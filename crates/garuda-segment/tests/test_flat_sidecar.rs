mod common;

use common::{field_name, stored_record, temp_root};
use garuda_index_scalar::prefilter_doc_ids;
use garuda_segment::{
    FlatSearchRequest, PersistedSegment, SegmentFilter, SegmentSearchRequest,
    read_persisted_segment, search_persisted, segment_meta, write_persisted_segment,
};
use garuda_storage::{read_file, segment_flat_index_path, segment_scalar_index_path};
use garuda_types::{
    CollectionName, CollectionSchema, DenseVector, DistanceMetric, DocId, FilterExpr,
    InternalDocId, Nullability, ScalarCompareOp, ScalarFieldSchema, ScalarIndexState,
    ScalarPredicate, ScalarPrefilter, ScalarType, ScalarValue, SegmentId, StatusCode, TopK,
    VectorDimension, VectorFieldSchema, VectorIndexState,
};

#[test]
fn persisted_flat_sidecar_roundtrips_exact_search() {
    let root = temp_root("segment-flat-sidecar");
    let schema = schema(VectorIndexState::DefaultFlat, ScalarIndexState::None);
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", "alpha", [0.9, 0.1, 0.0, 0.0]),
            stored_record(3, "doc-3", "beta", [0.0, 1.0, 0.0, 0.0]),
        ],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment with flat sidecar");
    let reopened = read_persisted_segment(&root, &segment.meta, &schema)
        .expect("read segment with flat sidecar");

    let hits = search_persisted(
        &reopened,
        SegmentSearchRequest::Flat(FlatSearchRequest {
            metric: DistanceMetric::Cosine,
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            top_k: TopK::new(3).expect("valid top_k"),
            filter: SegmentFilter::All,
        }),
        None,
        &garuda_meta::DeleteStore::new(),
    )
    .expect("search reopened segment");

    assert_eq!(
        hits.iter()
            .map(|hit| hit.record.doc.id.clone())
            .collect::<Vec<_>>(),
        vec![
            DocId::parse("doc-1").expect("valid doc id"),
            DocId::parse("doc-2").expect("valid doc id"),
            DocId::parse("doc-3").expect("valid doc id"),
        ]
    );
}

#[test]
fn missing_or_invalid_flat_sidecar_fails_reopen_for_persisted_flat_segments() {
    let root = temp_root("segment-flat-sidecar-invalid");
    let schema = schema(VectorIndexState::DefaultFlat, ScalarIndexState::None);
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0])],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment");

    std::fs::remove_file(segment_flat_index_path(&root, segment.meta.id)).expect("remove sidecar");
    let missing = read_persisted_segment(&root, &segment.meta, &schema)
        .expect_err("missing sidecar should fail");
    assert_eq!(missing.code, StatusCode::NotFound);

    write_persisted_segment(&root, &segment, &schema).expect("rewrite segment");
    std::fs::write(segment_flat_index_path(&root, segment.meta.id), b"broken")
        .expect("corrupt sidecar");
    let invalid = read_persisted_segment(&root, &segment.meta, &schema)
        .expect_err("invalid sidecar should fail");
    assert_eq!(invalid.code, StatusCode::Internal);

    let sidecar_bytes =
        read_file(&segment_flat_index_path(&root, segment.meta.id)).expect("read corrupt sidecar");
    assert_eq!(sidecar_bytes, b"broken");
}

#[test]
fn filtered_exact_search_does_not_truncate_matching_hits_before_filtering() {
    let root = temp_root("segment-flat-filtered");
    let schema = schema(VectorIndexState::DefaultFlat, ScalarIndexState::None);
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", "beta", [0.0, 1.0, 0.0, 0.0]),
        ],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment");
    let reopened = read_persisted_segment(&root, &segment.meta, &schema).expect("read segment");

    let filter = FilterExpr::Eq(
        "category".to_string(),
        ScalarValue::String("beta".to_string()),
    );
    let hits = search_persisted(
        &reopened,
        SegmentSearchRequest::Flat(FlatSearchRequest {
            metric: DistanceMetric::Cosine,
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            top_k: TopK::new(1).expect("valid top_k"),
            filter: SegmentFilter::Matching(&filter),
        }),
        None,
        &garuda_meta::DeleteStore::new(),
    )
    .expect("search reopened segment");

    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].record.doc.id,
        DocId::parse("doc-2").expect("valid doc id")
    );
}

#[test]
fn scalar_prefilter_does_not_drop_farther_allowed_flat_hit() {
    let root = temp_root("segment-flat-scalar-prefilter");
    let schema = schema(VectorIndexState::DefaultFlat, ScalarIndexState::Indexed);
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", "beta", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", "beta", [0.95, 0.05, 0.0, 0.0]),
            stored_record(3, "doc-3", "alpha", [0.0, 1.0, 0.0, 0.0]),
        ],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment");
    let reopened = read_persisted_segment(&root, &segment.meta, &schema).expect("read segment");

    let allowed_doc_ids = prefilter_doc_ids(
        &ScalarPrefilter::And(vec![ScalarPredicate {
            field: field_name("category"),
            op: ScalarCompareOp::Eq,
            value: ScalarValue::String("alpha".to_string()),
        }]),
        &reopened.scalar_indexes,
    )
    .expect("scalar prefilter");

    let hits = search_persisted(
        &reopened,
        SegmentSearchRequest::Flat(FlatSearchRequest {
            metric: DistanceMetric::Cosine,
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            top_k: TopK::new(1).expect("valid top_k"),
            filter: SegmentFilter::All,
        }),
        Some(&allowed_doc_ids),
        &garuda_meta::DeleteStore::new(),
    )
    .expect("search reopened segment");

    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].record.doc.id,
        DocId::parse("doc-3").expect("valid doc id")
    );
}

#[test]
fn persisted_scalar_sidecar_roundtrips() {
    let root = temp_root("segment-scalar-sidecar");
    let schema = schema(VectorIndexState::DefaultFlat, ScalarIndexState::Indexed);
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", "beta", [0.0, 1.0, 0.0, 0.0]),
        ],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment");
    let reopened = read_persisted_segment(&root, &segment.meta, &schema).expect("read segment");

    let matching = prefilter_doc_ids(
        &ScalarPrefilter::And(vec![ScalarPredicate {
            field: field_name("category"),
            op: ScalarCompareOp::Eq,
            value: ScalarValue::String("alpha".to_string()),
        }]),
        &reopened.scalar_indexes,
    )
    .expect("scalar prefilter");

    assert_eq!(matching.len(), 1);
    assert!(matching.contains(&InternalDocId::new(1).expect("valid doc id")));
}

#[test]
fn missing_scalar_sidecar_fails_reopen_for_indexed_scalar_field() {
    let root = temp_root("segment-scalar-sidecar-missing");
    let schema = schema(VectorIndexState::DefaultFlat, ScalarIndexState::Indexed);
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0])],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment");

    std::fs::remove_file(segment_scalar_index_path(
        &root,
        segment.meta.id,
        &field_name("category"),
    ))
    .expect("remove scalar sidecar");

    let error =
        read_persisted_segment(&root, &segment.meta, &schema).expect_err("missing scalar sidecar");
    assert_eq!(error.code, StatusCode::NotFound);
}

fn schema(vector_indexes: VectorIndexState, scalar_index: ScalarIndexState) -> CollectionSchema {
    CollectionSchema {
        name: CollectionName::parse("docs").expect("valid name"),
        primary_key: field_name("pk"),
        fields: vec![
            ScalarFieldSchema {
                name: field_name("pk"),
                field_type: ScalarType::String,
                index: ScalarIndexState::None,
                nullability: Nullability::Required,
                default_value: None,
            },
            ScalarFieldSchema {
                name: field_name("category"),
                field_type: ScalarType::String,
                index: scalar_index,
                nullability: Nullability::Required,
                default_value: None,
            },
        ],
        vector: VectorFieldSchema {
            name: field_name("embedding"),
            dimension: VectorDimension::new(4).expect("valid dimension"),
            metric: DistanceMetric::Cosine,
            indexes: vector_indexes,
        },
    }
}
