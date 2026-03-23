mod common;

use common::{field_name, stored_record, temp_root};
use garuda_index_scalar::prefilter_doc_ids;
use garuda_segment::{
    HnswSegmentSearchRequest, PersistedSegment, RecordState, SegmentFilter, SegmentSearchRequest,
    read_persisted_segment, search_persisted, segment_meta, write_persisted_segment,
};
use garuda_storage::{read_file, segment_hnsw_index_path};
use garuda_types::{
    CollectionName, CollectionSchema, DenseVector, DistanceMetric, DocId, FilterExpr,
    HnswIndexParams, Nullability, ScalarCompareOp, ScalarFieldSchema, ScalarIndexState,
    ScalarPredicate, ScalarPrefilter, ScalarType, ScalarValue, SegmentId, StatusCode, TopK,
    VectorDimension, VectorFieldSchema, VectorIndexState,
};

#[test]
fn persisted_hnsw_sidecar_roundtrips_search() {
    let root = temp_root("segment-hnsw-sidecar");
    let schema = schema();
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", "alpha", [0.9, 0.1, 0.0, 0.0]),
            stored_record(3, "doc-3", "beta", [0.0, 1.0, 0.0, 0.0]),
        ],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment with hnsw sidecar");
    let reopened = read_persisted_segment(&root, &segment.meta, &schema)
        .expect("read segment with hnsw sidecar");

    let hits = search_persisted(
        &reopened,
        SegmentSearchRequest::Hnsw(HnswSegmentSearchRequest {
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            top_k: TopK::new(3).expect("valid top_k"),
            ef_search: HnswIndexParams::default().ef_search,
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
fn missing_or_invalid_hnsw_sidecar_fails_reopen_for_persisted_hnsw_segments() {
    let root = temp_root("segment-hnsw-sidecar-invalid");
    let schema = schema();
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0])],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment");

    std::fs::remove_file(segment_hnsw_index_path(&root, segment.meta.id)).expect("remove sidecar");
    let missing = read_persisted_segment(&root, &segment.meta, &schema)
        .expect_err("missing sidecar should fail");
    assert_eq!(missing.code, StatusCode::NotFound);

    write_persisted_segment(&root, &segment, &schema).expect("rewrite segment");
    std::fs::write(segment_hnsw_index_path(&root, segment.meta.id), b"broken")
        .expect("corrupt sidecar");
    let invalid = read_persisted_segment(&root, &segment.meta, &schema)
        .expect_err("invalid sidecar should fail");
    assert_eq!(invalid.code, StatusCode::Internal);

    let sidecar_bytes =
        read_file(&segment_hnsw_index_path(&root, segment.meta.id)).expect("read corrupt sidecar");
    assert_eq!(sidecar_bytes, b"broken");
}

#[test]
fn filtered_hnsw_search_does_not_truncate_matching_hits_before_filtering() {
    let root = temp_root("segment-hnsw-filtered");
    let schema = schema();
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
        SegmentSearchRequest::Hnsw(HnswSegmentSearchRequest {
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            top_k: TopK::new(1).expect("valid top_k"),
            ef_search: HnswIndexParams::default().ef_search,
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
fn scalar_prefilter_does_not_drop_farther_allowed_hnsw_hit() {
    let root = temp_root("segment-hnsw-scalar-prefilter");
    let schema = CollectionSchema {
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
                index: ScalarIndexState::Indexed,
                nullability: Nullability::Required,
                default_value: None,
            },
        ],
        vector: VectorFieldSchema {
            name: field_name("embedding"),
            dimension: VectorDimension::new(4).expect("valid dimension"),
            metric: DistanceMetric::Cosine,
            indexes: VectorIndexState::HnswOnly(HnswIndexParams::default()),
        },
    };
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
        SegmentSearchRequest::Hnsw(HnswSegmentSearchRequest {
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            top_k: TopK::new(1).expect("valid top_k"),
            ef_search: HnswIndexParams::default().ef_search,
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
fn stale_hnsw_sidecar_fails_reopen_when_live_entries_change() {
    let root = temp_root("segment-hnsw-sidecar-stale");
    let schema = schema();
    let mut segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", "beta", [0.0, 1.0, 0.0, 0.0]),
        ],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment");
    let original_sidecar =
        read_file(&segment_hnsw_index_path(&root, segment.meta.id)).expect("read sidecar");

    segment.records[1].state = RecordState::Deleted;
    segment.rebuild_search_resources(&schema);
    write_persisted_segment(&root, &segment, &schema).expect("write stale segment");
    std::fs::write(
        segment_hnsw_index_path(&root, segment.meta.id),
        original_sidecar,
    )
    .expect("restore stale sidecar");

    let error = read_persisted_segment(&root, &segment.meta, &schema)
        .expect_err("stale sidecar should fail");
    assert_eq!(error.code, StatusCode::Internal);
}

fn schema() -> CollectionSchema {
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
                index: ScalarIndexState::None,
                nullability: Nullability::Required,
                default_value: None,
            },
        ],
        vector: VectorFieldSchema {
            name: field_name("embedding"),
            dimension: VectorDimension::new(4).expect("valid dimension"),
            metric: DistanceMetric::Cosine,
            indexes: VectorIndexState::HnswOnly(HnswIndexParams::default()),
        },
    }
}
