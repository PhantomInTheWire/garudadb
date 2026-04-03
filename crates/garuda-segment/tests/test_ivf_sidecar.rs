use garuda_segment::{
    PersistedSegment, SegmentExecutionRequest, SegmentFilter, SegmentFilterContext, WritingSegment,
    read_persisted_segment, search_persisted, segment_meta, write_persisted_segment,
};
use garuda_index_scalar::prefilter_doc_ids;
use garuda_storage::{read_file, segment_ivf_index_path};
use garuda_types::{
    AnnBudgetPolicy, CollectionName, CollectionSchema, DenseVector, DistanceMetric, Doc, DocId,
    FieldName, InternalDocId, IvfIndexParams, IvfRecallPlan, Nullability, RecallPlan,
    ScalarCompareOp, ScalarFieldSchema, ScalarIndexState, ScalarPredicate, ScalarType,
    ScalarValue, SegmentId, StatusCode, TopK, VectorDimension, VectorFieldSchema, VectorIndexState,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn persisted_ivf_sidecar_roundtrips_search() {
    let root = temp_root("segment-ivf-sidecar");
    let schema = schema();
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", [0.9, 0.1, 0.0, 0.0]),
            stored_record(3, "doc-3", [0.0, 1.0, 0.0, 0.0]),
        ],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment with ivf sidecar");
    let reopened = read_persisted_segment(&root, &segment.meta, &schema)
        .expect("read segment with ivf sidecar");

    let hits = search_persisted(
        &reopened,
        SegmentExecutionRequest {
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            metric: DistanceMetric::Cosine,
            recall: RecallPlan::Ivf(IvfRecallPlan {
                top_k: TopK::new(3).expect("valid top_k"),
                nprobe: schema
                    .vector
                    .indexes
                    .ivf_params()
                    .expect("ivf params")
                    .n_probe,
                budget: AnnBudgetPolicy::Requested,
            }),
            filter: SegmentFilterContext {
                allowed_doc_ids: None,
                delete_store: Some(&garuda_meta::DeleteStore::new()),
                residual: SegmentFilter::All,
            },
        },
    )
    .expect("search reopened segment");

    assert_eq!(hits.len(), 3);
    assert_eq!(
        hits[0].record.doc.id,
        DocId::parse("doc-1").expect("valid doc id")
    );
}

#[test]
fn missing_or_invalid_ivf_sidecar_fails_reopen() {
    let root = temp_root("segment-ivf-sidecar-invalid");
    let schema = schema();
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![stored_record(1, "doc-1", [1.0, 0.0, 0.0, 0.0])],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment");

    std::fs::remove_file(segment_ivf_index_path(&root, segment.meta.id)).expect("remove sidecar");
    let missing = read_persisted_segment(&root, &segment.meta, &schema)
        .expect_err("missing sidecar should fail");
    assert_eq!(missing.code, StatusCode::NotFound);

    write_persisted_segment(&root, &segment, &schema).expect("rewrite segment");
    std::fs::write(segment_ivf_index_path(&root, segment.meta.id), b"broken")
        .expect("corrupt sidecar");
    let invalid = read_persisted_segment(&root, &segment.meta, &schema)
        .expect_err("invalid sidecar should fail");
    assert_eq!(invalid.code, StatusCode::Internal);

    let sidecar_bytes =
        read_file(&segment_ivf_index_path(&root, segment.meta.id)).expect("read corrupt sidecar");
    assert_eq!(sidecar_bytes, b"broken");
}

#[test]
fn writing_ivf_index_should_contain_each_live_record_once() {
    let schema = schema();
    let records = vec![
        stored_record(1, "doc-1", [1.0, 0.0, 0.0, 0.0]),
        stored_record(2, "doc-2", [0.9, 0.1, 0.0, 0.0]),
        stored_record(3, "doc-3", [0.0, 1.0, 0.0, 0.0]),
    ];
    let segment = WritingSegment::new(segment_meta(SegmentId::new_unchecked(0)), records, &schema);

    let index = segment.ivf_index.expect("ivf index");
    assert_eq!(segment.meta.doc_count, 3);
    assert_eq!(index.list_count(), 3);
}

#[test]
fn write_persisted_segment_should_compact_ivf_sidecar_after_incremental_delete() {
    let root = temp_root("segment-ivf-sidecar-compact-after-delete");
    let schema = schema();
    let mut segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", [0.9, 0.1, 0.0, 0.0]),
            stored_record(3, "doc-3", [0.0, 1.0, 0.0, 0.0]),
        ],
        &schema,
    );

    assert!(segment.mark_deleted(InternalDocId::new(2).expect("doc id")));
    write_persisted_segment(&root, &segment, &schema).expect("write compact ivf sidecar");

    let reopened =
        read_persisted_segment(&root, &segment.meta, &schema).expect("reopen compact ivf sidecar");
    assert_eq!(reopened.meta.doc_count, 2);

    let hits = search_persisted(
        &reopened,
        SegmentExecutionRequest {
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            metric: DistanceMetric::Cosine,
            recall: RecallPlan::Ivf(IvfRecallPlan {
                top_k: TopK::new(3).expect("valid top_k"),
                nprobe: schema
                    .vector
                    .indexes
                    .ivf_params()
                    .expect("ivf params")
                    .n_probe,
                budget: AnnBudgetPolicy::Requested,
            }),
            filter: SegmentFilterContext {
                allowed_doc_ids: None,
                delete_store: Some(&garuda_meta::DeleteStore::new()),
                residual: SegmentFilter::All,
            },
        },
    )
    .expect("search reopened segment");

    assert_eq!(hits.len(), 2);
    assert!(
        hits.iter()
            .all(|hit| hit.record.doc.id != DocId::parse("doc-2").expect("valid doc id"))
    );
}

#[test]
fn read_persisted_segment_skips_ivf_sidecar_when_doc_count_is_zero() {
    let root = temp_root("segment-ivf-sidecar-empty");
    let schema = schema();
    let mut segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![stored_record(1, "doc-1", [1.0, 0.0, 0.0, 0.0])],
        &schema,
    );

    assert!(segment.mark_deleted(InternalDocId::new(1).expect("doc id")));
    write_persisted_segment(&root, &segment, &schema).expect("write segment");

    let reopened =
        read_persisted_segment(&root, &segment.meta, &schema).expect("reopen empty segment");
    assert_eq!(reopened.meta.doc_count, 0);
    assert!(reopened.ivf_index.is_none());
}

#[test]
fn scalar_prefilter_with_only_deleted_visible_matches_returns_no_ivf_hits() {
    let root = temp_root("segment-ivf-prefilter-deleted-only");
    let schema = indexed_schema();
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_indexed_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]),
            stored_indexed_record(2, "doc-2", "beta", [0.0, 1.0, 0.0, 0.0]),
        ],
        &schema,
    );
    write_persisted_segment(&root, &segment, &schema).expect("write segment");
    let reopened = read_persisted_segment(&root, &segment.meta, &schema).expect("read segment");

    let allowed_doc_ids = prefilter_doc_ids(
        Some(&[ScalarPredicate {
            field: field_name("category"),
            op: ScalarCompareOp::Eq,
            value: ScalarValue::String("alpha".to_string()),
        }]),
        &reopened.scalar_indexes,
    )
    .expect("scalar prefilter");
    let mut delete_store = garuda_meta::DeleteStore::new();
    delete_store.insert(InternalDocId::new(1).expect("doc id"));

    let hits = search_persisted(
        &reopened,
        SegmentExecutionRequest {
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            metric: DistanceMetric::Cosine,
            recall: RecallPlan::Ivf(IvfRecallPlan {
                top_k: TopK::new(1).expect("valid top_k"),
                nprobe: schema
                    .vector
                    .indexes
                    .ivf_params()
                    .expect("ivf params")
                    .n_probe,
                budget: AnnBudgetPolicy::AdaptiveFiltered,
            }),
            filter: SegmentFilterContext {
                allowed_doc_ids: Some(&allowed_doc_ids),
                delete_store: Some(&delete_store),
                residual: SegmentFilter::All,
            },
        },
    )
    .expect("search reopened segment");

    assert!(hits.is_empty());
}

fn temp_root(prefix: &str) -> PathBuf {
    let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("garudadb-segment-{prefix}-{ts}-{nonce}"));
    std::fs::create_dir_all(&path).expect("create temp root");
    path
}

fn stored_record(internal_doc_id: u64, id: &str, vector: [f32; 4]) -> garuda_segment::StoredRecord {
    garuda_segment::StoredRecord {
        doc_id: InternalDocId::new(internal_doc_id).expect("valid internal doc id"),
        state: garuda_segment::RecordState::Live,
        doc: Doc::new(
            DocId::parse(id).expect("valid doc id"),
            BTreeMap::from([(
                "pk".to_string(),
                garuda_types::ScalarValue::String(id.to_string()),
            )]),
            DenseVector::parse(vector.to_vec()).expect("valid vector"),
        ),
    }
}

fn stored_indexed_record(
    internal_doc_id: u64,
    id: &str,
    category: &str,
    vector: [f32; 4],
) -> garuda_segment::StoredRecord {
    garuda_segment::StoredRecord {
        doc_id: InternalDocId::new(internal_doc_id).expect("valid internal doc id"),
        state: garuda_segment::RecordState::Live,
        doc: Doc::new(
            DocId::parse(id).expect("valid doc id"),
            BTreeMap::from([
                (
                    "pk".to_string(),
                    garuda_types::ScalarValue::String(id.to_string()),
                ),
                (
                    "category".to_string(),
                    garuda_types::ScalarValue::String(category.to_string()),
                ),
            ]),
            DenseVector::parse(vector.to_vec()).expect("valid vector"),
        ),
    }
}

fn schema() -> CollectionSchema {
    CollectionSchema {
        name: CollectionName::parse("docs").expect("valid name"),
        primary_key: field_name("pk"),
        fields: Vec::new(),
        vector: VectorFieldSchema {
            name: field_name("embedding"),
            dimension: VectorDimension::new(4).expect("valid dimension"),
            metric: DistanceMetric::Cosine,
            indexes: VectorIndexState::IvfOnly(IvfIndexParams::default()),
        },
    }
}

fn indexed_schema() -> CollectionSchema {
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
                index: ScalarIndexState::Indexed,
                nullability: Nullability::Required,
                default_value: None,
            },
        ],
        vector: VectorFieldSchema {
            name: field_name("embedding"),
            dimension: VectorDimension::new(4).expect("valid dimension"),
            metric: DistanceMetric::Cosine,
            indexes: VectorIndexState::IvfOnly(IvfIndexParams::default()),
        },
    }
}

fn field_name(value: &str) -> FieldName {
    FieldName::parse(value).expect("valid field name")
}
