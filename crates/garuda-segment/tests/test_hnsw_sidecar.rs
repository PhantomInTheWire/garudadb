use garuda_segment::{
    HnswSegmentSearchRequest, PersistedSegment, RecordState, SearchVisibility, SegmentFilter,
    StoredRecord, read_persisted_segment, search_persisted_hnsw, segment_meta,
    write_persisted_segment,
};
use garuda_storage::{read_file, segment_hnsw_index_path};
use garuda_types::{
    DenseVector, DistanceMetric, Doc, DocId, FieldName, FilterExpr, HnswIndexParams, InternalDocId,
    ScalarValue, SegmentId, StatusCode, TopK, VectorDimension, VectorFieldSchema, VectorIndexState,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn persisted_hnsw_sidecar_roundtrips_search() {
    let root = temp_root("segment-hnsw-sidecar");
    let vector_field = vector_field(VectorIndexState::HnswOnly(HnswIndexParams::default()));
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", "alpha", [0.9, 0.1, 0.0, 0.0]),
            stored_record(3, "doc-3", "beta", [0.0, 1.0, 0.0, 0.0]),
        ],
        &vector_field,
    );
    write_persisted_segment(&root, &segment, &vector_field)
        .expect("write segment with hnsw sidecar");
    let reopened = read_persisted_segment(&root, &segment.meta, &vector_field)
        .expect("read segment with hnsw sidecar");

    let hits = search_persisted_hnsw(
        &reopened,
        HnswSegmentSearchRequest {
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            top_k: TopK::new(3).expect("valid top_k"),
            ef_search: HnswIndexParams::default().ef_search,
            filter: SegmentFilter::All,
        },
        SearchVisibility::All,
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
    let vector_field = vector_field(VectorIndexState::HnswOnly(HnswIndexParams::default()));
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0])],
        &vector_field,
    );
    write_persisted_segment(&root, &segment, &vector_field).expect("write segment");

    std::fs::remove_file(segment_hnsw_index_path(&root, segment.meta.id)).expect("remove sidecar");
    let missing = read_persisted_segment(&root, &segment.meta, &vector_field)
        .expect_err("missing sidecar should fail");
    assert_eq!(missing.code, StatusCode::NotFound);

    write_persisted_segment(&root, &segment, &vector_field).expect("rewrite segment");
    std::fs::write(segment_hnsw_index_path(&root, segment.meta.id), b"broken")
        .expect("corrupt sidecar");
    let invalid = read_persisted_segment(&root, &segment.meta, &vector_field)
        .expect_err("invalid sidecar should fail");
    assert_eq!(invalid.code, StatusCode::Internal);

    let sidecar_bytes =
        read_file(&segment_hnsw_index_path(&root, segment.meta.id)).expect("read corrupt sidecar");
    assert_eq!(sidecar_bytes, b"broken");
}

#[test]
fn filtered_hnsw_search_does_not_truncate_matching_hits_before_filtering() {
    let root = temp_root("segment-hnsw-filtered");
    let vector_field = vector_field(VectorIndexState::HnswOnly(HnswIndexParams::default()));
    let segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", "beta", [0.0, 1.0, 0.0, 0.0]),
        ],
        &vector_field,
    );
    write_persisted_segment(&root, &segment, &vector_field).expect("write segment");
    let reopened =
        read_persisted_segment(&root, &segment.meta, &vector_field).expect("read segment");

    let filter = FilterExpr::Eq(
        "category".to_string(),
        ScalarValue::String("beta".to_string()),
    );
    let hits = search_persisted_hnsw(
        &reopened,
        HnswSegmentSearchRequest {
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            top_k: TopK::new(1).expect("valid top_k"),
            ef_search: HnswIndexParams::default().ef_search,
            filter: SegmentFilter::Matching(&filter),
        },
        SearchVisibility::All,
    )
    .expect("search reopened segment");

    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].record.doc.id,
        DocId::parse("doc-2").expect("valid doc id")
    );
}

#[test]
fn stale_hnsw_sidecar_fails_reopen_when_live_entries_change() {
    let root = temp_root("segment-hnsw-sidecar-stale");
    let vector_field = vector_field(VectorIndexState::HnswOnly(HnswIndexParams::default()));
    let mut segment = PersistedSegment::new(
        segment_meta(SegmentId::new_unchecked(1)),
        vec![
            stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]),
            stored_record(2, "doc-2", "beta", [0.0, 1.0, 0.0, 0.0]),
        ],
        &vector_field,
    );
    write_persisted_segment(&root, &segment, &vector_field).expect("write segment");

    segment.records[1].state = RecordState::Deleted;
    segment.sync_meta();
    write_persisted_segment(&root, &segment, &vector_field).expect("write stale segment");

    let error = read_persisted_segment(&root, &segment.meta, &vector_field)
        .expect_err("stale sidecar should fail");
    assert_eq!(error.code, StatusCode::Internal);
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

fn stored_record(internal_doc_id: u64, id: &str, category: &str, vector: [f32; 4]) -> StoredRecord {
    StoredRecord {
        doc_id: InternalDocId::new(internal_doc_id).expect("valid internal doc id"),
        state: RecordState::Live,
        doc: Doc::new(
            DocId::parse(id).expect("valid doc id"),
            BTreeMap::from([
                ("pk".to_string(), ScalarValue::String(id.to_string())),
                (
                    "category".to_string(),
                    ScalarValue::String(category.to_string()),
                ),
            ]),
            DenseVector::parse(vector.to_vec()).expect("valid vector"),
        ),
    }
}

fn vector_field(indexes: VectorIndexState) -> VectorFieldSchema {
    VectorFieldSchema {
        name: FieldName::parse("embedding").expect("valid field name"),
        dimension: VectorDimension::new(4).expect("valid dimension"),
        metric: DistanceMetric::Cosine,
        indexes,
    }
}
