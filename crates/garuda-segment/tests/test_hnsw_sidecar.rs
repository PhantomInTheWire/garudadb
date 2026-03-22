use garuda_segment::{
    HnswSegmentSearchRequest, SegmentFile, SegmentFilter, SegmentKind, StoredRecord, read_segment,
    search_hnsw, segment_meta, sync_segment, write_segment,
};
use garuda_storage::{read_file, segment_hnsw_index_path};
use garuda_types::{
    DenseVector, DistanceMetric, Doc, DocId, FieldName, HnswIndexParams, IndexParams,
    InternalDocId, SegmentId, StatusCode, TopK, VectorDimension, VectorFieldSchema,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn persisted_hnsw_sidecar_roundtrips_search() {
    let root = temp_root("segment-hnsw-sidecar");
    let mut segment = SegmentFile::new(
        segment_meta(SegmentId::new_unchecked(1)),
        Vec::new(),
        SegmentKind::Persisted,
        &vector_field(IndexParams::Hnsw(HnswIndexParams::default())),
    );
    segment
        .records
        .push(stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]));
    segment
        .records
        .push(stored_record(2, "doc-2", "alpha", [0.9, 0.1, 0.0, 0.0]));
    segment
        .records
        .push(stored_record(3, "doc-3", "beta", [0.0, 1.0, 0.0, 0.0]));
    let vector_field = vector_field(IndexParams::Hnsw(HnswIndexParams::default()));
    sync_segment(&mut segment, &vector_field);
    write_segment(&root, &segment, &vector_field).expect("write segment with hnsw sidecar");

    let reopened =
        read_segment(&root, &segment.meta, &vector_field).expect("read segment with hnsw sidecar");

    let hits = search_hnsw(
        &reopened,
        HnswSegmentSearchRequest {
            query_vector: &DenseVector::parse(vec![1.0, 0.0, 0.0, 0.0]).expect("valid vector"),
            top_k: TopK::new(3).expect("valid top_k"),
            ef_search: HnswIndexParams::default().ef_search,
            filter: SegmentFilter::All,
        },
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
    let mut segment = SegmentFile::new(
        segment_meta(SegmentId::new_unchecked(1)),
        Vec::new(),
        SegmentKind::Persisted,
        &vector_field(IndexParams::Hnsw(HnswIndexParams::default())),
    );
    segment
        .records
        .push(stored_record(1, "doc-1", "alpha", [1.0, 0.0, 0.0, 0.0]));
    let vector_field = vector_field(IndexParams::Hnsw(HnswIndexParams::default()));
    sync_segment(&mut segment, &vector_field);
    write_segment(&root, &segment, &vector_field).expect("write segment");

    std::fs::remove_file(segment_hnsw_index_path(&root, segment.meta.id)).expect("remove sidecar");
    let missing =
        read_segment(&root, &segment.meta, &vector_field).expect_err("missing sidecar should fail");
    assert_eq!(missing.code, StatusCode::NotFound);

    write_segment(&root, &segment, &vector_field).expect("rewrite segment");
    std::fs::write(segment_hnsw_index_path(&root, segment.meta.id), b"broken")
        .expect("corrupt sidecar");
    let invalid =
        read_segment(&root, &segment.meta, &vector_field).expect_err("invalid sidecar should fail");
    assert_eq!(invalid.code, StatusCode::Internal);

    let sidecar_bytes =
        read_file(&segment_hnsw_index_path(&root, segment.meta.id)).expect("read corrupt sidecar");
    assert_eq!(sidecar_bytes, b"broken");
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

fn vector_field(index: IndexParams) -> VectorFieldSchema {
    VectorFieldSchema {
        name: FieldName::parse("embedding").expect("valid field name"),
        dimension: VectorDimension::new(4).expect("valid dimension"),
        metric: DistanceMetric::Cosine,
        index,
    }
}
