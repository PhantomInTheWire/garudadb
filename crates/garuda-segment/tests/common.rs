use garuda_segment::{RecordState, StoredRecord};
use garuda_types::{
    DenseVector, Doc, DocId, FieldName, InternalDocId, ScalarValue,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn temp_root(prefix: &str) -> PathBuf {
    let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("garudadb-segment-{prefix}-{ts}-{nonce}"));
    std::fs::create_dir_all(&path).expect("create temp root");
    path
}

pub fn stored_record(
    internal_doc_id: u64,
    id: &str,
    category: &str,
    vector: [f32; 4],
) -> StoredRecord {
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

pub fn field_name(name: &str) -> FieldName {
    FieldName::parse(name).expect("valid field name")
}
