use garuda_engine::{Collection, Database};
use garuda_types::{
    AccessMode, CollectionName, CollectionOptions, CollectionSchema, DenseVector, DistanceMetric,
    Doc, DocId, FieldName, FlatIndexParams, IndexParams, Nullability, ScalarFieldSchema,
    ScalarType, ScalarValue, StorageAccess, TopK, VectorDimension, VectorFieldSchema,
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
    let path = std::env::temp_dir().join(format!("garudadb-{prefix}-{ts}-{nonce}"));
    std::fs::create_dir_all(&path).expect("create temp root");
    path
}

pub fn default_schema(name: &str) -> CollectionSchema {
    CollectionSchema {
        name: collection_name(name),
        primary_key: field_name("pk"),
        fields: vec![
            ScalarFieldSchema {
                name: field_name("pk"),
                field_type: ScalarType::String,
                nullability: Nullability::Required,
                default_value: None,
            },
            ScalarFieldSchema {
                name: field_name("rank"),
                field_type: ScalarType::Int64,
                nullability: Nullability::Required,
                default_value: None,
            },
            ScalarFieldSchema {
                name: field_name("category"),
                field_type: ScalarType::String,
                nullability: Nullability::Required,
                default_value: None,
            },
            ScalarFieldSchema {
                name: field_name("score"),
                field_type: ScalarType::Float64,
                nullability: Nullability::Required,
                default_value: None,
            },
        ],
        vector: VectorFieldSchema {
            name: field_name("embedding"),
            dimension: VectorDimension::new(4).expect("valid dimension"),
            metric: DistanceMetric::Cosine,
            index: IndexParams::Flat(FlatIndexParams),
        },
    }
}

pub fn default_options() -> CollectionOptions {
    CollectionOptions {
        access_mode: AccessMode::ReadWrite,
        storage_access: StorageAccess::MmapPreferred,
        segment_max_docs: 2,
    }
}

pub fn database(prefix: &str) -> (PathBuf, Database) {
    let root = temp_root(prefix);
    let db = Database::open(&root).expect("open db root");
    (root, db)
}

pub fn build_doc(id: &str, rank: i64, category: &str, score: f64, vector: [f32; 4]) -> Doc {
    let mut fields = BTreeMap::new();
    fields.insert("pk".to_string(), ScalarValue::String(id.to_string()));
    fields.insert("rank".to_string(), ScalarValue::Int64(rank));
    fields.insert(
        "category".to_string(),
        ScalarValue::String(category.to_string()),
    );
    fields.insert("score".to_string(), ScalarValue::Float64(score));
    Doc::new(
        doc_id(id),
        fields,
        DenseVector::parse(vector.to_vec()).expect("valid non-empty vector"),
    )
}

pub fn doc_id(value: &str) -> DocId {
    DocId::parse(value).expect("valid doc id")
}

pub fn collection_name(value: &str) -> CollectionName {
    CollectionName::parse(value).expect("valid collection name")
}

pub fn field_name(value: &str) -> FieldName {
    FieldName::parse(value).expect("valid field name")
}

pub fn dense_vector(values: Vec<f32>) -> DenseVector {
    DenseVector::parse(values).expect("valid non-empty vector")
}

pub fn top_k(value: usize) -> TopK {
    TopK::new(value).expect("valid top_k")
}

pub fn seed_collection(collection: &Collection) {
    let docs = vec![
        build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]),
        build_doc("doc-2", 2, "alpha", 0.8, [0.9, 0.1, 0.0, 0.0]),
        build_doc("doc-3", 3, "beta", 0.7, [0.0, 1.0, 0.0, 0.0]),
        build_doc("doc-4", 4, "gamma", 0.6, [0.0, 0.0, 1.0, 0.0]),
    ];
    let results = collection.insert(docs);
    assert!(results.iter().all(|result| result.status.is_ok()));
}

pub fn seed_more_collection_docs(collection: &Collection) {
    let docs = vec![
        build_doc("doc-5", 5, "alpha", 0.5, [1.0, 0.0, 0.1, 0.0]),
        build_doc("doc-6", 6, "beta", 0.4, [0.0, 1.0, 0.1, 0.0]),
        build_doc("doc-7", 7, "beta", 0.3, [0.0, 1.0, 0.0, 0.1]),
        build_doc("doc-8", 8, "gamma", 0.2, [0.0, 0.0, 1.0, 0.1]),
    ];
    let results = collection.insert(docs);
    assert!(results.iter().all(|result| result.status.is_ok()));
}
