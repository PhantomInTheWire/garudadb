use garuda_engine::{Collection, Database};
use garuda_types::{
    CollectionOptions, CollectionSchema, DistanceMetric, Doc, FlatIndexParams, IndexParams,
    ScalarFieldSchema, ScalarType, ScalarValue, VectorFieldSchema,
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
        name: name.to_string(),
        primary_key: "pk".to_string(),
        fields: vec![
            ScalarFieldSchema {
                name: "pk".to_string(),
                field_type: ScalarType::String,
                nullable: false,
            },
            ScalarFieldSchema {
                name: "rank".to_string(),
                field_type: ScalarType::Int64,
                nullable: false,
            },
            ScalarFieldSchema {
                name: "category".to_string(),
                field_type: ScalarType::String,
                nullable: false,
            },
            ScalarFieldSchema {
                name: "score".to_string(),
                field_type: ScalarType::Float64,
                nullable: false,
            },
        ],
        vector: VectorFieldSchema {
            name: "embedding".to_string(),
            dimension: 4,
            metric: DistanceMetric::Cosine,
            index: IndexParams::Flat(FlatIndexParams),
        },
    }
}

pub fn schema_with_dimension(name: &str, dimension: usize) -> CollectionSchema {
    let mut schema = default_schema(name);
    schema.vector.dimension = dimension;
    schema
}

pub fn schema_with_vector_name(name: &str, vector_name: &str) -> CollectionSchema {
    let mut schema = default_schema(name);
    schema.vector.name = vector_name.to_string();
    schema
}

pub fn schema_with_duplicate_field(name: &str) -> CollectionSchema {
    let mut schema = default_schema(name);
    schema.fields.push(schema.fields[0].clone());
    schema
}

pub fn schema_missing_primary_field(name: &str) -> CollectionSchema {
    let mut schema = default_schema(name);
    schema.primary_key = "missing_pk".to_string();
    schema
}

pub fn default_options() -> CollectionOptions {
    CollectionOptions {
        read_only: false,
        enable_mmap: true,
        segment_max_docs: 2,
    }
}

pub fn read_only_options() -> CollectionOptions {
    CollectionOptions {
        read_only: true,
        ..default_options()
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
    Doc::new(id, fields, vector.to_vec())
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
