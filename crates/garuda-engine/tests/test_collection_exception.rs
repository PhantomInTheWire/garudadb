mod common;

use common::{
    build_doc, collection_name, database, default_options, default_schema, dense_vector, doc_id,
    field_name, seed_collection,
};
use garuda_types::{Doc, HnswIndexParams, IndexParams, ScalarType, ScalarValue, TopK};
use std::fs;

#[test]
fn rejects_duplicate_ids_wrong_dimensions_invalid_filters_and_wrong_index_targets() {
    let (_root, db) = database("exception");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let first = collection.insert(vec![build_doc(
        "doc-1",
        1,
        "alpha",
        0.9,
        [1.0, 0.0, 0.0, 0.0],
    )]);
    assert!(first[0].status.is_ok());

    let duplicate = collection.insert(vec![build_doc(
        "doc-1",
        2,
        "beta",
        0.8,
        [0.0, 1.0, 0.0, 0.0],
    )]);
    assert_eq!(
        duplicate[0].status.code,
        garuda_types::StatusCode::AlreadyExists
    );

    let wrong_dimension = collection.insert(vec![Doc {
        id: doc_id("doc-bad"),
        fields: build_doc("doc-bad", 1, "alpha", 0.1, [1.0, 0.0, 0.0, 0.0]).fields,
        vector: dense_vector(vec![1.0, 2.0]),
        score: None,
    }]);
    assert_eq!(
        wrong_dimension[0].status.code,
        garuda_types::StatusCode::InvalidArgument
    );

    let query = collection.query(garuda_types::VectorQuery {
        field_name: field_name("embedding"),
        source: garuda_types::QueryVectorSource::Vector(dense_vector(vec![1.0, 0.0, 0.0, 0.0])),
        top_k: TopK::new(10).expect("valid top_k"),
        filter: Some("category = ".to_string()),
        vector_projection: garuda_types::VectorProjection::Exclude,
        output_fields: None,
        ef_search: None,
    });
    assert!(query.is_err());

    let bad_index = collection.create_index(
        &field_name("rank"),
        IndexParams::Hnsw(HnswIndexParams::default()),
    );
    assert!(bad_index.is_err());

    let fetched = collection.fetch(vec![doc_id("doc-1")]);
    assert_eq!(
        fetched["doc-1"].fields["category"],
        ScalarValue::String("alpha".to_string())
    );
}

#[test]
fn rejects_inserts_without_vectors() {
    let (_root, db) = database("exception-missing-vector");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let missing_vector = collection.insert(vec![Doc {
        id: doc_id("doc-no-vector"),
        fields: build_doc("doc-no-vector", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]).fields,
        vector: garuda_types::DenseVector::default(),
        score: None,
    }]);

    assert_eq!(
        missing_vector[0].status.code,
        garuda_types::StatusCode::InvalidArgument
    );

    let fetched = collection.fetch(vec![doc_id("doc-no-vector")]);
    assert!(fetched.is_empty());
}

#[test]
fn rejects_null_for_non_nullable_fields() {
    let (_root, db) = database("exception-null-required");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let mut doc = build_doc("doc-null-rank", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]);
    doc.fields.insert("rank".to_string(), ScalarValue::Null);

    let result = collection.insert(vec![doc]);
    assert_eq!(
        result[0].status.code,
        garuda_types::StatusCode::InvalidArgument
    );

    let fetched = collection.fetch(vec![doc_id("doc-null-rank")]);
    assert!(fetched.is_empty());
}

#[test]
fn rejects_invalid_default_values_during_collection_creation() {
    let (_root, db) = database("exception-invalid-default-on-create");

    let mut wrong_type_schema = default_schema("docs");
    wrong_type_schema
        .fields
        .push(garuda_types::ScalarFieldSchema {
            name: field_name("is_public"),
            field_type: ScalarType::Bool,
            nullability: garuda_types::Nullability::Required,
            default_value: Some(ScalarValue::String(String::from("yes"))),
        });

    let wrong_type = db.create_collection(wrong_type_schema, default_options());
    assert!(wrong_type.is_err());
}

#[test]
fn rejects_non_string_or_nullable_primary_keys_and_mismatched_primary_key_values() {
    let (_root, db) = database("exception-primary-key-invariants");

    let mut non_string_pk_schema = default_schema("docs-non-string-pk");
    non_string_pk_schema.fields[0].field_type = ScalarType::Int64;
    assert!(
        db.create_collection(non_string_pk_schema, default_options())
            .is_err()
    );

    let mut nullable_pk_schema = default_schema("docs-nullable-pk");
    nullable_pk_schema.fields[0].nullability = garuda_types::Nullability::Nullable;
    assert!(
        db.create_collection(nullable_pk_schema, default_options())
            .is_err()
    );

    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let mut wrong_insert = build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]);
    wrong_insert.fields.insert(
        String::from("pk"),
        ScalarValue::String(String::from("other-doc")),
    );

    let insert_result = collection.insert(vec![wrong_insert]);
    assert_eq!(
        insert_result[0].status.code,
        garuda_types::StatusCode::InvalidArgument
    );

    let ok_result = collection.insert(vec![build_doc(
        "doc-1",
        1,
        "alpha",
        0.9,
        [1.0, 0.0, 0.0, 0.0],
    )]);
    assert!(ok_result[0].status.is_ok());

    let mut wrong_update = Doc {
        id: doc_id("doc-1"),
        fields: std::collections::BTreeMap::new(),
        vector: garuda_types::DenseVector::default(),
        score: None,
    };
    wrong_update.fields.insert(
        String::from("pk"),
        ScalarValue::String(String::from("other-doc")),
    );

    let update_result = collection.update(vec![wrong_update]);
    assert_eq!(
        update_result[0].status.code,
        garuda_types::StatusCode::InvalidArgument
    );

    let fetched = collection.fetch(vec![doc_id("doc-1")]);
    assert_eq!(
        fetched["doc-1"].fields["pk"],
        ScalarValue::String(String::from("doc-1"))
    );
}

#[test]
fn add_column_rolls_back_state_when_persist_fails() {
    let (_root, db) = database("exception-ddl-persist-rollback");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let collection_dir = collection.path();
    let metadata = fs::metadata(&collection_dir).expect("collection dir metadata");
    let mut permissions = metadata.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&collection_dir, permissions).expect("make collection dir readonly");

    let result = collection.add_column(garuda_types::ScalarFieldSchema {
        name: field_name("is_public"),
        field_type: ScalarType::Bool,
        nullability: garuda_types::Nullability::Required,
        default_value: Some(ScalarValue::Bool(true)),
    });

    assert!(result.is_err());
    assert!(
        !collection
            .schema()
            .fields
            .iter()
            .any(|field| field.name == field_name("is_public"))
    );
}

#[test]
fn failed_persist_does_not_publish_new_manifest_on_reopen() {
    let (_root, db) = database("exception-ddl-manifest-commit-order");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let segment_dir = collection.path().join("1");
    fs::create_dir_all(&segment_dir).expect("create segment dir");

    let metadata = fs::metadata(&segment_dir).expect("segment dir metadata");
    let mut permissions = metadata.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&segment_dir, permissions).expect("make segment dir readonly");

    let result = collection.add_column(garuda_types::ScalarFieldSchema {
        name: field_name("is_public"),
        field_type: ScalarType::Bool,
        nullability: garuda_types::Nullability::Required,
        default_value: Some(ScalarValue::Bool(true)),
    });

    assert!(result.is_err());
    drop(collection);

    let reopened = db
        .open_collection(&common::collection_name("docs"))
        .expect("reopen collection");
    assert!(
        !reopened
            .schema()
            .fields
            .iter()
            .any(|field| field.name == field_name("is_public"))
    );
}

#[test]
fn failed_checkpoint_restores_rewritten_segment_files() {
    let (_root, db) = database("exception-checkpoint-segment-rollback");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let collection_dir = collection.path();
    for segment_id in ["0", "1", "2"] {
        fs::create_dir_all(collection_dir.join(segment_id)).expect("create segment dir");
    }

    let mut root_permissions = fs::metadata(&collection_dir)
        .expect("collection dir metadata")
        .permissions();
    root_permissions.set_readonly(true);
    fs::set_permissions(&collection_dir, root_permissions).expect("make root readonly");

    for segment_id in ["0", "1", "2"] {
        let segment_dir = collection_dir.join(segment_id);
        let mut segment_permissions = fs::metadata(&segment_dir)
            .expect("segment dir metadata")
            .permissions();
        segment_permissions.set_readonly(false);
        fs::set_permissions(&segment_dir, segment_permissions)
            .expect("restore segment dir write access");
    }

    let result = collection.add_column(garuda_types::ScalarFieldSchema {
        name: field_name("is_public"),
        field_type: ScalarType::Bool,
        nullability: garuda_types::Nullability::Required,
        default_value: Some(ScalarValue::Bool(true)),
    });

    assert!(result.is_err());
    drop(collection);

    let reopened = db
        .open_collection(&common::collection_name("docs"))
        .expect("reopen collection");
    let fetched = reopened.fetch(vec![doc_id("doc-1")]);
    let doc = fetched.get(&doc_id("doc-1")).expect("doc present");

    assert!(
        !reopened
            .schema()
            .fields
            .iter()
            .any(|field| field.name == field_name("is_public"))
    );
    assert!(!doc.fields.contains_key("is_public"));
}

#[test]
fn manifest_write_failure_keeps_wal_for_recovery() {
    let (root, db) = database("exception-manifest-wal");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let insert = collection.insert(vec![build_doc(
        "doc-rollback",
        1,
        "alpha",
        0.9,
        [1.0, 0.0, 0.0, 0.0],
    )]);
    assert!(insert[0].status.is_ok());

    let manifest_path = root.join("docs").join("manifest.1");
    fs::create_dir_all(&manifest_path).expect("create blocking manifest dir");

    assert!(collection.flush().is_err());
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen after wal-backed recovery");
    assert_eq!(reopened.fetch(vec![doc_id("doc-rollback")]).len(), 1);
    assert!(manifest_path.is_dir());
}
