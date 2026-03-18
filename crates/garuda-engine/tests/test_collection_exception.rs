mod common;

use common::{
    build_doc, database, default_options, default_schema, dense_vector, doc_id, field_name,
    seed_collection,
};
use garuda_types::{Doc, IndexKind, ScalarType, ScalarValue};
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
        vector: vec![1.0, 2.0],
        score: None,
    }]);
    assert_eq!(
        wrong_dimension[0].status.code,
        garuda_types::StatusCode::InvalidArgument
    );

    let query = collection.query(garuda_types::VectorQuery {
        field_name: field_name("embedding"),
        vector: Some(dense_vector(vec![1.0, 0.0, 0.0, 0.0])),
        id: None,
        top_k: 10,
        filter: Some("category = ".to_string()),
        include_vector: false,
        output_fields: None,
        ef_search: None,
    });
    assert!(query.is_err());

    let bad_index = collection.create_index(&field_name("rank"), IndexKind::Hnsw);
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
        vector: Vec::new(),
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
            nullable: false,
            default_value: Some(ScalarValue::String(String::from("yes"))),
        });

    let wrong_type = db.create_collection(wrong_type_schema, default_options());
    assert!(wrong_type.is_err());
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
        nullable: false,
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
        nullable: false,
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
        nullable: false,
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
