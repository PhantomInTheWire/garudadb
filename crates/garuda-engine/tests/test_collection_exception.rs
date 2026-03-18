mod common;

use common::{
    build_doc, database, default_options, default_schema, dense_vector, doc_id, field_name,
};
use garuda_types::{Doc, IndexKind, ScalarValue};

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
