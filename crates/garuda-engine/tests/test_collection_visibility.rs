mod common;

use common::{
    build_doc, database, default_options, default_schema, dense_vector, doc_id, field_name,
    seed_collection,
};
use garuda_types::VectorQuery;

#[test]
fn deleted_documents_should_disappear_from_fetch_and_query() {
    let (_root, db) = database("visibility-delete");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let deleted = collection.delete(vec![doc_id("doc-2")]);
    assert!(deleted[0].status.is_ok());

    let fetched = collection.fetch(vec![doc_id("doc-2")]);
    assert!(fetched.is_empty());

    let results = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            10,
        ))
        .expect("query after delete");
    assert!(!results.iter().any(|doc| doc.id == doc_id("doc-2")));
}

#[test]
fn update_and_upsert_should_change_visible_document_version_only_once() {
    let (_root, db) = database("visibility-update-upsert");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let inserted = collection.insert(vec![build_doc(
        "doc-1",
        1,
        "alpha",
        0.9,
        [1.0, 0.0, 0.0, 0.0],
    )]);
    assert!(inserted[0].status.is_ok());

    let updated = collection.update(vec![build_doc(
        "doc-1",
        99,
        "beta",
        0.1,
        [0.0, 1.0, 0.0, 0.0],
    )]);
    assert!(updated[0].status.is_ok());

    let upserted = collection.upsert(vec![build_doc(
        "doc-1",
        100,
        "gamma",
        0.2,
        [0.0, 0.0, 1.0, 0.0],
    )]);
    assert!(upserted[0].status.is_ok());

    let fetched = collection.fetch(vec![doc_id("doc-1")]);
    assert_eq!(fetched.len(), 1);
    assert_eq!(
        fetched["doc-1"].fields["rank"],
        garuda_types::ScalarValue::Int64(100)
    );
}
