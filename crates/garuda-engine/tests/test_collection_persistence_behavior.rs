mod common;

use common::{build_doc, database, default_options, default_schema, doc_id, seed_collection};
use garuda_types::VectorQuery;

#[test]
fn writes_deletes_and_schema_changes_should_persist_across_reopen() {
    let (_root, db) = database("persistence");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.insert(vec![build_doc(
        "doc-9",
        9,
        "alpha",
        0.1,
        [1.0, 0.0, 0.0, 0.1],
    )]);
    collection.delete(vec![doc_id("doc-3")]);
    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&common::collection_name("docs"))
        .expect("reopen");
    let fetched = reopened.fetch(vec![doc_id("doc-9"), doc_id("doc-3")]);
    assert!(fetched.contains_key(&doc_id("doc-9")));
    assert!(!fetched.contains_key(&doc_id("doc-3")));

    let results = reopened
        .query(VectorQuery::by_vector(
            common::field_name("embedding"),
            common::dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            10,
        ))
        .expect("query");
    assert!(!results.iter().any(|doc| doc.id == doc_id("doc-3")));
}

#[test]
fn reopening_twice_should_not_change_user_visible_state() {
    let (_root, db) = database("persistence-double-open");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");
    drop(collection);

    let once = db
        .open_collection(&common::collection_name("docs"))
        .expect("open once");
    drop(once);
    let twice = db
        .open_collection(&common::collection_name("docs"))
        .expect("open twice");
    assert_eq!(twice.stats().doc_count, 4);
    assert_eq!(twice.schema(), default_schema("docs"));
}
