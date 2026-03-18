mod common;

use common::{build_doc, collection_name, database, default_options, default_schema, doc_id};

#[test]
fn reopen_after_unflushed_writes_should_follow_a_clear_recovery_policy() {
    let (_root, db) = database("recovery-unflushed");
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
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen after wal-backed write");
    let fetched = reopened.fetch(vec![doc_id("doc-1")]);
    assert!(fetched.contains_key(&doc_id("doc-1")));
}

#[test]
fn reopen_after_delete_then_flush_should_not_resurrect_tombstoned_docs() {
    let (_root, db) = database("recovery-delete");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    common::seed_collection(&collection);
    let deleted = collection.delete(vec![doc_id("doc-2")]);
    assert!(deleted[0].status.is_ok());
    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen");
    assert!(reopened.fetch(vec![doc_id("doc-2")]).is_empty());
}
