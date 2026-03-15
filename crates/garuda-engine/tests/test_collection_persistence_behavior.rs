mod common;

use common::{build_doc, database, default_options, default_schema, seed_collection};
use garuda_types::VectorQuery;

#[test]
fn writes_deletes_and_schema_changes_should_persist_across_reopen() {
    let (_root, db) = database("persistence");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection
        .insert(vec![build_doc(
            "doc-9",
            9,
            "alpha",
            0.1,
            [1.0, 0.0, 0.0, 0.1],
        )]);
    collection.delete(vec!["doc-3".to_string()]);
    collection.flush().expect("flush");

    let reopened = db.open_collection("docs").expect("reopen");
    let fetched = reopened.fetch(vec!["doc-9".to_string(), "doc-3".to_string()]);
    assert!(fetched.contains_key("doc-9"));
    assert!(!fetched.contains_key("doc-3"));

    let results = reopened
        .query(VectorQuery::by_vector("embedding", vec![1.0, 0.0, 0.0, 0.0], 10))
        .expect("query");
    assert!(!results.iter().any(|doc| doc.id == "doc-3"));
}

#[test]
fn reopening_twice_should_not_change_user_visible_state() {
    let (_root, db) = database("persistence-double-open");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");

    let once = db.open_collection("docs").expect("open once");
    let twice = db.open_collection("docs").expect("open twice");
    assert_eq!(once.stats(), twice.stats());
    assert_eq!(once.schema(), twice.schema());
}
