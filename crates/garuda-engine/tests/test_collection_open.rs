mod common;

use common::{database, default_options, default_schema, seed_collection};

#[test]
fn open_collection_after_writes_preserves_data() {
    let (_root, db) = database("open");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");

    let reopened = db.open_collection("docs").expect("reopen");
    assert_eq!(reopened.stats().doc_count, 4);
    let fetched = reopened.fetch(vec!["doc-1".to_string(), "doc-4".to_string()]);
    assert_eq!(fetched.len(), 2);
}
