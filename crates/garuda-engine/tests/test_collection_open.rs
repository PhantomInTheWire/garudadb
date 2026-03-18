mod common;

use common::{collection_name, database, default_options, default_schema, doc_id, seed_collection};

#[test]
fn open_collection_after_writes_preserves_data() {
    let (_root, db) = database("open");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen");
    assert_eq!(reopened.stats().doc_count, 4);
    let fetched = reopened.fetch(vec![doc_id("doc-1"), doc_id("doc-4")]);
    assert_eq!(fetched.len(), 2);
}
