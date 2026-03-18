mod common;

use common::{build_doc, collection_name, database, default_options, default_schema, doc_id};

#[test]
fn reopen_uses_latest_visible_document_for_a_reused_doc_id() {
    let (_root, db) = database("storage-id-map-latest");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let first = build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]);
    let second = build_doc("doc-1", 9, "beta", 0.1, [0.0, 1.0, 0.0, 0.0]);

    let insert = collection.insert(vec![first]);
    assert!(insert[0].status.is_ok());

    let upsert = collection.upsert(vec![second]);
    assert!(upsert[0].status.is_ok());

    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    let fetched = reopened.fetch(vec![doc_id("doc-1")]);
    let doc = fetched.get(&doc_id("doc-1")).expect("doc present");

    assert_eq!(doc.fields["rank"], garuda_types::ScalarValue::Int64(9));
    assert_eq!(
        doc.fields["category"],
        garuda_types::ScalarValue::String(String::from("beta"))
    );
}

#[test]
fn delete_then_reinsert_same_doc_id_survives_reopen() {
    let (_root, db) = database("storage-id-map-reinsert");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let first = build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]);
    let second = build_doc("doc-1", 7, "gamma", 0.4, [0.0, 0.0, 1.0, 0.0]);

    let insert = collection.insert(vec![first]);
    assert!(insert[0].status.is_ok());

    let delete = collection.delete(vec![doc_id("doc-1")]);
    assert!(delete[0].status.is_ok());

    let reinsert = collection.insert(vec![second]);
    assert!(reinsert[0].status.is_ok());

    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    let fetched = reopened.fetch(vec![doc_id("doc-1")]);
    let doc = fetched.get(&doc_id("doc-1")).expect("doc present");

    assert_eq!(doc.fields["rank"], garuda_types::ScalarValue::Int64(7));
    assert_eq!(
        doc.fields["category"],
        garuda_types::ScalarValue::String(String::from("gamma"))
    );
}
