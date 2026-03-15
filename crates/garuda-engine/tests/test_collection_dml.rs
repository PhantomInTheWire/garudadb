mod common;

use common::{build_doc, database, default_options, default_schema, seed_collection};

#[test]
fn insert_fetch_update_upsert_delete_flow_works() {
    let (_root, db) = database("dml");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let insert = collection.insert(vec![build_doc(
        "doc-1",
        1,
        "alpha",
        0.9,
        [1.0, 0.0, 0.0, 0.0],
    )]);
    assert!(insert[0].status.is_ok());
    assert_eq!(collection.fetch(vec!["doc-1".to_string()]).len(), 1);

    let update = collection.update(vec![build_doc(
        "doc-1",
        11,
        "alpha",
        0.99,
        [1.0, 0.0, 0.0, 0.0],
    )]);
    assert!(update[0].status.is_ok());
    let fetched = collection.fetch(vec!["doc-1".to_string()]);
    assert_eq!(
        fetched["doc-1"].fields["rank"],
        garuda_types::ScalarValue::Int64(11)
    );

    let upsert = collection.upsert(vec![build_doc(
        "doc-2",
        2,
        "beta",
        0.8,
        [0.0, 1.0, 0.0, 0.0],
    )]);
    assert!(upsert[0].status.is_ok());
    assert_eq!(collection.stats().doc_count, 2);

    let delete = collection.delete(vec!["doc-1".to_string()]);
    assert!(delete[0].status.is_ok());
    assert!(collection.fetch(vec!["doc-1".to_string()]).is_empty());
}

#[test]
fn insert_rolls_over_segments() {
    let (_root, db) = database("dml-rollover");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    assert!(collection.stats().segment_count >= 2);
}
