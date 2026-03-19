mod common;

use common::{
    database, default_options, default_schema, dense_vector, doc_id, field_name, seed_collection,
};
use garuda_types::{OptimizeOptions, ScalarValue, VectorQuery};

#[test]
fn optimize_preserves_query_results() {
    let (_root, db) = database("optimize");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let before = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            4,
        ))
        .expect("query before");
    collection.optimize(OptimizeOptions).expect("optimize");
    let after = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            4,
        ))
        .expect("query after");

    assert_eq!(
        before.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn optimize_keeps_tail_segment_documents_visible() {
    let (_root, db) = database("optimize-tail");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    collection.optimize(OptimizeOptions).expect("optimize");

    let fetched = collection.fetch(vec![
        doc_id("doc-1"),
        doc_id("doc-2"),
        doc_id("doc-3"),
        doc_id("doc-4"),
    ]);

    assert_eq!(fetched.len(), 4);
}

#[test]
fn optimize_after_delete_then_reopen_keeps_live_set_intact() {
    let (_root, db) = database("optimize-delete-reopen");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let deleted = collection.delete(vec![doc_id("doc-2")]);
    assert!(deleted[0].status.is_ok());

    collection.optimize(OptimizeOptions).expect("optimize");
    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&common::collection_name("docs"))
        .expect("reopen collection");

    assert!(reopened.fetch(vec![doc_id("doc-2")]).is_empty());

    let fetched = reopened.fetch(vec![doc_id("doc-1"), doc_id("doc-3"), doc_id("doc-4")]);
    assert_eq!(fetched.len(), 3);
    assert_eq!(
        fetched["doc-1"].fields["category"],
        ScalarValue::String("alpha".to_string())
    );
}

#[test]
fn optimize_default_options_should_preserve_current_behavior() {
    let (_root, db) = database("optimize-default-options");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let before = collection.stats();
    collection
        .optimize(OptimizeOptions::default())
        .expect("optimize with default options");
    let after = collection.stats();

    assert_eq!(before.doc_count, after.doc_count);
    assert!(after.segment_count >= 1);
}
