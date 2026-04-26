mod common;

use common::{
    build_doc, collection_name, database, default_options, default_schema, dense_vector, doc_id,
    field_name, seed_collection,
};
use garuda_types::{OptimizeOptions, ScalarValue, StatusCode, VectorQuery};
use std::thread;

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
            common::top_k(4),
        ))
        .expect("query before");
    collection.optimize(OptimizeOptions).expect("optimize");
    let after = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(4),
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
fn optimize_persists_without_extra_flush() {
    let (_root, db) = database("optimize-no-extra-flush");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    collection.optimize(OptimizeOptions).expect("optimize");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    let fetched = reopened.fetch(vec![
        doc_id("doc-1"),
        doc_id("doc-2"),
        doc_id("doc-3"),
        doc_id("doc-4"),
    ]);

    assert_eq!(fetched.len(), 4);
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
        .optimize(OptimizeOptions)
        .expect("optimize with default options");
    let after = collection.stats();

    assert_eq!(before.doc_count, after.doc_count);
    assert!(after.segment_count >= 1);
}

#[test]
fn optimize_racing_write_remains_visible_and_durable() {
    let (_root, db) = database("optimize-racing-write");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    let docs = (0..1_000)
        .map(|rank| {
            build_doc(
                &format!("doc-{rank}"),
                rank,
                "bulk",
                rank as f64,
                [1.0, rank as f32, 0.0, 0.0],
            )
        })
        .collect();
    let results = collection.insert(docs);
    assert!(results.iter().all(|result| result.status.is_ok()));

    let optimizer = collection.clone();
    let optimize = thread::spawn(move || optimizer.optimize(OptimizeOptions));
    let results = collection.insert(vec![build_doc(
        "racing-doc",
        1_001,
        "race",
        1_001.0,
        [0.0, 1.0, 0.0, 0.0],
    )]);
    assert!(results[0].status.is_ok());

    if let Err(status) = optimize.join().expect("optimize thread") {
        assert_eq!(status.code, StatusCode::FailedPrecondition);
        assert!(status.message.contains("concurrent collection mutation"));
    }

    assert!(
        collection
            .fetch(vec![doc_id("racing-doc")])
            .contains_key(&doc_id("racing-doc"))
    );
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    assert!(
        reopened
            .fetch(vec![doc_id("racing-doc")])
            .contains_key(&doc_id("racing-doc"))
    );
}
