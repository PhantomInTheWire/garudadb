mod common;

use common::{database, default_options, default_schema, seed_collection, seed_more_collection_docs};
use garuda_types::VectorQuery;

#[test]
fn fetch_and_query_should_span_multiple_segments_transparently() {
    let (_root, db) = database("multisegment");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let fetched = collection.fetch(vec![
        "doc-1".to_string(),
        "doc-4".to_string(),
        "doc-8".to_string(),
    ]);
    assert_eq!(fetched.len(), 3);

    let results = collection
        .query(VectorQuery::by_vector("embedding", vec![0.0, 1.0, 0.0, 0.0], 8))
        .expect("query");
    assert!(results.len() >= 3);
}

#[test]
fn optimize_should_be_idempotent_at_the_logical_api_level() {
    let (_root, db) = database("multisegment-optimize");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    collection.optimize().expect("first optimize");
    let after_first = collection
        .query(VectorQuery::by_vector("embedding", vec![1.0, 0.0, 0.0, 0.0], 8))
        .expect("query after first");

    collection.optimize().expect("second optimize");
    let after_second = collection
        .query(VectorQuery::by_vector("embedding", vec![1.0, 0.0, 0.0, 0.0], 8))
        .expect("query after second");

    assert_eq!(
        after_first.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after_second.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}
