mod common;

use common::{
    database, default_options, default_schema, dense_vector, doc_id, field_name, seed_collection,
    seed_more_collection_docs,
};
use garuda_types::{OptimizeOptions, VectorQuery};

#[test]
fn fetch_and_query_should_span_multiple_segments_transparently() {
    let (_root, db) = database("multisegment");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let fetched = collection.fetch(vec![doc_id("doc-1"), doc_id("doc-4"), doc_id("doc-8")]);
    assert_eq!(fetched.len(), 3);

    let results = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
            common::top_k(8),
        ))
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

    collection
        .optimize(OptimizeOptions)
        .expect("first optimize");
    let after_first = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(8),
        ))
        .expect("query after first");

    collection
        .optimize(OptimizeOptions)
        .expect("second optimize");
    let after_second = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(8),
        ))
        .expect("query after second");

    assert_eq!(
        after_first
            .iter()
            .map(|doc| doc.id.clone())
            .collect::<Vec<_>>(),
        after_second
            .iter()
            .map(|doc| doc.id.clone())
            .collect::<Vec<_>>()
    );
}
