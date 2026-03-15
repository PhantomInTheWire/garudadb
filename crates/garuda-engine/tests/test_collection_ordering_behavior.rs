mod common;

use common::{database, default_options, default_schema};
use garuda_types::VectorQuery;

#[test]
fn top_k_should_cap_results_without_returning_more_documents() {
    let (_root, db) = database("ordering-topk");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    common::seed_collection(&collection);
    common::seed_more_collection_docs(&collection);

    let results = collection
        .query(VectorQuery::by_vector("embedding", vec![1.0, 0.0, 0.0, 0.0], 3))
        .expect("query");
    assert!(results.len() <= 3);
}

#[test]
fn repeated_queries_should_not_duplicate_documents_in_one_result_set() {
    let (_root, db) = database("ordering-no-duplicates");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    common::seed_collection(&collection);
    common::seed_more_collection_docs(&collection);

    let results = collection
        .query(VectorQuery::by_vector("embedding", vec![0.0, 1.0, 0.0, 0.0], 10))
        .expect("query");
    let mut ids = results.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), results.len());
}
