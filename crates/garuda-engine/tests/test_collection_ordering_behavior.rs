mod common;

use common::{build_doc, database, default_options, default_schema, dense_vector, field_name};
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
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(3),
        ))
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
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
            common::top_k(10),
        ))
        .expect("query");
    let mut ids = results.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), results.len());
}

#[test]
fn top_k_should_apply_public_doc_id_tie_break_after_segment_merge() {
    let (_root, db) = database("ordering-public-doc-id-tie-break");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let results = collection.insert(vec![
        build_doc("doc-b", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]),
        build_doc("doc-a", 2, "alpha", 0.8, [1.0, 0.0, 0.0, 0.0]),
    ]);
    assert!(results.iter().all(|result| result.status.is_ok()));

    let results = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(1),
        ))
        .expect("query");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id.as_str(), "doc-a");
}
