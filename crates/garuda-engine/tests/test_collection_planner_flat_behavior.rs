mod common;

use common::{
    collection_name, database, default_options, default_schema, dense_vector, doc_id, field_name,
    seed_collection, seed_more_collection_docs,
};
use garuda_types::{OptimizeOptions, VectorQuery};

#[test]
fn filtered_flat_query_should_merge_exact_results_across_segments() {
    let (_root, db) = database("planner-flat-filtered-merge");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
        common::top_k(3),
    );
    query.filter = Some("category = 'beta'".to_string());

    let results = collection.query(query).expect("filtered query");

    assert_eq!(
        results.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        vec![doc_id("doc-3"), doc_id("doc-6"), doc_id("doc-7")]
    );
}

#[test]
fn filtered_flat_query_should_keep_same_results_after_reopen_and_optimize() {
    let (_root, db) = database("planner-flat-filtered-stability");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let before = filtered_beta_query(&collection);
    collection.flush().expect("flush");
    collection.optimize(OptimizeOptions).expect("optimize");
    let after_optimize = filtered_beta_query(&collection);
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    let after_reopen = filtered_beta_query(&reopened);

    let before_ids: Vec<_> = before.iter().map(|doc| doc.id.clone()).collect();
    let optimize_ids: Vec<_> = after_optimize.iter().map(|doc| doc.id.clone()).collect();
    let reopen_ids: Vec<_> = after_reopen.iter().map(|doc| doc.id.clone()).collect();

    assert_eq!(before_ids, optimize_ids);
    assert_eq!(before_ids, reopen_ids);
}

fn filtered_beta_query(collection: &garuda_engine::Collection) -> Vec<garuda_types::Doc> {
    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
        common::top_k(3),
    );
    query.filter = Some("category = 'beta'".to_string());

    collection.query(query).expect("filtered query")
}
