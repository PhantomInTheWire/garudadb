mod common;

use common::{
    collection_name, database, default_options, default_schema, dense_vector, doc_id, field_name,
    seed_collection, seed_more_collection_docs,
};
use garuda_types::{OptimizeOptions, VectorQuery};

#[test]
fn flat_query_results_should_survive_reopen_with_identical_ordering() {
    let (_root, db) = database("flat-reopen");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let before = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(5),
        ))
        .expect("query before reopen");
    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    let after = reopened
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(5),
        ))
        .expect("query after reopen");

    assert_eq!(
        before.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn flat_query_should_merge_exact_results_across_multiple_segments() {
    let (_root, db) = database("flat-multisegment");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let results = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
            common::top_k(3),
        ))
        .expect("query");

    assert_eq!(
        results.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        vec![doc_id("doc-3"), doc_id("doc-6"), doc_id("doc-7")]
    );
}

#[test]
fn flat_query_by_document_id_should_match_query_by_vector() {
    let (_root, db) = database("flat-query-by-id");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let by_vector = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query by vector");
    let by_id = collection
        .query(VectorQuery::by_id(
            field_name("embedding"),
            doc_id("doc-1"),
            common::top_k(4),
        ))
        .expect("query by id");

    assert_eq!(
        by_vector
            .iter()
            .map(|doc| doc.id.clone())
            .collect::<Vec<_>>(),
        by_id.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn optimize_should_preserve_exact_flat_query_results() {
    let (_root, db) = database("flat-optimize");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let before = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query before optimize");

    collection
        .optimize(OptimizeOptions)
        .expect("optimize collection");

    let after = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query after optimize");

    assert_eq!(
        before.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}
