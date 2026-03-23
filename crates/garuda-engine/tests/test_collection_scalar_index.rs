mod common;

use common::{
    database, default_options, default_schema, dense_vector, doc_id, field_name, seed_collection,
    seed_more_collection_docs, top_k,
};
use garuda_types::{IndexParams, ScalarIndexParams, VectorQuery};

#[test]
fn create_scalar_index_on_persisted_string_field_survives_reopen_and_filters_query() {
    let (_root, db) = database("scalar-index-persisted-string");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");

    collection
        .create_index(
            &field_name("category"),
            IndexParams::Scalar(ScalarIndexParams),
        )
        .expect("create scalar index on category");
    collection.flush().expect("flush indexed collection");
    drop(collection);

    let reopened = db
        .open_collection(&common::collection_name("docs"))
        .expect("reopen collection");

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        top_k(10),
    );
    query.filter = Some("category = 'alpha'".to_string());

    let results = reopened.query(query).expect("query with scalar filter");
    let result_ids: Vec<_> = results.into_iter().map(|doc| doc.id).collect();
    assert_eq!(result_ids, vec![doc_id("doc-1"), doc_id("doc-2")]);
}

#[test]
fn create_scalar_index_on_writing_int_field_supports_range_filter() {
    let (_root, db) = database("scalar-index-writing-range");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    collection
        .create_index(&field_name("rank"), IndexParams::Scalar(ScalarIndexParams))
        .expect("create scalar index on rank");

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
        top_k(10),
    );
    query.filter = Some("rank >= 6".to_string());

    let results = collection.query(query).expect("query with range filter");
    let result_ids: Vec<_> = results.into_iter().map(|doc| doc.id).collect();
    assert_eq!(
        result_ids,
        vec![doc_id("doc-6"), doc_id("doc-7"), doc_id("doc-8")]
    );
}
