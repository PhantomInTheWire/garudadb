mod common;

use common::{
    collection_name, database, default_options, default_schema, dense_vector, field_name,
    seed_collection, seed_more_collection_docs,
};
use garuda_types::{HnswIndexParams, IndexKind, IndexParams, VectorQuery};

#[test]
fn creating_index_before_and_after_data_should_preserve_logical_results() {
    let (_root, db) = database("index-lifecycle");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("create empty index");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let indexed = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            5,
        ))
        .expect("indexed query");

    collection
        .drop_index(&field_name("embedding"))
        .expect("drop index");
    let flat = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            5,
        ))
        .expect("flat query");

    assert_eq!(
        indexed.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        flat.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn index_choice_should_survive_reopen() {
    let (_root, db) = database("index-reopen");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("create hnsw");
    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen");
    assert_eq!(reopened.schema().vector.index.kind(), IndexKind::Hnsw);
}

#[test]
fn hnsw_index_params_should_roundtrip_through_reopen() {
    let (_root, db) = database("index-params-roundtrip");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let params = HnswIndexParams {
        m: 32,
        ef_construction: 128,
        ef_search: 96,
    };

    collection
        .create_index(&field_name("embedding"), IndexParams::Hnsw(params.clone()))
        .expect("create hnsw with custom params");
    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen");
    assert_eq!(reopened.schema().vector.index, IndexParams::Hnsw(params));
}
