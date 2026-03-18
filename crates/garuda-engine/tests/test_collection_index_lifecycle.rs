mod common;

use common::{
    collection_name, database, default_options, default_schema, dense_vector, field_name,
    seed_collection, seed_more_collection_docs,
};
use garuda_types::{IndexKind, VectorQuery};

#[test]
fn creating_index_before_and_after_data_should_preserve_logical_results() {
    let (_root, db) = database("index-lifecycle");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    collection
        .create_index(&field_name("embedding"), IndexKind::Hnsw)
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
        .create_index(&field_name("embedding"), IndexKind::Hnsw)
        .expect("create hnsw");
    collection.flush().expect("flush");

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen");
    assert_eq!(reopened.schema().vector.index.kind(), IndexKind::Hnsw);
}
