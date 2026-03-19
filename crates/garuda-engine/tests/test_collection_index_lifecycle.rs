mod common;

use common::{
    collection_name, database, default_options, default_schema, dense_vector, field_name,
    seed_collection, seed_more_collection_docs,
};
use garuda_storage::{WRITING_SEGMENT_ID, segment_flat_index_path};
use garuda_types::{
    FlatIndexParams, HnswIndexParams, IndexKind, IndexParams, SegmentId, VectorQuery,
};

const FIRST_PERSISTED_SEGMENT_ID: SegmentId = SegmentId::new_unchecked(1);

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
            common::top_k(5),
        ))
        .expect("indexed query");

    collection
        .drop_index(&field_name("embedding"))
        .expect("drop index");
    let flat = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(5),
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

#[test]
fn flat_index_sidecars_should_survive_reopen_and_hnsw_switch_should_remove_them() {
    let (root, db) = database("flat-sidecars");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);
    collection.flush().expect("flush");

    assert!(
        root.join("docs").join("1").exists(),
        "persisted segment should exist"
    );
    assert!(segment_flat_index_path(&root.join("docs"), FIRST_PERSISTED_SEGMENT_ID).exists());
    assert!(
        !segment_flat_index_path(&root.join("docs"), WRITING_SEGMENT_ID).exists(),
        "writing segment should stay record-backed"
    );

    let before = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(5),
        ))
        .expect("query before reopen");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen");
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

    reopened
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("switch to hnsw compatibility mode");
    assert!(
        !segment_flat_index_path(&root.join("docs"), FIRST_PERSISTED_SEGMENT_ID).exists(),
        "non-flat index mode should remove persisted flat sidecars"
    );
}

#[test]
fn create_flat_index_on_existing_persisted_data_should_create_sidecars_and_preserve_results() {
    let (root, db) = database("flat-create-existing");
    let mut schema = default_schema("docs");
    schema.vector.index = IndexParams::Hnsw(HnswIndexParams::default());

    let collection = db
        .create_collection(schema, default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);
    collection.flush().expect("flush");

    let collection_root = root.join("docs");
    assert!(
        !segment_flat_index_path(&collection_root, FIRST_PERSISTED_SEGMENT_ID).exists(),
        "non-flat schema should not persist flat sidecars"
    );

    let before = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query before create_index(flat)");

    collection
        .create_index(&field_name("embedding"), IndexParams::Flat(FlatIndexParams))
        .expect("create flat index");

    assert!(segment_flat_index_path(&collection_root, FIRST_PERSISTED_SEGMENT_ID).exists());

    let after = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query after create_index(flat)");

    assert_eq!(
        before.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}
