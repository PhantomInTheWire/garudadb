mod common;

use common::{
    collection_name, database, default_options, default_schema, dense_vector, field_name,
    seed_collection, seed_more_collection_docs,
};
use garuda_storage::{WRITING_SEGMENT_ID, segment_ivf_index_path};
use garuda_types::{IndexKind, IndexParams, IvfIndexParams, VectorIndexState, VectorQuery};

const FIRST_PERSISTED_SEGMENT_ID: garuda_types::SegmentId =
    garuda_types::SegmentId::new_unchecked(1);

#[test]
fn create_index_reopen_and_drop_should_roundtrip_ivf_sidecars() {
    let (root, db) = database("ivf-sidecars");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);
    collection.flush().expect("flush");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Ivf(IvfIndexParams::default()),
        )
        .expect("create ivf");

    let collection_root = root.join("docs");
    assert!(segment_ivf_index_path(&collection_root, FIRST_PERSISTED_SEGMENT_ID).exists());
    assert!(!segment_ivf_index_path(&collection_root, WRITING_SEGMENT_ID).exists());

    let before = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query before reopen");
    drop(collection);

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen");
    assert_eq!(
        reopened.schema().vector.indexes.default_kind(),
        IndexKind::Ivf
    );

    let after = reopened
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query after reopen");
    assert_eq!(
        before.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );

    reopened
        .drop_index(&field_name("embedding"), IndexKind::Ivf)
        .expect("drop ivf");
    assert!(!segment_ivf_index_path(&collection_root, FIRST_PERSISTED_SEGMENT_ID).exists());
}

#[test]
fn create_flat_index_on_ivf_only_collection_should_preserve_results() {
    let (root, db) = database("ivf-flat-enable");
    let mut schema = default_schema("docs");
    schema.vector.indexes = VectorIndexState::IvfOnly(IvfIndexParams::default());

    let collection = db
        .create_collection(schema, default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);
    collection.flush().expect("flush");

    let before = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query before create flat");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Flat(garuda_types::FlatIndexParams),
        )
        .expect("create flat");

    assert!(root.join("docs").join("1").exists());
    let after = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query after create flat");

    assert_eq!(
        before.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}
