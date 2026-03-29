mod common;

use common::{
    build_doc, collection_name, database, default_options, default_schema, dense_vector, doc_id,
    field_name, seed_collection, seed_more_collection_docs,
};
use garuda_storage::{WRITING_SEGMENT_ID, segment_ivf_index_path};
use garuda_types::{
    IndexKind, IndexParams, IvfIndexParams, IvfProbeCount, VectorIndexState, VectorQuery,
    VectorSearch,
};

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

#[test]
fn ivf_writing_segment_should_return_each_inserted_record_once() {
    let (_root, db) = database("ivf-writing-segment");
    let mut schema = default_schema("docs");
    schema.vector.indexes = VectorIndexState::IvfOnly(IvfIndexParams::default());
    let mut options = default_options();
    options.segment_max_docs = 100;

    let collection = db
        .create_collection(schema, options)
        .expect("create collection");

    let results = collection.insert(vec![
        build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]),
        build_doc("doc-2", 2, "alpha", 0.8, [0.9, 0.1, 0.0, 0.0]),
        build_doc("doc-3", 3, "beta", 0.7, [0.0, 1.0, 0.0, 0.0]),
        build_doc("doc-4", 4, "gamma", 0.6, [0.0, 0.0, 1.0, 0.0]),
    ]);
    assert!(results.iter().all(|result| result.status.is_ok()));
    assert_eq!(collection.stats().doc_count, 4);

    let docs = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query");

    assert_eq!(docs.len(), 4);
    assert_eq!(
        docs.iter()
            .map(|doc| doc.id.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        4
    );
}

#[test]
fn ivf_writing_segment_should_preserve_results_after_flush() {
    let (_root, db) = database("ivf-writing-flush");
    let mut schema = default_schema("docs");
    schema.vector.indexes = VectorIndexState::IvfOnly(IvfIndexParams::default());
    let mut options = default_options();
    options.segment_max_docs = 100;

    let collection = db
        .create_collection(schema, options)
        .expect("create collection");

    let results = collection.insert(vec![
        build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]),
        build_doc("doc-2", 2, "alpha", 0.8, [0.9, 0.1, 0.0, 0.0]),
        build_doc("doc-3", 3, "beta", 0.7, [0.0, 1.0, 0.0, 0.0]),
        build_doc("doc-4", 4, "gamma", 0.6, [0.0, 0.0, 1.0, 0.0]),
    ]);
    assert!(results.iter().all(|result| result.status.is_ok()));

    let before = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query before flush");

    collection.flush().expect("flush");

    let after = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("query after flush");

    assert_eq!(
        before.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn delete_on_persisted_segment_with_small_nprobe_still_returns_live_hit() {
    let (_root, db) = database("ivf-persisted-delete-small-nprobe");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Ivf(IvfIndexParams::default()),
        )
        .expect("create ivf index");

    let deleted = collection.delete(vec![doc_id("doc-1")]);
    assert!(deleted.iter().all(|result| result.status.is_ok()));

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(1),
    );
    query.search = VectorSearch::Ivf {
        nprobe: IvfProbeCount::new(1).expect("valid nprobe"),
    };

    let results = collection.query(query).expect("query after delete");
    assert_eq!(results[0].id.as_str(), "doc-2");
}
