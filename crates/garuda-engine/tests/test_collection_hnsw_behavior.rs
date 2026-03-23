mod common;

use common::{
    build_doc, database, default_options, default_schema, dense_vector, doc_id, field_name,
    seed_collection, top_k,
};
use garuda_types::{
    CollectionOptions, HnswEfConstruction, HnswEfSearch, HnswIndexParams, HnswM,
    HnswMinNeighborCount, HnswPruneWidth, HnswScalingFactor, IndexParams, StatusCode, VectorQuery,
};

#[test]
fn create_index_rebuilds_hnsw_with_new_scaling_factor() {
    let (_root, db) = database("hnsw-scaling");
    let collection = db
        .create_collection(default_schema("docs"), single_segment_options())
        .expect("create collection");
    seed_scaling_fixture(&collection);

    let flat = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.47413385, 0.9022627, 0.0, 0.0]),
            top_k(1),
        ))
        .expect("flat query");
    assert_eq!(flat[0].id.as_str(), "doc-4");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(hnsw_params(1, 50, 1, 1, 1, 1)),
        )
        .expect("create high scaling hnsw index");

    let high_scaling = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.47413385, 0.9022627, 0.0, 0.0]),
            top_k(1),
        ))
        .expect("high scaling query");
    assert_eq!(high_scaling[0].id.as_str(), "doc-5");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(hnsw_params(1, 2, 1, 1, 1, 1)),
        )
        .expect("create low scaling hnsw index");

    let low_scaling = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.47413385, 0.9022627, 0.0, 0.0]),
            top_k(1),
        ))
        .expect("low scaling query");
    assert_eq!(low_scaling[0].id.as_str(), "doc-4");
}

#[test]
fn create_index_rebuilds_hnsw_with_new_prune_width() {
    let (_root, db) = database("hnsw-prune-width");
    let collection = db
        .create_collection(default_schema("docs"), single_segment_options())
        .expect("create collection");
    seed_prune_fixture(&collection);

    let flat = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![-0.4794442, 0.9057386, 0.0, 0.0]),
            top_k(3),
        ))
        .expect("flat query");
    assert_eq!(
        flat.iter().map(|doc| doc.id.as_str()).collect::<Vec<_>>(),
        vec!["doc-6", "doc-5", "doc-2"]
    );

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(hnsw_params(2, 50, 4, 1, 2, 4)),
        )
        .expect("create narrow prune hnsw index");

    let narrow_prune = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![-0.4794442, 0.9057386, 0.0, 0.0]),
            top_k(3),
        ))
        .expect("narrow prune query");
    assert_eq!(
        narrow_prune
            .iter()
            .map(|doc| doc.id.as_str())
            .collect::<Vec<_>>(),
        vec!["doc-6", "doc-2", "doc-1"]
    );

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(hnsw_params(2, 50, 4, 4, 2, 4)),
        )
        .expect("create wide prune hnsw index");

    let wide_prune = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![-0.4794442, 0.9057386, 0.0, 0.0]),
            top_k(3),
        ))
        .expect("wide prune query");
    assert_eq!(
        wide_prune
            .iter()
            .map(|doc| doc.id.as_str())
            .collect::<Vec<_>>(),
        vec!["doc-6", "doc-5", "doc-2"]
    );
}

#[test]
fn create_index_rejects_hnsw_min_neighbor_count_above_max_neighbors() {
    let (_root, db) = database("hnsw-invalid-min-neighbors");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let status = collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(hnsw_params(2, 50, 8, 8, 3, 8)),
        )
        .expect_err("invalid hnsw params should fail");
    assert_eq!(status.code, StatusCode::InvalidArgument);
}

#[test]
fn delete_on_persisted_segment_updates_hnsw_results() {
    let (_root, db) = database("hnsw-persisted-delete");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("create hnsw index");

    let deleted = collection.delete(vec![doc_id("doc-1")]);
    assert!(deleted.iter().all(|result| result.status.is_ok()));

    let results = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            top_k(1),
        ))
        .expect("query after delete");

    assert_eq!(results[0].id.as_str(), "doc-2");
}

#[test]
fn delete_on_persisted_segment_with_small_ef_search_still_returns_live_hit() {
    let (_root, db) = database("hnsw-persisted-delete-small-ef-search");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("create hnsw index");

    let deleted = collection.delete(vec![doc_id("doc-1")]);
    assert!(deleted.iter().all(|result| result.status.is_ok()));

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        top_k(1),
    );
    query.search = garuda_types::VectorSearch::Hnsw {
        ef_search: HnswEfSearch::new(1).expect("valid ef_search"),
    };

    let results = collection.query(query).expect("query after delete");
    assert_eq!(results[0].id.as_str(), "doc-2");
}

#[test]
fn update_on_persisted_segment_with_small_ef_search_removes_old_vector() {
    let (_root, db) = database("hnsw-persisted-update-small-ef-search");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("create hnsw index");

    let updated = collection.update(vec![build_doc(
        "doc-1",
        1,
        "alpha",
        0.9,
        [0.0, 1.0, 0.0, 0.0],
    )]);
    assert!(updated.iter().all(|result| result.status.is_ok()));

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        top_k(1),
    );
    query.search = garuda_types::VectorSearch::Hnsw {
        ef_search: HnswEfSearch::new(1).expect("valid ef_search"),
    };

    let results = collection.query(query).expect("query after update");
    assert_eq!(results[0].id.as_str(), "doc-2");
}

#[test]
fn upsert_on_writing_segment_updates_hnsw_results() {
    let (_root, db) = database("hnsw-writing-upsert");
    let collection = db
        .create_collection(default_schema("docs"), single_segment_options())
        .expect("create collection");
    seed_collection(&collection);

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("create hnsw index");

    let upserted = collection.upsert(vec![build_doc(
        "doc-1",
        1,
        "alpha",
        0.9,
        [0.0, 1.0, 0.0, 0.0],
    )]);
    assert!(upserted.iter().all(|result| result.status.is_ok()));

    let results = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            top_k(1),
        ))
        .expect("query after upsert");

    assert_eq!(results[0].id.as_str(), "doc-2");
}

fn hnsw_params(
    max_neighbors: u32,
    scaling_factor: u32,
    ef_construction: u32,
    prune_width: u32,
    min_neighbor_count: u32,
    ef_search: u32,
) -> HnswIndexParams {
    HnswIndexParams {
        max_neighbors: HnswM::new(max_neighbors).expect("valid max_neighbors"),
        scaling_factor: HnswScalingFactor::new(scaling_factor).expect("valid scaling_factor"),
        ef_construction: HnswEfConstruction::new(ef_construction).expect("valid ef_construction"),
        prune_width: HnswPruneWidth::new(prune_width).expect("valid prune_width"),
        min_neighbor_count: HnswMinNeighborCount::new(min_neighbor_count)
            .expect("valid min_neighbor_count"),
        ef_search: HnswEfSearch::new(ef_search).expect("valid ef_search"),
    }
}

fn seed_scaling_fixture(collection: &garuda_engine::Collection) {
    let docs = vec![
        build_doc(
            "doc-1",
            1,
            "alpha",
            0.1,
            [-0.16671813, -0.9393654, 0.0, 0.0],
        ),
        build_doc(
            "doc-2",
            2,
            "alpha",
            0.2,
            [-0.7148936, -0.38582748, 0.0, 0.0],
        ),
        build_doc("doc-3", 3, "beta", 0.3, [-0.87854815, -0.7147157, 0.0, 0.0]),
        build_doc("doc-4", 4, "beta", 0.4, [0.3036996, 0.08503449, 0.0, 0.0]),
        build_doc("doc-5", 5, "gamma", 0.5, [0.23736966, -0.4550056, 0.0, 0.0]),
    ];
    let results = collection.insert(docs);
    assert!(results.iter().all(|result| result.status.is_ok()));
}

fn seed_prune_fixture(collection: &garuda_engine::Collection) {
    let docs = vec![
        build_doc(
            "doc-1",
            1,
            "alpha",
            0.1,
            [-0.24059594, -0.12569392, 0.0, 0.0],
        ),
        build_doc(
            "doc-2",
            2,
            "alpha",
            0.2,
            [-0.70171404, -0.30555004, 0.0, 0.0],
        ),
        build_doc(
            "doc-3",
            3,
            "beta",
            0.3,
            [-0.038252234, -0.33283275, 0.0, 0.0],
        ),
        build_doc("doc-4", 4, "beta", 0.4, [0.1485604, -0.48624796, 0.0, 0.0]),
        build_doc("doc-5", 5, "gamma", 0.5, [0.5103195, 0.35472643, 0.0, 0.0]),
        build_doc(
            "doc-6",
            6,
            "gamma",
            0.6,
            [-0.61180323, 0.73381424, 0.0, 0.0],
        ),
    ];
    let results = collection.insert(docs);
    assert!(results.iter().all(|result| result.status.is_ok()));
}

fn single_segment_options() -> CollectionOptions {
    let mut options = default_options();
    options.segment_max_docs = 16;
    options
}
