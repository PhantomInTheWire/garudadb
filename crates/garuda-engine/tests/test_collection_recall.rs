mod common;

use common::{database, default_options, default_schema, seed_collection};
use garuda_types::{IndexKind, VectorQuery};

#[test]
fn hnsw_matches_flat_ground_truth_for_small_fixture() {
    let (_root, db) = database("recall");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let flat = collection
        .query(VectorQuery::by_vector(
            "embedding",
            vec![1.0, 0.0, 0.0, 0.0],
            3,
        ))
        .expect("flat query");
    collection
        .create_index("embedding", IndexKind::Hnsw)
        .expect("create hnsw");
    let hnsw = collection
        .query(VectorQuery::by_vector(
            "embedding",
            vec![1.0, 0.0, 0.0, 0.0],
            3,
        ))
        .expect("hnsw query");

    let flat_ids: Vec<_> = flat.iter().map(|doc| doc.id.clone()).collect();
    let hnsw_ids: Vec<_> = hnsw.iter().map(|doc| doc.id.clone()).collect();
    assert_eq!(flat_ids, hnsw_ids);
}
