mod common;

use common::{
    database, default_options, default_schema, dense_vector, field_name, seed_collection,
};
use garuda_types::{HnswIndexParams, IndexParams, IvfIndexParams, VectorQuery};

#[test]
fn hnsw_matches_flat_ground_truth_for_small_fixture() {
    let (_root, db) = database("recall");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let flat = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(3),
        ))
        .expect("flat query");
    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("create hnsw");
    let hnsw = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(3),
        ))
        .expect("hnsw query");

    let flat_ids: Vec<_> = flat.iter().map(|doc| doc.id.clone()).collect();
    let hnsw_ids: Vec<_> = hnsw.iter().map(|doc| doc.id.clone()).collect();
    assert_eq!(flat_ids, hnsw_ids);
}

#[test]
fn ivf_matches_flat_ground_truth_for_small_fixture() {
    let (_root, db) = database("ivf-recall");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let flat = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(3),
        ))
        .expect("flat query");
    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Ivf(IvfIndexParams::default()),
        )
        .expect("create ivf");
    let ivf = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(3),
        ))
        .expect("ivf query");

    let flat_ids: Vec<_> = flat.iter().map(|doc| doc.id.clone()).collect();
    let ivf_ids: Vec<_> = ivf.iter().map(|doc| doc.id.clone()).collect();
    assert_eq!(flat_ids, ivf_ids);
}

#[test]
fn filtered_hnsw_matches_flat_ground_truth_for_small_fixture() {
    let (_root, db) = database("filtered-hnsw-recall");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let mut flat_query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(2),
    );
    flat_query.filter = Some("category = 'alpha'".to_string());
    let flat = collection.query(flat_query).expect("flat filtered query");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("create hnsw");

    let mut hnsw_query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(2),
    );
    hnsw_query.filter = Some("category = 'alpha'".to_string());
    let hnsw = collection.query(hnsw_query).expect("hnsw filtered query");

    let flat_ids: Vec<_> = flat.iter().map(|doc| doc.id.clone()).collect();
    let hnsw_ids: Vec<_> = hnsw.iter().map(|doc| doc.id.clone()).collect();
    assert_eq!(flat_ids, hnsw_ids);
}

#[test]
fn filtered_ivf_matches_flat_ground_truth_for_small_fixture() {
    let (_root, db) = database("filtered-ivf-recall");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let mut flat_query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(2),
    );
    flat_query.filter = Some("category = 'alpha'".to_string());
    let flat = collection.query(flat_query).expect("flat filtered query");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Ivf(IvfIndexParams::default()),
        )
        .expect("create ivf");

    let mut ivf_query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(2),
    );
    ivf_query.filter = Some("category = 'alpha'".to_string());
    let ivf = collection.query(ivf_query).expect("ivf filtered query");

    let flat_ids: Vec<_> = flat.iter().map(|doc| doc.id.clone()).collect();
    let ivf_ids: Vec<_> = ivf.iter().map(|doc| doc.id.clone()).collect();
    assert_eq!(flat_ids, ivf_ids);
}
