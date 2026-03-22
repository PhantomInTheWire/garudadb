mod common;

use common::{
    build_doc, database, default_options, default_schema, dense_vector, doc_id, field_name,
    seed_collection,
};
use garuda_types::{
    HnswEfConstruction, HnswEfSearch, HnswIndexParams, HnswM, HnswMinNeighborCount, HnswPruneWidth,
    HnswScalingFactor, IndexParams, VectorQuery,
};

#[test]
fn query_by_document_id_should_behave_like_query_by_its_stored_vector() {
    let (_root, db) = database("query-by-id");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let by_id = collection.query(VectorQuery::by_id(
        field_name("embedding"),
        doc_id("doc-1"),
        common::top_k(3),
    ));
    assert!(by_id.is_ok());
}

#[test]
fn query_by_document_id_should_match_query_by_vector_results() {
    let (_root, db) = database("query-by-id-parity");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

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
fn include_vector_and_output_fields_should_control_result_shape() {
    let (_root, db) = database("query-shape");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(3),
    );
    query.vector_projection = garuda_types::VectorProjection::Include;
    query.output_fields = Some(vec!["category".to_string()]);
    let results = collection.query(query).expect("query");
    assert!(!results.is_empty());
    assert!(!results[0].vector.is_empty());
    assert!(results[0].fields.contains_key("category"));
    assert!(!results[0].fields.contains_key("rank"));
}

#[test]
fn same_query_should_be_stable_across_repeated_calls() {
    let (_root, db) = database("query-stability");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let first = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("first query");
    let second = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            common::top_k(4),
        ))
        .expect("second query");

    assert_eq!(
        first.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        second.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn query_by_document_id_should_match_query_by_vector_results_under_hnsw() {
    let (_root, db) = database("query-by-id-hnsw-parity");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("create hnsw");

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
fn hnsw_query_should_honor_ef_search_override() {
    let (_root, db) = database("query-hnsw-ef-search");
    let collection = db
        .create_collection(default_schema("docs"), single_segment_options())
        .expect("create collection");
    seed_ef_fixture(&collection);
    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(hnsw_params(3, 50, 8, 8, 3, 1)),
        )
        .expect("create hnsw");

    let low_ef = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.4001776, 0.9763744, 0.0, 0.0]),
            common::top_k(1),
        ))
        .expect("low ef_search query");
    assert_eq!(low_ef[0].id, doc_id("doc-1"));

    let mut high_ef_query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![0.4001776, 0.9763744, 0.0, 0.0]),
        common::top_k(1),
    );
    high_ef_query.ef_search = Some(HnswEfSearch::new(8).expect("valid ef_search"));

    let high_ef = collection
        .query(high_ef_query)
        .expect("high ef_search query");
    assert_eq!(high_ef[0].id, doc_id("doc-5"));
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

fn seed_ef_fixture(collection: &garuda_engine::Collection) {
    let docs = vec![
        build_doc("doc-1", 1, "alpha", 0.1, [-0.8360677, 0.50163317, 0.0, 0.0]),
        build_doc("doc-2", 2, "alpha", 0.2, [0.56510925, -0.8752184, 0.0, 0.0]),
        build_doc(
            "doc-3",
            3,
            "beta",
            0.3,
            [-0.14134157, -0.32616633, 0.0, 0.0],
        ),
        build_doc("doc-4", 4, "beta", 0.4, [-0.9092026, -0.81424975, 0.0, 0.0]),
        build_doc(
            "doc-5",
            5,
            "gamma",
            0.5,
            [0.93141365, -0.11878121, 0.0, 0.0],
        ),
    ];
    let results = collection.insert(docs);
    assert!(results.iter().all(|result| result.status.is_ok()));
}

fn single_segment_options() -> garuda_types::CollectionOptions {
    let mut options = default_options();
    options.segment_max_docs = 16;
    options
}
