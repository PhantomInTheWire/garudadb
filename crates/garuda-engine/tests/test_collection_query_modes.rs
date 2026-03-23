mod common;

use common::{
    database, default_options, default_schema, dense_vector, doc_id, field_name, seed_collection,
};
use garuda_types::{
    HnswIndexParams, IndexParams, IvfIndexParams, IvfProbeCount, VectorQuery, VectorSearch,
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
fn query_by_document_id_should_match_query_by_vector_results_under_ivf() {
    let (_root, db) = database("query-by-id-ivf-parity");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Ivf(IvfIndexParams::default()),
        )
        .expect("create ivf");

    let mut by_vector_query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(4),
    );
    by_vector_query.search = VectorSearch::Default;
    let by_vector = collection.query(by_vector_query).expect("query by vector");

    let mut by_id_query =
        VectorQuery::by_id(field_name("embedding"), doc_id("doc-1"), common::top_k(4));
    by_id_query.search = VectorSearch::Default;
    let by_id = collection.query(by_id_query).expect("query by id");

    assert_eq!(
        by_vector
            .iter()
            .map(|doc| doc.id.clone())
            .collect::<Vec<_>>(),
        by_id.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn ivf_query_should_accept_public_nprobe_override() {
    let (_root, db) = database("query-ivf-nprobe");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Ivf(IvfIndexParams::default()),
        )
        .expect("create ivf");

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(4),
    );
    query.search = VectorSearch::Ivf {
        nprobe: IvfProbeCount::new(1).expect("nprobe"),
    };

    let results = collection.query(query).expect("ivf query");
    assert!(!results.is_empty());
}

#[test]
fn mismatched_query_override_should_fail() {
    let (_root, db) = database("query-mismatched-override");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(4),
    );
    query.search = VectorSearch::Ivf {
        nprobe: IvfProbeCount::new(1).expect("nprobe"),
    };

    let error = collection
        .query(query)
        .expect_err("mismatched override should fail");
    assert_eq!(error.code, garuda_types::StatusCode::InvalidArgument);
}
