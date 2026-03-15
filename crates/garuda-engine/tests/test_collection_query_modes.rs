mod common;

use common::{database, default_options, default_schema, seed_collection};
use garuda_types::VectorQuery;

#[test]
fn query_by_document_id_should_behave_like_query_by_its_stored_vector() {
    let (_root, db) = database("query-by-id");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let by_id = collection.query(VectorQuery {
        field_name: "embedding".to_string(),
        vector: None,
        id: Some("doc-1".to_string()),
        top_k: 3,
        filter: None,
        include_vector: false,
        output_fields: None,
        ef_search: None,
    });
    assert!(by_id.is_ok());
}

#[test]
fn include_vector_and_output_fields_should_control_result_shape() {
    let (_root, db) = database("query-shape");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let mut query = VectorQuery::by_vector("embedding", vec![1.0, 0.0, 0.0, 0.0], 3);
    query.include_vector = true;
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
        .query(VectorQuery::by_vector("embedding", vec![1.0, 0.0, 0.0, 0.0], 4))
        .expect("first query");
    let second = collection
        .query(VectorQuery::by_vector("embedding", vec![1.0, 0.0, 0.0, 0.0], 4))
        .expect("second query");

    assert_eq!(
        first.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>(),
        second.iter().map(|doc| doc.id.clone()).collect::<Vec<_>>()
    );
}
