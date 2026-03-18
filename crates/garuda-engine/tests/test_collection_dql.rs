mod common;

use common::{
    database, default_options, default_schema, dense_vector, doc_id, field_name, seed_collection,
};
use garuda_types::VectorQuery;

#[test]
fn vector_query_returns_ranked_hits() {
    let (_root, db) = database("dql");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let results = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
            2,
        ))
        .expect("query");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, doc_id("doc-1"));
    assert!(results[0].score.unwrap() >= results[1].score.unwrap());
    assert!(results[0].vector.is_empty());
}

#[test]
fn filtered_query_supports_equality_range_and_boolean_operators() {
    let (_root, db) = database("dql-filter");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        10,
    );
    query.filter = Some("category = 'alpha' AND rank >= 2".to_string());
    query.output_fields = Some(vec!["rank".to_string(), "category".to_string()]);
    let results = collection.query(query).expect("query");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, doc_id("doc-2"));
    assert!(results[0].fields.contains_key("rank"));
    assert!(!results[0].fields.contains_key("score"));
}
