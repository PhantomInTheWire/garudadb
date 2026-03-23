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
            common::top_k(2),
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
        common::top_k(10),
    );
    query.filter = Some("category = 'alpha' AND rank >= 2".to_string());
    query.output_fields = Some(vec!["rank".to_string(), "category".to_string()]);
    let results = collection.query(query).expect("query");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, doc_id("doc-2"));
    assert!(results[0].fields.contains_key("rank"));
    assert!(!results[0].fields.contains_key("score"));
}

#[test]
fn filtered_query_supports_like_contains_and_is_null() {
    let (_root, db) = database("dql-string-and-null-filter");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    collection
        .add_column(garuda_types::ScalarFieldSchema {
            name: field_name("nickname"),
            field_type: garuda_types::ScalarType::String,
            index: garuda_types::ScalarIndexState::None,
            nullability: garuda_types::Nullability::Nullable,
            default_value: Some(garuda_types::ScalarValue::Null),
        })
        .expect("add nullable column");

    let mut update = garuda_types::Doc {
        id: doc_id("doc-1"),
        fields: std::collections::BTreeMap::new(),
        vector: garuda_types::DenseVector::default(),
        score: None,
    };
    update.fields.insert(
        "nickname".to_string(),
        garuda_types::ScalarValue::String("alphacat".to_string()),
    );
    let statuses = collection.update(vec![update]);
    assert!(statuses[0].status.is_ok());

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(10),
    );
    query.filter = Some("category LIKE 'alp%' AND category CONTAINS 'pha'".to_string());
    let like_results = collection.query(query).expect("like contains query");
    assert_eq!(like_results.len(), 2);
    assert_eq!(like_results[0].id, doc_id("doc-1"));
    assert_eq!(like_results[1].id, doc_id("doc-2"));

    let mut exact_like_query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(10),
    );
    exact_like_query.filter = Some("category LIKE 'alpha'".to_string());
    let exact_like_results = collection
        .query(exact_like_query)
        .expect("exact like query");
    let exact_like_ids: Vec<_> = exact_like_results.into_iter().map(|doc| doc.id).collect();
    assert_eq!(exact_like_ids, vec![doc_id("doc-1"), doc_id("doc-2")]);

    let mut null_query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(10),
    );
    null_query.filter = Some("nickname IS NULL".to_string());
    let null_results = collection.query(null_query).expect("is null query");
    let ids: Vec<_> = null_results.into_iter().map(|doc| doc.id).collect();
    assert_eq!(ids, vec![doc_id("doc-2"), doc_id("doc-3"), doc_id("doc-4")]);
}
