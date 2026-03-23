mod common;

use common::{
    database, default_options, default_schema, dense_vector, field_name, seed_collection,
    seed_more_collection_docs,
};
use garuda_types::DocId;
use garuda_types::VectorQuery;

#[test]
fn filters_should_respect_boolean_grouping_and_not_leak_non_matching_docs() {
    let (_root, db) = database("filter-grouping");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let mut query = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
        common::top_k(10),
    );
    query.filter = Some("(category = 'beta' OR category = 'gamma') AND rank >= 4".to_string());
    let results = collection.query(query).expect("query");

    assert!(!results.is_empty());
    assert!(results.iter().all(|doc| {
        let cat = &doc.fields["category"];
        let rank = &doc.fields["rank"];
        matches!(cat, garuda_types::ScalarValue::String(v) if v == "beta" || v == "gamma")
            && matches!(rank, garuda_types::ScalarValue::Int64(v) if *v >= 4)
    }));
}

#[test]
fn filters_on_unknown_fields_or_invalid_types_should_error() {
    let (_root, db) = database("filter-invalid");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let mut unknown_field = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(5),
    );
    unknown_field.filter = Some("missing_field = 'x'".to_string());
    assert!(collection.query(unknown_field).is_err());

    let mut wrong_type = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(5),
    );
    wrong_type.filter = Some("rank = 'not-a-number'".to_string());
    assert!(collection.query(wrong_type).is_err());

    let mut bad_like = VectorQuery::by_vector(
        field_name("embedding"),
        dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(5),
    );
    bad_like.filter = Some("category LIKE 'a%b%c'".to_string());
    assert!(collection.query(bad_like).is_err());
}

#[test]
fn delete_by_filter_should_remove_only_matching_documents() {
    let (_root, db) = database("filter-delete-by-filter");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);
    seed_more_collection_docs(&collection);

    let deleted = collection.delete_by_filter("category = 'beta' AND rank >= 6");
    assert!(deleted.is_ok(), "delete_by_filter should succeed");

    let remaining = collection
        .query(VectorQuery::by_vector(
            field_name("embedding"),
            dense_vector(vec![0.0, 1.0, 0.0, 0.0]),
            common::top_k(10),
        ))
        .expect("query after delete by filter");

    let remaining_ids: Vec<DocId> = remaining.into_iter().map(|doc| doc.id).collect();
    assert!(!remaining_ids.contains(&common::doc_id("doc-6")));
    assert!(!remaining_ids.contains(&common::doc_id("doc-7")));
    assert!(remaining_ids.contains(&common::doc_id("doc-3")));
    assert!(remaining_ids.contains(&common::doc_id("doc-4")));
}
