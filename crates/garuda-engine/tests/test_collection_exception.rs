mod common;

use common::{build_doc, database, default_options, default_schema};
use garuda_types::{IndexKind, ScalarValue, VectorQuery};

#[test]
fn rejects_duplicate_ids_wrong_dimensions_invalid_filters_and_wrong_index_targets() {
    let (_root, db) = database("exception");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let first = collection.insert(vec![build_doc(
        "doc-1",
        1,
        "alpha",
        0.9,
        [1.0, 0.0, 0.0, 0.0],
    )]);
    assert!(first[0].status.is_ok());

    let duplicate = collection.insert(vec![build_doc(
        "doc-1",
        2,
        "beta",
        0.8,
        [0.0, 1.0, 0.0, 0.0],
    )]);
    assert_eq!(
        duplicate[0].status.code,
        garuda_types::StatusCode::AlreadyExists
    );

    let wrong_dimension = collection.insert(vec![garuda_types::Doc::new(
        "doc-bad",
        build_doc("doc-bad", 1, "alpha", 0.1, [1.0, 0.0, 0.0, 0.0]).fields,
        vec![1.0, 2.0],
    )]);
    assert_eq!(
        wrong_dimension[0].status.code,
        garuda_types::StatusCode::InvalidArgument
    );

    let query = collection.query(garuda_types::VectorQuery {
        field_name: "embedding".to_string(),
        vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
        id: None,
        top_k: 10,
        filter: Some("category = ".to_string()),
        include_vector: false,
        output_fields: None,
        ef_search: None,
    });
    assert!(query.is_err());

    let bad_index = collection.create_index("rank", IndexKind::Hnsw);
    assert!(bad_index.is_err());

    let fetched = collection.fetch(vec!["doc-1".to_string()]);
    assert_eq!(
        fetched["doc-1"].fields["category"],
        ScalarValue::String("alpha".to_string())
    );
}
