mod common;

use common::{database, default_options, default_schema, doc_id, field_name, seed_collection};
use garuda_types::{
    DenseVector, Doc, HnswIndexParams, IndexKind, IndexParams, IvfIndexParams,
    ScalarFieldSchema, ScalarIndexState, ScalarType, ScalarValue, VectorIndexState,
};
use std::collections::BTreeMap;

#[test]
fn create_drop_index_and_column_ddl_roundtrip() {
    let (_root, db) = database("ddl");
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
    assert_eq!(
        collection.schema().vector.indexes.default_kind(),
        IndexKind::Hnsw
    );

    collection
        .add_column(ScalarFieldSchema {
            name: field_name("flag"),
            field_type: ScalarType::Bool,
            index: ScalarIndexState::None,
            nullability: garuda_types::Nullability::Nullable,
            default_value: None,
        })
        .expect("add column");
    let fetched = collection.fetch(vec![doc_id("doc-1")]);
    assert_eq!(fetched["doc-1"].fields["flag"], ScalarValue::Null);

    collection
        .alter_column(&field_name("flag"), &field_name("is_flagged"))
        .expect("rename column");
    let fetched = collection.fetch(vec![doc_id("doc-1")]);
    assert!(fetched["doc-1"].fields.contains_key("is_flagged"));

    collection
        .drop_column(&field_name("is_flagged"))
        .expect("drop column");
    collection
        .drop_index(&field_name("embedding"), IndexKind::Hnsw)
        .expect("drop index");
    assert_eq!(
        collection.schema().vector.indexes,
        VectorIndexState::DefaultFlat
    );
}

#[test]
fn create_and_drop_scalar_index_uses_scalar_field_state() {
    let (_root, db) = database("ddl-scalar-index");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    collection
        .create_index(
            &field_name("category"),
            IndexParams::Scalar(garuda_types::ScalarIndexParams),
        )
        .expect("create scalar index");
    assert!(
        collection
            .schema()
            .fields
            .iter()
            .find(|field| field.name == field_name("category"))
            .expect("category field")
            .index
            .is_indexed()
    );

    collection
        .drop_index(&field_name("category"), IndexKind::Scalar)
        .expect("drop scalar index");
    assert!(
        !collection
            .schema()
            .fields
            .iter()
            .find(|field| field.name == field_name("category"))
            .expect("category field")
            .index
            .is_indexed()
    );
}

#[test]
fn create_index_rejects_wrong_field_kind() {
    let (_root, db) = database("ddl-wrong-index-kind");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let vector_on_scalar = collection.create_index(
        &field_name("category"),
        IndexParams::Hnsw(HnswIndexParams::default()),
    );
    assert!(vector_on_scalar.is_err());
    assert_eq!(
        collection.schema().vector.indexes,
        VectorIndexState::DefaultFlat
    );

    let scalar_on_vector = collection.create_index(
        &field_name("embedding"),
        IndexParams::Scalar(garuda_types::ScalarIndexParams),
    );
    assert!(scalar_on_vector.is_err());
    assert_eq!(
        collection.schema().vector.indexes,
        VectorIndexState::DefaultFlat
    );
}

#[test]
fn invalid_drop_index_leaves_vector_state_unchanged() {
    let (_root, db) = database("ddl-invalid-drop-index");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    collection
        .create_index(
            &field_name("embedding"),
            IndexParams::Hnsw(HnswIndexParams::default()),
        )
        .expect("create hnsw index");

    let result = collection.drop_index(&field_name("embedding"), IndexKind::Scalar);
    assert!(result.is_err());
    assert_eq!(
        collection.schema().vector.indexes.default_kind(),
        IndexKind::Hnsw
    );
}

#[test]
fn drop_index_should_fail_when_kind_is_not_enabled() {
    let (_root, db) = database("ddl-drop-invalid-index");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let error = collection
        .drop_index(&field_name("embedding"), IndexKind::Hnsw)
        .expect_err("dropping missing hnsw should fail");
    assert_eq!(error.code, garuda_types::StatusCode::InvalidArgument);

    let error = collection
        .drop_index(&field_name("embedding"), IndexKind::Flat)
        .expect_err("dropping missing flat should fail");
    assert_eq!(error.code, garuda_types::StatusCode::InvalidArgument);
}

#[test]
fn create_and_drop_ivf_index_roundtrips_vector_state() {
    let (_root, db) = database("ddl-ivf-index");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    collection
        .create_index(&field_name("embedding"), IndexParams::Ivf(IvfIndexParams::default()))
        .expect("create ivf index");
    assert_eq!(collection.schema().vector.indexes.default_kind(), IndexKind::Ivf);

    collection
        .drop_index(&field_name("embedding"), IndexKind::Ivf)
        .expect("drop ivf index");
    assert_eq!(collection.schema().vector.indexes, VectorIndexState::DefaultFlat);
}

#[test]
fn rename_column_rejects_existing_scalar_and_vector_names() {
    let (_root, db) = database("ddl-rename-conflict");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let rename_to_scalar = collection.alter_column(&field_name("rank"), &field_name("category"));
    assert!(rename_to_scalar.is_err());

    let rename_to_vector = collection.alter_column(&field_name("rank"), &field_name("embedding"));
    assert!(rename_to_vector.is_err());

    let fetched = collection.fetch(vec![doc_id("doc-1")]);
    assert!(fetched["doc-1"].fields.contains_key("rank"));
    assert!(fetched["doc-1"].fields.contains_key("category"));
}

#[test]
fn add_non_nullable_column_requires_a_backfill_strategy() {
    let (_root, db) = database("ddl-add-required-column");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let result = collection.add_column(ScalarFieldSchema {
        name: field_name("required_flag"),
        field_type: ScalarType::Bool,
        index: ScalarIndexState::None,
        nullability: garuda_types::Nullability::Required,
        default_value: None,
    });

    assert!(result.is_err());

    let fetched = collection.fetch(vec![doc_id("doc-1")]);
    assert!(!fetched["doc-1"].fields.contains_key("required_flag"));
}

#[test]
fn add_column_with_default_backfills_existing_rows() {
    let (_root, db) = database("ddl-add-default-column");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    collection
        .add_column(ScalarFieldSchema {
            name: field_name("is_public"),
            field_type: ScalarType::Bool,
            index: ScalarIndexState::None,
            nullability: garuda_types::Nullability::Required,
            default_value: Some(ScalarValue::Bool(true)),
        })
        .expect("add column with default");

    let fetched = collection.fetch(vec![doc_id("doc-1"), doc_id("doc-2")]);
    assert_eq!(
        fetched["doc-1"].fields["is_public"],
        ScalarValue::Bool(true)
    );
    assert_eq!(
        fetched["doc-2"].fields["is_public"],
        ScalarValue::Bool(true)
    );
}

#[test]
fn add_column_rejects_default_with_wrong_type() {
    let (_root, db) = database("ddl-add-invalid-default");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    let result = collection.add_column(ScalarFieldSchema {
        name: field_name("is_public"),
        field_type: ScalarType::Bool,
        index: ScalarIndexState::None,
        nullability: garuda_types::Nullability::Required,
        default_value: Some(ScalarValue::String(String::from("yes"))),
    });

    assert!(result.is_err());

    let fetched = collection.fetch(vec![doc_id("doc-1")]);
    assert!(!fetched["doc-1"].fields.contains_key("is_public"));
}

#[test]
fn inserts_apply_schema_defaults_after_add_column() {
    let (_root, db) = database("ddl-insert-applies-default");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    collection
        .add_column(ScalarFieldSchema {
            name: field_name("is_public"),
            field_type: ScalarType::Bool,
            index: ScalarIndexState::None,
            nullability: garuda_types::Nullability::Required,
            default_value: Some(ScalarValue::Bool(true)),
        })
        .expect("add column with default");

    let mut fields = BTreeMap::new();
    fields.insert(
        String::from("pk"),
        ScalarValue::String(String::from("doc-3")),
    );
    fields.insert(String::from("rank"), ScalarValue::Int64(3));
    fields.insert(
        String::from("category"),
        ScalarValue::String(String::from("gamma")),
    );
    fields.insert(String::from("score"), ScalarValue::Float64(0.7));

    let result = collection.insert(vec![Doc::new(
        doc_id("doc-3"),
        fields,
        DenseVector::parse(vec![0.0, 0.0, 1.0, 0.0]).expect("valid vector"),
    )]);

    assert!(result[0].status.is_ok());

    let fetched = collection.fetch(vec![doc_id("doc-3")]);
    assert_eq!(
        fetched["doc-3"].fields["is_public"],
        ScalarValue::Bool(true)
    );
}

#[test]
fn renamed_column_survives_reopen_and_query_projection_uses_new_name() {
    let (_root, db) = database("ddl-rename-reopen");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    collection
        .alter_column(&field_name("category"), &field_name("label"))
        .expect("rename column");
    collection.flush().expect("flush");
    drop(collection);

    let reopened = db
        .open_collection(&common::collection_name("docs"))
        .expect("reopen collection");

    let fetched = reopened.fetch(vec![doc_id("doc-1")]);
    assert_eq!(
        fetched["doc-1"].fields["label"],
        ScalarValue::String("alpha".to_string())
    );
    assert!(!fetched["doc-1"].fields.contains_key("category"));

    let mut query = garuda_types::VectorQuery::by_vector(
        field_name("embedding"),
        common::dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(2),
    );
    query.output_fields = Some(vec!["label".to_string()]);
    let results = reopened.query(query).expect("query");
    assert!(results.iter().all(|doc| doc.fields.contains_key("label")));
}

#[test]
fn renamed_indexed_scalar_column_remains_queryable() {
    let (_root, db) = database("ddl-rename-indexed-scalar");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    collection
        .create_index(
            &field_name("category"),
            IndexParams::Scalar(garuda_types::ScalarIndexParams),
        )
        .expect("create scalar index");
    collection
        .alter_column(&field_name("category"), &field_name("label"))
        .expect("rename indexed column");

    let mut query = garuda_types::VectorQuery::by_vector(
        field_name("embedding"),
        common::dense_vector(vec![1.0, 0.0, 0.0, 0.0]),
        common::top_k(10),
    );
    query.filter = Some("label = 'alpha'".to_string());

    let results = collection
        .query(query)
        .expect("query renamed indexed field");
    let ids: Vec<_> = results.into_iter().map(|doc| doc.id).collect();
    assert_eq!(ids, vec![doc_id("doc-1"), doc_id("doc-2")]);
}
