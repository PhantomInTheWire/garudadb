mod common;

use common::{database, default_options, default_schema, doc_id, field_name, seed_collection};
use garuda_types::{IndexKind, ScalarFieldSchema, ScalarType, ScalarValue};

#[test]
fn create_drop_index_and_column_ddl_roundtrip() {
    let (_root, db) = database("ddl");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    seed_collection(&collection);

    collection
        .create_index(&field_name("embedding"), IndexKind::Hnsw)
        .expect("create hnsw index");
    assert_eq!(collection.schema().vector.index.kind(), IndexKind::Hnsw);

    collection
        .add_column(ScalarFieldSchema {
            name: field_name("flag"),
            field_type: ScalarType::Bool,
            nullable: true,
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
        .drop_index(&field_name("embedding"))
        .expect("drop index");
    assert_eq!(collection.schema().vector.index.kind(), IndexKind::Flat);
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
        nullable: false,
    });

    assert!(result.is_err());

    let fetched = collection.fetch(vec![doc_id("doc-1")]);
    assert!(!fetched["doc-1"].fields.contains_key("required_flag"));
}
