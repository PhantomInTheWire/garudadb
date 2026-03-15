mod common;

use common::{
    database, default_options, default_schema, read_only_options, schema_missing_primary_field,
    schema_with_dimension, schema_with_duplicate_field, schema_with_vector_name,
};

#[test]
fn rejects_invalid_schema_shapes_and_unknown_collections() {
    let (_root, db) = database("create-open-validation");

    assert!(db.open_collection("does-not-exist").is_err());
    assert!(db
        .create_collection(schema_with_dimension("zero-dim", 0), default_options())
        .is_err());
    assert!(db
        .create_collection(schema_with_duplicate_field("dup-field"), default_options())
        .is_err());
    assert!(db
        .create_collection(schema_missing_primary_field("missing-pk"), default_options())
        .is_err());
    assert!(db
        .create_collection(schema_with_vector_name("bad-vector", "pk"), default_options())
        .is_err());
}

#[test]
fn rejects_invalid_collection_names_and_duplicate_collection_creation() {
    let (_root, db) = database("create-open-names");

    assert!(db
        .create_collection(default_schema(""), default_options())
        .is_err());
    assert!(db
        .create_collection(default_schema("name with spaces"), default_options())
        .is_err());

    let first = db.create_collection(default_schema("docs"), default_options());
    assert!(first.is_ok());
    let second = db.create_collection(default_schema("docs"), default_options());
    assert!(second.is_err());
}

#[test]
fn read_only_collection_open_should_preserve_non_mutating_contract() {
    let (_root, db) = database("create-open-read-only");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create base collection");
    collection.flush().expect("flush");

    let reopened = db.open_collection("docs");
    assert!(reopened.is_ok());

    let readonly_create = db.create_collection(default_schema("readonly"), read_only_options());
    assert!(readonly_create.is_err());
}
