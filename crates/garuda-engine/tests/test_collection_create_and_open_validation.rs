mod common;
#[path = "support/schema_helpers.rs"]
mod schema_helpers;

use common::{
    collection_name, database, default_options, default_schema,
};
use schema_helpers::{
    read_only_options, schema_missing_primary_field, schema_with_dimension,
    schema_with_duplicate_field, schema_with_vector_name,
};
use garuda_types::{CollectionName, StatusCode};

#[test]
fn rejects_invalid_schema_shapes_and_unknown_collections() {
    let (_root, db) = database("create-open-validation");

    assert!(
        db.open_collection(&collection_name("does-not-exist"))
            .is_err()
    );
    assert!(
        db.create_collection(schema_with_dimension("zero-dim", 0), default_options())
            .is_err()
    );
    assert!(
        db.create_collection(schema_with_duplicate_field("dup-field"), default_options())
            .is_err()
    );
    assert!(
        db.create_collection(
            schema_missing_primary_field("missing-pk"),
            default_options()
        )
        .is_err()
    );
    assert!(
        db.create_collection(
            schema_with_vector_name("bad-vector", "pk"),
            default_options()
        )
        .is_err()
    );
}

#[test]
fn rejects_invalid_collection_names_and_duplicate_collection_creation() {
    let (_root, db) = database("create-open-names");

    assert!(CollectionName::parse("").is_err());
    assert!(CollectionName::parse("name with spaces").is_err());

    let first = db.create_collection(default_schema("docs"), default_options());
    assert!(first.is_ok());
    let second = db.create_collection(default_schema("docs"), default_options());
    assert!(second.is_err());
}

#[test]
fn rejects_creation_when_collection_directory_already_exists() {
    let (root, db) = database("create-open-existing-dir");
    std::fs::create_dir(root.join("docs")).expect("create existing collection dir");

    let result = db.create_collection(default_schema("docs"), default_options());
    match result {
        Ok(_) => panic!("create_collection should fail for an existing directory"),
        Err(status) => assert_eq!(status.code, StatusCode::AlreadyExists),
    }
}

#[test]
fn read_only_collection_open_should_preserve_non_mutating_contract() {
    let (_root, db) = database("create-open-read-only");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create base collection");
    collection.flush().expect("flush");
    drop(collection);

    let reopened = db.open_collection(&collection_name("docs"));
    assert!(reopened.is_ok());

    let readonly_create = db.create_collection(default_schema("readonly"), read_only_options());
    assert!(readonly_create.is_err());
}
