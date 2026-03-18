mod common;

use common::{collection_name, database, default_options, default_schema, seed_collection};
use std::fs;

#[test]
fn reopen_uses_current_version_and_ignores_stale_temp_version_file() {
    let (root, db) = database("storage-version-temp");
    let schema = default_schema("docs");

    let collection = db
        .create_collection(schema.clone(), default_options())
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");

    let temp_version_path = root.join("docs").join("VERSION.json.tmp");
    fs::write(&temp_version_path, b"{\"corrupt\":true}").expect("write stale temp file");

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    assert_eq!(reopened.schema(), schema);
    assert_eq!(reopened.stats().doc_count, 4);
}

#[test]
fn corrupt_current_version_file_blocks_reopen() {
    let (root, db) = database("storage-version-corrupt");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");
    collection.flush().expect("flush");

    let version_path = root.join("docs").join("VERSION.json");
    fs::write(&version_path, b"{not-json").expect("corrupt version");

    let result = db.open_collection(&collection_name("docs"));
    assert!(result.is_err());
}

#[test]
fn flush_keeps_version_file_present_without_temp_artifacts() {
    let (root, db) = database("storage-version-current");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    seed_collection(&collection);
    collection.flush().expect("flush");

    let version_path = root.join("docs").join("VERSION.json");
    let temp_path = root.join("docs").join("VERSION.json.tmp");

    assert!(version_path.exists());
    assert!(!temp_path.exists());
}
