mod common;

use common::{collection_name, database, default_schema, seed_collection};
use std::fs;
use garuda_types::{CollectionName, CollectionOptions};

#[test]
fn reopen_uses_latest_committed_manifest_version() {
    let (root, db) = database("storage-version-latest");
    let schema = default_schema("docs");

    let collection = db
        .create_collection(schema.clone(), options_with_segment_max_docs(8))
        .expect("create collection");
    collection.flush().expect("initial flush");
    seed_collection(&collection);
    collection.flush().expect("second flush");
    drop(collection);

    let version_paths = manifest_version_paths(&root, "docs");
    assert_eq!(version_paths.len(), 1, "only latest manifest should remain");
    assert_eq!(
        version_paths[0].file_name().and_then(|name| name.to_str()),
        Some("manifest.2")
    );

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    assert_eq!(reopened.schema(), schema);
    assert_eq!(reopened.stats().doc_count, 4);
}

#[test]
fn stale_older_manifest_version_does_not_override_latest_state() {
    let (root, db) = database("storage-version-stale");
    let collection = db
        .create_collection(default_schema("docs"), options_with_segment_max_docs(8))
        .expect("create collection");
    seed_collection(&collection);
    collection.flush().expect("flush");
    drop(collection);

    let collection_dir = collection_dir(&root, "docs");
    let stale_version_path = collection_dir.join("manifest.0");
    fs::write(&stale_version_path, b"{\"corrupt\":true}").expect("write stale manifest");

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    assert_eq!(reopened.stats().doc_count, 4);
}

#[test]
fn latest_manifest_corruption_blocks_reopen() {
    let (root, db) = database("storage-version-corrupt");
    let collection = db
        .create_collection(default_schema("docs"), options_with_segment_max_docs(8))
        .expect("create collection");
    collection.flush().expect("flush");
    drop(collection);

    let version_paths = manifest_version_paths(&root, "docs");
    assert_eq!(version_paths.len(), 1);

    fs::write(&version_paths[0], b"not-a-valid-manifest").expect("corrupt latest manifest");

    let result = db.open_collection(&collection_name("docs"));
    assert!(result.is_err());
}

#[test]
fn flush_leaves_no_temp_manifest_artifacts() {
    let (root, db) = database("storage-version-current");
    let collection = db
        .create_collection(default_schema("docs"), options_with_segment_max_docs(8))
        .expect("create collection");

    seed_collection(&collection);
    collection.flush().expect("flush");
    drop(collection);

    let collection_dir = collection_dir(&root, "docs");
    let version_paths = manifest_version_paths(&root, "docs");

    assert_eq!(version_paths.len(), 1);
    assert!(!collection_dir.join("manifest.1.tmp").exists());
    assert!(!collection_dir.join("manifest.2.tmp").exists());
    assert!(!collection_dir.join("VERSION.json").exists());
}

fn options_with_segment_max_docs(segment_max_docs: usize) -> CollectionOptions {
    CollectionOptions {
        segment_max_docs,
        ..common::default_options()
    }
}

fn collection_dir(root: &std::path::Path, name: &str) -> std::path::PathBuf {
    root.join(CollectionName::parse(name).expect("valid collection name").as_str())
}

fn manifest_version_paths(root: &std::path::Path, name: &str) -> Vec<std::path::PathBuf> {
    garuda_storage::manifest_paths(&collection_dir(root, name))
        .expect("read manifest paths")
        .into_iter()
        .map(|(_, path)| path)
        .collect()
}
