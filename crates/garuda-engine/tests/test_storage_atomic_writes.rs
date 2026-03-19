mod common;
#[path = "support/storage_helpers.rs"]
mod storage_helpers;

use common::{collection_name, database, default_schema};
use storage_helpers::{manifest_version_paths, options_with_segment_max_docs};

#[test]
fn atomic_write_keeps_only_final_file() {
    let (root, db) = database("storage-atomic-write");
    let schema = default_schema("docs");

    let collection = db
        .create_collection(schema.clone(), options_with_segment_max_docs(8))
        .expect("create collection");

    collection.flush().expect("flush");
    drop(collection);

    let version_paths = manifest_version_paths(&root, "docs");
    let temp_path = root.join("docs").join("manifest.1.tmp");
    assert_eq!(version_paths.len(), 1);
    assert_eq!(
        version_paths[0].file_name().and_then(|name| name.to_str()),
        Some("manifest.1")
    );
    assert!(!temp_path.exists());
    assert!(!root.join("docs").join("VERSION.json").exists());

    let reopened = db
        .open_collection(&collection_name("docs"))
        .expect("reopen collection");
    assert_eq!(reopened.schema(), schema);
}
