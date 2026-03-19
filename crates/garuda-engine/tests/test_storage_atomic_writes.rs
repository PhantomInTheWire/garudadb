mod common;

use common::{collection_name, database, default_schema};
use garuda_types::{CollectionName, CollectionOptions};

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

fn options_with_segment_max_docs(segment_max_docs: usize) -> CollectionOptions {
    CollectionOptions {
        segment_max_docs,
        ..common::default_options()
    }
}

fn manifest_version_paths(root: &std::path::Path, name: &str) -> Vec<std::path::PathBuf> {
    let collection_dir = root.join(CollectionName::parse(name).expect("valid collection name").as_str());
    garuda_storage::manifest_paths(&collection_dir)
        .expect("read manifest paths")
        .into_iter()
        .map(|(_, path)| path)
        .collect()
}
