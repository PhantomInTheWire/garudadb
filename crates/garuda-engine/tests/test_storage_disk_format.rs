mod common;

use common::{build_doc, database, default_options, default_schema};
use garuda_types::CollectionName;

#[test]
fn collection_uses_binary_engine_files_instead_of_json_artifacts() {
    let (root, db) = database("storage-disk-format");
    let collection = db
        .create_collection(default_schema("docs"), default_options())
        .expect("create collection");

    let results = collection.insert(vec![
        build_doc("doc-1", 1, "alpha", 0.9, [1.0, 0.0, 0.0, 0.0]),
        build_doc("doc-2", 2, "alpha", 0.8, [0.9, 0.1, 0.0, 0.0]),
    ]);
    assert!(results.iter().all(|result| result.status.is_ok()));
    collection.flush().expect("flush");
    drop(collection);

    let collection_dir = collection_dir(&root, "docs");
    let idmap_paths = storage_snapshot_paths(&collection_dir, "idmap.");
    let delete_paths = storage_snapshot_paths(&collection_dir, "del.");

    assert_eq!(idmap_paths.len(), 1);
    assert_eq!(delete_paths.len(), 1);
    assert!(collection_dir.join("0").exists());
    assert!(collection_dir.join("0").join("data.seg").exists());
    assert!(collection_dir.join("0").join("data.wal").exists());
    assert!(collection_dir.join("1").join("data.seg").exists());

    assert_eq!(
        idmap_paths[0].file_name().and_then(|name| name.to_str()),
        Some("idmap.1")
    );
    assert_eq!(
        delete_paths[0].file_name().and_then(|name| name.to_str()),
        Some("del.1")
    );

    assert!(!collection_dir.join("IDMAP.json").exists());
    assert!(!collection_dir.join("DELETE_STORE.json").exists());
    assert!(!collection_dir.join("VERSION.json").exists());
    assert!(!collection_dir.join("segments").exists());
}

fn collection_dir(root: &std::path::Path, name: &str) -> std::path::PathBuf {
    root.join(CollectionName::parse(name).expect("valid collection name").as_str())
}

fn storage_snapshot_paths(
    collection_dir: &std::path::Path,
    prefix: &str,
) -> Vec<std::path::PathBuf> {
    let mut paths = std::fs::read_dir(collection_dir)
        .expect("read collection dir")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                return false;
            };

            file_name.starts_with(prefix)
        })
        .collect::<Vec<_>>();

    paths.sort();
    paths
}
