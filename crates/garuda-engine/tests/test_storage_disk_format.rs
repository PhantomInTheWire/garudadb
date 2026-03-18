mod common;

use common::{
    build_doc, collection_dir, database, default_options, default_schema, storage_snapshot_paths,
};

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
    let idmap_paths = storage_snapshot_paths(&root, "docs", "idmap.");
    let delete_paths = storage_snapshot_paths(&root, "docs", "del.");

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
