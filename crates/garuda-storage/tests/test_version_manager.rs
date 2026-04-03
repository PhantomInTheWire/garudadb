use garuda_storage::VersionManager;

#[test]
fn exists_should_error_when_collection_path_is_not_a_directory() {
    let root = std::env::temp_dir().join(format!(
        "garudadb-version-manager-file-{}",
        std::process::id()
    ));
    std::fs::write(&root, b"not a directory").expect("create sentinel file");

    let manager = VersionManager::new(&root);
    let status = manager.exists().expect_err("file path should fail");

    assert_eq!(status.code, garuda_types::StatusCode::Internal);

    std::fs::remove_file(root).expect("remove sentinel file");
}
