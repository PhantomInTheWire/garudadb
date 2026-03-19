use crate::common::default_options;
use garuda_types::CollectionOptions;
use std::path::{Path, PathBuf};

pub fn collection_dir(root: &Path, name: &str) -> PathBuf {
    root.join(name)
}

pub fn manifest_version_paths(root: &Path, name: &str) -> Vec<PathBuf> {
    let collection_dir = collection_dir(root, name);
    let mut paths = std::fs::read_dir(collection_dir)
        .expect("read collection dir")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                return false;
            };

            file_name.starts_with("manifest.")
        })
        .collect::<Vec<_>>();

    paths.sort();
    paths
}

pub fn storage_snapshot_paths(root: &Path, name: &str, prefix: &str) -> Vec<PathBuf> {
    let collection_dir = collection_dir(root, name);
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

pub fn options_with_segment_max_docs(segment_max_docs: usize) -> CollectionOptions {
    CollectionOptions {
        segment_max_docs,
        ..default_options()
    }
}
