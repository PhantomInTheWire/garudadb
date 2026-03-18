use crate::io::{create_dir_all, create_empty_file, remove_file, sync_directory};
use garuda_types::{CollectionName, ManifestVersionId, SnapshotId, Status, StatusCode};
use std::path::{Path, PathBuf};

pub const LOCK_FILE_NAME: &str = "LOCK";
pub const MANIFEST_FILE_PREFIX: &str = "manifest.";
pub const ID_MAP_FILE_PREFIX: &str = "idmap.";
pub const DELETE_FILE_PREFIX: &str = "del.";
pub const DATA_SEG_FILE_NAME: &str = "data.seg";
pub const DATA_WAL_FILE_NAME: &str = "data.wal";
pub const WRITING_SEGMENT_ID: u64 = 0;

pub fn ensure_database_root(path: &Path) -> Result<(), Status> {
    create_dir_all(path, "failed to create database root")
}

pub fn collection_dir(root: &Path, name: &CollectionName) -> PathBuf {
    root.join(name.as_str())
}

pub fn ensure_new_collection_dir(path: &Path) -> Result<(), Status> {
    if path.exists() {
        return Err(Status::err(
            StatusCode::AlreadyExists,
            "collection directory already exists",
        ));
    }

    create_dir_all(path, "failed to create collection directory")?;
    create_empty_file(&path.join(LOCK_FILE_NAME), "failed to create lock file")?;
    sync_directory(path)?;

    Ok(())
}

pub fn ensure_existing_collection_dir(path: &Path) -> Result<(), Status> {
    if path.exists() && !manifest_paths(path)?.is_empty() {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::NotFound,
        "collection directory does not exist",
    ))
}

pub fn manifest_path(root: &Path, version_id: ManifestVersionId) -> PathBuf {
    root.join(format!("{MANIFEST_FILE_PREFIX}{}", version_id.get()))
}

pub fn id_map_snapshot_path(root: &Path, snapshot_id: SnapshotId) -> PathBuf {
    root.join(format!("{ID_MAP_FILE_PREFIX}{}", snapshot_id.get()))
}

pub fn delete_snapshot_path(root: &Path, snapshot_id: SnapshotId) -> PathBuf {
    root.join(format!("{DELETE_FILE_PREFIX}{}", snapshot_id.get()))
}

pub fn segment_dir(root: &Path, segment_id: u64) -> PathBuf {
    root.join(segment_id.to_string())
}

pub fn segment_data_path(root: &Path, segment_id: u64) -> PathBuf {
    segment_dir(root, segment_id).join(DATA_SEG_FILE_NAME)
}

pub fn segment_wal_path(root: &Path, segment_id: u64) -> PathBuf {
    segment_dir(root, segment_id).join(DATA_WAL_FILE_NAME)
}

pub fn manifest_paths(root: &Path) -> Result<Vec<(ManifestVersionId, PathBuf)>, Status> {
    let mut paths = Vec::new();
    let entries = std::fs::read_dir(root).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!(
                "failed to read collection directory {}: {error}",
                root.display()
            ),
        )
    })?;

    for entry in entries {
        let entry = entry.map_err(|error| {
            Status::err(
                StatusCode::Internal,
                format!("failed to read directory entry: {error}"),
            )
        })?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };

        if !file_name.starts_with(MANIFEST_FILE_PREFIX) {
            continue;
        }

        let suffix = &file_name[MANIFEST_FILE_PREFIX.len()..];
        let Ok(version_id) = suffix.parse::<u64>() else {
            continue;
        };

        paths.push((ManifestVersionId::new(version_id), entry.path()));
    }

    paths.sort_by_key(|(version_id, _)| version_id.get());
    Ok(paths)
}

pub fn remove_path_if_exists(path: &Path) -> Result<(), Status> {
    if !path.exists() {
        return Ok(());
    }

    if path.is_dir() {
        std::fs::remove_dir_all(path).map_err(|error| {
            Status::err(
                StatusCode::Internal,
                format!("failed to remove directory {}: {error}", path.display()),
            )
        })?;
    } else {
        remove_file(path)?;
    }

    Ok(())
}
