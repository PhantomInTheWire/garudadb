use garuda_types::{CollectionName, Status, StatusCode};
use std::path::{Path, PathBuf};

use crate::storage_io::{create_dir_all, create_empty_file};
use crate::version::VersionStore;

pub const LOCK_FILE_NAME: &str = "LOCK";
pub const SEGMENTS_DIR_NAME: &str = "segments";

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

    create_dir_all(
        &path.join(SEGMENTS_DIR_NAME),
        "failed to create collection directory",
    )?;
    create_empty_file(&path.join(LOCK_FILE_NAME), "failed to create lock file")?;

    Ok(())
}

pub fn ensure_existing_collection_dir(path: &Path) -> Result<(), Status> {
    if path.exists() && VersionStore::new(path).exists() {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::NotFound,
        "collection directory does not exist",
    ))
}
