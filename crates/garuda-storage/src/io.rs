use garuda_types::{Status, StatusCode};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn create_dir_all(path: &Path, message: &str) -> Result<(), Status> {
    fs::create_dir_all(path)
        .map_err(|error| Status::err(StatusCode::Internal, format!("{message}: {error}")))
}

pub fn create_empty_file(path: &Path, message: &str) -> Result<(), Status> {
    let parent = path.parent().ok_or_else(|| {
        Status::err(
            StatusCode::Internal,
            "cannot determine parent directory for file creation",
        )
    })?;

    create_dir_all(parent, "failed to create parent directory")?;

    File::create(path)
        .map(|_| ())
        .map_err(|error| Status::err(StatusCode::Internal, format!("{message}: {error}")))
}

pub fn read_file(path: &Path) -> Result<Vec<u8>, Status> {
    let mut file = File::open(path).map_err(|error| {
        Status::err(
            StatusCode::NotFound,
            format!("failed to open file {}: {error}", path.display()),
        )
    })?;

    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to read file {}: {error}", path.display()),
        )
    })?;

    Ok(bytes)
}

pub fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<(), Status> {
    let parent = path.parent().ok_or_else(|| {
        Status::err(
            StatusCode::Internal,
            "cannot determine parent directory for write",
        )
    })?;

    create_dir_all(parent, "failed to create parent directory")?;

    let temp_path = temp_path(path);
    write_temp_file(&temp_path, bytes)?;
    rename_path(&temp_path, path)?;
    sync_directory(parent)?;

    Ok(())
}

pub fn rename_path(from: &Path, to: &Path) -> Result<(), Status> {
    fs::rename(from, to).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!(
                "failed to rename {} to {}: {error}",
                from.display(),
                to.display()
            ),
        )
    })
}

pub fn remove_file(path: &Path) -> Result<(), Status> {
    fs::remove_file(path).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to remove file {}: {error}", path.display()),
        )
    })
}

pub fn sync_directory(path: &Path) -> Result<(), Status> {
    let dir = OpenOptions::new().read(true).open(path).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to open directory {}: {error}", path.display()),
        )
    })?;

    dir.sync_all().map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to sync directory {}: {error}", path.display()),
        )
    })
}

fn temp_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("temp");

    path.with_file_name(format!("{file_name}.tmp"))
}

fn write_temp_file(path: &Path, bytes: &[u8]) -> Result<(), Status> {
    let mut file = File::create(path).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to create temp file {}: {error}", path.display()),
        )
    })?;

    file.write_all(bytes).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to write temp file {}: {error}", path.display()),
        )
    })?;

    file.sync_all().map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to sync temp file {}: {error}", path.display()),
        )
    })?;

    Ok(())
}
