use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use garuda_types::{Status, StatusCode};

pub(crate) fn create_dir_all(path: &Path, message: &str) -> Result<(), Status> {
    fs::create_dir_all(path)
        .map_err(|error| Status::err(StatusCode::Internal, format!("{message}: {error}")))
}

pub(crate) fn create_empty_file(path: &Path, message: &str) -> Result<(), Status> {
    File::create(path)
        .map(|_| ())
        .map_err(|error| Status::err(StatusCode::Internal, format!("{message}: {error}")))
}

pub(crate) fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Status> {
    let mut file = File::open(path).map_err(|error| {
        Status::err(
            StatusCode::NotFound,
            format!("failed to open file: {error}"),
        )
    })?;

    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to read file: {error}"),
        )
    })?;

    serde_json::from_slice(&bytes).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to parse file: {error}"),
        )
    })
}

pub(crate) fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), Status> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to serialize file: {error}"),
        )
    })?;

    write_bytes_atomically(path, &bytes)
}

fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<(), Status> {
    let parent = parent_dir(path)?;
    create_dir_all(parent, "failed to create parent directory")?;

    let temp_path = temp_path_for(path);
    write_temp_file(&temp_path, bytes)?;
    rename_file(&temp_path, path)?;

    Ok(())
}

fn parent_dir(path: &Path) -> Result<&Path, Status> {
    path.parent().ok_or_else(|| {
        Status::err(
            StatusCode::Internal,
            "cannot determine parent directory for write",
        )
    })
}

fn temp_path_for(path: &Path) -> PathBuf {
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
            format!("failed to create temp file: {error}"),
        )
    })?;

    file.write_all(bytes).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to write temp file: {error}"),
        )
    })?;

    file.sync_all().map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to sync temp file: {error}"),
        )
    })?;

    Ok(())
}

fn rename_file(from: &Path, to: &Path) -> Result<(), Status> {
    fs::rename(from, to).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to replace file atomically: {error}"),
        )
    })
}
