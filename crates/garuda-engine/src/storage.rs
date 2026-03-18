use garuda_types::{CollectionName, Doc, DocId, Manifest, SegmentMeta, Status, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const LOCK_FILE_NAME: &str = "LOCK";
pub const VERSION_FILE_NAME: &str = "VERSION.json";
pub const ID_MAP_FILE_NAME: &str = "IDMAP.json";
pub const DELETE_STORE_FILE_NAME: &str = "DELETE_STORE.json";
pub const SEGMENTS_DIR_NAME: &str = "segments";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredRecord {
    pub doc_id: u64,
    pub deleted: bool,
    pub doc: Doc,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentFile {
    pub meta: SegmentMeta,
    pub records: Vec<StoredRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct IdMapFile {
    pub entries: HashMap<DocId, u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct DeleteStoreFile {
    pub deleted_doc_ids: Vec<u64>,
}

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
    if path.exists() && path.join(VERSION_FILE_NAME).exists() {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::NotFound,
        "collection directory does not exist",
    ))
}

pub fn write_manifest(path: &Path, manifest: &Manifest) -> Result<(), Status> {
    write_json_file(&path.join(VERSION_FILE_NAME), manifest)
}

pub fn read_manifest(path: &Path) -> Result<Manifest, Status> {
    read_json_file(&path.join(VERSION_FILE_NAME))
}

pub fn write_id_map(path: &Path, entries: &HashMap<DocId, u64>) -> Result<(), Status> {
    let file = IdMapFile {
        entries: entries.clone(),
    };
    write_json_file(&path.join(ID_MAP_FILE_NAME), &file)
}

pub fn read_id_map(path: &Path) -> Result<HashMap<DocId, u64>, Status> {
    let file: IdMapFile = read_json_file(&path.join(ID_MAP_FILE_NAME))?;
    Ok(file.entries)
}

pub fn write_delete_store(path: &Path, deleted_doc_ids: &HashSet<u64>) -> Result<(), Status> {
    let mut ids = deleted_doc_ids.iter().copied().collect::<Vec<_>>();
    ids.sort_unstable();
    let file = DeleteStoreFile {
        deleted_doc_ids: ids,
    };
    write_json_file(&path.join(DELETE_STORE_FILE_NAME), &file)
}

pub fn read_delete_store(path: &Path) -> Result<HashSet<u64>, Status> {
    let file: DeleteStoreFile = read_json_file(&path.join(DELETE_STORE_FILE_NAME))?;
    Ok(file.deleted_doc_ids.into_iter().collect())
}

pub fn write_segment(path: &Path, segment: &SegmentFile) -> Result<(), Status> {
    let file_path = segment_path(path, segment.meta.id);
    write_json_file(&file_path, segment)
}

pub fn read_segment(path: &Path, meta: &SegmentMeta) -> Result<SegmentFile, Status> {
    read_json_file(&segment_path(path, meta.id))
}

pub fn segment_meta(id: u64) -> SegmentMeta {
    SegmentMeta {
        id,
        path: segment_file_name(id),
        min_doc_id: None,
        max_doc_id: None,
        doc_count: 0,
    }
}

pub(crate) fn segment_file_name(id: u64) -> String {
    format!("{SEGMENTS_DIR_NAME}/segment-{id:020}.json")
}

pub fn sync_segment_meta(segment: &mut SegmentFile) {
    let mut min_doc_id = None;
    let mut max_doc_id = None;
    let mut live_doc_count = 0usize;

    for record in &segment.records {
        min_doc_id =
            Some(min_doc_id.map_or(record.doc_id, |current: u64| current.min(record.doc_id)));
        max_doc_id =
            Some(max_doc_id.map_or(record.doc_id, |current: u64| current.max(record.doc_id)));
        if !record.deleted {
            live_doc_count += 1;
        }
    }

    segment.meta.min_doc_id = min_doc_id;
    segment.meta.max_doc_id = max_doc_id;
    segment.meta.doc_count = live_doc_count;
}

fn create_dir_all(path: &Path, message: &str) -> Result<(), Status> {
    fs::create_dir_all(path)
        .map_err(|error| Status::err(StatusCode::Internal, format!("{message}: {error}")))
}

fn create_empty_file(path: &Path, message: &str) -> Result<(), Status> {
    File::create(path)
        .map(|_| ())
        .map_err(|error| Status::err(StatusCode::Internal, format!("{message}: {error}")))
}

fn segment_path(path: &Path, segment_id: u64) -> PathBuf {
    path.join(SEGMENTS_DIR_NAME)
        .join(format!("segment-{segment_id:020}.json"))
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), Status> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to serialize file: {error}"),
        )
    })?;

    write_bytes_atomically(path, &bytes)
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Status> {
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
