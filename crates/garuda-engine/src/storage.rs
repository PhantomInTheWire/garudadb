use garuda_types::{CollectionName, Doc, SegmentMeta, Status, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::storage_io::{create_dir_all, create_empty_file, read_json_file, write_json_file};
use crate::version::VersionStore;

pub const LOCK_FILE_NAME: &str = "LOCK";
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
    if path.exists() && VersionStore::new(path).exists() {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::NotFound,
        "collection directory does not exist",
    ))
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

fn segment_path(path: &Path, segment_id: u64) -> PathBuf {
    path.join(SEGMENTS_DIR_NAME)
        .join(format!("segment-{segment_id:020}.json"))
}
