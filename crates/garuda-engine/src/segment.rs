use crate::storage::SEGMENTS_DIR_NAME;
use crate::storage_io::{read_json_file, write_json_file};
use garuda_types::{Doc, SegmentMeta, Status};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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

pub fn segment_file_name(id: u64) -> String {
    format!("{SEGMENTS_DIR_NAME}/segment-{id:020}.json")
}

pub fn sync_segment_meta(segment: &mut SegmentFile) {
    let mut min_doc_id = None;
    let mut max_doc_id = None;
    let mut live_doc_count = 0usize;

    for record in &segment.records {
        let record_id = record.doc_id;

        min_doc_id = Some(min_doc_id.unwrap_or(record_id).min(record_id));
        max_doc_id = Some(max_doc_id.unwrap_or(record_id).max(record_id));

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
