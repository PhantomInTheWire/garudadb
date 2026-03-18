mod codec;
mod wal;

use codec::{decode_segment, encode_segment};
use garuda_storage::{
    create_dir_all, read_file, remove_path_if_exists, segment_data_path, segment_dir,
    segment_wal_path, write_file_atomically,
};
use garuda_types::{Doc, DocId, SegmentMeta, Status};
pub use wal::{WalOp, append_wal_ops, read_wal_ops, reset_wal};

#[derive(Clone, Debug, PartialEq)]
pub struct StoredRecord {
    pub doc_id: u64,
    pub deleted: bool,
    pub doc: Doc,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SegmentFile {
    pub meta: SegmentMeta,
    pub records: Vec<StoredRecord>,
}

pub fn segment_file_name(segment_id: u64) -> String {
    segment_id.to_string()
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

pub fn ensure_segment_files(root: &std::path::Path, segment_id: u64) -> Result<(), Status> {
    let segment_dir = segment_dir(root, segment_id);
    create_dir_all(&segment_dir, "failed to create segment directory")?;

    if !segment_data_path(root, segment_id).exists() {
        let segment = SegmentFile {
            meta: segment_meta(segment_id),
            records: Vec::new(),
        };
        write_segment(root, &segment)?;
    }

    if !segment_wal_path(root, segment_id).exists() {
        reset_wal(root, segment_id)?;
    }

    Ok(())
}

pub fn write_segment(root: &std::path::Path, segment: &SegmentFile) -> Result<(), Status> {
    let bytes = encode_segment(segment)?;
    write_file_atomically(&segment_data_path(root, segment.meta.id), &bytes)
}

pub fn read_segment(root: &std::path::Path, meta: &SegmentMeta) -> Result<SegmentFile, Status> {
    let bytes = read_file(&segment_data_path(root, meta.id))?;
    let mut segment = decode_segment(&bytes)?;
    segment.meta.path = meta.path.clone();
    Ok(segment)
}

pub fn remove_segment(root: &std::path::Path, segment_id: u64) -> Result<(), Status> {
    remove_path_if_exists(&segment_dir(root, segment_id))
}

pub fn doc_exists(records: &[StoredRecord], id: &DocId) -> bool {
    records
        .iter()
        .any(|record| record.doc.id == *id && !record.deleted)
}
