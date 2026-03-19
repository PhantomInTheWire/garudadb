mod codec;
mod wal;

use codec::{decode_segment, encode_segment};
use garuda_index_flat::{FlatIndex, FlatIndexEntry};
use garuda_meta::evaluate_filter;
use garuda_storage::{
    create_dir_all, read_file, remove_path_if_exists, segment_data_path, segment_dir,
    segment_wal_path, write_file_atomically,
};
use garuda_types::{
    DenseVector, DistanceMetric, Doc, DocId, FilterExpr, InternalDocId, SegmentId, SegmentMeta,
    Status, TopK, VectorDimension,
};
pub use wal::{WalOp, append_wal_ops, read_wal_ops, reset_wal};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordState {
    Live,
    Deleted,
}

impl RecordState {
    pub fn to_tag(self) -> u8 {
        match self {
            Self::Live => 0,
            Self::Deleted => 1,
        }
    }

    pub fn from_tag(tag: u8) -> Result<Self, Status> {
        match tag {
            0 => Ok(Self::Live),
            1 => Ok(Self::Deleted),
            _ => Err(Status::err(
                garuda_types::StatusCode::Internal,
                "unrecognized record state tag",
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StoredRecord {
    pub doc_id: InternalDocId,
    pub state: RecordState,
    pub doc: Doc,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SegmentFile {
    pub meta: SegmentMeta,
    pub records: Vec<StoredRecord>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SegmentSearchHit {
    pub record: StoredRecord,
    pub score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SegmentFilter<'a> {
    All,
    Matching(&'a FilterExpr),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactSearchRequest<'a> {
    pub metric: DistanceMetric,
    pub query_vector: &'a DenseVector,
    pub top_k: TopK,
    pub filter: SegmentFilter<'a>,
}

pub fn segment_file_name(segment_id: SegmentId) -> String {
    segment_id.get().to_string()
}

pub fn segment_meta(id: SegmentId) -> SegmentMeta {
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

        if matches!(record.state, RecordState::Live) {
            live_doc_count += 1;
        }
    }

    segment.meta.min_doc_id = min_doc_id;
    segment.meta.max_doc_id = max_doc_id;
    segment.meta.doc_count = live_doc_count;
}

pub fn ensure_segment_files(root: &std::path::Path, segment_id: SegmentId) -> Result<(), Status> {
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

pub fn remove_segment(root: &std::path::Path, segment_id: SegmentId) -> Result<(), Status> {
    remove_path_if_exists(&segment_dir(root, segment_id))
}

pub fn doc_exists(records: &[StoredRecord], id: &DocId) -> bool {
    records
        .iter()
        .any(|record| record.doc.id == *id && matches!(record.state, RecordState::Live))
}

pub fn exact_search(
    segment: &SegmentFile,
    request: ExactSearchRequest<'_>,
) -> Result<Vec<SegmentSearchHit>, Status> {
    let mut entries = Vec::new();
    let mut records = std::collections::HashMap::new();

    for record in &segment.records {
        if matches!(record.state, RecordState::Deleted) {
            continue;
        }

        if let SegmentFilter::Matching(filter) = request.filter {
            if !evaluate_filter(filter, &record.doc.fields) {
                continue;
            }
        }

        entries.push(FlatIndexEntry::new(
            record.doc_id,
            record.doc.vector.clone(),
        ));
        records.insert(record.doc_id, record.clone());
    }

    if entries.is_empty() {
        return Ok(Vec::new());
    }

    let dimension = VectorDimension::new(entries[0].vector.len())?;
    let index = FlatIndex::build(dimension, entries)?;
    let hits = index.search(request.metric, request.query_vector, request.top_k)?;
    let mut search_hits = Vec::with_capacity(hits.len());

    for hit in hits {
        let Some(record) = records.remove(&hit.doc_id) else {
            return Err(Status::err(
                garuda_types::StatusCode::Internal,
                "flat index hit missing backing record",
            ));
        };

        search_hits.push(SegmentSearchHit {
            record,
            score: hit.score,
        });
    }

    Ok(search_hits)
}
