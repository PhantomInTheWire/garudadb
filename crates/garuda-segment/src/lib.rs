mod codec;
mod wal;

use codec::{decode_flat_index, decode_segment, encode_flat_index, encode_segment};
use garuda_index_flat::{FlatIndex, FlatIndexEntry};
use garuda_meta::evaluate_filter;
use garuda_storage::{
    WRITING_SEGMENT_ID, create_dir_all, read_file, remove_path_if_exists, segment_data_path,
    segment_dir, segment_flat_index_path, segment_wal_path, write_file_atomically,
};
use garuda_types::{
    DenseVector, DistanceMetric, Doc, DocId, FilterExpr, IndexParams, InternalDocId, SegmentId,
    SegmentMeta, Status, StatusCode, TopK, VectorDimension, VectorFieldSchema,
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
                StatusCode::Internal,
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
    vector_search: VectorSearchState,
}

#[derive(Clone, Debug, PartialEq)]
enum VectorSearchState {
    ScanRecords,
    UseFlatIndex(FlatIndex),
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

impl SegmentFile {
    pub fn new_writing(meta: SegmentMeta) -> Self {
        Self {
            meta,
            records: Vec::new(),
            vector_search: VectorSearchState::ScanRecords,
        }
    }

    pub fn new_persisted(meta: SegmentMeta) -> Self {
        Self {
            meta,
            records: Vec::new(),
            vector_search: VectorSearchState::ScanRecords,
        }
    }

    fn new_persisted_with_records(meta: SegmentMeta, records: Vec<StoredRecord>) -> Self {
        Self {
            meta,
            records,
            vector_search: VectorSearchState::ScanRecords,
        }
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

pub fn sync_vector_search(segment: &mut SegmentFile, vector_field: &VectorFieldSchema) {
    if !should_use_persisted_flat(segment.meta.id, vector_field, segment.meta.doc_count) {
        segment.vector_search = VectorSearchState::ScanRecords;
        return;
    }

    let entries = flat_index_entries(segment);
    let dimension = vector_field.dimension;
    let flat_index = FlatIndex::build(dimension, entries)
        .expect("validated segment records should match the vector field dimension");

    segment.vector_search = VectorSearchState::UseFlatIndex(flat_index);
}

pub fn ensure_segment_files(root: &std::path::Path, segment_id: SegmentId) -> Result<(), Status> {
    let segment_dir = segment_dir(root, segment_id);
    create_dir_all(&segment_dir, "failed to create segment directory")?;

    if !segment_data_path(root, segment_id).exists() {
        let segment = SegmentFile::new_writing(segment_meta(segment_id));
        let bytes = encode_segment(&segment)?;
        write_file_atomically(&segment_data_path(root, segment_id), &bytes)?;
    }

    if !segment_wal_path(root, segment_id).exists() {
        reset_wal(root, segment_id)?;
    }

    Ok(())
}

pub fn write_segment(
    root: &std::path::Path,
    segment: &SegmentFile,
    vector_field: &VectorFieldSchema,
) -> Result<(), Status> {
    let bytes = encode_segment(segment)?;
    write_file_atomically(&segment_data_path(root, segment.meta.id), &bytes)?;

    if should_use_persisted_flat(segment.meta.id, vector_field, segment.meta.doc_count) {
        let sidecar = encode_flat_index(flat_index_entries(segment), vector_field)?;
        write_file_atomically(&segment_flat_index_path(root, segment.meta.id), &sidecar)?;
        return Ok(());
    }

    remove_path_if_exists(&segment_flat_index_path(root, segment.meta.id))?;
    Ok(())
}

pub fn read_segment(
    root: &std::path::Path,
    meta: &SegmentMeta,
    vector_field: &VectorFieldSchema,
) -> Result<SegmentFile, Status> {
    let bytes = read_file(&segment_data_path(root, meta.id))?;
    let mut segment = decode_segment(&bytes)?;
    segment.meta.path = meta.path.clone();
    segment.vector_search = load_vector_search_state(root, &segment, vector_field)?;
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
    let mut records = std::collections::HashMap::new();

    for record in &segment.records {
        if matches!(record.state, RecordState::Deleted) {
            continue;
        }

        records.insert(record.doc_id, record.clone());
    }

    if records.is_empty() {
        return Ok(Vec::new());
    }

    let search_top_k = search_top_k(request, records.len())?;
    let hits = match &segment.vector_search {
        VectorSearchState::ScanRecords => scan_records(segment, request, search_top_k)?,
        VectorSearchState::UseFlatIndex(index) => {
            index.search(request.metric, request.query_vector, search_top_k)?
        }
    };
    let mut search_hits = Vec::with_capacity(hits.len());

    for hit in hits {
        let Some(record) = records.remove(&hit.doc_id) else {
            return Err(Status::err(
                StatusCode::Internal,
                "flat index hit missing backing record",
            ));
        };

        if let SegmentFilter::Matching(filter) = request.filter {
            if !evaluate_filter(filter, &record.doc.fields) {
                continue;
            }
        }

        search_hits.push(SegmentSearchHit {
            record,
            score: hit.score,
        });
    }

    Ok(search_hits)
}

fn scan_records(
    segment: &SegmentFile,
    request: ExactSearchRequest<'_>,
    top_k: TopK,
) -> Result<Vec<garuda_index_flat::FlatSearchHit>, Status> {
    let entries = flat_index_entries(segment);
    if entries.is_empty() {
        return Ok(Vec::new());
    }

    let dimension = VectorDimension::new(entries[0].vector.len())?;
    let index = FlatIndex::build(dimension, entries)?;
    index.search(request.metric, request.query_vector, top_k)
}

fn flat_index_entries(segment: &SegmentFile) -> Vec<FlatIndexEntry> {
    let mut entries = Vec::with_capacity(segment.meta.doc_count);

    for record in &segment.records {
        if matches!(record.state, RecordState::Deleted) {
            continue;
        }

        entries.push(FlatIndexEntry::new(
            record.doc_id,
            record.doc.vector.clone(),
        ));
    }

    entries
}

fn load_vector_search_state(
    root: &std::path::Path,
    segment: &SegmentFile,
    vector_field: &VectorFieldSchema,
) -> Result<VectorSearchState, Status> {
    if !should_use_persisted_flat(segment.meta.id, vector_field, segment.meta.doc_count) {
        return Ok(VectorSearchState::ScanRecords);
    }

    let bytes = read_file(&segment_flat_index_path(root, segment.meta.id))?;
    let flat_index = decode_flat_index(&bytes, vector_field)?;
    let expected_len = segment.meta.doc_count;

    if flat_index.len() != expected_len {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted flat index does not match segment live doc count",
        ));
    }

    Ok(VectorSearchState::UseFlatIndex(flat_index))
}

fn should_use_persisted_flat(
    segment_id: SegmentId,
    vector_field: &VectorFieldSchema,
    doc_count: usize,
) -> bool {
    if segment_id == WRITING_SEGMENT_ID || doc_count == 0 {
        return false;
    }

    matches!(vector_field.index, IndexParams::Flat(_))
}

fn search_top_k(request: ExactSearchRequest<'_>, live_doc_count: usize) -> Result<TopK, Status> {
    if matches!(request.filter, SegmentFilter::All) {
        return Ok(request.top_k);
    }

    TopK::new(live_doc_count).map_err(|_| {
        Status::err(
            StatusCode::Internal,
            "filtered exact search requires at least one live document",
        )
    })
}
