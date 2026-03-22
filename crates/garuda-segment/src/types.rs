use crate::index::build_vector_search_state;
use garuda_index_flat::FlatIndex;
use garuda_index_hnsw::HnswIndex;
use garuda_types::{
    DenseVector, DistanceMetric, Doc, FilterExpr, HnswEfSearch, InternalDocId, SegmentId,
    SegmentMeta, Status, StatusCode, TopK, VectorFieldSchema,
};

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
    pub(crate) kind: SegmentKind,
    pub meta: SegmentMeta,
    pub records: Vec<StoredRecord>,
    pub(crate) flat_index: Option<FlatIndex>,
    pub(crate) hnsw_index: Option<HnswIndex>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentKind {
    Writing,
    Persisted,
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
pub struct FlatSearchRequest<'a> {
    pub metric: DistanceMetric,
    pub query_vector: &'a DenseVector,
    pub top_k: TopK,
    pub filter: SegmentFilter<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HnswSegmentSearchRequest<'a> {
    pub query_vector: &'a DenseVector,
    pub top_k: TopK,
    pub ef_search: HnswEfSearch,
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
    pub fn new(
        meta: SegmentMeta,
        records: Vec<StoredRecord>,
        kind: SegmentKind,
        vector_field: &VectorFieldSchema,
    ) -> Self {
        let mut meta = meta;
        sync_segment_meta_fields(&mut meta, &records);
        let (flat_index, hnsw_index) = build_vector_search_state(vector_field, &meta, &records);

        Self {
            kind,
            meta,
            records,
            flat_index,
            hnsw_index,
        }
    }

    pub fn is_writing(&self) -> bool {
        matches!(self.kind, SegmentKind::Writing)
    }

    pub fn mark_persisted(&mut self) {
        self.kind = SegmentKind::Persisted;
    }

    pub fn sync_meta(&mut self) {
        sync_segment_meta_fields(&mut self.meta, &self.records);
    }
}

impl SegmentKind {
    pub(crate) fn from_segment_id(id: SegmentId) -> Self {
        if id == garuda_storage::WRITING_SEGMENT_ID {
            return Self::Writing;
        }

        Self::Persisted
    }
}

pub(crate) fn sync_segment_meta_fields(meta: &mut SegmentMeta, records: &[StoredRecord]) {
    let mut min_doc_id = None;
    let mut max_doc_id = None;
    let mut live_doc_count = 0usize;

    for record in records {
        let record_id = record.doc_id;

        min_doc_id = Some(min_doc_id.unwrap_or(record_id).min(record_id));
        max_doc_id = Some(max_doc_id.unwrap_or(record_id).max(record_id));

        if matches!(record.state, RecordState::Live) {
            live_doc_count += 1;
        }
    }

    meta.min_doc_id = min_doc_id;
    meta.max_doc_id = max_doc_id;
    meta.doc_count = live_doc_count;
}
