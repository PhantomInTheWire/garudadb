use crate::index::{build_persisted_search_resources, build_writing_search_resources};
use garuda_index_flat::{FlatIndex, WritingFlatIndex};
use garuda_index_hnsw::{HnswIndex, WritingHnswIndex};
use garuda_index_scalar::ScalarIndex;
use garuda_types::{
    CollectionSchema, DenseVector, DistanceMetric, Doc, FieldName, FilterExpr, HnswEfSearch,
    InternalDocId, ScalarPrefilter, SegmentId, SegmentMeta, Status, StatusCode, TopK,
};
use std::collections::{BTreeMap, HashSet};

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
pub struct WritingSegment {
    pub meta: SegmentMeta,
    pub records: Vec<StoredRecord>,
    pub flat_index: Option<WritingFlatIndex>,
    pub hnsw_index: Option<WritingHnswIndex>,
    pub scalar_indexes: BTreeMap<FieldName, ScalarIndex>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PersistedSegment {
    pub meta: SegmentMeta,
    pub records: Vec<StoredRecord>,
    pub flat_index: Option<FlatIndex>,
    pub hnsw_index: Option<HnswIndex>,
    pub scalar_indexes: BTreeMap<FieldName, ScalarIndex>,
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

impl WritingSegment {
    pub fn new(meta: SegmentMeta, records: Vec<StoredRecord>, schema: &CollectionSchema) -> Self {
        let mut meta = meta;
        sync_segment_meta_fields(&mut meta, &records);
        let resources = build_writing_search_resources(schema, &records);

        Self {
            meta,
            records,
            flat_index: resources.flat_index,
            hnsw_index: resources.hnsw_index,
            scalar_indexes: resources.scalar_indexes,
        }
    }

    pub fn sync_meta(&mut self) {
        sync_segment_meta_fields(&mut self.meta, &self.records);
    }
    pub fn prefilter_doc_ids(&self, prefilter: &ScalarPrefilter) -> Option<HashSet<InternalDocId>> {
        prefilter_doc_ids(prefilter, &self.scalar_indexes)
    }
}

impl PersistedSegment {
    pub fn new(meta: SegmentMeta, records: Vec<StoredRecord>, schema: &CollectionSchema) -> Self {
        let mut meta = meta;
        sync_segment_meta_fields(&mut meta, &records);
        let resources = build_persisted_search_resources(schema, &meta, &records);

        Self {
            meta,
            records,
            flat_index: resources.flat_index,
            hnsw_index: resources.hnsw_index,
            scalar_indexes: resources.scalar_indexes,
        }
    }

    pub fn sync_meta(&mut self) {
        sync_segment_meta_fields(&mut self.meta, &self.records);
    }

    pub fn rebuild_search_resources(&mut self, schema: &CollectionSchema) {
        self.sync_meta();
        let resources = build_persisted_search_resources(schema, &self.meta, &self.records);
        self.flat_index = resources.flat_index;
        self.hnsw_index = resources.hnsw_index;
        self.scalar_indexes = resources.scalar_indexes;
    }

    pub fn prefilter_doc_ids(&self, prefilter: &ScalarPrefilter) -> Option<HashSet<InternalDocId>> {
        prefilter_doc_ids(prefilter, &self.scalar_indexes)
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

fn prefilter_doc_ids(
    prefilter: &ScalarPrefilter,
    scalar_indexes: &BTreeMap<FieldName, ScalarIndex>,
) -> Option<HashSet<InternalDocId>> {
    let ScalarPrefilter::And(predicates) = prefilter else {
        return None;
    };

    let mut doc_ids: Option<HashSet<InternalDocId>> = None;

    for predicate in predicates {
        let index = scalar_indexes
            .get(&predicate.field)
            .expect("planned scalar prefilter should have an index");
        let matching_doc_ids = index.matching_doc_ids(predicate);

        doc_ids = Some(match doc_ids {
            None => matching_doc_ids,
            Some(doc_ids) => doc_ids
                .into_iter()
                .filter(|doc_id| matching_doc_ids.contains(doc_id))
                .collect(),
        });
    }

    Some(doc_ids.unwrap_or_default())
}
