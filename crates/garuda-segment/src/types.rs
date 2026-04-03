use crate::index::{build_persisted_search_resources, build_writing_search_resources};
use garuda_index_flat::{FlatIndex, WritingFlatIndex};
use garuda_index_hnsw::{HnswIndex, WritingHnswIndex};
use garuda_index_ivf::{IvfIndex, WritingIvfIndex};
use garuda_index_scalar::ScalarIndex;
use garuda_types::{
    CollectionSchema, DenseVector, DistanceMetric, Doc, FieldName, FilterExpr, HnswEfSearch,
    InternalDocId, IvfProbeCount, RemoveResult, SegmentId, SegmentMeta, Status, StatusCode, TopK,
};
use std::collections::{BTreeMap, HashMap};

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
    record_indexes: HashMap<InternalDocId, usize>,
    pub flat_index: Option<WritingFlatIndex>,
    pub hnsw_index: Option<WritingHnswIndex>,
    pub ivf_index: Option<WritingIvfIndex>,
    pub scalar_indexes: BTreeMap<FieldName, ScalarIndex>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PersistedSegment {
    pub meta: SegmentMeta,
    pub records: Vec<StoredRecord>,
    record_indexes: HashMap<InternalDocId, usize>,
    pub flat_index: Option<FlatIndex>,
    pub hnsw_index: Option<HnswIndex>,
    pub ivf_index: Option<IvfIndex>,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IvfSegmentSearchRequest<'a> {
    pub query_vector: &'a DenseVector,
    pub top_k: TopK,
    pub nprobe: IvfProbeCount,
    pub filter: SegmentFilter<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SegmentSearchRequest<'a> {
    Flat(FlatSearchRequest<'a>),
    Hnsw(HnswSegmentSearchRequest<'a>),
    Ivf(IvfSegmentSearchRequest<'a>),
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
        let record_indexes = record_indexes(&records);
        let resources = build_writing_search_resources(schema, &records);

        Self {
            meta,
            records,
            record_indexes,
            flat_index: resources.flat_index,
            hnsw_index: resources.hnsw_index,
            ivf_index: resources.ivf_index,
            scalar_indexes: resources.scalar_indexes,
        }
    }

    pub fn sync_meta(&mut self) {
        sync_segment_meta_fields(&mut self.meta, &self.records);
    }

    pub fn seal(self, segment_id: SegmentId, schema: &CollectionSchema) -> PersistedSegment {
        let mut meta = self.meta;
        meta.id = segment_id;
        meta.path = segment_file_name(segment_id);
        PersistedSegment::new(meta, self.records, schema)
    }

    pub fn push_record(&mut self, record: StoredRecord) {
        let record_index = self.records.len();
        self.meta.min_doc_id = Some(
            self.meta
                .min_doc_id
                .map_or(record.doc_id, |min| min.min(record.doc_id)),
        );
        self.meta.max_doc_id = Some(
            self.meta
                .max_doc_id
                .map_or(record.doc_id, |max| max.max(record.doc_id)),
        );
        if matches!(record.state, RecordState::Live) {
            self.meta.doc_count += 1;
        }
        self.record_indexes.insert(record.doc_id, record_index);
        self.records.push(record);
    }

    pub fn mark_deleted(&mut self, doc_id: InternalDocId) -> bool {
        mark_deleted_record(
            &mut self.records,
            &self.record_indexes,
            &mut self.meta,
            &mut self.scalar_indexes,
            doc_id,
            |doc_id| {
                if let Some(index) = &mut self.flat_index {
                    assert_eq!(index.remove(doc_id), RemoveResult::Removed);
                }
            },
            |doc_id| {
                if let Some(index) = &mut self.hnsw_index {
                    assert_eq!(index.remove(doc_id), RemoveResult::Removed);
                }
            },
            |doc_id| {
                if let Some(index) = &mut self.ivf_index {
                    assert_eq!(index.remove(doc_id), RemoveResult::Removed);
                }
            },
        )
        .is_some()
    }
}

impl PersistedSegment {
    pub fn new(meta: SegmentMeta, records: Vec<StoredRecord>, schema: &CollectionSchema) -> Self {
        let mut meta = meta;
        sync_segment_meta_fields(&mut meta, &records);
        let resources = build_persisted_search_resources(schema, &meta, &records);

        Self::from_parts(meta, records, resources)
    }

    pub(crate) fn from_parts(
        meta: SegmentMeta,
        records: Vec<StoredRecord>,
        resources: crate::index::PersistedSearchResources,
    ) -> Self {
        let record_indexes = record_indexes(&records);

        Self {
            meta,
            records,
            record_indexes,
            flat_index: resources.flat_index,
            hnsw_index: resources.hnsw_index,
            ivf_index: resources.ivf_index,
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
        self.ivf_index = resources.ivf_index;
        self.scalar_indexes = resources.scalar_indexes;
    }

    pub fn mark_deleted(&mut self, doc_id: InternalDocId) -> bool {
        mark_deleted_record(
            &mut self.records,
            &self.record_indexes,
            &mut self.meta,
            &mut self.scalar_indexes,
            doc_id,
            |doc_id| {
                if let Some(index) = &mut self.flat_index {
                    assert_eq!(index.remove(doc_id), RemoveResult::Removed);
                }
            },
            |doc_id| {
                if let Some(index) = &mut self.hnsw_index {
                    assert_eq!(index.remove(doc_id), RemoveResult::Removed);
                }
            },
            |doc_id| {
                if let Some(index) = &mut self.ivf_index {
                    assert_eq!(index.remove(doc_id), RemoveResult::Removed);
                }
            },
        )
        .is_some()
    }
}

fn mark_deleted_record(
    records: &mut [StoredRecord],
    record_indexes: &HashMap<InternalDocId, usize>,
    meta: &mut SegmentMeta,
    scalar_indexes: &mut BTreeMap<FieldName, ScalarIndex>,
    doc_id: InternalDocId,
    remove_from_flat: impl FnOnce(InternalDocId),
    remove_from_hnsw: impl FnOnce(InternalDocId),
    remove_from_ivf: impl FnOnce(InternalDocId),
) -> Option<()> {
    let &record_index = record_indexes.get(&doc_id)?;
    if !matches!(records[record_index].state, RecordState::Live) {
        return None;
    }
    let scalar_fields = deleted_record_scalar_fields(&records[record_index], scalar_indexes);
    records[record_index].state = RecordState::Deleted;
    assert!(
        meta.doc_count > 0,
        "segment live doc count should include deleted record"
    );
    meta.doc_count -= 1;
    remove_from_flat(doc_id);
    remove_from_ivf(doc_id);
    remove_from_hnsw(doc_id);
    remove_from_scalar_indexes(scalar_indexes, doc_id, scalar_fields);
    Some(())
}

fn record_indexes(records: &[StoredRecord]) -> HashMap<InternalDocId, usize> {
    let mut indexes = HashMap::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        indexes.insert(record.doc_id, index);
    }
    indexes
}

fn deleted_record_scalar_fields(
    record: &StoredRecord,
    scalar_indexes: &BTreeMap<FieldName, ScalarIndex>,
) -> Vec<(FieldName, garuda_types::ScalarValue)> {
    let mut values = Vec::with_capacity(scalar_indexes.len());

    for field in scalar_indexes.keys() {
        let value = record
            .doc
            .fields
            .get(field.as_str())
            .expect("validated indexed scalar field should exist")
            .clone();
        values.push((field.clone(), value));
    }

    values
}

fn remove_from_scalar_indexes(
    scalar_indexes: &mut BTreeMap<FieldName, ScalarIndex>,
    doc_id: InternalDocId,
    scalar_fields: Vec<(FieldName, garuda_types::ScalarValue)>,
) {
    for (field, value) in scalar_fields {
        let index = scalar_indexes
            .get_mut(&field)
            .expect("enabled scalar index should exist");
        assert_eq!(index.remove(doc_id, &value), RemoveResult::Removed);
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
