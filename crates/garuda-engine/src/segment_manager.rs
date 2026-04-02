use garuda_index_ivf::IvfBuildEntry;
use garuda_meta::MetadataStore;
use garuda_segment::{
    segment_file_name, segment_meta, PersistedSegment, RecordState, StoredRecord, WritingSegment,
};
use garuda_storage::WRITING_SEGMENT_ID;
use garuda_types::{CollectionSchema, Doc, InternalDocId, SegmentId};

#[derive(Clone)]
pub(crate) struct SegmentManager {
    persisted_segments: Vec<PersistedSegment>,
    writing_segment: WritingSegment,
}

impl SegmentManager {
    pub(crate) fn new(
        persisted_segments: Vec<PersistedSegment>,
        writing_segment: WritingSegment,
    ) -> Self {
        Self {
            persisted_segments,
            writing_segment,
        }
    }

    pub(crate) fn empty_writing_segment(schema: &CollectionSchema) -> WritingSegment {
        WritingSegment::new(segment_meta(WRITING_SEGMENT_ID), Vec::new(), schema)
    }

    pub(crate) fn persisted_segments(&self) -> &[PersistedSegment] {
        &self.persisted_segments
    }

    pub(crate) fn writing_segment(&self) -> &WritingSegment {
        &self.writing_segment
    }

    pub(crate) fn apply_to_all_records(&mut self, mut apply: impl FnMut(&mut [StoredRecord])) {
        for segment in &mut self.persisted_segments {
            apply(&mut segment.records);
            segment.sync_meta();
        }

        apply(&mut self.writing_segment.records);
        self.writing_segment.sync_meta();
    }

    pub(crate) fn segment_count(&self) -> usize {
        self.persisted_segments.len() + 1
    }

    pub(crate) fn all_live_records(
        &self,
        is_deleted: impl Fn(InternalDocId) -> bool,
    ) -> Vec<StoredRecord> {
        let persisted_capacity = self
            .persisted_segments
            .iter()
            .map(|segment| segment.records.len())
            .sum::<usize>();
        let mut records =
            Vec::with_capacity(persisted_capacity + self.writing_segment.records.len());

        for segment in &self.persisted_segments {
            collect_visible_records(&segment.records, &is_deleted, &mut records);
        }

        collect_visible_records(&self.writing_segment.records, &is_deleted, &mut records);
        records.sort_by_key(|record| record.doc_id);
        records
    }

    pub(crate) fn record_by_internal_id(&self, doc_id: InternalDocId) -> Option<&StoredRecord> {
        if let Some(record) =
            record_in_segment_by_internal_id(&self.writing_segment.records, doc_id)
        {
            return Some(record);
        }

        for segment in &self.persisted_segments {
            if !segment_contains_doc_id(&segment.meta, doc_id) {
                continue;
            }

            if let Some(record) = record_in_segment_by_internal_id(&segment.records, doc_id) {
                return Some(record);
            }
        }

        None
    }

    pub(crate) fn rebuild_metadata(
        &self,
        meta: &mut MetadataStore,
        deleted_doc_ids: &[InternalDocId],
    ) {
        meta.clear();

        for &doc_id in deleted_doc_ids {
            meta.mark_deleted(doc_id);
        }

        for segment in &self.persisted_segments {
            index_segment_meta(&segment.records, meta);
        }

        index_segment_meta(&self.writing_segment.records, meta);
    }

    pub(crate) fn rebuild_indexes(&mut self, schema: &CollectionSchema) {
        self.persisted_segments = self
            .persisted_segments
            .iter()
            .map(|segment| {
                PersistedSegment::new(segment.meta.clone(), segment.records.clone(), schema)
            })
            .collect();
        self.writing_segment = WritingSegment::new(
            self.writing_segment.meta.clone(),
            self.writing_segment.records.clone(),
            schema,
        );
    }

    pub(crate) fn append_new_record(
        &mut self,
        doc_id: InternalDocId,
        doc: Doc,
        next_segment_id: &mut SegmentId,
        segment_max_docs: usize,
        schema: &CollectionSchema,
    ) {
        self.writing_segment.records.push(StoredRecord {
            doc_id,
            state: RecordState::Live,
            doc: doc.clone(),
        });

        if let Some(index) = &mut self.writing_segment.flat_index {
            index.insert(doc_id, doc.vector.clone());
        }

        if let Some(index) = &mut self.writing_segment.hnsw_index {
            index.insert(doc_id, doc.vector.clone());
        }

        if let Some(index) = &mut self.writing_segment.ivf_index {
            index.insert(IvfBuildEntry::new(doc_id, doc.vector.clone()));
        }

        for field in &schema.fields {
            if !field.is_indexed() {
                continue;
            }

            let value = doc
                .fields
                .get(field.name.as_str())
                .expect("validated indexed scalar field should exist");
            let index = self
                .writing_segment
                .scalar_indexes
                .get_mut(&field.name)
                .expect("enabled writing scalar index should exist");
            index.insert(doc_id, value);
        }

        self.writing_segment.sync_meta();
        self.rotate_writing_segment_if_needed(next_segment_id, segment_max_docs, schema);
    }

    pub(crate) fn mark_writing_deleted(&mut self, doc_id: InternalDocId) -> bool {
        self.writing_segment.mark_deleted(doc_id)
    }

    pub(crate) fn mark_deleted(&mut self, doc_id: InternalDocId) -> bool {
        if self.mark_writing_deleted(doc_id) {
            return true;
        }

        for index in 0..self.persisted_segments.len() {
            if !segment_contains_doc_id(&self.persisted_segments[index].meta, doc_id) {
                continue;
            }

            let segment = &mut self.persisted_segments[index];
            if segment.mark_deleted(doc_id) {
                return true;
            }
        }

        false
    }

    pub(crate) fn optimize(
        &mut self,
        next_segment_id: &mut SegmentId,
        segment_max_docs: usize,
        schema: &CollectionSchema,
        is_deleted: impl Fn(InternalDocId) -> bool,
    ) {
        let all_live_records = self.all_live_records(is_deleted);
        let rebuilt_capacity =
            (all_live_records.len() + segment_max_docs.saturating_sub(1)) / segment_max_docs.max(1);
        let mut rebuilt_segments = Vec::with_capacity(rebuilt_capacity);
        let mut current_segment = Self::empty_writing_segment(schema);

        for record in all_live_records {
            if current_segment.records.len() >= segment_max_docs {
                seal_segment(
                    &mut rebuilt_segments,
                    current_segment,
                    next_segment_id,
                    schema,
                );
                current_segment = Self::empty_writing_segment(schema);
            }

            current_segment.records.push(record);
        }

        if !current_segment.records.is_empty() {
            seal_segment(
                &mut rebuilt_segments,
                current_segment,
                next_segment_id,
                schema,
            );
        }

        self.persisted_segments = rebuilt_segments;
        self.writing_segment = Self::empty_writing_segment(schema);
    }

    pub(crate) fn seal_writing_segment(
        &mut self,
        next_segment_id: &mut SegmentId,
        schema: &CollectionSchema,
    ) {
        if self.writing_segment.meta.doc_count == 0 {
            self.writing_segment = Self::empty_writing_segment(schema);
            return;
        }

        let segment_id = *next_segment_id;
        *next_segment_id = next_segment_id.next();
        let writing = std::mem::replace(
            &mut self.writing_segment,
            Self::empty_writing_segment(schema),
        );
        self.persisted_segments
            .push(writing.seal(segment_id, schema));
    }

    fn rotate_writing_segment_if_needed(
        &mut self,
        next_segment_id: &mut SegmentId,
        segment_max_docs: usize,
        schema: &CollectionSchema,
    ) {
        if self.writing_segment.meta.doc_count < segment_max_docs {
            return;
        }

        self.seal_writing_segment(next_segment_id, schema);
    }
}

fn collect_visible_records(
    records: &[StoredRecord],
    is_deleted: &impl Fn(InternalDocId) -> bool,
    out: &mut Vec<StoredRecord>,
) {
    for record in records {
        if matches!(record.state, RecordState::Deleted) || is_deleted(record.doc_id) {
            continue;
        }

        out.push(record.clone());
    }
}

fn segment_contains_doc_id(meta: &garuda_types::SegmentMeta, doc_id: InternalDocId) -> bool {
    let Some(min_doc_id) = meta.min_doc_id else {
        return false;
    };
    let Some(max_doc_id) = meta.max_doc_id else {
        return false;
    };

    min_doc_id <= doc_id && doc_id <= max_doc_id
}

fn record_in_segment_by_internal_id(
    records: &[StoredRecord],
    doc_id: InternalDocId,
) -> Option<&StoredRecord> {
    records
        .iter()
        .find(|record| record.doc_id == doc_id && matches!(record.state, RecordState::Live))
}

fn seal_segment(
    rebuilt_segments: &mut Vec<PersistedSegment>,
    current_segment: WritingSegment,
    next_segment_id: &mut SegmentId,
    schema: &CollectionSchema,
) {
    let new_id = *next_segment_id;
    *next_segment_id = next_segment_id.next();
    let mut segment = current_segment.seal(new_id, schema);
    segment.meta.path = segment_file_name(new_id);
    rebuilt_segments.push(segment);
}

fn index_segment_meta(records: &[StoredRecord], meta: &mut MetadataStore) {
    for record in records {
        if matches!(record.state, RecordState::Deleted) {
            meta.mark_deleted(record.doc_id);
            continue;
        }

        meta.index_live_doc(record.doc.id.clone(), record.doc_id);
    }
}
