use garuda_segment::{
    RecordState, SegmentFile, SegmentKind, StoredRecord, rebuild_search_resources,
    segment_file_name, segment_meta,
};
use garuda_storage::WRITING_SEGMENT_ID;
use garuda_types::{Doc, DocId, InternalDocId, SegmentId, VectorFieldSchema};

#[derive(Clone)]
pub(crate) struct SegmentManager {
    persisted_segments: Vec<SegmentFile>,
    writing_segment: SegmentFile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TouchedSegment {
    Writing,
    Persisted(usize),
}

impl SegmentManager {
    pub(crate) fn new(persisted_segments: Vec<SegmentFile>, writing_segment: SegmentFile) -> Self {
        Self {
            persisted_segments,
            writing_segment,
        }
    }

    pub(crate) fn empty_writing_segment(vector_field: &VectorFieldSchema) -> SegmentFile {
        SegmentFile::new(
            segment_meta(WRITING_SEGMENT_ID),
            Vec::new(),
            SegmentKind::Writing,
            vector_field,
        )
    }

    pub(crate) fn persisted_segments(&self) -> &[SegmentFile] {
        &self.persisted_segments
    }

    pub(crate) fn persisted_segments_mut(&mut self) -> &mut [SegmentFile] {
        &mut self.persisted_segments
    }

    pub(crate) fn writing_segment(&self) -> &SegmentFile {
        &self.writing_segment
    }

    pub(crate) fn writing_segment_mut(&mut self) -> &mut SegmentFile {
        &mut self.writing_segment
    }

    pub(crate) fn segment_count(&self) -> usize {
        self.persisted_segments.len() + 1
    }

    pub(crate) fn all_live_records(&self) -> Vec<StoredRecord> {
        let persisted_capacity = self
            .persisted_segments
            .iter()
            .map(|segment| segment.records.len())
            .sum::<usize>();
        let mut records =
            Vec::with_capacity(persisted_capacity + self.writing_segment.records.len());

        collect_live_records(self.persisted_segments(), &mut records);
        collect_live_records_from_segment(self.writing_segment(), &mut records);

        records.sort_by_key(|record| record.doc_id);
        records
    }

    pub(crate) fn record_by_internal_id(&self, doc_id: InternalDocId) -> Option<&StoredRecord> {
        if let Some(record) = record_in_segment_by_internal_id(&self.writing_segment, doc_id) {
            return Some(record);
        }

        for segment in &self.persisted_segments {
            if !segment_contains_doc_id(segment, doc_id) {
                continue;
            }

            if let Some(record) = record_in_segment_by_internal_id(segment, doc_id) {
                return Some(record);
            }
        }

        None
    }

    pub(crate) fn append_new_record(
        &mut self,
        doc_id: InternalDocId,
        doc: Doc,
        next_segment_id: &mut SegmentId,
        segment_max_docs: usize,
        vector_field: &VectorFieldSchema,
    ) {
        self.writing_segment.records.push(StoredRecord {
            doc_id,
            state: RecordState::Live,
            doc,
        });

        rebuild_search_resources(&mut self.writing_segment, vector_field);
        self.rotate_writing_segment_if_needed(next_segment_id, segment_max_docs, vector_field);
    }

    pub(crate) fn update_doc(
        &mut self,
        id: &DocId,
        doc: Doc,
        vector_field: &VectorFieldSchema,
    ) -> bool {
        let Some(touched) = self.mutate_doc(id, |record| record.doc = doc.clone()) else {
            return false;
        };

        self.rebuild_touched_segment(touched, vector_field);
        true
    }

    pub(crate) fn delete_doc(&mut self, id: &DocId, vector_field: &VectorFieldSchema) -> bool {
        let Some(touched) = self.mark_deleted(id) else {
            return false;
        };

        self.rebuild_touched_segment(touched, vector_field);
        true
    }

    pub(crate) fn mark_deleted(&mut self, id: &DocId) -> Option<TouchedSegment> {
        self.mutate_doc(id, |record| record.state = RecordState::Deleted)
    }

    pub(crate) fn optimize(
        &mut self,
        next_segment_id: &mut SegmentId,
        segment_max_docs: usize,
        vector_field: &VectorFieldSchema,
    ) {
        let all_live_records = self.all_live_records();
        let rebuilt_capacity =
            (all_live_records.len() + segment_max_docs.saturating_sub(1)) / segment_max_docs.max(1);
        let mut rebuilt_segments = Vec::with_capacity(rebuilt_capacity);
        let mut current_segment = Self::empty_writing_segment(vector_field);

        for record in all_live_records {
            if current_segment.records.len() >= segment_max_docs {
                seal_segment(&mut rebuilt_segments, current_segment, next_segment_id);
                current_segment = Self::empty_writing_segment(vector_field);
            }

            current_segment.records.push(record);
        }

        if current_segment.records.is_empty() {
            self.persisted_segments = rebuilt_segments;
            self.writing_segment = Self::empty_writing_segment(vector_field);
            return;
        }

        seal_segment(&mut rebuilt_segments, current_segment, next_segment_id);
        self.persisted_segments = rebuilt_segments;
        self.writing_segment = Self::empty_writing_segment(vector_field);
    }

    fn rotate_writing_segment_if_needed(
        &mut self,
        next_segment_id: &mut SegmentId,
        segment_max_docs: usize,
        vector_field: &VectorFieldSchema,
    ) {
        if self.writing_segment.meta.doc_count < segment_max_docs {
            return;
        }

        let new_id = *next_segment_id;
        *next_segment_id = next_segment_id.next();

        let mut persisted = std::mem::replace(
            &mut self.writing_segment,
            Self::empty_writing_segment(vector_field),
        );
        persisted.mark_persisted();
        persisted.meta.id = new_id;
        persisted.meta.path = segment_file_name(new_id);
        persisted.sync_meta();
        self.persisted_segments.push(persisted);
    }

    pub(crate) fn rebuild_touched_segment(
        &mut self,
        touched: TouchedSegment,
        vector_field: &VectorFieldSchema,
    ) {
        match touched {
            TouchedSegment::Writing => {
                rebuild_search_resources(&mut self.writing_segment, vector_field);
            }
            TouchedSegment::Persisted(index) => {
                rebuild_search_resources(&mut self.persisted_segments[index], vector_field);
            }
        }
    }

    fn mutate_doc(
        &mut self,
        id: &DocId,
        mut mutate: impl FnMut(&mut StoredRecord),
    ) -> Option<TouchedSegment> {
        let touched = self.find_live_doc(id)?;

        match touched {
            TouchedSegment::Writing => {
                let record = live_record_in_segment(&mut self.writing_segment, id)
                    .expect("writing segment record should exist");
                mutate(record);
            }
            TouchedSegment::Persisted(index) => {
                let record = live_record_in_segment(&mut self.persisted_segments[index], id)
                    .expect("persisted segment record should exist");
                mutate(record);
            }
        }

        Some(touched)
    }

    fn find_live_doc(&self, id: &DocId) -> Option<TouchedSegment> {
        if has_live_doc(&self.writing_segment, id) {
            return Some(TouchedSegment::Writing);
        }

        for (index, segment) in self.persisted_segments.iter().enumerate() {
            if has_live_doc(segment, id) {
                return Some(TouchedSegment::Persisted(index));
            }
        }

        None
    }
}

fn collect_live_records(segments: &[SegmentFile], out: &mut Vec<StoredRecord>) {
    for segment in segments {
        collect_live_records_from_segment(segment, out);
    }
}

fn collect_live_records_from_segment(segment: &SegmentFile, out: &mut Vec<StoredRecord>) {
    for record in &segment.records {
        if matches!(record.state, RecordState::Deleted) {
            continue;
        }

        out.push(record.clone());
    }
}

fn segment_contains_doc_id(segment: &SegmentFile, doc_id: InternalDocId) -> bool {
    let Some(min_doc_id) = segment.meta.min_doc_id else {
        return false;
    };
    let Some(max_doc_id) = segment.meta.max_doc_id else {
        return false;
    };

    min_doc_id <= doc_id && doc_id <= max_doc_id
}

fn record_in_segment_by_internal_id(
    segment: &SegmentFile,
    doc_id: InternalDocId,
) -> Option<&StoredRecord> {
    segment
        .records
        .iter()
        .find(|record| record.doc_id == doc_id && matches!(record.state, RecordState::Live))
}

fn seal_segment(
    rebuilt_segments: &mut Vec<SegmentFile>,
    mut segment: SegmentFile,
    next_segment_id: &mut SegmentId,
) {
    segment.mark_persisted();
    segment.sync_meta();
    segment.meta.id = *next_segment_id;
    segment.meta.path = segment_file_name(*next_segment_id);
    *next_segment_id = next_segment_id.next();
    rebuilt_segments.push(segment);
}

fn live_record_in_segment<'a>(
    segment: &'a mut SegmentFile,
    id: &DocId,
) -> Option<&'a mut StoredRecord> {
    for record in &mut segment.records {
        if record.doc.id != *id || matches!(record.state, RecordState::Deleted) {
            continue;
        }

        return Some(record);
    }

    None
}

fn has_live_doc(segment: &SegmentFile, id: &DocId) -> bool {
    segment
        .records
        .iter()
        .any(|record| record.doc.id == *id && matches!(record.state, RecordState::Live))
}
