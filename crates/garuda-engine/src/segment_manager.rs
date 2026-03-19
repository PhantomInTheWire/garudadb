use garuda_segment::{
    RecordState, SegmentFile, StoredRecord, segment_file_name, segment_meta, sync_segment_meta,
};
use garuda_storage::WRITING_SEGMENT_ID;
use garuda_types::{Doc, DocId};

#[derive(Clone)]
pub(crate) struct SegmentManager {
    persisted_segments: Vec<SegmentFile>,
    writing_segment: SegmentFile,
}

impl SegmentManager {
    pub(crate) fn new(persisted_segments: Vec<SegmentFile>, writing_segment: SegmentFile) -> Self {
        Self {
            persisted_segments,
            writing_segment,
        }
    }

    pub(crate) fn empty_writing_segment() -> SegmentFile {
        SegmentFile {
            meta: segment_meta(WRITING_SEGMENT_ID),
            records: Vec::new(),
        }
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
        let mut records = Vec::with_capacity(persisted_capacity + self.writing_segment.records.len());

        collect_live_records(self.persisted_segments(), &mut records);
        collect_live_records_from_segment(self.writing_segment(), &mut records);

        records.sort_by_key(|record| record.doc_id);
        records
    }

    pub(crate) fn find_live_record(&self, id: &DocId) -> Option<&StoredRecord> {
        for record in &self.writing_segment.records {
            if record.doc.id == *id && matches!(record.state, RecordState::Live) {
                return Some(record);
            }
        }

        for segment in &self.persisted_segments {
            for record in &segment.records {
                if record.doc.id == *id && matches!(record.state, RecordState::Live) {
                    return Some(record);
                }
            }
        }

        None
    }

    pub(crate) fn find_live_record_mut(&mut self, id: &DocId) -> Option<&mut StoredRecord> {
        for record in &mut self.writing_segment.records {
            if record.doc.id == *id && matches!(record.state, RecordState::Live) {
                return Some(record);
            }
        }

        for segment in &mut self.persisted_segments {
            for record in &mut segment.records {
                if record.doc.id == *id && matches!(record.state, RecordState::Live) {
                    return Some(record);
                }
            }
        }

        None
    }

    pub(crate) fn append_new_record(
        &mut self,
        doc_id: u64,
        doc: Doc,
        next_segment_id: &mut u64,
        segment_max_docs: usize,
    ) {
        self.writing_segment.records.push(StoredRecord {
            doc_id,
            state: RecordState::Live,
            doc,
        });

        sync_segment_meta(&mut self.writing_segment);
        self.rotate_writing_segment_if_needed(next_segment_id, segment_max_docs);
    }

    pub(crate) fn optimize(&mut self, next_segment_id: &mut u64, segment_max_docs: usize) {
        let all_live_records = self.all_live_records();
        let mut rebuilt_segments = Vec::new();
        let mut current_segment = Self::empty_writing_segment();

        for record in all_live_records {
            if current_segment.records.len() >= segment_max_docs {
                seal_segment(&mut rebuilt_segments, current_segment, next_segment_id);
                current_segment = Self::empty_writing_segment();
            }

            current_segment.records.push(record);
        }

        if current_segment.records.is_empty() {
            self.persisted_segments = rebuilt_segments;
            self.writing_segment = Self::empty_writing_segment();
            return;
        }

        seal_segment(&mut rebuilt_segments, current_segment, next_segment_id);
        self.persisted_segments = rebuilt_segments;
        self.writing_segment = Self::empty_writing_segment();
    }

    fn rotate_writing_segment_if_needed(
        &mut self,
        next_segment_id: &mut u64,
        segment_max_docs: usize,
    ) {
        if self.writing_segment.meta.doc_count < segment_max_docs {
            return;
        }

        let new_id = *next_segment_id;
        *next_segment_id += 1;

        let mut persisted = std::mem::replace(&mut self.writing_segment, Self::empty_writing_segment());
        persisted.meta.id = new_id;
        persisted.meta.path = segment_file_name(new_id);
        sync_segment_meta(&mut persisted);
        self.persisted_segments.push(persisted);
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

fn seal_segment(
    rebuilt_segments: &mut Vec<SegmentFile>,
    mut segment: SegmentFile,
    next_segment_id: &mut u64,
) {
    sync_segment_meta(&mut segment);
    segment.meta.id = *next_segment_id;
    segment.meta.path = segment_file_name(*next_segment_id);
    *next_segment_id += 1;
    rebuilt_segments.push(segment);
}
