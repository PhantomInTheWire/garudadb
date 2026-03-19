use crate::catalog::CollectionCatalog;
use crate::segment_manager::SegmentManager;
use crate::validation::{apply_schema_defaults, validate_doc};
use garuda_meta::MetadataStore;
use garuda_segment::{RecordState, SegmentFile, StoredRecord, sync_segment_meta};
use garuda_types::{Doc, DocId, StatusCode, WriteResult};
use std::path::PathBuf;

#[derive(Clone, Copy)]
pub(crate) enum WriteMode {
    Insert,
    Upsert,
}

#[derive(Clone)]
pub(crate) struct CollectionRuntime {
    pub(crate) path: PathBuf,
    pub(crate) catalog: CollectionCatalog,
    pub(crate) segments: SegmentManager,
    pub(crate) meta: MetadataStore,
}

impl CollectionRuntime {
    pub(crate) fn insert_doc(&mut self, doc: Doc, mode: WriteMode) -> WriteResult {
        let mut doc = doc;
        apply_schema_defaults(&self.catalog.schema, &mut doc);

        if let Err(status) = validate_doc(&self.catalog.schema, &doc) {
            return WriteResult::err(doc.id, status.code, status.message);
        }

        if self.find_live_record(&doc.id).is_some() && matches!(mode, WriteMode::Insert) {
            return WriteResult::err(doc.id, StatusCode::AlreadyExists, "document already exists");
        }

        if matches!(mode, WriteMode::Upsert) {
            self.delete_existing_if_present(&doc.id);
        }

        let inserted_doc_id = doc.id.clone();
        self.append_new_record(doc);
        self.finish_mutation();
        WriteResult::ok(inserted_doc_id)
    }

    pub(crate) fn update_doc(&mut self, doc: Doc) -> WriteResult {
        let Some(existing_doc) = self
            .find_live_record(&doc.id)
            .map(|record| record.doc.clone())
        else {
            return WriteResult::err(doc.id, StatusCode::NotFound, "document not found");
        };

        let merged_doc = merge_docs(&existing_doc, &doc);
        if let Err(status) = validate_doc(&self.catalog.schema, &merged_doc) {
            return WriteResult::err(doc.id, status.code, status.message);
        }

        let Some(record) = self.find_live_record_mut(&doc.id) else {
            return WriteResult::err(doc.id, StatusCode::NotFound, "document not found");
        };

        record.doc = merged_doc;
        self.finish_mutation();
        WriteResult::ok(doc.id)
    }

    pub(crate) fn delete_doc(&mut self, id: &DocId) -> WriteResult {
        let Some(record) = self.find_live_record_mut(id) else {
            return WriteResult::err(id.clone(), StatusCode::NotFound, "document not found");
        };

        record.state = RecordState::Deleted;
        self.finish_mutation();
        WriteResult::ok(id.clone())
    }

    pub(crate) fn rebuild_indexes(&mut self) {
        self.meta.clear();

        for segment in self.segments.persisted_segments_mut() {
            index_segment(segment, &mut self.meta);
        }

        index_segment(self.segments.writing_segment_mut(), &mut self.meta);
    }

    pub(crate) fn live_doc_count(&self) -> usize {
        self.segments.all_live_records().len()
    }

    pub(crate) fn all_live_docs(&self) -> Vec<Doc> {
        self.segments
            .all_live_records()
            .into_iter()
            .map(|record| record.doc)
            .collect()
    }

    pub(crate) fn all_live_records(&self) -> Vec<StoredRecord> {
        self.segments.all_live_records()
    }

    pub(crate) fn find_live_record(&self, id: &DocId) -> Option<&StoredRecord> {
        self.segments.find_live_record(id)
    }

    pub(crate) fn find_live_record_mut(&mut self, id: &DocId) -> Option<&mut StoredRecord> {
        self.segments.find_live_record_mut(id)
    }
    fn append_new_record(&mut self, doc: Doc) {
        let doc_id = self.catalog.next_doc_id;
        self.catalog.next_doc_id += 1;

        self.segments.append_new_record(
            doc_id,
            doc,
            &mut self.catalog.next_segment_id,
            self.catalog.options.segment_max_docs,
        );
    }

    fn finish_mutation(&mut self) {
        self.rebuild_indexes();
    }

    fn delete_existing_if_present(&mut self, id: &DocId) {
        let Some(record) = self.find_live_record_mut(id) else {
            return;
        };

        record.state = RecordState::Deleted;
    }
}

fn index_segment(segment: &mut SegmentFile, meta: &mut MetadataStore) {
    sync_segment_meta(segment);

    for record in &segment.records {
        if matches!(record.state, RecordState::Deleted) {
            meta.mark_deleted(record.doc_id);
            continue;
        }

        meta.index_live_doc(record.doc.id.clone(), record.doc_id);
    }
}

fn merge_docs(existing: &Doc, incoming: &Doc) -> Doc {
    let mut merged = existing.clone();

    for (key, value) in &incoming.fields {
        merged.fields.insert(key.clone(), value.clone());
    }

    if !incoming.vector.is_empty() {
        merged.vector = incoming.vector.clone();
    }

    merged
}
