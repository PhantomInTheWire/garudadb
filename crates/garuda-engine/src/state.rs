use crate::catalog::CollectionCatalog;
use crate::segment_manager::{SegmentManager, TouchedSegment};
use crate::validation::{apply_schema_defaults, validate_doc};
use garuda_meta::MetadataStore;
use garuda_segment::{RecordState, SegmentFile, StoredRecord, rebuild_search_resources};
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

        if self.record(&doc.id).is_some() && matches!(mode, WriteMode::Insert) {
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
        let Some(existing_doc) = self.record(&doc.id).map(|record| record.doc.clone()) else {
            return WriteResult::err(doc.id, StatusCode::NotFound, "document not found");
        };

        let merged_doc = merge_docs(&existing_doc, &doc);
        if let Err(status) = validate_doc(&self.catalog.schema, &merged_doc) {
            return WriteResult::err(doc.id, status.code, status.message);
        }

        if !self
            .segments
            .update_doc(&doc.id, merged_doc, &self.catalog.schema.vector)
        {
            return WriteResult::err(doc.id, StatusCode::NotFound, "document not found");
        }

        self.finish_mutation();
        WriteResult::ok(doc.id)
    }

    pub(crate) fn delete_doc(&mut self, id: DocId) -> WriteResult {
        if !self.segments.delete_doc(&id, &self.catalog.schema.vector) {
            return WriteResult::err(id, StatusCode::NotFound, "document not found");
        }

        self.finish_mutation();
        WriteResult::ok(id)
    }

    pub(crate) fn rebuild_indexes(&mut self) {
        self.meta.clear();
        let vector_field = self.catalog.schema.vector.clone();

        for segment in self.segments.persisted_segments_mut() {
            rebuild_search_resources(segment, &vector_field);
            index_segment_meta(segment, &mut self.meta);
        }

        let writing_segment = self.segments.writing_segment_mut();
        rebuild_search_resources(writing_segment, &vector_field);
        index_segment_meta(writing_segment, &mut self.meta);
    }

    pub(crate) fn live_doc_count(&self) -> usize {
        self.segments.all_live_records().len()
    }

    pub(crate) fn all_live_records(&self) -> Vec<StoredRecord> {
        self.segments.all_live_records()
    }

    pub(crate) fn record(&self, id: &DocId) -> Option<&StoredRecord> {
        let internal_doc_id = self.meta.internal_doc_id(id)?;
        if self.meta.is_deleted(internal_doc_id) {
            return None;
        }

        self.segments.record_by_internal_id(internal_doc_id)
    }

    fn append_new_record(&mut self, doc: Doc) {
        let doc_id = self.catalog.next_doc_id;
        self.catalog.next_doc_id = self.catalog.next_doc_id.next();

        self.segments.append_new_record(
            doc_id,
            doc,
            &mut self.catalog.next_segment_id,
            self.catalog.options.segment_max_docs,
            &self.catalog.schema.vector,
        );
    }

    fn finish_mutation(&mut self) {
        self.refresh_metadata();
    }

    fn delete_existing_if_present(&mut self, id: &DocId) {
        let Some(touched) = self.segments.mark_deleted(id) else {
            return;
        };

        if matches!(touched, TouchedSegment::Writing) {
            return;
        }

        self.segments
            .rebuild_touched_segment(touched, &self.catalog.schema.vector);
    }
}

impl CollectionRuntime {
    fn refresh_metadata(&mut self) {
        self.meta.clear();

        for segment in self.segments.persisted_segments() {
            index_segment_meta(segment, &mut self.meta);
        }

        index_segment_meta(self.segments.writing_segment(), &mut self.meta);
    }
}

fn index_segment_meta(segment: &SegmentFile, meta: &mut MetadataStore) {
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
