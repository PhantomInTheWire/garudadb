use crate::segment_manager::SegmentManager;
use crate::validation::{apply_schema_defaults, validate_doc};
use garuda_meta::{DeleteStore, IdMap};
use garuda_segment::{RecordState, SegmentFile, StoredRecord, WalOp, sync_segment_meta};
use garuda_types::{Doc, DocId, Manifest, Status, StatusCode, WriteResult};
use std::path::PathBuf;

#[derive(Clone, Copy)]
pub(crate) enum WriteMode {
    Insert,
    Upsert,
}

#[derive(Clone)]
pub(crate) struct CollectionState {
    pub(crate) path: PathBuf,
    pub(crate) manifest: Manifest,
    pub(crate) segments: SegmentManager,
    pub(crate) id_map: IdMap,
    pub(crate) delete_store: DeleteStore,
}

impl CollectionState {
    pub(crate) fn apply_wal_op(&mut self, op: &WalOp) -> Result<(), Status> {
        if self.is_redundant_wal_op(op) {
            return Ok(());
        }

        let result = match op {
            WalOp::Insert(doc) => self.insert_doc(doc.clone(), WriteMode::Insert),
            WalOp::Upsert(doc) => self.insert_doc(doc.clone(), WriteMode::Upsert),
            WalOp::Update(doc) => self.update_doc(doc.clone()),
            WalOp::Delete(doc_id) => self.delete_doc(doc_id),
        };

        if result.status.is_ok() {
            return Ok(());
        }

        if matches!(op, WalOp::Update(_)) && result.status.code == StatusCode::NotFound {
            return Ok(());
        }

        Err(Status::err(result.status.code, result.status.message))
    }

    fn is_redundant_wal_op(&self, op: &WalOp) -> bool {
        match op {
            WalOp::Insert(doc) => self.insert_already_applied(doc),
            WalOp::Upsert(doc) => self.live_doc_matches(doc),
            WalOp::Update(doc) => {
                self.update_already_applied(doc) || self.find_live_record(&doc.id).is_none()
            }
            WalOp::Delete(doc_id) => self.find_live_record(doc_id).is_none(),
        }
    }

    pub(crate) fn insert_doc(&mut self, doc: Doc, mode: WriteMode) -> WriteResult {
        let mut doc = doc;
        apply_schema_defaults(&self.manifest.schema, &mut doc);

        if let Err(status) = validate_doc(&self.manifest.schema, &doc) {
            return WriteResult::err(doc.id, status.code, status.message);
        }

        if self.find_live_record(&doc.id).is_some() && matches!(mode, WriteMode::Insert) {
            return WriteResult::err(doc.id, StatusCode::AlreadyExists, "document already exists");
        }

        if matches!(mode, WriteMode::Upsert) {
            self.delete_existing_if_present(&doc.id);
        }

        self.append_new_record(doc.clone());
        self.finish_mutation();
        WriteResult::ok(doc.id)
    }

    pub(crate) fn update_doc(&mut self, doc: Doc) -> WriteResult {
        let schema = self.manifest.schema.clone();

        let Some(record) = self.find_live_record_mut(&doc.id) else {
            return WriteResult::err(doc.id, StatusCode::NotFound, "document not found");
        };

        let merged_doc = merge_docs(&record.doc, &doc);
        if let Err(status) = validate_doc(&schema, &merged_doc) {
            return WriteResult::err(doc.id, status.code, status.message);
        }

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
        self.id_map.clear();
        self.delete_store.clear();

        for segment in self.segments.persisted_segments_mut() {
            index_segment(segment, &mut self.id_map, &mut self.delete_store);
        }

        index_segment(
            self.segments.writing_segment_mut(),
            &mut self.id_map,
            &mut self.delete_store,
        );
    }

    pub(crate) fn refresh_manifest(&mut self) {
        self.segments.refresh_manifest(&mut self.manifest);
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
        let doc_id = self.manifest.next_doc_id;
        self.manifest.next_doc_id += 1;

        self.segments.append_new_record(
            doc_id,
            doc,
            &mut self.manifest.next_segment_id,
            self.manifest.options.segment_max_docs,
        );
    }

    fn finish_mutation(&mut self) {
        self.rebuild_indexes();
        self.refresh_manifest();
    }

    fn delete_existing_if_present(&mut self, id: &DocId) {
        let Some(record) = self.find_live_record_mut(id) else {
            return;
        };

        record.state = RecordState::Deleted;
    }

    fn live_doc_matches(&self, doc: &Doc) -> bool {
        let Some(record) = self.find_live_record(&doc.id) else {
            return false;
        };

        record.doc == *doc
    }

    fn insert_already_applied(&self, doc: &Doc) -> bool {
        self.find_live_record(&doc.id).is_some()
    }

    fn update_already_applied(&self, doc: &Doc) -> bool {
        let Some(record) = self.find_live_record(&doc.id) else {
            return false;
        };

        let merged_doc = merge_docs(&record.doc, doc);
        merged_doc == record.doc
    }
}

fn index_segment(segment: &mut SegmentFile, id_map: &mut IdMap, delete_store: &mut DeleteStore) {
    sync_segment_meta(segment);

    for record in &segment.records {
        if matches!(record.state, RecordState::Deleted) {
            delete_store.insert(record.doc_id);
            continue;
        }

        id_map.insert(record.doc.id.clone(), record.doc_id);
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
