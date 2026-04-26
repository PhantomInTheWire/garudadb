use crate::segment_manager::SegmentManager;
use crate::validation::{apply_schema_defaults, validate_doc};
use garuda_meta::MetadataStore;
use garuda_segment::StoredRecord;
use garuda_types::{
    CollectionOptions, CollectionSchema, Doc, DocId, InternalDocId, ManifestVersionId, SegmentId,
    SnapshotId, StatusCode, WriteResult,
};
use std::path::PathBuf;

#[derive(Clone, Copy)]
pub(crate) enum WriteMode {
    Insert,
    Upsert,
}

#[derive(Clone)]
pub(crate) struct CollectionRuntime {
    pub(crate) path: PathBuf,
    pub(crate) schema: CollectionSchema,
    pub(crate) options: CollectionOptions,
    pub(crate) next_doc_id: InternalDocId,
    pub(crate) next_segment_id: SegmentId,
    pub(crate) id_map_snapshot_id: SnapshotId,
    pub(crate) delete_snapshot_id: SnapshotId,
    pub(crate) manifest_version_id: ManifestVersionId,
    pub(crate) revision: u64,
    pub(crate) segments: SegmentManager,
    pub(crate) meta: MetadataStore,
}

impl CollectionRuntime {
    pub(crate) fn insert_doc(&mut self, doc: Doc, mode: WriteMode) -> WriteResult {
        let mut doc = doc;
        apply_schema_defaults(&self.schema, &mut doc);

        if let Err(status) = validate_doc(&self.schema, &doc) {
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
        self.refresh_metadata();
        self.revision += 1;
        WriteResult::ok(inserted_doc_id)
    }

    pub(crate) fn update_doc(&mut self, doc: Doc) -> WriteResult {
        let Some(existing_doc) = self.record(&doc.id).map(|record| record.doc.clone()) else {
            return WriteResult::err(doc.id, StatusCode::NotFound, "document not found");
        };

        let merged_doc = merge_docs(&existing_doc, &doc);
        if let Err(status) = validate_doc(&self.schema, &merged_doc) {
            return WriteResult::err(doc.id, status.code, status.message);
        }

        self.delete_existing_if_present(&doc.id);
        self.append_new_record(merged_doc);
        self.refresh_metadata();
        self.revision += 1;
        WriteResult::ok(doc.id)
    }

    pub(crate) fn delete_doc(&mut self, id: DocId) -> WriteResult {
        if !self.delete_existing_if_present(&id) {
            return WriteResult::err(id, StatusCode::NotFound, "document not found");
        }

        self.revision += 1;
        WriteResult::ok(id)
    }

    pub(crate) fn rebuild_indexes(&mut self) {
        self.segments.rebuild_indexes(&self.schema);
        self.refresh_metadata();
    }

    pub(crate) fn live_doc_count(&self) -> usize {
        self.all_live_records().len()
    }

    pub(crate) fn all_live_records(&self) -> Vec<StoredRecord> {
        self.segments
            .all_live_records(|doc_id| self.meta.is_deleted(doc_id))
    }

    pub(crate) fn record(&self, id: &DocId) -> Option<&StoredRecord> {
        let internal_doc_id = self.meta.internal_doc_id(id)?;
        if self.meta.is_deleted(internal_doc_id) {
            return None;
        }

        self.segments.record_by_internal_id(internal_doc_id)
    }

    fn append_new_record(&mut self, doc: Doc) {
        let doc_id = self.next_doc_id;
        self.next_doc_id = self.next_doc_id.next();

        self.segments.append_new_record(
            doc_id,
            doc,
            &mut self.next_segment_id,
            self.options.segment_max_docs,
            &self.schema,
        );
    }

    fn delete_existing_if_present(&mut self, id: &DocId) -> bool {
        let Some(doc_id) = self.meta.internal_doc_id(id) else {
            return false;
        };
        if self.meta.is_deleted(doc_id) {
            return false;
        }

        if !self.segments.mark_deleted(doc_id) {
            return false;
        }

        self.meta.mark_deleted(doc_id);
        true
    }
}

impl CollectionRuntime {
    fn refresh_metadata(&mut self) {
        let deleted_doc_ids: Vec<_> = self.meta.deleted_doc_ids().copied().collect();
        self.segments
            .rebuild_metadata(&mut self.meta, &deleted_doc_ids);
    }
}

pub(crate) fn merge_docs(existing: &Doc, incoming: &Doc) -> Doc {
    let mut merged = existing.clone();

    for (key, value) in &incoming.fields {
        merged.fields.insert(key.clone(), value.clone());
    }

    if !incoming.vector.is_empty() {
        merged.vector = incoming.vector.clone();
    }

    merged
}
