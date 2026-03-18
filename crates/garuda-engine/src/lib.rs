mod bootstrap;
mod ddl;
mod filter;
mod filter_parser;
mod query;
mod schema;
mod scoring;
mod state;
mod storage;
mod storage_io;
mod validation;
mod version;

use bootstrap::{create_collection_state, load_collection_state};
use ddl::{
    backfill_new_column, drop_column_from_schema, drop_column_from_state,
    ensure_column_can_be_added, ensure_vector_index_field, rename_column_in_schema,
    rename_column_in_state, set_vector_index_kind,
};
use filter::evaluate_filter;
use query::{apply_query_projection, parse_query_filter, resolve_query_vector};
use schema::{validate_create_options, validate_schema};
use scoring::score_doc;
use state::CollectionState;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use storage::{
    collection_dir, ensure_database_root, ensure_existing_collection_dir,
    ensure_new_collection_dir, sync_segment_meta, write_delete_store, write_id_map, write_segment,
};
use validation::validate_field_default;
use version::VersionStore;

use garuda_types::{
    CollectionName, CollectionOptions, CollectionSchema, CollectionStats, Doc, DocId, FieldName,
    IndexKind, ScalarFieldSchema, Status, StatusCode, VectorQuery, WriteResult,
};

#[derive(Clone)]
pub struct Database {
    root: PathBuf,
}

#[derive(Clone)]
pub struct Collection {
    inner: Arc<RwLock<CollectionState>>,
}

impl Database {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, Status> {
        ensure_database_root(root.as_ref())?;

        Ok(Self {
            root: root.as_ref().to_path_buf(),
        })
    }

    pub fn create_collection(
        &self,
        schema: CollectionSchema,
        options: CollectionOptions,
    ) -> Result<Collection, Status> {
        validate_schema(&schema)?;
        validate_create_options(&options)?;

        let path = collection_dir(&self.root, &schema.name);
        ensure_new_collection_dir(&path)?;

        let collection = Collection {
            inner: Arc::new(RwLock::new(create_collection_state(path, schema, options))),
        };
        collection.persist_all()?;

        Ok(collection)
    }

    pub fn open_collection(&self, name: &CollectionName) -> Result<Collection, Status> {
        let path = collection_dir(&self.root, name);
        ensure_existing_collection_dir(&path)?;

        Ok(Collection {
            inner: Arc::new(RwLock::new(load_collection_state(path)?)),
        })
    }
}

impl Collection {
    pub fn path(&self) -> PathBuf {
        self.read_state().path.clone()
    }

    pub fn schema(&self) -> CollectionSchema {
        self.read_state().manifest.schema.clone()
    }

    pub fn options(&self) -> CollectionOptions {
        self.read_state().manifest.options.clone()
    }

    pub fn stats(&self) -> CollectionStats {
        let state = self.read_state();

        CollectionStats {
            doc_count: state.live_doc_count(),
            segment_count: state.persisted_segments.len() + 1,
        }
    }

    pub fn flush(&self) -> Result<(), Status> {
        self.persist_all()
    }

    pub fn create_index(&self, field_name: &FieldName, kind: IndexKind) -> Result<(), Status> {
        self.mutate_and_persist(|state| {
            ensure_vector_index_field(state, field_name)?;
            set_vector_index_kind(state, kind);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn drop_index(&self, field_name: &FieldName) -> Result<(), Status> {
        self.create_index(field_name, IndexKind::Flat)
    }

    pub fn add_column(&self, field: ScalarFieldSchema) -> Result<(), Status> {
        self.mutate_and_persist(|state| {
            ensure_column_can_be_added(state, &field)?;
            validate_field_default(&field)?;

            state.manifest.schema.fields.push(field.clone());
            backfill_new_column(state, &field);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn alter_column(&self, old_name: &FieldName, new_name: &FieldName) -> Result<(), Status> {
        self.mutate_and_persist(|state| {
            rename_column_in_schema(state, old_name, new_name)?;
            rename_column_in_state(state, old_name, new_name);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn drop_column(&self, name: &FieldName) -> Result<(), Status> {
        self.mutate_and_persist(|state| {
            drop_column_from_schema(state, name)?;
            drop_column_from_state(state, name);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn optimize(&self) -> Result<(), Status> {
        self.mutate_and_persist(|state| {
            optimize_segments(state);
            state.rebuild_indexes();
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn insert(&self, docs: Vec<Doc>) -> Vec<WriteResult> {
        self.mutate_writes_and_persist(|state| {
            run_doc_writes(state, docs, |state, doc| state.insert_doc(doc, false))
        })
    }

    pub fn upsert(&self, docs: Vec<Doc>) -> Vec<WriteResult> {
        self.mutate_writes_and_persist(|state| {
            run_doc_writes(state, docs, |state, doc| state.insert_doc(doc, true))
        })
    }

    pub fn update(&self, docs: Vec<Doc>) -> Vec<WriteResult> {
        self.mutate_writes_and_persist(|state| {
            run_doc_writes(state, docs, |state, doc| state.update_doc(doc))
        })
    }

    pub fn delete(&self, ids: Vec<DocId>) -> Vec<WriteResult> {
        self.mutate_writes_and_persist(|state| {
            let mut results = Vec::new();

            for id in ids {
                results.push(state.delete_doc(&id));
            }

            results
        })
    }

    pub fn fetch(&self, ids: Vec<DocId>) -> HashMap<DocId, Doc> {
        let state = self.read_state();
        let mut docs = HashMap::new();

        for id in ids {
            let Some(doc) = fetch_doc(&state, &id) else {
                continue;
            };

            docs.insert(id, doc);
        }

        docs
    }

    pub fn query(&self, query: VectorQuery) -> Result<Vec<Doc>, Status> {
        let state = self.read_state();
        ensure_query_uses_known_vector_field(&state, &query)?;

        let filter = parse_query_filter(query.filter.as_deref(), &state.manifest.schema)?;
        let query_vector = resolve_query_vector(&query, &state)?;
        let docs = collect_matching_docs(&state, &query, filter.as_ref())?;

        Ok(score_and_sort_docs(&state, docs, &query_vector, &query))
    }

    fn persist_all(&self) -> Result<(), Status> {
        let state = self.read_state();

        write_id_map(&state.path, &state.id_map)?;
        write_delete_store(&state.path, &state.deleted_doc_ids)?;
        write_segment(&state.path, &state.writing_segment)?;

        for segment in &state.persisted_segments {
            write_segment(&state.path, segment)?;
        }

        VersionStore::new(&state.path).write_manifest(&state.manifest)?;
        Ok(())
    }

    fn restore_state(&self, snapshot: CollectionState) {
        let mut state = self.write_state();
        *state = snapshot;
    }

    fn mutate_and_persist(
        &self,
        mutate: impl FnOnce(&mut CollectionState) -> Result<(), Status>,
    ) -> Result<(), Status> {
        let mut state = self.write_state();
        let snapshot = state.clone();

        mutate(&mut state)?;
        drop(state);

        if let Err(status) = self.persist_all() {
            self.restore_state(snapshot);
            return Err(status);
        }

        Ok(())
    }

    fn mutate_writes_and_persist(
        &self,
        mutate: impl FnOnce(&mut CollectionState) -> Vec<WriteResult>,
    ) -> Vec<WriteResult> {
        let mut state = self.write_state();
        let snapshot = state.clone();
        let mut results = mutate(&mut state);
        drop(state);

        if let Err(status) = self.persist_all() {
            self.restore_state(snapshot);
            mark_persist_failure(&mut results, &status);
        }

        results
    }

    fn read_state(&self) -> std::sync::RwLockReadGuard<'_, CollectionState> {
        self.inner.read().expect("collection lock poisoned")
    }

    fn write_state(&self) -> std::sync::RwLockWriteGuard<'_, CollectionState> {
        self.inner.write().expect("collection lock poisoned")
    }
}

fn run_doc_writes(
    state: &mut CollectionState,
    docs: Vec<Doc>,
    write_one: impl Fn(&mut CollectionState, Doc) -> WriteResult,
) -> Vec<WriteResult> {
    let mut results = Vec::new();

    for doc in docs {
        results.push(write_one(state, doc));
    }

    results
}

fn fetch_doc(state: &CollectionState, id: &DocId) -> Option<Doc> {
    let record = state.find_live_record(id)?;
    let mut doc = record.doc.clone();
    doc.score = Some(0.0);
    Some(doc)
}

fn ensure_query_uses_known_vector_field(
    state: &CollectionState,
    query: &VectorQuery,
) -> Result<(), Status> {
    if query.field_name == state.manifest.schema.vector.name {
        return Ok(());
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "unknown vector field",
    ))
}

fn collect_matching_docs(
    state: &CollectionState,
    query: &VectorQuery,
    filter: Option<&garuda_types::FilterExpr>,
) -> Result<Vec<Doc>, Status> {
    let mut docs = state.all_live_docs();

    if let Some(filter) = filter {
        docs.retain(|doc| evaluate_filter(filter, &doc.fields));
    }

    if query.top_k == 0 {
        return Ok(Vec::new());
    }

    Ok(docs)
}

fn score_and_sort_docs(
    state: &CollectionState,
    docs: Vec<Doc>,
    query_vector: &[f32],
    query: &VectorQuery,
) -> Vec<Doc> {
    let mut scored_docs = Vec::new();

    for mut doc in docs {
        doc.score = Some(score_doc(
            state.manifest.schema.vector.metric,
            query_vector,
            &doc.vector,
        ));
        apply_query_projection(&mut doc, query);
        scored_docs.push(doc);
    }

    scored_docs.sort_by(|lhs, rhs| {
        rhs.score
            .unwrap_or_default()
            .total_cmp(&lhs.score.unwrap_or_default())
            .then_with(|| lhs.id.cmp(&rhs.id))
    });
    scored_docs.truncate(query.top_k);

    scored_docs
}

fn optimize_segments(state: &mut CollectionState) {
    let all_live_records = state.all_live_records();
    let mut rebuilt_segments = Vec::new();
    let mut current_segment = CollectionState::empty_writing_segment();

    for record in all_live_records {
        if current_segment.records.len() >= state.manifest.options.segment_max_docs {
            seal_segment(state, &mut rebuilt_segments, current_segment);
            current_segment = CollectionState::empty_writing_segment();
        }

        current_segment.records.push(record);
    }

    if current_segment.records.is_empty() {
        state.persisted_segments = rebuilt_segments;
        state.writing_segment = CollectionState::empty_writing_segment();
        return;
    }

    seal_segment(state, &mut rebuilt_segments, current_segment);
    state.persisted_segments = rebuilt_segments;
    state.writing_segment = CollectionState::empty_writing_segment();
}

fn seal_segment(
    state: &mut CollectionState,
    rebuilt_segments: &mut Vec<storage::SegmentFile>,
    mut segment: storage::SegmentFile,
) {
    sync_segment_meta(&mut segment);
    segment.meta.id = state.manifest.next_segment_id;
    segment.meta.path = storage::segment_file_name(state.manifest.next_segment_id);
    state.manifest.next_segment_id += 1;
    rebuilt_segments.push(segment);
}

fn mark_persist_failure(results: &mut [WriteResult], status: &Status) {
    for result in results {
        if !result.status.is_ok() {
            continue;
        }

        result.status = Status::err(status.code.clone(), status.message.clone());
    }
}
