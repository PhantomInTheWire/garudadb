mod bootstrap;
mod ddl;
mod delete_store;
mod filter;
mod filter_parser;
mod lock;
mod optimize;
mod persistence;
mod query;
mod schema;
mod state;
mod storage;
mod validation;

use bootstrap::{create_collection_state, load_collection_state};
use ddl::{
    backfill_new_column, drop_column_from_schema, drop_column_from_state,
    ensure_column_can_be_added, ensure_vector_index_field, rename_column_in_schema,
    rename_column_in_state, set_vector_index_kind,
};
use filter::evaluate_filter;
use garuda_math::score_doc;
use lock::CollectionLock;
use optimize::optimize_segments;
use persistence::checkpoint_state;
use query::{apply_query_projection, parse_query_filter, resolve_query_vector};
use schema::{validate_create_options, validate_schema};
use state::CollectionState;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use storage::{
    collection_dir, ensure_database_root, ensure_existing_collection_dir, ensure_new_collection_dir,
};
use validation::validate_field_default;

use garuda_segment::{WalOp, append_wal_ops};
use garuda_storage::WRITING_SEGMENT_ID;
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
    _lock: Arc<CollectionLock>,
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
        let lock = Arc::new(CollectionLock::acquire(&path)?);

        let collection = Collection {
            inner: Arc::new(RwLock::new(create_collection_state(path, schema, options))),
            _lock: lock,
        };
        collection.checkpoint()?;

        Ok(collection)
    }

    pub fn open_collection(&self, name: &CollectionName) -> Result<Collection, Status> {
        let path = collection_dir(&self.root, name);
        ensure_existing_collection_dir(&path)?;
        let lock = Arc::new(CollectionLock::acquire(&path)?);

        Ok(Collection {
            inner: Arc::new(RwLock::new(load_collection_state(path)?)),
            _lock: lock,
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
        self.checkpoint()
    }

    pub fn create_index(&self, field_name: &FieldName, kind: IndexKind) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
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
        self.mutate_and_checkpoint(|state| {
            ensure_column_can_be_added(state, &field)?;
            validate_field_default(&field)?;

            state.manifest.schema.fields.push(field.clone());
            backfill_new_column(state, &field);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn alter_column(&self, old_name: &FieldName, new_name: &FieldName) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            rename_column_in_schema(state, old_name, new_name)?;
            rename_column_in_state(state, old_name, new_name);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn drop_column(&self, name: &FieldName) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            drop_column_from_schema(state, name)?;
            drop_column_from_state(state, name);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn optimize(&self) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            optimize_segments(state);
            state.rebuild_indexes();
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn insert(&self, docs: Vec<Doc>) -> Vec<WriteResult> {
        self.apply_doc_write_batch(docs, WalOp::Insert, |state, doc| {
            state.insert_doc(doc, false)
        })
    }

    pub fn upsert(&self, docs: Vec<Doc>) -> Vec<WriteResult> {
        self.apply_doc_write_batch(docs, WalOp::Upsert, |state, doc| {
            state.insert_doc(doc, true)
        })
    }

    pub fn update(&self, docs: Vec<Doc>) -> Vec<WriteResult> {
        self.apply_doc_write_batch(docs, WalOp::Update, |state, doc| state.update_doc(doc))
    }

    pub fn delete(&self, ids: Vec<DocId>) -> Vec<WriteResult> {
        let mut state = self.write_state();
        let snapshot = state.clone();
        let mut results = Vec::new();
        let mut wal_ops = Vec::new();

        for id in ids {
            let result = state.delete_doc(&id);
            if result.status.is_ok() {
                wal_ops.push(WalOp::Delete(id.clone()));
            }

            results.push(result);
        }

        let persist_result = if wal_ops.is_empty() {
            Ok(())
        } else {
            append_wal_ops(&state.path, WRITING_SEGMENT_ID, &wal_ops)
        };

        if let Err(status) = persist_result {
            *state = snapshot;
            mark_persist_failure(&mut results, &status);
        }

        results
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

    fn checkpoint(&self) -> Result<(), Status> {
        let mut state = self.write_state();
        let snapshot = state.clone();

        if let Err(status) = checkpoint_state(&mut state) {
            *state = snapshot;
            return Err(status);
        }

        Ok(())
    }

    fn mutate_and_checkpoint(
        &self,
        mutate: impl FnOnce(&mut CollectionState) -> Result<(), Status>,
    ) -> Result<(), Status> {
        let mut state = self.write_state();
        let snapshot = state.clone();

        mutate(&mut state)?;
        if let Err(status) = checkpoint_state(&mut state) {
            *state = snapshot;
            return Err(status);
        }

        Ok(())
    }

    fn apply_doc_write_batch(
        &self,
        docs: Vec<Doc>,
        wal_op: impl Fn(Doc) -> WalOp,
        write_one: impl Fn(&mut CollectionState, Doc) -> WriteResult,
    ) -> Vec<WriteResult> {
        let mut state = self.write_state();
        let snapshot = state.clone();
        let mut results = Vec::new();
        let mut wal_ops = Vec::new();

        for doc in docs {
            let wal_doc = doc.clone();
            let result = write_one(&mut state, doc);
            if result.status.is_ok() {
                wal_ops.push(wal_op(wal_doc));
            }

            results.push(result);
        }

        let persist_result = if wal_ops.is_empty() {
            Ok(())
        } else {
            append_wal_ops(&state.path, WRITING_SEGMENT_ID, &wal_ops)
        };

        if let Err(status) = persist_result {
            *state = snapshot;
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

fn mark_persist_failure(results: &mut [WriteResult], status: &Status) {
    for result in results {
        if !result.status.is_ok() {
            continue;
        }

        result.status = Status::err(status.code.clone(), status.message.clone());
    }
}
