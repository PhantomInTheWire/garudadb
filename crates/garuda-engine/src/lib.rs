mod checkpoint_service;
mod filter;
mod filter_parser;
mod lock;
mod optimize;
mod query;
mod recovery_service;
mod schema_ddl;
mod schema;
mod segment_ddl;
mod segment_manager;
mod state;
mod storage;
mod validation;
mod write_service;

use checkpoint_service::checkpoint_state;
use garuda_math::score_doc;
use garuda_meta::evaluate_filter;
use lock::CollectionLock;
use optimize::optimize_segments;
use query::{apply_query_projection, parse_query_filter, resolve_query_vector};
use recovery_service::{create_collection_state, load_collection_state};
use schema::{validate_create_options, validate_schema};
use schema_ddl::{
    drop_column as drop_column_from_schema, ensure_column_can_be_added,
    ensure_vector_index_field, flat_index_params, rename_column as rename_column_in_schema,
    set_vector_index_params,
};
use segment_ddl::{
    backfill_new_column, drop_column as drop_column_from_state,
    rename_column as rename_column_in_state,
};
use state::CollectionState;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use storage::{
    collection_dir, ensure_database_root, ensure_existing_collection_dir, ensure_new_collection_dir,
};
use validation::validate_field_default;
use write_service::{WriteCommand, apply_write_command};

use garuda_types::{
    CollectionName, CollectionOptions, CollectionSchema, CollectionStats, Doc, DocId, FieldName,
    IndexParams, OptimizeOptions, ScalarFieldSchema, Status, StatusCode, VectorQuery, WriteResult,
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
            segment_count: state.segments.segment_count(),
        }
    }

    pub fn flush(&self) -> Result<(), Status> {
        self.checkpoint()
    }

    pub fn create_index(&self, field_name: &FieldName, params: IndexParams) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            ensure_vector_index_field(&state.manifest.schema, field_name)?;
            set_vector_index_params(&mut state.manifest.schema, params);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn drop_index(&self, field_name: &FieldName) -> Result<(), Status> {
        self.create_index(field_name, flat_index_params())
    }

    pub fn add_column(&self, field: ScalarFieldSchema) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            ensure_column_can_be_added(&state.manifest.schema, &field)?;
            validate_field_default(&field)?;

            state.manifest.schema.fields.push(field.clone());
            backfill_new_column(&mut state.segments, &field);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn alter_column(&self, old_name: &FieldName, new_name: &FieldName) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            rename_column_in_schema(&mut state.manifest.schema, old_name, new_name)?;
            rename_column_in_state(&mut state.segments, old_name, new_name);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn drop_column(&self, name: &FieldName) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            drop_column_from_schema(&mut state.manifest.schema, name)?;
            drop_column_from_state(&mut state.segments, name);
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn optimize(&self, _options: OptimizeOptions) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            optimize_segments(
                &mut state.segments,
                &mut state.manifest.next_segment_id,
                state.manifest.options.segment_max_docs,
            );
            state.rebuild_indexes();
            state.refresh_manifest();

            Ok(())
        })
    }

    pub fn insert(&self, docs: Vec<Doc>) -> Vec<WriteResult> {
        self.apply_write_command(WriteCommand::Insert(docs))
    }

    pub fn upsert(&self, docs: Vec<Doc>) -> Vec<WriteResult> {
        self.apply_write_command(WriteCommand::Upsert(docs))
    }

    pub fn update(&self, docs: Vec<Doc>) -> Vec<WriteResult> {
        self.apply_write_command(WriteCommand::Update(docs))
    }

    pub fn delete(&self, ids: Vec<DocId>) -> Vec<WriteResult> {
        self.apply_write_command(WriteCommand::Delete(ids))
    }

    pub fn delete_by_filter(&self, raw_filter: &str) -> Result<(), Status> {
        let state = self.read_state();
        let filter = parse_query_filter(Some(raw_filter), &state.manifest.schema)?;
        let Some(filter) = filter else {
            return Ok(());
        };

        let ids = collect_matching_doc_ids(&state, &filter);
        drop(state);

        for result in self.delete(ids) {
            if result.status.is_ok() {
                continue;
            }

            return Err(result.status);
        }

        Ok(())
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

    fn apply_write_command(&self, command: WriteCommand) -> Vec<WriteResult> {
        let mut state = self.write_state();
        apply_write_command(&mut state, command)
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

fn collect_matching_doc_ids(
    state: &CollectionState,
    filter: &garuda_types::FilterExpr,
) -> Vec<DocId> {
    let mut ids = Vec::new();

    for record in state.all_live_records() {
        if !evaluate_filter(filter, &record.doc.fields) {
            continue;
        }

        ids.push(record.doc.id);
    }

    ids
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
