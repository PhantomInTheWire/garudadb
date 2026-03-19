mod catalog;
mod checkpoint_service;
mod filter;
mod filter_parser;
mod lock;
mod query;
mod recovery_service;
mod schema;
mod schema_ddl;
mod segment_ddl;
mod segment_manager;
mod state;
mod storage;
mod validation;
mod write_service;

use checkpoint_service::checkpoint_state;
use lock::CollectionLock;
use query::execute_query;
use recovery_service::{create_collection_state, load_collection_state};
use schema::{validate_create_options, validate_schema};
use schema_ddl::{
    drop_column as drop_column_from_schema, ensure_column_can_be_added, ensure_vector_index_field,
    flat_index_params, rename_column as rename_column_in_schema, set_vector_index_params,
};
use segment_ddl::{
    backfill_new_column, drop_column as drop_column_from_state,
    rename_column as rename_column_in_state,
};
use state::CollectionRuntime;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use storage::{
    collection_dir, ensure_database_root, ensure_existing_collection_dir, ensure_new_collection_dir,
};
use validation::validate_field_default;
use write_service::{WriteCommand, apply_delete_by_filter, apply_write_command};

use garuda_types::{
    CollectionName, CollectionOptions, CollectionSchema, CollectionStats, Doc, DocId, FieldName,
    IndexParams, OptimizeOptions, ScalarFieldSchema, Status, VectorQuery, WriteResult,
};

#[derive(Clone)]
pub struct Database {
    root: PathBuf,
}

#[derive(Clone)]
pub struct Collection {
    inner: Arc<RwLock<CollectionRuntime>>,
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
        self.read_state().catalog.schema.clone()
    }

    pub fn options(&self) -> CollectionOptions {
        self.read_state().catalog.options.clone()
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
            ensure_vector_index_field(&state.catalog.schema, field_name)?;
            set_vector_index_params(&mut state.catalog.schema, params);

            Ok(())
        })
    }

    pub fn drop_index(&self, field_name: &FieldName) -> Result<(), Status> {
        self.create_index(field_name, flat_index_params())
    }

    pub fn add_column(&self, field: ScalarFieldSchema) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            ensure_column_can_be_added(&state.catalog.schema, &field)?;
            validate_field_default(&field)?;

            backfill_new_column(&mut state.segments, &field);
            state.catalog.schema.fields.push(field);

            Ok(())
        })
    }

    pub fn alter_column(&self, old_name: &FieldName, new_name: &FieldName) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            rename_column_in_schema(&mut state.catalog.schema, old_name, new_name)?;
            rename_column_in_state(&mut state.segments, old_name, new_name);

            Ok(())
        })
    }

    pub fn drop_column(&self, name: &FieldName) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            drop_column_from_schema(&mut state.catalog.schema, name)?;
            drop_column_from_state(&mut state.segments, name);

            Ok(())
        })
    }

    pub fn optimize(&self, _options: OptimizeOptions) -> Result<(), Status> {
        self.mutate_and_checkpoint(|state| {
            state.segments.optimize(
                &mut state.catalog.next_segment_id,
                state.catalog.options.segment_max_docs,
            );
            state.rebuild_indexes();

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
        let mut state = self.write_state();
        apply_delete_by_filter(&mut state, raw_filter)
    }

    pub fn fetch(&self, ids: Vec<DocId>) -> HashMap<DocId, Doc> {
        let state = self.read_state();
        let mut docs = HashMap::with_capacity(ids.len());

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
        execute_query(&state, query)
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
        mutate: impl FnOnce(&mut CollectionRuntime) -> Result<(), Status>,
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

    fn read_state(&self) -> std::sync::RwLockReadGuard<'_, CollectionRuntime> {
        self.inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write_state(&self) -> std::sync::RwLockWriteGuard<'_, CollectionRuntime> {
        self.inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn fetch_doc(state: &CollectionRuntime, id: &DocId) -> Option<Doc> {
    let record = state.record(id)?;
    let mut doc = record.doc.clone();
    doc.score = Some(0.0);
    Some(doc)
}
