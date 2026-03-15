use garuda_types::{
    CollectionOptions, CollectionSchema, CollectionStats, Doc, IndexKind, ScalarFieldSchema,
    Status, VectorQuery, WriteResult,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct Database {
    root: PathBuf,
}

impl Database {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, Status> {
        Ok(Self {
            root: root.as_ref().to_path_buf(),
        })
    }

    pub fn create_collection(
        &self,
        _schema: CollectionSchema,
        _options: CollectionOptions,
    ) -> Result<Collection, Status> {
        let _ = &self.root;
        unimplemented!(
            "GarudaDB implementation intentionally removed; only e2e tests should define the contract for now"
        )
    }

    pub fn open_collection(&self, _name: &str) -> Result<Collection, Status> {
        let _ = &self.root;
        unimplemented!(
            "GarudaDB implementation intentionally removed; only e2e tests should define the contract for now"
        )
    }
}

#[derive(Clone)]
pub struct Collection;

impl Collection {
    pub fn path(&self) -> PathBuf {
        unimplemented!("contract only")
    }

    pub fn schema(&self) -> CollectionSchema {
        unimplemented!("contract only")
    }

    pub fn options(&self) -> CollectionOptions {
        unimplemented!("contract only")
    }

    pub fn stats(&self) -> CollectionStats {
        unimplemented!("contract only")
    }

    pub fn flush(&self) -> Result<(), Status> {
        unimplemented!("contract only")
    }

    pub fn create_index(&self, _field_name: &str, _kind: IndexKind) -> Result<(), Status> {
        unimplemented!("contract only")
    }

    pub fn drop_index(&self, _field_name: &str) -> Result<(), Status> {
        unimplemented!("contract only")
    }

    pub fn add_column(&self, _field: ScalarFieldSchema) -> Result<(), Status> {
        unimplemented!("contract only")
    }

    pub fn alter_column(&self, _old_name: &str, _new_name: &str) -> Result<(), Status> {
        unimplemented!("contract only")
    }

    pub fn drop_column(&self, _name: &str) -> Result<(), Status> {
        unimplemented!("contract only")
    }

    pub fn optimize(&self) -> Result<(), Status> {
        unimplemented!("contract only")
    }

    pub fn insert(&self, _docs: Vec<Doc>) -> Vec<WriteResult> {
        unimplemented!("contract only")
    }

    pub fn upsert(&self, _docs: Vec<Doc>) -> Vec<WriteResult> {
        unimplemented!("contract only")
    }

    pub fn update(&self, _docs: Vec<Doc>) -> Vec<WriteResult> {
        unimplemented!("contract only")
    }

    pub fn delete(&self, _ids: Vec<String>) -> Vec<WriteResult> {
        unimplemented!("contract only")
    }

    pub fn fetch(&self, _ids: Vec<String>) -> HashMap<String, Doc> {
        unimplemented!("contract only")
    }

    pub fn query(&self, _query: VectorQuery) -> Result<Vec<Doc>, Status> {
        unimplemented!("contract only")
    }
}
