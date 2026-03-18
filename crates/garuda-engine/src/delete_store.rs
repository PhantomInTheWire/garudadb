use crate::storage_io::{read_json_file, write_json_file};
use garuda_types::Status;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

pub const DELETE_STORE_FILE_NAME: &str = "DELETE_STORE.json";

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct DeleteStoreFile {
    deleted_doc_ids: Vec<u64>,
}

pub fn write_delete_store(path: &Path, deleted_doc_ids: &HashSet<u64>) -> Result<(), Status> {
    let mut ids = deleted_doc_ids.iter().copied().collect::<Vec<_>>();
    ids.sort_unstable();

    let file = DeleteStoreFile {
        deleted_doc_ids: ids,
    };

    write_json_file(&path.join(DELETE_STORE_FILE_NAME), &file)
}

pub fn read_delete_store(path: &Path) -> Result<HashSet<u64>, Status> {
    let file: DeleteStoreFile = read_json_file(&path.join(DELETE_STORE_FILE_NAME))?;
    Ok(file.deleted_doc_ids.into_iter().collect())
}
