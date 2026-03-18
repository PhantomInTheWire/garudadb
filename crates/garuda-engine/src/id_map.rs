use crate::storage_io::{read_json_file, write_json_file};
use garuda_types::{DocId, Status};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub const ID_MAP_FILE_NAME: &str = "IDMAP.json";

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct IdMapFile {
    entries: HashMap<DocId, u64>,
}

pub(crate) fn write_id_map(path: &Path, entries: &HashMap<DocId, u64>) -> Result<(), Status> {
    let file = IdMapFile {
        entries: entries.clone(),
    };

    write_json_file(&path.join(ID_MAP_FILE_NAME), &file)
}

pub(crate) fn read_id_map(path: &Path) -> Result<HashMap<DocId, u64>, Status> {
    let file: IdMapFile = read_json_file(&path.join(ID_MAP_FILE_NAME))?;
    Ok(file.entries)
}
