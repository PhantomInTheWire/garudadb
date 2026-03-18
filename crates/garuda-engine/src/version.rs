use crate::storage_io::{read_json_file, write_json_file};
use garuda_types::Manifest;
use std::path::{Path, PathBuf};

use garuda_types::Status;

pub const VERSION_FILE_NAME: &str = "VERSION.json";

#[derive(Clone, Debug)]
pub(crate) struct VersionStore {
    collection_path: PathBuf,
}

impl VersionStore {
    pub(crate) fn new(collection_path: impl AsRef<Path>) -> Self {
        Self {
            collection_path: collection_path.as_ref().to_path_buf(),
        }
    }

    pub(crate) fn exists(&self) -> bool {
        self.current_path().exists()
    }

    pub(crate) fn read_manifest(&self) -> Result<Manifest, Status> {
        read_json_file(&self.current_path())
    }

    pub(crate) fn write_manifest(&self, manifest: &Manifest) -> Result<(), Status> {
        write_json_file(&self.current_path(), manifest)
    }

    fn current_path(&self) -> PathBuf {
        self.collection_path.join(VERSION_FILE_NAME)
    }
}
