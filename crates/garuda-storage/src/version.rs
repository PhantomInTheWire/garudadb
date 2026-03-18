use crate::codec::{decode_manifest, encode_manifest};
use crate::io::{read_file, remove_file, write_file_atomically};
use crate::layout::{manifest_path, manifest_paths};
use garuda_types::{Manifest, ManifestVersionId, Status, StatusCode};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct VersionManager {
    collection_path: PathBuf,
}

impl VersionManager {
    pub fn new(collection_path: impl AsRef<Path>) -> Self {
        Self {
            collection_path: collection_path.as_ref().to_path_buf(),
        }
    }

    pub fn exists(&self) -> bool {
        manifest_paths(&self.collection_path)
            .map(|paths| !paths.is_empty())
            .unwrap_or(false)
    }

    pub fn read_latest_manifest(&self) -> Result<Manifest, Status> {
        let (_, path) = latest_manifest_path(&self.collection_path)?;
        let bytes = read_file(&path)?;
        decode_manifest(&bytes)
    }

    pub fn write_manifest(&self, manifest: &Manifest) -> Result<(), Status> {
        let path = manifest_path(&self.collection_path, manifest.manifest_version_id);
        let bytes = encode_manifest(manifest)?;
        write_file_atomically(&path, &bytes)?;
        let _ = self.remove_stale_manifests(manifest.manifest_version_id);
        Ok(())
    }

    pub fn next_manifest_version(&self) -> Result<ManifestVersionId, Status> {
        if !self.exists() {
            return Ok(ManifestVersionId::new(0));
        }

        let (current, _) = latest_manifest_path(&self.collection_path)?;
        Ok(current.next())
    }

    fn remove_stale_manifests(&self, keep: ManifestVersionId) -> Result<(), Status> {
        for (version_id, path) in manifest_paths(&self.collection_path)? {
            if version_id == keep {
                continue;
            }

            remove_file(&path)?;
        }

        Ok(())
    }
}

fn latest_manifest_path(root: &Path) -> Result<(ManifestVersionId, PathBuf), Status> {
    manifest_paths(root)?
        .into_iter()
        .last()
        .ok_or_else(|| Status::err(StatusCode::NotFound, "manifest not found"))
}
