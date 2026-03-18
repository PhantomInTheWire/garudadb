use crate::catalog::CollectionCatalog;
use crate::segment_manager::SegmentManager;
use crate::state::CollectionRuntime;
use crate::write_service::replay_wal_ops;
use garuda_meta::{DeleteStore, IdMap, MetadataStore};
use garuda_segment::{SegmentFile, read_segment, read_wal_ops, sync_segment_meta};
use garuda_storage::{
    VersionManager, WRITING_SEGMENT_ID, read_delete_snapshot, read_id_map_snapshot,
};
use garuda_types::{
    CollectionOptions, CollectionSchema, Manifest, ManifestVersionId, SegmentMeta, SnapshotId,
    Status,
};
use std::path::{Path, PathBuf};

const INITIAL_DOC_ID: u64 = 1;
const INITIAL_SEGMENT_ID: u64 = 1;
const INITIAL_SNAPSHOT_ID: u64 = 0;
const INITIAL_MANIFEST_VERSION_ID: u64 = 0;

pub(crate) fn create_collection_state(
    path: PathBuf,
    schema: CollectionSchema,
    options: CollectionOptions,
) -> CollectionRuntime {
    let writing_segment = SegmentManager::empty_writing_segment();
    let manifest = Manifest {
        schema,
        options,
        next_doc_id: INITIAL_DOC_ID,
        next_segment_id: INITIAL_SEGMENT_ID,
        id_map_snapshot_id: SnapshotId::new(INITIAL_SNAPSHOT_ID),
        delete_snapshot_id: SnapshotId::new(INITIAL_SNAPSHOT_ID),
        manifest_version_id: ManifestVersionId::new(INITIAL_MANIFEST_VERSION_ID),
        writing_segment: writing_segment.meta.clone(),
        persisted_segments: Vec::new(),
    };

    CollectionRuntime {
        path,
        catalog: CollectionCatalog::from_manifest(manifest),
        segments: SegmentManager::new(Vec::new(), writing_segment),
        meta: MetadataStore::new(),
    }
}

pub(crate) fn load_collection_state(path: PathBuf) -> Result<CollectionRuntime, Status> {
    let manifest = VersionManager::new(&path).read_latest_manifest()?;
    let writing_segment = load_segment(&path, &manifest.writing_segment)?;
    let persisted_segments = load_persisted_segments(&path, &manifest)?;
    let id_map = IdMap::from(read_id_map_snapshot(&path, manifest.id_map_snapshot_id)?);
    let delete_store = DeleteStore::from(read_delete_snapshot(&path, manifest.delete_snapshot_id)?);

    let mut state = CollectionRuntime {
        path,
        catalog: CollectionCatalog::from_manifest(manifest),
        segments: SegmentManager::new(persisted_segments, writing_segment),
        meta: MetadataStore::from_parts(id_map, delete_store),
    };

    let wal_ops = read_wal_ops(&state.path, WRITING_SEGMENT_ID)?;
    replay_wal_ops(&mut state, wal_ops)?;

    Ok(state)
}

fn load_persisted_segments(path: &Path, manifest: &Manifest) -> Result<Vec<SegmentFile>, Status> {
    let mut segments = Vec::new();

    for meta in &manifest.persisted_segments {
        segments.push(load_segment(path, meta)?);
    }

    Ok(segments)
}

fn load_segment(path: &Path, meta: &SegmentMeta) -> Result<SegmentFile, Status> {
    let mut segment = read_segment(path, meta)?;
    sync_segment_meta(&mut segment);
    Ok(segment)
}
