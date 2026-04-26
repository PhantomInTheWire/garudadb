use crate::schema::validate_schema;
use crate::segment_manager::SegmentManager;
use crate::state::CollectionRuntime;
use crate::write_service::replay_wal_ops;
use garuda_meta::{DeleteStore, IdMap, MetadataStore};
use garuda_segment::{
    PersistedSegment, read_persisted_segment, read_wal_ops, read_writing_segment,
};
use garuda_storage::{
    VersionManager, WRITING_SEGMENT_ID, read_delete_snapshot, read_id_map_snapshot,
};
use garuda_types::{
    AccessMode, CollectionOptions, CollectionSchema, InternalDocId, Manifest, ManifestVersionId,
    SegmentId, SnapshotId, Status, StatusCode,
};
use std::path::{Path, PathBuf};

const INITIAL_DOC_ID: InternalDocId = InternalDocId::new_unchecked(1);
const INITIAL_SEGMENT_ID: SegmentId = SegmentId::new_unchecked(1);
const INITIAL_SNAPSHOT_ID: u64 = 0;
const INITIAL_MANIFEST_VERSION_ID: u64 = 0;

pub(crate) fn create_collection_state(
    path: PathBuf,
    schema: CollectionSchema,
    options: CollectionOptions,
) -> CollectionRuntime {
    let writing_segment = SegmentManager::empty_writing_segment(&schema);

    CollectionRuntime {
        path,
        schema,
        options,
        next_doc_id: INITIAL_DOC_ID,
        next_segment_id: INITIAL_SEGMENT_ID,
        id_map_snapshot_id: SnapshotId::new(INITIAL_SNAPSHOT_ID),
        delete_snapshot_id: SnapshotId::new(INITIAL_SNAPSHOT_ID),
        manifest_version_id: ManifestVersionId::new(INITIAL_MANIFEST_VERSION_ID),
        revision: 0,
        segments: SegmentManager::new(Vec::new(), writing_segment),
        meta: MetadataStore::new(),
    }
}

pub(crate) fn load_collection_state(path: PathBuf) -> Result<CollectionRuntime, Status> {
    let manifest = VersionManager::new(&path).read_latest_manifest()?;
    validate_schema(&manifest.schema)?;
    let writing_segment = read_writing_segment(&path, &manifest.writing_segment, &manifest.schema)?;
    let persisted_segments = load_persisted_segments(&path, &manifest)?;
    let id_map = IdMap::from(read_id_map_snapshot(&path, manifest.id_map_snapshot_id)?);
    let delete_store = DeleteStore::from(read_delete_snapshot(&path, manifest.delete_snapshot_id)?);

    let mut state = CollectionRuntime {
        path,
        schema: manifest.schema,
        options: manifest.options,
        next_doc_id: manifest.next_doc_id,
        next_segment_id: manifest.next_segment_id,
        id_map_snapshot_id: manifest.id_map_snapshot_id,
        delete_snapshot_id: manifest.delete_snapshot_id,
        manifest_version_id: manifest.manifest_version_id,
        revision: 0,
        segments: SegmentManager::new(persisted_segments, writing_segment),
        meta: MetadataStore::from_parts(id_map, delete_store),
    };

    let wal_ops = read_wal_ops(&state.path, WRITING_SEGMENT_ID)?;
    if matches!(state.options.access_mode, AccessMode::ReadOnly) && !wal_ops.is_empty() {
        return Err(Status::err(
            StatusCode::FailedPrecondition,
            "read-only collection cannot reopen with pending WAL operations",
        ));
    }
    replay_wal_ops(&mut state, wal_ops)?;

    Ok(state)
}

fn load_persisted_segments(
    path: &Path,
    manifest: &Manifest,
) -> Result<Vec<PersistedSegment>, Status> {
    let mut segments = Vec::new();

    for meta in &manifest.persisted_segments {
        segments.push(read_persisted_segment(path, meta, &manifest.schema)?);
    }

    Ok(segments)
}
