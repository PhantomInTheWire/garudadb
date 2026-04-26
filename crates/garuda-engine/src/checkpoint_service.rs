use crate::state::CollectionRuntime;
use garuda_segment::{reset_wal, write_persisted_segment, write_writing_segment};
use garuda_storage::{
    SnapshotKind, VersionManager, WRITING_SEGMENT_ID, delete_snapshot_path, id_map_snapshot_path,
    manifest_path, read_file, remove_old_snapshots, remove_path_if_exists, segment_data_path,
    segment_dir, segment_flat_index_path, segment_hnsw_index_path, segment_ivf_index_path,
    segment_scalar_index_dir, write_delete_snapshot, write_file_atomically, write_id_map_snapshot,
};
use garuda_types::{SegmentId, Status, StatusCode};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub(crate) fn checkpoint_state(state: &mut CollectionRuntime) -> Result<(), Status> {
    let staged = stage_checkpoint(state.clone())?;
    *state = publish_checkpoint(staged)?;

    Ok(())
}

pub(crate) struct StagedCheckpoint {
    state: CollectionRuntime,
    version_manager: VersionManager,
    rollback: CheckpointFiles,
}

pub(crate) fn stage_checkpoint(mut state: CollectionRuntime) -> Result<StagedCheckpoint, Status> {
    let version_manager = VersionManager::new(&state.path);
    let had_manifest = version_manager.exists()?;

    if had_manifest {
        state.id_map_snapshot_id = state.id_map_snapshot_id.next();
        state.delete_snapshot_id = state.delete_snapshot_id.next();
        state.manifest_version_id = state.manifest_version_id.next();
    }

    state
        .segments
        .seal_writing_segment(&mut state.next_segment_id, &state.schema);
    state.rebuild_indexes();

    let rollback = capture_checkpoint_files(&state)?;
    let persist_result = write_checkpoint_data_files(&state);

    if let Err(status) = persist_result {
        rollback.restore()?;
        return Err(status);
    }

    Ok(StagedCheckpoint {
        state,
        version_manager,
        rollback,
    })
}

pub(crate) fn publish_checkpoint(staged: StagedCheckpoint) -> Result<CollectionRuntime, Status> {
    let StagedCheckpoint {
        state,
        version_manager,
        rollback,
    } = staged;
    if let Err(status) = write_checkpoint_manifest(&state, &version_manager) {
        rollback.restore()?;
        return Err(status);
    }

    let _ = reset_wal(&state.path, WRITING_SEGMENT_ID);

    let _ = version_manager.remove_stale_manifests(state.manifest_version_id);
    let _ = remove_old_snapshots(&state.path, SnapshotKind::IdMap, state.id_map_snapshot_id);
    let _ = remove_old_snapshots(&state.path, SnapshotKind::Delete, state.delete_snapshot_id);
    let _ = remove_stale_segment_dirs(&state);

    Ok(state)
}

pub(crate) fn discard_staged_checkpoint(staged: StagedCheckpoint) -> Result<(), Status> {
    staged.rollback.restore()
}

fn write_checkpoint_manifest(
    state: &CollectionRuntime,
    version_manager: &VersionManager,
) -> Result<(), Status> {
    let manifest = garuda_types::Manifest {
        schema: state.schema.clone(),
        options: state.options.clone(),
        next_doc_id: state.next_doc_id,
        next_segment_id: state.next_segment_id,
        id_map_snapshot_id: state.id_map_snapshot_id,
        delete_snapshot_id: state.delete_snapshot_id,
        manifest_version_id: state.manifest_version_id,
        writing_segment: state.segments.writing_segment().meta.clone(),
        persisted_segments: state
            .segments
            .persisted_segments()
            .iter()
            .map(|segment| segment.meta.clone())
            .collect(),
    };

    version_manager.write_manifest(&manifest)
}

fn write_checkpoint_data_files(state: &CollectionRuntime) -> Result<(), Status> {
    write_all_segments(state)?;
    write_id_map_snapshot(
        &state.path,
        state.id_map_snapshot_id,
        state
            .meta
            .id_map_entries()
            .map(|(doc_id, internal_doc_id)| (doc_id.clone(), *internal_doc_id)),
    )?;
    write_delete_snapshot(
        &state.path,
        state.delete_snapshot_id,
        state.meta.deleted_doc_ids().copied(),
    )?;

    Ok(())
}

fn write_all_segments(state: &CollectionRuntime) -> Result<(), Status> {
    for segment in state.segments.persisted_segments() {
        write_persisted_segment(&state.path, segment, &state.schema)?;
    }

    write_writing_segment(&state.path, state.segments.writing_segment(), &state.schema)
}

fn remove_stale_segment_dirs(state: &CollectionRuntime) -> Result<(), Status> {
    let mut live_segment_ids =
        HashSet::with_capacity(state.segments.persisted_segments().len() + 1);
    live_segment_ids.insert(WRITING_SEGMENT_ID);

    for segment in state.segments.persisted_segments() {
        live_segment_ids.insert(segment.meta.id);
    }

    let entries = std::fs::read_dir(&state.path).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!(
                "failed to read collection directory {}: {error}",
                state.path.display()
            ),
        )
    })?;

    for entry in entries {
        let entry = entry.map_err(|error| {
            Status::err(
                StatusCode::Internal,
                format!("failed to read directory entry: {error}"),
            )
        })?;

        let Ok(file_type) = entry.file_type() else {
            continue;
        };

        if !file_type.is_dir() {
            continue;
        }

        let Some(file_name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };

        let Ok(segment_id) = file_name.parse::<u64>() else {
            continue;
        };
        let segment_id = SegmentId::new_unchecked(segment_id);

        if live_segment_ids.contains(&segment_id) {
            continue;
        }

        remove_path_if_exists(&segment_dir(&state.path, segment_id))?;
    }

    Ok(())
}

fn capture_checkpoint_files(state: &CollectionRuntime) -> Result<CheckpointFiles, Status> {
    let mut files = Vec::new();

    for segment in state.segments.persisted_segments() {
        files.push(capture_path(&segment_data_path(
            &state.path,
            segment.meta.id,
        ))?);
        files.push(capture_path(&segment_flat_index_path(
            &state.path,
            segment.meta.id,
        ))?);
        files.push(capture_path(&segment_hnsw_index_path(
            &state.path,
            segment.meta.id,
        ))?);
        files.push(capture_path(&segment_ivf_index_path(
            &state.path,
            segment.meta.id,
        ))?);
        files.push(capture_path(&segment_scalar_index_dir(
            &state.path,
            segment.meta.id,
        ))?);
    }

    files.push(capture_path(&segment_data_path(
        &state.path,
        WRITING_SEGMENT_ID,
    ))?);
    files.push(capture_path(&segment_flat_index_path(
        &state.path,
        WRITING_SEGMENT_ID,
    ))?);
    files.push(capture_path(&segment_hnsw_index_path(
        &state.path,
        WRITING_SEGMENT_ID,
    ))?);
    files.push(capture_path(&segment_ivf_index_path(
        &state.path,
        WRITING_SEGMENT_ID,
    ))?);
    files.push(capture_path(&segment_scalar_index_dir(
        &state.path,
        WRITING_SEGMENT_ID,
    ))?);
    files.push(capture_path(&id_map_snapshot_path(
        &state.path,
        state.id_map_snapshot_id,
    ))?);
    files.push(capture_path(&delete_snapshot_path(
        &state.path,
        state.delete_snapshot_id,
    ))?);
    files.push(capture_path(&manifest_path(
        &state.path,
        state.manifest_version_id,
    ))?);

    Ok(CheckpointFiles { files })
}

fn capture_path(path: &Path) -> Result<PathBackup, Status> {
    if !path.exists() {
        return Ok(PathBackup {
            path: path.to_path_buf(),
            state: PathState::Missing,
        });
    }

    if path.is_dir() {
        return Ok(PathBackup {
            path: path.to_path_buf(),
            state: PathState::Dir(capture_dir_entries(path, path)?),
        });
    }

    Ok(PathBackup {
        path: path.to_path_buf(),
        state: PathState::File(read_file(path)?),
    })
}

fn capture_dir_entries(root: &Path, path: &Path) -> Result<Vec<DirEntryBackup>, Status> {
    let mut entries = Vec::new();
    let read_dir = std::fs::read_dir(path).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to read directory {}: {error}", path.display()),
        )
    })?;

    for entry in read_dir {
        let entry = entry.map_err(|error| {
            Status::err(
                StatusCode::Internal,
                format!("failed to read directory entry: {error}"),
            )
        })?;
        let child_path = entry.path();

        if child_path.is_dir() {
            entries.extend(capture_dir_entries(root, &child_path)?);
            continue;
        }

        let relative_path = child_path
            .strip_prefix(root)
            .expect("directory child should strip prefix")
            .to_path_buf();
        entries.push(DirEntryBackup {
            relative_path,
            bytes: read_file(&child_path)?,
        });
    }

    Ok(entries)
}

struct CheckpointFiles {
    files: Vec<PathBackup>,
}

impl CheckpointFiles {
    fn restore(self) -> Result<(), Status> {
        for file in self.files {
            restore_path(file)?;
        }

        Ok(())
    }
}

struct PathBackup {
    path: PathBuf,
    state: PathState,
}

enum PathState {
    Missing,
    File(Vec<u8>),
    Dir(Vec<DirEntryBackup>),
}

struct DirEntryBackup {
    relative_path: PathBuf,
    bytes: Vec<u8>,
}

fn restore_path(path: PathBackup) -> Result<(), Status> {
    match path.state {
        PathState::Missing => remove_path_if_exists(&path.path),
        PathState::File(bytes) => restore_file(&path.path, &bytes),
        PathState::Dir(entries) => restore_dir(&path.path, entries),
    }
}

fn restore_file(path: &Path, bytes: &[u8]) -> Result<(), Status> {
    let parent = path.parent().ok_or_else(|| {
        Status::err(
            StatusCode::Internal,
            format!("cannot determine parent for {}", path.display()),
        )
    })?;

    std::fs::create_dir_all(parent).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to create directory {}: {error}", parent.display()),
        )
    })?;

    write_file_atomically(path, bytes)
}

fn restore_dir(path: &Path, entries: Vec<DirEntryBackup>) -> Result<(), Status> {
    remove_path_if_exists(path)?;
    std::fs::create_dir_all(path).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to create directory {}: {error}", path.display()),
        )
    })?;

    for entry in entries {
        let entry_path = path.join(entry.relative_path);
        restore_file(&entry_path, &entry.bytes)?;
    }

    Ok(())
}
