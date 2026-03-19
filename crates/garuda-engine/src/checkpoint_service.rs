use crate::state::CollectionRuntime;
use garuda_segment::{reset_wal, write_segment};
use garuda_storage::{
    SnapshotKind, VersionManager, WRITING_SEGMENT_ID, delete_snapshot_path, id_map_snapshot_path,
    manifest_path, read_file, remove_old_snapshots, remove_path_if_exists, segment_data_path,
    segment_dir, segment_wal_path, write_delete_snapshot, write_file_atomically,
    write_id_map_snapshot,
};
use garuda_types::{Status, StatusCode};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const FIXED_CHECKPOINT_FILE_COUNT: usize = 5;

pub(crate) fn checkpoint_state(state: &mut CollectionRuntime) -> Result<(), Status> {
    let version_manager = VersionManager::new(&state.path);
    let had_manifest = version_manager.exists();

    if had_manifest {
        state.catalog.id_map_snapshot_id = state.catalog.id_map_snapshot_id.next();
        state.catalog.delete_snapshot_id = state.catalog.delete_snapshot_id.next();
        state.catalog.manifest_version_id = state.catalog.manifest_version_id.next();
    }

    let rollback = capture_checkpoint_files(state)?;
    let persist_result = write_checkpoint_files(state, &version_manager);

    if let Err(status) = persist_result {
        rollback.restore()?;
        return Err(status);
    }

    let _ = remove_old_snapshots(
        &state.path,
        SnapshotKind::IdMap,
        state.catalog.id_map_snapshot_id,
    );
    let _ = remove_old_snapshots(
        &state.path,
        SnapshotKind::Delete,
        state.catalog.delete_snapshot_id,
    );

    let _ = remove_stale_segment_dirs(state);

    Ok(())
}

fn write_checkpoint_files(
    state: &CollectionRuntime,
    version_manager: &VersionManager,
) -> Result<(), Status> {
    let manifest = state.catalog.to_manifest(
        state.segments.writing_segment(),
        state.segments.persisted_segments(),
    );
    write_all_segments(state)?;
    write_id_map_snapshot(
        &state.path,
        state.catalog.id_map_snapshot_id,
        state
            .meta
            .id_map_entries()
            .map(|(doc_id, internal_doc_id)| (doc_id.as_str().to_string(), *internal_doc_id)),
    )?;
    write_delete_snapshot(
        &state.path,
        state.catalog.delete_snapshot_id,
        state.meta.deleted_doc_ids().copied(),
    )?;
    version_manager.write_manifest(&manifest)?;

    if reset_wal(&state.path, WRITING_SEGMENT_ID).is_err() {
        return Ok(());
    }

    Ok(())
}

fn write_all_segments(state: &CollectionRuntime) -> Result<(), Status> {
    for segment in state.segments.persisted_segments() {
        write_segment(&state.path, segment)?;
    }

    write_segment(&state.path, state.segments.writing_segment())
}

fn remove_stale_segment_dirs(state: &CollectionRuntime) -> Result<(), Status> {
    let mut live_segment_ids = HashSet::with_capacity(state.segments.persisted_segments().len() + 1);
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

        if live_segment_ids.contains(&segment_id) {
            continue;
        }

        remove_path_if_exists(&segment_dir(&state.path, segment_id))?;
    }

    Ok(())
}

fn capture_checkpoint_files(state: &CollectionRuntime) -> Result<CheckpointFiles, Status> {
    // Each rollback snapshot always captures the writing segment, id-map snapshot,
    // delete snapshot, writing WAL, and manifest, in addition to every persisted segment.
    let mut files = Vec::with_capacity(
        state.segments.persisted_segments().len() + FIXED_CHECKPOINT_FILE_COUNT,
    );

    for segment in state.segments.persisted_segments() {
        files.push(capture_file(&segment_data_path(
            &state.path,
            segment.meta.id,
        ))?);
    }

    files.push(capture_file(&segment_data_path(
        &state.path,
        WRITING_SEGMENT_ID,
    ))?);
    files.push(capture_file(&id_map_snapshot_path(
        &state.path,
        state.catalog.id_map_snapshot_id,
    ))?);
    files.push(capture_file(&delete_snapshot_path(
        &state.path,
        state.catalog.delete_snapshot_id,
    ))?);
    files.push(capture_file(&segment_wal_path(
        &state.path,
        WRITING_SEGMENT_ID,
    ))?);
    files.push(capture_file(&manifest_path(
        &state.path,
        state.catalog.manifest_version_id,
    ))?);

    Ok(CheckpointFiles { files })
}

fn capture_file(path: &Path) -> Result<FileBackup, Status> {
    let original_bytes = if path.exists() {
        Some(read_file(path)?)
    } else {
        None
    };

    Ok(FileBackup {
        path: path.to_path_buf(),
        original_bytes,
    })
}

struct CheckpointFiles {
    files: Vec<FileBackup>,
}

impl CheckpointFiles {
    fn restore(self) -> Result<(), Status> {
        for file in self.files {
            restore_file(file)?;
        }

        Ok(())
    }
}

struct FileBackup {
    path: PathBuf,
    original_bytes: Option<Vec<u8>>,
}

fn restore_file(file: FileBackup) -> Result<(), Status> {
    let Some(original_bytes) = file.original_bytes else {
        remove_path_if_exists(&file.path)?;
        return Ok(());
    };

    let parent = file.path.parent().ok_or_else(|| {
        Status::err(
            StatusCode::Internal,
            format!("cannot determine parent for {}", file.path.display()),
        )
    })?;

    std::fs::create_dir_all(parent).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!("failed to create directory {}: {error}", parent.display()),
        )
    })?;

    write_file_atomically(&file.path, &original_bytes)
}
