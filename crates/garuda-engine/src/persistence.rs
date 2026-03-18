use crate::state::CollectionState;
use garuda_segment::{reset_wal, write_segment};
use garuda_storage::{
    SnapshotKind, VersionManager, WRITING_SEGMENT_ID, remove_old_snapshots, remove_path_if_exists,
    segment_dir, write_delete_snapshot, write_id_map_snapshot,
};
use garuda_types::Status;
use std::collections::HashSet;

pub(crate) fn checkpoint_state(state: &mut CollectionState) -> Result<(), Status> {
    let version_manager = VersionManager::new(&state.path);
    let had_manifest = version_manager.exists();

    if had_manifest {
        state.manifest.id_map_snapshot_id = state.manifest.id_map_snapshot_id.next();
        state.manifest.delete_snapshot_id = state.manifest.delete_snapshot_id.next();
        state.manifest.manifest_version_id = state.manifest.manifest_version_id.next();
    }

    state.refresh_manifest();

    write_all_segments(state)?;
    write_id_map_snapshot(
        &state.path,
        state.manifest.id_map_snapshot_id,
        &state.id_map,
    )?;
    write_delete_snapshot(
        &state.path,
        state.manifest.delete_snapshot_id,
        &state.deleted_doc_ids,
    )?;
    version_manager.write_manifest(&state.manifest)?;

    let _ = reset_wal(&state.path, WRITING_SEGMENT_ID);
    let _ = remove_old_snapshots(
        &state.path,
        SnapshotKind::IdMap,
        state.manifest.id_map_snapshot_id,
    );
    let _ = remove_old_snapshots(
        &state.path,
        SnapshotKind::Delete,
        state.manifest.delete_snapshot_id,
    );

    let _ = remove_stale_segment_dirs(state);

    Ok(())
}

fn write_all_segments(state: &CollectionState) -> Result<(), Status> {
    for segment in &state.persisted_segments {
        write_segment(&state.path, segment)?;
    }

    write_segment(&state.path, &state.writing_segment)
}

fn remove_stale_segment_dirs(state: &CollectionState) -> Result<(), Status> {
    let mut live_segment_ids = HashSet::new();
    live_segment_ids.insert(WRITING_SEGMENT_ID);

    for segment in &state.persisted_segments {
        live_segment_ids.insert(segment.meta.id);
    }

    let entries = std::fs::read_dir(&state.path).map_err(|error| {
        garuda_types::Status::err(
            garuda_types::StatusCode::Internal,
            format!(
                "failed to read collection directory {}: {error}",
                state.path.display()
            ),
        )
    })?;

    for entry in entries {
        let entry = entry.map_err(|error| {
            garuda_types::Status::err(
                garuda_types::StatusCode::Internal,
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
