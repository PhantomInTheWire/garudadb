use crate::codec::{decode_delete_snapshot, decode_id_map, encode_delete_snapshot, encode_id_map};
use crate::io::{read_file, remove_file, write_file_atomically};
use crate::layout::{
    DELETE_FILE_PREFIX, ID_MAP_FILE_PREFIX, delete_snapshot_path, id_map_snapshot_path,
};
use garuda_types::{DocId, SnapshotId, Status, StatusCode};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotKind {
    IdMap,
    Delete,
}

pub fn write_id_map_snapshot(
    root: &Path,
    snapshot_id: SnapshotId,
    entries: &HashMap<DocId, u64>,
) -> Result<(), Status> {
    let mut ordered_entries = entries
        .iter()
        .map(|(doc_id, internal_doc_id)| (doc_id.as_str().to_string(), *internal_doc_id))
        .collect::<Vec<_>>();
    ordered_entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

    let bytes = encode_id_map(&ordered_entries)?;
    write_file_atomically(&id_map_snapshot_path(root, snapshot_id), &bytes)
}

pub fn read_id_map_snapshot(
    root: &Path,
    snapshot_id: SnapshotId,
) -> Result<HashMap<DocId, u64>, Status> {
    let bytes = read_file(&id_map_snapshot_path(root, snapshot_id))?;
    let entries = decode_id_map(&bytes)?;
    let mut id_map = HashMap::new();

    for (doc_id, internal_doc_id) in entries {
        id_map.insert(DocId::parse(doc_id)?, internal_doc_id);
    }

    Ok(id_map)
}

pub fn write_delete_snapshot(
    root: &Path,
    snapshot_id: SnapshotId,
    deleted_doc_ids: &HashSet<u64>,
) -> Result<(), Status> {
    let mut ids = deleted_doc_ids.iter().copied().collect::<Vec<_>>();
    ids.sort_unstable();

    let bytes = encode_delete_snapshot(&ids)?;
    write_file_atomically(&delete_snapshot_path(root, snapshot_id), &bytes)
}

pub fn read_delete_snapshot(root: &Path, snapshot_id: SnapshotId) -> Result<HashSet<u64>, Status> {
    let bytes = read_file(&delete_snapshot_path(root, snapshot_id))?;
    let ids = decode_delete_snapshot(&bytes)?;
    Ok(ids.into_iter().collect())
}

pub fn remove_old_snapshots(
    root: &Path,
    kind: SnapshotKind,
    keep: SnapshotId,
) -> Result<(), Status> {
    let entries = std::fs::read_dir(root).map_err(|error| {
        Status::err(
            StatusCode::Internal,
            format!(
                "failed to read collection directory {}: {error}",
                root.display()
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
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };

        let prefix = snapshot_prefix(kind);
        if !file_name.starts_with(prefix) {
            continue;
        }

        let suffix = &file_name[prefix.len()..];
        let Ok(snapshot_id) = suffix.parse::<u64>() else {
            continue;
        };

        if snapshot_id == keep.get() {
            continue;
        }

        remove_file(&entry.path())?;
    }

    Ok(())
}

fn snapshot_prefix(kind: SnapshotKind) -> &'static str {
    match kind {
        SnapshotKind::IdMap => ID_MAP_FILE_PREFIX,
        SnapshotKind::Delete => DELETE_FILE_PREFIX,
    }
}
