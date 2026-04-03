use crate::state::IvfEntryIndex;
use crate::{IvfBuildEntry, IvfIndexConfig, IvfStoredLists};
use garuda_types::{InternalDocId, Status, StatusCode};
use std::collections::HashMap;

pub(super) fn build_list_entry_indexes(
    entries: &[IvfBuildEntry],
    doc_ids_by_list: &[Vec<InternalDocId>],
) -> Result<Vec<Vec<IvfEntryIndex>>, Status> {
    let mut entry_index_by_doc_id = HashMap::with_capacity(entries.len());

    for (entry_index, entry) in entries.iter().enumerate() {
        entry_index_by_doc_id.insert(entry.doc_id, entry_index);
    }

    let mut list_entry_indexes = Vec::with_capacity(doc_ids_by_list.len());

    for doc_ids in doc_ids_by_list {
        let mut indexes = Vec::with_capacity(doc_ids.len());

        for &doc_id in doc_ids {
            let Some(entry_index) = entry_index_by_doc_id.remove(&doc_id) else {
                return Err(Status::err(
                    StatusCode::Internal,
                    "persisted ivf lists do not match live entries",
                ));
            };

            indexes.push(IvfEntryIndex::new(entry_index));
        }

        list_entry_indexes.push(indexes);
    }

    if entry_index_by_doc_id.is_empty() {
        return Ok(list_entry_indexes);
    }

    Err(Status::err(
        StatusCode::Internal,
        "persisted ivf lists do not match live entries",
    ))
}

pub(super) fn validate_stored_lists(
    config: &IvfIndexConfig,
    entries: &[IvfBuildEntry],
    stored: &IvfStoredLists,
) -> Result<(), Status> {
    let expected_list_count = config.list_count(entries.len());
    if stored.centroids.len() != expected_list_count {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted ivf centroid count does not match live entries",
        ));
    }

    if stored.doc_ids_by_list.len() != expected_list_count {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted ivf list count does not match live entries",
        ));
    }

    for centroid in stored.centroids.iter() {
        if centroid.len() == config.dimension.get() {
            continue;
        }

        return Err(Status::err(
            StatusCode::Internal,
            "persisted ivf centroid dimension does not match index dimension",
        ));
    }

    Ok(())
}
