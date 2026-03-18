use crate::id_map::read_id_map;
use crate::state::CollectionState;
use crate::storage::{self, read_delete_store, read_segment, sync_segment_meta};
use crate::version::VersionStore;
use garuda_types::{CollectionOptions, CollectionSchema, Manifest, SegmentMeta, Status};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

pub(crate) fn create_collection_state(
    path: PathBuf,
    schema: CollectionSchema,
    options: CollectionOptions,
) -> CollectionState {
    let writing_segment = CollectionState::empty_writing_segment();
    let manifest = Manifest {
        schema,
        options,
        next_doc_id: 1,
        next_segment_id: 1,
        writing_segment: writing_segment.meta.clone(),
        persisted_segments: Vec::new(),
    };

    CollectionState {
        path,
        manifest,
        persisted_segments: Vec::new(),
        writing_segment,
        id_map: HashMap::new(),
        deleted_doc_ids: HashSet::new(),
    }
}

pub(crate) fn load_collection_state(path: PathBuf) -> Result<CollectionState, Status> {
    let manifest = VersionStore::new(&path).read_manifest()?;
    let writing_segment = load_segment(&path, &manifest.writing_segment)?;
    let persisted_segments = load_persisted_segments(&path, &manifest)?;
    let id_map = read_id_map(&path).unwrap_or_default();
    let deleted_doc_ids = read_delete_store(&path).unwrap_or_default();

    let mut state = CollectionState {
        path,
        manifest,
        persisted_segments,
        writing_segment,
        id_map,
        deleted_doc_ids,
    };
    state.rebuild_indexes();

    Ok(state)
}

fn load_persisted_segments(
    path: &Path,
    manifest: &Manifest,
) -> Result<Vec<storage::SegmentFile>, Status> {
    let mut segments = Vec::new();

    for meta in &manifest.persisted_segments {
        segments.push(load_segment(path, meta)?);
    }

    Ok(segments)
}

fn load_segment(path: &Path, meta: &SegmentMeta) -> Result<storage::SegmentFile, Status> {
    let mut segment = read_segment(path, meta)?;
    sync_segment_meta(&mut segment);
    Ok(segment)
}
