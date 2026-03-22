use crate::codec::{decode_flat_index, decode_hnsw_graph};
use crate::types::{RecordState, SegmentFile};
use garuda_index_flat::{FlatIndex, FlatIndexEntry};
use garuda_index_hnsw::{HnswBuildConfig, HnswBuildEntry, HnswIndex, HnswIndexConfig};
use garuda_storage::{read_file, segment_flat_index_path, segment_hnsw_index_path};
use garuda_types::{SegmentId, SegmentMeta, Status, StatusCode, VectorFieldSchema};

pub(crate) fn build_vector_search_state(
    vector_field: &VectorFieldSchema,
    meta: &SegmentMeta,
    records: &[crate::StoredRecord],
) -> (Option<FlatIndex>, Option<HnswIndex>) {
    let flat_index = build_flat_search_state(vector_field, meta, records);
    let hnsw_index = build_hnsw_search_state(vector_field, meta, records);
    (flat_index, hnsw_index)
}

fn build_flat_search_state(
    vector_field: &VectorFieldSchema,
    meta: &SegmentMeta,
    records: &[crate::StoredRecord],
) -> Option<FlatIndex> {
    if !vector_field.indexes.has_flat() {
        return None;
    }

    let entries = flat_index_entries(records, meta.doc_count);
    let index = FlatIndex::build(vector_field.dimension, entries)
        .expect("validated segment records should match the vector field dimension");
    Some(index)
}

fn build_hnsw_search_state(
    vector_field: &VectorFieldSchema,
    meta: &SegmentMeta,
    records: &[crate::StoredRecord],
) -> Option<HnswIndex> {
    let Some(params) = vector_field.indexes.hnsw_params() else {
        return None;
    };

    let config = hnsw_index_config(vector_field, params);
    let entries = hnsw_build_entries(&config, records, meta.doc_count);
    Some(HnswIndex::build(config, entries))
}

pub(crate) fn load_vector_search_state(
    root: &std::path::Path,
    segment_id: SegmentId,
    meta: &SegmentMeta,
    records: &[crate::StoredRecord],
    vector_field: &VectorFieldSchema,
) -> Result<(Option<FlatIndex>, Option<HnswIndex>), Status> {
    let flat_index = load_flat_search_state(root, segment_id, meta, vector_field)?;
    let hnsw_index = load_hnsw_search_state(root, segment_id, meta, records, vector_field)?;
    Ok((flat_index, hnsw_index))
}

pub(crate) fn hnsw_index_config(
    vector_field: &VectorFieldSchema,
    params: &garuda_types::HnswIndexParams,
) -> HnswIndexConfig {
    HnswIndexConfig::new(
        vector_field.dimension,
        vector_field.metric,
        HnswBuildConfig::new(
            params.neighbor_config().expect("validated hnsw params"),
            params.scaling_factor,
            params.ef_construction,
            params.prune_width,
        ),
    )
}

fn load_flat_search_state(
    root: &std::path::Path,
    segment_id: SegmentId,
    meta: &SegmentMeta,
    vector_field: &VectorFieldSchema,
) -> Result<Option<FlatIndex>, Status> {
    if !vector_field.indexes.has_flat() {
        return Ok(None);
    }

    let bytes = read_file(&segment_flat_index_path(root, segment_id))?;
    let flat_index = decode_flat_index(&bytes, vector_field)?;

    if flat_index.len() != meta.doc_count {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted flat index does not match segment live doc count",
        ));
    }

    Ok(Some(flat_index))
}

fn load_hnsw_search_state(
    root: &std::path::Path,
    segment_id: SegmentId,
    meta: &SegmentMeta,
    records: &[crate::StoredRecord],
    vector_field: &VectorFieldSchema,
) -> Result<Option<HnswIndex>, Status> {
    let Some(params) = vector_field.indexes.hnsw_params() else {
        return Ok(None);
    };

    let bytes = read_file(&segment_hnsw_index_path(root, segment_id))?;
    let graph = decode_hnsw_graph(&bytes, vector_field, meta.doc_count)?;
    let config = hnsw_index_config(vector_field, params);
    let entries = hnsw_build_entries(&config, records, meta.doc_count);

    Ok(Some(HnswIndex::from_parts(config, entries, graph)))
}

pub(crate) fn flat_index_entries(
    records: &[crate::StoredRecord],
    live_doc_count: usize,
) -> Vec<FlatIndexEntry> {
    let mut entries = Vec::with_capacity(live_doc_count);

    for record in records {
        if matches!(record.state, RecordState::Deleted) {
            continue;
        }

        entries.push(FlatIndexEntry::new(
            record.doc_id,
            record.doc.vector.clone(),
        ));
    }

    entries
}

pub(crate) fn hnsw_build_entries(
    config: &HnswIndexConfig,
    records: &[crate::StoredRecord],
    live_doc_count: usize,
) -> Vec<HnswBuildEntry> {
    let mut entries = Vec::with_capacity(live_doc_count);

    for record in records {
        if matches!(record.state, RecordState::Deleted) {
            continue;
        }

        entries.push(
            HnswBuildEntry::new(config, record.doc_id, record.doc.vector.clone())
                .expect("validated segment records should match the vector field dimension"),
        );
    }

    entries
}

pub(crate) fn should_use_persisted_flat(
    segment: &SegmentFile,
    vector_field: &VectorFieldSchema,
) -> bool {
    if segment.is_writing() || segment.meta.doc_count == 0 {
        return false;
    }

    vector_field.indexes.has_flat()
}

pub(crate) fn should_persist_hnsw(segment: &SegmentFile, vector_field: &VectorFieldSchema) -> bool {
    !segment.is_writing() && segment.meta.doc_count != 0 && vector_field.indexes.has_hnsw()
}
