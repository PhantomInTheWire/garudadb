use crate::codec::{decode_flat_index, decode_hnsw_graph};
use crate::types::{RecordState, SegmentFile};
use garuda_index_flat::{FlatIndex, FlatIndexEntry};
use garuda_index_hnsw::{HnswBuildConfig, HnswBuildEntry, HnswIndex, HnswIndexConfig};
use garuda_storage::{read_file, segment_flat_index_path, segment_hnsw_index_path};
use garuda_types::{IndexParams, SegmentId, SegmentMeta, Status, StatusCode, VectorFieldSchema};

pub(crate) fn build_vector_search_state(
    vector_field: &VectorFieldSchema,
    meta: &SegmentMeta,
    records: &[crate::StoredRecord],
) -> (Option<FlatIndex>, Option<HnswIndex>) {
    match &vector_field.index {
        IndexParams::Flat(_) => build_flat_search_state(vector_field, meta, records),
        IndexParams::Hnsw(params) => build_hnsw_search_state(vector_field, meta, records, params),
    }
}

fn build_flat_search_state(
    vector_field: &VectorFieldSchema,
    meta: &SegmentMeta,
    records: &[crate::StoredRecord],
) -> (Option<FlatIndex>, Option<HnswIndex>) {
    let entries = flat_index_entries(records, meta.doc_count);
    let index = FlatIndex::build(vector_field.dimension, entries)
        .expect("validated segment records should match the vector field dimension");
    (Some(index), None)
}

fn build_hnsw_search_state(
    vector_field: &VectorFieldSchema,
    meta: &SegmentMeta,
    records: &[crate::StoredRecord],
    params: &garuda_types::HnswIndexParams,
) -> (Option<FlatIndex>, Option<HnswIndex>) {
    let config = hnsw_index_config(vector_field, params);
    let entries = hnsw_build_entries(&config, records, meta.doc_count);
    let index = HnswIndex::build(config, entries);
    (None, Some(index))
}

pub(crate) fn load_vector_search_state(
    root: &std::path::Path,
    segment_id: SegmentId,
    meta: &SegmentMeta,
    records: &[crate::StoredRecord],
    vector_field: &VectorFieldSchema,
) -> Result<(Option<FlatIndex>, Option<HnswIndex>), Status> {
    match &vector_field.index {
        IndexParams::Flat(_) => load_flat_search_state(root, segment_id, meta, vector_field),
        IndexParams::Hnsw(params) => {
            let bytes = read_file(&segment_hnsw_index_path(root, segment_id))?;
            let graph = decode_hnsw_graph(&bytes, vector_field, meta.doc_count)?;
            let config = hnsw_index_config(vector_field, params);
            let entries = hnsw_build_entries(&config, records, meta.doc_count);
            Ok((None, Some(HnswIndex::from_parts(config, entries, graph))))
        }
    }
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
) -> Result<(Option<FlatIndex>, Option<HnswIndex>), Status> {
    let bytes = read_file(&segment_flat_index_path(root, segment_id))?;
    let flat_index = decode_flat_index(&bytes, vector_field)?;

    if flat_index.len() != meta.doc_count {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted flat index does not match segment live doc count",
        ));
    }

    Ok((Some(flat_index), None))
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

    matches!(vector_field.index, IndexParams::Flat(_))
}

pub(crate) fn should_persist_hnsw(segment: &SegmentFile) -> bool {
    !segment.is_writing() && segment.meta.doc_count != 0
}
