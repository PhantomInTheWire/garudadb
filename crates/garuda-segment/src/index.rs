use crate::codec::{decode_flat_index, decode_hnsw_graph};
use crate::segment_file_name;
use crate::types::{PersistedSegment, RecordState, StoredRecord, WritingSegment};
use garuda_index_flat::{FlatIndex, FlatIndexEntry, WritingFlatIndex};
use garuda_index_hnsw::{
    HnswBuildConfig, HnswBuildEntry, HnswIndex, HnswIndexConfig, WritingHnswIndex,
};
use garuda_storage::{read_file, segment_flat_index_path, segment_hnsw_index_path};
use garuda_types::{SegmentId, SegmentMeta, Status, StatusCode, VectorFieldSchema};

pub(crate) fn build_writing_search_resources(
    vector_field: &VectorFieldSchema,
    records: &[StoredRecord],
) -> (Option<WritingFlatIndex>, Option<WritingHnswIndex>) {
    let mut flat_index = if vector_field.indexes.has_flat() {
        Some(WritingFlatIndex::new(vector_field.dimension))
    } else {
        None
    };
    let mut hnsw_index = vector_field
        .indexes
        .hnsw_params()
        .map(|params| WritingHnswIndex::new(hnsw_index_config(vector_field, params)));

    for record in records {
        if matches!(record.state, RecordState::Deleted) {
            continue;
        }

        if let Some(index) = &mut flat_index {
            index.insert(record.doc_id, record.doc.vector.clone());
        }

        if let Some(index) = &mut hnsw_index {
            index.insert(record.doc_id, record.doc.vector.clone());
        }
    }

    (flat_index, hnsw_index)
}

pub(crate) fn build_persisted_search_resources(
    vector_field: &VectorFieldSchema,
    meta: &SegmentMeta,
    records: &[StoredRecord],
) -> (Option<FlatIndex>, Option<HnswIndex>) {
    let flat_index = build_flat_index(vector_field, meta, records);
    let hnsw_index = build_hnsw_index(vector_field, meta, records);
    (flat_index, hnsw_index)
}

pub(crate) fn load_persisted_search_resources(
    root: &std::path::Path,
    segment_id: SegmentId,
    meta: &SegmentMeta,
    records: &[StoredRecord],
    vector_field: &VectorFieldSchema,
) -> Result<(Option<FlatIndex>, Option<HnswIndex>), Status> {
    let flat_index = load_flat_index(root, segment_id, meta, vector_field)?;
    let hnsw_index = load_hnsw_index(root, segment_id, meta, records, vector_field)?;
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

fn build_flat_index(
    vector_field: &VectorFieldSchema,
    meta: &SegmentMeta,
    records: &[StoredRecord],
) -> Option<FlatIndex> {
    if !vector_field.indexes.has_flat() {
        return None;
    }

    let entries = flat_index_entries(records, meta.doc_count);
    let index = FlatIndex::build(vector_field.dimension, entries)
        .expect("validated segment records should match the vector field dimension");
    Some(index)
}

fn build_hnsw_index(
    vector_field: &VectorFieldSchema,
    meta: &SegmentMeta,
    records: &[StoredRecord],
) -> Option<HnswIndex> {
    let Some(params) = vector_field.indexes.hnsw_params() else {
        return None;
    };

    let config = hnsw_index_config(vector_field, params);
    let entries = hnsw_build_entries(&config, records, meta.doc_count);
    Some(HnswIndex::build(config, entries))
}

fn load_flat_index(
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

fn load_hnsw_index(
    root: &std::path::Path,
    segment_id: SegmentId,
    meta: &SegmentMeta,
    records: &[StoredRecord],
    vector_field: &VectorFieldSchema,
) -> Result<Option<HnswIndex>, Status> {
    let Some(params) = vector_field.indexes.hnsw_params() else {
        return Ok(None);
    };

    let bytes = read_file(&segment_hnsw_index_path(root, segment_id))?;
    let graph = decode_hnsw_graph(&bytes, vector_field, meta.doc_count)?;
    let config = hnsw_index_config(vector_field, params);
    let entries = hnsw_build_entries(&config, records, meta.doc_count);

    if graph.node_count() != entries.len() {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted hnsw graph does not match rebuilt live entries",
        ));
    }

    Ok(Some(HnswIndex::from_parts(config, entries, graph)))
}

pub(crate) fn flat_index_entries(
    records: &[StoredRecord],
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
    records: &[StoredRecord],
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

pub(crate) fn should_persist_flat(vector_field: &VectorFieldSchema, live_doc_count: usize) -> bool {
    vector_field.indexes.has_flat() && live_doc_count != 0
}

pub(crate) fn should_persist_hnsw(vector_field: &VectorFieldSchema, live_doc_count: usize) -> bool {
    vector_field.indexes.has_hnsw() && live_doc_count != 0
}

pub(crate) fn persistable_flat_entries_from_writing(
    segment: &WritingSegment,
) -> Vec<FlatIndexEntry> {
    let index = segment
        .flat_index
        .as_ref()
        .expect("enabled writing flat state should exist");
    index.entries().to_vec()
}

pub(crate) fn into_persisted_segment(
    segment: WritingSegment,
    segment_id: SegmentId,
    vector_field: &VectorFieldSchema,
) -> PersistedSegment {
    let mut meta = segment.meta;
    meta.id = segment_id;
    meta.path = segment_file_name(segment_id);
    PersistedSegment::new(meta, segment.records, vector_field)
}
