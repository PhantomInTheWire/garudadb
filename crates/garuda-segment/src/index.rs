use crate::codec::{decode_flat_index, decode_hnsw_graph, decode_ivf_index, decode_scalar_index};
use crate::types::{RecordState, StoredRecord, WritingSegment};
use garuda_index_flat::{FlatIndex, FlatIndexEntry, WritingFlatIndex};
use garuda_index_hnsw::{
    HnswBuildConfig, HnswBuildEntry, HnswIndex, HnswIndexConfig, WritingHnswIndex,
};
use garuda_index_ivf::{IvfBuildEntry, IvfIndex, IvfIndexConfig, WritingIvfIndex};
use garuda_index_scalar::ScalarIndex;
use garuda_storage::{
    read_file, segment_flat_index_path, segment_hnsw_index_path, segment_ivf_index_path,
    segment_scalar_index_path,
};
use garuda_types::{
    CollectionSchema, FieldName, HnswIndexParams, IvfIndexParams, ScalarFieldSchema, SegmentId,
    SegmentMeta, Status, StatusCode,
};
use std::collections::BTreeMap;

pub(crate) struct WritingSearchResources {
    pub flat_index: Option<WritingFlatIndex>,
    pub hnsw_index: Option<WritingHnswIndex>,
    pub ivf_index: Option<WritingIvfIndex>,
    pub scalar_indexes: BTreeMap<FieldName, ScalarIndex>,
}

pub(crate) struct PersistedSearchResources {
    pub flat_index: Option<FlatIndex>,
    pub hnsw_index: Option<HnswIndex>,
    pub ivf_index: Option<IvfIndex>,
    pub scalar_indexes: BTreeMap<FieldName, ScalarIndex>,
}

pub(crate) fn build_writing_search_resources(
    schema: &CollectionSchema,
    records: &[StoredRecord],
) -> WritingSearchResources {
    let mut flat_index = if schema.vector.indexes.has_flat() {
        Some(WritingFlatIndex::new(schema.vector.dimension))
    } else {
        None
    };
    let mut hnsw_index = schema
        .vector
        .indexes
        .hnsw_params()
        .map(|params| WritingHnswIndex::new(hnsw_index_config(schema, params)));
    let ivf_index = schema
        .vector
        .indexes
        .ivf_params()
        .cloned()
        .map(|_| build_writing_ivf_index(schema, records));
    let mut scalar_indexes = build_scalar_indexes(&schema.fields);

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

        index_scalar_fields(&mut scalar_indexes, &schema.fields, record);
    }

    WritingSearchResources {
        flat_index,
        hnsw_index,
        ivf_index,
        scalar_indexes,
    }
}

pub(crate) fn build_persisted_search_resources(
    schema: &CollectionSchema,
    meta: &SegmentMeta,
    records: &[StoredRecord],
) -> PersistedSearchResources {
    PersistedSearchResources {
        flat_index: build_flat_index(schema, meta, records),
        hnsw_index: build_hnsw_index(schema, meta, records),
        ivf_index: build_ivf_index(schema, meta, records),
        scalar_indexes: build_persisted_scalar_indexes(&schema.fields, records),
    }
}

pub(crate) fn build_persisted_vector_resources(
    schema: &CollectionSchema,
    meta: &SegmentMeta,
    records: &[StoredRecord],
) -> (Option<HnswIndex>, Option<IvfIndex>) {
    (
        build_hnsw_index(schema, meta, records),
        build_ivf_index(schema, meta, records),
    )
}

pub(crate) fn load_persisted_search_resources(
    root: &std::path::Path,
    segment_id: SegmentId,
    schema: &CollectionSchema,
    meta: &SegmentMeta,
    records: &[StoredRecord],
) -> Result<PersistedSearchResources, Status> {
    Ok(PersistedSearchResources {
        flat_index: load_flat_index(root, segment_id, meta, schema)?,
        hnsw_index: load_hnsw_index(root, segment_id, meta, records, schema)?,
        ivf_index: load_ivf_index(root, segment_id, meta, records, schema)?,
        scalar_indexes: load_scalar_indexes(root, segment_id, &schema.fields, meta.doc_count)?,
    })
}

pub(crate) fn hnsw_index_config(
    schema: &CollectionSchema,
    params: &HnswIndexParams,
) -> HnswIndexConfig {
    HnswIndexConfig::new(
        schema.vector.dimension,
        schema.vector.metric,
        HnswBuildConfig::new(
            params.neighbor_config().expect("validated hnsw params"),
            params.scaling_factor,
            params.ef_construction,
            params.prune_width,
        ),
    )
}

pub(crate) fn ivf_index_config(
    schema: &CollectionSchema,
    params: IvfIndexParams,
) -> IvfIndexConfig {
    IvfIndexConfig::new(schema.vector.dimension, schema.vector.metric, params)
}

fn build_flat_index(
    schema: &CollectionSchema,
    meta: &SegmentMeta,
    records: &[StoredRecord],
) -> Option<FlatIndex> {
    if !schema.vector.indexes.has_flat() {
        return None;
    }

    let entries = flat_index_entries(records, meta.doc_count);
    let index = FlatIndex::build(schema.vector.dimension, entries)
        .expect("validated segment records should match the vector field dimension");
    Some(index)
}

fn build_hnsw_index(
    schema: &CollectionSchema,
    meta: &SegmentMeta,
    records: &[StoredRecord],
) -> Option<HnswIndex> {
    let params = schema.vector.indexes.hnsw_params()?;

    let config = hnsw_index_config(schema, params);
    let entries = hnsw_build_entries(&config, records, meta.doc_count);
    Some(HnswIndex::build(config, entries))
}

fn build_writing_ivf_index(schema: &CollectionSchema, records: &[StoredRecord]) -> WritingIvfIndex {
    let params = schema
        .vector
        .indexes
        .ivf_params()
        .cloned()
        .expect("ivf index must be enabled");
    let config = ivf_index_config(schema, params);
    let entries = ivf_build_entries(records, live_doc_count(records));
    WritingIvfIndex::from_entries_incremental(config, entries)
}

fn load_flat_index(
    root: &std::path::Path,
    segment_id: SegmentId,
    meta: &SegmentMeta,
    schema: &CollectionSchema,
) -> Result<Option<FlatIndex>, Status> {
    if !schema.vector.indexes.has_flat() {
        return Ok(None);
    }

    let bytes = read_file(&segment_flat_index_path(root, segment_id))?;
    let flat_index = decode_flat_index(&bytes, &schema.vector)?;

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
    schema: &CollectionSchema,
) -> Result<Option<HnswIndex>, Status> {
    let Some(params) = schema.vector.indexes.hnsw_params() else {
        return Ok(None);
    };

    let bytes = read_file(&segment_hnsw_index_path(root, segment_id))?;
    let graph = decode_hnsw_graph(&bytes, &schema.vector, meta.doc_count)?;
    let config = hnsw_index_config(schema, params);
    let entries = hnsw_build_entries(&config, records, meta.doc_count);

    if graph.node_count() != entries.len() {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted hnsw graph does not match rebuilt live entries",
        ));
    }

    Ok(Some(HnswIndex::from_parts(config, entries, graph)))
}

fn load_ivf_index(
    root: &std::path::Path,
    segment_id: SegmentId,
    meta: &SegmentMeta,
    records: &[StoredRecord],
    schema: &CollectionSchema,
) -> Result<Option<IvfIndex>, Status> {
    let Some(params) = schema.vector.indexes.ivf_params().cloned() else {
        return Ok(None);
    };

    let bytes = read_file(&segment_ivf_index_path(root, segment_id))?;
    let stored = decode_ivf_index(&bytes, &schema.vector)?;
    let config = ivf_index_config(schema, params);
    let entries = ivf_build_entries(records, meta.doc_count);

    if entries.len() != meta.doc_count {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted ivf index does not match segment live doc count",
        ));
    }

    Ok(Some(IvfIndex::from_parts(config, entries, stored)?))
}

fn build_scalar_indexes(fields: &[ScalarFieldSchema]) -> BTreeMap<FieldName, ScalarIndex> {
    let mut indexes = BTreeMap::new();

    for field in fields {
        if !field.is_indexed() {
            continue;
        }

        indexes.insert(field.name.clone(), ScalarIndex::new(field.field_type));
    }

    indexes
}

fn build_persisted_scalar_indexes(
    fields: &[ScalarFieldSchema],
    records: &[StoredRecord],
) -> BTreeMap<FieldName, ScalarIndex> {
    let mut indexes = build_scalar_indexes(fields);

    for record in records {
        if matches!(record.state, RecordState::Deleted) {
            continue;
        }

        index_scalar_fields(&mut indexes, fields, record);
    }

    indexes
}

fn load_scalar_indexes(
    root: &std::path::Path,
    segment_id: SegmentId,
    fields: &[ScalarFieldSchema],
    live_doc_count: usize,
) -> Result<BTreeMap<FieldName, ScalarIndex>, Status> {
    let mut indexes = BTreeMap::new();

    for field in fields {
        if !field.is_indexed() {
            continue;
        }

        if live_doc_count == 0 {
            indexes.insert(field.name.clone(), ScalarIndex::new(field.field_type));
            continue;
        }

        let bytes = read_file(&segment_scalar_index_path(root, segment_id, &field.name))?;
        let index = decode_scalar_index(&bytes, field, live_doc_count)?;
        indexes.insert(field.name.clone(), index);
    }

    Ok(indexes)
}

fn index_scalar_fields(
    indexes: &mut BTreeMap<FieldName, ScalarIndex>,
    fields: &[ScalarFieldSchema],
    record: &StoredRecord,
) {
    for field in fields {
        if !field.is_indexed() {
            continue;
        }

        let value = record
            .doc
            .fields
            .get(field.name.as_str())
            .expect("validated scalar field should exist");
        let index = indexes
            .get_mut(&field.name)
            .expect("enabled scalar index should exist");
        index.insert(record.doc_id, value);
    }
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

pub(crate) fn ivf_build_entries(
    records: &[StoredRecord],
    live_doc_count: usize,
) -> Vec<IvfBuildEntry> {
    let mut entries = Vec::with_capacity(live_doc_count);

    for record in records {
        if matches!(record.state, RecordState::Deleted) {
            continue;
        }

        entries.push(IvfBuildEntry::new(record.doc_id, record.doc.vector.clone()));
    }

    entries
}

pub(crate) fn should_persist_flat(schema: &CollectionSchema, live_doc_count: usize) -> bool {
    schema.vector.indexes.has_flat() && live_doc_count != 0
}

pub(crate) fn should_persist_hnsw(schema: &CollectionSchema, live_doc_count: usize) -> bool {
    schema.vector.indexes.has_hnsw() && live_doc_count != 0
}

pub(crate) fn should_persist_ivf(schema: &CollectionSchema, live_doc_count: usize) -> bool {
    schema.vector.indexes.has_ivf() && live_doc_count != 0
}

pub(crate) fn indexed_scalar_fields(
    schema: &CollectionSchema,
) -> impl Iterator<Item = &ScalarFieldSchema> {
    schema.fields.iter().filter(|field| field.is_indexed())
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

fn build_ivf_index(
    schema: &CollectionSchema,
    meta: &SegmentMeta,
    records: &[StoredRecord],
) -> Option<IvfIndex> {
    let params = schema.vector.indexes.ivf_params().cloned()?;
    Some(IvfIndex::build(
        ivf_index_config(schema, params),
        ivf_build_entries(records, meta.doc_count),
    ))
}

fn live_doc_count(records: &[StoredRecord]) -> usize {
    records
        .iter()
        .filter(|record| matches!(record.state, RecordState::Live))
        .count()
}
