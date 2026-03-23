use crate::{
    BinaryReader, BinaryWriter, read_scalar_value, read_segment_meta, write_scalar_value,
    write_segment_meta,
};
use garuda_types::{
    AccessMode, CollectionName, CollectionOptions, CollectionSchema, DistanceMetric, DocId,
    FieldName, HnswEfConstruction, HnswEfSearch, HnswIndexParams, HnswM, HnswMinNeighborCount,
    HnswPruneWidth, HnswScalingFactor, IndexKind, InternalDocId, Manifest, ManifestVersionId,
    Nullability, ScalarFieldSchema, ScalarIndexState, ScalarType, ScalarValue, SegmentId,
    SnapshotId, Status, StatusCode, StorageAccess, VectorDimension, VectorIndexState,
};

const MANIFEST_MAGIC: &[u8; 8] = b"GRDMAN01";
const ID_MAP_MAGIC: &[u8; 8] = b"GRDIDM01";
const DELETE_MAGIC: &[u8; 8] = b"GRDDEL01";
const FORMAT_VERSION: u16 = 1;

pub fn encode_manifest(manifest: &Manifest) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(MANIFEST_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    write_collection_schema(&mut writer, &manifest.schema)?;
    write_collection_options(&mut writer, &manifest.options);
    writer.write_u64(manifest.next_doc_id.get());
    writer.write_u64(manifest.next_segment_id.get());
    writer.write_u64(manifest.id_map_snapshot_id.get());
    writer.write_u64(manifest.delete_snapshot_id.get());
    writer.write_u64(manifest.manifest_version_id.get());
    write_segment_meta(&mut writer, &manifest.writing_segment)?;
    writer.write_len(manifest.persisted_segments.len())?;

    for meta in &manifest.persisted_segments {
        write_segment_meta(&mut writer, meta)?;
    }

    Ok(writer.finish())
}

pub fn decode_manifest(bytes: &[u8]) -> Result<Manifest, Status> {
    let mut reader = BinaryReader::new(bytes, MANIFEST_MAGIC, "storage")?;
    reader.expect_u16(FORMAT_VERSION)?;

    let schema = read_collection_schema(&mut reader)?;
    let options = read_collection_options(&mut reader)?;
    let next_doc_id = InternalDocId::new(reader.read_u64()?)?;
    let next_segment_id = SegmentId::new_unchecked(reader.read_u64()?);
    let id_map_snapshot_id = SnapshotId::new(reader.read_u64()?);
    let delete_snapshot_id = SnapshotId::new(reader.read_u64()?);
    let manifest_version_id = ManifestVersionId::new(reader.read_u64()?);
    let writing_segment = read_segment_meta(&mut reader)?;
    let persisted_segment_count = reader.read_len()?;
    let mut persisted_segments = Vec::with_capacity(persisted_segment_count);

    for _ in 0..persisted_segment_count {
        persisted_segments.push(read_segment_meta(&mut reader)?);
    }

    reader.finish()?;

    Ok(Manifest {
        schema,
        options,
        next_doc_id,
        next_segment_id,
        id_map_snapshot_id,
        delete_snapshot_id,
        manifest_version_id,
        writing_segment,
        persisted_segments,
    })
}

pub fn encode_id_map(entries: &[(DocId, InternalDocId)]) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(ID_MAP_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    writer.write_len(entries.len())?;

    for (doc_id, internal_doc_id) in entries {
        writer.write_string(doc_id.as_str())?;
        writer.write_u64(internal_doc_id.get());
    }

    Ok(writer.finish())
}

pub fn decode_id_map(bytes: &[u8]) -> Result<Vec<(String, InternalDocId)>, Status> {
    let mut reader = BinaryReader::new(bytes, ID_MAP_MAGIC, "storage")?;
    reader.expect_u16(FORMAT_VERSION)?;
    let count = reader.read_len()?;
    let mut entries = Vec::with_capacity(count);

    for _ in 0..count {
        entries.push((
            reader.read_string()?,
            InternalDocId::new(reader.read_u64()?)?,
        ));
    }

    reader.finish()?;
    Ok(entries)
}

pub fn encode_delete_snapshot(ids: &[InternalDocId]) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(DELETE_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    writer.write_len(ids.len())?;

    for id in ids {
        writer.write_u64(id.get());
    }

    Ok(writer.finish())
}

pub fn decode_delete_snapshot(bytes: &[u8]) -> Result<Vec<InternalDocId>, Status> {
    let mut reader = BinaryReader::new(bytes, DELETE_MAGIC, "storage")?;
    reader.expect_u16(FORMAT_VERSION)?;
    let count = reader.read_len()?;
    let mut ids = Vec::with_capacity(count);

    for _ in 0..count {
        ids.push(InternalDocId::new(reader.read_u64()?)?);
    }

    reader.finish()?;
    Ok(ids)
}

fn write_collection_schema(
    writer: &mut BinaryWriter,
    schema: &CollectionSchema,
) -> Result<(), Status> {
    writer.write_string(schema.name.as_str())?;
    writer.write_string(schema.primary_key.as_str())?;
    writer.write_len(schema.fields.len())?;

    for field in &schema.fields {
        write_scalar_field_schema(writer, field)?;
    }

    writer.write_string(schema.vector.name.as_str())?;
    writer.write_len(schema.vector.dimension.get())?;
    writer.write_u8(schema.vector.metric.to_tag());
    write_vector_index_state(writer, &schema.vector.indexes)?;

    Ok(())
}

fn read_collection_schema(reader: &mut BinaryReader<'_>) -> Result<CollectionSchema, Status> {
    let name = CollectionName::parse(reader.read_string()?)?;
    let primary_key = FieldName::parse(reader.read_string()?)?;
    let field_count = reader.read_len()?;
    let mut fields = Vec::with_capacity(field_count);

    for _ in 0..field_count {
        fields.push(read_scalar_field_schema(reader)?);
    }

    let vector_name = FieldName::parse(reader.read_string()?)?;
    let dimension = VectorDimension::new(reader.read_len()?)?;
    let metric = DistanceMetric::from_tag(reader.read_u8()?)?;
    let indexes = read_vector_index_state(reader)?;

    Ok(CollectionSchema {
        name,
        primary_key,
        fields,
        vector: garuda_types::VectorFieldSchema {
            name: vector_name,
            dimension,
            metric,
            indexes,
        },
    })
}

fn write_collection_options(writer: &mut BinaryWriter, options: &CollectionOptions) {
    writer.write_u8(options.access_mode.to_tag());
    writer.write_u8(options.storage_access.to_tag());
    writer.write_u64(options.segment_max_docs as u64);
}

fn read_collection_options(reader: &mut BinaryReader<'_>) -> Result<CollectionOptions, Status> {
    let access_mode = AccessMode::from_tag(reader.read_u8()?)?;
    let storage_access = StorageAccess::from_tag(reader.read_u8()?)?;
    let segment_max_docs = reader.read_u64()? as usize;

    Ok(CollectionOptions {
        access_mode,
        storage_access,
        segment_max_docs,
    })
}

fn write_scalar_field_schema(
    writer: &mut BinaryWriter,
    field: &ScalarFieldSchema,
) -> Result<(), Status> {
    writer.write_string(field.name.as_str())?;
    writer.write_u8(field.field_type.to_tag());
    writer.write_u8(write_scalar_index_state(field.index));
    writer.write_u8(field.nullability.to_tag());
    write_optional_scalar_value(writer, field.default_value.as_ref())?;
    Ok(())
}

fn read_scalar_field_schema(reader: &mut BinaryReader<'_>) -> Result<ScalarFieldSchema, Status> {
    let name = FieldName::parse(reader.read_string()?)?;
    let field_type = ScalarType::from_tag(reader.read_u8()?)?;
    let index = read_scalar_index_state(reader.read_u8()?)?;
    let nullability = Nullability::from_tag(reader.read_u8()?)?;
    let default_value = read_optional_scalar_value(reader)?;

    Ok(ScalarFieldSchema {
        name,
        field_type,
        index,
        nullability,
        default_value,
    })
}

fn write_vector_index_state(
    writer: &mut BinaryWriter,
    indexes: &VectorIndexState,
) -> Result<(), Status> {
    match indexes {
        VectorIndexState::DefaultFlat => {
            writer.write_u8(0);
        }
        VectorIndexState::HnswOnly(params) => {
            writer.write_u8(1);
            write_hnsw_index_params(writer, params);
        }
        VectorIndexState::FlatAndHnsw { default, hnsw } => {
            writer.write_u8(2);
            writer.write_u8(write_index_kind(*default));
            write_hnsw_index_params(writer, hnsw);
        }
    }

    Ok(())
}

fn read_vector_index_state(reader: &mut BinaryReader<'_>) -> Result<VectorIndexState, Status> {
    match reader.read_u8()? {
        0 => Ok(VectorIndexState::DefaultFlat),
        1 => Ok(VectorIndexState::HnswOnly(read_hnsw_index_params(reader)?)),
        2 => Ok(VectorIndexState::FlatAndHnsw {
            default: read_index_kind(reader.read_u8()?)?,
            hnsw: read_hnsw_index_params(reader)?,
        }),
        _ => Err(Status::err(
            StatusCode::Internal,
            "unrecognized vector index state tag",
        )),
    }
}

fn write_hnsw_index_params(writer: &mut BinaryWriter, params: &HnswIndexParams) {
    writer.write_u64(params.max_neighbors.get() as u64);
    writer.write_u64(params.scaling_factor.get() as u64);
    writer.write_u64(params.ef_construction.get() as u64);
    writer.write_u64(params.prune_width.get() as u64);
    writer.write_u64(params.min_neighbor_count.get() as u64);
    writer.write_u64(params.ef_search.get() as u64);
}

fn read_hnsw_index_params(reader: &mut BinaryReader<'_>) -> Result<HnswIndexParams, Status> {
    Ok(HnswIndexParams {
        max_neighbors: HnswM::from_persisted_u64(reader.read_u64()?)?,
        scaling_factor: HnswScalingFactor::from_persisted_u64(reader.read_u64()?)?,
        ef_construction: HnswEfConstruction::from_persisted_u64(reader.read_u64()?)?,
        prune_width: HnswPruneWidth::from_persisted_u64(reader.read_u64()?)?,
        min_neighbor_count: HnswMinNeighborCount::from_persisted_u64(reader.read_u64()?)?,
        ef_search: HnswEfSearch::from_persisted_u64(reader.read_u64()?)?,
    })
}

fn write_index_kind(kind: IndexKind) -> u8 {
    match kind {
        IndexKind::Flat => 0,
        IndexKind::Hnsw => 1,
        IndexKind::Scalar => 2,
    }
}

fn read_index_kind(tag: u8) -> Result<IndexKind, Status> {
    match tag {
        0 => Ok(IndexKind::Flat),
        1 => Ok(IndexKind::Hnsw),
        2 => Ok(IndexKind::Scalar),
        _ => Err(Status::err(
            StatusCode::Internal,
            "unrecognized index kind tag",
        )),
    }
}

fn write_scalar_index_state(state: ScalarIndexState) -> u8 {
    match state {
        ScalarIndexState::None => 0,
        ScalarIndexState::Indexed => 1,
    }
}

fn read_scalar_index_state(tag: u8) -> Result<ScalarIndexState, Status> {
    match tag {
        0 => Ok(ScalarIndexState::None),
        1 => Ok(ScalarIndexState::Indexed),
        _ => Err(Status::err(
            StatusCode::Internal,
            "unrecognized scalar index state tag",
        )),
    }
}

fn write_optional_scalar_value(
    writer: &mut BinaryWriter,
    value: Option<&ScalarValue>,
) -> Result<(), Status> {
    match value {
        Some(value) => {
            writer.write_bool(true);
            write_scalar_value(writer, value)?;
        }
        None => writer.write_bool(false),
    }

    Ok(())
}

fn read_optional_scalar_value(
    reader: &mut BinaryReader<'_>,
) -> Result<Option<ScalarValue>, Status> {
    if !reader.read_bool()? {
        return Ok(None);
    }

    Ok(Some(read_scalar_value(reader)?))
}
