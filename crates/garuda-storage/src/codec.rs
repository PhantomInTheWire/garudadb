use garuda_types::{
    CollectionName, CollectionOptions, CollectionSchema, DistanceMetric, FieldName,
    FlatIndexParams, HnswIndexParams, IndexParams, Manifest, ManifestVersionId, ScalarFieldSchema,
    ScalarType, ScalarValue, SegmentMeta, SnapshotId, Status, StatusCode,
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
    writer.write_u64(manifest.next_doc_id);
    writer.write_u64(manifest.next_segment_id);
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
    let mut reader = BinaryReader::new(bytes, MANIFEST_MAGIC)?;
    reader.expect_u16(FORMAT_VERSION)?;

    let schema = read_collection_schema(&mut reader)?;
    let options = read_collection_options(&mut reader)?;
    let next_doc_id = reader.read_u64()?;
    let next_segment_id = reader.read_u64()?;
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

pub fn encode_id_map(entries: &[(String, u64)]) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(ID_MAP_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    writer.write_len(entries.len())?;

    for (doc_id, internal_doc_id) in entries {
        writer.write_string(doc_id)?;
        writer.write_u64(*internal_doc_id);
    }

    Ok(writer.finish())
}

pub fn decode_id_map(bytes: &[u8]) -> Result<Vec<(String, u64)>, Status> {
    let mut reader = BinaryReader::new(bytes, ID_MAP_MAGIC)?;
    reader.expect_u16(FORMAT_VERSION)?;
    let count = reader.read_len()?;
    let mut entries = Vec::with_capacity(count);

    for _ in 0..count {
        entries.push((reader.read_string()?, reader.read_u64()?));
    }

    reader.finish()?;
    Ok(entries)
}

pub fn encode_delete_snapshot(ids: &[u64]) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(DELETE_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    writer.write_len(ids.len())?;

    for id in ids {
        writer.write_u64(*id);
    }

    Ok(writer.finish())
}

pub fn decode_delete_snapshot(bytes: &[u8]) -> Result<Vec<u64>, Status> {
    let mut reader = BinaryReader::new(bytes, DELETE_MAGIC)?;
    reader.expect_u16(FORMAT_VERSION)?;
    let count = reader.read_len()?;
    let mut ids = Vec::with_capacity(count);

    for _ in 0..count {
        ids.push(reader.read_u64()?);
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
    writer.write_len(schema.vector.dimension)?;
    writer.write_u8(metric_tag(schema.vector.metric));
    write_index_params(writer, &schema.vector.index)?;

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
    let dimension = reader.read_len()?;
    let metric = read_metric(reader.read_u8()?)?;
    let index = read_index_params(reader)?;

    Ok(CollectionSchema {
        name,
        primary_key,
        fields,
        vector: garuda_types::VectorFieldSchema {
            name: vector_name,
            dimension,
            metric,
            index,
        },
    })
}

fn write_collection_options(writer: &mut BinaryWriter, options: &CollectionOptions) {
    writer.write_bool(options.read_only);
    writer.write_bool(options.enable_mmap);
    writer.write_u64(options.segment_max_docs as u64);
}

fn read_collection_options(reader: &mut BinaryReader<'_>) -> Result<CollectionOptions, Status> {
    let read_only = reader.read_bool()?;
    let enable_mmap = reader.read_bool()?;
    let segment_max_docs = reader.read_u64()? as usize;

    Ok(CollectionOptions {
        read_only,
        enable_mmap,
        segment_max_docs,
    })
}

fn write_scalar_field_schema(
    writer: &mut BinaryWriter,
    field: &ScalarFieldSchema,
) -> Result<(), Status> {
    writer.write_string(field.name.as_str())?;
    writer.write_u8(scalar_type_tag(field.field_type));
    writer.write_bool(field.nullable);
    write_optional_scalar_value(writer, field.default_value.as_ref())?;
    Ok(())
}

fn read_scalar_field_schema(reader: &mut BinaryReader<'_>) -> Result<ScalarFieldSchema, Status> {
    let name = FieldName::parse(reader.read_string()?)?;
    let field_type = read_scalar_type(reader.read_u8()?)?;
    let nullable = reader.read_bool()?;
    let default_value = read_optional_scalar_value(reader)?;

    Ok(ScalarFieldSchema {
        name,
        field_type,
        nullable,
        default_value,
    })
}

fn write_index_params(writer: &mut BinaryWriter, index: &IndexParams) -> Result<(), Status> {
    match index {
        IndexParams::Flat(_) => {
            writer.write_u8(0);
        }
        IndexParams::Hnsw(params) => {
            writer.write_u8(1);
            writer.write_u64(params.m as u64);
            writer.write_u64(params.ef_construction as u64);
            writer.write_u64(params.ef_search as u64);
        }
    }

    Ok(())
}

fn read_index_params(reader: &mut BinaryReader<'_>) -> Result<IndexParams, Status> {
    match reader.read_u8()? {
        0 => Ok(IndexParams::Flat(FlatIndexParams)),
        1 => Ok(IndexParams::Hnsw(HnswIndexParams {
            m: reader.read_u64()? as usize,
            ef_construction: reader.read_u64()? as usize,
            ef_search: reader.read_u64()? as usize,
        })),
        _ => Err(Status::err(
            StatusCode::Internal,
            "unrecognized index params tag",
        )),
    }
}

fn write_segment_meta(writer: &mut BinaryWriter, meta: &SegmentMeta) -> Result<(), Status> {
    writer.write_u64(meta.id);
    writer.write_string(&meta.path)?;
    writer.write_optional_u64(meta.min_doc_id);
    writer.write_optional_u64(meta.max_doc_id);
    writer.write_u64(meta.doc_count as u64);
    Ok(())
}

fn read_segment_meta(reader: &mut BinaryReader<'_>) -> Result<SegmentMeta, Status> {
    Ok(SegmentMeta {
        id: reader.read_u64()?,
        path: reader.read_string()?,
        min_doc_id: reader.read_optional_u64()?,
        max_doc_id: reader.read_optional_u64()?,
        doc_count: reader.read_u64()? as usize,
    })
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

fn write_scalar_value(writer: &mut BinaryWriter, value: &ScalarValue) -> Result<(), Status> {
    match value {
        ScalarValue::Bool(value) => {
            writer.write_u8(0);
            writer.write_bool(*value);
        }
        ScalarValue::Int64(value) => {
            writer.write_u8(1);
            writer.write_i64(*value);
        }
        ScalarValue::Float64(value) => {
            writer.write_u8(2);
            writer.write_f64(*value);
        }
        ScalarValue::String(value) => {
            writer.write_u8(3);
            writer.write_string(value)?;
        }
        ScalarValue::Null => {
            writer.write_u8(4);
        }
    }

    Ok(())
}

fn read_scalar_value(reader: &mut BinaryReader<'_>) -> Result<ScalarValue, Status> {
    match reader.read_u8()? {
        0 => Ok(ScalarValue::Bool(reader.read_bool()?)),
        1 => Ok(ScalarValue::Int64(reader.read_i64()?)),
        2 => Ok(ScalarValue::Float64(reader.read_f64()?)),
        3 => Ok(ScalarValue::String(reader.read_string()?)),
        4 => Ok(ScalarValue::Null),
        _ => Err(Status::err(
            StatusCode::Internal,
            "unrecognized scalar value tag",
        )),
    }
}

fn metric_tag(metric: DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::Cosine => 0,
        DistanceMetric::InnerProduct => 1,
        DistanceMetric::L2 => 2,
    }
}

fn read_metric(tag: u8) -> Result<DistanceMetric, Status> {
    match tag {
        0 => Ok(DistanceMetric::Cosine),
        1 => Ok(DistanceMetric::InnerProduct),
        2 => Ok(DistanceMetric::L2),
        _ => Err(Status::err(StatusCode::Internal, "unrecognized metric tag")),
    }
}

fn scalar_type_tag(scalar_type: ScalarType) -> u8 {
    match scalar_type {
        ScalarType::Bool => 0,
        ScalarType::Int64 => 1,
        ScalarType::Float64 => 2,
        ScalarType::String => 3,
    }
}

fn read_scalar_type(tag: u8) -> Result<ScalarType, Status> {
    match tag {
        0 => Ok(ScalarType::Bool),
        1 => Ok(ScalarType::Int64),
        2 => Ok(ScalarType::Float64),
        3 => Ok(ScalarType::String),
        _ => Err(Status::err(
            StatusCode::Internal,
            "unrecognized scalar type tag",
        )),
    }
}

struct BinaryWriter {
    bytes: Vec<u8>,
}

impl BinaryWriter {
    fn new(magic: &[u8; 8]) -> Self {
        let mut bytes = Vec::with_capacity(magic.len() + std::mem::size_of::<u32>());
        bytes.extend_from_slice(magic);
        Self { bytes }
    }

    fn finish(mut self) -> Vec<u8> {
        let checksum = checksum(&self.bytes);
        self.bytes.extend_from_slice(&checksum.to_le_bytes());
        self.bytes
    }

    fn write_bool(&mut self, value: bool) {
        self.bytes.push(u8::from(value));
    }

    fn write_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn write_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_f64(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_len(&mut self, value: usize) -> Result<(), Status> {
        let value = u64::try_from(value)
            .map_err(|_| Status::err(StatusCode::Internal, "value exceeds supported range"))?;

        self.write_u64(value);
        Ok(())
    }

    fn write_string(&mut self, value: &str) -> Result<(), Status> {
        self.write_len(value.len())?;
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn write_optional_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.write_bool(true);
                self.write_u64(value);
            }
            None => self.write_bool(false),
        }
    }
}

struct BinaryReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
    payload_len: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(bytes: &'a [u8], magic: &[u8; 8]) -> Result<Self, Status> {
        if bytes.len() < magic.len() + std::mem::size_of::<u32>() {
            return Err(Status::err(
                StatusCode::Internal,
                "storage file is too short",
            ));
        }

        if &bytes[..magic.len()] != magic {
            return Err(Status::err(
                StatusCode::Internal,
                "unexpected storage file magic",
            ));
        }

        let payload_len = bytes.len() - std::mem::size_of::<u32>();
        let expected_checksum = u32::from_le_bytes(
            bytes[payload_len..]
                .try_into()
                .expect("checksum length is fixed"),
        );
        let actual_checksum = checksum(&bytes[..payload_len]);

        if expected_checksum != actual_checksum {
            return Err(Status::err(
                StatusCode::Internal,
                "storage checksum mismatch",
            ));
        }

        Ok(Self {
            bytes,
            cursor: magic.len(),
            payload_len,
        })
    }

    fn finish(&self) -> Result<(), Status> {
        if self.cursor == self.payload_len {
            return Ok(());
        }

        Err(Status::err(
            StatusCode::Internal,
            "storage file contains trailing bytes",
        ))
    }

    fn expect_u16(&mut self, expected: u16) -> Result<(), Status> {
        let actual = self.read_u16()?;
        if actual == expected {
            return Ok(());
        }

        Err(Status::err(
            StatusCode::Internal,
            "unsupported storage format version",
        ))
    }

    fn read_bool(&mut self) -> Result<bool, Status> {
        Ok(self.read_u8()? != 0)
    }

    fn read_u8(&mut self) -> Result<u8, Status> {
        let bytes = self.read_exact(1)?;
        Ok(bytes[0])
    }

    fn read_u16(&mut self) -> Result<u16, Status> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    fn read_u64(&mut self) -> Result<u64, Status> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    fn read_i64(&mut self) -> Result<i64, Status> {
        let bytes = self.read_exact(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    fn read_f64(&mut self) -> Result<f64, Status> {
        let bytes = self.read_exact(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    fn read_len(&mut self) -> Result<usize, Status> {
        let value = self.read_u64()?;
        usize::try_from(value)
            .map_err(|_| Status::err(StatusCode::Internal, "value exceeds supported range"))
    }

    fn read_string(&mut self) -> Result<String, Status> {
        let len = self.read_len()?;
        let bytes = self.read_exact(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|error| Status::err(StatusCode::Internal, format!("invalid utf-8: {error}")))
    }

    fn read_optional_u64(&mut self) -> Result<Option<u64>, Status> {
        if !self.read_bool()? {
            return Ok(None);
        }

        Ok(Some(self.read_u64()?))
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], Status> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or_else(|| Status::err(StatusCode::Internal, "storage read overflow"))?;

        if end > self.payload_len {
            return Err(Status::err(
                StatusCode::Internal,
                "storage file ended unexpectedly",
            ));
        }

        let bytes = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(bytes)
    }
}

fn checksum(bytes: &[u8]) -> u32 {
    let mut hash = 2_166_136_261u32;

    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(16_777_619);
    }

    hash
}
