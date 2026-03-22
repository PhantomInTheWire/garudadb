use crate::{RecordState, SegmentFile, StoredRecord};
use garuda_index_flat::{FlatIndex, FlatIndexEntry};
use garuda_types::{
    DenseVector, DistanceMetric, Doc, DocId, IndexParams, InternalDocId, ScalarValue, SegmentId,
    SegmentMeta, Status, StatusCode, VectorDimension, VectorFieldSchema,
};
use garuda_types::{HnswGraph, HnswLevel, HnswNeighborLimits};
const SEGMENT_MAGIC: &[u8; 8] = b"GRDSEG01";
const FLAT_INDEX_MAGIC: &[u8; 8] = b"GRDFLT01";
const HNSW_INDEX_MAGIC: &[u8; 8] = b"GRDHNS01";
const FORMAT_VERSION: u16 = 1;
const FNV_OFFSET_BASIS: u32 = 2_166_136_261;
const FNV_PRIME: u32 = 16_777_619;

pub fn encode_segment(segment: &SegmentFile) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(SEGMENT_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    write_segment_meta(&mut writer, &segment.meta)?;
    writer.write_len(segment.records.len())?;

    for record in &segment.records {
        writer.write_u64(record.doc_id.get());
        writer.write_u8(record.state.to_tag());
        write_doc(&mut writer, &record.doc)?;
    }

    Ok(writer.finish())
}

pub fn decode_segment(bytes: &[u8]) -> Result<SegmentFile, Status> {
    let mut reader = BinaryReader::new(bytes, SEGMENT_MAGIC)?;
    reader.expect_u16(FORMAT_VERSION)?;
    let meta = read_segment_meta(&mut reader)?;
    let record_count = reader.read_len()?;
    let mut records = Vec::with_capacity(record_count);

    for _ in 0..record_count {
        let doc_id = InternalDocId::new(reader.read_u64()?)?;
        let state = RecordState::from_tag(reader.read_u8()?)?;
        let doc = read_doc(&mut reader)?;
        records.push(StoredRecord { doc_id, state, doc });
    }

    reader.finish()?;

    Ok(SegmentFile::new_persisted_with_records(meta, records))
}

pub fn encode_doc_payload(doc: &Doc) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(b"GRDDOC01");
    writer.write_u16(FORMAT_VERSION);
    write_doc(&mut writer, doc)?;
    Ok(writer.finish())
}

pub fn encode_flat_index(
    entries: Vec<FlatIndexEntry>,
    vector_field: &VectorFieldSchema,
) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(FLAT_INDEX_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    writer.write_u64(vector_field.dimension.get() as u64);
    writer.write_u8(vector_field.metric.to_tag());
    writer.write_len(entries.len())?;

    for entry in entries {
        writer.write_u64(entry.doc_id.get());
        writer.write_len(entry.vector.len())?;

        for value in entry.vector.as_slice() {
            writer.write_f32(*value);
        }
    }

    Ok(writer.finish())
}

pub fn decode_flat_index(
    bytes: &[u8],
    vector_field: &VectorFieldSchema,
) -> Result<FlatIndex, Status> {
    let mut reader = BinaryReader::new(bytes, FLAT_INDEX_MAGIC)?;
    reader.expect_u16(FORMAT_VERSION)?;

    let dimension = VectorDimension::new(reader.read_u64()? as usize)?;
    if dimension != vector_field.dimension {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted flat index dimension does not match schema",
        ));
    }

    let metric = DistanceMetric::from_tag(reader.read_u8()?)?;
    if metric != vector_field.metric {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted flat index metric does not match schema",
        ));
    }

    if !matches!(vector_field.index, IndexParams::Flat(_)) {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted flat index requested for a non-flat vector field",
        ));
    }

    let entry_count = reader.read_len()?;
    let mut entries = Vec::with_capacity(entry_count);

    for _ in 0..entry_count {
        let doc_id = InternalDocId::new(reader.read_u64()?)?;
        let vector_len = reader.read_len()?;
        let mut vector = Vec::with_capacity(vector_len);

        for _ in 0..vector_len {
            vector.push(reader.read_f32()?);
        }

        entries.push(FlatIndexEntry::new(doc_id, DenseVector::parse(vector)?));
    }

    reader.finish()?;
    FlatIndex::build(dimension, entries)
}

pub fn encode_hnsw_graph(graph: &HnswGraph) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(HNSW_INDEX_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    writer.write_len(graph.node_count())?;
    writer.write_len(graph.level_count())?;

    for &level in graph.node_levels() {
        writer.write_len(level.get())?;
    }

    for raw_level in 0..graph.level_count() {
        let level = HnswLevel::new(raw_level);

        for raw_node in 0..graph.node_count() {
            let node = garuda_types::NodeIndex::new(raw_node);
            let neighbors = graph.neighbors(level, node);
            writer.write_len(neighbors.len())?;

            for &neighbor in neighbors {
                writer.write_len(neighbor.get())?;
            }
        }
    }

    Ok(writer.finish())
}

pub fn decode_hnsw_graph(
    bytes: &[u8],
    vector_field: &VectorFieldSchema,
    entry_count: usize,
) -> Result<HnswGraph, Status> {
    let mut reader = BinaryReader::new(bytes, HNSW_INDEX_MAGIC)?;
    reader.expect_u16(FORMAT_VERSION)?;

    if !matches!(&vector_field.index, IndexParams::Hnsw(_)) {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted hnsw index requested for a non-hnsw vector field",
        ));
    }

    let node_count = reader.read_len()?;
    let level_count = reader.read_len()?;
    let mut node_levels = Vec::with_capacity(node_count);

    for _ in 0..node_count {
        node_levels.push(HnswLevel::new(reader.read_len()?));
    }

    let mut levels = Vec::with_capacity(level_count);

    for _ in 0..level_count {
        let mut nodes = Vec::with_capacity(node_count);

        for _ in 0..node_count {
            let neighbor_count = reader.read_len()?;
            let mut neighbors = Vec::with_capacity(neighbor_count);

            for _ in 0..neighbor_count {
                neighbors.push(garuda_types::NodeIndex::new(reader.read_len()?));
            }

            nodes.push(neighbors);
        }

        levels.push(nodes);
    }

    reader.finish()?;

    let IndexParams::Hnsw(params) = &vector_field.index else {
        unreachable!("validated hnsw vector field")
    };

    HnswGraph::from_parts(
        node_levels,
        levels,
        entry_count,
        HnswNeighborLimits::new(params.max_neighbors),
    )
}

pub fn decode_doc_payload(bytes: &[u8]) -> Result<Doc, Status> {
    let mut reader = BinaryReader::new(bytes, b"GRDDOC01")?;
    reader.expect_u16(FORMAT_VERSION)?;
    let doc = read_doc(&mut reader)?;
    reader.finish()?;
    Ok(doc)
}

fn write_segment_meta(writer: &mut BinaryWriter, meta: &SegmentMeta) -> Result<(), Status> {
    writer.write_u64(meta.id.get());
    writer.write_string(&meta.path)?;
    writer.write_optional_internal_doc_id(meta.min_doc_id);
    writer.write_optional_internal_doc_id(meta.max_doc_id);
    writer.write_u64(meta.doc_count as u64);
    Ok(())
}

fn read_segment_meta(reader: &mut BinaryReader<'_>) -> Result<SegmentMeta, Status> {
    Ok(SegmentMeta {
        id: SegmentId::new_unchecked(reader.read_u64()?),
        path: reader.read_string()?,
        min_doc_id: reader.read_optional_internal_doc_id()?,
        max_doc_id: reader.read_optional_internal_doc_id()?,
        doc_count: reader.read_u64()? as usize,
    })
}

fn write_doc(writer: &mut BinaryWriter, doc: &Doc) -> Result<(), Status> {
    writer.write_string(doc.id.as_str())?;
    writer.write_len(doc.fields.len())?;

    for (name, value) in &doc.fields {
        writer.write_string(name)?;
        write_scalar_value(writer, value)?;
    }

    writer.write_len(doc.vector.len())?;
    for value in doc.vector.as_slice() {
        writer.write_f32(*value);
    }

    writer.write_optional_f32(doc.score);
    Ok(())
}

fn read_doc(reader: &mut BinaryReader<'_>) -> Result<Doc, Status> {
    let id = DocId::parse(reader.read_string()?)?;
    let field_count = reader.read_len()?;
    let mut fields = std::collections::BTreeMap::new();

    for _ in 0..field_count {
        fields.insert(reader.read_string()?, read_scalar_value(reader)?);
    }

    let vector_len = reader.read_len()?;
    let mut vector = Vec::with_capacity(vector_len);

    for _ in 0..vector_len {
        vector.push(reader.read_f32()?);
    }

    let score = reader.read_optional_f32()?;

    Ok(Doc {
        id,
        fields,
        vector: DenseVector::parse(vector)?,
        score,
    })
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
        ScalarValue::Null => writer.write_u8(4),
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

    fn write_f32(&mut self, value: f32) {
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

    fn write_optional_internal_doc_id(&mut self, value: Option<InternalDocId>) {
        self.write_optional_u64(value.map(InternalDocId::get));
    }

    fn write_optional_f32(&mut self, value: Option<f32>) {
        match value {
            Some(value) => {
                self.write_bool(true);
                self.write_f32(value);
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
                "segment file is too short",
            ));
        }

        if &bytes[..magic.len()] != magic {
            return Err(Status::err(
                StatusCode::Internal,
                "unexpected segment file magic",
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
                "segment checksum mismatch",
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
            "segment file contains trailing bytes",
        ))
    }

    fn expect_u16(&mut self, expected: u16) -> Result<(), Status> {
        let actual = self.read_u16()?;
        if actual == expected {
            return Ok(());
        }

        Err(Status::err(
            StatusCode::Internal,
            "unsupported segment format version",
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

    fn read_f32(&mut self) -> Result<f32, Status> {
        let bytes = self.read_exact(4)?;
        Ok(f32::from_le_bytes(bytes.try_into().expect("fixed length")))
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

    fn read_optional_internal_doc_id(&mut self) -> Result<Option<InternalDocId>, Status> {
        let Some(value) = self.read_optional_u64()? else {
            return Ok(None);
        };

        Ok(Some(InternalDocId::new(value)?))
    }

    fn read_optional_f32(&mut self) -> Result<Option<f32>, Status> {
        if !self.read_bool()? {
            return Ok(None);
        }

        Ok(Some(self.read_f32()?))
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], Status> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or_else(|| Status::err(StatusCode::Internal, "segment read overflow"))?;

        if end > self.payload_len {
            return Err(Status::err(
                StatusCode::Internal,
                "segment file ended unexpectedly",
            ));
        }

        let bytes = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(bytes)
    }
}

pub(crate) fn checksum(bytes: &[u8]) -> u32 {
    let mut hash = FNV_OFFSET_BASIS;

    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}
