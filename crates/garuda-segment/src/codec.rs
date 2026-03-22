use crate::doc_codec::{read_doc, write_doc};
use crate::{RecordState, StoredRecord};
use garuda_index_flat::{FlatIndex, FlatIndexEntry};
use garuda_storage::{BinaryReader, BinaryWriter, read_segment_meta, write_segment_meta};
use garuda_types::{
    DenseVector, DistanceMetric, InternalDocId, SegmentMeta, Status, StatusCode, VectorDimension,
    VectorFieldSchema,
};
use garuda_types::{HnswGraph, HnswLevel, HnswNeighborLimits};
const SEGMENT_MAGIC: &[u8; 8] = b"GRDSEG01";
const FLAT_INDEX_MAGIC: &[u8; 8] = b"GRDFLT01";
const HNSW_INDEX_MAGIC: &[u8; 8] = b"GRDHNS01";
const FORMAT_VERSION: u16 = 1;
pub fn encode_segment(meta: &SegmentMeta, records: &[StoredRecord]) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(SEGMENT_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    write_segment_meta(&mut writer, meta)?;
    writer.write_len(records.len())?;

    for record in records {
        writer.write_u64(record.doc_id.get());
        writer.write_u8(record.state.to_tag());
        write_doc(&mut writer, &record.doc)?;
    }

    Ok(writer.finish())
}

pub fn encode_empty_segment(meta: &SegmentMeta) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(SEGMENT_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    write_segment_meta(&mut writer, meta)?;
    writer.write_len(0)?;
    Ok(writer.finish())
}

pub struct DecodedSegment {
    pub meta: SegmentMeta,
    pub records: Vec<StoredRecord>,
}

pub fn decode_segment(bytes: &[u8]) -> Result<DecodedSegment, Status> {
    let mut reader = BinaryReader::new(bytes, SEGMENT_MAGIC, "segment")?;
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

    Ok(DecodedSegment { meta, records })
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
    let mut reader = BinaryReader::new(bytes, FLAT_INDEX_MAGIC, "segment")?;
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

    if !vector_field.indexes.has_flat() {
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
    let mut reader = BinaryReader::new(bytes, HNSW_INDEX_MAGIC, "segment")?;
    reader.expect_u16(FORMAT_VERSION)?;

    if !vector_field.indexes.has_hnsw() {
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

    let params = vector_field
        .indexes
        .hnsw_params()
        .expect("validated hnsw vector field");

    HnswGraph::from_parts(
        node_levels,
        levels,
        entry_count,
        HnswNeighborLimits::new(params.max_neighbors),
    )
}
