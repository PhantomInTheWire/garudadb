use crate::doc_codec::{read_doc, write_doc};
use crate::{RecordState, StoredRecord};
use garuda_index_flat::{FlatIndex, FlatIndexEntry};
use garuda_index_scalar::{ScalarIndex, ScalarIndexData};
use garuda_storage::{BinaryReader, BinaryWriter, read_segment_meta, write_segment_meta};
use garuda_types::{
    DenseVector, DistanceMetric, InternalDocId, NodeIndex, ScalarFieldSchema, SegmentMeta, Status,
    StatusCode, VectorDimension, VectorFieldSchema,
};
use garuda_types::{HnswGraph, HnswLevel, HnswNeighborLimits};
const SEGMENT_MAGIC: &[u8; 8] = b"GRDSEG01";
const FLAT_INDEX_MAGIC: &[u8; 8] = b"GRDFLT01";
const HNSW_INDEX_MAGIC: &[u8; 8] = b"GRDHNS01";
const SCALAR_INDEX_MAGIC: &[u8; 8] = b"GRDSCL01";
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
            let node = NodeIndex::new(raw_node);
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
                neighbors.push(NodeIndex::new(reader.read_len()?));
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

pub fn encode_scalar_index(index: &ScalarIndex, entry_count: usize) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(SCALAR_INDEX_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    writer.write_len(entry_count)?;

    match index.data() {
        ScalarIndexData::Bool {
            false_doc_ids,
            true_doc_ids,
        } => {
            writer.write_u8(0);
            writer.write_len(false_doc_ids.len())?;
            for doc_id in false_doc_ids {
                writer.write_u64(doc_id.get());
            }

            writer.write_len(true_doc_ids.len())?;
            for doc_id in true_doc_ids {
                writer.write_u64(doc_id.get());
            }
        }
        ScalarIndexData::Int64(postings) => {
            writer.write_u8(1);
            writer.write_len(postings.len())?;

            for (value, doc_ids) in postings {
                writer.write_i64(value);
                write_posting_list(&mut writer, &doc_ids)?;
            }
        }
        ScalarIndexData::Float64(postings) => {
            writer.write_u8(2);
            writer.write_len(postings.len())?;

            for (value, doc_ids) in postings {
                writer.write_f64(value);
                write_posting_list(&mut writer, &doc_ids)?;
            }
        }
        ScalarIndexData::String(postings) => {
            writer.write_u8(3);
            writer.write_len(postings.len())?;

            for (value, doc_ids) in postings {
                writer.write_string(&value)?;
                write_posting_list(&mut writer, &doc_ids)?;
            }
        }
    }

    Ok(writer.finish())
}

pub fn decode_scalar_index(
    bytes: &[u8],
    field: &ScalarFieldSchema,
    entry_count: usize,
) -> Result<ScalarIndex, Status> {
    let mut reader = BinaryReader::new(bytes, SCALAR_INDEX_MAGIC, "segment")?;
    reader.expect_u16(FORMAT_VERSION)?;

    if reader.read_len()? != entry_count {
        return Err(Status::err(
            StatusCode::Internal,
            "persisted scalar index does not match segment live doc count",
        ));
    }

    let index = match reader.read_u8()? {
        0 => ScalarIndex::from_data(ScalarIndexData::Bool {
            false_doc_ids: read_posting_list(&mut reader)?,
            true_doc_ids: read_posting_list(&mut reader)?,
        }),
        1 => ScalarIndex::from_data(ScalarIndexData::Int64(read_i64_postings(&mut reader)?)),
        2 => ScalarIndex::from_data(ScalarIndexData::Float64(read_f64_postings(&mut reader)?)),
        3 => ScalarIndex::from_data(ScalarIndexData::String(read_string_postings(&mut reader)?)),
        _ => {
            return Err(Status::err(
                StatusCode::Internal,
                "unrecognized scalar index tag",
            ));
        }
    };

    reader.finish()?;

    let expected = ScalarIndex::new(field.field_type);
    if std::mem::discriminant(&index) == std::mem::discriminant(&expected) {
        return Ok(index);
    }

    Err(Status::err(
        StatusCode::Internal,
        "persisted scalar index type does not match schema",
    ))
}

fn write_posting_list(writer: &mut BinaryWriter, doc_ids: &[InternalDocId]) -> Result<(), Status> {
    writer.write_len(doc_ids.len())?;

    for doc_id in doc_ids {
        writer.write_u64(doc_id.get());
    }

    Ok(())
}

fn read_posting_list(reader: &mut BinaryReader<'_>) -> Result<Vec<InternalDocId>, Status> {
    let posting_count = reader.read_len()?;
    let mut doc_ids = Vec::with_capacity(posting_count);

    for _ in 0..posting_count {
        doc_ids.push(InternalDocId::new(reader.read_u64()?)?);
    }

    Ok(doc_ids)
}

macro_rules! read_postings {
    ($reader:expr, $read_value:ident) => {{
        let posting_count = $reader.read_len()?;
        let mut postings = Vec::with_capacity(posting_count);

        for _ in 0..posting_count {
            postings.push(($reader.$read_value()?, read_posting_list($reader)?));
        }

        Ok(postings)
    }};
}

fn read_i64_postings(
    reader: &mut BinaryReader<'_>,
) -> Result<Vec<(i64, Vec<InternalDocId>)>, Status> {
    read_postings!(reader, read_i64)
}

fn read_f64_postings(
    reader: &mut BinaryReader<'_>,
) -> Result<Vec<(f64, Vec<InternalDocId>)>, Status> {
    read_postings!(reader, read_f64)
}

fn read_string_postings(
    reader: &mut BinaryReader<'_>,
) -> Result<Vec<(String, Vec<InternalDocId>)>, Status> {
    read_postings!(reader, read_string)
}
