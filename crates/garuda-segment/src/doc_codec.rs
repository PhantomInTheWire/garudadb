use garuda_storage::{BinaryReader, BinaryWriter, read_scalar_value, write_scalar_value};
use garuda_types::{DenseVector, Doc, DocId, Status};

const DOC_MAGIC: &[u8; 8] = b"GRDDOC01";
const FORMAT_VERSION: u16 = 1;

pub fn encode_doc_payload(doc: &Doc) -> Result<Vec<u8>, Status> {
    let mut writer = BinaryWriter::new(DOC_MAGIC);
    writer.write_u16(FORMAT_VERSION);
    write_doc(&mut writer, doc)?;
    Ok(writer.finish())
}

pub fn decode_doc_payload(bytes: &[u8]) -> Result<Doc, Status> {
    let mut reader = BinaryReader::new(bytes, DOC_MAGIC, "segment")?;
    reader.expect_u16(FORMAT_VERSION)?;
    let doc = read_doc(&mut reader)?;
    reader.finish()?;
    Ok(doc)
}

pub(crate) fn write_doc(writer: &mut BinaryWriter, doc: &Doc) -> Result<(), Status> {
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

pub(crate) fn read_doc(reader: &mut BinaryReader<'_>) -> Result<Doc, Status> {
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
