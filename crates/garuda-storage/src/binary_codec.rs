use garuda_types::{InternalDocId, ScalarValue, SegmentId, SegmentMeta, Status, StatusCode};

const FNV_OFFSET_BASIS: u32 = 2_166_136_261;
const FNV_PRIME: u32 = 16_777_619;

pub struct BinaryWriter {
    bytes: Vec<u8>,
}

impl BinaryWriter {
    pub fn new(magic: &[u8; 8]) -> Self {
        let mut bytes = Vec::with_capacity(magic.len() + std::mem::size_of::<u32>());
        bytes.extend_from_slice(magic);
        Self { bytes }
    }

    pub fn finish(mut self) -> Vec<u8> {
        let checksum = checksum(&self.bytes);
        self.bytes.extend_from_slice(&checksum.to_le_bytes());
        self.bytes
    }

    pub fn write_bool(&mut self, value: bool) {
        self.bytes.push(u8::from(value));
    }

    pub fn write_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub fn write_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_f32(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_f64(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_len(&mut self, value: usize) -> Result<(), Status> {
        let value = u64::try_from(value)
            .map_err(|_| Status::err(StatusCode::Internal, "value exceeds supported range"))?;
        self.write_u64(value);
        Ok(())
    }

    pub fn write_string(&mut self, value: &str) -> Result<(), Status> {
        self.write_len(value.len())?;
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    pub fn write_optional_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.write_bool(true);
                self.write_u64(value);
            }
            None => self.write_bool(false),
        }
    }

    pub fn write_optional_internal_doc_id(&mut self, value: Option<InternalDocId>) {
        self.write_optional_u64(value.map(InternalDocId::get));
    }

    pub fn write_optional_f32(&mut self, value: Option<f32>) {
        match value {
            Some(value) => {
                self.write_bool(true);
                self.write_f32(value);
            }
            None => self.write_bool(false),
        }
    }
}

pub struct BinaryReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
    payload_len: usize,
    file_kind: &'static str,
}

impl<'a> BinaryReader<'a> {
    pub fn new(bytes: &'a [u8], magic: &[u8; 8], file_kind: &'static str) -> Result<Self, Status> {
        if bytes.len() < magic.len() + std::mem::size_of::<u32>() {
            return Err(Status::err(
                StatusCode::Internal,
                format!("{file_kind} file is too short"),
            ));
        }

        if &bytes[..magic.len()] != magic {
            return Err(Status::err(
                StatusCode::Internal,
                format!("unexpected {file_kind} file magic"),
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
                format!("{file_kind} checksum mismatch"),
            ));
        }

        Ok(Self {
            bytes,
            cursor: magic.len(),
            payload_len,
            file_kind,
        })
    }

    pub fn finish(&self) -> Result<(), Status> {
        if self.cursor == self.payload_len {
            return Ok(());
        }

        Err(Status::err(
            StatusCode::Internal,
            format!("{} file contains trailing bytes", self.file_kind),
        ))
    }

    pub fn expect_u16(&mut self, expected: u16) -> Result<(), Status> {
        let actual = self.read_u16()?;
        if actual == expected {
            return Ok(());
        }

        Err(Status::err(
            StatusCode::Internal,
            format!("unsupported {} format version", self.file_kind),
        ))
    }

    pub fn read_bool(&mut self) -> Result<bool, Status> {
        Ok(self.read_u8()? != 0)
    }

    pub fn read_u8(&mut self) -> Result<u8, Status> {
        let bytes = self.read_exact(1)?;
        Ok(bytes[0])
    }

    pub fn read_u16(&mut self) -> Result<u16, Status> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    pub fn read_u64(&mut self) -> Result<u64, Status> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    pub fn read_i64(&mut self) -> Result<i64, Status> {
        let bytes = self.read_exact(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    pub fn read_f32(&mut self) -> Result<f32, Status> {
        let bytes = self.read_exact(4)?;
        Ok(f32::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    pub fn read_f64(&mut self) -> Result<f64, Status> {
        let bytes = self.read_exact(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    pub fn read_len(&mut self) -> Result<usize, Status> {
        let value = self.read_u64()?;
        usize::try_from(value)
            .map_err(|_| Status::err(StatusCode::Internal, "value exceeds supported range"))
    }

    pub fn read_string(&mut self) -> Result<String, Status> {
        let len = self.read_len()?;
        let bytes = self.read_exact(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|error| Status::err(StatusCode::Internal, format!("invalid utf-8: {error}")))
    }

    pub fn read_optional_u64(&mut self) -> Result<Option<u64>, Status> {
        if !self.read_bool()? {
            return Ok(None);
        }

        Ok(Some(self.read_u64()?))
    }

    pub fn read_optional_internal_doc_id(&mut self) -> Result<Option<InternalDocId>, Status> {
        let Some(value) = self.read_optional_u64()? else {
            return Ok(None);
        };

        Ok(Some(InternalDocId::new(value)?))
    }

    pub fn read_optional_f32(&mut self) -> Result<Option<f32>, Status> {
        if !self.read_bool()? {
            return Ok(None);
        }

        Ok(Some(self.read_f32()?))
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], Status> {
        let end = self.cursor.checked_add(len).ok_or_else(|| {
            Status::err(
                StatusCode::Internal,
                format!("{} read overflow", self.file_kind),
            )
        })?;

        if end > self.payload_len {
            return Err(Status::err(
                StatusCode::Internal,
                format!("{} file ended unexpectedly", self.file_kind),
            ));
        }

        let bytes = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(bytes)
    }
}

pub fn write_segment_meta(writer: &mut BinaryWriter, meta: &SegmentMeta) -> Result<(), Status> {
    writer.write_u64(meta.id.get());
    writer.write_string(&meta.path)?;
    writer.write_optional_internal_doc_id(meta.min_doc_id);
    writer.write_optional_internal_doc_id(meta.max_doc_id);
    writer.write_u64(meta.doc_count as u64);
    Ok(())
}

pub fn read_segment_meta(reader: &mut BinaryReader<'_>) -> Result<SegmentMeta, Status> {
    Ok(SegmentMeta {
        id: SegmentId::new_unchecked(reader.read_u64()?),
        path: reader.read_string()?,
        min_doc_id: reader.read_optional_internal_doc_id()?,
        max_doc_id: reader.read_optional_internal_doc_id()?,
        doc_count: reader.read_u64()? as usize,
    })
}

pub fn write_scalar_value(writer: &mut BinaryWriter, value: &ScalarValue) -> Result<(), Status> {
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

pub fn read_scalar_value(reader: &mut BinaryReader<'_>) -> Result<ScalarValue, Status> {
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

pub fn checksum(bytes: &[u8]) -> u32 {
    let mut hash = FNV_OFFSET_BASIS;

    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}
