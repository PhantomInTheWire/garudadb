use crate::codec::{checksum, decode_doc_payload, encode_doc_payload};
use garuda_storage::{read_file, segment_wal_path, write_file_atomically};
use garuda_types::{Doc, DocId, Status, StatusCode};

const WAL_MAGIC: &[u8; 8] = b"GRDWAL01";
const FORMAT_VERSION: u16 = 1;
const INSERT_OP_TAG: u8 = 0;
const UPSERT_OP_TAG: u8 = 1;
const UPDATE_OP_TAG: u8 = 2;
const DELETE_OP_TAG: u8 = 3;

#[derive(Clone, Debug)]
pub enum WalOp {
    Insert(Doc),
    Upsert(Doc),
    Update(Doc),
    Delete(DocId),
}

pub fn append_wal_ops(
    root: &std::path::Path,
    segment_id: u64,
    ops: &[WalOp],
) -> Result<(), Status> {
    let wal_path = segment_wal_path(root, segment_id);
    let mut bytes = if wal_path.exists() {
        read_file(&wal_path)?
    } else {
        new_wal_bytes()
    };

    if ops.is_empty() {
        write_file_atomically(&wal_path, &bytes)?;
        return Ok(());
    }

    validate_header(&bytes)?;
    bytes.truncate(bytes.len() - std::mem::size_of::<u32>());

    for op in ops {
        append_op(&mut bytes, op)?;
    }

    let checksum = checksum(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    write_file_atomically(&wal_path, &bytes)
}

pub fn read_wal_ops(root: &std::path::Path, segment_id: u64) -> Result<Vec<WalOp>, Status> {
    let wal_path = segment_wal_path(root, segment_id);
    if !wal_path.exists() {
        return Ok(Vec::new());
    }

    let bytes = read_file(&wal_path)?;
    validate_header(&bytes)?;

    let payload_len = bytes.len() - std::mem::size_of::<u32>();
    let mut cursor = WAL_MAGIC.len() + std::mem::size_of::<u16>();
    let mut ops = Vec::new();

    while cursor < payload_len {
        let tag = bytes[cursor];
        cursor += 1;

        let header_remaining = payload_len.saturating_sub(cursor);
        if header_remaining < 8 {
            return Err(Status::err(
                StatusCode::Internal,
                "wal entry header is truncated",
            ));
        }

        let payload_size =
            u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().expect("fixed length"))
                as usize;
        cursor += 8;

        let Some(end) = cursor.checked_add(payload_size) else {
            return Err(Status::err(
                StatusCode::Internal,
                "wal entry exceeds file payload",
            ));
        };

        if end > payload_len {
            return Err(Status::err(
                StatusCode::Internal,
                "wal entry exceeds file payload",
            ));
        }

        let payload = &bytes[cursor..end];
        cursor = end;

        ops.push(match tag {
            INSERT_OP_TAG => WalOp::Insert(decode_doc_payload(payload)?),
            UPSERT_OP_TAG => WalOp::Upsert(decode_doc_payload(payload)?),
            UPDATE_OP_TAG => WalOp::Update(decode_doc_payload(payload)?),
            DELETE_OP_TAG => WalOp::Delete(DocId::parse(read_string_payload(payload)?)?),
            _ => {
                return Err(Status::err(
                    StatusCode::Internal,
                    "unrecognized wal operation tag",
                ));
            }
        });
    }

    Ok(ops)
}

pub fn reset_wal(root: &std::path::Path, segment_id: u64) -> Result<(), Status> {
    write_file_atomically(&segment_wal_path(root, segment_id), &new_wal_bytes())
}

fn append_op(bytes: &mut Vec<u8>, op: &WalOp) -> Result<(), Status> {
    match op {
        WalOp::Insert(doc) => append_payload(bytes, INSERT_OP_TAG, &encode_doc_payload(doc)?),
        WalOp::Upsert(doc) => append_payload(bytes, UPSERT_OP_TAG, &encode_doc_payload(doc)?),
        WalOp::Update(doc) => append_payload(bytes, UPDATE_OP_TAG, &encode_doc_payload(doc)?),
        WalOp::Delete(doc_id) => append_payload(bytes, DELETE_OP_TAG, doc_id.as_str().as_bytes()),
    }

    Ok(())
}

fn append_payload(bytes: &mut Vec<u8>, tag: u8, payload: &[u8]) {
    bytes.push(tag);
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
}

fn new_wal_bytes() -> Vec<u8> {
    let mut bytes = WAL_MAGIC.to_vec();
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    let checksum = checksum(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    bytes
}

fn validate_header(bytes: &[u8]) -> Result<(), Status> {
    if bytes.len() < WAL_MAGIC.len() + std::mem::size_of::<u16>() + std::mem::size_of::<u32>() {
        return Err(Status::err(StatusCode::Internal, "wal file is too short"));
    }

    if &bytes[..WAL_MAGIC.len()] != WAL_MAGIC {
        return Err(Status::err(
            StatusCode::Internal,
            "unexpected wal file magic",
        ));
    }

    let version_offset = WAL_MAGIC.len();
    let version_end = version_offset + std::mem::size_of::<u16>();
    let version = u16::from_le_bytes(
        bytes[version_offset..version_end]
            .try_into()
            .expect("fixed length"),
    );

    if version != FORMAT_VERSION {
        return Err(Status::err(
            StatusCode::Internal,
            "unsupported wal format version",
        ));
    }

    let payload_len = bytes.len() - std::mem::size_of::<u32>();
    let expected_checksum = u32::from_le_bytes(
        bytes[payload_len..]
            .try_into()
            .expect("checksum length is fixed"),
    );
    let actual_checksum = checksum(&bytes[..payload_len]);

    if expected_checksum == actual_checksum {
        return Ok(());
    }

    Err(Status::err(StatusCode::Internal, "wal checksum mismatch"))
}

fn read_string_payload(payload: &[u8]) -> Result<String, Status> {
    String::from_utf8(payload.to_vec())
        .map_err(|error| Status::err(StatusCode::Internal, format!("invalid utf-8: {error}")))
}
