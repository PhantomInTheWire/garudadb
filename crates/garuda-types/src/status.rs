use crate::{CollectionOptions, CollectionSchema, DocId, InternalDocId, SegmentId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatusCode {
    Ok,
    InvalidArgument,
    NotFound,
    AlreadyExists,
    FailedPrecondition,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Status {
    pub code: StatusCode,
    pub message: String,
}

impl Status {
    pub fn ok() -> Self {
        Self {
            code: StatusCode::Ok,
            message: String::new(),
        }
    }

    pub fn err(code: StatusCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn is_ok(&self) -> bool {
        self.code == StatusCode::Ok
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WriteResult {
    pub id: DocId,
    pub status: Status,
}

impl WriteResult {
    pub fn ok(id: DocId) -> Self {
        Self {
            id,
            status: Status::ok(),
        }
    }

    pub fn err(id: DocId, code: StatusCode, message: impl Into<String>) -> Self {
        Self {
            id,
            status: Status::err(code, message),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CollectionStats {
    pub doc_count: usize,
    pub segment_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub id: SegmentId,
    pub path: String,
    pub min_doc_id: Option<InternalDocId>,
    pub max_doc_id: Option<InternalDocId>,
    pub doc_count: usize,
}

impl SegmentMeta {
    pub fn contains_doc_id(&self, doc_id: InternalDocId) -> bool {
        let Some(min_doc_id) = self.min_doc_id else {
            return false;
        };
        let Some(max_doc_id) = self.max_doc_id else {
            return false;
        };

        min_doc_id <= doc_id && doc_id <= max_doc_id
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: CollectionSchema,
    pub options: CollectionOptions,
    pub next_doc_id: InternalDocId,
    pub next_segment_id: SegmentId,
    pub id_map_snapshot_id: crate::SnapshotId,
    pub delete_snapshot_id: crate::SnapshotId,
    pub manifest_version_id: crate::ManifestVersionId,
    pub writing_segment: SegmentMeta,
    pub persisted_segments: Vec<SegmentMeta>,
}
