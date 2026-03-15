use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type DocId = u64;
pub type SegmentId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    InnerProduct,
    L2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarType {
    Bool,
    Int64,
    Float64,
    String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKind {
    Flat,
    Hnsw,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatIndexParams;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswIndexParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswIndexParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexParams {
    Flat(FlatIndexParams),
    Hnsw(HnswIndexParams),
}

impl IndexParams {
    pub fn kind(&self) -> IndexKind {
        match self {
            Self::Flat(_) => IndexKind::Flat,
            Self::Hnsw(_) => IndexKind::Hnsw,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarFieldSchema {
    pub name: String,
    pub field_type: ScalarType,
    pub nullable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorFieldSchema {
    pub name: String,
    pub dimension: usize,
    pub metric: DistanceMetric,
    pub index: IndexParams,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionSchema {
    pub name: String,
    pub primary_key: String,
    pub fields: Vec<ScalarFieldSchema>,
    pub vector: VectorFieldSchema,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionOptions {
    pub read_only: bool,
    pub enable_mmap: bool,
    pub segment_max_docs: usize,
}

impl Default for CollectionOptions {
    fn default() -> Self {
        Self {
            read_only: false,
            enable_mmap: true,
            segment_max_docs: 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalarValue {
    Bool(bool),
    Int64(i64),
    Float64(f64),
    String(String),
    Null,
}

impl ScalarValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::Int64(_) => "int64",
            Self::Float64(_) => "float64",
            Self::String(_) => "string",
            Self::Null => "null",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Doc {
    pub id: String,
    pub fields: BTreeMap<String, ScalarValue>,
    pub vector: Vec<f32>,
    pub score: Option<f32>,
}

impl Doc {
    pub fn new(
        id: impl Into<String>,
        fields: BTreeMap<String, ScalarValue>,
        vector: Vec<f32>,
    ) -> Self {
        Self {
            id: id.into(),
            fields,
            vector,
            score: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterExpr {
    Eq(String, ScalarValue),
    Ne(String, ScalarValue),
    Gt(String, ScalarValue),
    Gte(String, ScalarValue),
    Lt(String, ScalarValue),
    Lte(String, ScalarValue),
    And(Box<FilterExpr>, Box<FilterExpr>),
    Or(Box<FilterExpr>, Box<FilterExpr>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorQuery {
    pub field_name: String,
    pub vector: Option<Vec<f32>>,
    pub id: Option<String>,
    pub top_k: usize,
    pub filter: Option<String>,
    pub include_vector: bool,
    pub output_fields: Option<Vec<String>>,
    pub ef_search: Option<usize>,
}

impl VectorQuery {
    pub fn by_vector(field_name: impl Into<String>, vector: Vec<f32>, top_k: usize) -> Self {
        Self {
            field_name: field_name.into(),
            vector: Some(vector),
            id: None,
            top_k,
            filter: None,
            include_vector: false,
            output_fields: None,
            ef_search: None,
        }
    }
}

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
    pub id: String,
    pub status: Status,
}

impl WriteResult {
    pub fn ok(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            status: Status::ok(),
        }
    }

    pub fn err(id: impl Into<String>, code: StatusCode, message: impl Into<String>) -> Self {
        Self {
            id: id.into(),
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
    pub min_doc_id: Option<DocId>,
    pub max_doc_id: Option<DocId>,
    pub doc_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: CollectionSchema,
    pub options: CollectionOptions,
    pub next_doc_id: DocId,
    pub next_segment_id: SegmentId,
    pub writing_segment: SegmentMeta,
    pub persisted_segments: Vec<SegmentMeta>,
}
