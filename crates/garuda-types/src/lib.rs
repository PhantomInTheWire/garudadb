use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CollectionName(String);

impl CollectionName {
    pub fn parse(value: impl Into<String>) -> Result<Self, Status> {
        let value = value.into();

        if value.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "collection name cannot be empty",
            ));
        }

        let valid = value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-');

        if !valid {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "collection name may only use letters, numbers, '_' and '-'",
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CollectionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Borrow<str> for CollectionName {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FieldName(String);

impl FieldName {
    pub fn parse(value: impl Into<String>) -> Result<Self, Status> {
        let value = value.into();

        if value.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name cannot be empty",
            ));
        }

        let mut chars = value.chars();
        let Some(first) = chars.next() else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name cannot be empty",
            ));
        };

        if !first.is_ascii_alphabetic() && first != '_' {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name must start with a letter or '_'",
            ));
        }

        let valid = chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
        if !valid {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name may only use letters, numbers, and '_'",
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FieldName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Borrow<str> for FieldName {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DocId(String);

impl DocId {
    pub fn parse(value: impl Into<String>) -> Result<Self, Status> {
        let value = value.into();

        if value.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "document id cannot be empty",
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Borrow<str> for DocId {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

pub type InternalDocId = u64;
pub type SegmentId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ManifestVersionId(u64);

impl ManifestVersionId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SnapshotId(u64);

impl SnapshotId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StorageFormatVersion(u16);

impl StorageFormatVersion {
    pub fn new(value: u16) -> Self {
        Self(value)
    }

    pub fn get(self) -> u16 {
        self.0
    }
}

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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OptimizeOptions;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarFieldSchema {
    pub name: FieldName,
    pub field_type: ScalarType,
    pub nullable: bool,
    pub default_value: Option<ScalarValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorFieldSchema {
    pub name: FieldName,
    pub dimension: usize,
    pub metric: DistanceMetric,
    pub index: IndexParams,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionSchema {
    pub name: CollectionName,
    pub primary_key: FieldName,
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
    pub id: DocId,
    pub fields: BTreeMap<String, ScalarValue>,
    pub vector: Vec<f32>,
    pub score: Option<f32>,
}

impl Doc {
    pub fn new(id: DocId, fields: BTreeMap<String, ScalarValue>, vector: DenseVector) -> Self {
        Self {
            id,
            fields,
            vector: vector.into_vec(),
            score: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseVector(Vec<f32>);

impl DenseVector {
    pub fn parse(values: Vec<f32>) -> Result<Self, Status> {
        if values.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "vector cannot be empty",
            ));
        }

        Ok(Self(values))
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.0
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
pub enum QueryVectorSource {
    Vector(DenseVector),
    DocumentId(DocId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorQuery {
    pub field_name: FieldName,
    pub source: QueryVectorSource,
    pub top_k: usize,
    pub filter: Option<String>,
    pub include_vector: bool,
    pub output_fields: Option<Vec<String>>,
    pub ef_search: Option<usize>,
}

impl VectorQuery {
    pub fn by_vector(field_name: FieldName, vector: DenseVector, top_k: usize) -> Self {
        Self {
            field_name,
            source: QueryVectorSource::Vector(vector),
            top_k,
            filter: None,
            include_vector: false,
            output_fields: None,
            ef_search: None,
        }
    }

    pub fn by_id(field_name: FieldName, id: DocId, top_k: usize) -> Self {
        Self {
            field_name,
            source: QueryVectorSource::DocumentId(id),
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: CollectionSchema,
    pub options: CollectionOptions,
    pub next_doc_id: InternalDocId,
    pub next_segment_id: SegmentId,
    pub id_map_snapshot_id: SnapshotId,
    pub delete_snapshot_id: SnapshotId,
    pub manifest_version_id: ManifestVersionId,
    pub writing_segment: SegmentMeta,
    pub persisted_segments: Vec<SegmentMeta>,
}
