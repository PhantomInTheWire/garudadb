use crate::{
    CollectionName, DistanceMetric, FieldName, HnswIndexParams, Status, StatusCode,
    VectorDimension,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarType {
    Bool,
    Int64,
    Float64,
    String,
}

impl ScalarType {
    pub fn to_tag(self) -> u8 {
        match self {
            Self::Bool => 0,
            Self::Int64 => 1,
            Self::Float64 => 2,
            Self::String => 3,
        }
    }

    pub fn from_tag(tag: u8) -> Result<Self, Status> {
        match tag {
            0 => Ok(Self::Bool),
            1 => Ok(Self::Int64),
            2 => Ok(Self::Float64),
            3 => Ok(Self::String),
            _ => Err(Status::err(
                StatusCode::Internal,
                "unrecognized scalar type tag",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKind {
    Flat,
    Hnsw,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatIndexParams;

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
pub enum VectorIndexState {
    DefaultFlat,
    HnswOnly(HnswIndexParams),
    FlatAndHnsw {
        default: IndexKind,
        hnsw: HnswIndexParams,
    },
}

impl VectorIndexState {
    pub fn default_kind(&self) -> IndexKind {
        match self {
            Self::DefaultFlat => IndexKind::Flat,
            Self::HnswOnly(_) => IndexKind::Hnsw,
            Self::FlatAndHnsw { default, .. } => *default,
        }
    }

    pub fn has_flat(&self) -> bool {
        match self {
            Self::DefaultFlat | Self::FlatAndHnsw { .. } => true,
            Self::HnswOnly(_) => false,
        }
    }

    pub fn has_hnsw(&self) -> bool {
        match self {
            Self::DefaultFlat => false,
            Self::HnswOnly(_) | Self::FlatAndHnsw { .. } => true,
        }
    }

    pub fn hnsw_params(&self) -> Option<&HnswIndexParams> {
        match self {
            Self::DefaultFlat => None,
            Self::HnswOnly(params) => Some(params),
            Self::FlatAndHnsw { hnsw, .. } => Some(hnsw),
        }
    }

    pub fn enable(self, params: IndexParams) -> Self {
        match params {
            IndexParams::Flat(_) => match self {
                Self::DefaultFlat => Self::DefaultFlat,
                Self::HnswOnly(hnsw) => Self::FlatAndHnsw {
                    default: IndexKind::Hnsw,
                    hnsw,
                },
                Self::FlatAndHnsw { default, hnsw } => Self::FlatAndHnsw { default, hnsw },
            },
            IndexParams::Hnsw(hnsw) => match self {
                Self::DefaultFlat => Self::FlatAndHnsw {
                    default: IndexKind::Hnsw,
                    hnsw,
                },
                Self::HnswOnly(_) => Self::HnswOnly(hnsw),
                Self::FlatAndHnsw { .. } => Self::FlatAndHnsw {
                    default: IndexKind::Hnsw,
                    hnsw,
                },
            },
        }
    }

    pub fn drop(self, kind: IndexKind) -> Self {
        match (self, kind) {
            (Self::DefaultFlat, _) => Self::DefaultFlat,
            (Self::HnswOnly(hnsw), IndexKind::Flat) => Self::HnswOnly(hnsw),
            (Self::HnswOnly(_), IndexKind::Hnsw) => Self::DefaultFlat,
            (Self::FlatAndHnsw { hnsw, .. }, IndexKind::Flat) => Self::HnswOnly(hnsw),
            (Self::FlatAndHnsw { .. }, IndexKind::Hnsw) => Self::DefaultFlat,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Nullability {
    Nullable,
    Required,
}

impl Nullability {
    pub fn is_nullable(self) -> bool {
        matches!(self, Self::Nullable)
    }

    pub fn to_tag(self) -> u8 {
        match self {
            Self::Nullable => 0,
            Self::Required => 1,
        }
    }

    pub fn from_tag(tag: u8) -> Result<Self, Status> {
        match tag {
            0 => Ok(Self::Nullable),
            1 => Ok(Self::Required),
            _ => Err(Status::err(
                StatusCode::Internal,
                "unrecognized nullability tag",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarFieldSchema {
    pub name: FieldName,
    pub field_type: ScalarType,
    pub nullability: Nullability,
    pub default_value: Option<crate::ScalarValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessMode {
    ReadWrite,
    ReadOnly,
}

impl AccessMode {
    pub fn to_tag(self) -> u8 {
        match self {
            Self::ReadWrite => 0,
            Self::ReadOnly => 1,
        }
    }

    pub fn from_tag(tag: u8) -> Result<Self, Status> {
        match tag {
            0 => Ok(Self::ReadWrite),
            1 => Ok(Self::ReadOnly),
            _ => Err(Status::err(
                StatusCode::Internal,
                "unrecognized access mode tag",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageAccess {
    StandardIo,
    MmapPreferred,
}

impl StorageAccess {
    pub fn to_tag(self) -> u8 {
        match self {
            Self::StandardIo => 0,
            Self::MmapPreferred => 1,
        }
    }

    pub fn from_tag(tag: u8) -> Result<Self, Status> {
        match tag {
            0 => Ok(Self::StandardIo),
            1 => Ok(Self::MmapPreferred),
            _ => Err(Status::err(
                StatusCode::Internal,
                "unrecognized storage access tag",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorFieldSchema {
    pub name: FieldName,
    pub dimension: VectorDimension,
    pub metric: DistanceMetric,
    pub indexes: VectorIndexState,
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
    pub access_mode: AccessMode,
    pub storage_access: StorageAccess,
    pub segment_max_docs: usize,
}

impl Default for CollectionOptions {
    fn default() -> Self {
        Self {
            access_mode: AccessMode::ReadWrite,
            storage_access: StorageAccess::MmapPreferred,
            segment_max_docs: 1024,
        }
    }
}
