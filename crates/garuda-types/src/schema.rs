use crate::{
    CollectionName, DistanceMetric, FieldName, HnswIndexParams, IvfIndexParams, Status, StatusCode,
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
    Ivf,
    Scalar,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatIndexParams;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarIndexParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarIndexState {
    None,
    Indexed,
}

impl ScalarIndexState {
    pub fn is_indexed(self) -> bool {
        matches!(self, Self::Indexed)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexParams {
    Flat(FlatIndexParams),
    Hnsw(HnswIndexParams),
    Ivf(IvfIndexParams),
    Scalar(ScalarIndexParams),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VectorIndexState {
    DefaultFlat,
    HnswOnly(HnswIndexParams),
    IvfOnly(IvfIndexParams),
    FlatAndHnsw {
        default: IndexKind,
        hnsw: HnswIndexParams,
    },
    FlatAndIvf {
        default: IndexKind,
        ivf: IvfIndexParams,
    },
}

impl VectorIndexState {
    pub fn default_kind(&self) -> IndexKind {
        match self {
            Self::DefaultFlat => IndexKind::Flat,
            Self::HnswOnly(_) => IndexKind::Hnsw,
            Self::IvfOnly(_) => IndexKind::Ivf,
            Self::FlatAndHnsw { default, .. } => *default,
            Self::FlatAndIvf { default, .. } => *default,
        }
    }

    pub fn has_flat(&self) -> bool {
        match self {
            Self::DefaultFlat | Self::FlatAndHnsw { .. } | Self::FlatAndIvf { .. } => true,
            Self::HnswOnly(_) | Self::IvfOnly(_) => false,
        }
    }

    pub fn has_hnsw(&self) -> bool {
        match self {
            Self::DefaultFlat => false,
            Self::HnswOnly(_) | Self::FlatAndHnsw { .. } => true,
            Self::IvfOnly(_) | Self::FlatAndIvf { .. } => false,
        }
    }

    pub fn has_ivf(&self) -> bool {
        match self {
            Self::DefaultFlat | Self::HnswOnly(_) | Self::FlatAndHnsw { .. } => false,
            Self::IvfOnly(_) | Self::FlatAndIvf { .. } => true,
        }
    }

    pub fn hnsw_params(&self) -> Option<&HnswIndexParams> {
        match self {
            Self::DefaultFlat => None,
            Self::HnswOnly(params) => Some(params),
            Self::FlatAndHnsw { hnsw, .. } => Some(hnsw),
            Self::IvfOnly(_) | Self::FlatAndIvf { .. } => None,
        }
    }

    pub fn ivf_params(&self) -> Option<&IvfIndexParams> {
        match self {
            Self::DefaultFlat | Self::HnswOnly(_) | Self::FlatAndHnsw { .. } => None,
            Self::IvfOnly(params) => Some(params),
            Self::FlatAndIvf { ivf, .. } => Some(ivf),
        }
    }

    pub fn enable(self, params: IndexParams) -> Result<Self, Status> {
        match params {
            IndexParams::Flat(_) => Ok(match self {
                Self::DefaultFlat => Self::DefaultFlat,
                Self::HnswOnly(hnsw) => Self::FlatAndHnsw {
                    default: IndexKind::Hnsw,
                    hnsw,
                },
                Self::IvfOnly(ivf) => Self::FlatAndIvf {
                    default: IndexKind::Ivf,
                    ivf,
                },
                Self::FlatAndHnsw { default, hnsw } => Self::FlatAndHnsw { default, hnsw },
                Self::FlatAndIvf { default, ivf } => Self::FlatAndIvf { default, ivf },
            }),
            IndexParams::Hnsw(hnsw) => Ok(match self {
                Self::DefaultFlat => Self::FlatAndHnsw {
                    default: IndexKind::Hnsw,
                    hnsw,
                },
                Self::HnswOnly(_) => Self::HnswOnly(hnsw),
                Self::FlatAndHnsw { .. } => Self::FlatAndHnsw {
                    default: IndexKind::Hnsw,
                    hnsw,
                },
                Self::IvfOnly(_) | Self::FlatAndIvf { .. } => {
                    return Err(Status::err(
                        StatusCode::InvalidArgument,
                        "cannot enable hnsw while ivf is enabled",
                    ));
                }
            }),
            IndexParams::Ivf(ivf) => Ok(match self {
                Self::DefaultFlat => Self::FlatAndIvf {
                    default: IndexKind::Ivf,
                    ivf,
                },
                Self::IvfOnly(_) => Self::IvfOnly(ivf),
                Self::FlatAndIvf { .. } => Self::FlatAndIvf {
                    default: IndexKind::Ivf,
                    ivf,
                },
                Self::HnswOnly(_) | Self::FlatAndHnsw { .. } => {
                    return Err(Status::err(
                        StatusCode::InvalidArgument,
                        "cannot enable ivf while hnsw is enabled",
                    ));
                }
            }),
            IndexParams::Scalar(_) => Err(Status::err(
                StatusCode::InvalidArgument,
                "cannot create a scalar index on the vector field",
            )),
        }
    }

    pub fn drop(self, kind: IndexKind) -> Result<Self, Status> {
        match (self, kind) {
            (Self::DefaultFlat, IndexKind::Flat) => Ok(Self::DefaultFlat),
            (Self::DefaultFlat, _) => Err(Status::err(
                StatusCode::InvalidArgument,
                "index kind is not enabled",
            )),
            (Self::HnswOnly(hnsw), IndexKind::Flat) => Ok(Self::HnswOnly(hnsw)),
            (Self::HnswOnly(_), IndexKind::Hnsw) => Ok(Self::DefaultFlat),
            (Self::HnswOnly(_), IndexKind::Ivf) => Err(Status::err(
                StatusCode::InvalidArgument,
                "index kind is not enabled",
            )),
            (Self::HnswOnly(_), IndexKind::Scalar) => Err(Status::err(
                StatusCode::InvalidArgument,
                "cannot drop a scalar index from the vector field",
            )),
            (Self::IvfOnly(ivf), IndexKind::Flat) => Ok(Self::IvfOnly(ivf)),
            (Self::IvfOnly(_), IndexKind::Ivf) => Ok(Self::DefaultFlat),
            (Self::IvfOnly(_), IndexKind::Hnsw) => Err(Status::err(
                StatusCode::InvalidArgument,
                "index kind is not enabled",
            )),
            (Self::IvfOnly(_), IndexKind::Scalar) => Err(Status::err(
                StatusCode::InvalidArgument,
                "cannot drop a scalar index from the vector field",
            )),
            (Self::FlatAndHnsw { hnsw, .. }, IndexKind::Flat) => Ok(Self::HnswOnly(hnsw)),
            (Self::FlatAndHnsw { .. }, IndexKind::Hnsw) => Ok(Self::DefaultFlat),
            (Self::FlatAndHnsw { .. }, IndexKind::Ivf) => Err(Status::err(
                StatusCode::InvalidArgument,
                "index kind is not enabled",
            )),
            (Self::FlatAndHnsw { .. }, IndexKind::Scalar) => Err(Status::err(
                StatusCode::InvalidArgument,
                "cannot drop a scalar index from the vector field",
            )),
            (Self::FlatAndIvf { ivf, .. }, IndexKind::Flat) => Ok(Self::IvfOnly(ivf)),
            (Self::FlatAndIvf { .. }, IndexKind::Ivf) => Ok(Self::DefaultFlat),
            (Self::FlatAndIvf { .. }, IndexKind::Hnsw) => Err(Status::err(
                StatusCode::InvalidArgument,
                "index kind is not enabled",
            )),
            (Self::FlatAndIvf { .. }, IndexKind::Scalar) => Err(Status::err(
                StatusCode::InvalidArgument,
                "cannot drop a scalar index from the vector field",
            )),
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
    pub index: ScalarIndexState,
    pub nullability: Nullability,
    pub default_value: Option<crate::ScalarValue>,
}

impl ScalarFieldSchema {
    pub fn is_indexed(&self) -> bool {
        self.index.is_indexed()
    }
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

impl CollectionSchema {
    pub fn scalar_field(&self, field_name: &FieldName) -> Option<&ScalarFieldSchema> {
        self.fields.iter().find(|field| field.name == *field_name)
    }

    pub fn scalar_field_by_name(&self, field_name: &str) -> Option<&ScalarFieldSchema> {
        self.fields
            .iter()
            .find(|field| field.name.as_str() == field_name)
    }

    pub fn scalar_field_mut(&mut self, field_name: &FieldName) -> Option<&mut ScalarFieldSchema> {
        self.fields
            .iter_mut()
            .find(|field| field.name == *field_name)
    }
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
