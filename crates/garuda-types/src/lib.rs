mod hnsw;
mod ids;
mod ivf;
mod query;
mod schema;
mod status;

pub use hnsw::*;
pub use ids::*;
pub use ivf::*;
pub use query::*;
pub use schema::*;
pub use status::*;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VectorDimension(usize);

impl VectorDimension {
    pub fn new(value: usize) -> Result<Self, Status> {
        if value == 0 {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "vector dimension must be greater than zero",
            ));
        }

        Ok(Self(value))
    }

    pub fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TopK(usize);

impl TopK {
    pub fn new(value: usize) -> Result<Self, Status> {
        if value == 0 {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "top_k must be greater than zero",
            ));
        }

        Ok(Self(value))
    }

    pub fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    InnerProduct,
    L2,
}

impl DistanceMetric {
    pub fn to_tag(self) -> u8 {
        match self {
            Self::Cosine => 0,
            Self::InnerProduct => 1,
            Self::L2 => 2,
        }
    }

    pub fn from_tag(tag: u8) -> Result<Self, Status> {
        match tag {
            0 => Ok(Self::Cosine),
            1 => Ok(Self::InnerProduct),
            2 => Ok(Self::L2),
            _ => Err(Status::err(StatusCode::Internal, "unrecognized metric tag")),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OptimizeOptions;
