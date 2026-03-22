use crate::{DocId, FieldName, HnswEfSearch, Status, StatusCode, TopK};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    pub vector: DenseVector,
    pub score: Option<f32>,
}

impl Doc {
    pub fn new(id: DocId, fields: BTreeMap<String, ScalarValue>, vector: DenseVector) -> Self {
        Self {
            id,
            fields,
            vector,
            score: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DenseVector(Vec<f32>);

impl DenseVector {
    pub fn parse(values: Vec<f32>) -> Result<Self, Status> {
        if values.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "vector cannot be empty",
            ));
        }

        if values.iter().any(|value| !value.is_finite()) {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "vector values must be finite",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorProjection {
    Include,
    Exclude,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorQuery {
    pub field_name: FieldName,
    pub source: QueryVectorSource,
    pub top_k: TopK,
    pub filter: Option<String>,
    pub vector_projection: VectorProjection,
    pub output_fields: Option<Vec<String>>,
    pub ef_search: Option<HnswEfSearch>,
}

impl VectorQuery {
    pub fn by_vector(field_name: FieldName, vector: DenseVector, top_k: TopK) -> Self {
        Self {
            field_name,
            source: QueryVectorSource::Vector(vector),
            top_k,
            filter: None,
            vector_projection: VectorProjection::Exclude,
            output_fields: None,
            ef_search: None,
        }
    }

    pub fn by_id(field_name: FieldName, id: DocId, top_k: TopK) -> Self {
        Self {
            field_name,
            source: QueryVectorSource::DocumentId(id),
            top_k,
            filter: None,
            vector_projection: VectorProjection::Exclude,
            output_fields: None,
            ef_search: None,
        }
    }
}
