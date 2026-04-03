use garuda_types::{
    CollectionName, CollectionOptions, CollectionSchema, DistanceMetric, FieldName,
    HnswIndexParams, IndexKind, IvfIndexParams, Nullability, ScalarIndexState, ScalarType,
    ScalarValue, Status, StorageAccess, VectorSearch,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::num::NonZeroU32;
use std::path::Path;

use crate::cli::{
    CreateIndexKind, IndexKindArg, MetricArg, NullabilityArg, QuerySearch, ScalarIndexStateArg,
    ScalarTypeArg, StorageAccessArg,
};

#[derive(Deserialize)]
struct JsonDoc {
    id: String,
    fields: BTreeMap<String, serde_json::Value>,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct CreateCollectionFile {
    schema: CollectionSchema,
    options: CollectionOptions,
}

impl From<MetricArg> for DistanceMetric {
    fn from(value: MetricArg) -> Self {
        match value {
            MetricArg::Cosine => DistanceMetric::Cosine,
            MetricArg::InnerProduct => DistanceMetric::InnerProduct,
            MetricArg::L2 => DistanceMetric::L2,
        }
    }
}

impl From<StorageAccessArg> for StorageAccess {
    fn from(value: StorageAccessArg) -> Self {
        match value {
            StorageAccessArg::StandardIo => StorageAccess::StandardIo,
            StorageAccessArg::MmapPreferred => StorageAccess::MmapPreferred,
        }
    }
}

impl From<ScalarTypeArg> for ScalarType {
    fn from(value: ScalarTypeArg) -> Self {
        match value {
            ScalarTypeArg::Bool => ScalarType::Bool,
            ScalarTypeArg::Int64 => ScalarType::Int64,
            ScalarTypeArg::Float64 => ScalarType::Float64,
            ScalarTypeArg::String => ScalarType::String,
        }
    }
}

impl From<NullabilityArg> for Nullability {
    fn from(value: NullabilityArg) -> Self {
        match value {
            NullabilityArg::Required => Nullability::Required,
            NullabilityArg::Nullable => Nullability::Nullable,
        }
    }
}

impl From<ScalarIndexStateArg> for ScalarIndexState {
    fn from(value: ScalarIndexStateArg) -> Self {
        match value {
            ScalarIndexStateArg::None => ScalarIndexState::None,
            ScalarIndexStateArg::Indexed => ScalarIndexState::Indexed,
        }
    }
}

impl From<IndexKindArg> for IndexKind {
    fn from(value: IndexKindArg) -> Self {
        match value {
            IndexKindArg::Flat => IndexKind::Flat,
            IndexKindArg::Hnsw => IndexKind::Hnsw,
            IndexKindArg::Ivf => IndexKind::Ivf,
            IndexKindArg::Scalar => IndexKind::Scalar,
        }
    }
}

pub fn print_json(value: &impl serde::Serialize) -> Result<(), String> {
    let output = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    println!("{output}");
    Ok(())
}

pub fn parse_collection_name(value: &str) -> Result<CollectionName, String> {
    CollectionName::parse(value).map_err(|status| status.message)
}

pub fn parse_doc_id(value: String) -> Result<garuda_types::DocId, String> {
    garuda_types::DocId::parse(value).map_err(|status| status.message)
}

pub fn parse_field_name(value: String) -> Result<FieldName, String> {
    FieldName::parse(value).map_err(|status| status.message)
}

pub fn field_name(value: &str) -> FieldName {
    FieldName::parse(value).expect("hardcoded field name should be valid")
}

pub fn read_collection_file(path: &Path) -> Result<(CollectionSchema, CollectionOptions), String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|error| error.to_string())?;
    let file: CreateCollectionFile =
        serde_json::from_str(&contents).map_err(|error| error.to_string())?;
    Ok((file.schema, file.options))
}

pub fn read_jsonl_docs(path: &Path) -> Result<Vec<garuda_types::Doc>, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let reader = BufReader::new(file);
    let mut docs = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|error| error.to_string())?;
        if line.trim().is_empty() {
            continue;
        }

        let doc: JsonDoc = serde_json::from_str(&line).map_err(|error| error.to_string())?;
        docs.push(garuda_types::Doc::new(
            parse_doc_id(doc.id)?,
            parse_scalar_fields(doc.fields)?,
            garuda_types::DenseVector::parse(doc.vector).map_err(|status| status.message)?,
        ));
    }

    Ok(docs)
}

pub fn parse_vector_arg(raw: &str) -> Result<garuda_types::DenseVector, String> {
    let values = raw
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f32>()
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    garuda_types::DenseVector::parse(values).map_err(|status| status.message)
}

pub fn parse_query_search(search: Option<QuerySearch>) -> Result<VectorSearch, String> {
    match search {
        None => Ok(VectorSearch::Default),
        Some(QuerySearch::Hnsw { ef_search }) => Ok(VectorSearch::Hnsw { ef_search }),
        Some(QuerySearch::Ivf { nprobe }) => Ok(VectorSearch::Ivf { nprobe }),
    }
}

pub fn parse_index_params(kind: CreateIndexKind) -> Result<garuda_types::IndexParams, String> {
    match kind {
        CreateIndexKind::Flat => Ok(garuda_types::IndexParams::Flat(
            garuda_types::FlatIndexParams,
        )),
        CreateIndexKind::Scalar => Ok(garuda_types::IndexParams::Scalar(
            garuda_types::ScalarIndexParams,
        )),
        CreateIndexKind::Hnsw(args) => Ok(garuda_types::IndexParams::Hnsw(HnswIndexParams {
            max_neighbors: args.max_neighbors,
            scaling_factor: args.scaling_factor,
            ef_construction: args.ef_construction,
            prune_width: args.prune_width,
            min_neighbor_count: args.min_neighbor_count,
            ef_search: args.ef_search,
        })),
        CreateIndexKind::Ivf(args) => Ok(garuda_types::IndexParams::Ivf(IvfIndexParams {
            n_list: args.n_list,
            n_probe: args.n_probe,
            training_iterations: args.training_iterations,
        })),
    }
}

pub fn parse_scalar_json_literal(raw: &str) -> Result<ScalarValue, String> {
    let value = serde_json::from_str(raw).map_err(|error| error.to_string())?;
    parse_scalar_value(value)
}

fn parse_scalar_fields(
    fields: BTreeMap<String, serde_json::Value>,
) -> Result<BTreeMap<String, ScalarValue>, String> {
    fields
        .into_iter()
        .map(|(name, value)| Ok((name, parse_scalar_value(value)?)))
        .collect()
}

fn parse_scalar_value(value: serde_json::Value) -> Result<ScalarValue, String> {
    match value {
        serde_json::Value::Null => Ok(ScalarValue::Null),
        serde_json::Value::Bool(value) => Ok(ScalarValue::Bool(value)),
        serde_json::Value::Number(value) => {
            if value.is_i64() {
                let value = value.as_i64().expect("i64 json number");
                return Ok(ScalarValue::Int64(value));
            }
            if value.is_u64() {
                return Err(String::from("unsigned integer literal exceeds i64 range"));
            }
            if value.is_f64() {
                let value = value.as_f64().expect("f64 json number");
                return Ok(ScalarValue::Float64(value));
            }
            Err(String::from("unsupported numeric value"))
        }
        serde_json::Value::String(value) => Ok(ScalarValue::String(value)),
        _ => Err(String::from("field values must be scalar")),
    }
}

pub fn parse_non_zero_u32<T>(
    value: &str,
    parse: impl FnOnce(u32) -> Result<T, Status>,
) -> Result<T, String> {
    let value = value
        .parse::<NonZeroU32>()
        .map_err(|error| error.to_string())?;
    parse(value.get()).map_err(|status| status.message)
}
