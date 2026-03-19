use clap::{Parser, Subcommand};
use garuda_engine::Database;
use garuda_types::{
    CollectionName, CollectionOptions, CollectionSchema, DistanceMetric, FieldName,
    FlatIndexParams, HnswIndexParams, IndexKind, IndexParams, Nullability, ScalarFieldSchema,
    ScalarType, ScalarValue, TopK, VectorDimension, VectorFieldSchema, VectorQuery,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::num::NonZeroUsize;
use std::path::PathBuf;

const PRIMARY_KEY_FIELD: &str = "pk";
const VECTOR_FIELD: &str = "embedding";

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Create {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        dimension: NonZeroUsize,
    },
    InsertJsonl {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        path: PathBuf,
    },
    Query {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        #[arg(long)]
        vector: String,
        #[arg(long)]
        top_k: usize,
    },
    Fetch {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        id: String,
    },
    CreateIndex {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        field: String,
        kind: String,
    },
    Stats {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
    },
}

#[derive(Deserialize)]
struct JsonDoc {
    id: String,
    fields: BTreeMap<String, serde_json::Value>,
    vector: Vec<f32>,
}

fn main() -> Result<(), String> {
    let Cli { root, command } = Cli::parse();
    let db = Database::open(&root).map_err(|status| status.message)?;

    match command {
        Commands::Init => {
            println!("{}", root.display());
            Ok(())
        }
        Commands::Create { name, dimension } => {
            let schema = CollectionSchema {
                name: name.clone(),
                primary_key: field_name(PRIMARY_KEY_FIELD),
                fields: vec![
                    ScalarFieldSchema {
                        name: field_name(PRIMARY_KEY_FIELD),
                        field_type: ScalarType::String,
                        nullability: Nullability::Required,
                        default_value: None,
                    },
                    ScalarFieldSchema {
                        name: field_name("rank"),
                        field_type: ScalarType::Int64,
                        nullability: Nullability::Required,
                        default_value: None,
                    },
                    ScalarFieldSchema {
                        name: field_name("category"),
                        field_type: ScalarType::String,
                        nullability: Nullability::Required,
                        default_value: None,
                    },
                    ScalarFieldSchema {
                        name: field_name("score"),
                        field_type: ScalarType::Float64,
                        nullability: Nullability::Required,
                        default_value: None,
                    },
                ],
                vector: VectorFieldSchema {
                    name: field_name(VECTOR_FIELD),
                    dimension: VectorDimension::new(dimension.get())
                        .expect("cli dimension is non-zero"),
                    metric: DistanceMetric::Cosine,
                    index: IndexParams::Flat(FlatIndexParams),
                },
            };

            db.create_collection(schema, CollectionOptions::default())
                .map_err(|status| status.message)?;
            println!("{name}");
            Ok(())
        }
        Commands::InsertJsonl { name, path } => {
            let collection = db.open_collection(&name).map_err(|status| status.message)?;
            let docs = read_jsonl_docs(&path)?;
            let results = collection.insert(docs);

            if results.iter().all(|result| result.status.is_ok()) {
                println!("{}", results.len());
                return Ok(());
            }

            let first_error = results
                .into_iter()
                .find(|result| !result.status.is_ok())
                .expect("insert should have a failure");
            Err(first_error.status.message)
        }
        Commands::Query {
            name,
            vector,
            top_k,
        } => {
            let collection = db.open_collection(&name).map_err(|status| status.message)?;
            let query = VectorQuery::by_vector(
                field_name(VECTOR_FIELD),
                parse_vector_arg(&vector)?,
                TopK::new(top_k).map_err(|status| status.message)?,
            );
            let docs = collection.query(query).map_err(|status| status.message)?;
            let output = serde_json::to_string_pretty(&docs).map_err(|error| error.to_string())?;
            println!("{output}");
            Ok(())
        }
        Commands::Fetch { name, id } => {
            let collection = db.open_collection(&name).map_err(|status| status.message)?;
            let docs =
                collection.fetch(vec![garuda_types::DocId::parse(id).map_err(|s| s.message)?]);
            let output = serde_json::to_string_pretty(&docs).map_err(|error| error.to_string())?;
            println!("{output}");
            Ok(())
        }
        Commands::CreateIndex { name, field, kind } => {
            let collection = db.open_collection(&name).map_err(|status| status.message)?;
            collection
                .create_index(
                    &FieldName::parse(field).map_err(|status| status.message)?,
                    index_params(parse_index_kind(&kind)?),
                )
                .map_err(|status| status.message)?;
            println!("ok");
            Ok(())
        }
        Commands::Stats { name } => {
            let collection = db.open_collection(&name).map_err(|status| status.message)?;
            let output = serde_json::to_string_pretty(&collection.stats())
                .map_err(|error| error.to_string())?;
            println!("{output}");
            Ok(())
        }
    }
}

fn parse_collection_name(value: &str) -> Result<CollectionName, String> {
    CollectionName::parse(value).map_err(|status| status.message)
}

fn field_name(value: &str) -> FieldName {
    FieldName::parse(value).expect("hardcoded field name should be valid")
}

fn read_jsonl_docs(path: &PathBuf) -> Result<Vec<garuda_types::Doc>, String> {
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
            garuda_types::DocId::parse(doc.id).map_err(|status| status.message)?,
            parse_scalar_fields(doc.fields)?,
            garuda_types::DenseVector::parse(doc.vector).map_err(|status| status.message)?,
        ));
    }

    Ok(docs)
}

fn parse_vector_arg(raw: &str) -> Result<garuda_types::DenseVector, String> {
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

fn parse_index_kind(raw: &str) -> Result<IndexKind, String> {
    match raw {
        "flat" => Ok(IndexKind::Flat),
        "hnsw" => Ok(IndexKind::Hnsw),
        _ => Err(format!("unsupported index kind: {raw}")),
    }
}

fn index_params(kind: IndexKind) -> IndexParams {
    match kind {
        IndexKind::Flat => IndexParams::Flat(FlatIndexParams),
        IndexKind::Hnsw => IndexParams::Hnsw(HnswIndexParams::default()),
    }
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
            if let Some(value) = value.as_i64() {
                return Ok(ScalarValue::Int64(value));
            }

            if let Some(value) = value.as_f64() {
                return Ok(ScalarValue::Float64(value));
            }

            Err(String::from("unsupported numeric value"))
        }
        serde_json::Value::String(value) => Ok(ScalarValue::String(value)),
        _ => Err(String::from("field values must be scalar")),
    }
}
