use clap::{Parser, Subcommand};
use garuda_engine::Database;
use garuda_types::{
    CollectionName, CollectionOptions, CollectionSchema, DistanceMetric, FieldName,
    FlatIndexParams, IndexParams, ScalarFieldSchema, ScalarType, VectorFieldSchema,
};
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
    Stats {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
    },
}

fn main() -> Result<(), String> {
    let Cli { root, command } = Cli::parse();

    match command {
        Commands::Init => {
            Database::open(&root).map_err(|status| status.message)?;
            println!("{}", root.display());
            Ok(())
        }
        Commands::Create { name, dimension } => {
            let db = Database::open(&root).map_err(|status| status.message)?;
            let schema = CollectionSchema {
                name: name.clone(),
                primary_key: field_name(PRIMARY_KEY_FIELD),
                fields: vec![ScalarFieldSchema {
                    name: field_name(PRIMARY_KEY_FIELD),
                    field_type: ScalarType::String,
                    nullable: false,
                    default_value: None,
                }],
                vector: VectorFieldSchema {
                    name: field_name(VECTOR_FIELD),
                    dimension: dimension.get(),
                    metric: DistanceMetric::Cosine,
                    index: IndexParams::Flat(FlatIndexParams),
                },
            };
            db.create_collection(schema, CollectionOptions::default())
                .map_err(|status| status.message)?;
            println!("{name}");
            Ok(())
        }
        Commands::Stats { name } => {
            let db = Database::open(&root).map_err(|status| status.message)?;
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
