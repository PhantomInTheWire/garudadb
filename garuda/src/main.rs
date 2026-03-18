use clap::{Parser, Subcommand};
use garuda_engine::Database;
use garuda_types::{
    CollectionName, CollectionOptions, CollectionSchema, DistanceMetric, FieldName,
    FlatIndexParams, IndexParams, ScalarFieldSchema, ScalarType, VectorFieldSchema,
};
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Create { name: String, dimension: usize },
    Stats { name: String },
    Doctor,
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    let db = Database::open(&cli.root).map_err(|status| status.message)?;

    match cli.command {
        None | Some(Commands::Doctor) => {
            println!("ok");
            Ok(())
        }
        Some(Commands::Init) => {
            println!("{}", cli.root.display());
            Ok(())
        }
        Some(Commands::Create { name, dimension }) => {
            let schema = CollectionSchema {
                name: CollectionName::parse(name.clone()).map_err(|status| status.message)?,
                primary_key: FieldName::parse("pk").map_err(|status| status.message)?,
                fields: vec![ScalarFieldSchema {
                    name: FieldName::parse("pk").map_err(|status| status.message)?,
                    field_type: ScalarType::String,
                    nullable: false,
                    default_value: None,
                }],
                vector: VectorFieldSchema {
                    name: FieldName::parse("embedding").map_err(|status| status.message)?,
                    dimension,
                    metric: DistanceMetric::Cosine,
                    index: IndexParams::Flat(FlatIndexParams),
                },
            };
            db.create_collection(schema, CollectionOptions::default())
                .map_err(|status| status.message)?;
            println!("{name}");
            Ok(())
        }
        Some(Commands::Stats { name }) => {
            let name = CollectionName::parse(name).map_err(|status| status.message)?;
            let collection = db.open_collection(&name).map_err(|status| status.message)?;
            let output = serde_json::to_string_pretty(&collection.stats())
                .map_err(|error| error.to_string())?;
            println!("{output}");
            Ok(())
        }
    }
}
