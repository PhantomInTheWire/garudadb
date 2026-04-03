use clap::{Args, Parser, Subcommand, ValueEnum};
use garuda_types::CollectionName;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;

use crate::parsing::parse_collection_name;

#[derive(Parser)]
pub struct Cli {
    #[arg(long, default_value = ".")]
    pub root: PathBuf,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Init,
    Create {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        dimension: NonZeroUsize,
        #[arg(long, value_enum, default_value_t = MetricArg::Cosine)]
        metric: MetricArg,
        #[arg(long)]
        segment_max_docs: Option<usize>,
        #[arg(long, value_enum)]
        storage_access: Option<StorageAccessArg>,
    },
    CreateFromSchema {
        path: PathBuf,
    },
    InsertJsonl {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        path: PathBuf,
    },
    UpsertJsonl {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        path: PathBuf,
    },
    UpdateJsonl {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        path: PathBuf,
    },
    DeleteIds {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        #[arg(required = true, num_args = 1..)]
        ids: Vec<String>,
    },
    DeleteFilter {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        #[arg(long)]
        filter: String,
    },
    Query {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        #[command(subcommand)]
        source: QuerySource,
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
        #[command(subcommand)]
        kind: CreateIndexKind,
    },
    DropIndex {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        field: String,
        kind: IndexKindArg,
    },
    AddColumn {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        column: String,
        #[arg(long = "type", value_enum)]
        field_type: ScalarTypeArg,
        #[arg(long, value_enum, default_value_t = NullabilityArg::Required)]
        nullability: NullabilityArg,
        #[arg(long, value_enum, default_value_t = ScalarIndexStateArg::None)]
        index: ScalarIndexStateArg,
        #[arg(long)]
        default: Option<String>,
    },
    RenameColumn {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        old: String,
        new: String,
    },
    DropColumn {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        column: String,
    },
    Flush {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
    },
    Optimize {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
    },
    Schema {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
    },
    Options {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
    },
    Stats {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
    },
}

#[derive(Subcommand)]
pub enum QuerySource {
    Vector(QueryArgs),
    ById(QueryArgs),
}

#[derive(Args)]
pub struct QueryArgs {
    #[arg(long)]
    pub value: String,
    #[arg(long)]
    pub top_k: usize,
    #[arg(long)]
    pub filter: Option<String>,
    #[arg(long)]
    pub include_vector: bool,
    #[arg(long, value_delimiter = ',')]
    pub fields: Option<Vec<String>>,
    #[command(subcommand)]
    pub search: Option<QuerySearch>,
}

#[derive(Subcommand)]
pub enum QuerySearch {
    Hnsw {
        #[arg(long)]
        ef_search: NonZeroU32,
    },
    Ivf {
        #[arg(long)]
        nprobe: NonZeroU32,
    },
}

#[derive(Subcommand)]
pub enum CreateIndexKind {
    Flat,
    Scalar,
    Hnsw {
        #[arg(long)]
        max_neighbors: Option<NonZeroU32>,
        #[arg(long)]
        scaling_factor: Option<NonZeroU32>,
        #[arg(long)]
        ef_construction: Option<NonZeroU32>,
        #[arg(long)]
        prune_width: Option<NonZeroU32>,
        #[arg(long)]
        min_neighbor_count: Option<NonZeroU32>,
        #[arg(long)]
        ef_search: Option<NonZeroU32>,
    },
    Ivf {
        #[arg(long = "n-list")]
        n_list: Option<NonZeroU32>,
        #[arg(long = "n-probe")]
        n_probe: Option<NonZeroU32>,
        #[arg(long)]
        training_iterations: Option<NonZeroU32>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
pub enum MetricArg {
    Cosine,
    InnerProduct,
    L2,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum StorageAccessArg {
    StandardIo,
    MmapPreferred,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ScalarTypeArg {
    Bool,
    Int64,
    Float64,
    String,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum NullabilityArg {
    Required,
    Nullable,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ScalarIndexStateArg {
    None,
    Indexed,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum IndexKindArg {
    Flat,
    Hnsw,
    Ivf,
    Scalar,
}
