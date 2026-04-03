use clap::{Args, Parser, Subcommand, ValueEnum};
use garuda_types::{
    CollectionName, CollectionOptions, DenseVector, DocId, HnswEfConstruction, HnswEfSearch,
    HnswIndexParams, HnswM, HnswMinNeighborCount, HnswPruneWidth, HnswScalingFactor,
    IvfIndexParams, IvfListCount, IvfProbeCount, IvfTrainingIterations, VectorProjection,
};
use std::num::NonZeroUsize;
use std::path::PathBuf;

use crate::parsing::{parse_collection_name, parse_doc_id, parse_non_zero_u32, parse_vector_arg};

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
    #[command(flatten)]
    Run(RunnableCommand),
}

#[derive(Subcommand)]
pub enum RunnableCommand {
    Create {
        #[arg(value_parser = parse_collection_name)]
        name: CollectionName,
        dimension: NonZeroUsize,
        #[arg(long, value_enum, default_value_t = MetricArg::Cosine)]
        metric: MetricArg,
        #[arg(long, default_value_t = CollectionOptions::DEFAULT_SEGMENT_MAX_DOCS)]
        segment_max_docs: usize,
        #[arg(long, value_enum, default_value_t = StorageAccessArg::MmapPreferred)]
        storage_access: StorageAccessArg,
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
    Vector(VectorQueryArgs),
    ById(ByIdQueryArgs),
}

#[derive(Args)]
pub struct QueryOptions {
    #[arg(long)]
    pub top_k: usize,
    #[arg(long)]
    pub filter: Option<String>,
    #[arg(long, value_enum, default_value_t = VectorProjectionArg::Exclude)]
    pub vector_projection: VectorProjectionArg,
    #[arg(long, value_delimiter = ',')]
    pub fields: Option<Vec<String>>,
    #[command(subcommand)]
    pub search: Option<QuerySearch>,
}

#[derive(Args)]
pub struct VectorQueryArgs {
    #[arg(long, value_parser = parse_vector_arg)]
    pub value: DenseVector,
    #[command(flatten)]
    pub options: QueryOptions,
}

#[derive(Args)]
pub struct ByIdQueryArgs {
    #[arg(long, value_parser = |value: &str| parse_doc_id(value.to_owned()))]
    pub value: DocId,
    #[command(flatten)]
    pub options: QueryOptions,
}

#[derive(Subcommand)]
pub enum QuerySearch {
    Hnsw {
        #[arg(long, value_parser = |value: &str| parse_non_zero_u32(value, HnswEfSearch::new))]
        ef_search: HnswEfSearch,
    },
    Ivf {
        #[arg(long, value_parser = |value: &str| parse_non_zero_u32(value, IvfProbeCount::new))]
        nprobe: IvfProbeCount,
    },
}

#[derive(Subcommand)]
pub enum CreateIndexKind {
    Flat,
    Scalar,
    Hnsw {
        #[arg(long, value_parser = |value: &str| parse_non_zero_u32(value, HnswM::new), default_value_t = HnswIndexParams::default().max_neighbors)]
        max_neighbors: HnswM,
        #[arg(long, value_parser = |value: &str| parse_non_zero_u32(value, HnswScalingFactor::new), default_value_t = HnswIndexParams::default().scaling_factor)]
        scaling_factor: HnswScalingFactor,
        #[arg(long, value_parser = |value: &str| parse_non_zero_u32(value, HnswEfConstruction::new), default_value_t = HnswIndexParams::default().ef_construction)]
        ef_construction: HnswEfConstruction,
        #[arg(long, value_parser = |value: &str| parse_non_zero_u32(value, HnswPruneWidth::new), default_value_t = HnswIndexParams::default().prune_width)]
        prune_width: HnswPruneWidth,
        #[arg(long, value_parser = |value: &str| parse_non_zero_u32(value, HnswMinNeighborCount::new), default_value_t = HnswIndexParams::default().min_neighbor_count)]
        min_neighbor_count: HnswMinNeighborCount,
        #[arg(long, value_parser = |value: &str| parse_non_zero_u32(value, HnswEfSearch::new), default_value_t = HnswIndexParams::default().ef_search)]
        ef_search: HnswEfSearch,
    },
    Ivf {
        #[arg(long = "n-list", value_parser = |value: &str| parse_non_zero_u32(value, IvfListCount::new), default_value_t = IvfIndexParams::default().n_list)]
        n_list: IvfListCount,
        #[arg(long = "n-probe", value_parser = |value: &str| parse_non_zero_u32(value, IvfProbeCount::new), default_value_t = IvfIndexParams::default().n_probe)]
        n_probe: IvfProbeCount,
        #[arg(long, value_parser = |value: &str| parse_non_zero_u32(value, IvfTrainingIterations::new), default_value_t = IvfIndexParams::default().training_iterations)]
        training_iterations: IvfTrainingIterations,
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

#[derive(Clone, Copy, ValueEnum)]
pub enum VectorProjectionArg {
    Include,
    Exclude,
}

impl From<VectorProjectionArg> for VectorProjection {
    fn from(value: VectorProjectionArg) -> Self {
        match value {
            VectorProjectionArg::Include => VectorProjection::Include,
            VectorProjectionArg::Exclude => VectorProjection::Exclude,
        }
    }
}
