use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroU32;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CollectionName(String);

impl CollectionName {
    pub fn parse(value: impl Into<String>) -> Result<Self, Status> {
        let value = value.into();

        if value.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "collection name cannot be empty",
            ));
        }

        let valid = value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-');

        if !valid {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "collection name may only use letters, numbers, '_' and '-'",
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CollectionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Borrow<str> for CollectionName {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FieldName(String);

impl FieldName {
    pub fn parse(value: impl Into<String>) -> Result<Self, Status> {
        let value = value.into();

        if value.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name cannot be empty",
            ));
        }

        let mut chars = value.chars();
        let Some(first) = chars.next() else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name cannot be empty",
            ));
        };

        if !first.is_ascii_alphabetic() && first != '_' {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name must start with a letter or '_'",
            ));
        }

        let valid = chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
        if !valid {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "field name may only use letters, numbers, and '_'",
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FieldName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Borrow<str> for FieldName {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DocId(String);

impl DocId {
    pub fn parse(value: impl Into<String>) -> Result<Self, Status> {
        let value = value.into();

        if value.is_empty() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "document id cannot be empty",
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Borrow<str> for DocId {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct InternalDocId(u64);

impl InternalDocId {
    pub const fn new_unchecked(value: u64) -> Self {
        Self(value)
    }

    pub fn new(value: u64) -> Result<Self, Status> {
        if value == 0 {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "internal doc id must be greater than zero",
            ));
        }

        Ok(Self(value))
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SegmentId(u64);

impl SegmentId {
    pub const fn new_unchecked(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeIndex(usize);

impl NodeIndex {
    pub fn new(value: usize) -> Self {
        Self(value)
    }

    pub fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HnswLevel(usize);

impl HnswLevel {
    pub fn new(value: usize) -> Self {
        Self(value)
    }

    pub fn get(self) -> usize {
        self.0
    }
}

pub const HNSW_LEVEL_ZERO_NEIGHBOR_MULTIPLIER: usize = 2;
pub const HNSW_MAX_GRAPH_LEVEL: usize = 15;

type NodeNeighbors = Vec<NodeIndex>;
type HnswLevelAdjacency = Vec<NodeNeighbors>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswGraph {
    node_levels: Vec<HnswLevel>,
    levels: Vec<HnswLevelAdjacency>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswNeighborLimits {
    level_zero: usize,
    upper_levels: usize,
}

impl HnswNeighborLimits {
    pub fn new(m: HnswM) -> Self {
        let upper_levels = m.get() as usize;

        Self {
            level_zero: upper_levels * HNSW_LEVEL_ZERO_NEIGHBOR_MULTIPLIER,
            upper_levels,
        }
    }

    pub fn for_level(self, level: HnswLevel) -> usize {
        match level.get() {
            0 => self.level_zero,
            _ => self.upper_levels,
        }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ManifestVersionId(u64);

impl ManifestVersionId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SnapshotId(u64);

impl SnapshotId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StorageFormatVersion(u16);

impl StorageFormatVersion {
    pub fn new(value: u16) -> Self {
        Self(value)
    }

    pub fn get(self) -> u16 {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswM(NonZeroU32);

impl HnswM {
    pub fn new(value: u32) -> Result<Self, Status> {
        let Some(value) = NonZeroU32::new(value) else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "hnsw m must be greater than zero",
            ));
        };

        Ok(Self(value))
    }

    pub fn from_persisted_u64(value: u64) -> Result<Self, Status> {
        let value = u32::try_from(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw m exceeds u32::MAX in persisted manifest",
            )
        })?;

        Self::new(value)
            .map_err(|_| Status::err(StatusCode::Internal, "hnsw m must be greater than zero"))
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswScalingFactor(NonZeroU32);

impl HnswScalingFactor {
    const MIN_SCALING_FACTOR: u32 = 2;

    pub fn new(value: u32) -> Result<Self, Status> {
        let Some(value) = NonZeroU32::new(value) else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "hnsw scaling_factor must be greater than one",
            ));
        };

        if value.get() < Self::MIN_SCALING_FACTOR {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "hnsw scaling_factor must be greater than one",
            ));
        }

        Ok(Self(value))
    }

    pub fn from_persisted_u64(value: u64) -> Result<Self, Status> {
        let value = u32::try_from(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw scaling_factor exceeds u32::MAX in persisted manifest",
            )
        })?;

        Self::new(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw scaling_factor must be greater than one",
            )
        })
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswPruneWidth(NonZeroU32);

impl HnswPruneWidth {
    pub fn new(value: u32) -> Result<Self, Status> {
        let Some(value) = NonZeroU32::new(value) else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "hnsw prune_width must be greater than zero",
            ));
        };

        Ok(Self(value))
    }

    pub fn from_persisted_u64(value: u64) -> Result<Self, Status> {
        let value = u32::try_from(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw prune_width exceeds u32::MAX in persisted manifest",
            )
        })?;

        Self::new(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw prune_width must be greater than zero",
            )
        })
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswMinNeighborCount(NonZeroU32);

impl HnswMinNeighborCount {
    pub fn new(value: u32) -> Result<Self, Status> {
        let Some(value) = NonZeroU32::new(value) else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "hnsw min_neighbor_count must be greater than zero",
            ));
        };

        Ok(Self(value))
    }

    pub fn from_persisted_u64(value: u64) -> Result<Self, Status> {
        let value = u32::try_from(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw min_neighbor_count exceeds u32::MAX in persisted manifest",
            )
        })?;

        Self::new(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw min_neighbor_count must be greater than zero",
            )
        })
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

impl HnswGraph {
    pub fn new(node_levels: Vec<HnswLevel>) -> Self {
        let level_count = max(&node_levels).map_or(1, |level| level.get() + 1);
        let node_count = node_levels.len();

        Self {
            node_levels,
            levels: vec![vec![Vec::new(); node_count]; level_count],
        }
    }

    pub fn from_parts(
        node_levels: Vec<HnswLevel>,
        levels: Vec<HnswLevelAdjacency>,
        entry_count: usize,
        neighbor_limits: HnswNeighborLimits,
    ) -> Result<Self, Status> {
        let graph = Self {
            node_levels,
            levels,
        };
        graph.validate(entry_count, neighbor_limits)?;
        Ok(graph)
    }

    pub fn node_count(&self) -> usize {
        self.node_levels.len()
    }

    pub fn node_level(&self, node: NodeIndex) -> HnswLevel {
        self.node_levels[node.get()]
    }

    pub fn node_levels(&self) -> &[HnswLevel] {
        &self.node_levels
    }

    pub fn max_level(&self) -> HnswLevel {
        assert!(!self.node_levels.is_empty(), "hnsw graph max level");
        HnswLevel::new(self.levels.len() - 1)
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn entry_point(&self) -> NodeIndex {
        let max_level = self.max_level();

        self.node_levels
            .iter()
            .position(|&level| level == max_level)
            .map(NodeIndex::new)
            .expect("hnsw graph entry point")
    }

    pub fn neighbors(&self, level: HnswLevel, node: NodeIndex) -> &[NodeIndex] {
        &self.levels[level.get()][node.get()]
    }

    pub fn replace_neighbors(
        &mut self,
        level: HnswLevel,
        node: NodeIndex,
        neighbors: Vec<NodeIndex>,
    ) {
        self.levels[level.get()][node.get()] = neighbors;
    }

    pub fn add_neighbor(&mut self, level: HnswLevel, node: NodeIndex, neighbor: NodeIndex) {
        self.levels[level.get()][node.get()].push(neighbor);
    }

    fn validate(
        &self,
        entry_count: usize,
        neighbor_limits: HnswNeighborLimits,
    ) -> Result<(), Status> {
        if self.node_count() != entry_count {
            return Err(Status::err(
                StatusCode::Internal,
                "persisted hnsw graph node levels do not match entry count",
            ));
        }

        if self.levels.is_empty() {
            return Err(Status::err(
                StatusCode::Internal,
                "persisted hnsw graph has no levels",
            ));
        }

        if entry_count == 0 {
            if self.levels.len() == 1 && self.levels[0].is_empty() {
                return Ok(());
            }

            return Err(Status::err(
                StatusCode::Internal,
                "persisted empty hnsw graph has an invalid shape",
            ));
        }

        if self.level_count() != self.max_level().get() + 1 {
            return Err(Status::err(
                StatusCode::Internal,
                "persisted hnsw graph level count does not match node levels",
            ));
        }

        if !self.node_levels.contains(&self.max_level()) {
            return Err(Status::err(
                StatusCode::Internal,
                "persisted hnsw graph has no node at the top level",
            ));
        }

        for level in &self.levels {
            if level.len() != entry_count {
                return Err(Status::err(
                    StatusCode::Internal,
                    "persisted hnsw graph levels do not match entry count",
                ));
            }
        }

        for (level, nodes) in self.levels.iter().enumerate() {
            let max_neighbors = neighbor_limits.for_level(HnswLevel::new(level));

            for neighbors in nodes {
                if neighbors.len() <= max_neighbors {
                    continue;
                }

                return Err(Status::err(
                    StatusCode::Internal,
                    "persisted hnsw graph exceeds configured neighbor limit",
                ));
            }
        }

        Ok(())
    }
}

fn max<T: Copy + Ord>(values: &[T]) -> Option<T> {
    values.iter().copied().max()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswNeighborConfig {
    m: HnswM,
    min_neighbor_count: HnswMinNeighborCount,
}

impl HnswNeighborConfig {
    pub fn new(m: HnswM, min_neighbor_count: HnswMinNeighborCount) -> Result<Self, Status> {
        if min_neighbor_count.get() > m.get() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "hnsw min_neighbor_count must not exceed m",
            ));
        }

        Ok(Self {
            m,
            min_neighbor_count,
        })
    }

    pub fn m(self) -> HnswM {
        self.m
    }

    pub fn min_neighbor_count(self) -> HnswMinNeighborCount {
        self.min_neighbor_count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswEfConstruction(NonZeroU32);

impl HnswEfConstruction {
    pub fn new(value: u32) -> Result<Self, Status> {
        let Some(value) = NonZeroU32::new(value) else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "hnsw ef_construction must be greater than zero",
            ));
        };

        Ok(Self(value))
    }

    pub fn from_persisted_u64(value: u64) -> Result<Self, Status> {
        let value = u32::try_from(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw ef_construction exceeds u32::MAX in persisted manifest",
            )
        })?;

        Self::new(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw ef_construction must be greater than zero",
            )
        })
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswEfSearch(NonZeroU32);

impl HnswEfSearch {
    pub fn new(value: u32) -> Result<Self, Status> {
        let Some(value) = NonZeroU32::new(value) else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "hnsw ef_search must be greater than zero",
            ));
        };

        Ok(Self(value))
    }

    pub fn from_persisted_u64(value: u64) -> Result<Self, Status> {
        let value = u32::try_from(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw ef_search exceeds u32::MAX in persisted manifest",
            )
        })?;

        Self::new(value).map_err(|_| {
            Status::err(
                StatusCode::Internal,
                "hnsw ef_search must be greater than zero",
            )
        })
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswIndexParams {
    pub m: HnswM,
    pub scaling_factor: HnswScalingFactor,
    pub ef_construction: HnswEfConstruction,
    pub prune_width: HnswPruneWidth,
    pub min_neighbor_count: HnswMinNeighborCount,
    pub ef_search: HnswEfSearch,
}

impl Default for HnswIndexParams {
    fn default() -> Self {
        Self {
            m: HnswM::new(16).expect("default hnsw m should be valid"),
            scaling_factor: HnswScalingFactor::new(50)
                .expect("default hnsw scaling_factor should be valid"),
            ef_construction: HnswEfConstruction::new(200)
                .expect("default hnsw ef_construction should be valid"),
            prune_width: HnswPruneWidth::new(16).expect("default hnsw prune_width should be valid"),
            min_neighbor_count: HnswMinNeighborCount::new(8)
                .expect("default hnsw min_neighbor_count should be valid"),
            ef_search: HnswEfSearch::new(64).expect("default hnsw ef_search should be valid"),
        }
    }
}

impl HnswIndexParams {
    pub fn neighbor_config(&self) -> Result<HnswNeighborConfig, Status> {
        HnswNeighborConfig::new(self.m, self.min_neighbor_count)
    }
}

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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OptimizeOptions;

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
    pub default_value: Option<ScalarValue>,
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
    pub index: IndexParams,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatusCode {
    Ok,
    InvalidArgument,
    NotFound,
    AlreadyExists,
    FailedPrecondition,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Status {
    pub code: StatusCode,
    pub message: String,
}

impl Status {
    pub fn ok() -> Self {
        Self {
            code: StatusCode::Ok,
            message: String::new(),
        }
    }

    pub fn err(code: StatusCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn is_ok(&self) -> bool {
        self.code == StatusCode::Ok
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WriteResult {
    pub id: DocId,
    pub status: Status,
}

impl WriteResult {
    pub fn ok(id: DocId) -> Self {
        Self {
            id,
            status: Status::ok(),
        }
    }

    pub fn err(id: DocId, code: StatusCode, message: impl Into<String>) -> Self {
        Self {
            id,
            status: Status::err(code, message),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CollectionStats {
    pub doc_count: usize,
    pub segment_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub id: SegmentId,
    pub path: String,
    pub min_doc_id: Option<InternalDocId>,
    pub max_doc_id: Option<InternalDocId>,
    pub doc_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: CollectionSchema,
    pub options: CollectionOptions,
    pub next_doc_id: InternalDocId,
    pub next_segment_id: SegmentId,
    pub id_map_snapshot_id: SnapshotId,
    pub delete_snapshot_id: SnapshotId,
    pub manifest_version_id: ManifestVersionId,
    pub writing_segment: SegmentMeta,
    pub persisted_segments: Vec<SegmentMeta>,
}
