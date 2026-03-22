use crate::{Status, StatusCode};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

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
    pub fn new(max_neighbors: HnswM) -> Self {
        let upper_levels = max_neighbors.get() as usize;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswNeighborConfig {
    max_neighbors: HnswM,
    min_neighbor_count: HnswMinNeighborCount,
}

impl HnswNeighborConfig {
    pub fn new(
        max_neighbors: HnswM,
        min_neighbor_count: HnswMinNeighborCount,
    ) -> Result<Self, Status> {
        if min_neighbor_count.get() > max_neighbors.get() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "hnsw min_neighbor_count must not exceed m",
            ));
        }

        Ok(Self {
            max_neighbors,
            min_neighbor_count,
        })
    }

    pub fn max_neighbors(self) -> HnswM {
        self.max_neighbors
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
    pub max_neighbors: HnswM,
    pub scaling_factor: HnswScalingFactor,
    pub ef_construction: HnswEfConstruction,
    pub prune_width: HnswPruneWidth,
    pub min_neighbor_count: HnswMinNeighborCount,
    pub ef_search: HnswEfSearch,
}

impl Default for HnswIndexParams {
    fn default() -> Self {
        Self {
            max_neighbors: HnswM::new(16).expect("default hnsw max_neighbors should be valid"),
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
        HnswNeighborConfig::new(self.max_neighbors, self.min_neighbor_count)
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

    fn max_node_level(&self) -> HnswLevel {
        max(&self.node_levels).expect("hnsw graph max node level")
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
            .rposition(|&level| level == max_level)
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

    pub fn push_node(&mut self, level: HnswLevel) -> NodeIndex {
        let node = NodeIndex::new(self.node_levels.len());
        self.node_levels.push(level);

        while self.levels.len() <= level.get() {
            self.levels.push(Vec::new());
        }

        for neighbors_by_node in &mut self.levels {
            neighbors_by_node.push(Vec::new());
        }

        node
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

        if self.level_count() != self.max_node_level().get() + 1 {
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
            let level = HnswLevel::new(level);
            let max_neighbors = neighbor_limits.for_level(level);

            for (raw_node, neighbors) in nodes.iter().enumerate() {
                let node = NodeIndex::new(raw_node);

                if self.node_level(node) < level && !neighbors.is_empty() {
                    return Err(Status::err(
                        StatusCode::Internal,
                        "persisted hnsw graph has edges above a node's level",
                    ));
                }

                if neighbors.len() > max_neighbors {
                    return Err(Status::err(
                        StatusCode::Internal,
                        "persisted hnsw graph exceeds configured neighbor limit",
                    ));
                }

                for &neighbor in neighbors {
                    if neighbor.get() >= entry_count {
                        return Err(Status::err(
                            StatusCode::Internal,
                            "persisted hnsw graph has an out-of-bounds neighbor",
                        ));
                    }

                    if self.node_level(neighbor) < level {
                        return Err(Status::err(
                            StatusCode::Internal,
                            "persisted hnsw graph has a neighbor above its level",
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

fn max<T: Copy + Ord>(values: &[T]) -> Option<T> {
    values.iter().copied().max()
}
