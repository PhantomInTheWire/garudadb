use garuda_math::score_doc;
use garuda_types::{
    DenseVector, DistanceMetric, HNSW_MAX_GRAPH_LEVEL, HnswEfConstruction, HnswEfSearch, HnswGraph,
    HnswLevel, HnswM, HnswMinNeighborCount, HnswNeighborConfig, HnswNeighborLimits, HnswPruneWidth,
    HnswScalingFactor, InternalDocId, NodeIndex, Status, StatusCode, TopK, VectorDimension,
};

mod build;
mod compare;
mod heap;
mod search;

use build::sample_node_levels;

#[derive(Clone, Debug, PartialEq)]
pub struct HnswBuildConfig {
    pub neighbors: HnswNeighborConfig,
    pub scaling_factor: HnswScalingFactor,
    pub ef_construction: HnswEfConstruction,
    pub prune_width: HnswPruneWidth,
}

impl HnswBuildConfig {
    pub fn new(
        neighbors: HnswNeighborConfig,
        scaling_factor: HnswScalingFactor,
        ef_construction: HnswEfConstruction,
        prune_width: HnswPruneWidth,
    ) -> Self {
        Self {
            neighbors,
            scaling_factor,
            ef_construction,
            prune_width,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HnswIndexConfig {
    pub dimension: VectorDimension,
    pub metric: DistanceMetric,
    pub build: HnswBuildConfig,
}

impl HnswIndexConfig {
    pub fn new(dimension: VectorDimension, metric: DistanceMetric, build: HnswBuildConfig) -> Self {
        Self {
            dimension,
            metric,
            build,
        }
    }

    pub fn max_neighbors(&self) -> HnswM {
        self.build.neighbors.max_neighbors()
    }

    pub fn min_neighbor_count(&self) -> HnswMinNeighborCount {
        self.build.neighbors.min_neighbor_count()
    }

    fn neighbor_limits(&self) -> HnswNeighborLimits {
        HnswNeighborLimits::new(self.max_neighbors())
    }

    fn build_candidate_limit(&self, level: HnswLevel) -> usize {
        self.neighbor_limits()
            .for_level(level)
            .max(self.build.ef_construction.get() as usize)
            .max(self.build.prune_width.get() as usize)
    }

    fn max_graph_level(&self) -> usize {
        HNSW_MAX_GRAPH_LEVEL
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HnswBuildEntry {
    doc_id: InternalDocId,
    vector: DenseVector,
}

impl HnswBuildEntry {
    pub fn new(
        config: &HnswIndexConfig,
        doc_id: InternalDocId,
        vector: DenseVector,
    ) -> Result<Self, Status> {
        if vector.len() != config.dimension.get() {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "hnsw index entry dimension does not match index dimension",
            ));
        }

        Ok(Self { doc_id, vector })
    }

    pub fn doc_id(&self) -> InternalDocId {
        self.doc_id
    }

    pub fn vector(&self) -> &DenseVector {
        &self.vector
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HnswSearchRequest<'a> {
    pub query_vector: &'a DenseVector,
    pub limit: TopK,
    pub ef_search: HnswEfSearch,
}

impl<'a> HnswSearchRequest<'a> {
    pub fn new(query_vector: &'a DenseVector, limit: TopK, ef_search: HnswEfSearch) -> Self {
        Self {
            query_vector,
            limit,
            ef_search,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HnswHit {
    pub doc_id: InternalDocId,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HnswIndex {
    config: HnswIndexConfig,
    entries: Vec<HnswBuildEntry>,
    graph: HnswGraph,
}

impl HnswIndex {
    pub fn build(config: HnswIndexConfig, entries: Vec<HnswBuildEntry>) -> Self {
        let node_levels = sample_node_levels(&config, &entries);
        let mut index = Self {
            config,
            graph: HnswGraph::new(node_levels),
            entries,
        };

        if index.entries.is_empty() {
            return index;
        }

        let mut entry_point = NodeIndex::new(0);
        let mut max_level = index.graph.node_level(entry_point);

        for raw_index in 1..index.entries.len() {
            let node = NodeIndex::new(raw_index);
            index.insert_node(node, entry_point, max_level);

            let node_level = index.graph.node_level(node);
            if node_level > max_level {
                entry_point = node;
                max_level = node_level;
            }
        }

        index
    }

    pub fn from_parts(
        config: HnswIndexConfig,
        entries: Vec<HnswBuildEntry>,
        graph: HnswGraph,
    ) -> Self {
        Self {
            config,
            entries,
            graph,
        }
    }

    pub fn config(&self) -> &HnswIndexConfig {
        &self.config
    }

    pub fn search(&self, request: HnswSearchRequest<'_>) -> Result<Vec<HnswHit>, Status> {
        self.execute_search(request)
    }

    pub fn entries(&self) -> &[HnswBuildEntry] {
        &self.entries
    }

    pub fn graph(&self) -> &HnswGraph {
        &self.graph
    }

    fn entry(&self, node: NodeIndex) -> &HnswBuildEntry {
        &self.entries[node.get()]
    }

    fn doc_id(&self, node: NodeIndex) -> InternalDocId {
        self.entry(node).doc_id()
    }

    fn vector(&self, node: NodeIndex) -> &DenseVector {
        self.entry(node).vector()
    }

    fn score_node(&self, query_vector: &DenseVector, node: NodeIndex) -> f32 {
        score_doc(
            self.config.metric,
            query_vector.as_slice(),
            self.vector(node).as_slice(),
        )
    }
}
