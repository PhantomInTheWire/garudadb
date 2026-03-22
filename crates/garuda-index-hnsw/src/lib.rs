use garuda_types::{
    DenseVector, DistanceMetric, HNSW_MAX_GRAPH_LEVEL, HnswEfConstruction, HnswEfSearch, HnswGraph,
    HnswLevel, HnswM, HnswMinNeighborCount, HnswNeighborConfig, HnswPruneWidth, HnswScalingFactor,
    InternalDocId, NodeIndex, Status, StatusCode, TopK, VectorDimension,
};

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

    pub fn m(&self) -> HnswM {
        self.build.neighbors.m()
    }

    pub fn min_neighbor_count(&self) -> HnswMinNeighborCount {
        self.build.neighbors.min_neighbor_count()
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

        Self {
            config,
            graph: HnswGraph::new(node_levels),
            entries,
        }
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

    pub fn entries(&self) -> &[HnswBuildEntry] {
        &self.entries
    }

    pub fn graph(&self) -> &HnswGraph {
        &self.graph
    }
}

fn sample_node_levels(config: &HnswIndexConfig, entries: &[HnswBuildEntry]) -> Vec<HnswLevel> {
    let mut node_levels = Vec::with_capacity(entries.len());

    for (index, entry) in entries.iter().enumerate() {
        node_levels.push(sample_node_level(
            config,
            NodeIndex::new(index),
            entry.doc_id(),
        ));
    }

    node_levels
}

fn sample_node_level(
    config: &HnswIndexConfig,
    node: NodeIndex,
    doc_id: InternalDocId,
) -> HnswLevel {
    let scale = config.build.scaling_factor.get() as usize; // scale factor = 50
    let max_level = config.max_graph_level(); // max graph level = 15
    let mut level = 0usize;
    let mut value = node.get() + (doc_id.get() as usize);

    while level < max_level && value % scale == 0 {
        level += 1;
        value /= scale.max(2);
    }

    HnswLevel::new(level)
}
