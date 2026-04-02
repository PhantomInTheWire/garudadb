//! HNSW index construction and approximate nearest-neighbor search.
//!
//! This crate implements a small, deterministic HNSW variant around three
//! pieces of state:
//!
//! - `HnswIndexConfig`: vector dimension, distance metric, and build knobs.
//! - `Vec<HnswBuildEntry>`: the stored `(doc_id, vector)` rows, addressed by
//!   `NodeIndex`.
//! - `HnswGraph`: one adjacency list per level, plus a sampled level for each
//!   node.
//!
//! `HnswGraph` itself is defined in `garuda_types`. This crate owns the
//! scoring, search, insertion, and pruning logic.
//!
//! Implementation outline:
//!
//! 1. Build-time data model
//! - Nodes are append-only. `NodeIndex` is the position in `entries`.
//! - Each node stores its vector once. Graph edges only reference
//!   `NodeIndex`.
//! - Level 0 allows `2 * M` neighbors. Upper levels allow `M` neighbors.
//! - Scores are always "higher is better". `score_doc` maps metrics into that
//!   form, including `L2` by negating distance.
//!
//! 2. Level sampling
//! - Levels are not sampled from RNG state.
//! - `sample_node_level` hashes `(node_index, doc_id)` into `(0, 1)` and then
//!   applies `floor(-ln(sample) / ln(scaling_factor))`.
//! - Levels are capped at `HNSW_MAX_GRAPH_LEVEL`.
//! - This makes level assignment reproducible across rebuilds from the same live
//!   entries.
//!
//! 3. Bulk build
//! - `HnswIndex::build` samples every node level first, creates an empty graph
//!   with that shape, then inserts nodes in input order.
//! - Node 0 is the initial entry point.
//! - The tracked build entry point is replaced only when a newly inserted node
//!   reaches a higher level than every earlier node.
//! - If a new node creates a brand new top level, there are naturally no edges
//!   to add above the old max level because no older node exists there.
//!
//! 4. Incremental insert
//! - `insert` appends the new entry, samples its level, grows the graph, and
//!   reuses the current graph entry point plus current max level.
//! - The first inserted node becomes a one-node graph.
//! - Later inserts use the same insertion routine as bulk build.
//!
//! 5. Greedy descent for entry-point selection
//! - `select_entry_point` is the standard greedy walk within one level.
//! - Starting from the current entry point, it repeatedly moves to any neighbor
//!   with a strictly better score.
//! - Score ties break toward lower `doc_id`, which makes the walk deterministic.
//! - Search uses this from the top layer down to level 1. Build uses the same
//!   walk to descend from levels above the new node's own top layer.
//!
//! 6. Layer search
//! - `search_layer` is the shared local graph search used by both query-time
//!   search and build-time neighbor discovery.
//! - It keeps:
//!   - `visited`: a `HashSet<NodeIndex>`.
//!   - `candidates`: a max-heap of frontier nodes ordered by best score first.
//!   - `results`: a min-heap wrapper (`WorstScoredNode`) that keeps the current
//!     best `candidate_limit` nodes and exposes the current worst one.
//! - The loop pops the best frontier node. If that node is already worse than
//!   the current worst result, the search stops.
//! - Otherwise it visits the node's neighbors, scores unseen neighbors, and
//!   inserts them when either:
//!   - the result heap still has capacity, or
//!   - the neighbor is better than the current worst result.
//! - Returned candidates are sorted best-first.
//!
//! 7. Query-time search
//! - `search` validates query dimension, returns early on an empty index, and
//!   computes `candidate_limit = min(len, max(top_k, ef_search))`.
//! - It greedily descends from the graph entry point to a level-0 entry point.
//! - It runs `search_layer` on level 0.
//! - It converts nodes back into `HnswHit { doc_id, score }`, sorts by score
//!   descending and `doc_id` ascending, and truncates to `top_k`.
//! - Deterministic tie-breaking is therefore preserved in both traversal and
//!   final output ordering.
//!
//! 8. Build-time neighbor selection
//! - For each level from the node's insertion top level down to level 0,
//!   insertion runs `search_layer` from the current entry point.
//! - The layer-specific candidate budget is
//!   `max(level_neighbor_limit, ef_construction, prune_width)`.
//! - The best candidate from one level becomes the entry point for the next
//!   lower level.
//! - The resulting candidate set is then pruned to the final neighbor list for
//!   that level.
//!
//! 9. Pruning rule
//! - Candidates are first sorted by score descending, then `doc_id` ascending.
//! - The list is truncated to `prune_width` before diversity pruning.
//! - A candidate is accepted if:
//!   - it is not the node itself,
//!   - it is not already selected, and
//!   - it is "distinct" from already selected neighbors.
//! - Distinctness means: for every already selected neighbor `s`, the score
//!   between `candidate` and `s` must be strictly less than the candidate's
//!   score to the inserted node. If `candidate` is at least as close to an
//!   existing selected neighbor as it is to the node being inserted, the
//!   candidate is treated as redundant and skipped.
//! - If diversity pruning leaves fewer than `min_neighbor_count` neighbors, the
//!   code backfills from the best remaining candidates without the distinctness
//!   check until it reaches `min_neighbor_count` or the level limit.
//!
//! 10. Reverse edges
//! - After a node chooses its outgoing neighbors for a level, every chosen
//!   neighbor is updated to include a reverse edge back to the node.
//! - Reverse-edge insertion is not append-only. If the neighbor is full, the
//!   code re-runs the same pruning logic over `existing_neighbors + new_node`
//!   and replaces the neighbor list with the pruned result.
//! - This keeps edge counts bounded and makes reverse links follow the same
//!   selection policy as forward links.
//!
//! 11. Entry point semantics
//! - `HnswGraph::entry_point()` returns the last node that appears on the graph's
//!   highest level.
//! - That matches build behavior because nodes are appended and a newly created
//!   top-level node becomes the newest node on the max level.
//! - Query-time search always starts from that node.
//!
//! 12. Persisted graph invariants
//! - `HnswGraph::from_parts` validates persisted structure before the graph is
//!   accepted.
//! - Validation checks entry count, level count, presence of a top-level node,
//!   per-level adjacency shape, per-level neighbor limits, out-of-bounds edges,
//!   and edges that point above either endpoint's sampled level.
//! - `HnswIndex::from_parts` then pairs that graph with rebuilt live entries.
//!
//! 13. Determinism and non-goals
//! - Determinism is deliberate: level sampling is hash-based, comparisons use
//!   total float ordering, and equal scores break by lower `doc_id`.
//! - The implementation is single-threaded and in-memory.
//! - There is no lazy deletion, tombstone-aware graph maintenance, or heuristic
//!   that depends on external randomness.
//! - The crate aims for a predictable HNSW core that is easy to rebuild from the
//!   segment's live vectors and easy to validate when loaded from disk.

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
#[cfg(test)]
mod search_tests;

use build::{sample_node_level, sample_node_levels};

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
pub struct WritingHnswIndex {
    index: HnswIndex,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HnswIndex {
    config: HnswIndexConfig,
    entries: Vec<HnswBuildEntry>,
    graph: HnswGraph,
}

impl HnswIndex {
    pub fn empty(config: HnswIndexConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            graph: HnswGraph::new(Vec::new()),
        }
    }

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

    pub fn insert(&mut self, entry: HnswBuildEntry) {
        if self.entries.is_empty() {
            let level = sample_node_level(&self.config, NodeIndex::new(0), entry.doc_id());
            self.graph.push_node(level);
            self.entries.push(entry);
            return;
        }

        let entry_point = self.graph.entry_point();
        let max_level = self.graph.max_level();
        let node = self.graph.push_node(sample_node_level(
            &self.config,
            NodeIndex::new(self.entries.len()),
            entry.doc_id(),
        ));
        self.entries.push(entry);
        self.insert_node(node, entry_point, max_level);
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

impl WritingHnswIndex {
    pub fn new(config: HnswIndexConfig) -> Self {
        Self {
            index: HnswIndex::empty(config),
        }
    }

    pub fn insert(&mut self, doc_id: InternalDocId, vector: DenseVector) {
        let entry = HnswBuildEntry::new(self.index.config(), doc_id, vector)
            .expect("writing hnsw entry dimension");
        self.index.insert(entry);
    }

    pub fn search(
        &self,
        query_vector: &DenseVector,
        top_k: TopK,
        ef_search: HnswEfSearch,
    ) -> Result<Vec<HnswHit>, Status> {
        self.index
            .search(HnswSearchRequest::new(query_vector, top_k, ef_search))
    }

    pub fn graph(&self) -> &HnswGraph {
        self.index.graph()
    }
}
