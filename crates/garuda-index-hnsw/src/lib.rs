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
//!   descends from the current highest-level active entry point.
//! - The first inserted node becomes a one-node graph.
//! - Later inserts use the same insertion routine as bulk build.
//! - Runtime state tracks active nodes by `doc_id`, so deletions can mark nodes
//!   inactive without rebuilding the full graph.
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
//!   computes `candidate_limit = min(active_len, max(top_k, ef_search))`.
//! - It greedily descends from the graph entry point to a level-0 entry point.
//! - It runs `search_layer` on level 0.
//! - It converts nodes back into `HnswHit { doc_id, score }`, sorts by score
//!   descending and `doc_id` ascending, and truncates to `top_k`.
//! - Deterministic tie-breaking is therefore preserved in both traversal and
//!   final output ordering.
//! - Inactive nodes are skipped during traversal and are never returned.
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
//! 9.5. Delete-time local repair
//! - `remove(doc_id)` marks the node inactive and unlinks it from all neighbors
//!   on levels up to that node's top level.
//! - For each affected level, the implementation collects both former active
//!   outgoing neighbors of the removed node and active incoming neighbors from
//!   a maintained runtime reverse-edge index, then performs local
//!   score-ordered pairing.
//! - Pair priority is: higher vector similarity first, then lower doc-id
//!   tie-break.
//! - Selected pairs are linked bidirectionally while respecting per-level
//!   max-neighbor limits.
//! - The first pass prefers nodes below `min_neighbor_count`; the second pass
//!   fills remaining available degree by score order.
//! - This keeps the neighborhood connected and limits quality regression after
//!   repeated deletes without rebuilding the full graph.
//!
//! 10. Reverse edges
//! - After a node chooses its outgoing neighbors for a level, every chosen
//!   neighbor is updated to include a reverse edge back to the node.
//! - Reverse-edge insertion is not append-only. If the neighbor is full, the
//!   code re-runs the same pruning logic over `existing_neighbors + new_node`
//!   and replaces the neighbor list with the pruned result.
//! - Deleted neighbors are ignored during reverse-edge candidate pruning.
//! - Runtime state also maintains a derived incoming-edge index for each level
//!   so delete repair can find nodes that still point at a target without
//!   scanning every active node.
//! - This keeps edge counts bounded, makes reverse links follow the same
//!   selection policy as forward links, and keeps incoming-neighbor lookup
//!   proportional to in-degree.
//!
//! 11. Entry point semantics
//! - Build tracks insertion descent from the current top-level entry point
//!   selected during build progression.
//! - Runtime state caches an active entry point at the current highest level.
//! - Query-time search starts from the cached active entry point and descends
//!   greedily from there.
//! - This keeps query entry-point selection valid after deletions mark nodes
//!   inactive without scanning all historical nodes.
//!
//! 12. Persisted graph invariants
//! - `HnswGraph::from_parts` validates persisted structure before the graph is
//!   accepted.
//! - Validation checks entry count, level count, presence of a top-level node,
//!   per-level adjacency shape, per-level neighbor limits, out-of-bounds edges,
//!   and edges that point above either endpoint's sampled level.
//! - `HnswIndex::from_parts` then pairs that graph with rebuilt live entries
//!   and reconstructs its runtime reverse-edge index from outgoing adjacency.
//!
//! 13. Determinism and non-goals
//! - Determinism is deliberate: level sampling is hash-based, comparisons use
//!   total float ordering, and equal scores break by lower `doc_id`.
//! - The implementation is single-threaded and in-memory.
//! - Deletion is lazy at the storage level (`NodeIndex` slots are not compacted
//!   in-place), but traversal and local graph repair are deletion-aware.
//! - There is no background compactor inside this crate and no heuristic that
//!   depends on external randomness.
//! - The crate aims for a predictable HNSW core that is easy to rebuild from the
//!   segment's live vectors and easy to validate when loaded from disk.
//! - Deletion currently marks nodes inactive and filters traversal/results; graph
//!   neighborhood repair is local and degree-aware.
//! - The incoming-edge index is derived runtime state owned by `HnswIndex`, not
//!   part of the persisted `HnswGraph` format.

use garuda_math::score_doc;
use garuda_types::{
    DenseVector, DistanceMetric, HNSW_MAX_GRAPH_LEVEL, HnswEfConstruction, HnswEfSearch, HnswGraph,
    HnswLevel, HnswM, HnswMinNeighborCount, HnswNeighborConfig, HnswNeighborLimits, HnswPruneWidth,
    HnswScalingFactor, InternalDocId, NodeIndex, RemoveResult, Status, StatusCode, TopK,
    VectorDimension,
};
use std::collections::HashMap;

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
    reverse_edges: Vec<Vec<Vec<NodeIndex>>>,
    node_states: Vec<HnswNodeState>,
    node_by_doc_id: HashMap<InternalDocId, NodeIndex>,
    active_entry_point: Option<(NodeIndex, HnswLevel)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HnswNodeState {
    Active,
    Deleted,
}

impl HnswIndex {
    pub fn empty(config: HnswIndexConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            graph: HnswGraph::new(Vec::new()),
            reverse_edges: Vec::new(),
            node_states: Vec::new(),
            node_by_doc_id: HashMap::new(),
            active_entry_point: None,
        }
    }

    pub fn build(config: HnswIndexConfig, entries: Vec<HnswBuildEntry>) -> Self {
        let node_levels = sample_node_levels(&config, &entries);
        let mut index = Self {
            config,
            graph: HnswGraph::new(node_levels),
            entries,
            reverse_edges: Vec::new(),
            node_states: Vec::new(),
            node_by_doc_id: HashMap::new(),
            active_entry_point: None,
        };

        index.node_states = vec![HnswNodeState::Active; index.entries.len()];
        index.rebuild_node_by_doc_id();
        index.rebuild_reverse_edges();

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
        let mut index = Self {
            config,
            entries,
            graph,
            reverse_edges: Vec::new(),
            node_states: Vec::new(),
            node_by_doc_id: HashMap::new(),
            active_entry_point: None,
        };
        index.node_states = vec![HnswNodeState::Active; index.entries.len()];
        index.rebuild_node_by_doc_id();
        index.rebuild_reverse_edges();
        index
    }

    pub fn config(&self) -> &HnswIndexConfig {
        &self.config
    }

    pub fn search(&self, request: HnswSearchRequest<'_>) -> Result<Vec<HnswHit>, Status> {
        self.execute_search(request)
    }

    pub fn insert(&mut self, entry: HnswBuildEntry) {
        if self.active_len() == 0 {
            let node_level = sample_node_level(
                &self.config,
                NodeIndex::new(self.entries.len()),
                entry.doc_id(),
            );
            let node = self.graph.push_node(node_level);
            self.entries.push(entry);
            self.node_states.push(HnswNodeState::Active);
            self.extend_reverse_edges_for_new_node(node_level);
            let replaced = self
                .node_by_doc_id
                .insert(self.entries[node.get()].doc_id(), node);
            assert!(
                replaced.is_none(),
                "hnsw first active insert should not replace doc id"
            );
            self.active_entry_point = Some((node, self.graph.node_level(node)));
            return;
        }

        let (entry_point, max_level) = self
            .active_entry_point_and_level()
            .expect("hnsw insert should have active entry point");
        let node_level = sample_node_level(
            &self.config,
            NodeIndex::new(self.entries.len()),
            entry.doc_id(),
        );
        let node = self.graph.push_node(node_level);
        self.entries.push(entry);
        self.node_states.push(HnswNodeState::Active);
        self.extend_reverse_edges_for_new_node(node_level);
        let replaced = self
            .node_by_doc_id
            .insert(self.entries[node.get()].doc_id(), node);
        assert!(
            replaced.is_none(),
            "hnsw insert should not replace existing doc id"
        );
        self.maybe_update_active_entry_point(node);
        self.insert_node(node, entry_point, max_level);
    }

    pub fn remove(&mut self, doc_id: InternalDocId) -> RemoveResult {
        let Some(node) = self.node_by_doc_id.remove(&doc_id) else {
            return RemoveResult::Missing;
        };

        assert!(self.is_active(node), "hnsw removed doc should be active");
        self.remove_node_and_repair(node);
        self.node_states[node.get()] = HnswNodeState::Deleted;
        if self
            .active_entry_point
            .is_some_and(|(entry_point, _)| entry_point == node)
        {
            self.refresh_active_entry_point();
        }
        RemoveResult::Removed
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

    fn is_active(&self, node: NodeIndex) -> bool {
        matches!(self.node_states[node.get()], HnswNodeState::Active)
    }

    fn active_len(&self) -> usize {
        self.node_by_doc_id.len()
    }

    fn active_entry_point_and_level(&self) -> Option<(NodeIndex, HnswLevel)> {
        self.active_entry_point
    }

    fn rebuild_node_by_doc_id(&mut self) {
        assert_eq!(
            self.node_states.len(),
            self.entries.len(),
            "hnsw node states should align with entries"
        );
        self.node_by_doc_id.clear();
        self.node_by_doc_id.reserve(self.entries.len());

        for (raw_node, entry) in self.entries.iter().enumerate() {
            if !matches!(self.node_states[raw_node], HnswNodeState::Active) {
                continue;
            }

            let replaced = self
                .node_by_doc_id
                .insert(entry.doc_id(), NodeIndex::new(raw_node));
            assert!(replaced.is_none(), "hnsw active doc ids should be unique");
        }

        self.refresh_active_entry_point();
    }

    fn rebuild_reverse_edges(&mut self) {
        self.reverse_edges =
            vec![vec![Vec::new(); self.graph.node_count()]; self.graph.level_count()];

        for raw_node in 0..self.graph.node_count() {
            let node = NodeIndex::new(raw_node);
            let node_level = self.graph.node_level(node).get();
            for raw_level in 0..=node_level {
                let level = HnswLevel::new(raw_level);
                for &neighbor in self.graph.neighbors(level, node) {
                    self.reverse_edges[level.get()][neighbor.get()].push(node);
                }
            }
        }
    }

    fn extend_reverse_edges_for_new_node(&mut self, node_level: HnswLevel) {
        while self.reverse_edges.len() <= node_level.get() {
            self.reverse_edges.push(Vec::new());
        }

        for incoming_by_node in &mut self.reverse_edges {
            incoming_by_node.resize_with(self.graph.node_count(), Vec::new);
        }
    }

    fn replace_neighbors(&mut self, level: HnswLevel, node: NodeIndex, neighbors: Vec<NodeIndex>) {
        let old_neighbors = self.graph.neighbors(level, node).to_vec();

        for &old_neighbor in &old_neighbors {
            self.reverse_edges[level.get()][old_neighbor.get()].retain(|&source| source != node);
        }

        self.graph.replace_neighbors(level, node, neighbors);

        let new_neighbors = self.graph.neighbors(level, node).to_vec();
        for neighbor in new_neighbors {
            let incoming = &mut self.reverse_edges[level.get()][neighbor.get()];
            if !incoming.contains(&node) {
                incoming.push(node);
            }
        }
    }

    fn maybe_update_active_entry_point(&mut self, node: NodeIndex) {
        let node_level = self.graph.node_level(node);
        let Some((entry_point, entry_level)) = self.active_entry_point else {
            self.active_entry_point = Some((node, node_level));
            return;
        };

        if node_level > entry_level || (node_level == entry_level && node.get() > entry_point.get())
        {
            self.active_entry_point = Some((node, node_level));
        }
    }

    fn refresh_active_entry_point(&mut self) {
        if self.node_by_doc_id.is_empty() {
            self.active_entry_point = None;
            return;
        }

        let max_level = self.graph.max_level().get();
        for raw_level in (0..=max_level).rev() {
            let level = HnswLevel::new(raw_level);
            let mut best_node: Option<NodeIndex> = None;

            for &node in self.node_by_doc_id.values() {
                if self.graph.node_level(node) < level {
                    continue;
                }

                match best_node {
                    None => best_node = Some(node),
                    Some(current) if node.get() > current.get() => best_node = Some(node),
                    _ => {}
                }
            }

            if let Some(node) = best_node {
                self.active_entry_point = Some((node, level));
                return;
            }
        }

        unreachable!("hnsw active nodes should yield an active entry point");
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

    pub fn remove(&mut self, doc_id: InternalDocId) -> RemoveResult {
        self.index.remove(doc_id)
    }

    pub fn graph(&self) -> &HnswGraph {
        self.index.graph()
    }
}
