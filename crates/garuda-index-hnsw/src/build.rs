use crate::{
    HnswBuildEntry, HnswIndex, HnswIndexConfig, HnswLevel, compare::compare_score_then_doc_id,
    heap::ScoredNode,
};
use garuda_math::score_doc;
use garuda_types::{InternalDocId, NodeIndex};

impl HnswIndex {
    pub(crate) fn insert_node(
        &mut self,
        node: NodeIndex,
        mut entry_point: NodeIndex,
        max_level: HnswLevel,
    ) {
        let node_level = self.graph.node_level(node);

        for level in ((node_level.get() + 1)..=max_level.get()).rev() {
            entry_point =
                self.select_entry_point(HnswLevel::new(level), self.vector(node), entry_point);
        }

        let top_level = HnswLevel::new(node_level.get().min(max_level.get()));
        let mut updates = Vec::with_capacity(top_level.get() + 1);

        for level in (0..=top_level.get()).rev() {
            let level = HnswLevel::new(level);
            let candidates = self.search_layer(
                level,
                entry_point,
                self.vector(node),
                self.config.build_candidate_limit(level),
            );

            if let Some(best_candidate) = best_retained_candidate(&candidates) {
                entry_point = best_candidate.index;
            }

            let neighbors = self.prune_to_max_neighbors(level, node, candidates);
            updates.push((level, neighbors));
        }

        for (level, neighbors) in updates {
            self.apply_neighbor_update(level, node, neighbors);
        }
    }

    fn apply_neighbor_update(
        &mut self,
        level: HnswLevel,
        node: NodeIndex,
        neighbors: Vec<NodeIndex>,
    ) {
        for &neighbor in &neighbors {
            self.add_reverse_neighbor(level, neighbor, node);
        }

        self.graph.replace_neighbors(level, node, neighbors);
    }

    fn add_reverse_neighbor(&mut self, level: HnswLevel, node: NodeIndex, neighbor: NodeIndex) {
        let neighbors = self.graph.neighbors(level, node);
        if neighbors.contains(&neighbor) {
            return;
        }

        let num_of_max_neighbors = self.config.neighbor_limits().for_level(level);
        let mut candidates = Vec::with_capacity(num_of_max_neighbors + 1);

        let node_vector = self.vector(node);
        for &existing_neighbor in neighbors {
            candidates.push(ScoredNode {
                index: existing_neighbor,
                score: self.score_node(node_vector, existing_neighbor),
            });
        }

        candidates.push(ScoredNode {
            index: neighbor,
            score: self.score_node(node_vector, neighbor),
        });

        let neighbors = self.prune_to_max_neighbors(level, node, candidates);
        self.graph.replace_neighbors(level, node, neighbors);
    }

    fn prune_to_max_neighbors(
        &self,
        level: HnswLevel,
        node: NodeIndex,
        mut candidates: Vec<ScoredNode>,
    ) -> Vec<NodeIndex> {
        candidates.sort_by(|left, right| {
            let left_doc_id = self.doc_id(left.index);
            let right_doc_id = self.doc_id(right.index);

            compare_score_then_doc_id(left.score, left_doc_id, right.score, right_doc_id)
        });

        let prune_width = self.config.build.prune_width.get() as usize;
        if candidates.len() > prune_width {
            candidates.truncate(prune_width);
        }

        let max_neighbors_count = self.config.neighbor_limits().for_level(level);
        let min_neighbor_count = self.config.min_neighbor_count().get() as usize;
        let mut neighbors = Vec::with_capacity(max_neighbors_count);

        for candidate in &candidates {
            let can_add_neighbor = candidate.index != node
                && !neighbors.contains(&candidate.index)
                && self.is_distinct_neighbor(candidate, &neighbors);

            if can_add_neighbor {
                neighbors.push(candidate.index);

                if neighbors.len() >= max_neighbors_count {
                    return neighbors;
                }
            }
        }

        if neighbors.len() < min_neighbor_count {
            for candidate in &candidates {
                let can_backfill_neighbor =
                    candidate.index != node && !neighbors.contains(&candidate.index);

                if can_backfill_neighbor {
                    neighbors.push(candidate.index);

                    if neighbors.len() >= min_neighbor_count
                        || neighbors.len() >= max_neighbors_count
                    {
                        break;
                    }
                }
            }
        }

        neighbors
    }

    fn is_distinct_neighbor(&self, candidate: &ScoredNode, neighbors: &[NodeIndex]) -> bool {
        let candidate_vector = self.vector(candidate.index);

        for &selected_neighbor in neighbors {
            let selected_vector = self.vector(selected_neighbor);
            let selected_score = score_doc(
                self.config.metric,
                candidate_vector.as_slice(),
                selected_vector.as_slice(),
            );

            if selected_score >= candidate.score {
                return false;
            }
        }

        true
    }
}

fn best_retained_candidate(candidates: &[ScoredNode]) -> Option<ScoredNode> {
    candidates.first().copied()
}

pub(crate) fn sample_node_levels(
    config: &HnswIndexConfig,
    entries: &[HnswBuildEntry],
) -> Vec<HnswLevel> {
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

pub(crate) fn sample_node_level(
    config: &HnswIndexConfig,
    node: NodeIndex,
    doc_id: InternalDocId,
) -> HnswLevel {
    let max_level = config.max_graph_level();
    if max_level == 0 {
        return HnswLevel::new(0);
    }

    let sample = hashed_unit_interval(node, doc_id);
    let scale = f64::from(config.build.scaling_factor.get()).ln();
    let level = (-sample.ln() / scale).floor() as usize;

    HnswLevel::new(level.min(max_level))
}

fn hashed_unit_interval(node: NodeIndex, doc_id: InternalDocId) -> f64 {
    const HASH_MIX: u64 = 0x9e37_79b9_7f4a_7c15;
    const HASH_STEP: u64 = 0xbf58_476d_1ce4_e5b9;
    const HASH_FINAL: u64 = 0x94d0_49bb_1331_11eb;

    let mut hash = node.get() as u64;
    hash ^= doc_id.get().wrapping_mul(HASH_MIX);
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(HASH_STEP);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(HASH_FINAL);
    hash ^= hash >> 31;

    (hash as f64 + 1.0) / (u64::MAX as f64 + 2.0)
}
