use crate::{
    HnswBuildEntry, HnswIndex, HnswIndexConfig, HnswLevel, compare::compare_score_then_doc_id,
    heap::ScoredNode,
};
use garuda_math::score_doc;
use garuda_types::{InternalDocId, NodeIndex};

impl HnswIndex {
    pub(crate) fn remove_node_and_repair(&mut self, node: NodeIndex) {
        let max_level = self.graph.node_level(node).get();

        for raw_level in 0..=max_level {
            let level = HnswLevel::new(raw_level);
            let former_neighbors = self
                .graph
                .neighbors(level, node)
                .iter()
                .copied()
                .filter(|&neighbor| self.is_active(neighbor))
                .collect::<Vec<_>>();

            let mut affected_neighbors = former_neighbors;
            for incoming in self.incoming_active_neighbors(level, node) {
                if affected_neighbors.contains(&incoming) {
                    continue;
                }

                affected_neighbors.push(incoming);
            }

            for &neighbor in &affected_neighbors {
                self.unlink_edge(level, neighbor, node);
            }
            self.replace_neighbors(level, node, Vec::new());

            self.repair_neighbors(level, &affected_neighbors);
        }
    }

    fn incoming_active_neighbors(&self, level: HnswLevel, node: NodeIndex) -> Vec<NodeIndex> {
        self.reverse_edges[level.get()][node.get()]
            .iter()
            .copied()
            .filter(|&candidate| self.is_active(candidate))
            .collect()
    }

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

            if let Some(best_candidate) = candidates.first().copied() {
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

        self.replace_neighbors(level, node, neighbors);
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
            if !self.is_active(existing_neighbor) {
                continue;
            }

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
        self.replace_neighbors(level, node, neighbors);
    }

    fn unlink_edge(&mut self, level: HnswLevel, from: NodeIndex, to: NodeIndex) {
        let mut neighbors = self.graph.neighbors(level, from).to_vec();
        let original_len = neighbors.len();
        neighbors.retain(|&neighbor| neighbor != to);
        if neighbors.len() == original_len {
            return;
        }

        self.replace_neighbors(level, from, neighbors);
    }

    fn repair_neighbors(&mut self, level: HnswLevel, neighbors: &[NodeIndex]) {
        if neighbors.len() < 2 {
            return;
        }

        let level_limit = self.config.neighbor_limits().for_level(level);
        let min_degree = self.config.min_neighbor_count().get() as usize;

        let mut degrees = Vec::with_capacity(neighbors.len());
        for &neighbor in neighbors {
            degrees.push(self.graph.neighbors(level, neighbor).len());
        }

        let mut pairs = Vec::new();
        for left in 0..neighbors.len() {
            for right in (left + 1)..neighbors.len() {
                let left_node = neighbors[left];
                let right_node = neighbors[right];

                if self.graph.neighbors(level, left_node).contains(&right_node) {
                    continue;
                }

                let score = score_doc(
                    self.config.metric,
                    self.vector(left_node).as_slice(),
                    self.vector(right_node).as_slice(),
                );
                pairs.push(RepairPair { left, right, score });
            }
        }

        pairs.sort_by(|a, b| {
            let score_order = b.score.total_cmp(&a.score);
            if score_order != std::cmp::Ordering::Equal {
                return score_order;
            }

            let a_left_doc = self.doc_id(neighbors[a.left]);
            let b_left_doc = self.doc_id(neighbors[b.left]);
            let left_doc_order = a_left_doc.cmp(&b_left_doc);
            if left_doc_order != std::cmp::Ordering::Equal {
                return left_doc_order;
            }

            let a_right_doc = self.doc_id(neighbors[a.right]);
            let b_right_doc = self.doc_id(neighbors[b.right]);
            a_right_doc.cmp(&b_right_doc)
        });

        for &pair in &pairs {
            let left = pair.left;
            let right = pair.right;

            if degrees[left] >= level_limit || degrees[right] >= level_limit {
                continue;
            }

            if degrees[left] >= min_degree && degrees[right] >= min_degree {
                continue;
            }

            let left_node = neighbors[left];
            let right_node = neighbors[right];
            self.link_edge(level, left_node, right_node);
            degrees[left] += 1;
            degrees[right] += 1;
        }

        for &pair in &pairs {
            let left = pair.left;
            let right = pair.right;

            if degrees[left] >= level_limit || degrees[right] >= level_limit {
                continue;
            }

            let left_node = neighbors[left];
            let right_node = neighbors[right];
            if self.graph.neighbors(level, left_node).contains(&right_node) {
                continue;
            }

            self.link_edge(level, left_node, right_node);
            degrees[left] += 1;
            degrees[right] += 1;
        }
    }

    fn link_edge(&mut self, level: HnswLevel, left: NodeIndex, right: NodeIndex) {
        let mut left_neighbors = self.graph.neighbors(level, left).to_vec();
        if !left_neighbors.contains(&right) {
            left_neighbors.push(right);
            self.replace_neighbors(level, left, left_neighbors);
        }

        let mut right_neighbors = self.graph.neighbors(level, right).to_vec();
        if !right_neighbors.contains(&left) {
            right_neighbors.push(left);
            self.replace_neighbors(level, right, right_neighbors);
        }
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

#[derive(Clone, Copy)]
struct RepairPair {
    left: usize,
    right: usize,
    score: f32,
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
