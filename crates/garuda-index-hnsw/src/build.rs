use crate::{
    HnswBuildEntry, HnswIndex, HnswLevel, compare::compare_score_then_doc_id, heap::ScoredNode,
};
use garuda_math::score_doc;
use garuda_types::{InternalDocId, NodeIndex};

impl HnswIndex {
    pub(crate) fn connect_new_node(&mut self, node: NodeIndex) {
        let node_level = self.graph.node_level(node);

        for raw_level in 0..=node_level.get() {
            let level = HnswLevel::new(raw_level);
            let neighbors = self.best_previous_neighbors(level, node);

            for &neighbor in &neighbors {
                self.add_reverse_neighbor(level, neighbor, node);
            }

            self.graph.replace_neighbors(level, node, neighbors);
        }
    }

    fn best_previous_neighbors(&self, level: HnswLevel, node: NodeIndex) -> Vec<NodeIndex> {
        let mut candidates = Vec::with_capacity(node.get());

        for raw_index in 0..node.get() {
            let candidate = NodeIndex::new(raw_index);

            if self.graph.node_level(candidate) < level {
                continue;
            }

            candidates.push(ScoredNode {
                index: candidate,
                score: self.score_node(self.entries[node.get()].vector(), candidate),
            });
        }

        self.prune_to_max_neighbors(level, node, candidates)
    }

    fn add_reverse_neighbor(&mut self, level: HnswLevel, node: NodeIndex, neighbor: NodeIndex) {
        let neighbors = self.graph.neighbors(level, node);
        if neighbors.contains(&neighbor) {
            return;
        }

        let num_of_max_neighbors = self.config.neighbor_limits().for_level(level);
        let mut candidates = Vec::with_capacity(num_of_max_neighbors + 1);

        let node_vector = self.entries[node.get()].vector();
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
            let left_doc_id = self.entries[left.index.get()].doc_id();
            let right_doc_id = self.entries[right.index.get()].doc_id();

            compare_score_then_doc_id(left.score, left_doc_id, right.score, right_doc_id)
        });

        let prune_width = self.config.build.prune_width.get() as usize;
        if candidates.len() > prune_width {
            candidates.truncate(prune_width);
        }

        let max_neighbors = self.config.neighbor_limits().for_level(level);
        let min_neighbor_count = self.config.min_neighbor_count().get() as usize;
        let mut neighbors = Vec::with_capacity(max_neighbors);

        for candidate in &candidates {
            if candidate.index == node || neighbors.contains(&candidate.index) {
                continue;
            }

            if !self.is_distinct_neighbor(node, candidate, &neighbors) {
                continue;
            }

            neighbors.push(candidate.index);

            if neighbors.len() >= max_neighbors {
                return neighbors;
            }
        }

        if neighbors.len() < min_neighbor_count {
            for candidate in &candidates {
                if candidate.index == node || neighbors.contains(&candidate.index) {
                    continue;
                }

                neighbors.push(candidate.index);

                if neighbors.len() >= min_neighbor_count || neighbors.len() >= max_neighbors {
                    return neighbors;
                }
            }
        }

        return neighbors;
    }

    fn is_distinct_neighbor(
        &self,
        node: NodeIndex,
        candidate: &ScoredNode,
        neighbors: &[NodeIndex],
    ) -> bool {
        let candidate_vector = self.entries[candidate.index.get()].vector();

        for &selected_neighbor in neighbors {
            let selected_vector = self.entries[selected_neighbor.get()].vector();
            let selected_score = score_doc(
                self.config.metric,
                candidate_vector.as_slice(),
                selected_vector.as_slice(),
            );

            if selected_score >= candidate.score {
                return false;
            }
        }

        let node_vector = self.entries[node.get()].vector();
        let node_score = score_doc(
            self.config.metric,
            candidate_vector.as_slice(),
            node_vector.as_slice(),
        );

        return node_score == candidate.score;
    }
}

pub(crate) fn sample_node_levels(
    config: &crate::HnswIndexConfig,
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

fn sample_node_level(
    config: &crate::HnswIndexConfig,
    node: NodeIndex,
    doc_id: InternalDocId,
) -> HnswLevel {
    let scale = config.build.scaling_factor.get() as usize;
    let max_level = config.max_graph_level();
    let mut level = 0usize;
    let mut value = node.get() + (doc_id.get() as usize);

    while level < max_level && value % scale == 0 {
        level += 1;
        value /= scale;
    }

    HnswLevel::new(level)
}
