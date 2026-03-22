use crate::{HnswBuildEntry, HnswIndex, HnswLevel, heap::ScoredNode};
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

        candidates.sort_by(|left, right| {
            right.score.total_cmp(&left.score).then_with(|| {
                self.entries[left.index.get()]
                    .doc_id()
                    .cmp(&self.entries[right.index.get()].doc_id())
            })
        });
        candidates.truncate(self.config.neighbor_limits().for_level(level));

        candidates
            .into_iter()
            .map(|candidate| candidate.index)
            .collect()
    }

    fn add_reverse_neighbor(&mut self, level: HnswLevel, node: NodeIndex, neighbor: NodeIndex) {
        if self.graph.neighbors(level, node).contains(&neighbor) {
            return;
        }

        if self.graph.neighbors(level, node).len() >= self.config.neighbor_limits().for_level(level)
        {
            return;
        }

        self.graph.add_neighbor(level, node, neighbor);
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
