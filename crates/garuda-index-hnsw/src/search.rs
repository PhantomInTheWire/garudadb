use crate::{
    HnswHit, HnswIndex, HnswSearchRequest,
    compare::compare_score_then_doc_id,
    heap::{ScoredNode, WorstScoredNode},
};
use garuda_types::{DenseVector, HnswLevel, NodeIndex, Status, StatusCode};
use std::collections::{BinaryHeap, HashSet};

impl HnswIndex {
    pub(crate) fn execute_search(
        &self,
        request: HnswSearchRequest<'_>,
    ) -> Result<Vec<HnswHit>, Status> {
        let query_dim = request.query_vector.len();
        let config_dim = self.config.dimension.get();

        if query_dim != config_dim {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "query vector dimension does not match hnsw index dimension",
            ));
        }

        if self.active_len() == 0 {
            return Ok(Vec::new());
        }

        let request_limit = request.limit.get();
        let ef_search = request.ef_search.get() as usize;
        let entries_len = self.active_len();

        let candidate_limit = request_limit.max(ef_search).min(entries_len);

        let entry_point = self.search_entry_point(request.query_vector);

        let candidates: Vec<ScoredNode> = self.search_layer(
            HnswLevel::new(0),
            entry_point,
            request.query_vector,
            candidate_limit,
        );

        Ok(self.top_hits(candidates, request_limit))
    }

    fn search_entry_point(&self, query_vector: &DenseVector) -> NodeIndex {
        let (mut entry_point, top_level) = self
            .active_entry_point_and_level()
            .expect("hnsw search should have active entry point");

        for level in (1..=top_level.get()).rev() {
            entry_point = self.select_entry_point(HnswLevel::new(level), query_vector, entry_point);
        }

        entry_point
    }

    fn top_hits(&self, candidates: Vec<ScoredNode>, limit: usize) -> Vec<HnswHit> {
        let mut hits = self.to_hnsw_hits(candidates);

        hits.sort_by(|left, right| {
            compare_score_then_doc_id(left.score, left.doc_id, right.score, right.doc_id)
        });
        hits.truncate(limit);
        hits
    }

    fn to_hnsw_hits(&self, candidates: Vec<ScoredNode>) -> Vec<HnswHit> {
        let mut hits = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            hits.push(HnswHit {
                doc_id: self.doc_id(candidate.index),
                score: candidate.score,
            });
        }

        hits
    }

    pub(crate) fn select_entry_point(
        &self,
        level: HnswLevel,
        query_vector: &DenseVector,
        mut entry_point: NodeIndex,
    ) -> NodeIndex {
        assert!(
            self.is_active(entry_point),
            "hnsw select entry point should be active"
        );
        let mut best_score = self.score_node(query_vector, entry_point);

        loop {
            let mut improved = false;

            for &neighbor in self.graph.neighbors(level, entry_point) {
                if !self.is_active(neighbor) {
                    continue;
                }

                let score = self.score_node(query_vector, neighbor);

                let neighbor_doc_id = self.doc_id(neighbor);
                let entry_point_doc_id = self.doc_id(entry_point);

                // either have a better score or same score but lower doc_id
                let is_better_neighbor = score > best_score
                    || (score == best_score && neighbor_doc_id < entry_point_doc_id);

                if is_better_neighbor {
                    entry_point = neighbor;
                    best_score = score;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        entry_point
    }

    pub(crate) fn search_layer(
        &self,
        level: HnswLevel,
        entry_point: NodeIndex,
        query_vector: &DenseVector,
        candidate_limit: usize,
    ) -> Vec<ScoredNode> {
        assert!(
            self.is_active(entry_point),
            "hnsw search-layer entry point should be active"
        );

        let mut visited = HashSet::with_capacity(candidate_limit);
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let entry = self.scored_node(query_vector, entry_point);

        visited.insert(entry_point);
        candidates.push(entry);
        results.push(WorstScoredNode::from(entry));

        while let Some(candidate) = candidates.pop() {
            let Some(worst_result) = results.peek() else {
                break;
            };

            if candidate.score < worst_result.score {
                break;
            }

            let neighbors = self.graph.neighbors(level, candidate.index);
            for &neighbor in neighbors {
                if !self.is_active(neighbor) {
                    continue;
                }

                if !visited.insert(neighbor) {
                    continue;
                }

                let scored_neighbor = self.scored_node(query_vector, neighbor);
                let has_capacity = results.len() < candidate_limit;
                let improves_worst = results.peek().is_some_and(|current_worst| {
                    let neighbor_doc_id = self.doc_id(scored_neighbor.index);
                    let worst_doc_id = self.doc_id(current_worst.index);

                    compare_score_then_doc_id(
                        scored_neighbor.score,
                        neighbor_doc_id,
                        current_worst.score,
                        worst_doc_id,
                    )
                    .is_lt()
                });

                if has_capacity || improves_worst {
                    candidates.push(scored_neighbor);

                    if !has_capacity {
                        results.pop();
                    }

                    results.push(WorstScoredNode::from(scored_neighbor));
                }
            }
        }

        collect_best_first(results)
    }

    fn scored_node(&self, query_vector: &DenseVector, node: NodeIndex) -> ScoredNode {
        ScoredNode {
            index: node,
            score: self.score_node(query_vector, node),
        }
    }
}

fn collect_best_first(mut results: BinaryHeap<WorstScoredNode>) -> Vec<ScoredNode> {
    let mut candidates = Vec::with_capacity(results.len());

    while let Some(candidate) = results.pop() {
        candidates.push(ScoredNode::from(candidate));
    }

    candidates.reverse();
    candidates
}
