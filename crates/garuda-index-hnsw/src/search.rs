use crate::{
    HnswHit, HnswIndex, HnswSearchRequest,
    compare::{compare_score_then_doc_id},
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

        let entries_are_empty = self.entries.is_empty();
        if entries_are_empty {
            return Ok(Vec::new());
        }

        let request_limit = request.limit.get();
        let ef_search = request.ef_search.get() as usize;
        let entries_len = self.entries.len();

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
        let mut entry_point = self.graph.entry_point();
        let max_level = self.graph.max_level().get();

        for level in (1..=max_level).rev() {
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
            let entry = &self.entries[candidate.index.get()];
            hits.push(HnswHit {
                doc_id: entry.doc_id(),
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
        let mut best_score = self.score_node(query_vector, entry_point);

        loop {
            let mut improved = false;

            for &neighbor in self.graph.neighbors(level, entry_point) {
                let score = self.score_node(query_vector, neighbor);

                let neighbor_doc_id = self.entries[neighbor.get()].doc_id();
                let entry_point_doc_id = self.entries[entry_point.get()].doc_id();

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

        return entry_point;
    }

    pub(crate) fn search_layer(
        &self,
        level: HnswLevel,
        entry_point: NodeIndex,
        query_vector: &DenseVector,
        candidate_limit: usize,
    ) -> Vec<ScoredNode> {
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
                if !visited.insert(neighbor) {
                    continue;
                }

                let scored_neighbor = self.scored_node(query_vector, neighbor);
                let has_capacity = results.len() < candidate_limit;
                let improves_worst = results
                    .peek()
                    .is_some_and(|current_worst| scored_neighbor.score >= current_worst.score);

                if has_capacity || improves_worst {
                    candidates.push(scored_neighbor);

                    if !has_capacity {
                        results.pop();
                    }

                    results.push(WorstScoredNode::from(scored_neighbor));
                }
            }
        }

        to_vec(results)
    }

    fn scored_node(&self, query_vector: &DenseVector, node: NodeIndex) -> ScoredNode {
        ScoredNode {
            index: node,
            score: self.score_node(query_vector, node),
        }
    }
}

fn to_vec(mut results: BinaryHeap<WorstScoredNode>) -> Vec<ScoredNode> {
    let mut best = Vec::with_capacity(results.len());

    while let Some(candidate) = results.pop() {
        best.push(ScoredNode::from(candidate));
    }

    best
}
