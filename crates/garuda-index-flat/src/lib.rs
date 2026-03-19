use garuda_math::score_doc;
use garuda_types::{DistanceMetric, InternalDocId, Status, StatusCode};

#[derive(Clone, Debug, PartialEq)]
pub struct FlatIndexEntry {
    pub doc_id: InternalDocId,
    pub vector: Vec<f32>,
}

impl FlatIndexEntry {
    pub fn new(doc_id: InternalDocId, vector: Vec<f32>) -> Self {
        Self { doc_id, vector }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlatSearchHit {
    pub doc_id: InternalDocId,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlatIndex {
    dimension: usize,
    entries: Vec<FlatIndexEntry>,
}

impl FlatIndex {
    pub fn build(dimension: usize, entries: Vec<FlatIndexEntry>) -> Result<Self, Status> {
        for entry in &entries {
            if entry.vector.len() == dimension {
                continue;
            }

            return Err(Status::err(
                StatusCode::InvalidArgument,
                "flat index entry dimension does not match index dimension",
            ));
        }

        Ok(Self { dimension, entries })
    }

    pub fn search(
        &self,
        metric: DistanceMetric,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<FlatSearchHit>, Status> {
        if query_vector.len() != self.dimension {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "query vector dimension does not match flat index dimension",
            ));
        }

        if top_k == 0 {
            return Ok(Vec::new());
        }

        let mut hits = Vec::with_capacity(self.entries.len());

        for entry in &self.entries {
            hits.push(FlatSearchHit {
                doc_id: entry.doc_id,
                score: score_doc(metric, query_vector, &entry.vector),
            });
        }

        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });
        hits.truncate(top_k);

        Ok(hits)
    }
}
