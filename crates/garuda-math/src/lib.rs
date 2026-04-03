//! Vector scoring helpers shared by the indexing crates.

use garuda_types::DistanceMetric;

mod simd;

pub fn l2_norm(vector: &[f32]) -> f32 {
    simd::l2_norm(vector)
}

pub fn score_doc(metric: DistanceMetric, query_vector: &[f32], candidate_vector: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => simd::cosine_similarity(query_vector, candidate_vector),
        DistanceMetric::InnerProduct => simd::dot(query_vector, candidate_vector),
        DistanceMetric::L2 => -simd::squared_l2(query_vector, candidate_vector).sqrt(),
    }
}

#[cfg(test)]
mod tests;
