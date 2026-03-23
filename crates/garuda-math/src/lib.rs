use garuda_types::DistanceMetric;

mod simd;

pub fn score_doc(metric: DistanceMetric, query_vector: &[f32], candidate_vector: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_similarity(query_vector, candidate_vector),
        DistanceMetric::InnerProduct => inner_product(query_vector, candidate_vector),
        DistanceMetric::L2 => negative_l2_distance(query_vector, candidate_vector),
    }
}

fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
    let accum = simd::cosine_accum(lhs, rhs);

    if accum.lhs_norm == 0.0 || accum.rhs_norm == 0.0 {
        return 0.0;
    }

    accum.dot / (accum.lhs_norm.sqrt() * accum.rhs_norm.sqrt())
}

fn inner_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    simd::dot(lhs, rhs)
}

fn negative_l2_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    -simd::squared_l2(lhs, rhs).sqrt()
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn dot_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for (left, right) in lhs.iter().zip(rhs.iter()) {
        sum += left * right;
    }

    sum
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn squared_l2_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for (left, right) in lhs.iter().zip(rhs.iter()) {
        let delta = left - right;
        sum += delta * delta;
    }

    sum
}

#[cfg(test)]
mod tests;
