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
    let dot = simd::dot(lhs, rhs);
    let lhs_norm = simd::dot(lhs, lhs);
    let rhs_norm = simd::dot(rhs, rhs);

    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        return 0.0;
    }

    dot / (lhs_norm.sqrt() * rhs_norm.sqrt())
}

fn inner_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    simd::dot(lhs, rhs)
}

fn negative_l2_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    -simd::squared_l2(lhs, rhs).sqrt()
}

#[cfg_attr(not(test), allow(dead_code))]
fn cosine_similarity_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    let dot = dot_scalar(lhs, rhs);
    let lhs_norm = dot_scalar(lhs, lhs);
    let rhs_norm = dot_scalar(rhs, rhs);

    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        return 0.0;
    }

    dot / (lhs_norm.sqrt() * rhs_norm.sqrt())
}

#[cfg_attr(not(test), allow(dead_code))]
fn inner_product_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    dot_scalar(lhs, rhs)
}

#[cfg_attr(not(test), allow(dead_code))]
fn negative_l2_distance_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    -squared_l2_scalar(lhs, rhs).sqrt()
}

pub(crate) fn dot_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for (left, right) in lhs.iter().zip(rhs.iter()) {
        sum += left * right;
    }

    sum
}

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
