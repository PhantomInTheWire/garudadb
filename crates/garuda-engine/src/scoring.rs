use garuda_types::DistanceMetric;

pub(crate) fn score_doc(
    metric: DistanceMetric,
    query_vector: &[f32],
    candidate_vector: &[f32],
) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_similarity(query_vector, candidate_vector),
        DistanceMetric::InnerProduct => inner_product(query_vector, candidate_vector),
        DistanceMetric::L2 => negative_l2_distance(query_vector, candidate_vector),
    }
}

fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut lhs_norm = 0.0f32;
    let mut rhs_norm = 0.0f32;

    for (left, right) in lhs.iter().zip(rhs.iter()) {
        dot += left * right;
        lhs_norm += left * left;
        rhs_norm += right * right;
    }

    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        return 0.0;
    }

    dot / (lhs_norm.sqrt() * rhs_norm.sqrt())
}

fn inner_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for (left, right) in lhs.iter().zip(rhs.iter()) {
        sum += left * right;
    }

    sum
}

fn negative_l2_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for (left, right) in lhs.iter().zip(rhs.iter()) {
        let delta = left - right;
        sum += delta * delta;
    }

    -sum.sqrt()
}
