use crate::score_doc;
use crate::simd;
use garuda_types::DistanceMetric;

const TEST_DIMS: &[usize] = &[
    1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 768, 1536,
];
const RANDOM_CASES_PER_DIM: usize = 16;
const ABS_TOLERANCE: f32 = 1.0e-4;
const REL_TOLERANCE: f32 = 2.0e-5;
const RANDOM_SEED: u64 = 0x1234_5678_9abc_def0;
const RANDOM_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const RANDOM_INCREMENT: u64 = 1_442_695_040_888_963_407;
const SCALE_SMALL: f32 = 1.0e-3;
const SCALE_MEDIUM: f32 = 0.5;
const SCALE_LARGE: f32 = 8.0;
const SCALE_XL: f32 = 64.0;

#[test]
fn cosine_scoring_handles_zero_norms() {
    let score = score_doc(DistanceMetric::Cosine, &[0.0, 0.0], &[1.0, 2.0]);
    assert_eq!(score, 0.0);
}

#[test]
fn inner_product_scoring_matches_dot_product() {
    let score = score_doc(DistanceMetric::InnerProduct, &[1.0, 2.0], &[3.0, 4.0]);
    assert_eq!(score, 11.0);
}

#[test]
fn l2_scoring_prefers_closer_vectors() {
    let close = score_doc(DistanceMetric::L2, &[1.0, 1.0], &[1.0, 2.0]);
    let far = score_doc(DistanceMetric::L2, &[1.0, 1.0], &[4.0, 5.0]);

    assert!(close > far);
}

#[test]
fn dispatched_dot_matches_scalar_oracle_for_curated_inputs() {
    for &dim in TEST_DIMS {
        for (lhs, rhs) in cases(dim) {
            let expected = simd::dot_scalar(&lhs, &rhs);
            let actual = simd::dot(&lhs, &rhs);
            assert_close(actual, expected, dim, "dot");
        }
    }
}

#[test]
fn dispatched_squared_l2_matches_scalar_oracle_for_curated_inputs() {
    for &dim in TEST_DIMS {
        for (lhs, rhs) in cases(dim) {
            let expected = simd::squared_l2_scalar(&lhs, &rhs);
            let actual = simd::squared_l2(&lhs, &rhs);
            assert_close(actual, expected, dim, "squared_l2");
        }
    }
}

#[test]
fn dispatched_metrics_match_scalar_oracle_for_curated_inputs() {
    for &dim in TEST_DIMS {
        for (lhs, rhs) in cases(dim) {
            assert_metric_matches_oracle(&lhs, &rhs, dim, "curated");
        }
    }
}

#[test]
fn dispatched_metrics_match_scalar_oracle_for_deterministic_random_inputs() {
    let mut seed = RANDOM_SEED;

    for &dim in TEST_DIMS {
        for _ in 0..RANDOM_CASES_PER_DIM {
            let lhs = random_vector(dim, &mut seed);
            let rhs = random_vector(dim, &mut seed);
            assert_metric_matches_oracle(&lhs, &rhs, dim, "random");
        }
    }
}

#[test]
fn cosine_oracle_returns_zero_when_query_is_zero_vector() {
    let lhs = vec![0.0; 31];
    let rhs = alternating_vector(31, 0.5, -1.5);

    let expected = cosine_similarity_scalar(&lhs, &rhs);
    let actual = score_doc(DistanceMetric::Cosine, &lhs, &rhs);

    assert_eq!(expected, 0.0);
    assert_eq!(actual, 0.0);
}

#[test]
fn cosine_oracle_returns_zero_when_candidate_is_zero_vector() {
    let lhs = alternating_vector(63, 1.25, -0.75);
    let rhs = vec![0.0; 63];

    let expected = cosine_similarity_scalar(&lhs, &rhs);
    let actual = score_doc(DistanceMetric::Cosine, &lhs, &rhs);

    assert_eq!(expected, 0.0);
    assert_eq!(actual, 0.0);
}

#[test]
#[should_panic]
fn score_doc_panics_when_cosine_vectors_have_mismatched_lengths() {
    score_doc(DistanceMetric::Cosine, &[1.0, 2.0], &[3.0]);
}

#[test]
#[should_panic]
fn score_doc_panics_when_inner_product_vectors_have_mismatched_lengths() {
    score_doc(DistanceMetric::InnerProduct, &[1.0, 2.0], &[3.0]);
}

#[test]
#[should_panic]
fn score_doc_panics_when_l2_vectors_have_mismatched_lengths() {
    score_doc(DistanceMetric::L2, &[1.0, 2.0], &[3.0]);
}

fn assert_metric_matches_oracle(lhs: &[f32], rhs: &[f32], dim: usize, label: &str) {
    let inner_expected = simd::dot_scalar(lhs, rhs);
    let inner_actual = score_doc(DistanceMetric::InnerProduct, lhs, rhs);
    assert_close(inner_actual, inner_expected, dim, label);

    let l2_expected = negative_l2_distance_scalar(lhs, rhs);
    let l2_actual = score_doc(DistanceMetric::L2, lhs, rhs);
    assert_close(l2_actual, l2_expected, dim, label);

    let cosine_expected = cosine_similarity_scalar(lhs, rhs);
    let cosine_actual = score_doc(DistanceMetric::Cosine, lhs, rhs);
    assert_close(cosine_actual, cosine_expected, dim, label);
}

fn cosine_similarity_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());

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

fn negative_l2_distance_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    -simd::squared_l2_scalar(lhs, rhs).sqrt()
}

fn cases(dim: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    vec![
        (
            alternating_vector(dim, 1.0, -1.0),
            alternating_vector(dim, -0.5, 2.0),
        ),
        (ramp_vector(dim, 0.25, 0.5), ramp_vector(dim, -0.75, 0.125)),
        (
            repeated_pattern(dim, &[SCALE_SMALL, -2.0e-3, 3.0e-3, -4.0e-3, 5.0e-3]),
            repeated_pattern(dim, &[-6.0e-3, 7.0e-3, -8.0e-3, 9.0e-3, -1.0e-2]),
        ),
        (
            repeated_pattern(dim, &[10.0, -20.0, 30.0, -40.0]),
            repeated_pattern(dim, &[5.5, -6.5, 7.5, -8.5]),
        ),
        (tail_heavy_vector(dim), tail_heavy_rhs(dim)),
    ]
}

fn alternating_vector(dim: usize, even: f32, odd: f32) -> Vec<f32> {
    (0..dim)
        .map(|index| if index % 2 == 0 { even } else { odd })
        .collect()
}

fn ramp_vector(dim: usize, start: f32, step: f32) -> Vec<f32> {
    (0..dim).map(|index| start + step * index as f32).collect()
}

fn repeated_pattern(dim: usize, pattern: &[f32]) -> Vec<f32> {
    (0..dim)
        .map(|index| pattern[index % pattern.len()])
        .collect()
}

fn tail_heavy_vector(dim: usize) -> Vec<f32> {
    let mut values = vec![0.0; dim];

    if let Some(last) = values.last_mut() {
        *last = 17.25;
    }
    if dim > 1 {
        values[dim - 2] = -9.5;
    }
    if dim > 2 {
        values[dim - 3] = 3.125;
    }

    values
}

fn tail_heavy_rhs(dim: usize) -> Vec<f32> {
    let mut values = tail_heavy_vector(dim);
    values.reverse();
    values
}

fn random_vector(dim: usize, seed: &mut u64) -> Vec<f32> {
    (0..dim).map(|_| next_random_value(seed)).collect()
}

fn next_random_value(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(RANDOM_MULTIPLIER)
        .wrapping_add(RANDOM_INCREMENT);

    let centered_bits = (*seed >> 32) as u32;
    let centered = (centered_bits as f32 / u32::MAX as f32) * 2.0 - 1.0;

    *seed = seed
        .wrapping_mul(RANDOM_MULTIPLIER)
        .wrapping_add(RANDOM_INCREMENT);

    let scale_bits = (*seed >> 32) as u32;
    let scale = match scale_bits % 4 {
        0 => SCALE_SMALL,
        1 => SCALE_MEDIUM,
        2 => SCALE_LARGE,
        _ => SCALE_XL,
    };

    centered * scale
}

fn assert_close(actual: f32, expected: f32, dim: usize, label: &str) {
    let diff = (actual - expected).abs();
    let allowed = ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs());

    assert!(
        diff <= allowed,
        "{label} mismatch for dim={dim}: actual={actual:?} expected={expected:?} diff={diff:?} allowed={allowed:?}"
    );
}
