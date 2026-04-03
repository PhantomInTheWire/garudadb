use garuda_math::l2_norm;

const TEST_DIMS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255,
];
const RANDOM_CASES_PER_DIM: usize = 8;
const RANDOM_SEED: u64 = 0x1234_5678_9abc_def0;
const RANDOM_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const RANDOM_INCREMENT: u64 = 1_442_695_040_888_963_407;
const ABS_TOLERANCE: f32 = 1.0e-5;
const REL_TOLERANCE: f32 = 2.0e-5;

#[test]
fn l2_norm_matches_scalar_oracle_for_curated_inputs() {
    for &dim in TEST_DIMS {
        for vector in cases(dim) {
            assert_close(l2_norm(&vector), l2_norm_scalar(&vector), dim, "curated");
        }
    }
}

#[test]
fn l2_norm_matches_scalar_oracle_for_deterministic_random_inputs() {
    let mut seed = RANDOM_SEED;

    for &dim in TEST_DIMS {
        for _ in 0..RANDOM_CASES_PER_DIM {
            let vector = random_vector(dim, &mut seed);
            assert_close(l2_norm(&vector), l2_norm_scalar(&vector), dim, "random");
        }
    }
}

fn l2_norm_scalar(vector: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for value in vector {
        sum += value * value;
    }

    sum.sqrt()
}

fn cases(dim: usize) -> Vec<Vec<f32>> {
    vec![
        vec![0.0; dim],
        alternating_vector(dim, 1.0, -1.0),
        ramp_vector(dim, -0.75, 0.125),
        repeated_pattern(dim, &[1.0e-3, -2.0e-3, 3.0e-3, -4.0e-3, 5.0e-3]),
        repeated_pattern(dim, &[10.0, -20.0, 30.0, -40.0]),
        tail_heavy_vector(dim),
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

fn random_vector(dim: usize, seed: &mut u64) -> Vec<f32> {
    (0..dim).map(|_| next_random_value(seed)).collect()
}

fn next_random_value(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(RANDOM_MULTIPLIER)
        .wrapping_add(RANDOM_INCREMENT);

    let bits = (*seed >> 32) as u32;
    (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn assert_close(actual: f32, expected: f32, dim: usize, label: &str) {
    let diff = (actual - expected).abs();
    let tolerance = ABS_TOLERANCE.max(expected.abs() * REL_TOLERANCE);

    assert!(
        diff <= tolerance,
        "{label} dim={dim} actual={actual} expected={expected} diff={diff} tolerance={tolerance}"
    );
}
