pub(crate) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    implementation::dot(lhs, rhs)
}

pub(crate) fn l2_norm(vector: &[f32]) -> f32 {
    implementation::l2_norm(vector)
}

pub(crate) fn squared_l2(lhs: &[f32], rhs: &[f32]) -> f32 {
    implementation::squared_l2(lhs, rhs)
}

pub(crate) fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
    implementation::cosine_similarity(lhs, rhs)
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

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn l2_norm_scalar(vector: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for value in vector {
        sum += value * value;
    }

    sum.sqrt()
}

#[cfg(not(target_arch = "aarch64"))]
mod implementation {
    use crate::simd::{dot_scalar, l2_norm_scalar, squared_l2_scalar};

    pub(super) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert_eq!(lhs.len(), rhs.len());
        dot_scalar(lhs, rhs)
    }

    pub(super) fn l2_norm(vector: &[f32]) -> f32 {
        l2_norm_scalar(vector)
    }

    pub(super) fn squared_l2(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert_eq!(lhs.len(), rhs.len());
        squared_l2_scalar(lhs, rhs)
    }

    pub(super) fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
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
}

#[cfg(target_arch = "aarch64")]
mod implementation {
    use std::arch::aarch64::*;

    pub(super) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
        unsafe { dot_neon(lhs, rhs) }
    }

    pub(super) fn l2_norm(vector: &[f32]) -> f32 {
        unsafe { l2_norm_neon(vector) }
    }

    pub(super) fn squared_l2(lhs: &[f32], rhs: &[f32]) -> f32 {
        unsafe { squared_l2_neon(lhs, rhs) }
    }

    pub(super) fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
        unsafe { cosine_similarity_neon(lhs, rhs) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn dot_neon(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert_eq!(lhs.len(), rhs.len());

        let mut index = 0usize;
        let len = lhs.len();
        let mut acc = vdupq_n_f32(0.0);

        while index + 4 <= len {
            let left = unsafe { vld1q_f32(lhs.as_ptr().add(index)) };
            let right = unsafe { vld1q_f32(rhs.as_ptr().add(index)) };
            acc = vfmaq_f32(acc, left, right);
            index += 4;
        }

        let mut sum = vaddvq_f32(acc);
        while index < len {
            sum += unsafe { *lhs.get_unchecked(index) * *rhs.get_unchecked(index) };
            index += 1;
        }

        sum
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn squared_l2_neon(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert_eq!(lhs.len(), rhs.len());

        let mut index = 0usize;
        let len = lhs.len();
        let mut acc = vdupq_n_f32(0.0);

        while index + 4 <= len {
            let left = unsafe { vld1q_f32(lhs.as_ptr().add(index)) };
            let right = unsafe { vld1q_f32(rhs.as_ptr().add(index)) };
            let delta = vsubq_f32(left, right);
            acc = vfmaq_f32(acc, delta, delta);
            index += 4;
        }

        let mut sum = vaddvq_f32(acc);
        while index < len {
            let delta = unsafe { *lhs.get_unchecked(index) - *rhs.get_unchecked(index) };
            sum += delta * delta;
            index += 1;
        }

        sum
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn l2_norm_neon(vector: &[f32]) -> f32 {
        let mut index = 0usize;
        let len = vector.len();
        let mut acc = vdupq_n_f32(0.0);

        while index + 4 <= len {
            let values = unsafe { vld1q_f32(vector.as_ptr().add(index)) };
            acc = vfmaq_f32(acc, values, values);
            index += 4;
        }

        let mut sum = vaddvq_f32(acc);
        while index < len {
            let value = unsafe { *vector.get_unchecked(index) };
            sum += value * value;
            index += 1;
        }

        sum.sqrt()
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn cosine_similarity_neon(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert_eq!(lhs.len(), rhs.len());

        let mut index = 0usize;
        let len = lhs.len();
        let mut dot = vdupq_n_f32(0.0);
        let mut lhs_norm = vdupq_n_f32(0.0);
        let mut rhs_norm = vdupq_n_f32(0.0);

        while index + 4 <= len {
            let left = unsafe { vld1q_f32(lhs.as_ptr().add(index)) };
            let right = unsafe { vld1q_f32(rhs.as_ptr().add(index)) };

            dot = vfmaq_f32(dot, left, right);
            lhs_norm = vfmaq_f32(lhs_norm, left, left);
            rhs_norm = vfmaq_f32(rhs_norm, right, right);
            index += 4;
        }

        let mut dot_sum = vaddvq_f32(dot);
        let mut lhs_norm_sum = vaddvq_f32(lhs_norm);
        let mut rhs_norm_sum = vaddvq_f32(rhs_norm);

        while index < len {
            let left = unsafe { *lhs.get_unchecked(index) };
            let right = unsafe { *rhs.get_unchecked(index) };

            dot_sum += left * right;
            lhs_norm_sum += left * left;
            rhs_norm_sum += right * right;
            index += 1;
        }

        if lhs_norm_sum == 0.0 || rhs_norm_sum == 0.0 {
            return 0.0;
        }

        dot_sum / (lhs_norm_sum.sqrt() * rhs_norm_sum.sqrt())
    }
}
