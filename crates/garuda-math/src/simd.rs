pub(crate) struct CosineAccum {
    pub(crate) dot: f32,
    pub(crate) lhs_norm: f32,
    pub(crate) rhs_norm: f32,
}

pub(crate) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    implementation::dot(lhs, rhs)
}

pub(crate) fn squared_l2(lhs: &[f32], rhs: &[f32]) -> f32 {
    implementation::squared_l2(lhs, rhs)
}

pub(crate) fn cosine_accum(lhs: &[f32], rhs: &[f32]) -> CosineAccum {
    implementation::cosine_accum(lhs, rhs)
}

#[cfg(not(target_arch = "aarch64"))]
mod implementation {
    use super::CosineAccum;

    pub(super) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
        crate::dot_scalar(lhs, rhs)
    }

    pub(super) fn squared_l2(lhs: &[f32], rhs: &[f32]) -> f32 {
        crate::squared_l2_scalar(lhs, rhs)
    }

    pub(super) fn cosine_accum(lhs: &[f32], rhs: &[f32]) -> CosineAccum {
        assert_eq!(lhs.len(), rhs.len());

        let mut dot = 0.0;
        let mut lhs_norm = 0.0;
        let mut rhs_norm = 0.0;

        for (left, right) in lhs.iter().zip(rhs.iter()) {
            dot += left * right;
            lhs_norm += left * left;
            rhs_norm += right * right;
        }

        CosineAccum {
            dot,
            lhs_norm,
            rhs_norm,
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod implementation {
    use super::CosineAccum;
    use std::arch::aarch64::*;

    pub(super) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
        unsafe { dot_neon(lhs, rhs) }
    }

    pub(super) fn squared_l2(lhs: &[f32], rhs: &[f32]) -> f32 {
        unsafe { squared_l2_neon(lhs, rhs) }
    }

    pub(super) fn cosine_accum(lhs: &[f32], rhs: &[f32]) -> CosineAccum {
        unsafe { cosine_accum_neon(lhs, rhs) }
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
    unsafe fn cosine_accum_neon(lhs: &[f32], rhs: &[f32]) -> CosineAccum {
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

        let mut accum = CosineAccum {
            dot: vaddvq_f32(dot),
            lhs_norm: vaddvq_f32(lhs_norm),
            rhs_norm: vaddvq_f32(rhs_norm),
        };

        while index < len {
            let left = unsafe { *lhs.get_unchecked(index) };
            let right = unsafe { *rhs.get_unchecked(index) };

            accum.dot += left * right;
            accum.lhs_norm += left * left;
            accum.rhs_norm += right * right;
            index += 1;
        }

        accum
    }
}
