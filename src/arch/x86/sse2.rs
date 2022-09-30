#![allow(non_camel_case_types)]

use core::fmt::{self, Debug};
use core::mem;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};
use core::slice;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::sse_macros::{float_type, impl_int, impl_ord_mask, int_type};
use crate::mask::*;
use crate::simd::{Bitwise, Float, Int, LanesEq, LanesOrd, Select, Simd};
use crate::{Arch, Task};

pub struct Sse2Impl;

impl Arch for Sse2Impl {
    type f32 = f32x4;
    type f64 = f64x2;

    type u8 = u8x16;
    type u16 = u16x8;
    type u32 = u32x4;
    type u64 = u64x2;

    type i8 = i8x16;
    type i16 = i16x8;
    type i32 = i32x4;
    type i64 = i64x2;

    type m8 = m8x16;
    type m16 = m16x8;
    type m32 = m32x4;
    type m64 = m64x2;

    #[inline(always)]
    fn invoke<T: Task>(task: T) -> T::Result {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner<T: Task>(task: T) -> T::Result {
            task.run::<Sse2Impl>()
        }

        unsafe { inner(task) }
    }
}

macro_rules! impl_int_mul {
    ($int8:ident, $int16:ident, $int32:ident, $int64:ident) => {
        impl Mul for $int8 {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse2")]
                unsafe fn inner(lhs: $int8, rhs: $int8) -> $int8 {
                    let lhs_odd = _mm_srli_epi16(lhs.0, 8);
                    let rhs_odd = _mm_srli_epi16(rhs.0, 8);
                    let even = _mm_mullo_epi16(lhs.0, rhs.0);
                    let odd = _mm_slli_epi16(_mm_mullo_epi16(lhs_odd, rhs_odd), 8);
                    let mask = _mm_set1_epi16(0x00FF);
                    $int8(_mm_or_si128(_mm_and_si128(mask, even), odd))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl Mul for $int16 {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse2")]
                unsafe fn inner(lhs: $int16, rhs: $int16) -> $int16 {
                    $int16(_mm_mullo_epi16(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl Mul for $int32 {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse2")]
                unsafe fn inner(lhs: $int32, rhs: $int32) -> $int32 {
                    let lhs_odd = _mm_srli_epi64(lhs.0, 32);
                    let rhs_odd = _mm_srli_epi64(rhs.0, 32);
                    let even = _mm_mul_epu32(lhs.0, rhs.0);
                    let odd = _mm_slli_epi64(_mm_mul_epu32(lhs_odd, rhs_odd), 32);
                    let mask = _mm_set1_epi64x(0xFFFFFFFF);
                    $int32(_mm_or_si128(_mm_and_si128(mask, even), odd))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl Mul for $int64 {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse2")]
                unsafe fn inner(lhs: $int64, rhs: $int64) -> $int64 {
                    let low_high = _mm_mul_epu32(lhs.0, _mm_srli_epi64(rhs.0, 32));
                    let high_low = _mm_mul_epu32(rhs.0, _mm_srli_epi64(lhs.0, 32));
                    let low_low = _mm_mul_epu32(lhs.0, rhs.0);
                    let high = _mm_slli_epi64(_mm_add_epi32(low_high, high_low), 32);
                    $int64(_mm_add_epi32(low_low, high))
                }

                unsafe { inner(self, rhs) }
            }
        }
    };
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn _mm_blendv_ps_fallback(a: __m128, b: __m128, mask: __m128) -> __m128 {
    _mm_or_ps(_mm_andnot_ps(mask, a), _mm_and_ps(mask, b))
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn _mm_blendv_pd_fallback(a: __m128d, b: __m128d, mask: __m128d) -> __m128d {
    _mm_or_pd(_mm_andnot_pd(mask, a), _mm_and_pd(mask, b))
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn _mm_blendv_epi8_fallback(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
    _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
}

float_type! {
    "sse2",
    f32x4, __m128, f32, 4, m32x4,
    _mm_set1_ps, _mm_loadu_ps, _mm_storeu_ps, _mm_castps_si128, _mm_castsi128_ps, _mm_blendv_ps_fallback,
    _mm_cmpeq_ps, _mm_cmpneq_ps, _mm_cmplt_ps, _mm_cmple_ps, _mm_cmpgt_ps, _mm_cmpge_ps,
    _mm_min_ps, _mm_max_ps, _mm_add_ps, _mm_sub_ps, _mm_mul_ps, _mm_div_ps, _mm_xor_ps,
}
float_type! {
    "sse2",
    f64x2, __m128d, f64, 2, m64x2,
    _mm_set1_pd, _mm_loadu_pd, _mm_storeu_pd, _mm_castpd_si128, _mm_castsi128_pd, _mm_blendv_pd_fallback,
    _mm_cmpeq_pd, _mm_cmpneq_pd, _mm_cmplt_pd, _mm_cmple_pd, _mm_cmpgt_pd, _mm_cmpge_pd,
    _mm_min_pd, _mm_max_pd, _mm_add_pd, _mm_sub_pd, _mm_mul_pd, _mm_div_pd, _mm_xor_pd,
}

int_type! { "sse2", u8x16, u8, 16, m8x16, _mm_set1_epi8, _mm_blendv_epi8_fallback }
int_type! { "sse2", u16x8, u16, 8, m16x8, _mm_set1_epi16, _mm_blendv_epi8_fallback }
int_type! { "sse2", u32x4, u32, 4, m32x4, _mm_set1_epi32, _mm_blendv_epi8_fallback }
int_type! { "sse2", u64x2, u64, 2, m64x2, _mm_set1_epi64x, _mm_blendv_epi8_fallback }
impl_int! { "sse2", u8x16, _mm_set1_epi8, _mm_add_epi8, _mm_sub_epi8 }
impl_int! { "sse2", u16x8, _mm_set1_epi16, _mm_add_epi16, _mm_sub_epi16 }
impl_int! { "sse2", u32x4, _mm_set1_epi32, _mm_add_epi32, _mm_sub_epi32 }
impl_int! { "sse2", u64x2, _mm_set1_epi64x, _mm_add_epi64, _mm_sub_epi64 }
impl_int_mul! { u8x16, u16x8, u32x4, u64x2 }

impl LanesEq for u8x16 {
    type Output = m8x16;

    #[inline(always)]
    fn eq(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &u8x16, rhs: &u8x16) -> m8x16 {
            m8x16(_mm_cmpeq_epi8(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesOrd for u8x16 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        !other.le(self)
    }

    #[inline(always)]
    fn le(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &u8x16, rhs: &u8x16) -> m8x16 {
            m8x16(_mm_cmpeq_epi8(lhs.0, _mm_min_epu8(lhs.0, rhs.0)))
        }

        unsafe { inner(self, other) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: u8x16, rhs: u8x16) -> u8x16 {
            u8x16(_mm_max_epu8(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: u8x16, rhs: u8x16) -> u8x16 {
            u8x16(_mm_min_epu8(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesEq for u16x8 {
    type Output = m16x8;

    #[inline(always)]
    fn eq(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &u16x8, rhs: &u16x8) -> m16x8 {
            m16x8(_mm_cmpeq_epi16(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesOrd for u16x8 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &u16x8, rhs: &u16x8) -> m16x8 {
            let bias = _mm_set1_epi16(i16::MIN);
            m16x8(_mm_cmplt_epi16(
                _mm_add_epi16(lhs.0, bias),
                _mm_add_epi16(rhs.0, bias),
            ))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesEq for u32x4 {
    type Output = m32x4;

    #[inline(always)]
    fn eq(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &u32x4, rhs: &u32x4) -> m32x4 {
            m32x4(_mm_cmpeq_epi32(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesOrd for u32x4 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &u32x4, rhs: &u32x4) -> m32x4 {
            let bias = _mm_set1_epi32(i32::MIN);
            m32x4(_mm_cmplt_epi32(
                _mm_add_epi32(lhs.0, bias),
                _mm_add_epi32(rhs.0, bias),
            ))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesEq for u64x2 {
    type Output = m64x2;

    #[inline(always)]
    fn eq(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &u64x2, rhs: &u64x2) -> m64x2 {
            // Compare high and low 32-bit integers separately, then swap and AND together
            let res = _mm_cmpeq_epi32(lhs.0, rhs.0);
            let swapped = _mm_shuffle_epi32(res, 0xB1);
            m64x2(_mm_and_si128(res, swapped))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesOrd for u64x2 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &u64x2, rhs: &u64x2) -> m64x2 {
            // If we split two 64-bit integers L and R into pairs of 32-bit integers (Lh, Ll) and
            // (Rh, Rl), L < R iff Lh < Rh || (Lh == Rh && Ll < Rl).
            //
            // Since we only have a signed 32-bit compare and we need to perform four unsigned
            // comparisons, we need to bias all four 32-bit integers.
            let bias = _mm_set1_epi32(i32::MIN);
            let lhs = _mm_add_epi32(lhs.0, bias);
            let rhs = _mm_add_epi32(rhs.0, bias);
            let lt = _mm_cmplt_epi32(lhs, rhs);
            let eq = _mm_cmpeq_epi32(lhs, rhs);
            // Copy Rh < Lh result down to the lower 32 bits
            let lt_low = _mm_shuffle_epi32(lt, 0xA0);
            let res = _mm_or_si128(lt, _mm_and_si128(eq, lt_low));
            // Copy the final result back to the upper 32 bits
            m64x2(_mm_shuffle_epi32(res, 0xF5))
        }

        unsafe { inner(self, other) }
    }
}

int_type! { "sse2", i8x16, i8, 16, m8x16, _mm_set1_epi8, _mm_blendv_epi8_fallback }
int_type! { "sse2", i16x8, i16, 8, m16x8, _mm_set1_epi16, _mm_blendv_epi8_fallback }
int_type! { "sse2", i32x4, i32, 4, m32x4, _mm_set1_epi32, _mm_blendv_epi8_fallback }
int_type! { "sse2", i64x2, i64, 2, m64x2, _mm_set1_epi64x, _mm_blendv_epi8_fallback }
impl_int! { "sse2", i8x16, _mm_set1_epi8, _mm_add_epi8, _mm_sub_epi8 }
impl_int! { "sse2", i16x8, _mm_set1_epi16, _mm_add_epi16, _mm_sub_epi16 }
impl_int! { "sse2", i32x4, _mm_set1_epi32, _mm_add_epi32, _mm_sub_epi32 }
impl_int! { "sse2", i64x2, _mm_set1_epi64x, _mm_add_epi64, _mm_sub_epi64 }
impl_int_mul! { i8x16, i16x8, i32x4, i64x2 }

impl LanesEq for i8x16 {
    type Output = m8x16;

    #[inline(always)]
    fn eq(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &i8x16, rhs: &i8x16) -> m8x16 {
            m8x16(_mm_cmpeq_epi8(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesOrd for i8x16 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &i8x16, rhs: &i8x16) -> m8x16 {
            m8x16(_mm_cmpgt_epi8(rhs.0, lhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesEq for i16x8 {
    type Output = m16x8;

    #[inline(always)]
    fn eq(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &i16x8, rhs: &i16x8) -> m16x8 {
            m16x8(_mm_cmpeq_epi16(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesOrd for i16x8 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &i16x8, rhs: &i16x8) -> m16x8 {
            m16x8(_mm_cmplt_epi16(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: i16x8, rhs: i16x8) -> i16x8 {
            i16x8(_mm_max_epi16(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: i16x8, rhs: i16x8) -> i16x8 {
            i16x8(_mm_min_epi16(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesEq for i32x4 {
    type Output = m32x4;

    #[inline(always)]
    fn eq(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &i32x4, rhs: &i32x4) -> m32x4 {
            m32x4(_mm_cmpeq_epi32(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesOrd for i32x4 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &i32x4, rhs: &i32x4) -> m32x4 {
            m32x4(_mm_cmplt_epi32(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesEq for i64x2 {
    type Output = m64x2;

    #[inline(always)]
    fn eq(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &i64x2, rhs: &i64x2) -> m64x2 {
            // Compare high and low 32-bit integers separately, then swap and AND together
            let res = _mm_cmpeq_epi32(lhs.0, rhs.0);
            let swapped = _mm_shuffle_epi32(res, 0xB1);
            m64x2(_mm_and_si128(res, swapped))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesOrd for i64x2 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(lhs: &i64x2, rhs: &i64x2) -> m64x2 {
            // If we split two 64-bit integers L and R into pairs of 32-bit integers (Lh, Ll) and
            // (Rh, Rl), L < R iff Lh < Rh || (Lh == Rh && Ll < Rl).
            //
            // Bias just the lower 32 bits, since we only have a signed 32-bit compare and we need
            // to perform an unsigned comparison on the lower bits.
            let bias = _mm_set_epi32(0, i32::MIN, 0, i32::MIN);
            let lhs = _mm_add_epi32(lhs.0, bias);
            let rhs = _mm_add_epi32(rhs.0, bias);
            let lt = _mm_cmplt_epi32(lhs, rhs);
            let eq = _mm_cmpeq_epi32(lhs, rhs);
            // Copy Rh < Lh result down to the lower 32 bits
            let lt_low = _mm_shuffle_epi32(lt, 0xA0);
            let res = _mm_or_si128(lt, _mm_and_si128(eq, lt_low));
            // Copy the final result back to the upper 32 bits
            m64x2(_mm_shuffle_epi32(res, 0xF5))
        }

        unsafe { inner(self, other) }
    }
}

int_type! { "sse2", m8x16, m8, 16, m8x16, _mm_set1_epi8, _mm_blendv_epi8_fallback }
int_type! { "sse2", m16x8, m16, 8, m16x8, _mm_set1_epi16, _mm_blendv_epi8_fallback }
int_type! { "sse2", m32x4, m32, 4, m32x4, _mm_set1_epi32, _mm_blendv_epi8_fallback }
int_type! { "sse2", m64x2, m64, 2, m64x2, _mm_set1_epi64x, _mm_blendv_epi8_fallback }
impl_ord_mask! { "sse2", m8x16 }
impl_ord_mask! { "sse2", m16x8 }
impl_ord_mask! { "sse2", m32x4 }
impl_ord_mask! { "sse2", m64x2 }

#[test]
fn u64_lt() {
    let lhs = u64x2::new(0);
    let rhs = u64x2::new(u32::MAX as u64);
    assert!(lhs.lt(&rhs)[0] == true.into(), "{} < {}", lhs[0], rhs[0]);
}

#[test]
fn i64_lt() {
    let lhs = i64x2::new(0);
    let rhs = i64x2::new(u32::MAX as i64);
    assert!(lhs.lt(&rhs)[0] == true.into(), "{} < {}", lhs[0], rhs[0]);
}
