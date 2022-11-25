#![allow(non_camel_case_types)]

use core::fmt::{self, Debug};
use core::mem;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};
use core::ops::{Shl, ShlAssign, Shr, ShrAssign};
use core::slice;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::sse_common::*;
use crate::mask::*;
use crate::simd::{Bitwise, Convert, Float, Int, LanesEq, LanesOrd, Narrow, Select, Simd, Widen};
use crate::{Arch, Convert16, Convert32, Convert64, Convert8, Task};

pub struct Sse4_2Impl;

impl Arch for Sse4_2Impl {
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

    const NAME: &'static str = "sse4.2";

    #[inline(always)]
    fn invoke<T: Task>(task: T) -> T::Result {
        #[inline]
        #[target_feature(enable = "sse4.2")]
        unsafe fn inner<T: Task>(task: T) -> T::Result {
            task.run::<Sse4_2Impl>()
        }

        unsafe { inner(task) }
    }
}

macro_rules! impl_ord_uint {
    ($uint:ident, $mask:ident, $cmpeq:ident, $max:ident, $min:ident) => {
        impl LanesEq for $uint {
            type Output = $mask;

            #[inline(always)]
            fn eq(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner(lhs: &$uint, rhs: &$uint) -> $mask {
                    $mask($cmpeq(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }

        impl LanesOrd for $uint {
            #[inline(always)]
            fn lt(&self, other: &Self) -> Self::Output {
                !other.le(self)
            }

            #[inline(always)]
            fn le(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner(lhs: &$uint, rhs: &$uint) -> $mask {
                    $mask($cmpeq(lhs.0, $min(lhs.0, rhs.0)))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner(lhs: $uint, rhs: $uint) -> $uint {
                    $uint($max(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn min(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner(lhs: $uint, rhs: $uint) -> $uint {
                    $uint($min(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }
    };
}

macro_rules! impl_ord_int {
    ($int:ident, $mask:ident, $cmpeq:ident, $cmplt:ident, $max:ident, $min:ident) => {
        impl LanesEq for $int {
            type Output = $mask;

            #[inline(always)]
            fn eq(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner(lhs: &$int, rhs: &$int) -> $mask {
                    $mask($cmpeq(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }

        impl LanesOrd for $int {
            #[inline(always)]
            fn lt(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner(lhs: &$int, rhs: &$int) -> $mask {
                    $mask($cmplt(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int($max(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn min(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int($min(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }
    };
}

macro_rules! impl_int_mul {
    ($int8:ident, $int16:ident, $int32:ident, $int64:ident) => {
        impl Mul for $int8 {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner(lhs: $int8, rhs: $int8) -> $int8 {
                    let lhs_odd = _mm_srli_epi16(lhs.0, 8);
                    let rhs_odd = _mm_srli_epi16(rhs.0, 8);
                    let even = _mm_mullo_epi16(lhs.0, rhs.0);
                    let odd = _mm_slli_epi16(_mm_mullo_epi16(lhs_odd, rhs_odd), 8);
                    let mask = _mm_set1_epi16(0x00FF);
                    $int8(_mm_blendv_epi8(odd, even, mask))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl Mul for $int16 {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse4.2")]
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
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner(lhs: $int32, rhs: $int32) -> $int32 {
                    $int32(_mm_mullo_epi32(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl Mul for $int64 {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "sse4.2")]
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

float_type! {
    "sse4.2",
    f32x4, __m128, f32, 4, m32x4,
    _mm_set1_ps, _mm_loadu_ps, _mm_storeu_ps, _mm_castps_si128, _mm_castsi128_ps, _mm_blendv_ps,
    _mm_cmpeq_ps, _mm_cmpneq_ps, _mm_cmplt_ps, _mm_cmple_ps, _mm_cmpgt_ps, _mm_cmpge_ps,
    _mm_min_ps, _mm_max_ps, _mm_add_ps, _mm_sub_ps, _mm_mul_ps, _mm_div_ps, _mm_xor_ps,
}
float_type! {
    "sse4.2",
    f64x2, __m128d, f64, 2, m64x2,
    _mm_set1_pd, _mm_loadu_pd, _mm_storeu_pd, _mm_castpd_si128, _mm_castsi128_pd, _mm_blendv_pd,
    _mm_cmpeq_pd, _mm_cmpneq_pd, _mm_cmplt_pd, _mm_cmple_pd, _mm_cmpgt_pd, _mm_cmpge_pd,
    _mm_min_pd, _mm_max_pd, _mm_add_pd, _mm_sub_pd, _mm_mul_pd, _mm_div_pd, _mm_xor_pd,
}
impl_convert32! { f32x4, Sse4_2Impl }
impl_convert64! { f64x2, Sse4_2Impl }

int_type! { "sse4.2", u8x16, u8, 16, m8x16, _mm_set1_epi8, _mm_blendv_epi8 }
int_type! { "sse4.2", u16x8, u16, 8, m16x8, _mm_set1_epi16, _mm_blendv_epi8 }
int_type! { "sse4.2", u32x4, u32, 4, m32x4, _mm_set1_epi32, _mm_blendv_epi8 }
int_type! { "sse4.2", u64x2, u64, 2, m64x2, _mm_set1_epi64x, _mm_blendv_epi8 }
impl_convert8! { u8x16, Sse4_2Impl }
impl_convert16! { u16x8, Sse4_2Impl }
impl_convert32! { u32x4, Sse4_2Impl }
impl_convert64! { u64x2, Sse4_2Impl }
impl_ord_uint! { u8x16, m8x16, _mm_cmpeq_epi8, _mm_max_epu8, _mm_min_epu8 }
impl_ord_uint! { u16x8, m16x8, _mm_cmpeq_epi16, _mm_max_epu16, _mm_min_epu16 }
impl_ord_uint! { u32x4, m32x4, _mm_cmpeq_epi32, _mm_max_epu32, _mm_min_epu32 }
impl_int! { "sse4.2", u8x16, u8, _mm_set1_epi8, _mm_add_epi8, _mm_sub_epi8, _mm_sll_epi8_fallback, _mm_srl_epi8_fallback }
impl_int! { "sse4.2", u16x8, u16, _mm_set1_epi16, _mm_add_epi16, _mm_sub_epi16, _mm_sll_epi16, _mm_srl_epi16 }
impl_int! { "sse4.2", u32x4, u32, _mm_set1_epi32, _mm_add_epi32, _mm_sub_epi32, _mm_sll_epi32, _mm_srl_epi32 }
impl_int! { "sse4.2", u64x2, u64, _mm_set1_epi64x, _mm_add_epi64, _mm_sub_epi64, _mm_sll_epi64, _mm_srl_epi64 }
impl_int_mul! { u8x16, u16x8, u32x4, u64x2 }

impl LanesEq for u64x2 {
    type Output = m64x2;

    #[inline(always)]
    fn eq(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse4.2")]
        unsafe fn inner(lhs: &u64x2, rhs: &u64x2) -> m64x2 {
            m64x2(_mm_cmpeq_epi64(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesOrd for u64x2 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse4.2")]
        unsafe fn inner(lhs: &u64x2, rhs: &u64x2) -> m64x2 {
            let bias = _mm_set1_epi64x(i64::MIN);
            m64x2(_mm_cmpgt_epi64(
                _mm_add_epi32(rhs.0, bias),
                _mm_add_epi32(lhs.0, bias),
            ))
        }

        unsafe { inner(self, other) }
    }
}

int_type! { "sse4.2", i8x16, i8, 16, m8x16, _mm_set1_epi8, _mm_blendv_epi8 }
int_type! { "sse4.2", i16x8, i16, 8, m16x8, _mm_set1_epi16, _mm_blendv_epi8 }
int_type! { "sse4.2", i32x4, i32, 4, m32x4, _mm_set1_epi32, _mm_blendv_epi8 }
int_type! { "sse4.2", i64x2, i64, 2, m64x2, _mm_set1_epi64x, _mm_blendv_epi8 }
impl_convert8! { i8x16, Sse4_2Impl }
impl_convert16! { i16x8, Sse4_2Impl }
impl_convert32! { i32x4, Sse4_2Impl }
impl_convert64! { i64x2, Sse4_2Impl }
impl_ord_int! { i8x16, m8x16, _mm_cmpeq_epi8, _mm_cmplt_epi8, _mm_max_epi8, _mm_min_epi8 }
impl_ord_int! { i16x8, m16x8, _mm_cmpeq_epi16, _mm_cmplt_epi16, _mm_max_epi16, _mm_min_epi16 }
impl_ord_int! { i32x4, m32x4, _mm_cmpeq_epi32, _mm_cmplt_epi32, _mm_max_epi32, _mm_min_epi32 }
impl_int! { "sse4.2", i8x16, i8, _mm_set1_epi8, _mm_add_epi8, _mm_sub_epi8, _mm_sll_epi8_fallback, _mm_sra_epi8_fallback }
impl_int! { "sse4.2", i16x8, i16, _mm_set1_epi16, _mm_add_epi16, _mm_sub_epi16, _mm_sll_epi16, _mm_sra_epi16 }
impl_int! { "sse4.2", i32x4, i32, _mm_set1_epi32, _mm_add_epi32, _mm_sub_epi32, _mm_sll_epi32, _mm_sra_epi32 }
impl_int! { "sse4.2", i64x2, i64, _mm_set1_epi64x, _mm_add_epi64, _mm_sub_epi64, _mm_sll_epi64, _mm_sra_epi64_fallback }
impl_int_mul! { i8x16, i16x8, i32x4, i64x2 }

impl LanesEq for i64x2 {
    type Output = m64x2;

    #[inline(always)]
    fn eq(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse4.2")]
        unsafe fn inner(lhs: &i64x2, rhs: &i64x2) -> m64x2 {
            m64x2(_mm_cmpeq_epi64(lhs.0, rhs.0))
        }

        unsafe { inner(self, other) }
    }
}

impl LanesOrd for i64x2 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "sse4.2")]
        unsafe fn inner(lhs: &i64x2, rhs: &i64x2) -> m64x2 {
            m64x2(_mm_cmpgt_epi64(rhs.0, lhs.0))
        }

        unsafe { inner(self, other) }
    }
}

int_type! { "sse4.2", m8x16, m8, 16, m8x16, _mm_set1_epi8, _mm_blendv_epi8 }
int_type! { "sse4.2", m16x8, m16, 8, m16x8, _mm_set1_epi16, _mm_blendv_epi8 }
int_type! { "sse4.2", m32x4, m32, 4, m32x4, _mm_set1_epi32, _mm_blendv_epi8 }
int_type! { "sse4.2", m64x2, m64, 2, m64x2, _mm_set1_epi64x, _mm_blendv_epi8 }
impl_ord_mask! { "sse4.2", m8x16 }
impl_ord_mask! { "sse4.2", m16x8 }
impl_ord_mask! { "sse4.2", m32x4 }
impl_ord_mask! { "sse4.2", m64x2 }
