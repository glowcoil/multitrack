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
use crate::simd::{Arch, Bitwise, Float, Int, LanesEq, LanesOrd, Select, Simd};

pub struct Sse4_2;

impl Arch for Sse4_2 {
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
}

macro_rules! impl_ord_uint {
    ($uint:ident, $mask:ident, $cmpeq:ident, $max:ident, $min:ident) => {
        impl LanesEq for $uint {
            type Output = $mask;

            fn eq(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cmpeq(self.0, other.0)) }
            }
        }

        impl LanesOrd for $uint {
            fn lt(&self, other: &Self) -> Self::Output {
                !other.le(self)
            }

            fn le(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cmpeq(self.0, $min(self.0, other.0))) }
            }

            fn max(self, other: Self) -> Self {
                unsafe { $uint($max(self.0, other.0)) }
            }

            fn min(self, other: Self) -> Self {
                unsafe { $uint($min(self.0, other.0)) }
            }
        }
    };
}

macro_rules! impl_ord_int {
    ($int:ident, $mask:ident, $cmpeq:ident, $cmplt:ident, $max:ident, $min:ident) => {
        impl LanesEq for $int {
            type Output = $mask;

            fn eq(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cmpeq(self.0, other.0)) }
            }
        }

        impl LanesOrd for $int {
            fn lt(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cmplt(self.0, other.0)) }
            }

            fn max(self, other: Self) -> Self {
                unsafe { $int($max(self.0, other.0)) }
            }

            fn min(self, other: Self) -> Self {
                unsafe { $int($min(self.0, other.0)) }
            }
        }
    };
}

macro_rules! impl_int_mul {
    ($int8:ident, $int16:ident, $int32:ident, $int64:ident) => {
        impl Mul for $int8 {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                unsafe {
                    let lhs_odd = _mm_srli_epi16(self.0, 8);
                    let rhs_odd = _mm_srli_epi16(rhs.0, 8);
                    let even = _mm_mullo_epi16(self.0, rhs.0);
                    let odd = _mm_slli_epi16(_mm_mullo_epi16(lhs_odd, rhs_odd), 8);
                    let mask = _mm_set1_epi16(0x00FF);
                    $int8(_mm_blendv_epi8(odd, even, mask))
                }
            }
        }

        impl Mul for $int16 {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                unsafe { $int16(_mm_mullo_epi16(self.0, rhs.0)) }
            }
        }

        impl Mul for $int32 {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                unsafe { $int32(_mm_mullo_epi32(self.0, rhs.0)) }
            }
        }

        impl Mul for $int64 {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                unsafe {
                    let low_high = _mm_mul_epu32(self.0, _mm_srli_epi64(rhs.0, 32));
                    let high_low = _mm_mul_epu32(rhs.0, _mm_srli_epi64(self.0, 32));
                    let low_low = _mm_mul_epu32(self.0, rhs.0);
                    let high = _mm_slli_epi64(_mm_add_epi32(low_high, high_low), 32);
                    $int64(_mm_add_epi32(low_low, high))
                }
            }
        }
    };
}

float_type! {
    f32x4, __m128, f32, 4, m32x4,
    _mm_set1_ps, _mm_loadu_ps, _mm_storeu_ps, _mm_castps_si128, _mm_castsi128_ps, _mm_blendv_ps,
    _mm_cmpeq_ps, _mm_cmpneq_ps, _mm_cmplt_ps, _mm_cmple_ps, _mm_cmpgt_ps, _mm_cmpge_ps,
    _mm_min_ps, _mm_max_ps, _mm_add_ps, _mm_sub_ps, _mm_mul_ps, _mm_div_ps, _mm_xor_ps,
}
float_type! {
    f64x2, __m128d, f64, 2, m64x2,
    _mm_set1_pd, _mm_loadu_pd, _mm_storeu_pd, _mm_castpd_si128, _mm_castsi128_pd, _mm_blendv_pd,
    _mm_cmpeq_pd, _mm_cmpneq_pd, _mm_cmplt_pd, _mm_cmple_pd, _mm_cmpgt_pd, _mm_cmpge_pd,
    _mm_min_pd, _mm_max_pd, _mm_add_pd, _mm_sub_pd, _mm_mul_pd, _mm_div_pd, _mm_xor_pd,
}

int_type! { u8x16, u8, 16, m8x16, _mm_set1_epi8, _mm_blendv_epi8 }
int_type! { u16x8, u16, 8, m16x8, _mm_set1_epi16, _mm_blendv_epi8 }
int_type! { u32x4, u32, 4, m32x4, _mm_set1_epi32, _mm_blendv_epi8 }
int_type! { u64x2, u64, 2, m64x2, _mm_set1_epi64x, _mm_blendv_epi8 }
impl_ord_uint! { u8x16, m8x16, _mm_cmpeq_epi8, _mm_max_epu8, _mm_min_epu8 }
impl_ord_uint! { u16x8, m16x8, _mm_cmpeq_epi16, _mm_max_epu16, _mm_min_epu16 }
impl_ord_uint! { u32x4, m32x4, _mm_cmpeq_epi32, _mm_max_epu32, _mm_min_epu32 }
impl_int! { u8x16, _mm_set1_epi8, _mm_add_epi8, _mm_sub_epi8 }
impl_int! { u16x8, _mm_set1_epi16, _mm_add_epi16, _mm_sub_epi16 }
impl_int! { u32x4, _mm_set1_epi32, _mm_add_epi32, _mm_sub_epi32 }
impl_int! { u64x2, _mm_set1_epi64x, _mm_add_epi64, _mm_sub_epi64 }
impl_int_mul! { u8x16, u16x8, u32x4, u64x2 }

impl LanesEq for u64x2 {
    type Output = m64x2;

    fn eq(&self, other: &Self) -> Self::Output {
        unsafe { m64x2(_mm_cmpeq_epi64(self.0, other.0)) }
    }
}

impl LanesOrd for u64x2 {
    fn lt(&self, other: &Self) -> Self::Output {
        unsafe {
            let bias = _mm_set1_epi64x(i64::MIN);
            m64x2(_mm_cmpgt_epi64(
                _mm_add_epi32(other.0, bias),
                _mm_add_epi32(self.0, bias),
            ))
        }
    }
}

int_type! { i8x16, i8, 16, m8x16, _mm_set1_epi8, _mm_blendv_epi8 }
int_type! { i16x8, i16, 8, m16x8, _mm_set1_epi16, _mm_blendv_epi8 }
int_type! { i32x4, i32, 4, m32x4, _mm_set1_epi32, _mm_blendv_epi8 }
int_type! { i64x2, i64, 2, m64x2, _mm_set1_epi64x, _mm_blendv_epi8 }
impl_ord_int! { i8x16, m8x16, _mm_cmpeq_epi8, _mm_cmplt_epi8, _mm_max_epi8, _mm_min_epi8 }
impl_ord_int! { i16x8, m16x8, _mm_cmpeq_epi16, _mm_cmplt_epi16, _mm_max_epi16, _mm_min_epi16 }
impl_ord_int! { i32x4, m32x4, _mm_cmpeq_epi32, _mm_cmplt_epi32, _mm_max_epi32, _mm_min_epi32 }
impl_int! { i8x16, _mm_set1_epi8, _mm_add_epi8, _mm_sub_epi8 }
impl_int! { i16x8, _mm_set1_epi16, _mm_add_epi16, _mm_sub_epi16 }
impl_int! { i32x4, _mm_set1_epi32, _mm_add_epi32, _mm_sub_epi32 }
impl_int! { i64x2, _mm_set1_epi64x, _mm_add_epi64, _mm_sub_epi64 }
impl_int_mul! { i8x16, i16x8, i32x4, i64x2 }

impl LanesEq for i64x2 {
    type Output = m64x2;

    fn eq(&self, other: &Self) -> Self::Output {
        unsafe { m64x2(_mm_cmpeq_epi64(self.0, other.0)) }
    }
}

impl LanesOrd for i64x2 {
    fn lt(&self, other: &Self) -> Self::Output {
        unsafe { m64x2(_mm_cmpgt_epi64(other.0, self.0)) }
    }
}

int_type! { m8x16, m8, 16, m8x16, _mm_set1_epi8, _mm_blendv_epi8 }
int_type! { m16x8, m16, 8, m16x8, _mm_set1_epi16, _mm_blendv_epi8 }
int_type! { m32x4, m32, 4, m32x4, _mm_set1_epi32, _mm_blendv_epi8 }
int_type! { m64x2, m64, 2, m64x2, _mm_set1_epi64x, _mm_blendv_epi8 }
impl_ord_mask! { m8x16 }
impl_ord_mask! { m16x8 }
impl_ord_mask! { m32x4 }
impl_ord_mask! { m64x2 }
