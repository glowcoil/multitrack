#![allow(non_camel_case_types)]

use crate::mask::*;
use crate::{Arch, Bitwise, Float, Int, LanesEq, LanesOrd, Select, Simd};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use core::fmt::{self, Debug};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};
use std::mem;
use std::slice;

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

macro_rules! float_type {
    (
        $float:ident, $inner:ident, $elem:ident, $lanes:literal, $mask:ident,
        $set:ident, $load:ident, $store:ident, $cast_to_int:ident, $cast_from_int:ident, $blend:ident,
        $cmpeq:ident, $cmpneq:ident, $cmplt:ident, $cmple:ident, $cmpgt:ident, $cmpge:ident,
        $min:ident, $max:ident, $add:ident, $sub:ident, $mul:ident, $div:ident, $xor:ident,
    ) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct $float($inner);

        impl Simd for $float {
            type Elem = $elem;
            type Mask = $mask;

            const LANES: usize = $lanes;

            fn new(elem: Self::Elem) -> Self {
                unsafe { $float($set(elem)) }
            }

            fn as_slice(&self) -> &[Self::Elem] {
                unsafe {
                    slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES)
                }
            }

            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                unsafe {
                    slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES)
                }
            }

            fn from_slice(slice: &[Self::Elem]) -> Self {
                assert!(slice.len() == Self::LANES);
                unsafe { $float($load(slice.as_ptr())) }
            }

            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                assert!(slice.len() == Self::LANES);
                unsafe {
                    $store(slice.as_mut_ptr(), self.0);
                }
            }

            fn align_slice(slice: &[Self::Elem]) -> (&[Self::Elem], &[Self], &[Self::Elem]) {
                unsafe { slice.align_to::<Self>() }
            }

            fn align_mut_slice(
                slice: &mut [Self::Elem],
            ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]) {
                unsafe { slice.align_to_mut::<Self>() }
            }
        }

        impl Debug for $float {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(self.as_slice(), fmt)
            }
        }

        impl Default for $float {
            fn default() -> Self {
                unsafe { mem::zeroed() }
            }
        }

        impl LanesEq for $float {
            type Output = $mask;

            fn eq(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmpeq(self.0, other.0))) }
            }

            fn ne(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmpneq(self.0, other.0))) }
            }
        }

        impl LanesOrd for $float {
            fn lt(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmplt(self.0, other.0))) }
            }

            fn le(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmple(self.0, other.0))) }
            }

            fn gt(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmpgt(self.0, other.0))) }
            }

            fn ge(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmpge(self.0, other.0))) }
            }

            fn max(self, other: Self) -> Self {
                unsafe { $float($max(self.0, other.0)) }
            }

            fn min(self, other: Self) -> Self {
                unsafe { $float($min(self.0, other.0)) }
            }
        }

        impl Index<usize> for $float {
            type Output = <Self as Simd>::Elem;

            fn index(&self, index: usize) -> &Self::Output {
                &self.as_slice()[index]
            }
        }

        impl IndexMut<usize> for $float {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }

        impl Select<$float> for $mask {
            fn select(self, if_true: $float, if_false: $float) -> $float {
                unsafe {
                    let mask = $cast_from_int(self.0);
                    $float($blend(if_false.0, if_true.0, mask))
                }
            }
        }

        impl Float for $float {}

        impl Add for $float {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                unsafe { $float($add(self.0, rhs.0)) }
            }
        }

        impl AddAssign for $float {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Sub for $float {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                unsafe { $float($sub(self.0, rhs.0)) }
            }
        }

        impl SubAssign for $float {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl Mul for $float {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                unsafe { $float($mul(self.0, rhs.0)) }
            }
        }

        impl MulAssign for $float {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Div for $float {
            type Output = Self;

            fn div(self, rhs: Self) -> Self {
                unsafe { $float($div(self.0, rhs.0)) }
            }
        }

        impl DivAssign for $float {
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }

        impl Neg for $float {
            type Output = Self;

            fn neg(self) -> Self {
                unsafe { $float($xor(self.0, $set(-0.0))) }
            }
        }
    };
}

macro_rules! int_type {
    ($int:ident, $elem:ident, $lanes:literal, $mask:ident, $set:ident, $cmp:ident) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct $int(__m128i);

        impl Simd for $int {
            type Elem = $elem;
            type Mask = $mask;

            const LANES: usize = $lanes;

            fn new(elem: Self::Elem) -> Self {
                unsafe { $int($set(mem::transmute(elem))) }
            }

            fn as_slice(&self) -> &[Self::Elem] {
                unsafe {
                    slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES)
                }
            }

            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                unsafe {
                    slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES)
                }
            }

            fn from_slice(slice: &[Self::Elem]) -> Self {
                assert!(slice.len() == Self::LANES);
                unsafe { $int(_mm_loadu_si128(slice.as_ptr() as *const __m128i)) }
            }

            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                assert!(slice.len() == Self::LANES);
                unsafe {
                    _mm_storeu_si128(slice.as_mut_ptr() as *mut __m128i, self.0);
                }
            }

            fn align_slice(slice: &[Self::Elem]) -> (&[Self::Elem], &[Self], &[Self::Elem]) {
                unsafe { slice.align_to::<Self>() }
            }

            fn align_mut_slice(
                slice: &mut [Self::Elem],
            ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]) {
                unsafe { slice.align_to_mut::<Self>() }
            }
        }

        impl Debug for $int {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(self.as_slice(), fmt)
            }
        }

        impl Default for $int {
            fn default() -> Self {
                unsafe { mem::zeroed() }
            }
        }

        impl LanesEq for $int {
            type Output = $mask;

            fn eq(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cmp(self.0, other.0)) }
            }
        }

        impl Index<usize> for $int {
            type Output = <Self as Simd>::Elem;

            fn index(&self, index: usize) -> &Self::Output {
                &self.as_slice()[index]
            }
        }

        impl IndexMut<usize> for $int {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }

        impl Bitwise for $int {}

        impl BitAnd for $int {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self::Output {
                unsafe { $int(_mm_and_si128(self.0, rhs.0)) }
            }
        }

        impl BitAndAssign for $int {
            fn bitand_assign(&mut self, rhs: Self) {
                *self = *self & rhs;
            }
        }

        impl BitOr for $int {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self::Output {
                unsafe { $int(_mm_or_si128(self.0, rhs.0)) }
            }
        }

        impl BitOrAssign for $int {
            fn bitor_assign(&mut self, rhs: Self) {
                *self = *self | rhs;
            }
        }

        impl BitXor for $int {
            type Output = Self;

            fn bitxor(self, rhs: Self) -> Self::Output {
                unsafe { $int(_mm_xor_si128(self.0, rhs.0)) }
            }
        }

        impl BitXorAssign for $int {
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = *self ^ rhs;
            }
        }

        impl Not for $int {
            type Output = Self;

            fn not(self) -> Self::Output {
                unsafe {
                    let zero = _mm_setzero_si128();
                    $int(_mm_andnot_si128(self.0, _mm_cmpeq_epi8(zero, zero)))
                }
            }
        }

        impl Select<$int> for $mask {
            fn select(self, if_true: $int, if_false: $int) -> $int {
                unsafe { $int(_mm_blendv_epi8(if_false.0, if_true.0, self.0)) }
            }
        }
    };
}

macro_rules! impl_ord_uint {
    ($uint:ident, $mask:ident, $cmpeq:ident, $max:ident, $min:ident) => {
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
    ($int:ident, $mask:ident, $cmplt:ident, $max:ident, $min:ident) => {
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

macro_rules! impl_ord_mask {
    ($mask:ident) => {
        impl LanesOrd for $mask {
            fn lt(&self, other: &Self) -> Self::Output {
                unsafe { $mask(_mm_andnot_si128(self.0, other.0)) }
            }

            fn le(&self, other: &Self) -> Self::Output {
                unsafe { $mask(_mm_or_si128(other.0, _mm_cmpeq_epi8(self.0, other.0))) }
            }

            fn max(self, other: Self) -> Self {
                unsafe { $mask(_mm_or_si128(self.0, other.0)) }
            }

            fn min(self, other: Self) -> Self {
                unsafe { $mask(_mm_and_si128(self.0, other.0)) }
            }
        }
    };
}

macro_rules! impl_int {
    ($int:ident, $set:ident, $add:ident, $sub:ident) => {
        impl Int for $int {}

        impl Add for $int {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                unsafe { $int($add(self.0, rhs.0)) }
            }
        }

        impl AddAssign for $int {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Sub for $int {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                unsafe { $int($sub(self.0, rhs.0)) }
            }
        }

        impl SubAssign for $int {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl MulAssign for $int {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Neg for $int {
            type Output = Self;

            fn neg(self) -> Self {
                unsafe { $int($sub($set(0), self.0)) }
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
                    let low_high = _mm_mullo_epi32(self.0, _mm_slli_epi64(rhs.0, 32));
                    let high_low = _mm_mullo_epi32(rhs.0, _mm_slli_epi64(self.0, 32));
                    let low_low = _mm_mul_epu32(self.0, rhs.0);
                    let high = _mm_add_epi32(low_high, high_low);
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

int_type! { u8x16, u8, 16, m8x16, _mm_set1_epi8, _mm_cmpeq_epi8 }
int_type! { u16x8, u16, 8, m16x8, _mm_set1_epi16, _mm_cmpeq_epi16 }
int_type! { u32x4, u32, 4, m32x4, _mm_set1_epi32, _mm_cmpeq_epi32 }
int_type! { u64x2, u64, 2, m64x2, _mm_set1_epi64x, _mm_cmpeq_epi64 }
impl_ord_uint! { u8x16, m8x16, _mm_cmpeq_epi8, _mm_max_epu8, _mm_min_epu8 }
impl_ord_uint! { u16x8, m16x8, _mm_cmpeq_epi16, _mm_max_epu16, _mm_min_epu16 }
impl_ord_uint! { u32x4, m32x4, _mm_cmpeq_epi32, _mm_max_epu32, _mm_min_epu32 }
impl_int! { u8x16, _mm_set1_epi8, _mm_add_epi8, _mm_sub_epi8 }
impl_int! { u16x8, _mm_set1_epi16, _mm_add_epi16, _mm_sub_epi16 }
impl_int! { u32x4, _mm_set1_epi32, _mm_add_epi32, _mm_sub_epi32 }
impl_int! { u64x2, _mm_set1_epi64x, _mm_add_epi64, _mm_sub_epi64 }
impl_int_mul! { u8x16, u16x8, u32x4, u64x2 }

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

int_type! { i8x16, i8, 16, m8x16, _mm_set1_epi8, _mm_cmpeq_epi8 }
int_type! { i16x8, i16, 8, m16x8, _mm_set1_epi16, _mm_cmpeq_epi16 }
int_type! { i32x4, i32, 4, m32x4, _mm_set1_epi32, _mm_cmpeq_epi32 }
int_type! { i64x2, i64, 2, m64x2, _mm_set1_epi64x, _mm_cmpeq_epi64 }
impl_ord_int! { i8x16, m8x16, _mm_cmplt_epi8, _mm_max_epi8, _mm_min_epi8 }
impl_ord_int! { i16x8, m16x8, _mm_cmplt_epi16, _mm_max_epi16, _mm_min_epi16 }
impl_ord_int! { i32x4, m32x4, _mm_cmplt_epi32, _mm_max_epi32, _mm_min_epi32 }
impl_int! { i8x16, _mm_set1_epi8, _mm_add_epi8, _mm_sub_epi8 }
impl_int! { i16x8, _mm_set1_epi16, _mm_add_epi16, _mm_sub_epi16 }
impl_int! { i32x4, _mm_set1_epi32, _mm_add_epi32, _mm_sub_epi32 }
impl_int! { i64x2, _mm_set1_epi64x, _mm_add_epi64, _mm_sub_epi64 }
impl_int_mul! { i8x16, i16x8, i32x4, i64x2 }

impl LanesOrd for i64x2 {
    fn lt(&self, other: &Self) -> Self::Output {
        unsafe { m64x2(_mm_cmpgt_epi64(other.0, self.0)) }
    }
}

int_type! { m8x16, m8, 16, m8x16, _mm_set1_epi8, _mm_cmpeq_epi8 }
int_type! { m16x8, m16, 8, m16x8, _mm_set1_epi16, _mm_cmpeq_epi16 }
int_type! { m32x4, m32, 4, m32x4, _mm_set1_epi32, _mm_cmpeq_epi32 }
int_type! { m64x2, m64, 2, m64x2, _mm_set1_epi64x, _mm_cmpeq_epi64 }
impl_ord_mask! { m8x16 }
impl_ord_mask! { m16x8 }
impl_ord_mask! { m32x4 }
impl_ord_mask! { m64x2 }
