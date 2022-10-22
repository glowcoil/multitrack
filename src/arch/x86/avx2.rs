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

use crate::mask::*;
use crate::simd::{Bitwise, Float, Int, LanesEq, LanesOrd, Select, Simd};
use crate::{Arch, Task};

pub struct Avx2Impl;

impl Arch for Avx2Impl {
    type f32 = f32x8;
    type f64 = f64x4;

    type u8 = u8x32;
    type u16 = u16x16;
    type u32 = u32x8;
    type u64 = u64x4;

    type i8 = i8x32;
    type i16 = i16x16;
    type i32 = i32x8;
    type i64 = i64x4;

    type m8 = m8x32;
    type m16 = m16x16;
    type m32 = m32x8;
    type m64 = m64x4;

    const NAME: &'static str = "avx2";

    #[inline(always)]
    fn invoke<T: Task>(task: T) -> T::Result {
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn inner<T: Task>(task: T) -> T::Result {
            task.run::<Avx2Impl>()
        }

        unsafe { inner(task) }
    }
}

macro_rules! float_type {
    (
        $float:ident, $inner:ident, $elem:ident, $lanes:literal, $mask:ident,
        $set:ident, $load:ident, $store:ident, $cast_to_int:ident, $cast_from_int:ident, $blend:ident,
        $cmp:ident, $max:ident, $min:ident, $add:ident, $sub:ident, $mul:ident, $div:ident, $xor:ident,
    ) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct $float($inner);

        impl Simd for $float {
            type Elem = $elem;
            type Mask = $mask;

            const LANES: usize = $lanes;

            #[inline(always)]
            fn new(elem: Self::Elem) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(elem: $elem) -> $float {
                    $float($set(elem))
                }

                unsafe { inner(elem) }
            }

            #[inline]
            fn as_slice(&self) -> &[Self::Elem] {
                unsafe {
                    slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES)
                }
            }

            #[inline]
            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                unsafe {
                    slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES)
                }
            }

            #[inline(always)]
            fn from_slice(slice: &[Self::Elem]) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(slice: &[<$float as Simd>::Elem]) -> $float {
                    assert!(slice.len() == <$float as Simd>::LANES);
                    $float($load(slice.as_ptr()))
                }

                unsafe { inner(slice) }
            }

            #[inline(always)]
            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(vec: &$float, slice: &mut [$elem]) {
                    assert!(slice.len() == <$float as Simd>::LANES);
                    $store(slice.as_mut_ptr(), vec.0);
                }

                unsafe { inner(self, slice) }
            }

            #[inline]
            fn align_slice(slice: &[Self::Elem]) -> (&[Self::Elem], &[Self], &[Self::Elem]) {
                unsafe { slice.align_to::<Self>() }
            }

            #[inline]
            fn align_mut_slice(
                slice: &mut [Self::Elem],
            ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]) {
                unsafe { slice.align_to_mut::<Self>() }
            }
        }

        impl Debug for $float {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(self.as_slice(), fmt)
            }
        }

        impl Default for $float {
            #[inline]
            fn default() -> Self {
                unsafe { mem::zeroed() }
            }
        }

        impl LanesEq for $float {
            type Output = $mask;

            #[inline(always)]
            fn eq(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    let res = $cmp(lhs.0, rhs.0, _CMP_EQ_OQ);
                    $mask($cast_to_int(res))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn ne(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    let res = $cmp(lhs.0, rhs.0, _CMP_NEQ_UQ);
                    $mask($cast_to_int(res))
                }

                unsafe { inner(self, other) }
            }
        }

        impl LanesOrd for $float {
            #[inline(always)]
            fn lt(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    let res = $cmp(lhs.0, rhs.0, _CMP_LT_OQ);
                    $mask($cast_to_int(res))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn le(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    let res = $cmp(lhs.0, rhs.0, _CMP_LE_OQ);
                    $mask($cast_to_int(res))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn gt(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    let res = $cmp(lhs.0, rhs.0, _CMP_GT_OQ);
                    $mask($cast_to_int(res))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn ge(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    let res = $cmp(lhs.0, rhs.0, _CMP_GE_OQ);
                    $mask($cast_to_int(res))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($max(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn min(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($min(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }

        impl Index<usize> for $float {
            type Output = <Self as Simd>::Elem;

            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.as_slice()[index]
            }
        }

        impl IndexMut<usize> for $float {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }

        impl Select<$float> for $mask {
            #[inline(always)]
            fn select(self, if_true: $float, if_false: $float) -> $float {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(mask: $mask, if_true: $float, if_false: $float) -> $float {
                    let mask = $cast_from_int(mask.0);
                    $float($blend(if_false.0, if_true.0, mask))
                }

                unsafe { inner(self, if_true, if_false) }
            }
        }

        impl Float for $float {}

        impl Add for $float {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($add(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl AddAssign for $float {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Sub for $float {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($sub(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl SubAssign for $float {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl Mul for $float {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($mul(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl MulAssign for $float {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Div for $float {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($div(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl DivAssign for $float {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }

        impl Neg for $float {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(vec: $float) -> $float {
                    $float($xor(vec.0, $set(-0.0)))
                }

                unsafe { inner(self) }
            }
        }
    };
}

macro_rules! int_type {
    ($int:ident, $elem:ident, $lanes:literal, $mask:ident, $set:ident, $cmp:ident) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct $int(__m256i);

        impl Simd for $int {
            type Elem = $elem;
            type Mask = $mask;

            const LANES: usize = $lanes;

            #[inline(always)]
            fn new(elem: Self::Elem) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(elem: $elem) -> $int {
                    $int($set(mem::transmute(elem)))
                }

                unsafe { inner(elem) }
            }

            #[inline]
            fn as_slice(&self) -> &[Self::Elem] {
                unsafe {
                    slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES)
                }
            }

            #[inline]
            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                unsafe {
                    slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES)
                }
            }

            #[inline(always)]
            fn from_slice(slice: &[Self::Elem]) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(slice: &[$elem]) -> $int {
                    assert!(slice.len() == <$int as Simd>::LANES);
                    $int(_mm256_loadu_si256(slice.as_ptr() as *const __m256i))
                }

                unsafe { inner(slice) }
            }

            #[inline(always)]
            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(vec: &$int, slice: &mut [$elem]) {
                    assert!(slice.len() == <$int as Simd>::LANES);
                    _mm256_storeu_si256(slice.as_mut_ptr() as *mut __m256i, vec.0);
                }

                unsafe {
                    inner(self, slice);
                }
            }

            #[inline]
            fn align_slice(slice: &[Self::Elem]) -> (&[Self::Elem], &[Self], &[Self::Elem]) {
                unsafe { slice.align_to::<Self>() }
            }

            #[inline]
            fn align_mut_slice(
                slice: &mut [Self::Elem],
            ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]) {
                unsafe { slice.align_to_mut::<Self>() }
            }
        }

        impl Debug for $int {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(self.as_slice(), fmt)
            }
        }

        impl Default for $int {
            #[inline]
            fn default() -> Self {
                unsafe { mem::zeroed() }
            }
        }

        impl LanesEq for $int {
            type Output = $mask;

            #[inline(always)]
            fn eq(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$int, rhs: &$int) -> $mask {
                    $mask($cmp(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }

        impl Index<usize> for $int {
            type Output = <Self as Simd>::Elem;

            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.as_slice()[index]
            }
        }

        impl IndexMut<usize> for $int {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }

        impl Bitwise for $int {}

        impl BitAnd for $int {
            type Output = Self;

            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int(_mm256_and_si256(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl BitAndAssign for $int {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = *self & rhs;
            }
        }

        impl BitOr for $int {
            type Output = Self;

            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int(_mm256_or_si256(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl BitOrAssign for $int {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = *self | rhs;
            }
        }

        impl BitXor for $int {
            type Output = Self;

            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int(_mm256_xor_si256(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl BitXorAssign for $int {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = *self ^ rhs;
            }
        }

        impl Not for $int {
            type Output = Self;

            #[inline(always)]
            fn not(self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(vec: $int) -> $int {
                    let zero = _mm256_setzero_si256();
                    $int(_mm256_andnot_si256(vec.0, _mm256_cmpeq_epi8(zero, zero)))
                }

                unsafe { inner(self) }
            }
        }

        impl Select<$int> for $mask {
            #[inline(always)]
            fn select(self, if_true: $int, if_false: $int) -> $int {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(mask: $mask, if_true: $int, if_false: $int) -> $int {
                    $int(_mm256_blendv_epi8(if_false.0, if_true.0, mask.0))
                }

                unsafe { inner(self, if_true, if_false) }
            }
        }
    };
}

macro_rules! impl_ord_uint {
    ($uint:ident, $mask:ident, $cmpeq:ident, $max:ident, $min:ident) => {
        impl LanesOrd for $uint {
            #[inline(always)]
            fn lt(&self, other: &Self) -> Self::Output {
                !other.le(self)
            }

            #[inline(always)]
            fn le(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$uint, rhs: &$uint) -> $mask {
                    $mask($cmpeq(lhs.0, $min(lhs.0, rhs.0)))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $uint, rhs: $uint) -> $uint {
                    $uint($max(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn min(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $uint, rhs: $uint) -> $uint {
                    $uint($min(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }
    };
}

macro_rules! impl_ord_int {
    ($int:ident, $mask:ident, $cmpgt:ident, $max:ident, $min:ident) => {
        impl LanesOrd for $int {
            #[inline(always)]
            fn lt(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$int, rhs: &$int) -> $mask {
                    $mask($cmpgt(rhs.0, lhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int($max(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn min(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int($min(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }
    };
}

macro_rules! impl_ord_mask {
    ($mask:ident) => {
        impl LanesOrd for $mask {
            #[inline(always)]
            fn lt(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$mask, rhs: &$mask) -> $mask {
                    $mask(_mm256_andnot_si256(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn le(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: &$mask, rhs: &$mask) -> $mask {
                    $mask(_mm256_or_si256(rhs.0, _mm256_cmpeq_epi8(lhs.0, rhs.0)))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $mask, rhs: $mask) -> $mask {
                    $mask(_mm256_or_si256(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn min(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $mask, rhs: $mask) -> $mask {
                    $mask(_mm256_and_si256(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }
    };
}

macro_rules! impl_int {
    ($int:ident, $set:ident, $add:ident, $sub:ident) => {
        impl Int for $int {}

        impl Add for $int {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int($add(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl AddAssign for $int {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Sub for $int {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int($sub(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl SubAssign for $int {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl MulAssign for $int {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Neg for $int {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(vec: $int) -> $int {
                    $int($sub($set(0), vec.0))
                }

                unsafe { inner(self) }
            }
        }

        impl Shl<usize> for $int {
            type Output = Self;

            #[inline]
            fn shl(self, _rhs: usize) -> Self {
                unimplemented!()
            }
        }

        impl ShlAssign<usize> for $int {
            #[inline]
            fn shl_assign(&mut self, rhs: usize) {
                *self = *self << rhs;
            }
        }

        impl Shr<usize> for $int {
            type Output = Self;

            #[inline]
            fn shr(self, _rhs: usize) -> Self {
                unimplemented!()
            }
        }

        impl ShrAssign<usize> for $int {
            #[inline]
            fn shr_assign(&mut self, rhs: usize) {
                *self = *self >> rhs;
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
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int8, rhs: $int8) -> $int8 {
                    let lhs_odd = _mm256_srli_epi16(lhs.0, 8);
                    let rhs_odd = _mm256_srli_epi16(rhs.0, 8);
                    let even = _mm256_mullo_epi16(lhs.0, rhs.0);
                    let odd = _mm256_slli_epi16(_mm256_mullo_epi16(lhs_odd, rhs_odd), 8);
                    let mask = _mm256_set1_epi32(0xFF00FF00u32 as i32);
                    $int8(_mm256_blendv_epi8(even, odd, mask))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl Mul for $int16 {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int16, rhs: $int16) -> $int16 {
                    $int16(_mm256_mullo_epi16(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl Mul for $int32 {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int32, rhs: $int32) -> $int32 {
                    $int32(_mm256_mullo_epi32(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl Mul for $int64 {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = "avx2")]
                unsafe fn inner(lhs: $int64, rhs: $int64) -> $int64 {
                    let low_high = _mm256_mullo_epi32(lhs.0, _mm256_slli_epi64(rhs.0, 32));
                    let high_low = _mm256_mullo_epi32(rhs.0, _mm256_slli_epi64(lhs.0, 32));
                    let low_low = _mm256_mul_epu32(lhs.0, rhs.0);
                    let high = _mm256_add_epi32(low_high, high_low);
                    $int64(_mm256_add_epi32(low_low, high))
                }

                unsafe { inner(self, rhs) }
            }
        }
    };
}

float_type! {
    f32x8, __m256, f32, 8, m32x8,
    _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_blendv_ps,
    _mm256_cmp_ps, _mm256_max_ps, _mm256_min_ps, _mm256_add_ps, _mm256_sub_ps, _mm256_mul_ps, _mm256_div_ps, _mm256_xor_ps,
}
float_type! {
    f64x4, __m256d, f64, 4, m64x4,
    _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd, _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_blendv_pd,
    _mm256_cmp_pd, _mm256_max_pd, _mm256_min_pd, _mm256_add_pd, _mm256_sub_pd, _mm256_mul_pd, _mm256_div_pd, _mm256_xor_pd,
}

int_type! { u8x32, u8, 32, m8x32, _mm256_set1_epi8, _mm256_cmpeq_epi8 }
int_type! { u16x16, u16, 16, m16x16, _mm256_set1_epi16, _mm256_cmpeq_epi16 }
int_type! { u32x8, u32, 8, m32x8, _mm256_set1_epi32, _mm256_cmpeq_epi32 }
int_type! { u64x4, u64, 4, m64x4, _mm256_set1_epi64x, _mm256_cmpeq_epi64 }
impl_ord_uint! { u8x32, m8x32, _mm256_cmpeq_epi8, _mm256_max_epu8, _mm256_min_epu8 }
impl_ord_uint! { u16x16, m16x16, _mm256_cmpeq_epi16, _mm256_max_epu16, _mm256_min_epu16 }
impl_ord_uint! { u32x8, m32x8, _mm256_cmpeq_epi32, _mm256_max_epu32, _mm256_min_epu32 }
impl_int! { u8x32, _mm256_set1_epi8, _mm256_add_epi8, _mm256_sub_epi8 }
impl_int! { u16x16, _mm256_set1_epi16, _mm256_add_epi16, _mm256_sub_epi16 }
impl_int! { u32x8, _mm256_set1_epi32, _mm256_add_epi32, _mm256_sub_epi32 }
impl_int! { u64x4, _mm256_set1_epi64x, _mm256_add_epi64, _mm256_sub_epi64 }
impl_int_mul! { u8x32, u16x16, u32x8, u64x4 }

// AVX2 lacks unsigned integer compares, but it does have unsigned integer min/max for 8, 16, and
// 32 bits. The impl_ord_uint macro thus implements le in terms of min and cmpeq. However, 64-bit
// unsigned integer min/max ops (_mm256_{min,max}_epu64) are only available on AVX512, so for u64x4
// we have to use a biased signed comparison (and then fall back to the default impls of min/max in
// terms of le and select).
impl LanesOrd for u64x4 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn inner(lhs: &u64x4, rhs: &u64x4) -> m64x4 {
            let bias = _mm256_set1_epi64x(i64::MIN);
            m64x4(_mm256_cmpgt_epi64(
                _mm256_add_epi32(rhs.0, bias),
                _mm256_add_epi32(lhs.0, bias),
            ))
        }

        unsafe { inner(self, other) }
    }
}

int_type! { i8x32, i8, 32, m8x32, _mm256_set1_epi8, _mm256_cmpeq_epi8 }
int_type! { i16x16, i16, 16, m16x16, _mm256_set1_epi16, _mm256_cmpeq_epi16 }
int_type! { i32x8, i32, 8, m32x8, _mm256_set1_epi32, _mm256_cmpeq_epi32 }
int_type! { i64x4, i64, 4, m64x4, _mm256_set1_epi64x, _mm256_cmpeq_epi64 }
impl_ord_int! { i8x32, m8x32, _mm256_cmpgt_epi8, _mm256_max_epi8, _mm256_min_epi8 }
impl_ord_int! { i16x16, m16x16, _mm256_cmpgt_epi16, _mm256_max_epi16, _mm256_min_epi16 }
impl_ord_int! { i32x8, m32x8, _mm256_cmpgt_epi32, _mm256_max_epi32, _mm256_min_epi32 }
impl_int! { i8x32, _mm256_set1_epi8, _mm256_add_epi8, _mm256_sub_epi8 }
impl_int! { i16x16, _mm256_set1_epi16, _mm256_add_epi16, _mm256_sub_epi16 }
impl_int! { i32x8, _mm256_set1_epi32, _mm256_add_epi32, _mm256_sub_epi32 }
impl_int! { i64x4, _mm256_set1_epi64x, _mm256_add_epi64, _mm256_sub_epi64 }
impl_int_mul! { i8x32, i16x16, i32x8, i64x4 }

// 64-bit integer min/max ops (_mm256_{min,max}_epi64) require AVX512, so for i64x4 we just fall
// back to the default impls of min and max in terms of le and select.
impl LanesOrd for i64x4 {
    #[inline(always)]
    fn lt(&self, other: &Self) -> Self::Output {
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn inner(lhs: &i64x4, rhs: &i64x4) -> m64x4 {
            m64x4(_mm256_cmpgt_epi64(rhs.0, lhs.0))
        }

        unsafe { inner(self, other) }
    }
}

int_type! { m8x32, m8, 32, m8x32, _mm256_set1_epi8, _mm256_cmpeq_epi8 }
int_type! { m16x16, m16, 16, m16x16, _mm256_set1_epi16, _mm256_cmpeq_epi16 }
int_type! { m32x8, m32, 8, m32x8, _mm256_set1_epi32, _mm256_cmpeq_epi32 }
int_type! { m64x4, m64, 4, m64x4, _mm256_set1_epi64x, _mm256_cmpeq_epi64 }
impl_ord_mask! { m8x32 }
impl_ord_mask! { m16x16 }
impl_ord_mask! { m32x8 }
impl_ord_mask! { m64x4 }
