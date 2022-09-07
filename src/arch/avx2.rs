#![allow(non_camel_case_types)]

use crate::{LanesEq, Mask, Select, Simd};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use core::fmt::{self, Debug};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};
use std::mem;
use std::slice;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct f32x8(__m256);

impl Simd for f32x8 {
    type Elem = f32;

    const LANES: usize = 8;

    fn new(elem: Self::Elem) -> Self {
        unsafe { f32x8(_mm256_set1_ps(elem)) }
    }

    fn as_slice(&self) -> &[Self::Elem] {
        unsafe { slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES) }
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        unsafe { slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES) }
    }

    fn from_slice(slice: &[Self::Elem]) -> Self {
        assert!(slice.len() == Self::LANES);
        unsafe { f32x8(_mm256_loadu_ps(slice.as_ptr())) }
    }

    fn write_to_slice(&self, slice: &mut [Self::Elem]) {
        assert!(slice.len() == Self::LANES);
        unsafe {
            _mm256_storeu_ps(slice.as_mut_ptr(), self.0);
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

impl Debug for f32x8 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), fmt)
    }
}

impl Default for f32x8 {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

impl LanesEq for f32x8 {
    type Output = m32x8;

    fn eq(&self, other: &Self) -> Self::Output {
        unsafe {
            let res = _mm256_cmp_ps(self.0, other.0, _CMP_EQ_OQ);
            m32x8(_mm256_castps_si256(res))
        }
    }

    fn ne(&self, other: &Self) -> Self::Output {
        unsafe {
            let res = _mm256_cmp_ps(self.0, other.0, _CMP_NEQ_OQ);
            m32x8(_mm256_castps_si256(res))
        }
    }
}

impl Index<usize> for f32x8 {
    type Output = <Self as Simd>::Elem;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl IndexMut<usize> for f32x8 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct m32x8(__m256i);

impl Mask for m32x8 {}

impl BitAnd for m32x8 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { m32x8(_mm256_and_si256(self.0, rhs.0)) }
    }
}

impl BitAndAssign for m32x8 {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl BitOr for m32x8 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { m32x8(_mm256_and_si256(self.0, rhs.0)) }
    }
}

impl BitOrAssign for m32x8 {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl BitXor for m32x8 {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { m32x8(_mm256_xor_si256(self.0, rhs.0)) }
    }
}

impl BitXorAssign for m32x8 {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl Not for m32x8 {
    type Output = Self;

    fn not(self) -> Self::Output {
        unsafe { m32x8(_mm256_andnot_si256(self.0, _mm256_set1_epi8(!0))) }
    }
}

impl Select<f32x8> for m32x8 {
    fn select(self, if_true: f32x8, if_false: f32x8) -> f32x8 {
        unsafe {
            let mask = _mm256_castsi256_ps(self.0);
            f32x8(_mm256_blendv_ps(if_false.0, if_true.0, mask))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let x = f32x8::new(3.0);
        let y = x.eq(&f32x8::new(3.0)).select(f32x8::new(0.0), f32x8::new(1.0));
        assert_eq!(y[0], 0.0);
    }
}
