#![allow(non_camel_case_types)]

use core::fmt::{self, Debug};
use core::num::Wrapping;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};
use core::ops::{Shl, ShlAssign, Shr, ShrAssign};
use core::slice;

use crate::mask::*;
use crate::simd::{Bitwise, Float, Int, LanesEq, LanesOrd, Select, Simd};
use crate::{Arch, Task};

pub struct ScalarImpl;

impl Arch for ScalarImpl {
    type f32 = f32x1;
    type f64 = f64x1;

    type u8 = u8x1;
    type u16 = u16x1;
    type u32 = u32x1;
    type u64 = u64x1;

    type i8 = i8x1;
    type i16 = i16x1;
    type i32 = i32x1;
    type i64 = i64x1;

    type m8 = m8x1;
    type m16 = m16x1;
    type m32 = m32x1;
    type m64 = m64x1;

    const NAME: &'static str = "scalar";

    #[inline]
    fn invoke<T: Task>(task: T) -> T::Result {
        task.run::<ScalarImpl>()
    }
}

macro_rules! scalar_type {
    ($scalar:ident, $inner:ident, $mask:ident) => {
        #[derive(Copy, Clone, Default)]
        #[repr(transparent)]
        pub struct $scalar($inner);

        impl Simd for $scalar {
            type Elem = $inner;
            type Mask = $mask;

            const LANES: usize = 1;

            #[inline]
            fn new(elem: Self::Elem) -> Self {
                $scalar(elem)
            }

            #[inline]
            fn as_slice(&self) -> &[Self::Elem] {
                slice::from_ref(&self.0)
            }

            #[inline]
            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                slice::from_mut(&mut self.0)
            }

            #[inline]
            fn from_slice(slice: &[Self::Elem]) -> Self {
                Self::new(slice[0])
            }

            #[inline]
            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                slice[0] = self.0;
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

        impl LanesEq for $scalar {
            type Output = $mask;

            #[inline]
            fn eq(&self, other: &$scalar) -> $mask {
                $mask((self.0 == other.0).into())
            }
        }

        impl LanesOrd for $scalar {
            #[inline]
            fn lt(&self, other: &$scalar) -> $mask {
                $mask((self.0 < other.0).into())
            }
        }

        impl Index<usize> for $scalar {
            type Output = $inner;

            #[inline]
            fn index(&self, index: usize) -> &$inner {
                assert!(index == 0);
                &self.0
            }
        }

        impl IndexMut<usize> for $scalar {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut $inner {
                assert!(index == 0);
                &mut self.0
            }
        }

        impl Debug for $scalar {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list().entry(&self.0).finish()
            }
        }

        impl Select<$scalar> for $mask {
            #[inline]
            fn select(self, if_true: $scalar, if_false: $scalar) -> $scalar {
                if self.0.into() {
                    if_true
                } else {
                    if_false
                }
            }
        }
    };
}

macro_rules! wrapping_scalar_type {
    ($scalar:ident, $inner:ident, $mask:ident) => {
        #[derive(Copy, Clone, Default)]
        #[repr(transparent)]
        pub struct $scalar(Wrapping<$inner>);

        impl Simd for $scalar {
            type Elem = $inner;
            type Mask = $mask;

            const LANES: usize = 1;

            #[inline]
            fn new(elem: Self::Elem) -> Self {
                $scalar(Wrapping(elem))
            }

            #[inline]
            fn as_slice(&self) -> &[Self::Elem] {
                slice::from_ref(&self.0 .0)
            }

            #[inline]
            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                slice::from_mut(&mut self.0 .0)
            }

            #[inline]
            fn from_slice(slice: &[Self::Elem]) -> Self {
                Self::new(slice[0])
            }

            #[inline]
            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                slice[0] = self.0 .0;
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

        impl LanesEq for $scalar {
            type Output = $mask;

            #[inline]
            fn eq(&self, other: &$scalar) -> $mask {
                $mask((self.0 == other.0).into())
            }
        }

        impl LanesOrd for $scalar {
            #[inline]
            fn lt(&self, other: &$scalar) -> $mask {
                $mask((self.0 < other.0).into())
            }
        }

        impl Index<usize> for $scalar {
            type Output = $inner;

            #[inline]
            fn index(&self, index: usize) -> &$inner {
                assert!(index == 0);
                &self.0 .0
            }
        }

        impl IndexMut<usize> for $scalar {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut $inner {
                assert!(index == 0);
                &mut self.0 .0
            }
        }

        impl Debug for $scalar {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list().entry(&self.0 .0).finish()
            }
        }

        impl Select<$scalar> for $mask {
            #[inline]
            fn select(self, if_true: $scalar, if_false: $scalar) -> $scalar {
                if self.0.into() {
                    if_true
                } else {
                    if_false
                }
            }
        }
    };
}

macro_rules! impl_float {
    ($float:ident) => {
        impl Float for $float {}

        impl Add for $float {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self {
                $float(self.0 + rhs.0)
            }
        }

        impl AddAssign for $float {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl Sub for $float {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                $float(self.0 - rhs.0)
            }
        }

        impl SubAssign for $float {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl Mul for $float {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self {
                $float(self.0 * rhs.0)
            }
        }

        impl MulAssign for $float {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                self.0 *= rhs.0;
            }
        }

        impl Div for $float {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self {
                $float(self.0 / rhs.0)
            }
        }

        impl DivAssign for $float {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                self.0 /= rhs.0;
            }
        }

        impl Neg for $float {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self {
                $float(-self.0)
            }
        }
    };
}

macro_rules! impl_int {
    ($int:ident) => {
        impl Int for $int {}

        impl Add for $int {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self {
                $int(self.0 + rhs.0)
            }
        }

        impl AddAssign for $int {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl Sub for $int {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                $int(self.0 - rhs.0)
            }
        }

        impl SubAssign for $int {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl Mul for $int {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self {
                $int(self.0 * rhs.0)
            }
        }

        impl MulAssign for $int {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                self.0 *= rhs.0;
            }
        }

        impl Div for $int {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self {
                $int(self.0 / rhs.0)
            }
        }

        impl DivAssign for $int {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                self.0 /= rhs.0;
            }
        }

        impl Neg for $int {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self {
                $int(-self.0)
            }
        }

        impl Shl<usize> for $int {
            type Output = Self;

            #[inline]
            fn shl(self, rhs: usize) -> Self {
                $int(self.0 << rhs)
            }
        }

        impl ShlAssign<usize> for $int {
            #[inline]
            fn shl_assign(&mut self, rhs: usize) {
                self.0 <<= rhs;
            }
        }

        impl Shr<usize> for $int {
            type Output = Self;

            #[inline]
            fn shr(self, rhs: usize) -> Self {
                $int(self.0 >> rhs)
            }
        }

        impl ShrAssign<usize> for $int {
            #[inline]
            fn shr_assign(&mut self, rhs: usize) {
                self.0 >>= rhs;
            }
        }
    };
}

macro_rules! impl_bitwise {
    ($bitwise:ident) => {
        impl Bitwise for $bitwise {}

        impl BitAnd for $bitwise {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self::Output {
                $bitwise(self.0 & rhs.0)
            }
        }

        impl BitAndAssign for $bitwise {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl BitOr for $bitwise {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                $bitwise(self.0 | rhs.0)
            }
        }

        impl BitOrAssign for $bitwise {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl BitXor for $bitwise {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                $bitwise(self.0 ^ rhs.0)
            }
        }

        impl BitXorAssign for $bitwise {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }

        impl Not for $bitwise {
            type Output = Self;

            #[inline]
            fn not(self) -> Self::Output {
                $bitwise(!self.0)
            }
        }
    };
}

scalar_type! { f32x1, f32, m32x1 }
scalar_type! { f64x1, f64, m64x1 }
impl_float! { f64x1 }
impl_float! { f32x1 }

wrapping_scalar_type! { u8x1, u8, m8x1 }
wrapping_scalar_type! { u16x1, u16, m16x1 }
wrapping_scalar_type! { u32x1, u32, m32x1 }
wrapping_scalar_type! { u64x1, u64, m64x1 }
impl_int! { u8x1 }
impl_int! { u16x1 }
impl_int! { u32x1 }
impl_int! { u64x1 }
impl_bitwise! { u8x1 }
impl_bitwise! { u16x1 }
impl_bitwise! { u32x1 }
impl_bitwise! { u64x1 }

wrapping_scalar_type! { i8x1, i8, m8x1 }
wrapping_scalar_type! { i16x1, i16, m16x1 }
wrapping_scalar_type! { i32x1, i32, m32x1 }
wrapping_scalar_type! { i64x1, i64, m64x1 }
impl_int! { i8x1 }
impl_int! { i16x1 }
impl_int! { i32x1 }
impl_int! { i64x1 }
impl_bitwise! { i8x1 }
impl_bitwise! { i16x1 }
impl_bitwise! { i32x1 }
impl_bitwise! { i64x1 }

scalar_type! { m8x1, m8, m8x1 }
scalar_type! { m16x1, m16, m16x1 }
scalar_type! { m32x1, m32, m32x1 }
scalar_type! { m64x1, m64, m64x1 }
impl_bitwise! { m8x1 }
impl_bitwise! { m16x1 }
impl_bitwise! { m32x1 }
impl_bitwise! { m64x1 }
