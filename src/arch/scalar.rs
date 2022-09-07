#![allow(non_camel_case_types)]

use crate::mask::*;
use crate::{Arch, LanesEq, LanesOrd, Mask, Num, Select, Simd};

use core::fmt::{self, Debug};
use core::num::Wrapping;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};
use core::slice;

macro_rules! scalar_type {
    ($scalar:ident, $inner:ident, $mask:ident) => {
        #[derive(Copy, Clone, Default)]
        #[repr(transparent)]
        pub struct $scalar($inner);

        impl Simd for $scalar {
            type Elem = $inner;

            const LANES: usize = 1;

            fn new(elem: Self::Elem) -> Self {
                $scalar(elem)
            }

            fn as_slice(&self) -> &[Self::Elem] {
                slice::from_ref(&self.0)
            }

            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                slice::from_mut(&mut self.0)
            }

            fn from_slice(slice: &[Self::Elem]) -> Self {
                Self::new(slice[0])
            }

            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                slice[0] = self.0;
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

        impl LanesEq for $scalar {
            type Output = $mask;

            fn eq(&self, other: &$scalar) -> $mask {
                $mask((self.0 == other.0).into())
            }
        }

        impl LanesOrd for $scalar {
            fn lt(&self, other: &$scalar) -> $mask {
                $mask((self.0 < other.0).into())
            }
        }

        impl Index<usize> for $scalar {
            type Output = $inner;

            fn index(&self, index: usize) -> &$inner {
                assert!(index == 0);
                &self.0
            }
        }

        impl IndexMut<usize> for $scalar {
            fn index_mut(&mut self, index: usize) -> &mut $inner {
                assert!(index == 0);
                &mut self.0
            }
        }

        impl Debug for $scalar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list().entry(&self.0).finish()
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

            const LANES: usize = 1;

            fn new(elem: Self::Elem) -> Self {
                $scalar(Wrapping(elem))
            }

            fn as_slice(&self) -> &[Self::Elem] {
                slice::from_ref(&self.0 .0)
            }

            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                slice::from_mut(&mut self.0 .0)
            }

            fn from_slice(slice: &[Self::Elem]) -> Self {
                Self::new(slice[0])
            }

            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                slice[0] = self.0 .0;
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

        impl LanesEq for $scalar {
            type Output = $mask;

            fn eq(&self, other: &$scalar) -> $mask {
                $mask((self.0 == other.0).into())
            }
        }

        impl LanesOrd for $scalar {
            fn lt(&self, other: &$scalar) -> $mask {
                $mask((self.0 < other.0).into())
            }
        }

        impl Index<usize> for $scalar {
            type Output = $inner;

            fn index(&self, index: usize) -> &$inner {
                assert!(index == 0);
                &self.0 .0
            }
        }

        impl IndexMut<usize> for $scalar {
            fn index_mut(&mut self, index: usize) -> &mut $inner {
                assert!(index == 0);
                &mut self.0 .0
            }
        }

        impl Debug for $scalar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list().entry(&self.0 .0).finish()
            }
        }
    };
}

macro_rules! impl_num {
    ($num:ident) => {
        impl Num for $num {}

        impl Add for $num {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                $num(self.0 + rhs.0)
            }
        }

        impl AddAssign for $num {
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl Sub for $num {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                $num(self.0 - rhs.0)
            }
        }

        impl SubAssign for $num {
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl Mul for $num {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                $num(self.0 * rhs.0)
            }
        }

        impl MulAssign for $num {
            fn mul_assign(&mut self, rhs: Self) {
                self.0 *= rhs.0;
            }
        }

        impl Div for $num {
            type Output = Self;

            fn div(self, rhs: Self) -> Self {
                $num(self.0 / rhs.0)
            }
        }

        impl DivAssign for $num {
            fn div_assign(&mut self, rhs: Self) {
                self.0 /= rhs.0;
            }
        }

        impl Rem for $num {
            type Output = Self;

            fn rem(self, rhs: Self) -> Self {
                $num(self.0 % rhs.0)
            }
        }

        impl RemAssign for $num {
            fn rem_assign(&mut self, rhs: Self) {
                self.0 %= rhs.0;
            }
        }

        impl Neg for $num {
            type Output = Self;

            fn neg(self) -> Self {
                $num(-self.0)
            }
        }
    };
}

macro_rules! impl_mask {
    ($mask:ident, { $($select:ident),* }) => {
        impl Mask for $mask {}

        impl BitAnd for $mask {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self::Output {
                $mask(self.0 & rhs.0)
            }
        }

        impl BitAndAssign for $mask {
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl BitOr for $mask {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self::Output {
                $mask(self.0 | rhs.0)
            }
        }

        impl BitOrAssign for $mask {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl BitXor for $mask {
            type Output = Self;

            fn bitxor(self, rhs: Self) -> Self::Output {
                $mask(self.0 ^ rhs.0)
            }
        }

        impl BitXorAssign for $mask {
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }

        impl Not for $mask {
            type Output = Self;

            fn not(self) -> Self::Output {
                $mask(!self.0)
            }
        }

        $(
            impl Select<$select> for $mask {
                fn select(self, if_true: $select, if_false: $select) -> $select {
                    if self.0.into() {
                        if_true
                    } else {
                        if_false
                    }
                }
            }
        )*
    };
}

pub struct Scalar;

impl Arch for Scalar {
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
}

scalar_type! { f32x1, f32, m32x1 }
scalar_type! { f64x1, f64, m64x1 }
impl_num! { f64x1 }
impl_num! { f32x1 }

wrapping_scalar_type! { u8x1, u8, m8x1 }
wrapping_scalar_type! { u16x1, u16, m16x1 }
wrapping_scalar_type! { u32x1, u32, m32x1 }
wrapping_scalar_type! { u64x1, u64, m64x1 }
impl_num! { u8x1 }
impl_num! { u16x1 }
impl_num! { u32x1 }
impl_num! { u64x1 }

wrapping_scalar_type! { i8x1, i8, m8x1 }
wrapping_scalar_type! { i16x1, i16, m16x1 }
wrapping_scalar_type! { i32x1, i32, m32x1 }
wrapping_scalar_type! { i64x1, i64, m64x1 }
impl_num! { i8x1 }
impl_num! { i16x1 }
impl_num! { i32x1 }
impl_num! { i64x1 }

scalar_type! { m8x1, m8, m8x1 }
scalar_type! { m16x1, m16, m16x1 }
scalar_type! { m32x1, m32, m32x1 }
scalar_type! { m64x1, m64, m64x1 }
impl_mask! { m8x1, { u8x1, i8x1, m8x1 } }
impl_mask! { m16x1, { u16x1, i16x1, m16x1 } }
impl_mask! { m32x1, { u32x1, i32x1, m32x1, f32x1 } }
impl_mask! { m64x1, { u64x1, i64x1, m64x1, f64x1 } }
