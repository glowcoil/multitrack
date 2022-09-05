#![allow(non_camel_case_types)]

use crate::mask::m32;
use crate::{Arch, Mask, Num, Select, Simd, LanesEq, LanesOrd};

use core::fmt::{self, Debug};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};
use core::slice;

pub struct Scalar;

impl Arch for Scalar {
    type f32 = f32x1;
    type m32 = m32x1;
}

#[derive(Copy, Clone, Default)]
pub struct f32x1(f32);

impl Simd for f32x1 {
    type Arch = Scalar;
    type Elem = f32;

    const LANES: usize = 1;

    fn new(elem: Self::Elem) -> Self {
        f32x1(elem)
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
}

impl LanesEq for f32x1 {
    type Output = m32x1;

    fn eq(&self, other: &f32x1) -> m32x1 {
        m32x1((self.0 == other.0).into())
    }
}

impl LanesOrd for f32x1 {
    fn lt(&self, other: &f32x1) -> m32x1 {
        m32x1((self.0 < other.0).into())
    }
}

impl Index<usize> for f32x1 {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        assert!(index == 0);
        &self.0
    }
}

impl IndexMut<usize> for f32x1 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        assert!(index == 0);
        &mut self.0
    }
}

impl Debug for f32x1 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple("Scalar::f32").field(&self.0).finish()
    }
}

impl Num for f32x1 {}

impl Add for f32x1 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        f32x1(self.0 + rhs.0)
    }
}

impl AddAssign for f32x1 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for f32x1 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        f32x1(self.0 - rhs.0)
    }
}

impl SubAssign for f32x1 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul for f32x1 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        f32x1(self.0 * rhs.0)
    }
}

impl MulAssign for f32x1 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Div for f32x1 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        f32x1(self.0 / rhs.0)
    }
}

impl DivAssign for f32x1 {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl Rem for f32x1 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        f32x1(self.0 % rhs.0)
    }
}

impl RemAssign for f32x1 {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl Neg for f32x1 {
    type Output = Self;

    fn neg(self) -> Self {
        f32x1(-self.0)
    }
}

#[derive(Copy, Clone, Default)]
pub struct m32x1(m32);

impl Simd for m32x1 {
    type Arch = Scalar;
    type Elem = m32;

    const LANES: usize = 1;

    fn new(elem: Self::Elem) -> Self {
        m32x1(elem)
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
}

impl LanesEq for m32x1 {
    type Output = m32x1;

    fn eq(&self, other: &m32x1) -> m32x1 {
        m32x1((self.0 == other.0).into())
    }
}

impl Index<usize> for m32x1 {
    type Output = m32;

    fn index(&self, index: usize) -> &m32 {
        assert!(index == 0);
        &self.0
    }
}

impl IndexMut<usize> for m32x1 {
    fn index_mut(&mut self, index: usize) -> &mut m32 {
        assert!(index == 0);
        &mut self.0
    }
}

impl Debug for m32x1 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple("Scalar::m32").field(&self.0).finish()
    }
}

impl Mask for m32x1 {}

impl BitAnd for m32x1 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        m32x1(self.0 & rhs.0)
    }
}

impl BitAndAssign for m32x1 {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for m32x1 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        m32x1(self.0 | rhs.0)
    }
}

impl BitOrAssign for m32x1 {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for m32x1 {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        m32x1(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for m32x1 {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Not for m32x1 {
    type Output = Self;

    fn not(self) -> Self::Output {
        m32x1(!self.0)
    }
}

impl Select<f32x1> for m32x1 {
    fn select(self, if_true: f32x1, if_false: f32x1) -> f32x1 {
        if self.0.into() {
            if_true
        } else {
            if_false
        }
    }
}

impl Select<m32x1> for m32x1 {
    fn select(self, if_true: m32x1, if_false: m32x1) -> m32x1 {
        if self.0.into() {
            if_true
        } else {
            if_false
        }
    }
}
