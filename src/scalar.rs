#![allow(non_camel_case_types)]

use crate::{Arch, Num, Simd};

use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use core::fmt::{self, Debug};
use core::ops::{Index, IndexMut};

pub struct Scalar;

impl Arch for Scalar {
    type f32 = f32x1;
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
