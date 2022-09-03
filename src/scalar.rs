#![allow(non_camel_case_types)]

use crate::{Arch, Simd};

use std::fmt::{self, Debug};
use std::ops::{Index, IndexMut};

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
