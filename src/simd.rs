use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};

use crate::mask::*;

#[allow(non_camel_case_types)]
pub trait Arch
where
    Self::m8: Select<Self::i8> + Select<Self::u8> + Select<Self::m8>,
    Self::m16: Select<Self::i16> + Select<Self::u16> + Select<Self::m16>,
    Self::m32: Select<Self::f32> + Select<Self::i32> + Select<Self::u32> + Select<Self::m32>,
    Self::m64: Select<Self::f64> + Select<Self::i64> + Select<Self::u64> + Select<Self::m64>,
{
    type f32: Simd<Elem = f32, Mask = Self::m32> + Float;
    type f64: Simd<Elem = f64, Mask = Self::m64> + Float;

    type u8: Simd<Elem = u8, Mask = Self::m8> + Int + Bitwise;
    type u16: Simd<Elem = u16, Mask = Self::m16> + Int + Bitwise;
    type u32: Simd<Elem = u32, Mask = Self::m32> + Int + Bitwise;
    type u64: Simd<Elem = u64, Mask = Self::m64> + Int + Bitwise;

    type i8: Simd<Elem = i8, Mask = Self::m8> + Int + Bitwise;
    type i16: Simd<Elem = i16, Mask = Self::m16> + Int + Bitwise;
    type i32: Simd<Elem = i32, Mask = Self::m32> + Int + Bitwise;
    type i64: Simd<Elem = i64, Mask = Self::m64> + Int + Bitwise;

    type m8: Simd<Elem = m8, Mask = Self::m8> + Bitwise;
    type m16: Simd<Elem = m16, Mask = Self::m16> + Bitwise;
    type m32: Simd<Elem = m32, Mask = Self::m32> + Bitwise;
    type m64: Simd<Elem = m64, Mask = Self::m64> + Bitwise;
}

pub trait Simd: Copy + Clone + Debug + Default + Send + Sync + Sized
where
    Self: LanesEq<Output = Self::Mask> + LanesOrd<Output = Self::Mask>,
    Self: Index<usize, Output = Self::Elem> + IndexMut<usize, Output = Self::Elem>,
{
    type Elem;
    type Mask: Select<Self>;

    const LANES: usize;

    fn new(elem: Self::Elem) -> Self;

    fn as_slice(&self) -> &[Self::Elem];
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
    fn from_slice(slice: &[Self::Elem]) -> Self;
    fn write_to_slice(&self, slice: &mut [Self::Elem]);
    fn align_slice(slice: &[Self::Elem]) -> (&[Self::Elem], &[Self], &[Self::Elem]);
    fn align_mut_slice(
        slice: &mut [Self::Elem],
    ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]);
}

pub trait Float: Sized
where
    Self: Add<Output = Self> + AddAssign,
    Self: Sub<Output = Self> + SubAssign,
    Self: Mul<Output = Self> + MulAssign,
    Self: Div<Output = Self> + DivAssign,
    Self: Neg<Output = Self>,
{
}

pub trait Int: Sized
where
    Self: Add<Output = Self> + AddAssign,
    Self: Sub<Output = Self> + SubAssign,
    Self: Mul<Output = Self> + MulAssign,
    Self: Neg<Output = Self>,
{
}

pub trait Bitwise: Sized
where
    Self: BitAnd<Output = Self> + BitAndAssign,
    Self: BitOr<Output = Self> + BitOrAssign,
    Self: BitXor<Output = Self> + BitXorAssign,
    Self: Not<Output = Self>,
{
}

pub trait LanesEq<Rhs = Self>: Sized {
    type Output: Bitwise + Select<Self>;

    fn eq(&self, other: &Self) -> Self::Output;

    fn ne(&self, other: &Self) -> Self::Output {
        !self.eq(other)
    }
}

pub trait LanesOrd<Rhs = Self>: LanesEq<Rhs> {
    fn lt(&self, other: &Self) -> Self::Output;

    fn le(&self, other: &Self) -> Self::Output {
        self.lt(other) | self.eq(other)
    }

    fn gt(&self, other: &Self) -> Self::Output {
        other.lt(self)
    }

    fn ge(&self, other: &Self) -> Self::Output {
        other.le(self)
    }

    fn max(self, other: Self) -> Self {
        other.lt(&self).select(self, other)
    }

    fn min(self, other: Self) -> Self {
        self.lt(&other).select(self, other)
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

pub trait Select<V> {
    fn select(self, if_true: V, if_false: V) -> V;
}
