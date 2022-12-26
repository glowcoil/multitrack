use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};
use core::ops::{Shl, ShlAssign, Shr, ShrAssign};

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
    Self: Shl<usize, Output = Self> + ShlAssign<usize>,
    Self: Shr<usize, Output = Self> + ShrAssign<usize>,
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

    #[inline(always)]
    fn ne(&self, other: &Self) -> Self::Output {
        !self.eq(other)
    }
}

pub trait LanesOrd<Rhs = Self>: LanesEq<Rhs> {
    fn lt(&self, other: &Self) -> Self::Output;

    #[inline(always)]
    fn le(&self, other: &Self) -> Self::Output {
        self.lt(other) | self.eq(other)
    }

    #[inline(always)]
    fn gt(&self, other: &Self) -> Self::Output {
        other.lt(self)
    }

    #[inline(always)]
    fn ge(&self, other: &Self) -> Self::Output {
        other.le(self)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        other.lt(&self).select(self, other)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.lt(&other).select(self, other)
    }

    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

pub trait Select<V> {
    fn select(self, if_true: V, if_false: V) -> V;
}

pub trait Convert<T> {
    fn convert(self) -> T;
}

pub trait Widen<T> {
    fn widen<F>(self, consume: F)
    where
        F: FnMut(T);
}

pub trait Narrow<T> {
    fn narrow<F>(produce: F) -> T
    where
        F: FnMut() -> Self;
}
