use core::fmt::Debug;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};

mod mask;
use mask::m32;

mod scalar;

#[allow(non_camel_case_types)]
pub trait Arch
where
    Self::m32: Select<Self::f32> + Select<Self::m32>,
{
    type f32: Simd<Arch = Self, Elem = f32> + Num;
    type m32: Simd<Arch = Self, Elem = m32> + Mask;
}

pub trait Simd
where
    Self: Copy + Clone + Debug + Default + Send + Sync + Sized,
    Self: Index<usize, Output = Self::Elem> + IndexMut<usize, Output = Self::Elem>,
{
    type Arch: Arch;
    type Elem;

    const LANES: usize;

    fn new(elem: Self::Elem) -> Self;
}

pub trait Num
where
    Self: Sized,
    Self: Add<Output = Self> + AddAssign,
    Self: Sub<Output = Self> + SubAssign,
    Self: Mul<Output = Self> + MulAssign,
    Self: Div<Output = Self> + DivAssign,
    Self: Rem<Output = Self> + RemAssign,
    Self: Neg,
{
}

pub trait Mask
where
    Self: Sized,
    Self: BitAnd + BitAndAssign,
    Self: BitOr + BitOrAssign,
    Self: BitXor + BitXorAssign,
    Self: Not,
{
}

pub trait Select<V> {
    fn select(self, if_true: V, if_false: V) -> V;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        fn f<A: Arch>() {
            let mut x = A::f32::new(0.0);
            x[0] = 2.0;

            let y = A::f32::new(1.0);

            let t = A::m32::new(false.into());
            let z = t.select(A::f32::new(0.0), x + y);

            assert_eq!(z[0], 3.0);
        }

        f::<scalar::Scalar>();
    }
}
