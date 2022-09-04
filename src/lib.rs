use core::fmt::Debug;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use core::ops::{Index, IndexMut};

mod scalar;

#[allow(non_camel_case_types)]
pub trait Arch {
    type f32: Simd<Elem = f32, Arch = Self> + Num;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        fn f<A: Arch>() {
            let mut x = A::f32::new(0.0);
            x[0] = 2.0;

            let y = A::f32::new(1.0);

            assert_eq!((x + y)[0], 3.0);
        }

        f::<scalar::Scalar>();
    }
}
