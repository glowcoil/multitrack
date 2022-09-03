use std::fmt::Debug;
use std::ops::{Index, IndexMut};

mod scalar;

#[allow(non_camel_case_types)]
pub trait Arch {
    type f32: Simd<Elem = f32, Arch = Self>;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        fn f<A: Arch>() {
            let mut x = A::f32::new(0.0);
            x[0] = 3.0;
        }

        f::<scalar::Scalar>();
    }
}
