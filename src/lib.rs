use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::ops::{Index, IndexMut};

pub mod mask;
use mask::*;

mod arch;

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

#[cfg(test)]
mod tests {
    use super::*;

    use arch::{avx2::Avx2, scalar::Scalar};

    fn test_ops<S>(
        type_: &str,
        values: &[S::Elem],
        eq: fn(&S::Elem, &S::Elem) -> bool,
        unary_ops: &[(fn(S) -> S, fn(S::Elem) -> S::Elem, &str)],
        binary_ops: &[(fn(S, S) -> S, fn(S::Elem, S::Elem) -> S::Elem, &str)],
        cmp_ops: &[(
            fn(&S, &S) -> S::Mask,
            fn(&S::Elem, &S::Elem) -> <S::Mask as Simd>::Elem,
            &str,
        )],
    ) where
        S: Simd,
        S::Elem: Copy + Debug,
        S::Mask: Simd,
        <S::Mask as Simd>::Elem: Debug + PartialEq,
    {
        for x in values.chunks(S::LANES) {
            for (vector, scalar, op) in unary_ops {
                let res = vector(S::from_slice(x));
                for (x, out) in x.iter().zip(res.as_slice().iter()) {
                    let scalar = scalar(*x);
                    assert!(
                        eq(&scalar, out),
                        "expected {}::{}({:?}) == {:?}, got {:?}",
                        type_,
                        op,
                        *x,
                        scalar,
                        *out
                    );
                }
            }
        }

        for x in values {
            for y in values.chunks(S::LANES) {
                for (vector, scalar, op) in binary_ops {
                    let res = vector(S::new(*x), S::from_slice(y));
                    for (y, out) in y.iter().zip(res.as_slice().iter()) {
                        let scalar = scalar(*x, *y);
                        assert!(
                            eq(&scalar, out),
                            "expected {}::{}({:?}, {:?}) == {:?}, got {:?}",
                            type_,
                            op,
                            *x,
                            *y,
                            scalar,
                            *out
                        );
                    }
                }

                for (vector, scalar, op) in cmp_ops {
                    let res = vector(&S::new(*x), &S::from_slice(y));
                    for (y, out) in y.iter().zip(res.as_slice().iter()) {
                        let scalar = scalar(x, y);
                        assert!(
                            &scalar == out,
                            "expected {}::{}({:?}, {:?}) == {:?}, got {:?}",
                            type_,
                            op,
                            *x,
                            *y,
                            scalar,
                            *out
                        );
                    }
                }
            }
        }
    }

    macro_rules! test_float {
        ($type:ident) => {{
            let values = [
                -1.0,
                -0.0,
                0.0,
                1.0,
                -$type::EPSILON,
                $type::EPSILON,
                $type::MIN,
                $type::MAX,
                $type::NEG_INFINITY,
                $type::INFINITY,
                $type::NAN,
            ]
            .into_iter()
            .cycle()
            .take(64)
            .collect::<Vec<$type>>();

            fn max(a: $type, b: $type) -> $type {
                if a > b {
                    a
                } else {
                    b
                }
            }

            fn min(a: $type, b: $type) -> $type {
                if a < b {
                    a
                } else {
                    b
                }
            }

            test_ops::<A::$type>(
                stringify!($type),
                &values,
                |x, y| x.to_bits() == y.to_bits(),
                &[(A::$type::neg, $type::neg, "neg")],
                &[
                    (A::$type::add, $type::add, "add"),
                    (A::$type::sub, $type::sub, "sub"),
                    (A::$type::mul, $type::mul, "mul"),
                    (A::$type::div, $type::div, "div"),
                    (A::$type::max, max, "max"),
                    (A::$type::min, min, "min"),
                ],
                &[
                    (A::$type::eq, |x, y| (x == y).into(), "eq"),
                    (A::$type::ne, |x, y| (x != y).into(), "ne"),
                    (A::$type::lt, |x, y| (x < y).into(), "lt"),
                    (A::$type::le, |x, y| (x <= y).into(), "le"),
                    (A::$type::gt, |x, y| (x > y).into(), "gt"),
                    (A::$type::ge, |x, y| (x >= y).into(), "ge"),
                ],
            );
        }};
    }

    macro_rules! test_int {
        ($type:ident) => {{
            let values = ($type::MIN..=$type::MAX)
                .step_by(1 << ($type::BITS as usize - 6))
                .take(64)
                .collect::<Vec<$type>>();

            test_ops::<A::$type>(
                stringify!($type),
                &values,
                $type::eq,
                &[
                    (A::$type::neg, $type::wrapping_neg, "neg"),
                    (A::$type::not, $type::not, "not"),
                ],
                &[
                    (A::$type::add, $type::wrapping_add, "add"),
                    (A::$type::sub, $type::wrapping_sub, "sub"),
                    (A::$type::mul, $type::wrapping_mul, "mul"),
                    (A::$type::bitand, $type::bitand, "bitand"),
                    (A::$type::bitor, $type::bitor, "bitor"),
                    (A::$type::bitxor, $type::bitxor, "bitxor"),
                    (A::$type::max, $type::max, "max"),
                    (A::$type::min, $type::min, "min"),
                ],
                &[
                    (A::$type::eq, |x, y| (x == y).into(), "eq"),
                    (A::$type::ne, |x, y| (x != y).into(), "ne"),
                    (A::$type::lt, |x, y| (x < y).into(), "lt"),
                    (A::$type::le, |x, y| (x <= y).into(), "le"),
                    (A::$type::gt, |x, y| (x > y).into(), "gt"),
                    (A::$type::ge, |x, y| (x >= y).into(), "ge"),
                ],
            );
        }};
    }

    macro_rules! test_mask {
        ($type:ident) => {{
            let values = [false.into(), true.into()]
                .into_iter()
                .cycle()
                .take(64)
                .collect::<Vec<$type>>();

            test_ops::<A::$type>(
                stringify!($type),
                &values,
                $type::eq,
                &[(A::$type::not, $type::not, "not")],
                &[
                    (A::$type::bitand, $type::bitand, "bitand"),
                    (A::$type::bitor, $type::bitor, "bitor"),
                    (A::$type::bitxor, $type::bitxor, "bitxor"),
                    (A::$type::max, $type::max, "max"),
                    (A::$type::min, $type::min, "min"),
                ],
                &[
                    (A::$type::eq, |x, y| (x == y).into(), "eq"),
                    (A::$type::ne, |x, y| (x != y).into(), "ne"),
                    (A::$type::lt, |x, y| (x < y).into(), "lt"),
                    (A::$type::le, |x, y| (x <= y).into(), "le"),
                    (A::$type::gt, |x, y| (x > y).into(), "gt"),
                    (A::$type::ge, |x, y| (x >= y).into(), "ge"),
                ],
            );
        }};
    }

    fn test_arch<A: Arch>() {
        test_float!(f32);
        test_float!(f64);

        test_int!(u8);
        test_int!(u16);
        test_int!(u32);
        test_int!(u64);

        test_int!(i8);
        test_int!(i16);
        test_int!(i32);
        test_int!(i64);

        test_mask!(m8);
        test_mask!(m16);
        test_mask!(m32);
        test_mask!(m64);
    }

    #[test]
    fn scalar() {
        test_arch::<Scalar>();
    }

    #[test]
    fn avx2() {
        test_arch::<Avx2>();
    }
}
