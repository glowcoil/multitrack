extern crate self as multitrack;

pub mod arch;
pub mod simd;

pub use multitrack_attributes::{dispatch, specialize};

use simd::*;

#[allow(non_camel_case_types)]
pub trait Arch
where
    Self::m8: Select<Self::i8> + Select<Self::u8> + Select<Self::m8>,
    Self::m16: Select<Self::i16> + Select<Self::u16> + Select<Self::m16>,
    Self::m32: Select<Self::f32> + Select<Self::i32> + Select<Self::u32> + Select<Self::m32>,
    Self::m64: Select<Self::f64> + Select<Self::i64> + Select<Self::u64> + Select<Self::m64>,
{
    type f32: Simd<Elem = f32, Mask = Self::m32> + AsSlice + Convert32<Self> + Float;
    type f64: Simd<Elem = f64, Mask = Self::m64> + AsSlice + Convert64<Self> + Float;

    type u8: Simd<Elem = u8, Mask = Self::m8> + AsSlice + Convert8<Self> + Int;
    type u16: Simd<Elem = u16, Mask = Self::m16> + AsSlice + Convert16<Self> + Int;
    type u32: Simd<Elem = u32, Mask = Self::m32> + AsSlice + Convert32<Self> + Int;
    type u64: Simd<Elem = u64, Mask = Self::m64> + AsSlice + Convert64<Self> + Int;

    type i8: Simd<Elem = i8, Mask = Self::m8> + AsSlice + Convert8<Self> + Int;
    type i16: Simd<Elem = i16, Mask = Self::m16> + AsSlice + Convert16<Self> + Int;
    type i32: Simd<Elem = i32, Mask = Self::m32> + AsSlice + Convert32<Self> + Int;
    type i64: Simd<Elem = i64, Mask = Self::m64> + AsSlice + Convert64<Self> + Int;

    type m8: Simd<Elem = bool, Mask = Self::m8> + Convert<Self::u8> + Mask;
    type m16: Simd<Elem = bool, Mask = Self::m16> + Convert<Self::u16> + Mask;
    type m32: Simd<Elem = bool, Mask = Self::m32> + Convert<Self::u32> + Mask;
    type m64: Simd<Elem = bool, Mask = Self::m64> + Convert<Self::u64> + Mask;

    const NAME: &'static str;

    fn invoke<T: Task>(task: T) -> T::Result;
}

pub trait Convert8<A: Arch + ?Sized>: Sized
where
    Self: Widen<A::f32> + Widen<A::f64>,
    Self: Convert<A::u8> + Widen<A::u16> + Widen<A::u32> + Widen<A::u64>,
    Self: Convert<A::i8> + Widen<A::i16> + Widen<A::i32> + Widen<A::i64>,
{
}

pub trait Convert16<A: Arch + ?Sized>: Sized
where
    Self: Widen<A::f32> + Widen<A::f64>,
    Self: Narrow<A::u8> + Convert<A::u16> + Widen<A::u32> + Widen<A::u64>,
    Self: Narrow<A::i8> + Convert<A::i16> + Widen<A::i32> + Widen<A::i64>,
{
}

pub trait Convert32<A: Arch + ?Sized>: Sized
where
    Self: Convert<A::f32> + Widen<A::f64>,
    Self: Narrow<A::u8> + Narrow<A::u16> + Convert<A::u32> + Widen<A::u64>,
    Self: Narrow<A::i8> + Narrow<A::i16> + Convert<A::i32> + Widen<A::i64>,
{
}

pub trait Convert64<A: Arch + ?Sized>: Sized
where
    Self: Narrow<A::f32> + Convert<A::f64>,
    Self: Narrow<A::u8> + Narrow<A::u16> + Narrow<A::u32> + Convert<A::u64>,
    Self: Narrow<A::i8> + Narrow<A::i16> + Narrow<A::i32> + Convert<A::i64>,
{
}

pub trait Task {
    type Result;

    fn run<A: Arch>(self) -> Self::Result;
}

pub trait Possible {
    fn supported() -> bool;

    unsafe fn invoke_unchecked<T: Task>(task: T) -> T::Result;

    #[inline]
    fn try_invoke<T: Task>(task: T) -> Option<T::Result> {
        if Self::supported() {
            Some(unsafe { Self::invoke_unchecked::<T>(task) })
        } else {
            None
        }
    }

    #[inline]
    unsafe fn specialize_unchecked<T: Task>() -> fn(T) -> T::Result {
        fn invoke<A: Possible + ?Sized, U: Task>(task: U) -> U::Result {
            unsafe { A::invoke_unchecked::<U>(task) }
        }

        invoke::<Self, T>
    }

    #[inline]
    fn specialize_unsafe<T: Task>() -> unsafe fn(T) -> T::Result {
        unsafe { Self::specialize_unchecked::<T>() }
    }

    #[inline]
    fn try_specialize<T: Task>() -> Option<fn(T) -> T::Result> {
        if Self::supported() {
            Some(unsafe { Self::specialize_unchecked::<T>() })
        } else {
            None
        }
    }
}

pub unsafe trait Supported: Possible {
    #[inline]
    fn invoke<T: Task>(task: T) -> T::Result {
        unsafe { Self::invoke_unchecked(task) }
    }

    #[inline]
    fn specialize<T: Task>() -> fn(T) -> T::Result {
        fn invoke<A: Supported + ?Sized, U: Task>(task: U) -> U::Result {
            A::invoke::<U>(task)
        }

        invoke::<Self, T>
    }
}

#[cfg(test)]
mod tests {
    use core::fmt::Debug;
    use core::mem;
    use core::ops::{Add, Div, Mul, Neg, Shl, Shr, Sub};
    use core::ops::{BitAnd, BitOr, BitXor, Not};
    use std::num::Wrapping;

    use crate::{arch::*, simd::*, Arch, Possible, Supported, Task};

    fn test_ops<S, U>(
        type_: &str,
        values: &[S::Elem],
        eq: fn(&S::Elem, &S::Elem) -> bool,
        unary_ops: &[(fn(S) -> S, fn(S::Elem) -> S::Elem, &str)],
        binary_ops: &[(fn(S, S) -> S, fn(S::Elem, S::Elem) -> S::Elem, &str)],
        shift_ops: &[(fn(S, usize) -> S, fn(S::Elem, usize) -> S::Elem, &str)],
        cmp_ops: &[(
            fn(&S, &S) -> S::Mask,
            fn(&S::Elem, &S::Elem) -> <S::Mask as Simd>::Elem,
            &str,
        )],
    ) where
        S: Simd + AsSlice,
        S::Elem: Copy + Debug,
        S::Mask: Simd,
        U: Simd<Mask = S::Mask> + AsSlice + LanesEq,
        <S::Mask as Simd>::Elem: From<bool> + Copy + Debug + PartialEq,
    {
        // let mask_values = [false.into(), true.into(), false.into()]
        //     .into_iter()
        //     .cycle()
        //     .take(S::LANES * 2)
        //     .collect::<Vec<<S::Mask as Simd>::Elem>>();

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
            for ys in values.chunks(S::LANES) {
                for (vector, scalar, op) in binary_ops {
                    let res = vector(S::new(*x), S::from_slice(ys));
                    for (y, out) in ys.iter().zip(res.as_slice().iter()) {
                        let scalar = scalar(*x, *y);
                        assert!(
                            eq(&scalar, out),
                            "expected {}::{}({:?}, {:?}) == {:?}, got {:?}",
                            type_,
                            op,
                            *x,
                            *y,
                            scalar,
                            *out,
                        );
                    }
                }

                // for (vector, scalar, op) in cmp_ops {
                //     let res = vector(&S::new(*x), &S::from_slice(ys));
                //     for (y, out) in ys.iter().zip(res.as_slice().iter()) {
                //         let scalar = scalar(x, y);
                //         assert!(
                //             &scalar == out,
                //             "expected {}::{}({:?}, {:?}) == {:?}, got {:?}",
                //             type_,
                //             op,
                //             *x,
                //             *y,
                //             scalar,
                //             *out,
                //         );
                //     }
                // }

                // for m in mask_values.chunks(S::LANES) {
                //     let res = S::Mask::from_slice(m).select(S::new(*x), S::from_slice(ys));
                //     for ((m, y), out) in m.iter().zip(ys.iter()).zip(res.as_slice().iter()) {
                //         let scalar = if *m == true.into() { *x } else { *y };
                //         assert!(
                //             eq(&scalar, out),
                //             "expected {}::Mask::select({:?}, {:?}, {:?}) == {:?}, got {:?}",
                //             type_,
                //             *m,
                //             *x,
                //             *y,
                //             scalar,
                //             *out,
                //         );
                //     }
                // }
            }
        }

        for xs in values.chunks(S::LANES) {
            for (vector, scalar, op) in shift_ops {
                for shift in 0..(mem::size_of::<S::Elem>() * 16) {
                    let res = vector(S::from_slice(xs), shift);
                    for (x, out) in xs.iter().zip(res.as_slice().iter()) {
                        let scalar = scalar(*x, shift);
                        assert!(
                            eq(&scalar, out),
                            "expected {}::{}({:?}, {:?}) == {:?}, got {:?}",
                            type_,
                            op,
                            *x,
                            shift,
                            scalar,
                            *out,
                        );
                    }
                }
            }
        }
    }

    macro_rules! test_float {
        ($type:ident, $uint:ident) => {{
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

            test_ops::<A::$type, A::$uint>(
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
                &[],
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
        ($type:ident, $uint:ident) => {{
            let values = ($type::MIN..=$type::MAX)
                .step_by((1 << ($type::BITS as usize - 7)) + 1)
                .take(64)
                .collect::<Vec<$type>>();

            test_ops::<A::$type, A::$uint>(
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
                    (A::$type::shl, |x, s| (Wrapping(x) << s).0, "shl"),
                    (A::$type::shr, |x, s| (Wrapping(x) >> s).0, "shr"),
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
        ($type:ident, $uint:ident) => {{
            let values = [0, 1]
                .into_iter()
                .cycle()
                .take(64)
                .collect::<Vec<$uint>>();

            test_ops::<A::$type, A::$uint>(
                stringify!($type),
                &values,
                bool::eq,
                &[(A::$type::not, bool::not, "not")],
                &[
                    (A::$type::bitand, bool::bitand, "bitand"),
                    (A::$type::bitor, bool::bitor, "bitor"),
                    (A::$type::bitxor, bool::bitxor, "bitxor"),
                    (A::$type::max, bool::max, "max"),
                    (A::$type::min, bool::min, "min"),
                ],
                &[],
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

    struct TestArch;

    impl Task for TestArch {
        type Result = ();

        fn run<A: Arch>(self) {
            test_float!(f32, u32);
            test_float!(f64, u64);

            test_int!(u8, u8);
            test_int!(u16, u16);
            test_int!(u32, u32);
            test_int!(u64, u64);

            test_int!(i8, u8);
            test_int!(i16, u16);
            test_int!(i32, u32);
            test_int!(i64, u64);

            // test_mask!(m8);
            // test_mask!(m16);
            // test_mask!(m32);
            // test_mask!(m64);
        }
    }

    #[test]
    fn scalar() {
        Scalar::invoke(TestArch);
    }

    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // mod x86 {
    //     use super::*;

    //     #[test]
    //     fn sse2() {
    //         Sse2::invoke(TestArch);
    //     }

    //     #[test]
    //     fn sse4_2() {
    //         Sse4_2::try_invoke(TestArch);
    //     }

    //     #[test]
    //     fn avx2() {
    //         Avx2::try_invoke(TestArch);
    //     }
    // }
}
