use crate::{Possible, Supported, Task};

mod avx2;
mod scalar;
mod sse2;
mod sse4_2;
mod sse_macros;

pub struct Scalar;

impl Possible for Scalar {
    fn supported() -> bool {
        true
    }

    unsafe fn specialize_unchecked<T: Task>() -> fn(T) -> T::Result {
        T::run::<scalar::ScalarImpl>
    }
}

impl Supported for Scalar {
    fn specialize<T: Task>() -> fn(T) -> T::Result {
        unsafe { Self::specialize_unchecked::<T>() }
    }
}

pub struct Sse2;

impl Possible for Sse2 {
    fn supported() -> bool {
        is_x86_feature_detected!("sse2")
    }

    unsafe fn specialize_unchecked<T: Task>() -> fn(T) -> T::Result {
        T::run::<sse2::Sse2Impl>
    }
}

#[cfg(target_feature = "sse2")]
impl Supported for Sse2 {
    fn specialize<T: Task>() -> fn(T) -> T::Result {
        unsafe { Self::specialize_unchecked::<T>() }
    }
}

pub struct Sse4_2;

impl Possible for Sse4_2 {
    fn supported() -> bool {
        is_x86_feature_detected!("sse4.2")
    }

    unsafe fn specialize_unchecked<T: Task>() -> fn(T) -> T::Result {
        T::run::<sse4_2::Sse4_2Impl>
    }
}

#[cfg(target_feature = "sse4.2")]
impl Supported for Sse4_2 {
    fn specialize<T: Task>() -> fn(T) -> T::Result {
        unsafe { Self::specialize_unchecked::<T>() }
    }
}

pub struct Avx2;

impl Possible for Avx2 {
    fn supported() -> bool {
        is_x86_feature_detected!("avx2")
    }

    unsafe fn specialize_unchecked<T: Task>() -> fn(T) -> T::Result {
        T::run::<avx2::Avx2Impl>
    }
}

#[cfg(target_feature = "avx2")]
impl Supported for Avx2 {
    fn specialize<T: Task>() -> fn(T) -> T::Result {
        unsafe { Self::specialize_unchecked::<T>() }
    }
}
