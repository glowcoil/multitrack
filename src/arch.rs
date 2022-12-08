use crate::{Arch, Possible, Supported, Task};

mod scalar;

// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// mod x86;
// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// pub use x86::*;

pub struct Scalar;

impl Possible for Scalar {
    #[inline]
    fn supported() -> bool {
        true
    }

    #[inline]
    unsafe fn invoke_unchecked<T: Task>(task: T) -> T::Result {
        scalar::ScalarImpl::invoke(task)
    }
}

unsafe impl Supported for Scalar {}
