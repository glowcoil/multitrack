#![allow(non_camel_case_types)]

use core::fmt::{self, Debug};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

macro_rules! mask_type {
    ($mask:ident, $inner:ty) => {
        #[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $mask($inner);

        impl $mask {
            pub const TRUE: $mask = $mask(!0);
            pub const FALSE: $mask = $mask(0);
        }

        impl From<bool> for $mask {
            fn from(value: bool) -> $mask {
                if value {
                    $mask::TRUE
                } else {
                    $mask::FALSE
                }
            }
        }

        impl From<$mask> for bool {
            fn from(value: $mask) -> bool {
                value == $mask::TRUE
            }
        }

        impl BitAnd for $mask {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self::Output {
                $mask(self.0 & rhs.0)
            }
        }

        impl BitAndAssign for $mask {
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl BitOr for $mask {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self::Output {
                $mask(self.0 | rhs.0)
            }
        }

        impl BitOrAssign for $mask {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl BitXor for $mask {
            type Output = Self;

            fn bitxor(self, rhs: Self) -> Self::Output {
                $mask(self.0 ^ rhs.0)
            }
        }

        impl BitXorAssign for $mask {
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }

        impl Not for $mask {
            type Output = Self;

            fn not(self) -> Self::Output {
                $mask(!self.0)
            }
        }

        impl Debug for $mask {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(&bool::from(*self), fmt)
            }
        }
    }
}

mask_type! { m8, u8 }
mask_type! { m16, u16 }
mask_type! { m32, u32 }
mask_type! { m64, u64 }
