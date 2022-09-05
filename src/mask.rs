#![allow(non_camel_case_types)]

use core::fmt::{self, Debug};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct m32(u32);

impl m32 {
    pub const TRUE: m32 = m32(!0);
    pub const FALSE: m32 = m32(0);
}

impl From<bool> for m32 {
    fn from(value: bool) -> m32 {
        if value {
            m32::TRUE
        } else {
            m32::FALSE
        }
    }
}

impl From<m32> for bool {
    fn from(value: m32) -> bool {
        value == m32::TRUE
    }
}

impl BitAnd for m32 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        m32(self.0 & rhs.0)
    }
}

impl BitAndAssign for m32 {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for m32 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        m32(self.0 | rhs.0)
    }
}

impl BitOrAssign for m32 {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for m32 {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        m32(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for m32 {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Not for m32 {
    type Output = Self;

    fn not(self) -> Self::Output {
        m32(!self.0)
    }
}

impl Debug for m32 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&bool::from(*self), fmt)
    }
}
