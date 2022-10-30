#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

macro_rules! float_type {
    (
        $feature:literal,
        $float:ident, $inner:ident, $elem:ident, $lanes:literal, $mask:ident,
        $set:ident, $load:ident, $store:ident, $cast_to_int:ident, $cast_from_int:ident, $blend:ident,
        $cmpeq:ident, $cmpneq:ident, $cmplt:ident, $cmple:ident, $cmpgt:ident, $cmpge:ident,
        $min:ident, $max:ident, $add:ident, $sub:ident, $mul:ident, $div:ident, $xor:ident,
    ) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct $float($inner);

        impl Simd for $float {
            type Elem = $elem;
            type Mask = $mask;

            const LANES: usize = $lanes;

            #[inline(always)]
            fn new(elem: Self::Elem) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(elem: $elem) -> $float {
                    $float($set(elem))
                }

                unsafe { inner(elem) }
            }

            #[inline]
            fn as_slice(&self) -> &[Self::Elem] {
                unsafe {
                    slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES)
                }
            }

            #[inline]
            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                unsafe {
                    slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES)
                }
            }

            #[inline(always)]
            fn from_slice(slice: &[Self::Elem]) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(slice: &[$elem]) -> $float {
                    assert!(slice.len() == <$float as Simd>::LANES);
                    $float($load(slice.as_ptr()))
                }

                unsafe { inner(slice) }
            }

            #[inline(always)]
            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(vec: &$float, slice: &mut [$elem]) {
                    assert!(slice.len() == <$float as Simd>::LANES);
                    $store(slice.as_mut_ptr(), vec.0);
                }

                unsafe { inner(self, slice) }
            }

            #[inline]
            fn align_slice(slice: &[Self::Elem]) -> (&[Self::Elem], &[Self], &[Self::Elem]) {
                unsafe { slice.align_to::<Self>() }
            }

            #[inline]
            fn align_mut_slice(
                slice: &mut [Self::Elem],
            ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]) {
                unsafe { slice.align_to_mut::<Self>() }
            }
        }

        impl Debug for $float {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(self.as_slice(), fmt)
            }
        }

        impl Default for $float {
            #[inline]
            fn default() -> Self {
                unsafe { mem::zeroed() }
            }
        }

        impl LanesEq for $float {
            type Output = $mask;

            #[inline(always)]
            fn eq(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    $mask($cast_to_int($cmpeq(lhs.0, rhs.0)))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn ne(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    $mask($cast_to_int($cmpneq(lhs.0, rhs.0)))
                }

                unsafe { inner(self, other) }
            }
        }

        impl LanesOrd for $float {
            #[inline(always)]
            fn lt(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    $mask($cast_to_int($cmplt(lhs.0, rhs.0)))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn le(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    $mask($cast_to_int($cmple(lhs.0, rhs.0)))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn gt(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    $mask($cast_to_int($cmpgt(lhs.0, rhs.0)))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn ge(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: &$float, rhs: &$float) -> $mask {
                    $mask($cast_to_int($cmpge(lhs.0, rhs.0)))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($max(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn min(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($min(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }

        impl Index<usize> for $float {
            type Output = <Self as Simd>::Elem;

            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.as_slice()[index]
            }
        }

        impl IndexMut<usize> for $float {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }

        impl Select<$float> for $mask {
            #[inline(always)]
            fn select(self, if_true: $float, if_false: $float) -> $float {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(mask: $mask, if_true: $float, if_false: $float) -> $float {
                    let mask = $cast_from_int(mask.0);
                    $float($blend(if_false.0, if_true.0, mask))
                }

                unsafe { inner(self, if_true, if_false) }
            }
        }

        impl Float for $float {}

        impl Add for $float {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($add(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl AddAssign for $float {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Sub for $float {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($sub(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl SubAssign for $float {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl Mul for $float {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($mul(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl MulAssign for $float {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Div for $float {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $float, rhs: $float) -> $float {
                    $float($div(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl DivAssign for $float {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }

        impl Neg for $float {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(vec: $float) -> $float {
                    $float($xor(vec.0, $set(-0.0)))
                }

                unsafe { inner(self) }
            }
        }
    };
}
pub(super) use float_type;

macro_rules! int_type {
    ($feature:literal, $int:ident, $elem:ident, $lanes:literal, $mask:ident, $set:ident, $blend:ident) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct $int(__m128i);

        impl Simd for $int {
            type Elem = $elem;
            type Mask = $mask;

            const LANES: usize = $lanes;

            #[inline(always)]
            fn new(elem: Self::Elem) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(elem: $elem) -> $int {
                    $int($set(mem::transmute(elem)))
                }

                unsafe { inner(elem) }
            }

            #[inline]
            fn as_slice(&self) -> &[Self::Elem] {
                unsafe {
                    slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES)
                }
            }

            #[inline]
            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                unsafe {
                    slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES)
                }
            }

            #[inline(always)]
            fn from_slice(slice: &[Self::Elem]) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(slice: &[$elem]) -> $int {
                    assert!(slice.len() == <$int as Simd>::LANES);
                    $int(_mm_loadu_si128(slice.as_ptr() as *const __m128i))
                }

                unsafe { inner(slice) }
            }

            #[inline(always)]
            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(vec: &$int, slice: &mut [$elem]) {
                    assert!(slice.len() == <$int as Simd>::LANES);
                    _mm_storeu_si128(slice.as_mut_ptr() as *mut __m128i, vec.0);
                }

                unsafe {
                    inner(self, slice);
                }
            }

            #[inline]
            fn align_slice(slice: &[Self::Elem]) -> (&[Self::Elem], &[Self], &[Self::Elem]) {
                unsafe { slice.align_to::<Self>() }
            }

            #[inline]
            fn align_mut_slice(
                slice: &mut [Self::Elem],
            ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]) {
                unsafe { slice.align_to_mut::<Self>() }
            }
        }

        impl Debug for $int {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(self.as_slice(), fmt)
            }
        }

        impl Default for $int {
            #[inline]
            fn default() -> Self {
                unsafe { mem::zeroed() }
            }
        }

        impl Index<usize> for $int {
            type Output = <Self as Simd>::Elem;

            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.as_slice()[index]
            }
        }

        impl IndexMut<usize> for $int {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }

        impl Bitwise for $int {}

        impl BitAnd for $int {
            type Output = Self;

            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int(_mm_and_si128(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl BitAndAssign for $int {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = *self & rhs;
            }
        }

        impl BitOr for $int {
            type Output = Self;

            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int(_mm_or_si128(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl BitOrAssign for $int {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = *self | rhs;
            }
        }

        impl BitXor for $int {
            type Output = Self;

            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int(_mm_xor_si128(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl BitXorAssign for $int {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = *self ^ rhs;
            }
        }

        impl Not for $int {
            type Output = Self;

            #[inline(always)]
            fn not(self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(vec: $int) -> $int {
                    let zero = _mm_setzero_si128();
                    $int(_mm_andnot_si128(vec.0, _mm_cmpeq_epi8(zero, zero)))
                }

                unsafe { inner(self) }
            }
        }

        impl Select<$int> for $mask {
            #[inline(always)]
            fn select(self, if_true: $int, if_false: $int) -> $int {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(mask: $mask, if_true: $int, if_false: $int) -> $int {
                    $int($blend(if_false.0, if_true.0, mask.0))
                }

                unsafe { inner(self, if_true, if_false) }
            }
        }
    };
}
pub(super) use int_type;

macro_rules! impl_ord_mask {
    ($feature:literal, $mask:ident) => {
        impl LanesEq for $mask {
            type Output = $mask;

            #[inline(always)]
            fn eq(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: &$mask, rhs: &$mask) -> $mask {
                    $mask(_mm_cmpeq_epi8(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn ne(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: &$mask, rhs: &$mask) -> $mask {
                    $mask(_mm_xor_si128(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }

        impl LanesOrd for $mask {
            #[inline(always)]
            fn lt(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: &$mask, rhs: &$mask) -> $mask {
                    $mask(_mm_andnot_si128(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn le(&self, other: &Self) -> Self::Output {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: &$mask, rhs: &$mask) -> $mask {
                    $mask(_mm_or_si128(rhs.0, _mm_cmpeq_epi8(lhs.0, rhs.0)))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $mask, rhs: $mask) -> $mask {
                    $mask(_mm_or_si128(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }

            #[inline(always)]
            fn min(self, other: Self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $mask, rhs: $mask) -> $mask {
                    $mask(_mm_and_si128(lhs.0, rhs.0))
                }

                unsafe { inner(self, other) }
            }
        }
    };
}
pub(super) use impl_ord_mask;

macro_rules! impl_int {
    ($feature:literal, $int:ident, $elem:ident, $set:ident, $add:ident, $sub:ident, $shl:ident, $shr:ident) => {
        impl Int for $int {}

        impl Add for $int {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int($add(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl AddAssign for $int {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Sub for $int {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $int, rhs: $int) -> $int {
                    $int($sub(lhs.0, rhs.0))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl SubAssign for $int {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl MulAssign for $int {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Neg for $int {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(vec: $int) -> $int {
                    $int($sub($set(0), vec.0))
                }

                unsafe { inner(self) }
            }
        }

        impl Shl<usize> for $int {
            type Output = Self;

            #[inline]
            fn shl(self, rhs: usize) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $int, rhs: usize) -> $int {
                    let shift = rhs & ($elem::BITS as usize - 1);
                    $int($shl(lhs.0, _mm_cvtsi64_si128(shift as i64)))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl ShlAssign<usize> for $int {
            #[inline]
            fn shl_assign(&mut self, rhs: usize) {
                *self = *self << rhs;
            }
        }

        impl Shr<usize> for $int {
            type Output = Self;

            #[inline]
            fn shr(self, rhs: usize) -> Self {
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(lhs: $int, rhs: usize) -> $int {
                    let shift = rhs & ($elem::BITS as usize - 1);
                    $int($shr(lhs.0, _mm_cvtsi64_si128(shift as i64)))
                }

                unsafe { inner(self, rhs) }
            }
        }

        impl ShrAssign<usize> for $int {
            #[inline]
            fn shr_assign(&mut self, rhs: usize) {
                *self = *self >> rhs;
            }
        }
    };
}
pub(super) use impl_int;

#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sll_epi8_fallback(_a: __m128i, _count: __m128i) -> __m128i {
    unimplemented!()
}

#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_srl_epi8_fallback(_a: __m128i, _count: __m128i) -> __m128i {
    unimplemented!()
}

#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sra_epi8_fallback(_a: __m128i, _count: __m128i) -> __m128i {
    unimplemented!()
}

#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_sra_epi64_fallback(_a: __m128i, _count: __m128i) -> __m128i {
    unimplemented!()
}
