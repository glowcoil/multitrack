macro_rules! float_type {
    (
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

            fn new(elem: Self::Elem) -> Self {
                unsafe { $float($set(elem)) }
            }

            fn as_slice(&self) -> &[Self::Elem] {
                unsafe {
                    slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES)
                }
            }

            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                unsafe {
                    slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES)
                }
            }

            fn from_slice(slice: &[Self::Elem]) -> Self {
                assert!(slice.len() == Self::LANES);
                unsafe { $float($load(slice.as_ptr())) }
            }

            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                assert!(slice.len() == Self::LANES);
                unsafe {
                    $store(slice.as_mut_ptr(), self.0);
                }
            }

            fn align_slice(slice: &[Self::Elem]) -> (&[Self::Elem], &[Self], &[Self::Elem]) {
                unsafe { slice.align_to::<Self>() }
            }

            fn align_mut_slice(
                slice: &mut [Self::Elem],
            ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]) {
                unsafe { slice.align_to_mut::<Self>() }
            }
        }

        impl Debug for $float {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(self.as_slice(), fmt)
            }
        }

        impl Default for $float {
            fn default() -> Self {
                unsafe { mem::zeroed() }
            }
        }

        impl LanesEq for $float {
            type Output = $mask;

            fn eq(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmpeq(self.0, other.0))) }
            }

            fn ne(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmpneq(self.0, other.0))) }
            }
        }

        impl LanesOrd for $float {
            fn lt(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmplt(self.0, other.0))) }
            }

            fn le(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmple(self.0, other.0))) }
            }

            fn gt(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmpgt(self.0, other.0))) }
            }

            fn ge(&self, other: &Self) -> Self::Output {
                unsafe { $mask($cast_to_int($cmpge(self.0, other.0))) }
            }

            fn max(self, other: Self) -> Self {
                unsafe { $float($max(self.0, other.0)) }
            }

            fn min(self, other: Self) -> Self {
                unsafe { $float($min(self.0, other.0)) }
            }
        }

        impl Index<usize> for $float {
            type Output = <Self as Simd>::Elem;

            fn index(&self, index: usize) -> &Self::Output {
                &self.as_slice()[index]
            }
        }

        impl IndexMut<usize> for $float {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }

        impl Select<$float> for $mask {
            fn select(self, if_true: $float, if_false: $float) -> $float {
                unsafe {
                    let mask = $cast_from_int(self.0);
                    $float($blend(if_false.0, if_true.0, mask))
                }
            }
        }

        impl Float for $float {}

        impl Add for $float {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                unsafe { $float($add(self.0, rhs.0)) }
            }
        }

        impl AddAssign for $float {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Sub for $float {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                unsafe { $float($sub(self.0, rhs.0)) }
            }
        }

        impl SubAssign for $float {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl Mul for $float {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                unsafe { $float($mul(self.0, rhs.0)) }
            }
        }

        impl MulAssign for $float {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Div for $float {
            type Output = Self;

            fn div(self, rhs: Self) -> Self {
                unsafe { $float($div(self.0, rhs.0)) }
            }
        }

        impl DivAssign for $float {
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }

        impl Neg for $float {
            type Output = Self;

            fn neg(self) -> Self {
                unsafe { $float($xor(self.0, $set(-0.0))) }
            }
        }
    };
}
pub(super) use float_type;

macro_rules! int_type {
    ($int:ident, $elem:ident, $lanes:literal, $mask:ident, $set:ident, $blend:ident) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct $int(__m128i);

        impl Simd for $int {
            type Elem = $elem;
            type Mask = $mask;

            const LANES: usize = $lanes;

            fn new(elem: Self::Elem) -> Self {
                unsafe { $int($set(mem::transmute(elem))) }
            }

            fn as_slice(&self) -> &[Self::Elem] {
                unsafe {
                    slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES)
                }
            }

            fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
                unsafe {
                    slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES)
                }
            }

            fn from_slice(slice: &[Self::Elem]) -> Self {
                assert!(slice.len() == Self::LANES);
                unsafe { $int(_mm_loadu_si128(slice.as_ptr() as *const __m128i)) }
            }

            fn write_to_slice(&self, slice: &mut [Self::Elem]) {
                assert!(slice.len() == Self::LANES);
                unsafe {
                    _mm_storeu_si128(slice.as_mut_ptr() as *mut __m128i, self.0);
                }
            }

            fn align_slice(slice: &[Self::Elem]) -> (&[Self::Elem], &[Self], &[Self::Elem]) {
                unsafe { slice.align_to::<Self>() }
            }

            fn align_mut_slice(
                slice: &mut [Self::Elem],
            ) -> (&mut [Self::Elem], &mut [Self], &mut [Self::Elem]) {
                unsafe { slice.align_to_mut::<Self>() }
            }
        }

        impl Debug for $int {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(self.as_slice(), fmt)
            }
        }

        impl Default for $int {
            fn default() -> Self {
                unsafe { mem::zeroed() }
            }
        }

        impl Index<usize> for $int {
            type Output = <Self as Simd>::Elem;

            fn index(&self, index: usize) -> &Self::Output {
                &self.as_slice()[index]
            }
        }

        impl IndexMut<usize> for $int {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }

        impl Bitwise for $int {}

        impl BitAnd for $int {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self::Output {
                unsafe { $int(_mm_and_si128(self.0, rhs.0)) }
            }
        }

        impl BitAndAssign for $int {
            fn bitand_assign(&mut self, rhs: Self) {
                *self = *self & rhs;
            }
        }

        impl BitOr for $int {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self::Output {
                unsafe { $int(_mm_or_si128(self.0, rhs.0)) }
            }
        }

        impl BitOrAssign for $int {
            fn bitor_assign(&mut self, rhs: Self) {
                *self = *self | rhs;
            }
        }

        impl BitXor for $int {
            type Output = Self;

            fn bitxor(self, rhs: Self) -> Self::Output {
                unsafe { $int(_mm_xor_si128(self.0, rhs.0)) }
            }
        }

        impl BitXorAssign for $int {
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = *self ^ rhs;
            }
        }

        impl Not for $int {
            type Output = Self;

            fn not(self) -> Self::Output {
                unsafe {
                    let zero = _mm_setzero_si128();
                    $int(_mm_andnot_si128(self.0, _mm_cmpeq_epi8(zero, zero)))
                }
            }
        }

        impl Select<$int> for $mask {
            fn select(self, if_true: $int, if_false: $int) -> $int {
                unsafe { $int($blend(if_false.0, if_true.0, self.0)) }
            }
        }
    };
}
pub(super) use int_type;

macro_rules! impl_ord_mask {
    ($mask:ident) => {
        impl LanesEq for $mask {
            type Output = $mask;

            fn eq(&self, other: &Self) -> Self::Output {
                unsafe { $mask(_mm_cmpeq_epi8(self.0, other.0)) }
            }

            fn ne(&self, other: &Self) -> Self::Output {
                unsafe { $mask(_mm_xor_si128(self.0, other.0)) }
            }
        }

        impl LanesOrd for $mask {
            fn lt(&self, other: &Self) -> Self::Output {
                unsafe { $mask(_mm_andnot_si128(self.0, other.0)) }
            }

            fn le(&self, other: &Self) -> Self::Output {
                unsafe { $mask(_mm_or_si128(other.0, _mm_cmpeq_epi8(self.0, other.0))) }
            }

            fn max(self, other: Self) -> Self {
                unsafe { $mask(_mm_or_si128(self.0, other.0)) }
            }

            fn min(self, other: Self) -> Self {
                unsafe { $mask(_mm_and_si128(self.0, other.0)) }
            }
        }
    };
}
pub(super) use impl_ord_mask;

macro_rules! impl_int {
    ($int:ident, $set:ident, $add:ident, $sub:ident) => {
        impl Int for $int {}

        impl Add for $int {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                unsafe { $int($add(self.0, rhs.0)) }
            }
        }

        impl AddAssign for $int {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Sub for $int {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                unsafe { $int($sub(self.0, rhs.0)) }
            }
        }

        impl SubAssign for $int {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl MulAssign for $int {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Neg for $int {
            type Output = Self;

            fn neg(self) -> Self {
                unsafe { $int($sub($set(0), self.0)) }
            }
        }
    };
}
pub(super) use impl_int;
