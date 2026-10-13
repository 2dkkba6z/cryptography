//! P-256 elliptic curve group operations.
//!
//! Points are represented in two forms:
//!
//! - [`AffinePoint`] — standard `(x, y)` coordinates, used for storage and
//!   table entries. Includes an `infinity` flag for the identity element.
//! - [`ProjectivePoint`] — Jacobian `(X : Y : Z)` coordinates, used during
//!   multi-step scalar multiplication to avoid per-step field inversions.
//!
//! Use [`ProjectivePoint::to_affine`] to convert back to affine coordinates.
//! Precomputed table builders normalize points in batches internally.

use core::ops::{Add, Neg, Sub};
use std::sync::OnceLock;

use crate::{Endianness, field::FieldElement, scalar::Scalar};

const BASE_WINDOWS: usize = 32;
const BASE_WINDOW_POINTS: usize = 256;

const CURVE_B: FieldElement = FieldElement::from_montgomery_limbs([
    0xd89c_df62_29c4_bddf,
    0xacf0_05cd_7884_3090,
    0xe5a2_20ab_f721_2ed6,
    0xdc30_061d_0487_4834,
]);

const GENERATOR_X: FieldElement = FieldElement::from_montgomery_limbs([
    0x79e7_30d4_18a9_143c,
    0x75ba_95fc_5fed_b601,
    0x79fb_732b_7762_2510,
    0x1890_5f76_a537_55c6,
]);

const GENERATOR_Y: FieldElement = FieldElement::from_montgomery_limbs([
    0xddf2_5357_ce95_560a,
    0x8b4a_b8e4_ba19_e45c,
    0xd2e8_8688_dd21_f325,
    0x8571_ff18_2588_5d85,
]);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffinePoint {
    x: FieldElement,
    y: FieldElement,
    infinity: bool,
}

impl AffinePoint {
    pub const IDENTITY: Self = Self {
        x: FieldElement::ZERO,
        y: FieldElement::ZERO,
        infinity: true,
    };

    pub const GENERATOR: Self = Self {
        x: GENERATOR_X,
        y: GENERATOR_Y,
        infinity: false,
    };

    #[inline]
    pub fn identity() -> Self {
        Self::IDENTITY
    }

    #[inline]
    pub fn generator() -> Self {
        Self::GENERATOR
    }

    /// Constructs a non-identity point from its coordinates.
    ///
    /// Returns `None` if the coordinates do not satisfy the curve equation.
    /// Use [`Self::IDENTITY`] for the identity point.
    #[inline]
    pub fn new(x: FieldElement, y: FieldElement) -> Option<Self> {
        let point = Self {
            x,
            y,
            infinity: false,
        };

        point.is_on_curve().then_some(point)
    }

    /// Parses a 64-byte `X || Y` encoding.
    ///
    /// Each coordinate uses the requested byte order. Exactly 64 zero
    /// bytes decode to the identity. All other inputs must have canonical
    /// coordinates and satisfy the curve equation, or this returns `None`.
    #[inline]
    pub fn from_uncompressed(bytes: &[u8; 64], endianness: Endianness) -> Option<Self> {
        if bytes == &[0u8; 64] {
            return Some(Self::IDENTITY);
        }

        let mut x_bytes = [0u8; 32];
        let mut y_bytes = [0u8; 32];
        x_bytes.copy_from_slice(&bytes[0..32]);
        y_bytes.copy_from_slice(&bytes[32..64]);

        if endianness == Endianness::Little {
            x_bytes.reverse();
            y_bytes.reverse();
        }

        Self::new(
            FieldElement::from_be_bytes(x_bytes)?,
            FieldElement::from_be_bytes(y_bytes)?,
        )
    }

    /// Parses a parity prefix followed by a 32-byte X-coordinate.
    ///
    /// Prefix `0x02` selects even Y; `0x03` selects odd Y. Only X uses the
    /// requested byte order. Returns `None` for an invalid prefix, a
    /// noncanonical X-coordinate, or an X-coordinate with no curve point.
    /// The identity has no compressed representation.
    #[inline]
    pub fn from_compressed(bytes: &[u8; 33], endianness: Endianness) -> Option<Self> {
        // `0x02`: Compressed point with an even Y-coordinate
        // `0x03`: Compressed point with an odd Y-coordinate
        if bytes[0] != 0x02 && bytes[0] != 0x03 {
            return None;
        }

        let mut x_bytes = [0u8; 32];
        x_bytes.copy_from_slice(&bytes[1..33]);

        if endianness == Endianness::Little {
            x_bytes.reverse();
        }

        let x = FieldElement::from_be_bytes(x_bytes)?;
        let rhs = x.square() * x - triple(x) + CURVE_B;
        let mut y = rhs.sqrt()?;

        if (y.to_be_bytes()[31] & 1) != (bytes[0] & 1) {
            y = -y;
        }

        ((y.to_be_bytes()[31] & 1) == (bytes[0] & 1)).then_some(Self {
            x,
            y,
            infinity: false,
        })
    }

    #[inline]
    pub fn to_projective(self) -> ProjectivePoint {
        if self.infinity {
            ProjectivePoint::IDENTITY
        } else {
            ProjectivePoint {
                x: self.x,
                y: self.y,
                z: FieldElement::ONE,
            }
        }
    }

    /// Serializes the affine point to a 64-byte uncompressed representation.
    ///
    /// As specified in SIMD-565, the point at infinity (identity) is canonically
    /// encoded as 64 bytes of zeroes. This ensures compatibility with the fixed-length
    /// layout expected by syscalls like `sol_curve_group_op`.
    #[inline]
    pub fn to_uncompressed(self, endianness: Endianness) -> [u8; 64] {
        if self.is_identity() {
            return [0u8; 64];
        }

        let mut x_bytes = self.x.to_be_bytes();
        let mut y_bytes = self.y.to_be_bytes();

        if endianness == Endianness::Little {
            x_bytes.reverse();
            y_bytes.reverse();
        }

        let mut out = [0u8; 64];
        out[0..32].copy_from_slice(&x_bytes);
        out[32..64].copy_from_slice(&y_bytes);
        out
    }

    /// Serializes the affine point to a 33-byte compressed representation.
    ///
    /// Returns `None` if the point is the point at infinity (identity).
    /// According to SIMD-565, decompression of the identity point is explicitly
    /// unsupported within the fixed 33-byte layout (e.g., for `sol_curve_decompress`),
    /// as it cannot satisfy the required `0x02` or `0x03` parity prefix. Enforcing
    /// this via `Option` ensures the output is always valid for the syscall.
    #[inline]
    pub fn to_compressed(self, endianness: Endianness) -> Option<[u8; 33]> {
        if self.is_identity() {
            return None;
        }

        let mut x_bytes = self.x.to_be_bytes();
        let y_bytes = self.y.to_be_bytes();

        let prefix = if (y_bytes[31] & 1) == 1 { 0x03 } else { 0x02 };

        if endianness == Endianness::Little {
            x_bytes.reverse();
        }

        let mut out = [0u8; 33];
        out[0] = prefix;
        out[1..33].copy_from_slice(&x_bytes);

        Some(out)
    }

    #[inline]
    pub fn is_identity(self) -> bool {
        self.infinity
    }

    #[inline]
    pub fn x(self) -> Option<FieldElement> {
        (!self.infinity).then_some(self.x)
    }

    #[inline]
    pub fn y(self) -> Option<FieldElement> {
        (!self.infinity).then_some(self.y)
    }

    #[inline]
    fn is_on_curve(self) -> bool {
        if self.infinity {
            return true;
        }

        let y2 = self.y.square();
        let x2 = self.x.square();
        let x3 = x2 * self.x;
        let three_x = triple(self.x);
        y2 == x3 - three_x + CURVE_B
    }
}

impl Neg for AffinePoint {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        if self.infinity {
            self
        } else {
            Self {
                x: self.x,
                y: -self.y,
                infinity: false,
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectivePoint {
    x: FieldElement,
    y: FieldElement,
    z: FieldElement,
}

impl PartialEq for ProjectivePoint {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let self_is_identity = self.z.is_zero();
        let other_is_identity = other.z.is_zero();

        if self_is_identity || other_is_identity {
            return self_is_identity && other_is_identity;
        }

        let self_z2 = self.z.square();
        let other_z2 = other.z.square();
        let self_z3 = self_z2 * self.z;
        let other_z3 = other_z2 * other.z;

        // x1 * z2^2 == x2 * z1^2
        // y1 * z2^3 == y2 * z1^3

        self.x * other_z2 == other.x * self_z2 && self.y * other_z3 == other.y * self_z3
    }
}

impl Eq for ProjectivePoint {}

impl ProjectivePoint {
    pub const IDENTITY: Self = Self {
        x: FieldElement::ZERO,
        y: FieldElement::ONE,
        z: FieldElement::ZERO,
    };

    pub const GENERATOR: Self = Self {
        x: GENERATOR_X,
        y: GENERATOR_Y,
        z: FieldElement::ONE,
    };

    #[inline]
    pub fn identity() -> Self {
        Self::IDENTITY
    }

    #[inline]
    pub fn generator() -> Self {
        Self::GENERATOR
    }

    #[inline]
    pub fn from_affine(point: AffinePoint) -> Self {
        point.to_projective()
    }

    #[inline]
    pub fn to_affine(self) -> AffinePoint {
        match self.z.invert() {
            Some(zinv) => {
                let zinv2 = zinv.square();
                AffinePoint {
                    x: self.x * zinv2,
                    y: self.y * zinv2 * zinv,
                    infinity: false,
                }
            }
            None => AffinePoint::IDENTITY,
        }
    }

    #[inline]
    pub fn to_uncompressed(self, endianness: Endianness) -> [u8; 64] {
        self.to_affine().to_uncompressed(endianness)
    }

    #[inline]
    pub fn is_identity(self) -> bool {
        self.z.is_zero()
    }

    #[inline]
    pub fn has_affine_x(self, x: FieldElement) -> bool {
        !self.is_identity() && self.x == x * self.z.square()
    }

    #[inline]
    pub fn double(self) -> Self {
        if self.is_identity() || self.y.is_zero() {
            return Self::IDENTITY;
        }

        // With t = 2Y and a = 4Y^2: S = X * a, 8Y^4 = a^2 / 2, Z3 = t * Z.
        let t = double(self.y);
        let a = t.square();
        let zz = self.z.square();
        let s = self.x * a;
        let m = triple((self.x - zz) * (self.x + zz));
        let x = m.square() - double(s);
        let y = m * (s - x) - a.square().half();
        let z = t * self.z;

        Self { x, y, z }
    }

    #[inline]
    pub fn add_mixed(self, rhs: AffinePoint) -> Self {
        if rhs.infinity {
            return self;
        }
        if self.is_identity() {
            return rhs.to_projective();
        }

        let z1z1 = self.z.square();
        let u2 = rhs.x * z1z1;
        let s2 = rhs.y * self.z * z1z1;
        let h = u2 - self.x;
        let slope_num = s2 - self.y;

        if h.is_zero() {
            return if slope_num.is_zero() {
                self.double()
            } else {
                Self::IDENTITY
            };
        }

        let hh = h.square();
        let hhh = h * hh;
        let v = self.x * hh;
        let x = slope_num.square() - hhh - double(v);
        let y = slope_num * (v - x) - self.y * hhh;
        let z = self.z * h;

        Self { x, y, z }
    }

    /// Multiplies this point by a canonical, big-endian scalar.
    ///
    /// Returns `None` when `scalar` is greater than or equal to the P-256
    /// group order. This routine is variable time and must only be used with
    /// public scalars.
    #[inline]
    pub fn mul_scalar_vartime(self, scalar: [u8; 32]) -> Option<Self> {
        Scalar::is_canonical(&scalar, Endianness::Big)
            .then(|| self.mul_scalar_vartime_unchecked(scalar))
    }

    /// Multiplies this point by arbitrary big-endian scalar bytes.
    ///
    /// Scalars greater than or equal to the P-256 group order are accepted;
    /// group arithmetic implicitly reduces them modulo the group order. This
    /// routine is variable time and must only be used with public scalars.
    #[inline]
    pub fn mul_scalar_vartime_unchecked(self, scalar: [u8; 32]) -> Self {
        // Odd multiples P, 3P, ..., 15P for a width-5 signed NAF.
        let affine_table = batch_normalize(odd_multiples::<8>(self));

        let naf = wnaf5(&scalar);
        let mut out = Self::IDENTITY;

        for &digit in naf.iter().rev() {
            if digit > 0 {
                out = out.double_add_mixed(affine_table[(digit >> 1) as usize]);
            } else if digit < 0 {
                out = out.double_add_mixed(-affine_table[((-digit) >> 1) as usize]);
            } else {
                out = out.double();
            }
        }

        out
    }

    /// Computes `2 * self + rhs` as `(self + rhs) + self`.
    ///
    /// The mixed addition also yields `self` rescaled to the `Z` of the sum,
    /// so the second addition uses the cheaper co-Z formula. Degenerate cases
    /// fall back to doubling followed by a mixed addition.
    #[inline]
    fn double_add_mixed(self, rhs: AffinePoint) -> Self {
        if rhs.infinity || self.is_identity() {
            return self.double().add_mixed(rhs);
        }

        let z1z1 = self.z.square();
        let u2 = rhs.x * z1z1;
        let s2 = rhs.y * self.z * z1z1;
        let h = u2 - self.x;
        let slope_num = s2 - self.y;

        if h.is_zero() {
            return self.double().add_mixed(rhs);
        }

        let hh = h.square();
        let hhh = h * hh;
        // self rescaled to Z = self.z * h.
        let x1 = self.x * hh;
        let y1 = self.y * hhh;
        let x2 = slope_num.square() - hhh - double(x1);
        let dx = x1 - x2;
        let y2 = slope_num * dx - y1;
        let z = self.z * h;

        if dx.is_zero() {
            return self.double().add_mixed(rhs);
        }

        let a = dx.square();
        let b = x2 * a;
        let c = x1 * a;
        let dy = y1 - y2;
        let x = dy.square() - b - c;
        let y = dy * (b - x) - y2 * (c - b);

        Self { x, y, z: z * dx }
    }

    /// Multiplies the generator by a canonical, big-endian scalar.
    ///
    /// Returns `None` when `scalar` is greater than or equal to the P-256
    /// group order. This routine is variable time and must only be used with
    /// public scalars.
    #[inline]
    pub fn fixed_base_scalar_mul_vartime(scalar: [u8; 32]) -> Option<Self> {
        Scalar::is_canonical(&scalar, Endianness::Big)
            .then(|| Self::fixed_base_scalar_mul_vartime_unchecked(scalar))
    }

    /// Multiplies the generator by arbitrary big-endian scalar bytes.
    ///
    /// Scalars greater than or equal to the P-256 group order are accepted;
    /// group arithmetic implicitly reduces them modulo the group order. This
    /// routine is variable time and must only be used with public scalars.
    #[inline]
    pub fn fixed_base_scalar_mul_vartime_unchecked(scalar: [u8; 32]) -> Self {
        mul_window8_vartime(generator_window8_table(), &scalar)
    }

    /// Computes a double-scalar multiplication with canonical, big-endian
    /// scalars.
    ///
    /// Returns `None` when either scalar is greater than or equal to the P-256
    /// group order. Both scalars are validated before group computation begins.
    #[inline]
    pub fn double_scalar_mul_vartime(
        generator_scalar: [u8; 32],
        point: AffinePoint,
        point_scalar: [u8; 32],
    ) -> Option<Self> {
        let canonical = Scalar::is_canonical(&generator_scalar, Endianness::Big)
            && Scalar::is_canonical(&point_scalar, Endianness::Big);

        canonical.then(|| {
            Self::double_scalar_mul_vartime_unchecked(generator_scalar, point, point_scalar)
        })
    }

    /// Computes a double-scalar multiplication with arbitrary big-endian
    /// scalar bytes.
    ///
    /// Scalars greater than or equal to the P-256 group order are accepted;
    /// group arithmetic implicitly reduces them modulo the group order. This
    /// routine is variable time and must only be used with public scalars.
    #[inline]
    pub fn double_scalar_mul_vartime_unchecked(
        generator_scalar: [u8; 32],
        point: AffinePoint,
        point_scalar: [u8; 32],
    ) -> Self {
        // Odd multiples G, 3G, ..., 127G for a width-8 signed NAF of the
        // generator scalar and P, 3P, ..., 15P for a width-5 one of the point
        // scalar.
        let generator_table = generator_odd_multiples_table();
        let point_table: [AffinePoint; 8] =
            batch_normalize(odd_multiples(Self::from_affine(point)));
        let generator_naf = wnaf::<8>(&generator_scalar);
        let point_naf = wnaf5(&point_scalar);
        let mut out = Self::IDENTITY;

        for i in (0..261).rev() {
            let g = generator_naf[i];
            let p = point_naf[i];
            if g != 0 {
                out = out.double_add_mixed(naf_entry(generator_table, g));
                out = add_naf_digit(out, &point_table, p);
            } else if p != 0 {
                out = out.double_add_mixed(naf_entry(&point_table, p));
            } else {
                out = out.double();
            }
        }

        out
    }

    /// Computes `sum(scalars[i] * points[i])` using variable-time table
    /// lookups.
    ///
    /// Returns `None` when `points` and `scalars` have different lengths or a
    /// scalar is greater than or equal to the P-256 group order. All scalars
    /// are validated before group computation begins.
    /// This routine is intended for public inputs, such as syscall MSM
    /// plumbing; do not use it with secret scalars.
    #[inline]
    pub fn multi_scalar_mul_vartime(points: &[AffinePoint], scalars: &[[u8; 32]]) -> Option<Self> {
        if points.len() != scalars.len() {
            return None;
        }

        if !scalars
            .iter()
            .all(|scalar| Scalar::is_canonical(scalar, Endianness::Big))
        {
            return None;
        }

        Self::multi_scalar_mul_vartime_unchecked(points, scalars)
    }

    /// Computes `sum(scalars[i] * points[i])` using variable-time table
    /// lookups and arbitrary big-endian scalar bytes.
    ///
    /// Returns `None` when `points` and `scalars` have different lengths.
    /// Scalars greater than or equal to the P-256 group order are accepted;
    /// group arithmetic implicitly reduces them modulo the group order. This
    /// routine is intended for public inputs only.
    #[inline]
    pub fn multi_scalar_mul_vartime_unchecked(
        points: &[AffinePoint],
        scalars: &[[u8; 32]],
    ) -> Option<Self> {
        if points.len() != scalars.len() {
            return None;
        }

        // Odd multiples P, 3P, ..., 15P per point for width-5 signed NAFs.
        // All tables share a single field inversion; P itself is already
        // affine and is not normalized again.
        let mut projective = Vec::with_capacity(points.len() * 7);
        for point in points {
            projective.extend_from_slice(&odd_multiples::<8>(Self::from_affine(*point))[1..]);
        }
        let normalized = batch_normalize_slice(&projective);
        let mut affine = Vec::with_capacity(points.len() * 8);
        for (point, multiples) in points.iter().zip(normalized.chunks_exact(7)) {
            affine.push(*point);
            affine.extend_from_slice(multiples);
        }
        let nafs: Vec<_> = scalars.iter().map(wnaf5).collect();
        let mut out = Self::IDENTITY;

        for i in (0..261).rev() {
            out = out.double();
            for (table, naf) in affine.chunks_exact(8).zip(&nafs) {
                let digit = naf[i];
                if digit > 0 {
                    out = out.add_mixed(table[(digit >> 1) as usize]);
                } else if digit < 0 {
                    out = out.add_mixed(-table[((-digit) >> 1) as usize]);
                }
            }
        }

        Some(out)
    }
}

impl Add for ProjectivePoint {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if self.is_identity() {
            return rhs;
        }
        if rhs.is_identity() {
            return self;
        }

        let z1z1 = self.z.square();
        let z2z2 = rhs.z.square();
        let u1 = self.x * z2z2;
        let u2 = rhs.x * z1z1;
        let s1 = self.y * rhs.z * z2z2;
        let s2 = rhs.y * self.z * z1z1;

        if u1 == u2 {
            return if s1 == s2 {
                self.double()
            } else {
                Self::IDENTITY
            };
        }

        let h = u2 - u1;
        let i = double(h).square();
        let j = h * i;
        let slope_num = double(s2 - s1);
        let v = u1 * i;
        let x = slope_num.square() - j - double(v);
        let y = slope_num * (v - x) - double(s1 * j);
        let z = double(self.z * rhs.z) * h;

        Self { x, y, z }
    }
}

impl Sub for ProjectivePoint {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Neg for ProjectivePoint {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            x: self.x,
            y: -self.y,
            z: self.z,
        }
    }
}

#[inline]
fn double(x: FieldElement) -> FieldElement {
    x + x
}

/// Odd multiples P, 3P, ..., 15P in Jacobian coordinates.
///
/// The addition formulas do not depend on the curve coefficient `a`, so the
/// additions of 2P are carried out on the isomorphic curve where 2P has
/// `Z = 1`, using mixed additions. Results are mapped back by multiplying
/// their `Z` by the `Z` of 2P.
#[inline]
fn odd_multiples<const N: usize>(base: ProjectivePoint) -> [ProjectivePoint; N] {
    let base2 = base.double();
    let mut table = [base; N];

    if base2.is_identity() {
        return odd_multiples_mixed(base, base2);
    }

    // Co-Z additions: P rescaled to the Z of 2P = (2Y * Z), and every sum
    // also yields 2P rescaled to the Z of that sum.
    let t = double(base.y);
    let tt = t.square();
    let mut step = base2;
    let mut entry = ProjectivePoint {
        x: base.x * tt,
        y: base.y * tt * t,
        z: base2.z,
    };

    for item in table.iter_mut().skip(1) {
        let h = entry.x - step.x;
        if h.is_zero() {
            return odd_multiples_mixed(base, base2);
        }
        let a = h.square();
        let b = step.x * a;
        let c = entry.x * a;
        let dy = entry.y - step.y;
        let e = step.y * (c - b);
        let x = dy.square() - b - c;
        let y = dy * (b - x) - e;
        let z = step.z * h;

        step = ProjectivePoint { x: b, y: e, z };
        entry = ProjectivePoint { x, y, z };
        *item = entry;
    }

    table
}

/// Odd multiples P, 3P, ..., 15P using mixed additions of 2P.
#[inline]
fn odd_multiples_mixed<const N: usize>(
    base: ProjectivePoint,
    base2: ProjectivePoint,
) -> [ProjectivePoint; N] {
    let mut table = [base; N];

    if base2.is_identity() {
        for i in 1..N {
            table[i] = table[i - 1] + base2;
        }
        return table;
    }

    let zz = base2.z.square();
    let step = AffinePoint {
        x: base2.x,
        y: base2.y,
        infinity: false,
    };
    let mut entry = ProjectivePoint {
        x: base.x * zz,
        y: base.y * (zz * base2.z),
        z: base.z,
    };

    for item in table.iter_mut().skip(1) {
        entry = entry.add_mixed(step);
        *item = ProjectivePoint {
            x: entry.x,
            y: entry.y,
            z: entry.z * base2.z,
        };
    }

    table
}

/// Width-5 non-adjacent form of a big-endian 256-bit integer, least
/// significant digit first. Every digit is zero or odd in `-15..=15`.
fn wnaf5(scalar: &[u8; 32]) -> [i8; 261] {
    wnaf::<5>(scalar)
}

/// Width-`W` non-adjacent form of a big-endian 256-bit integer, least
/// significant digit first. Every digit is zero or odd with absolute value
/// below `2^(W - 1)`. Only digits up to index 256 can be nonzero.
fn wnaf<const W: usize>(scalar: &[u8; 32]) -> [i8; 261] {
    let mut limbs = [0u64; 5];
    for (i, limb) in limbs.iter_mut().take(4).enumerate() {
        let start = 32 - 8 * (i + 1);
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&scalar[start..start + 8]);
        *limb = u64::from_be_bytes(bytes);
    }

    let mut naf = [0i8; 261];
    let mut pos = 0;
    let mut carry = 0u64;

    while pos < 256 {
        let idx = pos / 64;
        let bit = pos % 64;
        let bits = if bit + W < 64 {
            limbs[idx] >> bit
        } else {
            (limbs[idx] >> bit) | (limbs[idx + 1] << (64 - bit))
        };
        let window = carry + (bits & ((1 << W) - 1));

        if window & 1 == 0 {
            pos += 1;
            continue;
        }

        if window < 1 << (W - 1) {
            carry = 0;
            naf[pos] = window as i8;
        } else {
            carry = 1;
            naf[pos] = (window as i16 - (1 << W)) as i8;
        }
        pos += W;
    }
    // A final carry implies the scalar is at least 2^(pos - 1), so pos <= 256.
    if carry != 0 {
        naf[pos] = 1;
    }

    naf
}

#[inline]
fn triple(x: FieldElement) -> FieldElement {
    x + x + x
}

fn generator_window8_table() -> &'static [[AffinePoint; BASE_WINDOW_POINTS]; BASE_WINDOWS] {
    static TABLE: OnceLock<Box<[[AffinePoint; BASE_WINDOW_POINTS]; BASE_WINDOWS]>> =
        OnceLock::new();

    TABLE
        .get_or_init(|| build_window8_table(ProjectivePoint::GENERATOR))
        .as_ref()
}

/// Odd multiples G, 3G, ..., 127G for width-8 signed NAFs.
fn generator_odd_multiples_table() -> &'static [AffinePoint; 64] {
    static TABLE: OnceLock<[AffinePoint; 64]> = OnceLock::new();

    TABLE.get_or_init(|| batch_normalize(odd_multiples(ProjectivePoint::GENERATOR)))
}

/// Returns the odd multiple selected by a nonzero signed NAF digit.
#[inline]
fn naf_entry<const N: usize>(table: &[AffinePoint; N], digit: i8) -> AffinePoint {
    if digit > 0 {
        table[(digit >> 1) as usize]
    } else {
        -table[((-digit) >> 1) as usize]
    }
}

/// Adds the odd multiple selected by a signed NAF digit, if any.
#[inline]
fn add_naf_digit<const N: usize>(
    point: ProjectivePoint,
    table: &[AffinePoint; N],
    digit: i8,
) -> ProjectivePoint {
    if digit > 0 {
        point.add_mixed(table[(digit >> 1) as usize])
    } else if digit < 0 {
        point.add_mixed(-table[((-digit) >> 1) as usize])
    } else {
        point
    }
}

fn batch_normalize_slice(points: &[ProjectivePoint]) -> Vec<AffinePoint> {
    let mut products = vec![FieldElement::ONE; points.len()];
    let mut acc = FieldElement::ONE;

    for (product, point) in products.iter_mut().zip(points) {
        *product = acc;
        if !point.is_identity() {
            acc = acc * point.z;
        }
    }

    let mut out = vec![AffinePoint::IDENTITY; points.len()];
    let Some(mut acc_inverse) = acc.invert() else {
        return out;
    };

    for i in (0..points.len()).rev() {
        let point = points[i];
        if point.is_identity() {
            continue;
        }

        let z_inverse = acc_inverse * products[i];
        acc_inverse = acc_inverse * point.z;
        let z_inverse2 = z_inverse.square();
        out[i] = AffinePoint {
            x: point.x * z_inverse2,
            y: point.y * z_inverse2 * z_inverse,
            infinity: false,
        };
    }

    out
}

fn batch_normalize<const N: usize>(points: [ProjectivePoint; N]) -> [AffinePoint; N] {
    let mut products = [FieldElement::ONE; N];
    let mut acc = FieldElement::ONE;

    for (i, point) in points.iter().enumerate() {
        products[i] = acc;
        if !point.is_identity() {
            acc = acc * point.z;
        }
    }

    let Some(mut acc_inverse) = acc.invert() else {
        return [AffinePoint::IDENTITY; N];
    };
    let mut out = [AffinePoint::IDENTITY; N];

    for i in (0..N).rev() {
        let point = points[i];
        if point.is_identity() {
            continue;
        }

        let z_inverse = acc_inverse * products[i];
        acc_inverse = acc_inverse * point.z;
        let z_inverse2 = z_inverse.square();
        out[i] = AffinePoint {
            x: point.x * z_inverse2,
            y: point.y * z_inverse2 * z_inverse,
            infinity: false,
        };
    }

    out
}


fn build_window8_table(
    mut base: ProjectivePoint,
) -> Box<[[AffinePoint; BASE_WINDOW_POINTS]; BASE_WINDOWS]> {
    let mut rows = Vec::with_capacity(BASE_WINDOWS);

    for _ in 0..BASE_WINDOWS {
        rows.push(projective_window8_table(base));

        for _ in 0..8 {
            base = base.double();
        }
    }

    rows.into_boxed_slice()
        .try_into()
        .expect("fixed-point table has the expected number of rows")
}

fn projective_window8_table(base: ProjectivePoint) -> [AffinePoint; BASE_WINDOW_POINTS] {
    let mut projective = [ProjectivePoint::IDENTITY; BASE_WINDOW_POINTS];
    let mut multiple = ProjectivePoint::IDENTITY;

    for entry in projective.iter_mut().skip(1) {
        multiple = multiple + base;
        *entry = multiple;
    }

    batch_normalize(projective)
}

#[inline]
fn mul_window8_vartime(
    table: &[[AffinePoint; BASE_WINDOW_POINTS]; BASE_WINDOWS],
    scalar: &[u8; 32],
) -> ProjectivePoint {
    let mut out = ProjectivePoint::IDENTITY;

    for (window, byte) in scalar.iter().rev().enumerate() {
        out = out.add_mixed(table[window][*byte as usize]);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{AffinePoint, ProjectivePoint};
    use crate::{Endianness, field::FieldElement};
    use p256::{
        ProjectivePoint as P256ProjectivePoint, Scalar,
        elliptic_curve::{ff::PrimeField, group::Group, sec1::ToEncodedPoint},
    };

    const SCALAR: [u8; 32] = [
        0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
        0x77, 0x88,
    ];
    const SMALL_SCALAR: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5,
    ];
    const GROUP_ORDER: [u8; 32] = [
        0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xbc, 0xe6, 0xfa, 0xad, 0xa7, 0x17, 0x9e, 0x84, 0xf3, 0xb9, 0xca, 0xc2, 0xfc, 0x63,
        0x25, 0x51,
    ];

    fn assert_matches_p256(rust: ProjectivePoint, p256: P256ProjectivePoint) {
        let expected_identity = bool::from(p256.is_identity());
        assert_eq!(rust.is_identity(), expected_identity, "identity mismatch");

        let rust_bytes = rust.to_uncompressed(Endianness::Big);

        if expected_identity {
            assert_eq!(rust_bytes, [0u8; 64]);
        } else {
            let p256_bytes = p256.to_affine().to_encoded_point(false);
            assert_eq!(rust_bytes.as_slice(), &p256_bytes.as_bytes()[1..]);
        }
    }

    #[test]
    fn generator_matches_p256() {
        assert_matches_p256(
            ProjectivePoint::generator(),
            P256ProjectivePoint::generator(),
        );
    }

    #[test]
    fn parses_and_serializes_generator() {
        let bytes = ProjectivePoint::generator().to_uncompressed(Endianness::Big);
        assert_eq!(
            AffinePoint::from_uncompressed(&bytes, Endianness::Big).unwrap(),
            AffinePoint::generator()
        );
    }

    #[test]
    fn projective_equality_ignores_jacobian_scale() {
        let point = ProjectivePoint::generator().double();
        let normalized = ProjectivePoint::from_affine(point.to_affine());
        assert_eq!(point, normalized);

        let z = FieldElement::from_u64(7);
        let z2 = z.square();
        let scaled = ProjectivePoint {
            x: point.x * z2,
            y: point.y * z2 * z,
            z: point.z * z,
        };

        assert_eq!(point, scaled);
        assert_ne!(point, -point);
    }

    #[test]
    fn double_matches_p256() {
        let p256 = P256ProjectivePoint::generator().double();
        assert_matches_p256(ProjectivePoint::generator().double(), p256);
    }

    #[test]
    fn add_matches_p256() {
        let rust_g = ProjectivePoint::generator();
        let rust_2g = rust_g.double();
        let p256_g = P256ProjectivePoint::generator();
        let p256_2g = p256_g.double();

        assert_matches_p256(rust_2g + rust_g, p256_2g + p256_g);
    }

    #[test]
    fn mixed_add_matches_p256() {
        let rust_g = ProjectivePoint::generator();
        let rust_2g = rust_g.double();
        let p256_g = P256ProjectivePoint::generator();
        let p256_2g = p256_g.double();

        assert_matches_p256(
            rust_2g.add_mixed(AffinePoint::generator()),
            p256_2g + p256_g,
        );
    }

    #[test]
    fn scalar_mul_matches_p256() {
        let scalar = Option::<Scalar>::from(Scalar::from_repr(SCALAR.into())).unwrap();
        assert_matches_p256(
            ProjectivePoint::generator()
                .mul_scalar_vartime(SCALAR)
                .unwrap(),
            P256ProjectivePoint::generator() * scalar,
        );
    }

    #[test]
    fn fixed_base_scalar_mul_matches_p256() {
        let scalar = Option::<Scalar>::from(Scalar::from_repr(SCALAR.into())).unwrap();
        assert_matches_p256(
            ProjectivePoint::fixed_base_scalar_mul_vartime(SCALAR).unwrap(),
            P256ProjectivePoint::generator() * scalar,
        );
    }

    #[test]
    fn double_scalar_mul_matches_p256() {
        let scalar = Option::<Scalar>::from(Scalar::from_repr(SCALAR.into())).unwrap();
        let point = ProjectivePoint::generator().double().to_affine();
        let p256_point = P256ProjectivePoint::generator().double();

        assert_matches_p256(
            ProjectivePoint::double_scalar_mul_vartime(SCALAR, point, SCALAR).unwrap(),
            (P256ProjectivePoint::generator() * scalar) + (p256_point * scalar),
        );
    }

    #[test]
    fn multi_scalar_mul_matches_p256() {
        let scalar = Option::<Scalar>::from(Scalar::from_repr(SCALAR.into())).unwrap();
        let small_scalar = Option::<Scalar>::from(Scalar::from_repr(SMALL_SCALAR.into())).unwrap();
        let point = ProjectivePoint::generator().double().to_affine();
        let p256_point = P256ProjectivePoint::generator().double();

        assert_matches_p256(
            ProjectivePoint::multi_scalar_mul_vartime(
                &[AffinePoint::generator(), point],
                &[SCALAR, SMALL_SCALAR],
            )
            .unwrap(),
            (P256ProjectivePoint::generator() * scalar) + (p256_point * small_scalar),
        );
    }

    #[test]
    fn multi_scalar_mul_rejects_length_mismatch() {
        assert!(
            ProjectivePoint::multi_scalar_mul_vartime(&[AffinePoint::generator()], &[]).is_none()
        );
    }

    #[test]
    fn scalar_multiplication_rejects_non_canonical_scalars() {
        let generator = ProjectivePoint::generator();
        let affine_generator = AffinePoint::generator();

        assert!(generator.mul_scalar_vartime(GROUP_ORDER).is_none());
        assert!(ProjectivePoint::fixed_base_scalar_mul_vartime(GROUP_ORDER).is_none());
        assert!(
            ProjectivePoint::double_scalar_mul_vartime(GROUP_ORDER, affine_generator, SCALAR)
                .is_none()
        );
        assert!(
            ProjectivePoint::double_scalar_mul_vartime(SCALAR, affine_generator, GROUP_ORDER)
                .is_none()
        );
        assert!(
            ProjectivePoint::multi_scalar_mul_vartime(&[affine_generator], &[GROUP_ORDER])
                .is_none()
        );
    }

    #[test]
    fn unchecked_scalar_multiplication_accepts_non_canonical_scalars() {
        let generator = ProjectivePoint::generator();
        let affine_generator = AffinePoint::generator();

        assert_eq!(
            generator.mul_scalar_vartime_unchecked(GROUP_ORDER),
            ProjectivePoint::identity()
        );
        assert_eq!(
            ProjectivePoint::fixed_base_scalar_mul_vartime_unchecked(GROUP_ORDER),
            ProjectivePoint::identity()
        );
        assert_eq!(
            ProjectivePoint::double_scalar_mul_vartime_unchecked(
                GROUP_ORDER,
                affine_generator,
                GROUP_ORDER,
            ),
            ProjectivePoint::identity()
        );
        assert_eq!(
            ProjectivePoint::multi_scalar_mul_vartime_unchecked(
                &[affine_generator],
                &[GROUP_ORDER],
            )
            .unwrap(),
            ProjectivePoint::identity()
        );
    }

    #[test]
    fn parses_and_serializes_little_endian() {
        let point = AffinePoint::generator();

        let be_bytes = point.to_uncompressed(Endianness::Big);
        let le_bytes = point.to_uncompressed(Endianness::Little);

        assert_ne!(be_bytes, le_bytes);
        assert_eq!(
            AffinePoint::from_uncompressed(&le_bytes, Endianness::Little).unwrap(),
            point
        );
    }

    #[test]
    fn parses_and_serializes_compressed() {
        let point = AffinePoint::generator();

        let be_compressed = point.to_compressed(Endianness::Big).unwrap();
        let le_compressed = point.to_compressed(Endianness::Little).unwrap();

        assert_eq!(
            AffinePoint::from_compressed(&be_compressed, Endianness::Big).unwrap(),
            point
        );
        assert_eq!(
            AffinePoint::from_compressed(&le_compressed, Endianness::Little).unwrap(),
            point
        );
    }

    #[test]
    fn rejects_invalid_compressed_prefixes() {
        let mut bytes = [0u8; 33];

        // Test the SEC1 identity marker (0x00)
        assert!(AffinePoint::from_compressed(&bytes, Endianness::Big).is_none());

        // Test the unassigned marker (0x01)
        bytes[0] = 0x01;
        assert!(AffinePoint::from_compressed(&bytes, Endianness::Big).is_none());

        // Test the uncompressed marker (0x04)
        bytes[0] = 0x04;
        assert!(AffinePoint::from_compressed(&bytes, Endianness::Big).is_none());
    }

    #[test]
    fn enforces_identity_point_rules() {
        let identity = AffinePoint::IDENTITY;

        // Uncompressed correctly handles the 64-byte zero array
        assert_eq!(identity.to_uncompressed(Endianness::Big), [0u8; 64]);
        assert_eq!(
            AffinePoint::from_uncompressed(&[0u8; 64], Endianness::Big).unwrap(),
            identity
        );

        // Compressed correctly refuses to serialize the identity point
        assert!(identity.to_compressed(Endianness::Big).is_none());
    }

    #[test]
    fn identity_and_inverse_operations_match_p256() {
        let g = ProjectivePoint::GENERATOR;
        let identity = ProjectivePoint::IDENTITY;
        let reference_g = P256ProjectivePoint::generator();
        let reference_identity = P256ProjectivePoint::IDENTITY;

        for (rust, reference) in [
            (identity, reference_identity),
            (identity.double(), reference_identity.double()),
            (-identity, -reference_identity),
            (g + identity, reference_g + reference_identity),
            (identity + g, reference_identity + reference_g),
            (g - g, reference_g - reference_g),
            (g + (-g), reference_g + (-reference_g)),
            (g.add_mixed(AffinePoint::IDENTITY), reference_g),
            (identity.add_mixed(AffinePoint::GENERATOR), reference_g),
            (g.add_mixed(-AffinePoint::GENERATOR), reference_identity),
        ] {
            assert_matches_p256(rust, reference);
        }
    }

    #[test]
    fn zero_scalars_and_cancelling_msm_return_identity() {
        let zero = [0u8; 32];
        let g = AffinePoint::GENERATOR;

        let results = [
            ProjectivePoint::GENERATOR.mul_scalar_vartime(zero).unwrap(),
            ProjectivePoint::IDENTITY
                .mul_scalar_vartime(SCALAR)
                .unwrap(),
            ProjectivePoint::fixed_base_scalar_mul_vartime(zero).unwrap(),
            ProjectivePoint::double_scalar_mul_vartime(zero, g, zero).unwrap(),
            ProjectivePoint::multi_scalar_mul_vartime(&[], &[]).unwrap(),
            ProjectivePoint::multi_scalar_mul_vartime(&[g], &[zero]).unwrap(),
            ProjectivePoint::multi_scalar_mul_vartime(
                &[g, -g, AffinePoint::IDENTITY],
                &[SCALAR; 3],
            )
            .unwrap(),
        ];

        for result in results {
            assert_matches_p256(result, P256ProjectivePoint::IDENTITY);
        }
    }

    #[test]
    fn identity_points_do_not_bypass_scalar_validation() {
        let mut n_plus_one = GROUP_ORDER;
        n_plus_one[31] += 1;

        for invalid in [GROUP_ORDER, n_plus_one, [0xffu8; 32]] {
            assert!(
                ProjectivePoint::IDENTITY
                    .mul_scalar_vartime(invalid)
                    .is_none()
            );

            assert!(
                ProjectivePoint::double_scalar_mul_vartime(
                    [0u8; 32],
                    AffinePoint::IDENTITY,
                    invalid,
                )
                .is_none()
            );

            assert!(
                ProjectivePoint::double_scalar_mul_vartime(
                    invalid,
                    AffinePoint::IDENTITY,
                    [0u8; 32],
                )
                .is_none()
            );

            let points = [AffinePoint::IDENTITY; 3];

            for index in 0..points.len() {
                let mut scalars = [[0u8; 32]; 3];
                scalars[index] = invalid;

                assert!(
                    ProjectivePoint::multi_scalar_mul_vartime(&points, &scalars).is_none(),
                    "accepted noncanonical scalar at index {index}"
                );
            }
        }
    }
}
