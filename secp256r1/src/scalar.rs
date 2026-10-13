//! P-256 scalar field arithmetic.
//!
//! [`Scalar`] represents an element of GF(n) where
//! `n = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551`
//! is the P-256 group order. Internally scalars are stored in Montgomery
//! form. Use [`from_bytes`][Scalar::from_bytes] to parse a canonical scalar
//! in either byte order, and [`to_be_bytes`][Scalar::to_be_bytes] to serialize
//! it in canonical big-endian form.

use crate::Endianness;
use core::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Scalar {
    limbs: [u64; 4],
}

const MODULUS: [u64; 4] = [
    0xf3b9_cac2_fc63_2551,
    0xbce6_faad_a717_9e84,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_0000_0000,
];

const MODULUS_INV: u64 = 0xccd1_c8aa_ee00_bc4f;

const R2: [u64; 4] = [
    0x8324_4c95_be79_eea2,
    0x4699_799c_49bd_6fa6,
    0x2845_b239_2b6b_ec59,
    0x66e1_2d94_f3d9_5620,
];


impl Scalar {
    pub const ZERO: Self = Self { limbs: [0; 4] };
    pub const ONE: Self = Self {
        limbs: [
            0x0c46_353d_039c_daaf,
            0x4319_0552_58e8_617b,
            0x0000_0000_0000_0000,
            0x0000_0000_ffff_ffff,
        ],
    };

    /// Parses a canonical scalar in the requested byte order.
    ///
    /// Returns `None` if the integer is greater than or equal to the group
    /// order. Zero is accepted. This method does not reduce the input.
    #[inline]
    pub fn from_bytes(bytes: &[u8; 32], endianness: Endianness) -> Option<Self> {
        Self::from_canonical_limbs(limbs_from_bytes(bytes, endianness))
    }

    /// Reduces a 32-byte integer modulo the P-256 group order.
    ///
    /// Accepts every input value in the requested byte order. Use
    /// [`Self::from_bytes`] when noncanonical inputs must be rejected.
    #[inline]
    pub fn from_bytes_reduced(bytes: &[u8; 32], endianness: Endianness) -> Self {
        let mut limbs = limbs_from_bytes(bytes, endianness);

        if ge_limbs(limbs, MODULUS) {
            limbs = sub_limbs(limbs, MODULUS).0;
        }

        Self {
            limbs: montgomery_mul(limbs, R2),
        }
    }

    /// Returns `true` when `bytes` encodes an integer smaller than the P-256
    /// group order.
    ///
    /// This is the range check that [`from_bytes`][Scalar::from_bytes]
    /// performs, without the conversion into Montgomery form, for callers that
    /// only need to validate an encoding they intend to keep in byte form.
    #[inline]
    pub fn is_canonical(bytes: &[u8; 32], endianness: Endianness) -> bool {
        !ge_limbs(limbs_from_bytes(bytes, endianness), MODULUS)
    }

    /// Serializes this scalar as a canonical 32-byte big-endian integer.
    #[inline]
    pub fn to_be_bytes(self) -> [u8; 32] {
        be_bytes_from_limbs(from_montgomery(self.limbs))
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    #[inline]
    pub fn square(self) -> Self {
        Self {
            limbs: montgomery_square(self.limbs),
        }
    }

    #[inline]
    pub fn invert(self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        Some(Self {
            limbs: invert_canonical(self.limbs),
        })
    }

    #[inline]
    fn from_canonical_limbs(limbs: [u64; 4]) -> Option<Self> {
        if ge_limbs(limbs, MODULUS) {
            None
        } else {
            Some(Self {
                limbs: montgomery_mul(limbs, R2),
            })
        }
    }
}

impl Add for Scalar {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (sum, carry) = add_limbs(self.limbs, rhs.limbs);
        Self {
            limbs: reduce_sum(sum, carry),
        }
    }
}

impl Sub for Scalar {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let (difference, borrow) = sub_limbs(self.limbs, rhs.limbs);
        let (corrected, _) = add_limbs(difference, MODULUS);

        Self {
            limbs: if borrow == 0 { difference } else { corrected },
        }
    }
}

impl Mul for Scalar {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            limbs: montgomery_mul(self.limbs, rhs.limbs),
        }
    }
}

impl Neg for Scalar {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::ZERO - self
    }
}

const M62: i64 = (1 << 62) - 1;

/// Signed 62-bit limb representation: the value is `sum(l[i] * 2^(62 i))`.
/// Limbs 0..4 are kept in `[0, 2^62)` and the top limb carries the sign.
type Signed62 = [i64; 5];

const MODULUS_S62: Signed62 = to_signed62(MODULUS);

/// `-n^-1 mod 2^62`.
const MODULUS_NEG_INV62: i64 = (MODULUS_INV as i64) & M62;

const fn to_signed62(a: [u64; 4]) -> Signed62 {
    [
        (a[0] as i64) & M62,
        (((a[0] >> 62) | (a[1] << 2)) as i64) & M62,
        (((a[1] >> 60) | (a[2] << 4)) as i64) & M62,
        (((a[2] >> 58) | (a[3] << 6)) as i64) & M62,
        (a[3] >> 56) as i64,
    ]
}

#[inline(always)]
fn from_signed62(a: Signed62) -> [u64; 4] {
    [
        (a[0] as u64) | ((a[1] as u64) << 62),
        ((a[1] as u64) >> 2) | ((a[2] as u64) << 60),
        ((a[2] as u64) >> 4) | ((a[3] as u64) << 58),
        ((a[3] as u64) >> 6) | ((a[4] as u64) << 56),
    ]
}

/// Returns `a + sign * b` with limbs 0..4 normalized, `sign` in {-1, 1}.
#[inline(always)]
fn add_signed62(a: Signed62, b: Signed62, sign: i64) -> Signed62 {
    let mut out = [0; 5];
    let mut carry = 0i64;
    for i in 0..4 {
        carry += a[i] + sign * b[i];
        out[i] = carry & M62;
        carry >>= 62;
    }
    out[4] = a[4] + sign * b[4] + carry;
    out
}

/// Runs 62 variable-time divsteps on the low bits of `f` and `g` (with
/// `eta = -delta`), returning the new eta and the transition matrix
/// `[u, v, q, r]` scaled by `2^62`. Several steps are combined at once by
/// choosing a multiple `w` of `f` that clears the low bits of `g`.
#[inline(always)]
fn divsteps_62(mut eta: i64, mut f: u64, mut g: u64) -> (i64, [i64; 4]) {
    let (mut u, mut v, mut q, mut r) = (1u64, 0u64, 0u64, 1u64);
    let mut remaining = 62u32;
    loop {
        // A run of even steps only decrements eta and shifts.
        let zeros = (g | (u64::MAX << remaining)).trailing_zeros();
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= zeros as i64;
        remaining -= zeros;
        if remaining == 0 {
            break;
        }
        let w;
        if eta < 0 {
            eta = -eta;
            (f, g) = (g, f.wrapping_neg());
            (u, q) = (q, u.wrapping_neg());
            (v, r) = (r, v.wrapping_neg());
            let limit = (eta + 1).min(remaining as i64) as u32;
            let m = (u64::MAX >> (64 - limit)) & 63;
            // w = -g / f mod 2^6.
            w = f
                .wrapping_mul(g)
                .wrapping_mul(f.wrapping_mul(f).wrapping_sub(2))
                & m;
        } else {
            let limit = (eta + 1).min(remaining as i64) as u32;
            let m = (u64::MAX >> (64 - limit)) & 15;
            // w = -g / f mod 2^4.
            let inv = f.wrapping_add((f.wrapping_add(1) & 4) << 1);
            w = inv.wrapping_neg().wrapping_mul(g) & m;
        }
        g = g.wrapping_add(f.wrapping_mul(w));
        q = q.wrapping_add(u.wrapping_mul(w));
        r = r.wrapping_add(v.wrapping_mul(w));
    }
    (eta, [u as i64, v as i64, q as i64, r as i64])
}

/// Applies the matrix to the low `len` limbs of `f` and `g` (limb `len - 1`
/// carries the sign), dividing exactly by `2^62`.
#[inline(always)]
fn update_fg(f: &mut Signed62, g: &mut Signed62, len: usize, t: [i64; 4]) {
    let [u, v, q, r] = t.map(i128::from);
    let mut cf = u * f[0] as i128 + v * g[0] as i128;
    let mut cg = q * f[0] as i128 + r * g[0] as i128;
    cf >>= 62;
    cg >>= 62;
    for i in 1..len {
        cf += u * f[i] as i128 + v * g[i] as i128;
        cg += q * f[i] as i128 + r * g[i] as i128;
        f[i - 1] = (cf as i64) & M62;
        g[i - 1] = (cg as i64) & M62;
        cf >>= 62;
        cg >>= 62;
    }
    f[len - 1] = cf as i64;
    g[len - 1] = cg as i64;
}

/// Applies the matrix to `d` and `e` and divides by `2^62` modulo n.
#[inline(always)]
fn update_de(d: &mut Signed62, e: &mut Signed62, t: [i64; 4]) {
    let [u, v, q, r] = t.map(i128::from);
    let mut cd = u * d[0] as i128 + v * e[0] as i128;
    let mut ce = q * d[0] as i128 + r * e[0] as i128;
    let md = ((cd as i64).wrapping_mul(MODULUS_NEG_INV62) & M62) as i128;
    let me = ((ce as i64).wrapping_mul(MODULUS_NEG_INV62) & M62) as i128;
    cd += md * MODULUS_S62[0] as i128;
    ce += me * MODULUS_S62[0] as i128;
    cd >>= 62;
    ce >>= 62;
    for i in 1..5 {
        let n = MODULUS_S62[i] as i128;
        cd += u * d[i] as i128 + v * e[i] as i128 + md * n;
        ce += q * d[i] as i128 + r * e[i] as i128 + me * n;
        d[i - 1] = (cd as i64) & M62;
        e[i - 1] = (ce as i64) & M62;
        cd >>= 62;
        ce >>= 62;
    }
    d[4] = cd as i64;
    e[4] = ce as i64;
}

/// Returns `a^-1 * 2^512 mod n` for a canonical nonzero `a`, using
/// variable-time Bernstein-Yang divsteps. For Montgomery limbs `a = xR` this
/// is `x^-1 R`.
#[inline]
fn invert_canonical(a: [u64; 4]) -> [u64; 4] {
    // Invariants: f = d * a and g = e * a (mod n), scaled by R2.
    let mut f = MODULUS_S62;
    let mut g = to_signed62(a);
    let mut d = [0i64; 5];
    let mut e = to_signed62(R2);
    let mut eta = -1i64;
    let mut len = 5usize;

    loop {
        let (next_eta, t) = divsteps_62(eta, f[0] as u64, g[0] as u64);
        eta = next_eta;
        update_de(&mut d, &mut e, t);
        update_fg(&mut f, &mut g, len, t);
        if g[..len].iter().all(|&l| l == 0) {
            break;
        }
        // Drop the top limb once it is pure sign extension in both f and g.
        let (ft, gt) = (f[len - 1], g[len - 1]);
        if len > 1 && (ft == 0 || ft == -1) && (gt == 0 || gt == -1) {
            f[len - 2] |= ft << 62;
            g[len - 2] |= gt << 62;
            len -= 1;
        }
    }

    // f = +-1 here.
    if f[len - 1] < 0 {
        d = add_signed62([0; 5], d, -1);
    }
    while d[4] < 0 {
        d = add_signed62(d, MODULUS_S62, 1);
    }
    loop {
        let reduced = add_signed62(d, MODULUS_S62, -1);
        if reduced[4] < 0 {
            break;
        }
        d = reduced;
    }

    from_signed62(d)
}

#[inline(always)]
fn add_limbs(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], u64) {
    let (w0, carry) = adc(a[0], b[0], 0);
    let (w1, carry) = adc(a[1], b[1], carry);
    let (w2, carry) = adc(a[2], b[2], carry);
    let (w3, carry) = adc(a[3], b[3], carry);

    ([w0, w1, w2, w3], carry)
}

#[inline(always)]
fn sub_limbs(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], u64) {
    let (w0, borrow) = sbb(a[0], b[0], 0);
    let (w1, borrow) = sbb(a[1], b[1], borrow);
    let (w2, borrow) = sbb(a[2], b[2], borrow);
    let (w3, borrow) = sbb(a[3], b[3], borrow);

    ([w0, w1, w2, w3], borrow)
}

#[inline(always)]
fn reduce_sum(sum: [u64; 4], carry: u64) -> [u64; 4] {
    let (reduced, borrow) = sub_limbs(sum, MODULUS);

    if carry != 0 || borrow == 0 {
        reduced
    } else {
        sum
    }
}

#[inline(always)]
fn ge_limbs(a: [u64; 4], b: [u64; 4]) -> bool {
    sub_limbs(a, b).1 == 0
}

#[inline(always)]
fn mul_wide(a: [u64; 4], b: [u64; 4]) -> [u64; 8] {
    let (w0, carry) = mac(0, a[0], b[0], 0);
    let (w1, carry) = mac(0, a[0], b[1], carry);
    let (w2, carry) = mac(0, a[0], b[2], carry);
    let (w3, w4) = mac(0, a[0], b[3], carry);

    let (w1, carry) = mac(w1, a[1], b[0], 0);
    let (w2, carry) = mac(w2, a[1], b[1], carry);
    let (w3, carry) = mac(w3, a[1], b[2], carry);
    let (w4, w5) = mac(w4, a[1], b[3], carry);

    let (w2, carry) = mac(w2, a[2], b[0], 0);
    let (w3, carry) = mac(w3, a[2], b[1], carry);
    let (w4, carry) = mac(w4, a[2], b[2], carry);
    let (w5, w6) = mac(w5, a[2], b[3], carry);

    let (w3, carry) = mac(w3, a[3], b[0], 0);
    let (w4, carry) = mac(w4, a[3], b[1], carry);
    let (w5, carry) = mac(w5, a[3], b[2], carry);
    let (w6, w7) = mac(w6, a[3], b[3], carry);

    [w0, w1, w2, w3, w4, w5, w6, w7]
}

#[inline(always)]
fn montgomery_mul(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    montgomery_reduce(mul_wide(a, b))
}

#[inline(always)]
fn montgomery_square(a: [u64; 4]) -> [u64; 4] {
    let a0 = a[0] as u128;
    let a1 = a[1] as u128;
    let a2 = a[2] as u128;
    let a3 = a[3] as u128;

    let p00 = a0 * a0;
    let w0 = p00 as u64;
    let mut acc = p00 >> 64;
    let mut top = 0u64;

    let p01 = a0 * a1;
    top += (p01 >> 127) as u64;
    let (sum, overflow) = acc.overflowing_add(p01 << 1);
    let w1 = sum as u64;
    acc = sum >> 64;
    top += u64::from(overflow);

    acc |= (top as u128) << 64;
    top = 0;
    let p02 = a0 * a2;
    top += (p02 >> 127) as u64;
    let (sum, overflow) = acc.overflowing_add(p02 << 1);
    acc = sum;
    top += u64::from(overflow);
    let (sum, overflow) = acc.overflowing_add(a1 * a1);
    let w2 = sum as u64;
    acc = sum >> 64;
    top += u64::from(overflow);

    acc |= (top as u128) << 64;
    top = 0;
    let p03 = a0 * a3;
    top += (p03 >> 127) as u64;
    let (sum, overflow) = acc.overflowing_add(p03 << 1);
    acc = sum;
    top += u64::from(overflow);
    let p12 = a1 * a2;
    top += (p12 >> 127) as u64;
    let (sum, overflow) = acc.overflowing_add(p12 << 1);
    let w3 = sum as u64;
    acc = sum >> 64;
    top += u64::from(overflow);

    acc |= (top as u128) << 64;
    top = 0;
    let p13 = a1 * a3;
    top += (p13 >> 127) as u64;
    let (sum, overflow) = acc.overflowing_add(p13 << 1);
    acc = sum;
    top += u64::from(overflow);
    let (sum, overflow) = acc.overflowing_add(a2 * a2);
    let w4 = sum as u64;
    acc = sum >> 64;
    top += u64::from(overflow);

    acc |= (top as u128) << 64;
    top = 0;
    let p23 = a2 * a3;
    top += (p23 >> 127) as u64;
    let (sum, overflow) = acc.overflowing_add(p23 << 1);
    let w5 = sum as u64;
    acc = sum >> 64;
    top += u64::from(overflow);

    acc |= (top as u128) << 64;
    let (sum, overflow) = acc.overflowing_add(a3 * a3);
    let w6 = sum as u64;
    let w7 = (sum >> 64) as u64;
    debug_assert!(!overflow);

    montgomery_reduce_words(w0, w1, w2, w3, w4, w5, w6, w7)
}

#[inline(always)]
fn from_montgomery(a: [u64; 4]) -> [u64; 4] {
    montgomery_reduce([a[0], a[1], a[2], a[3], 0, 0, 0, 0])
}

#[inline(always)]
fn montgomery_reduce(input: [u64; 8]) -> [u64; 4] {
    montgomery_reduce_words(
        input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn montgomery_reduce_words(
    r0: u64,
    r1: u64,
    r2: u64,
    r3: u64,
    r4: u64,
    r5: u64,
    r6: u64,
    r7: u64,
) -> [u64; 4] {
    let k = r0.wrapping_mul(MODULUS_INV);
    let (_, carry) = mac(r0, k, MODULUS[0], 0);
    let (r1, carry) = mac(r1, k, MODULUS[1], carry);
    let (r2, carry) = mac(r2, k, MODULUS[2], carry);
    let (r3, carry) = mac(r3, k, MODULUS[3], carry);
    let (r4, carry2) = adc(r4, 0, carry);

    let k = r1.wrapping_mul(MODULUS_INV);
    let (_, carry) = mac(r1, k, MODULUS[0], 0);
    let (r2, carry) = mac(r2, k, MODULUS[1], carry);
    let (r3, carry) = mac(r3, k, MODULUS[2], carry);
    let (r4, carry) = mac(r4, k, MODULUS[3], carry);
    let (r5, carry2) = adc(r5, carry2, carry);

    let k = r2.wrapping_mul(MODULUS_INV);
    let (_, carry) = mac(r2, k, MODULUS[0], 0);
    let (r3, carry) = mac(r3, k, MODULUS[1], carry);
    let (r4, carry) = mac(r4, k, MODULUS[2], carry);
    let (r5, carry) = mac(r5, k, MODULUS[3], carry);
    let (r6, carry2) = adc(r6, carry2, carry);

    let k = r3.wrapping_mul(MODULUS_INV);
    let (_, carry) = mac(r3, k, MODULUS[0], 0);
    let (r4, carry) = mac(r4, k, MODULUS[1], carry);
    let (r5, carry) = mac(r5, k, MODULUS[2], carry);
    let (r6, carry) = mac(r6, k, MODULUS[3], carry);
    let (r7, r8) = adc(r7, carry2, carry);

    reduce_wide([r4, r5, r6, r7, r8])
}

#[inline(always)]
fn reduce_wide(value: [u64; 5]) -> [u64; 4] {
    let (w0, borrow) = sbb(value[0], MODULUS[0], 0);
    let (w1, borrow) = sbb(value[1], MODULUS[1], borrow);
    let (w2, borrow) = sbb(value[2], MODULUS[2], borrow);
    let (w3, borrow) = sbb(value[3], MODULUS[3], borrow);
    let (_, borrow) = sbb(value[4], 0, borrow);

    if borrow == 0 {
        [w0, w1, w2, w3]
    } else {
        [value[0], value[1], value[2], value[3]]
    }
}

#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let sum = (a as u128) + (b as u128) + (carry as u128);
    (sum as u64, (sum >> 64) as u64)
}

#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let difference = (a as u128).wrapping_sub((b as u128) + (borrow as u128));
    (difference as u64, u64::from((difference >> 127) != 0))
}

#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let product = (a as u128) + (b as u128) * (c as u128) + (carry as u128);
    (product as u64, (product >> 64) as u64)
}

#[inline]
fn limbs_from_bytes(bytes: &[u8; 32], endianness: Endianness) -> [u64; 4] {
    let mut be_bytes = *bytes;

    if endianness == Endianness::Little {
        be_bytes.reverse();
    }

    limbs_from_be_bytes(&be_bytes)
}

#[inline]
fn limbs_from_be_bytes(bytes: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];

    for (i, chunk) in bytes.as_chunks::<8>().0.iter().rev().enumerate() {
        limbs[i] = u64::from_be_bytes(*chunk);
    }

    limbs
}

#[inline]
fn be_bytes_from_limbs(limbs: [u64; 4]) -> [u8; 32] {
    let mut bytes = [0u8; 32];

    for (i, limb) in limbs.iter().rev().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_be_bytes());
    }

    bytes
}

#[cfg(test)]
mod tests {
    use super::Scalar;
    use crate::Endianness;
    use p256::{
        Scalar as P256Scalar,
        elliptic_curve::{bigint::U256, ff::PrimeField, ops::Reduce},
    };

    const A: [u8; 32] = [
        0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
        0x77, 0x88,
    ];
    const B: [u8; 32] = [
        0x0f, 0xed, 0xcb, 0xa9, 0x87, 0x65, 0x43, 0x21, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99,
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00, 0x80, 0x70, 0x60, 0x50, 0x40, 0x30,
        0x20, 0x10,
    ];
    const N: [u8; 32] = [
        0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xbc, 0xe6, 0xfa, 0xad, 0xa7, 0x17, 0x9e, 0x84, 0xf3, 0xb9, 0xca, 0xc2, 0xfc, 0x63,
        0x25, 0x51,
    ];

    fn p256_scalar(bytes: [u8; 32]) -> P256Scalar {
        Option::from(P256Scalar::from_repr(bytes.into())).unwrap()
    }

    fn assert_matches_p256(rust: Scalar, p256: P256Scalar) {
        let p256_bytes: [u8; 32] = p256.to_repr().into();
        assert_eq!(rust.to_be_bytes(), p256_bytes);
    }

    fn sample(mut seed: u64) -> [u8; 32] {
        let mut bytes = [0u8; 32];

        for chunk in bytes.as_chunks_mut::<8>().0 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            chunk.copy_from_slice(&seed.to_be_bytes());
        }

        bytes[0] &= 0x7f;
        bytes
    }

    #[test]
    fn rejects_non_canonical_order() {
        assert!(Scalar::from_bytes(&N, Endianness::Big).is_none());
    }

    #[test]
    fn is_canonical_agrees_with_from_bytes() {
        let mut n_minus_one = N;
        n_minus_one[31] -= 1;

        for bytes in [[0u8; 32], A, B, n_minus_one, N, [0xffu8; 32]] {
            assert_eq!(
                Scalar::is_canonical(&bytes, Endianness::Big),
                Scalar::from_bytes(&bytes, Endianness::Big).is_some(),
                "big endian {bytes:02x?}"
            );

            let mut le = bytes;
            le.reverse();
            assert_eq!(
                Scalar::is_canonical(&le, Endianness::Little),
                Scalar::from_bytes(&le, Endianness::Little).is_some(),
                "little endian {bytes:02x?}"
            );
        }

        for i in 0..128 {
            let bytes = sample(i);
            assert_eq!(
                Scalar::is_canonical(&bytes, Endianness::Big),
                Scalar::from_bytes(&bytes, Endianness::Big).is_some()
            );
        }
    }

    #[test]
    fn round_trips() {
        for bytes in [[0u8; 32], A, B] {
            assert_eq!(
                Scalar::from_bytes(&bytes, Endianness::Big)
                    .unwrap()
                    .to_be_bytes(),
                bytes
            );
        }
    }

    #[test]
    fn arithmetic_matches_p256() {
        let a = Scalar::from_bytes(&A, Endianness::Big).unwrap();
        let b = Scalar::from_bytes(&B, Endianness::Big).unwrap();
        let p256_a = p256_scalar(A);
        let p256_b = p256_scalar(B);

        assert_matches_p256(a + b, p256_a + p256_b);
        assert_matches_p256(a - b, p256_a - p256_b);
        assert_matches_p256(a * b, p256_a * p256_b);
        assert_matches_p256(a.square(), p256_a.square());
        assert_matches_p256(a.invert().unwrap(), Option::from(p256_a.invert()).unwrap());
    }

    #[test]
    fn reduced_bytes_match_p256() {
        let rust = Scalar::from_bytes_reduced(&N, Endianness::Big);
        let p256 = P256Scalar::reduce(U256::from_be_slice(&N));
        assert_matches_p256(rust, p256);
    }

    #[test]
    fn arithmetic_matches_p256_for_many_samples() {
        for i in 1..128 {
            let a = sample(i);
            let b = sample(i ^ 0xa5a5_a5a5_a5a5_a5a5);
            let rust_a = Scalar::from_bytes(&a, Endianness::Big).unwrap();
            let rust_b = Scalar::from_bytes(&b, Endianness::Big).unwrap();
            let p256_a = p256_scalar(a);
            let p256_b = p256_scalar(b);

            assert_matches_p256(rust_a + rust_b, p256_a + p256_b);
            assert_matches_p256(rust_a - rust_b, p256_a - p256_b);
            assert_matches_p256(rust_a * rust_b, p256_a * p256_b);
            assert_matches_p256(rust_a.square(), p256_a.square());
            assert_matches_p256(
                rust_a.invert().unwrap(),
                Option::from(p256_a.invert()).unwrap(),
            );
        }
    }

    #[test]
    fn parses_little_endian_scalars() {
        let mut le_a = A;
        le_a.reverse();

        let be_scalar = Scalar::from_bytes(&A, Endianness::Big).unwrap();
        let le_scalar = Scalar::from_bytes(&le_a, Endianness::Little).unwrap();

        assert_eq!(be_scalar, le_scalar);

        let mut le_n = N;
        le_n.reverse();

        let be_reduced = Scalar::from_bytes_reduced(&N, Endianness::Big);
        let le_reduced = Scalar::from_bytes_reduced(&le_n, Endianness::Little);

        assert_eq!(be_reduced, le_reduced);
    }
}
