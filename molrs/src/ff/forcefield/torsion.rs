//! Torsion algebra: the exact linear maps between every proper-torsion and
//! improper form molrs has.
//!
//! Every Class-I torsion form is a finite Fourier series in the dihedral angle
//! φ (LAMMPS's signed φ of the atoms I-J-K-L as stored, the angle every molrs
//! dihedral kernel computes):
//!
//! ```text
//! E(φ) = Σₙ₌₀ aₙ cos nφ + bₙ sin nφ           (FourierSeries)
//! ```
//!
//! Each form has an exact **embedding** into that series (`to_series`: the
//! energy is the same function of φ, constant included) and an exact
//! **projection** back (`from_series`), which either reproduces every
//! `n ≥ 1` coefficient or refuses with a [`TorsionRefusal`] naming the term
//! that prevents it. The forms, in the force-field IR (LAMMPS standard) (parameters as stored,
//! phases in degrees):
//!
//! | form | LAMMPS style | energy | image condition (`n ≥ 1` part) | constant |
//! |---|---|---|---|---|
//! | [`Periodic`] | `dihedral fourier` (molrs `periodic`) | Σₘ kₘ[1 + cos(nₘφ − γₘ)] | none: every series | implied, Σkₘ |
//! | [`Charmm`] | `dihedral charmm` | k[1 + cos(nφ − d)] | one order | implied, k |
//! | [`CosineTerm`] | molrs `improper periodic` | k[1 + cos(nφ − γ)] | one order | implied, k |
//! | [`SignedCosine`] | `dihedral harmonic`, `improper cvff` | k[1 + d cos nφ] | one order, bₙ = 0 | implied, k |
//! | [`Opls`] | `dihedral opls` | ½Σ kₙ[1 ∓ cos nφ] | bₙ = 0, n ≤ 4 | implied, ½Σkₙ |
//! | [`Class2`] | `dihedral class2` (torsion part) | Σₙ₌₁³ kₙ[1 − cos(nφ − φₙ)] | n ≤ 3 | implied, Σkₙ |
//! | [`MultiHarmonic`] | `dihedral multi/harmonic` | Σₙ₌₁⁵ Aₙ cosⁿ⁻¹φ | bₙ = 0, n ≤ 4 | free (exact) |
//! | [`NHarmonic`] | `dihedral nharmonic` | Σᵢ₌₁ᴺ Aᵢ cosⁱ⁻¹φ | bₙ = 0 | free (exact) |
//! | [`RyckaertBellemans`] | GROMACS funct 3, OpenMM `RBTorsionForce` | Σₙ₌₀⁵ Cₙ cosⁿ(φ − 180°) | bₙ = 0, n ≤ 5 | free (exact) |
//!
//! Every periodicity must be an integer ([`TorsionRefusal::NonIntegerPeriodicity`]);
//! LAMMPS requires that of every style above.
//!
//! # The constant term
//!
//! `a₀` is tracked: `to_series` is the exact energy, constant included, and
//! the polynomial forms (multi/harmonic, nharmonic, RB) carry it back exactly.
//! The other forms fix their constant by their other parameters, so their
//! `from_series` reproduces the series up to a constant offset
//! `a₀ − form.to_series().constant()`. A constant shifts no force, and
//! [`FourierSeries::canonical`] drops it: two torsions are the same physics
//! iff their canonical series are equal. In particular `ΣCₙ = 0` (an RB row
//! vanishing at φ = 180°) is **not** an image condition of the OPLS form once
//! the constant is dropped — only `C₅ = 0` is. RB is the polynomial
//! `multi/harmonic` / `nharmonic` exactly, constant included.
//!
//! # Exactness
//!
//! Phases are degrees, and multiples of 90° are evaluated exactly (`cos 180°`
//! is `−1`, `sin 180°` is `0`, not `1.2e−16`), so a 0°/180° phase lands as
//! `bₙ = 0` exactly and passes a sine-free projection. Coefficient tests are
//! exact (`!= 0.0`); a caller holding rounded input (GROMACS prints five
//! decimals) chops it first with [`FourierSeries::chopped`].
//!
//! # Outside the image
//!
//! - `improper harmonic`, `K(|φ| − χ0)²`, and LAMMPS `dihedral quadratic`,
//!   `K(φ − φ0)²`, are not finite Fourier series
//!   ([`TorsionRefusal::NotAFourierForm`]). A periodic improper and a harmonic
//!   one agree to second order about the minimum only:
//!   `K = n²·k/2` in LAMMPS's un-halved `K(χ − χ0)²` (`2k` at `n = 2`; the
//!   `k_h = n²k` of a ½-form harmonic) — see
//!   [`CosineTerm::second_order_harmonic`] and
//!   [`ImproperHarmonic::second_order_periodic`]. Their quartic terms differ
//!   by `−k n⁴ δ⁴/24`.
//! - `dihedral class2`'s cross terms (`mbt`, `ebt`, `at`, `aat`, `bb13`) couple
//!   the torsion to bonds and angles; they are outside the Class-I IR (and
//!   molrs stores none of them). [`Class2`] is the torsion part only.
//! - The out-of-plane impropers (`improper fourier` / molrs `uff_inversion`,
//!   `mmff_oop`) are functions of a Wilson angle, not of a dihedral.
//!
//! Units: every map is linear and unit-agnostic (energy in = energy out).
//! These live in `ff::forcefield`, not `ff::potential`, so readers and writers
//! never import from a kernel module; the kernels' agreement with the algebra
//! is tested here, on the registered kernels.

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign};

use super::Params;

// ── the series ──────────────────────────────────────────────────────────────

/// `E(φ) = Σₙ aₙ cos nφ + bₙ sin nφ`, `n = 0, 1, …`, φ in radians when
/// evaluated. `a[0]` is the constant term; `b[0]` multiplies `sin 0 = 0` and
/// is held at zero. Equality is exact on every coefficient, trailing zeros
/// aside.
#[derive(Debug, Clone, Default)]
pub struct FourierSeries {
    a: Vec<f64>,
    b: Vec<f64>,
}

impl FourierSeries {
    /// The zero series.
    pub fn zero() -> Self {
        Self::default()
    }

    /// The series with cosine coefficients `a` and sine coefficients `b`,
    /// both indexed from `n = 0` (the shorter is padded with zeros; `b[0]`,
    /// which multiplies `sin 0`, is dropped).
    pub fn new(a: Vec<f64>, b: Vec<f64>) -> Self {
        let mut s = Self::zero();
        for (n, &an) in a.iter().enumerate() {
            s.add_cos(n, an);
        }
        for (n, &bn) in b.iter().enumerate().skip(1) {
            s.add_sin(n, bn);
        }
        s
    }

    /// `aₙ` (zero beyond the stored length).
    pub fn a(&self, n: usize) -> f64 {
        self.a.get(n).copied().unwrap_or(0.0)
    }

    /// `bₙ` (zero beyond the stored length, and at `n = 0`).
    pub fn b(&self, n: usize) -> f64 {
        self.b.get(n).copied().unwrap_or(0.0)
    }

    /// The constant term `a₀`.
    pub fn constant(&self) -> f64 {
        self.a(0)
    }

    /// The highest `n ≥ 1` with `aₙ ≠ 0` or `bₙ ≠ 0`; 0 for a constant.
    pub fn order(&self) -> usize {
        (1..self.a.len())
            .rev()
            .find(|&n| self.a[n] != 0.0 || self.b[n] != 0.0)
            .unwrap_or(0)
    }

    /// The orders `n ≥ 1` with a non-zero coefficient, ascending.
    pub fn orders(&self) -> Vec<usize> {
        (1..=self.order())
            .filter(|&n| self.a(n) != 0.0 || self.b(n) != 0.0)
            .collect()
    }

    /// The energy at `phi` (radians).
    pub fn energy(&self, phi: f64) -> f64 {
        self.a
            .iter()
            .zip(&self.b)
            .enumerate()
            .map(|(n, (&an, &bn))| {
                let (s, c) = (n as f64 * phi).sin_cos();
                an * c + bn * s
            })
            .sum()
    }

    /// The canonical form: constant dropped, trailing zeros trimmed. Two
    /// torsions are the same physics iff their canonical series are equal.
    pub fn canonical(&self) -> Self {
        let len = self.order() + 1;
        let mut a = self.a.clone();
        let mut b = self.b.clone();
        a.resize(len, 0.0);
        b.resize(len, 0.0);
        a[0] = 0.0;
        Self { a, b }
    }

    /// Every coefficient with `|c| ≤ tol` set to exactly zero — for input
    /// rounded on print (the projections test coefficients exactly).
    pub fn chopped(&self, tol: f64) -> Self {
        let chop = |c: &f64| if c.abs() <= tol { 0.0 } else { *c };
        Self {
            a: self.a.iter().map(chop).collect(),
            b: self.b.iter().map(chop).collect(),
        }
    }

    /// `max |Δaₙ|, |Δbₙ|` over every order, the constant included.
    pub fn max_abs_diff(&self, other: &Self) -> f64 {
        let len = self.a.len().max(other.a.len());
        (0..len)
            .map(|n| {
                (self.a(n) - other.a(n))
                    .abs()
                    .max((self.b(n) - other.b(n)).abs())
            })
            .fold(0.0, f64::max)
    }

    fn grow(&mut self, n: usize) {
        if self.a.len() <= n {
            self.a.resize(n + 1, 0.0);
            self.b.resize(n + 1, 0.0);
        }
    }

    fn add_cos(&mut self, n: usize, c: f64) {
        self.grow(n);
        self.a[n] += c;
    }

    fn add_sin(&mut self, n: usize, s: f64) {
        if n > 0 {
            self.grow(n);
            self.b[n] += s;
        }
    }
}

impl PartialEq for FourierSeries {
    fn eq(&self, other: &Self) -> bool {
        let len = self.a.len().max(other.a.len());
        (0..len).all(|n| self.a(n) == other.a(n) && self.b(n) == other.b(n))
    }
}

impl AddAssign<&FourierSeries> for FourierSeries {
    fn add_assign(&mut self, rhs: &FourierSeries) {
        for n in 0..rhs.a.len() {
            self.add_cos(n, rhs.a[n]);
            self.add_sin(n, rhs.b[n]);
        }
    }
}

impl Add for FourierSeries {
    type Output = FourierSeries;
    fn add(mut self, rhs: FourierSeries) -> FourierSeries {
        self += &rhs;
        self
    }
}

/// Several rows on one quadruple are one torsion: their series summed.
impl Sum for FourierSeries {
    fn sum<I: Iterator<Item = FourierSeries>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

// ── refusals ────────────────────────────────────────────────────────────────

/// Why a conversion is not exact, naming the term that prevents it.
#[derive(Debug, Clone, PartialEq)]
pub enum TorsionRefusal {
    /// `bₙ ≠ 0` — a phase off {0°, 180°} — where the target is cosines only.
    SineTerm { n: usize, b: f64 },
    /// An order above the target's highest.
    OrderTooHigh { n: usize, max: usize },
    /// More than one order for a single-term target.
    MultiTerm { orders: Vec<usize> },
    /// A periodicity that is not an integer (or not finite).
    NonIntegerPeriodicity { periodicity: f64 },
    /// A `dihedral harmonic` / `improper cvff` sign that is not ±1.
    InvalidSign { sign: f64 },
    /// A form that is not a finite Fourier series in φ.
    NotAFourierForm {
        style: &'static str,
        reason: &'static str,
    },
    /// A second-order relation asked of a term with no curvature (k = 0 or n = 0).
    NoCurvature,
    /// A `(category, style)` that is not a torsion form of the algebra.
    UnknownStyle { category: String, style: String },
    /// A parameter the style requires is absent.
    MissingParam { style: &'static str, key: String },
}

impl fmt::Display for TorsionRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SineTerm { n, b } => write!(
                f,
                "sin({n}φ) coefficient {b} ≠ 0 (a phase other than 0° or 180°): \
                 the target form has cosines only"
            ),
            Self::OrderTooHigh { n, max } => write!(
                f,
                "order n = {n} is above the target form's highest order {max}"
            ),
            Self::MultiTerm { orders } => write!(
                f,
                "orders {orders:?} are present: the target form holds one term"
            ),
            Self::NonIntegerPeriodicity { periodicity } => {
                write!(f, "periodicity {periodicity} is not an integer")
            }
            Self::InvalidSign { sign } => write!(f, "sign {sign} is not ±1"),
            Self::NotAFourierForm { style, reason } => {
                write!(f, "{style} is not a Fourier form: {reason}")
            }
            Self::NoCurvature => write!(f, "the term has no curvature (k = 0 or n = 0)"),
            Self::UnknownStyle { category, style } => {
                write!(f, "`{category} {style}` is not a torsion form")
            }
            Self::MissingParam { style, key } => write!(f, "{style}: missing param `{key}`"),
        }
    }
}

impl std::error::Error for TorsionRefusal {}

type Result<T> = std::result::Result<T, TorsionRefusal>;

// ── helpers ─────────────────────────────────────────────────────────────────

/// `(cos d, sin d)` of `d` degrees, exact at multiples of 90°.
fn cos_sin_deg(deg: f64) -> (f64, f64) {
    let r = deg.rem_euclid(360.0);
    if r == 0.0 {
        (1.0, 0.0)
    } else if r == 90.0 {
        (0.0, 1.0)
    } else if r == 180.0 {
        (-1.0, 0.0)
    } else if r == 270.0 {
        (0.0, -1.0)
    } else {
        let (s, c) = r.to_radians().sin_cos();
        (c, s)
    }
}

/// The phase, in degrees on (−180°, 180°], of `c·cos x + s·sin x = A cos(x − phase)`;
/// exact on the axes, 0° for a zero amplitude.
fn phase_deg(s: f64, c: f64) -> f64 {
    if s == 0.0 {
        if c >= 0.0 { 0.0 } else { 180.0 }
    } else if c == 0.0 {
        if s > 0.0 { 90.0 } else { -90.0 }
    } else {
        s.atan2(c).to_degrees()
    }
}

/// `|n|` and the sign of `n` for an integral periodicity.
fn order_of(periodicity: f64) -> Result<(usize, f64)> {
    if !periodicity.is_finite() || periodicity.fract() != 0.0 {
        return Err(TorsionRefusal::NonIntegerPeriodicity { periodicity });
    }
    let sign = if periodicity < 0.0 { -1.0 } else { 1.0 };
    Ok((periodicity.abs() as usize, sign))
}

/// The series has only cosines, of order ≤ `max`.
fn require_cosines(s: &FourierSeries, max: Option<usize>) -> Result<()> {
    for n in 1..=s.order() {
        if let Some(max) = max
            && n > max
            && (s.a(n) != 0.0 || s.b(n) != 0.0)
        {
            return Err(TorsionRefusal::OrderTooHigh { n, max });
        }
        if s.b(n) != 0.0 {
            return Err(TorsionRefusal::SineTerm { n, b: s.b(n) });
        }
    }
    Ok(())
}

/// The one order `n ≥ 1` of a single-term series (`None` for a constant).
fn single_order(s: &FourierSeries) -> Result<Option<usize>> {
    let orders = s.orders();
    match orders.as_slice() {
        [] => Ok(None),
        [n] => Ok(Some(*n)),
        _ => Err(TorsionRefusal::MultiTerm { orders }),
    }
}

/// `Σₖ pₖ cosᵏφ` as a Fourier series: `cosᵏφ = 2⁻ᵏ Σⱼ C(k,j) cos((k − 2j)φ)`.
fn cos_poly_to_series(p: &[f64]) -> FourierSeries {
    let mut s = FourierSeries::zero();
    for (k, &pk) in p.iter().enumerate() {
        if pk == 0.0 {
            continue;
        }
        let scale = pk * 0.5_f64.powi(k as i32);
        let mut binom = 1.0; // C(k, j)
        for j in 0..=k {
            s.add_cos(k.abs_diff(2 * j), scale * binom);
            binom = binom * (k - j) as f64 / (j + 1) as f64;
        }
    }
    s
}

/// A cosine-only series as `Σₖ pₖ cosᵏφ`, `cos nφ = Tₙ(cos φ)` (Chebyshev),
/// with at least `min_len` coefficients.
fn series_to_cos_poly(s: &FourierSeries, min_len: usize) -> Vec<f64> {
    let order = s.order();
    let mut p = vec![0.0; (order + 1).max(min_len)];
    // T₀ = 1, T₁ = x, Tₙ₊₁ = 2x Tₙ − Tₙ₋₁ (integer coefficients, exact).
    let (mut prev, mut cur) = (vec![1.0], vec![0.0, 1.0]);
    p[0] += s.a(0);
    for n in 1..=order {
        let an = s.a(n);
        if an != 0.0 {
            for (i, &t) in cur.iter().enumerate() {
                p[i] += an * t;
            }
        }
        let mut next = vec![0.0; cur.len() + 1];
        for (i, &t) in cur.iter().enumerate() {
            next[i + 1] += 2.0 * t;
        }
        for (i, &t) in prev.iter().enumerate() {
            next[i] -= t;
        }
        prev = std::mem::replace(&mut cur, next);
    }
    p
}

// ── single cosine terms ─────────────────────────────────────────────────────

/// `k[1 + cos(nφ − γ)]`, `γ` = `phase` in degrees: one term of `dihedral
/// periodic` (LAMMPS `fourier`), and the whole of molrs `improper periodic`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosineTerm {
    pub k: f64,
    pub periodicity: f64,
    /// degrees
    pub phase: f64,
}

impl CosineTerm {
    /// The energy at `phi` (radians), straight from the formula.
    pub fn energy(&self, phi: f64) -> f64 {
        self.k * (1.0 + (self.periodicity * phi - self.phase.to_radians()).cos())
    }

    /// `k + k cos γ cos nφ + k sin γ sin nφ` (`n < 0`: `cos` is even, `sin` odd).
    pub fn to_series(&self) -> Result<FourierSeries> {
        let mut s = FourierSeries::zero();
        self.add_to(&mut s)?;
        Ok(s)
    }

    fn add_to(&self, s: &mut FourierSeries) -> Result<()> {
        let (n, sign) = order_of(self.periodicity)?;
        let (c, sn) = cos_sin_deg(self.phase);
        s.add_cos(0, self.k);
        s.add_cos(n, self.k * c);
        s.add_sin(n, sign * self.k * sn);
        Ok(())
    }

    /// `aₙ cos nφ + bₙ sin nφ` as `k cos(nφ − γ)`: `k = √(a² + b²) ≥ 0`,
    /// `γ ∈ (−180°, 180°]`.
    fn from_coefficients(n: usize, a: f64, b: f64) -> Self {
        Self {
            k: a.hypot(b),
            periodicity: n as f64,
            phase: phase_deg(b, a),
        }
    }

    /// The one term of a single-order series (constant offset aside). A
    /// constant series gives `k = 0, n = 1, γ = 0`.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::MultiTerm`] for more than one order.
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        Ok(match single_order(s)? {
            Some(n) => Self::from_coefficients(n, s.a(n), s.b(n)),
            None => Self {
                k: 0.0,
                periodicity: 1.0,
                phase: 0.0,
            },
        })
    }

    /// The harmonic improper `K(χ − χ0)²` (LAMMPS's un-halved `K`) that
    /// agrees with this term to second order about a minimum:
    /// `K = |k| n²/2`, `χ0` the minimum `φ*` (`nφ* − γ = 180°` for `k > 0`,
    /// `0°` for `k < 0`) nearest 0, as `|φ*|`. The quartic terms differ by
    /// `−|k| n⁴ δ⁴/24`.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::NoCurvature`] for `k = 0` or `n = 0`;
    /// [`TorsionRefusal::NonIntegerPeriodicity`].
    pub fn second_order_harmonic(&self) -> Result<ImproperHarmonic> {
        let (n, _) = order_of(self.periodicity)?;
        if n == 0 || self.k == 0.0 {
            return Err(TorsionRefusal::NoCurvature);
        }
        let shift = if self.k > 0.0 { 180.0 } else { 0.0 };
        let period = 360.0 / n as f64;
        let base = ((self.phase + shift) / self.periodicity).rem_euclid(period);
        Ok(ImproperHarmonic {
            k: self.k.abs() * (n * n) as f64 / 2.0,
            chi0: base.min(period - base),
        })
    }
}

/// `k[1 + d cos nφ]`, `d` = `sign` ∈ {+1, −1}: LAMMPS `dihedral harmonic` and
/// `improper cvff` (`K d n`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedCosine {
    pub k: f64,
    pub sign: f64,
    pub periodicity: f64,
}

impl SignedCosine {
    /// The energy at `phi` (radians), straight from the formula.
    pub fn energy(&self, phi: f64) -> f64 {
        self.k * (1.0 + self.sign * (self.periodicity * phi).cos())
    }

    /// `k + k d cos nφ`.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::InvalidSign`], [`TorsionRefusal::NonIntegerPeriodicity`].
    pub fn to_series(&self) -> Result<FourierSeries> {
        if self.sign != 1.0 && self.sign != -1.0 {
            return Err(TorsionRefusal::InvalidSign { sign: self.sign });
        }
        let (n, _) = order_of(self.periodicity)?;
        let mut s = FourierSeries::zero();
        s.add_cos(0, self.k);
        s.add_cos(n, self.k * self.sign);
        Ok(s)
    }

    /// `aₙ cos nφ` as `k = |aₙ|`, `d = sign(aₙ)` (constant offset aside). A
    /// constant series gives `k = 0, d = 1, n = 1`.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::MultiTerm`], [`TorsionRefusal::SineTerm`].
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        let Some(n) = single_order(s)? else {
            return Ok(Self {
                k: 0.0,
                sign: 1.0,
                periodicity: 1.0,
            });
        };
        require_cosines(s, None)?;
        let a = s.a(n);
        Ok(Self {
            k: a.abs(),
            sign: if a < 0.0 { -1.0 } else { 1.0 },
            periodicity: n as f64,
        })
    }
}

// ── proper forms ────────────────────────────────────────────────────────────

/// molrs `dihedral periodic` (LAMMPS `dihedral fourier`):
/// `Σₘ kₘ[1 + cos(nₘφ − γₘ)]`. Every series has this form.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Periodic {
    pub terms: Vec<CosineTerm>,
}

impl Periodic {
    /// The sum of the terms' series.
    pub fn to_series(&self) -> Result<FourierSeries> {
        let mut s = FourierSeries::zero();
        for t in &self.terms {
            t.add_to(&mut s)?;
        }
        Ok(s)
    }

    /// One term per order `n ≥ 1` present, ascending, `k ≥ 0`,
    /// `γ ∈ (−180°, 180°]` (constant offset aside). Never refuses.
    pub fn from_series(s: &FourierSeries) -> Self {
        Self {
            terms: s
                .orders()
                .into_iter()
                .map(|n| CosineTerm::from_coefficients(n, s.a(n), s.b(n)))
                .collect(),
        }
    }
}

/// LAMMPS `dihedral charmm`: `k[1 + cos(nφ − d)]`. `w` weights the 1-4 pair
/// the dihedral prices; it is not a torsion parameter, so the series does not
/// carry it and [`Charmm::from_series`] takes it from the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Charmm {
    pub term: CosineTerm,
    pub w: f64,
}

impl Charmm {
    pub fn to_series(&self) -> Result<FourierSeries> {
        self.term.to_series()
    }

    /// The single term of `s` (see [`CosineTerm::from_series`]) with 1-4
    /// weight `w`.
    pub fn from_series(s: &FourierSeries, w: f64) -> Result<Self> {
        Ok(Self {
            term: CosineTerm::from_series(s)?,
            w,
        })
    }
}

/// LAMMPS `dihedral opls`:
/// `½[k₁(1 + cos φ) + k₂(1 − cos 2φ) + k₃(1 + cos 3φ) + k₄(1 − cos 4φ)]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Opls {
    pub k: [f64; 4],
}

impl Opls {
    /// `a₀ = ½Σkₙ`, `aₙ = ±½kₙ` (`−` at even `n`).
    pub fn to_series(&self) -> FourierSeries {
        let [k1, k2, k3, k4] = self.k;
        FourierSeries::new(
            vec![
                0.5 * (k1 + k2 + k3 + k4),
                0.5 * k1,
                -0.5 * k2,
                0.5 * k3,
                -0.5 * k4,
            ],
            vec![],
        )
    }

    /// `kₙ = ±2aₙ` (constant offset aside).
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::OrderTooHigh`] above `n = 4`, [`TorsionRefusal::SineTerm`].
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        require_cosines(s, Some(4))?;
        Ok(Self {
            k: [2.0 * s.a(1), -2.0 * s.a(2), 2.0 * s.a(3), -2.0 * s.a(4)],
        })
    }
}

/// LAMMPS `dihedral class2`, torsion part: `Σₙ₌₁³ kₙ[1 − cos(nφ − φₙ)]`,
/// `φₙ` in degrees. The cross terms are outside the Class-I IR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Class2 {
    pub k: [f64; 3],
    /// degrees
    pub phi: [f64; 3],
}

impl Class2 {
    /// `a₀ = Σkₙ`, `aₙ = −kₙ cos φₙ`, `bₙ = −kₙ sin φₙ`.
    pub fn to_series(&self) -> FourierSeries {
        let mut s = FourierSeries::zero();
        for (i, (&k, &phi)) in self.k.iter().zip(&self.phi).enumerate() {
            let (c, sn) = cos_sin_deg(phi);
            s.add_cos(0, k);
            s.add_cos(i + 1, -k * c);
            s.add_sin(i + 1, -k * sn);
        }
        s
    }

    /// `kₙ = √(a² + b²) ≥ 0`, `φₙ` the phase of `(−aₙ, −bₙ)` on
    /// (−180°, 180°] (constant offset aside).
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::OrderTooHigh`] above `n = 3`.
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        if s.order() > 3 {
            return Err(TorsionRefusal::OrderTooHigh {
                n: s.order(),
                max: 3,
            });
        }
        let mut out = Self {
            k: [0.0; 3],
            phi: [0.0; 3],
        };
        for (i, (k, phi)) in out.k.iter_mut().zip(&mut out.phi).enumerate() {
            let (a, b) = (s.a(i + 1), s.b(i + 1));
            *k = a.hypot(b);
            *phi = phase_deg(-b, -a);
        }
        Ok(out)
    }
}

/// LAMMPS `dihedral nharmonic`: `Σᵢ₌₁ᴺ Aᵢ cosⁱ⁻¹φ` (`a[i−1]` = `Aᵢ`).
#[derive(Debug, Clone, PartialEq)]
pub struct NHarmonic {
    pub a: Vec<f64>,
}

impl NHarmonic {
    pub fn to_series(&self) -> FourierSeries {
        cos_poly_to_series(&self.a)
    }

    /// `cos nφ = Tₙ(cos φ)`, constant included; `N = order + 1` (at least 1).
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::SineTerm`].
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        require_cosines(s, None)?;
        Ok(Self {
            a: series_to_cos_poly(s, 1),
        })
    }
}

/// LAMMPS `dihedral multi/harmonic`: `Σₙ₌₁⁵ Aₙ cosⁿ⁻¹φ` — `nharmonic` at `N = 5`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiHarmonic {
    pub a: [f64; 5],
}

impl MultiHarmonic {
    pub fn to_series(&self) -> FourierSeries {
        cos_poly_to_series(&self.a)
    }

    /// As [`NHarmonic::from_series`], constant included.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::OrderTooHigh`] above `n = 4`, [`TorsionRefusal::SineTerm`].
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        require_cosines(s, Some(4))?;
        let p = series_to_cos_poly(s, 5);
        Ok(Self {
            a: [p[0], p[1], p[2], p[3], p[4]],
        })
    }
}

/// Ryckaert–Bellemans, `Σₙ₌₀⁵ Cₙ cosⁿψ` with `ψ = φ − 180°` (GROMACS funct 3,
/// OpenMM `RBTorsionForce`). Not a molrs style: since `cos ψ = −cos φ` it is
/// `multi/harmonic` / `nharmonic` with `Aₙ₊₁ = (−1)ⁿ Cₙ`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RyckaertBellemans {
    pub c: [f64; 6],
}

impl RyckaertBellemans {
    /// The `nharmonic` coefficients `Aₙ₊₁ = (−1)ⁿ Cₙ` (N = 6).
    pub fn to_nharmonic(&self) -> NHarmonic {
        NHarmonic {
            a: self
                .c
                .iter()
                .enumerate()
                .map(|(n, &c)| if n % 2 == 0 { c } else { -c })
                .collect(),
        }
    }

    pub fn to_series(&self) -> FourierSeries {
        self.to_nharmonic().to_series()
    }

    /// `Cₙ = (−1)ⁿ Aₙ₊₁`, constant included.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::OrderTooHigh`] above `n = 5`, [`TorsionRefusal::SineTerm`].
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        require_cosines(s, Some(5))?;
        let p = series_to_cos_poly(s, 6);
        let mut c = [0.0; 6];
        for (n, slot) in c.iter_mut().enumerate() {
            *slot = if n % 2 == 0 { p[n] } else { -p[n] };
        }
        Ok(Self { c })
    }
}

// ── the harmonic improper (not a Fourier form) ──────────────────────────────

/// LAMMPS `improper harmonic`: `K(χ − χ0)²`, `χ = |φ|`, `χ0` in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImproperHarmonic {
    pub k: f64,
    /// degrees
    pub chi0: f64,
}

impl ImproperHarmonic {
    /// The energy at `phi` (radians), straight from the formula.
    pub fn energy(&self, phi: f64) -> f64 {
        let d = phi.abs() - self.chi0.to_radians();
        self.k * d * d
    }

    /// Always refuses: `K(|φ| − χ0)²` is not a finite Fourier series.
    pub fn to_series(&self) -> Result<FourierSeries> {
        Err(not_fourier_improper_harmonic())
    }

    /// The periodic improper of periodicity `n` agreeing with this one to
    /// second order about `φ = χ0`: `k = 2K/n²`, `γ = nχ0 + 180°` (on
    /// (−180°, 180°]), the inverse of [`CosineTerm::second_order_harmonic`].
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::NonIntegerPeriodicity`]; [`TorsionRefusal::NoCurvature`]
    /// for `n = 0`.
    pub fn second_order_periodic(&self, periodicity: f64) -> Result<CosineTerm> {
        let (n, _) = order_of(periodicity)?;
        if n == 0 {
            return Err(TorsionRefusal::NoCurvature);
        }
        let n = n as f64;
        let mut phase = (n * self.chi0 + 180.0).rem_euclid(360.0);
        if phase > 180.0 {
            phase -= 360.0;
        }
        Ok(CosineTerm {
            k: 2.0 * self.k / (n * n),
            periodicity: n,
            phase,
        })
    }
}

fn not_fourier_improper_harmonic() -> TorsionRefusal {
    TorsionRefusal::NotAFourierForm {
        style: "improper harmonic",
        reason: "K(|φ| − χ0)² is not a finite Fourier series; a periodic improper \
                 matches it to second order only (K = n²k/2)",
    }
}

// ── forms as stored params ──────────────────────────────────────────────────

/// A torsion row of a molrs style, read from its stored [`Params`].
#[derive(Debug, Clone, PartialEq)]
pub enum TorsionForm {
    /// `dihedral periodic`
    Periodic(Periodic),
    /// `dihedral charmm`
    Charmm(Charmm),
    /// `dihedral opls`
    Opls(Opls),
    /// `dihedral multi/harmonic`
    MultiHarmonic(MultiHarmonic),
    /// `dihedral nharmonic`
    NHarmonic(NHarmonic),
    /// `dihedral harmonic`
    Harmonic(SignedCosine),
    /// `dihedral class2` (torsion part)
    Class2(Class2),
    /// `improper cvff`
    ImproperCvff(SignedCosine),
    /// `improper periodic`
    ImproperPeriodic(CosineTerm),
}

/// A Ryckaert–Bellemans row `Σₙ₌₀⁵ Cₙ cosⁿ(φ − 180°)` as the IR's polynomial
/// `Σ Aₙ₊₁ cosⁿφ`, `Aₙ₊₁ = (−1)ⁿ Cₙ` (`cos(φ − 180°) = −cos φ`), same energy
/// unit in and out: `dihedral multi/harmonic` (`a1..a5`) when `C₅ = 0`,
/// `dihedral nharmonic` (N = 6) otherwise — exact, constant included. The
/// GROMACS (funct 3) and OpenMM (`<RBTorsionForce>`) readers share it.
pub(crate) fn rb_polynomial(c: [f64; 6]) -> (&'static str, Params) {
    let (style, keep) = if c[5] == 0.0 {
        ("multi/harmonic", 5)
    } else {
        ("nharmonic", 6)
    };
    let mut params = Params::new();
    for (n, &cn) in c[..keep].iter().enumerate() {
        params.set(&format!("a{}", n + 1), if n % 2 == 0 { cn } else { -cn });
    }
    (style, params)
}

/// The `nharmonic` coefficients `a1..aN` of a params bag: contiguous from
/// `a1`, at least one, none past the first gap. Shared by the kernel, the
/// algebra and the LAMMPS writer, so the three read one `N`.
pub(crate) fn nharmonic_coefficients(p: &Params) -> std::result::Result<Vec<f64>, String> {
    let mut a = Vec::new();
    while let Some(v) = p.get(&format!("a{}", a.len() + 1)) {
        a.push(v);
    }
    if a.is_empty() {
        return Err("dihedral nharmonic: missing param `a1` (needs a1..aN, N ≥ 1)".into());
    }
    if let Some((key, _)) = p.iter().find(|(key, _)| {
        key.strip_prefix('a')
            .and_then(|i| i.parse::<usize>().ok())
            .is_some_and(|i| i > a.len())
    }) {
        return Err(format!(
            "dihedral nharmonic: `{key}` follows a gap after `a{}`",
            a.len()
        ));
    }
    Ok(a)
}

impl TorsionForm {
    /// The form of a row of `(category, style)` with params `p`, read with
    /// the defaults the kernel reads (an absent phase or OPLS/class2/
    /// multi-harmonic coefficient is 0; an absent charmm `w` is 0).
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::NotAFourierForm`] for `improper harmonic` and
    /// `dihedral quadratic`; [`TorsionRefusal::UnknownStyle`];
    /// [`TorsionRefusal::MissingParam`].
    pub fn from_params(category: &str, style: &str, p: &Params) -> Result<Self> {
        let need = |what: &'static str, key: &str| {
            p.get(key).ok_or_else(|| TorsionRefusal::MissingParam {
                style: what,
                key: key.to_owned(),
            })
        };
        let or0 = |key: &str| p.get(key).unwrap_or(0.0);
        Ok(match (category, style) {
            ("dihedral", "periodic") => {
                let what = "dihedral periodic";
                let mut terms = Vec::new();
                let mut m = 1;
                while let Some(k) = p.get(&format!("k{m}")) {
                    terms.push(CosineTerm {
                        k,
                        periodicity: need(what, &format!("periodicity{m}"))?,
                        phase: or0(&format!("phase{m}")),
                    });
                    m += 1;
                }
                if terms.is_empty() {
                    terms.push(CosineTerm {
                        k: need(what, "k")?,
                        periodicity: need(what, "periodicity")?,
                        phase: or0("phase"),
                    });
                }
                Self::Periodic(Periodic { terms })
            }
            ("dihedral", "charmm") => Self::Charmm(Charmm {
                term: CosineTerm {
                    k: need("dihedral charmm", "k")?,
                    periodicity: need("dihedral charmm", "periodicity")?,
                    phase: or0("phase"),
                },
                w: or0("w"),
            }),
            ("dihedral", "opls") => Self::Opls(Opls {
                k: [or0("k1"), or0("k2"), or0("k3"), or0("k4")],
            }),
            ("dihedral", "multi/harmonic") => Self::MultiHarmonic(MultiHarmonic {
                a: [or0("a1"), or0("a2"), or0("a3"), or0("a4"), or0("a5")],
            }),
            ("dihedral", "nharmonic") => Self::NHarmonic(NHarmonic {
                a: nharmonic_coefficients(p).map_err(|_| TorsionRefusal::MissingParam {
                    style: "dihedral nharmonic",
                    key: "a1..aN".into(),
                })?,
            }),
            ("dihedral", "harmonic") | ("improper", "cvff") => {
                let what = if category == "dihedral" {
                    "dihedral harmonic"
                } else {
                    "improper cvff"
                };
                let form = SignedCosine {
                    k: need(what, "k")?,
                    sign: need(what, "sign")?,
                    periodicity: need(what, "periodicity")?,
                };
                if category == "dihedral" {
                    Self::Harmonic(form)
                } else {
                    Self::ImproperCvff(form)
                }
            }
            ("dihedral", "class2") => Self::Class2(Class2 {
                k: [or0("k1"), or0("k2"), or0("k3")],
                phi: [or0("phi1"), or0("phi2"), or0("phi3")],
            }),
            ("improper", "periodic") => Self::ImproperPeriodic(CosineTerm {
                k: need("improper periodic", "k")?,
                periodicity: need("improper periodic", "periodicity")?,
                phase: or0("phase"),
            }),
            ("improper", "harmonic") => return Err(not_fourier_improper_harmonic()),
            ("dihedral", "quadratic") => {
                return Err(TorsionRefusal::NotAFourierForm {
                    style: "dihedral quadratic",
                    reason: "K(φ − φ0)² is not a finite Fourier series",
                });
            }
            _ => {
                return Err(TorsionRefusal::UnknownStyle {
                    category: category.to_owned(),
                    style: style.to_owned(),
                });
            }
        })
    }

    /// `(category, style)` of the form.
    pub fn style(&self) -> (&'static str, &'static str) {
        match self {
            Self::Periodic(_) => ("dihedral", "periodic"),
            Self::Charmm(_) => ("dihedral", "charmm"),
            Self::Opls(_) => ("dihedral", "opls"),
            Self::MultiHarmonic(_) => ("dihedral", "multi/harmonic"),
            Self::NHarmonic(_) => ("dihedral", "nharmonic"),
            Self::Harmonic(_) => ("dihedral", "harmonic"),
            Self::Class2(_) => ("dihedral", "class2"),
            Self::ImproperCvff(_) => ("improper", "cvff"),
            Self::ImproperPeriodic(_) => ("improper", "periodic"),
        }
    }

    /// The stored params of the form, in the spelling its kernel reads
    /// (`dihedral periodic` indexed `k<m>`/`periodicity<m>`/`phase<m>`; a
    /// termless one as one zero term).
    pub fn to_params(&self) -> Params {
        let mut p = Params::new();
        let mut set = |key: &str, v: f64| p.set(key, v);
        match self {
            Self::Periodic(f) => {
                let zero = [CosineTerm {
                    k: 0.0,
                    periodicity: 1.0,
                    phase: 0.0,
                }];
                let terms = if f.terms.is_empty() {
                    &zero[..]
                } else {
                    &f.terms
                };
                for (i, t) in terms.iter().enumerate() {
                    set(&format!("k{}", i + 1), t.k);
                    set(&format!("periodicity{}", i + 1), t.periodicity);
                    set(&format!("phase{}", i + 1), t.phase);
                }
            }
            Self::Charmm(f) => {
                set("k", f.term.k);
                set("periodicity", f.term.periodicity);
                set("phase", f.term.phase);
                set("w", f.w);
            }
            Self::ImproperPeriodic(t) => {
                set("k", t.k);
                set("periodicity", t.periodicity);
                set("phase", t.phase);
            }
            Self::Opls(f) => {
                for (i, &k) in f.k.iter().enumerate() {
                    set(&format!("k{}", i + 1), k);
                }
            }
            Self::MultiHarmonic(f) => {
                for (i, &a) in f.a.iter().enumerate() {
                    set(&format!("a{}", i + 1), a);
                }
            }
            Self::NHarmonic(f) => {
                for (i, &a) in f.a.iter().enumerate() {
                    set(&format!("a{}", i + 1), a);
                }
            }
            Self::Harmonic(f) | Self::ImproperCvff(f) => {
                set("k", f.k);
                set("sign", f.sign);
                set("periodicity", f.periodicity);
            }
            Self::Class2(f) => {
                for (i, (&k, &phi)) in f.k.iter().zip(&f.phi).enumerate() {
                    set(&format!("k{}", i + 1), k);
                    set(&format!("phi{}", i + 1), phi);
                }
            }
        }
        p
    }

    /// The exact series of the form, constant included.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::NonIntegerPeriodicity`], [`TorsionRefusal::InvalidSign`].
    pub fn to_series(&self) -> Result<FourierSeries> {
        match self {
            Self::Periodic(f) => f.to_series(),
            Self::Charmm(f) => f.to_series(),
            Self::Opls(f) => Ok(f.to_series()),
            Self::MultiHarmonic(f) => Ok(f.to_series()),
            Self::NHarmonic(f) => Ok(f.to_series()),
            Self::Harmonic(f) | Self::ImproperCvff(f) => f.to_series(),
            Self::Class2(f) => Ok(f.to_series()),
            Self::ImproperPeriodic(t) => t.to_series(),
        }
    }

    /// The form of `(category, style)` reproducing `s` (up to a constant
    /// where the form fixes its own). `dihedral charmm` gets `w = 0`: the 1-4
    /// weight is the source row's, not the series', so set it from there.
    ///
    /// # Errors
    ///
    /// The target's image condition (see the module table), or
    /// [`TorsionRefusal::UnknownStyle`] / [`TorsionRefusal::NotAFourierForm`].
    pub fn from_series(category: &str, style: &str, s: &FourierSeries) -> Result<Self> {
        Ok(match (category, style) {
            ("dihedral", "periodic") => Self::Periodic(Periodic::from_series(s)),
            ("dihedral", "charmm") => Self::Charmm(Charmm::from_series(s, 0.0)?),
            ("dihedral", "opls") => Self::Opls(Opls::from_series(s)?),
            ("dihedral", "multi/harmonic") => Self::MultiHarmonic(MultiHarmonic::from_series(s)?),
            ("dihedral", "nharmonic") => Self::NHarmonic(NHarmonic::from_series(s)?),
            ("dihedral", "harmonic") => Self::Harmonic(SignedCosine::from_series(s)?),
            ("dihedral", "class2") => Self::Class2(Class2::from_series(s)?),
            ("improper", "cvff") => Self::ImproperCvff(SignedCosine::from_series(s)?),
            ("improper", "periodic") => Self::ImproperPeriodic(CosineTerm::from_series(s)?),
            // Refused with the reason `from_params` gives.
            ("improper", "harmonic") | ("dihedral", "quadratic") => {
                return Self::from_params(category, style, &Params::new());
            }
            _ => {
                return Err(TorsionRefusal::UnknownStyle {
                    category: category.to_owned(),
                    style: style.to_owned(),
                });
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::f64::consts::PI;

    // ── random forms ──

    const SEED: u64 = 0x0070_1510_2026;
    const CASES: usize = 200;

    fn coeff(rng: &mut StdRng) -> f64 {
        rng.random_range(-5.0..5.0)
    }

    fn positive(rng: &mut StdRng) -> f64 {
        rng.random_range(0.05..5.0)
    }

    /// A phase on (−180°, 180°], a quarter of the time exactly 0° or 180°.
    fn phase(rng: &mut StdRng) -> f64 {
        match rng.random_range(0..8) {
            0 => 0.0,
            1 => 180.0,
            _ => {
                let p: f64 = rng.random_range(-180.0..180.0);
                if p == -180.0 { 180.0 } else { p }
            }
        }
    }

    fn sign(rng: &mut StdRng) -> f64 {
        if rng.random_range(0..2) == 0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Any term: negative k, n ∈ 0..=6, any phase (also beyond ±180°).
    fn any_term(rng: &mut StdRng) -> CosineTerm {
        CosineTerm {
            k: coeff(rng),
            periodicity: rng.random_range(0..=6) as f64,
            phase: rng.random_range(-540.0..540.0),
        }
    }

    /// A term `from_series` reproduces: k > 0, n ∈ 1..=6, γ ∈ (−180°, 180°].
    fn canonical_term(rng: &mut StdRng, n: usize) -> CosineTerm {
        CosineTerm {
            k: positive(rng),
            periodicity: n as f64,
            phase: phase(rng),
        }
    }

    fn random_series(rng: &mut StdRng, order: usize, sines: bool) -> FourierSeries {
        let a = (0..=order).map(|_| coeff(rng)).collect();
        let b = if sines {
            (0..=order).map(|_| coeff(rng)).collect()
        } else {
            vec![]
        };
        FourierSeries::new(a, b)
    }

    type Energy = Box<dyn Fn(f64) -> f64>;

    fn energy_fn(f: impl Fn(f64) -> f64 + 'static) -> Energy {
        Box::new(f)
    }

    /// Every form the algebra covers, from random coefficients (negative k,
    /// arbitrary phases, n ≤ 6), with its direct-formula energy.
    fn random_forms(rng: &mut StdRng) -> Vec<(TorsionForm, Energy)> {
        let periodic = Periodic {
            terms: (0..rng.random_range(1..=4))
                .map(|_| any_term(rng))
                .collect(),
        };
        let charmm = Charmm {
            term: any_term(rng),
            w: rng.random_range(0.0..1.0),
        };
        let improper = any_term(rng);
        let opls = Opls {
            k: [coeff(rng), coeff(rng), coeff(rng), coeff(rng)],
        };
        let multi = MultiHarmonic {
            a: [coeff(rng), coeff(rng), coeff(rng), coeff(rng), coeff(rng)],
        };
        let nharm = NHarmonic {
            a: (0..rng.random_range(1..=7)).map(|_| coeff(rng)).collect(),
        };
        let harmonic = SignedCosine {
            k: coeff(rng),
            sign: sign(rng),
            periodicity: rng.random_range(0..=6) as f64,
        };
        let cvff = SignedCosine {
            k: coeff(rng),
            sign: sign(rng),
            periodicity: rng.random_range(0..=6) as f64,
        };
        let class2 = Class2 {
            k: [coeff(rng), coeff(rng), coeff(rng)],
            phi: [phase(rng), phase(rng), rng.random_range(-540.0..540.0)],
        };
        let p2 = periodic.clone();
        let n2 = nharm.clone();
        vec![
            (
                TorsionForm::Periodic(periodic),
                energy_fn(move |phi| p2.terms.iter().map(|t| t.energy(phi)).sum()),
            ),
            (
                TorsionForm::Charmm(charmm),
                energy_fn(move |phi| charmm.term.energy(phi)),
            ),
            (
                TorsionForm::ImproperPeriodic(improper),
                energy_fn(move |phi| improper.energy(phi)),
            ),
            (
                TorsionForm::Opls(opls),
                energy_fn(move |phi| {
                    let [k1, k2, k3, k4] = opls.k;
                    0.5 * (k1 * (1.0 + phi.cos())
                        + k2 * (1.0 - (2.0 * phi).cos())
                        + k3 * (1.0 + (3.0 * phi).cos())
                        + k4 * (1.0 - (4.0 * phi).cos()))
                }),
            ),
            (
                TorsionForm::MultiHarmonic(multi),
                energy_fn(move |phi| {
                    multi
                        .a
                        .iter()
                        .enumerate()
                        .map(|(n, a)| a * phi.cos().powi(n as i32))
                        .sum()
                }),
            ),
            (
                TorsionForm::NHarmonic(nharm),
                energy_fn(move |phi| {
                    n2.a.iter()
                        .enumerate()
                        .map(|(n, a)| a * phi.cos().powi(n as i32))
                        .sum()
                }),
            ),
            (
                TorsionForm::Harmonic(harmonic),
                energy_fn(move |phi| harmonic.energy(phi)),
            ),
            (
                TorsionForm::ImproperCvff(cvff),
                energy_fn(move |phi| cvff.energy(phi)),
            ),
            (
                TorsionForm::Class2(class2),
                energy_fn(move |phi| {
                    class2
                        .k
                        .iter()
                        .zip(&class2.phi)
                        .enumerate()
                        .map(|(i, (k, p))| {
                            k * (1.0 - ((i + 1) as f64 * phi - p.to_radians()).cos())
                        })
                        .sum()
                }),
            ),
        ]
    }

    fn assert_energy(got: f64, want: f64, scale: f64, what: &str) {
        assert!(
            (got - want).abs() <= 1e-12 * scale.max(want.abs()),
            "{what}: {got} vs {want} (scale {scale})"
        );
    }

    /// E_form(φ) = E_series(φ) to 1e-12 relative for every embedding, the
    /// constant included.
    #[test]
    fn every_embedding_is_the_energy() {
        let mut rng = StdRng::seed_from_u64(SEED);
        for _ in 0..CASES {
            for (form, energy) in random_forms(&mut rng) {
                let s = form.to_series().expect("in the domain");
                let scale = 64.0 * s.max_abs_diff(&FourierSeries::zero());
                for _ in 0..8 {
                    let phi = rng.random_range(-PI..PI);
                    assert_energy(s.energy(phi), energy(phi), scale, &format!("{form:?}"));
                }
            }
        }
    }

    /// Form → series → form is the identity on a form's canonical inputs
    /// (k ≥ 0, distinct ascending orders, phases on (−180°, 180°]), and the
    /// same series on any input.
    #[test]
    fn round_trips_are_the_identity() {
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0);
        for _ in 0..CASES {
            // Every form: back through its own style gives the same series
            // (up to the constant for a fixed-constant form).
            for (form, _) in random_forms(&mut rng) {
                let s = form.to_series().unwrap();
                let (category, style) = form.style();
                let back = TorsionForm::from_series(category, style, &s)
                    .unwrap_or_else(|e| panic!("{form:?}: {e}"));
                let s2 = back.to_series().unwrap();
                let d = s.canonical().max_abs_diff(&s2.canonical());
                assert!(d < 1e-12 * 64.0, "{form:?} → {back:?}: Δ = {d}");
            }

            // Canonical inputs come back identical.
            let mut orders: Vec<usize> = (1..=6).filter(|_| rng.random_range(0..2) == 0).collect();
            if orders.is_empty() {
                orders.push(1);
            }
            let periodic = Periodic {
                terms: orders
                    .iter()
                    .map(|&n| canonical_term(&mut rng, n))
                    .collect(),
            };
            let back = Periodic::from_series(&periodic.to_series().unwrap());
            assert_eq!(back.terms.len(), periodic.terms.len());
            for (t, u) in periodic.terms.iter().zip(&back.terms) {
                assert!(
                    close(t.k, u.k) && t.periodicity == u.periodicity,
                    "{t:?} {u:?}"
                );
                assert!((t.phase - u.phase).abs() < 1e-10, "{t:?} {u:?}");
            }

            let n = rng.random_range(1..=6);
            let term = canonical_term(&mut rng, n);
            let charmm = Charmm { term, w: 0.5 };
            let back = Charmm::from_series(&charmm.to_series().unwrap(), 0.5).unwrap();
            assert!(close(back.term.k, term.k) && (back.term.phase - term.phase).abs() < 1e-10);
            assert_eq!((back.term.periodicity, back.w), (term.periodicity, 0.5));

            let sc = SignedCosine {
                k: positive(&mut rng),
                sign: sign(&mut rng),
                periodicity: rng.random_range(1..=6) as f64,
            };
            assert_eq!(SignedCosine::from_series(&sc.to_series().unwrap()), Ok(sc));

            let opls = Opls {
                k: [
                    coeff(&mut rng),
                    coeff(&mut rng),
                    coeff(&mut rng),
                    coeff(&mut rng),
                ],
            };
            assert_eq!(Opls::from_series(&opls.to_series()), Ok(opls));

            let multi = MultiHarmonic {
                a: [
                    coeff(&mut rng),
                    coeff(&mut rng),
                    coeff(&mut rng),
                    coeff(&mut rng),
                    coeff(&mut rng),
                ],
            };
            let back = MultiHarmonic::from_series(&multi.to_series()).unwrap();
            for (x, y) in multi.a.iter().zip(&back.a) {
                assert!((x - y).abs() < 1e-12 * 64.0, "{multi:?} → {back:?}");
            }

            // The top coefficient non-zero, so N survives the trip.
            let mut a: Vec<f64> = (0..rng.random_range(1..=7))
                .map(|_| coeff(&mut rng))
                .collect();
            let top = a.len() - 1;
            if top > 0 {
                a[top] = positive(&mut rng);
            }
            let nh = NHarmonic { a };
            let back = NHarmonic::from_series(&nh.to_series()).unwrap();
            assert_eq!(back.a.len(), nh.a.len(), "{nh:?} → {back:?}");
            for (x, y) in nh.a.iter().zip(&back.a) {
                assert!((x - y).abs() < 1e-12 * 64.0, "{nh:?} → {back:?}");
            }

            let rb = RyckaertBellemans {
                c: [0; 6].map(|_| coeff(&mut rng)),
            };
            let back = RyckaertBellemans::from_series(&rb.to_series()).unwrap();
            for (x, y) in rb.c.iter().zip(&back.c) {
                assert!((x - y).abs() < 1e-12 * 64.0, "{rb:?} → {back:?}");
            }

            let class2 = Class2 {
                k: [0; 3].map(|_| positive(&mut rng)),
                phi: [0; 3].map(|_| phase(&mut rng)),
            };
            let back = Class2::from_series(&class2.to_series()).unwrap();
            for i in 0..3 {
                assert!(close(class2.k[i], back.k[i]), "{class2:?} → {back:?}");
                assert!(
                    (class2.phi[i] - back.phi[i]).abs() < 1e-10,
                    "{class2:?} → {back:?}"
                );
            }

            // The params spelling round-trips too.
            for (form, _) in random_forms(&mut rng) {
                let (category, style) = form.style();
                let again = TorsionForm::from_params(category, style, &form.to_params()).unwrap();
                assert_eq!(again, form);
            }
        }
    }

    /// RB ↔ multi/harmonic ↔ OPLS ↔ periodic: every link is the same series,
    /// and the RB ↔ multi/harmonic link is the sign map `Aₙ₊₁ = (−1)ⁿCₙ`.
    #[test]
    fn the_rb_multi_harmonic_opls_periodic_chain() {
        let mut rng = StdRng::seed_from_u64(SEED + 2);
        for _ in 0..CASES {
            let mut c = [0; 6].map(|_| coeff(&mut rng));
            c[5] = 0.0; // in OPLS's image
            let rb = RyckaertBellemans { c };
            let s = rb.to_series();

            assert_eq!(
                rb.to_nharmonic().a,
                [c[0], -c[1], c[2], -c[3], c[4], -c[5]],
                "A(n+1) = (−1)ⁿ C(n), exactly"
            );
            let multi = MultiHarmonic::from_series(&s).unwrap();
            for (n, (&a, &cn)) in multi.a.iter().zip(&c).enumerate() {
                let want = if n % 2 == 0 { cn } else { -cn };
                assert!((a - want).abs() < 1e-12 * 64.0, "A{}: {a} vs {want}", n + 1);
            }
            let opls = Opls::from_series(&multi.to_series()).unwrap();
            let periodic = Periodic::from_series(&opls.to_series());
            let rb2 = RyckaertBellemans::from_series(&periodic.to_series().unwrap()).unwrap();

            for (name, other) in [
                ("multi/harmonic", multi.to_series()),
                ("opls", opls.to_series()),
                ("periodic", periodic.to_series().unwrap()),
                ("rb", rb2.to_series()),
            ] {
                let d = s.canonical().max_abs_diff(&other.canonical());
                assert!(d < 1e-12 * 64.0, "{name}: Δ = {d}");
            }
            for phi in [0.0, 0.7, PI / 2.0, 2.0, PI] {
                // RB and multi/harmonic carry the constant: equal energies.
                let e_rb: f64 = (0..6).map(|n| c[n] * (phi - PI).cos().powi(n as i32)).sum();
                assert_energy(multi.to_series().energy(phi), e_rb, 64.0 * 5.0, "multi");
                let rb1 = RyckaertBellemans::from_series(&s).unwrap();
                assert_energy(rb1.to_series().energy(phi), e_rb, 64.0 * 5.0, "rb");
                // Through periodic the constant is the periodic form's own.
                let offset = rb2.to_series().constant() - s.constant();
                assert_energy(
                    rb2.to_series().energy(phi) - offset,
                    e_rb,
                    64.0 * 5.0,
                    "rb via periodic",
                );
            }
            // Every OPLS phase is 0° or 180°.
            for t in &periodic.terms {
                assert!(t.phase == 0.0 || t.phase == 180.0, "{t:?}");
            }
        }
    }

    /// Out-of-image inputs refuse, naming the term.
    #[test]
    fn out_of_image_inputs_refuse_with_the_reason() {
        let mut rng = StdRng::seed_from_u64(SEED + 3);
        for _ in 0..CASES {
            // A phase off {0°, 180°}: a sine term at its order.
            let n = rng.random_range(1..=4);
            let p = loop {
                let p: f64 = rng.random_range(-179.0..179.0);
                if p.abs() > 1e-3 {
                    break p;
                }
            };
            let s = CosineTerm {
                k: positive(&mut rng),
                periodicity: n as f64,
                phase: p,
            }
            .to_series()
            .unwrap();
            for target in ["opls", "multi/harmonic", "nharmonic", "harmonic"] {
                match TorsionForm::from_series("dihedral", target, &s) {
                    Err(TorsionRefusal::SineTerm { n: m, .. }) => assert_eq!(m, n),
                    other => panic!("{target}: {other:?}"),
                }
            }
            assert!(matches!(
                RyckaertBellemans::from_series(&s),
                Err(TorsionRefusal::SineTerm { .. })
            ));
            assert!(matches!(
                TorsionForm::from_series("improper", "cvff", &s),
                Err(TorsionRefusal::SineTerm { .. })
            ));

            // An order above the form's highest.
            let mut a6 = vec![0.0; 7];
            a6[6] = 1.0;
            let top = random_series(&mut rng, 5, false) + FourierSeries::new(a6, vec![]);
            assert_eq!(top.order(), 6);
            for (target, max) in [("opls", 4), ("multi/harmonic", 4), ("class2", 3)] {
                match TorsionForm::from_series("dihedral", target, &top) {
                    Err(TorsionRefusal::OrderTooHigh { n, max: m }) => {
                        assert_eq!(m, max);
                        assert!(n > max);
                    }
                    other => panic!("{target}: {other:?}"),
                }
            }
            assert!(matches!(
                RyckaertBellemans::from_series(&top),
                Err(TorsionRefusal::OrderTooHigh { max: 5, .. })
            ));

            // Two orders into a single-term form.
            let two = canonical_term(&mut rng, 1).to_series().unwrap()
                + canonical_term(&mut rng, 3).to_series().unwrap();
            for (category, style) in [
                ("dihedral", "charmm"),
                ("dihedral", "harmonic"),
                ("improper", "cvff"),
                ("improper", "periodic"),
            ] {
                assert_eq!(
                    TorsionForm::from_series(category, style, &two),
                    Err(TorsionRefusal::MultiTerm { orders: vec![1, 3] }),
                    "{category} {style}"
                );
            }
        }

        // Non-integer n, a sign that is not ±1, a non-Fourier form.
        let half = CosineTerm {
            k: 1.0,
            periodicity: 2.5,
            phase: 0.0,
        };
        assert_eq!(
            half.to_series(),
            Err(TorsionRefusal::NonIntegerPeriodicity { periodicity: 2.5 })
        );
        let bad = SignedCosine {
            k: 1.0,
            sign: 0.5,
            periodicity: 2.0,
        };
        assert_eq!(
            bad.to_series(),
            Err(TorsionRefusal::InvalidSign { sign: 0.5 })
        );
        let harmonic = Params::from_pairs(&[("k", 10.0), ("chi0", 0.0)]);
        assert!(matches!(
            TorsionForm::from_params("improper", "harmonic", &harmonic),
            Err(TorsionRefusal::NotAFourierForm {
                style: "improper harmonic",
                ..
            })
        ));
        assert!(matches!(
            TorsionForm::from_series("improper", "harmonic", &FourierSeries::zero()),
            Err(TorsionRefusal::NotAFourierForm { .. })
        ));
        assert!(matches!(
            TorsionForm::from_params("dihedral", "quadratic", &Params::new()),
            Err(TorsionRefusal::NotAFourierForm { .. })
        ));
        assert!(matches!(
            TorsionForm::from_params("dihedral", "mmff_torsion", &Params::new()),
            Err(TorsionRefusal::UnknownStyle { .. })
        ));
    }

    /// A 0°/180° phase is a cosine, exactly: no rounding residue in `bₙ`.
    #[test]
    fn phases_on_the_axes_are_exact() {
        for (phase, a, b) in [
            (0.0, 1.0, 0.0),
            (180.0, -1.0, 0.0),
            (-180.0, -1.0, 0.0),
            (540.0, -1.0, 0.0),
            (90.0, 0.0, 1.0),
            (-90.0, 0.0, -1.0),
        ] {
            let s = CosineTerm {
                k: 1.0,
                periodicity: 2.0,
                phase,
            }
            .to_series()
            .unwrap();
            assert_eq!((s.a(2), s.b(2)), (a, b), "phase {phase}");
        }
    }

    /// Rows on one quadruple sum into one series; the canonical form drops
    /// the constant.
    #[test]
    fn rows_on_one_quadruple_sum() {
        let rows = [
            TorsionForm::Periodic(Periodic {
                terms: vec![CosineTerm {
                    k: 1.0,
                    periodicity: 3.0,
                    phase: 0.0,
                }],
            }),
            TorsionForm::Charmm(Charmm {
                term: CosineTerm {
                    k: 0.5,
                    periodicity: 3.0,
                    phase: 180.0,
                },
                w: 0.0,
            }),
            TorsionForm::Opls(Opls {
                k: [2.0, 0.0, 0.0, 0.0],
            }),
        ];
        let s: FourierSeries = rows.iter().map(|r| r.to_series().unwrap()).sum();
        // a0 = 1 + 0.5 + 1; a1 = 1 (opls ½k1); a3 = 1 − 0.5.
        assert_eq!(s, FourierSeries::new(vec![2.5, 1.0, 0.0, 0.5], vec![]));
        assert_eq!(
            s.canonical(),
            FourierSeries::new(vec![0.0, 1.0, 0.0, 0.5], vec![])
        );
        let charmm =
            Charmm::from_series(&FourierSeries::new(vec![9.0, 0.0, 0.0, 0.5], vec![]), 1.0)
                .unwrap();
        assert_eq!(
            charmm,
            Charmm {
                term: CosineTerm {
                    k: 0.5,
                    periodicity: 3.0,
                    phase: 0.0
                },
                w: 1.0
            }
        );
    }

    /// Periodic ↔ harmonic improper agree to second order about the minimum,
    /// with LAMMPS's un-halved K = n²k/2, and differ at fourth order by
    /// −k n⁴ δ⁴/24.
    #[test]
    fn periodic_and_harmonic_impropers_agree_to_second_order() {
        // AMBER's improper: k[1 + cos(2φ − 180°)] = k[1 − cos 2φ], minimum at 0.
        let amber = CosineTerm {
            k: 1.1,
            periodicity: 2.0,
            phase: 180.0,
        };
        let h = amber.second_order_harmonic().unwrap();
        assert_eq!(h, ImproperHarmonic { k: 2.2, chi0: 0.0 }, "K = 2k at n = 2");
        assert_eq!(h.second_order_periodic(2.0), Ok(amber));

        let mut rng = StdRng::seed_from_u64(SEED + 4);
        for _ in 0..CASES {
            let n = rng.random_range(1..=6);
            let t = CosineTerm {
                k: coeff(&mut rng),
                periodicity: n as f64,
                phase: phase(&mut rng),
            };
            let h = t.second_order_harmonic().unwrap();
            assert!((0.0..=180.0).contains(&h.chi0), "{h:?}");
            assert!((h.k - t.k.abs() * (n * n) as f64 / 2.0).abs() < 1e-15 * h.k);
            // The term's minimum is at +χ0 or −χ0; the harmonic's at both.
            let chi0 = h.chi0.to_radians();
            let side = if t.energy(chi0) <= t.energy(-chi0) {
                1.0
            } else {
                -1.0
            };
            let e_min = t.energy(side * chi0);
            // About it: E_p − E_p(min) = Kδ² − |k| n⁴ δ⁴/24 + O(δ⁶).
            let delta = 1e-2;
            for d in [delta, -delta] {
                if chi0 + d <= 0.0 {
                    continue;
                }
                let ep = t.energy(side * (chi0 + d)) - e_min;
                let eh = h.energy(side * (chi0 + d));
                let quartic = -t.k.abs() * (n as f64).powi(4) * d.powi(4) / 24.0;
                assert!(
                    (ep - eh - quartic).abs() < 1e-3 * quartic.abs() + 1e-15,
                    "{t:?} {h:?}: {ep} − {eh} vs {quartic}"
                );
            }
            // Back: a periodic term with the same minimum and curvature.
            let back = h.second_order_periodic(n as f64).unwrap();
            assert!((back.k - t.k.abs()).abs() < 1e-12 * t.k.abs());
            let back_h = back.second_order_harmonic().unwrap();
            assert!((back_h.k - h.k).abs() < 1e-12 * h.k);
            assert!((back_h.chi0 - h.chi0).abs() < 1e-9, "{h:?} {back_h:?}");
        }
        assert_eq!(
            CosineTerm {
                k: 1.0,
                periodicity: 0.0,
                phase: 0.0
            }
            .second_order_harmonic(),
            Err(TorsionRefusal::NoCurvature)
        );
        assert!(matches!(
            ImproperHarmonic { k: 1.0, chi0: 0.0 }.to_series(),
            Err(TorsionRefusal::NotAFourierForm { .. })
        ));
    }

    // ── the registered kernels price the series ──

    /// Every registered kernel of a form, evaluated through the compiler on a
    /// random 4-atom geometry, equals the series at that geometry's φ — so
    /// the algebra and the kernels cannot drift.
    #[test]
    fn registered_kernels_price_the_series() {
        use crate::ff::forcefield::ForceField;
        use crate::ff::potential::PotentialCompiler;
        use crate::ff::potential::geometry::compute_dihedral;
        use molrs::store::block::Block;
        use molrs::store::frame::Frame;
        use molrs::types::Idx;
        use ndarray::Array1;

        let mut rng = StdRng::seed_from_u64(SEED + 5);
        for _ in 0..40 {
            for (form, _) in random_forms(&mut rng) {
                // charmm w ≠ 0 is refused at compile time (no 1-4 pair kernel
                // yet); the series is w-free, so price the torsion at w = 0.
                let form = match form {
                    TorsionForm::Charmm(c) => TorsionForm::Charmm(Charmm { w: 0.0, ..c }),
                    f => f,
                };
                let (category, style) = form.style();
                let mut ff = ForceField::new("t");
                ff.def_style(category, style, Params::new())
                    .unwrap()
                    .def_type("t", &["a", "b", "c", "d"], form.to_params())
                    .unwrap();
                let mut block = Block::new();
                for (key, atom) in [("atomi", 0), ("atomj", 1), ("atomk", 2), ("atoml", 3)] {
                    block
                        .insert(key, Array1::from_vec(vec![atom as Idx]).into_dyn())
                        .unwrap();
                }
                block
                    .insert("type", Array1::from_vec(vec!["t".to_owned()]).into_dyn())
                    .unwrap();
                let mut frame = Frame::new();
                frame.insert(
                    if category == "dihedral" {
                        "dihedrals"
                    } else {
                        "impropers"
                    },
                    block,
                );
                let pots = PotentialCompiler::new(&ff)
                    .compile(&frame)
                    .unwrap_or_else(|e| panic!("{form:?}: {e}"));
                let s = form.to_series().unwrap();
                let scale = 64.0 * s.max_abs_diff(&FourierSeries::zero());
                for _ in 0..4 {
                    let coords: Vec<f64> = (0..12).map(|_| rng.random_range(-1.5..1.5)).collect();
                    let phi = compute_dihedral(&coords, 0, 1, 2, 3);
                    assert_energy(
                        pots.calc_energy(&coords),
                        s.energy(phi),
                        scale,
                        &format!("{category} {style} {form:?}"),
                    );
                }
            }
        }
    }
}
