//! Similarity-based missing-parameter estimator (force-field agnostic).
//!
//! [`ParameterEstimator`] fills bond / angle / dihedral parameters that a
//! force field does not cover, using the standard parmchk2-style analogy cascade
//! (exact → equivalent → corresponding → wildcard) plus a CGenFF-style additive
//! penalty (inner atoms weighted ×10) and, as a last resort, the GAFF empirical
//! fallback formulas (Badger bond `k`, mean-of-neighbours angle θ₀, the GAFF
//! Eq.5 angle `K_θ`, and a never-fabricate dihedral rule). **No ab-initio / QM
//! fitting is ever performed.**
//!
//! It is an opt-in no-match seam, not an eager pass: it implements the generic
//! [`ParameterInterpolator`] trait for the OPLS bonded-term query and is injected
//! via [`typify_bonded_with`](super::opls::assign::typify_bonded_with). Exact
//! matches always win first; with `strict=true` the interpolator is never
//! consulted; with no interpolator attached the assign path is byte-identical to
//! pre-interpolator behaviour.
//!
//! # Provenance convention (new)
//!
//! Every estimated term carries provenance written onto its [`Params`] so a
//! consumer can audit and tier the estimate:
//!
//! | key | type | meaning |
//! |---|---|---|
//! | `estimated` | numeric `1.0` | flag: this term was estimated, not matched |
//! | `estimate_penalty` | numeric | total CGenFF-style additive penalty (f64) |
//! | `estimate_method` | string | `"analogy"`, `"empirical"`, or `"generic-wildcard"` |
//! | `estimate_analog` | string | source type name copied from, or `""` |
//!
//! Penalty tiers (CGenFF bands): `< 10` reliable, `10–50` use with caution,
//! `> 50` poor (needs optimization). Inner atoms (the angle centre and the two
//! inner dihedral atoms) are weighted ×10, matching parmchk2's
//! `WEIGHT_BA_CTR` / `WEIGHT_TOR_CTR` and CGenFF's inner-atom weighting.
//!
//! # Units
//!
//! All inputs and outputs are in molrs internal units (Å, kcal/mol, radians, e).
//! Nearest-analog copying is unit-safe by construction (the analog's params come
//! from the same force-field table). The GAFF empirical formulas are calibrated
//! in AMBER units (Å, kcal/mol/rad²) which equal molrs units for bond/angle
//! force constants, so no extra conversion is applied — this is asserted by the
//! pinned empirical unit tests.

pub mod tables;

use std::collections::HashMap;
use std::str::FromStr;

use molrs::Element;

use crate::ff::forcefield::{ForceField, Params, StyleDefs};

use super::opls::assign::BondedTerm;
use super::opls::meta::OplsTypingMeta;
use tables::{EmpiricalTable, EquivTable};

/// 143.9 prefactor in the GAFF empirical angle force-constant formula
/// (Wang et al. 2004, Eq. 5). Units bake out to kcal/mol/rad².
const ANGLE_K_PREFACTOR: f64 = 143.9;

/// Generic interpolation seam for typifier parameter families.
///
/// `Term` is intentionally an associated type: OPLS bonded parameters use
/// [`BondedTerm`], while future typifiers can introduce their own query structs
/// for atom, pair, improper, stretch-bend, or charge-correction parameters
/// without changing this trait.
pub trait ParameterInterpolator {
    /// Query type understood by this interpolator.
    type Term;

    /// Interpolate parameters for `term`, or return `Ok(None)` to decline.
    fn interpolate(&self, term: &Self::Term) -> Result<Option<Params>, String>;
}

/// Reusable atom-type metadata used by parameter interpolation.
///
/// The estimator needs two pieces of typifier-side context: a type-to-class map
/// for class-keyed force fields and a type-to-element map for empirical
/// fallbacks. Keeping them in this context lets future typifiers construct the
/// same estimator without pretending to be OPLS.
#[derive(Debug, Clone, Default)]
pub struct TypifierParameterContext {
    type_to_class: HashMap<String, String>,
    type_to_element: HashMap<String, String>,
}

impl TypifierParameterContext {
    /// Create an empty interpolation context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a context from `(type_name, class_name)` pairs.
    pub fn from_type_classes<I, K, V>(classes: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut out = Self::new();
        for (name, class) in classes {
            out.insert_class(name, class);
        }
        out
    }

    /// Insert or replace a type-to-class entry.
    pub fn insert_class(&mut self, name: impl Into<String>, class: impl Into<String>) {
        self.type_to_class.insert(name.into(), class.into());
    }

    /// Insert or replace a type-to-element entry.
    pub fn insert_element(&mut self, name: impl Into<String>, element: impl Into<String>) {
        self.type_to_element.insert(name.into(), element.into());
    }

    /// Add element inference from force-field atom masses.
    pub fn with_forcefield_elements(mut self, ff: &ForceField) -> Self {
        self.type_to_element.extend(build_type_to_element(ff));
        self
    }

    fn class_of(&self, name: &str) -> Option<&str> {
        self.type_to_class.get(name).map(String::as_str)
    }

    fn element_of(&self, name: &str) -> Option<String> {
        self.type_to_element
            .get(name)
            .cloned()
            .or_else(|| element_from_token(name))
    }
}

/// Penalty tier for an estimate, following the CGenFF confidence bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyTier {
    /// `penalty < 10`: reliable.
    Reliable,
    /// `10 ≤ penalty ≤ 50`: use with caution, manual review advised.
    Caution,
    /// `penalty > 50`: poor, needs optimization.
    Poor,
}

impl PenaltyTier {
    /// Classify a total penalty into the CGenFF bands (`<10` / `10–50` / `>50`).
    /// The boundaries are inclusive on the lower band edge (`10.0 → Caution`,
    /// `50.0 → Caution`).
    pub fn of(penalty: f64) -> Self {
        if penalty < 10.0 {
            Self::Reliable
        } else if penalty <= 50.0 {
            Self::Caution
        } else {
            Self::Poor
        }
    }
}

/// How a missing parameter was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimateMethod {
    /// Copied verbatim from the nearest analog in the force-field table.
    Analogy,
    /// Computed from a GAFF empirical formula (Badger / mean-θ₀ / Eq.5).
    Empirical,
    /// Copied from a generic wildcard term (dihedral fallback).
    GenericWildcard,
}

impl EstimateMethod {
    /// The provenance string written onto the term (`estimate_method`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Analogy => "analogy",
            Self::Empirical => "empirical",
            Self::GenericWildcard => "generic-wildcard",
        }
    }
}

/// One pre-extracted candidate from a force-field bonded table: its type name,
/// endpoint patterns (class / type names), and the params to copy on a match.
struct AnalogCandidate {
    name: String,
    pattern: Vec<String>,
    params: Params,
}

/// Similarity-based missing-parameter estimator.
///
/// Constructed once from a [`ForceField`] + typifier metadata + the embedded
/// GAFF/parmchk2 constant tables. It caches the analog candidate tables and the
/// per-type element/class maps so each [`interpolate`](ParameterInterpolator::interpolate)
/// call is a cheap scan.
pub struct ParameterEstimator {
    /// Analog candidate tables (one per arity), scanned by the cascade.
    bonds: Vec<AnalogCandidate>,
    angles: Vec<AnalogCandidate>,
    dihedrals: Vec<AnalogCandidate>,
    /// Typifier-side type metadata used by the analogy and empirical fallback.
    context: TypifierParameterContext,
    /// GAFF Badger / angle empirical constants.
    empirical: EmpiricalTable,
    /// parmchk2 equivalent / corresponding substitution table + weights.
    equiv: EquivTable,
}

impl ParameterEstimator {
    /// Build an estimator from a force field + OPLS typing metadata.
    ///
    /// The analog candidate tables are read from the force field's
    /// `bond:harmonic` / `angle:harmonic` / `dihedral:opls` styles (the same
    /// source as chain-2's `CandidateTables`). The `type → class` map comes from
    /// `meta`; the `type → element` map is inferred from each atom type's
    /// tabulated mass (see [the element-sourcing note](#element-sourcing)).
    ///
    /// # Element sourcing
    ///
    /// The GAFF empirical fallback formulas need a per-atom **element**, but the
    /// OPLS reader keeps only `name` + `mass` per type (element dropped) and the
    /// typing metadata carries `class`, not element. Rather than plumb a new
    /// element channel through the reader, the estimator infers each type's
    /// element from its tabulated mass by nearest standard-atomic-mass match
    /// (`molrs::Element`). This is force-field agnostic — it works identically
    /// for GAFF — and the analogy cascade itself never needs an element (it
    /// works on type/class names); element is consulted only by the empirical
    /// last-resort fallback.
    pub fn new(ff: &ForceField, meta: &OplsTypingMeta) -> Self {
        let context = TypifierParameterContext::from_type_classes(
            meta.iter()
                .map(|(name, row)| (name.clone(), row.class.clone())),
        )
        .with_forcefield_elements(ff);

        Self::with_context(ff, context)
    }

    /// Build an estimator from a force field and an explicit interpolation
    /// context. This is the constructor future non-OPLS typifiers should use.
    pub fn with_context(ff: &ForceField, context: TypifierParameterContext) -> Self {
        let bonds = extract_candidates(ff, "bond", "harmonic");
        let angles = extract_candidates(ff, "angle", "harmonic");
        let dihedrals = extract_candidates(ff, "dihedral", "opls");

        Self {
            bonds,
            angles,
            dihedrals,
            context,
            empirical: EmpiricalTable::load(),
            equiv: EquivTable::load(),
        }
    }

    // -- public estimate entry points ---------------------------------------

    /// Estimate bond parameters (`k0`/`r0`) for an uncovered bond between two
    /// atom types, or `None` if nothing can be produced. Output is in molrs
    /// units (Å, kcal/mol/Å²) with provenance attached.
    pub fn estimate_bond(&self, types: &[String; 2]) -> Option<Params> {
        // Analogy cascade first (copy verbatim).
        if let Some(p) = self.analogy(
            &self.bonds,
            &[&types[0], &types[1]],
            &[false, false],
            "bond",
        ) {
            return Some(p);
        }
        // Empirical Badger fallback (needs both endpoint elements + a length).
        self.empirical_bond(types)
    }

    /// Estimate angle parameters (`k0`/`theta0`) for an uncovered angle, or
    /// `None`. Output in molrs units (radians, kcal/mol/rad²).
    pub fn estimate_angle(&self, types: &[String; 3]) -> Option<Params> {
        // Centre atom (index 1) is the inner atom → ×10 weighting.
        if let Some(p) = self.analogy(
            &self.angles,
            &[&types[0], &types[1], &types[2]],
            &[false, true, false],
            "angle",
        ) {
            return Some(p);
        }
        self.empirical_angle(types)
    }

    /// Estimate dihedral parameters for an uncovered dihedral, or `None`.
    ///
    /// Never fabricates a rigid barrier: prefers an analog (copying the whole
    /// multi-periodicity param group), then a generic wildcard term keyed on the
    /// two inner atoms, and finally a near-zero barrier with a high penalty.
    pub fn estimate_dihedral(&self, types: &[String; 4]) -> Option<Params> {
        // Inner two atoms (indices 1,2) weighted ×10.
        if let Some(p) = self.analogy(
            &self.dihedrals,
            &[&types[0], &types[1], &types[2], &types[3]],
            &[false, true, true, false],
            "torsion",
        ) {
            return Some(p);
        }
        self.generic_dihedral(types)
    }

    // -- analogy cascade ----------------------------------------------------

    /// Scan `table` for the lowest-penalty *specific* analog of `query`, copy
    /// its params verbatim, and attach provenance. `inner[i]` marks an inner
    /// atom (×10 weight). `arity` selects the penalty column
    /// (`"bond"`/`"angle"`/`"torsion"`). Wildcard-ended candidates are skipped —
    /// they are the generic-wildcard tier, handled separately (the cascade is
    /// exact → equivalent → corresponding → **wildcard**, so a true "analogy"
    /// copy is always a specific match). Returns `None` if no specific candidate
    /// of the right shape matches.
    fn analogy(
        &self,
        table: &[AnalogCandidate],
        query: &[&String],
        inner: &[bool],
        arity: &str,
    ) -> Option<Params> {
        let mut best: Option<(f64, &AnalogCandidate)> = None;
        for cand in table {
            if cand.pattern.len() != query.len() {
                continue;
            }
            // Skip wildcard-ended candidates — those belong to the generic tier.
            if cand.pattern.iter().any(|p| is_wildcard(p)) {
                continue;
            }
            let Some(penalty) = self.sequence_penalty(&cand.pattern, query, inner, arity) else {
                continue;
            };
            if best.is_none_or(|(p, _)| penalty < p) {
                best = Some((penalty, cand));
            }
        }
        let (penalty, cand) = best?;
        let mut params = cand.params.clone();
        write_provenance(&mut params, EstimateMethod::Analogy, penalty, &cand.name);
        Some(params)
    }

    /// Total substitution penalty of `pattern` against `query`, trying both
    /// orientations (bonded terms are reversal-symmetric) and taking the
    /// smaller. `None` if no orientation is a valid analog (some end is
    /// incompatible — different element and no tabulated correspondence).
    fn sequence_penalty(
        &self,
        pattern: &[String],
        query: &[&String],
        inner: &[bool],
        arity: &str,
    ) -> Option<f64> {
        let forward = self.oriented_penalty(pattern.iter(), query, inner, arity);
        let reversed = self.oriented_penalty(pattern.iter().rev(), query, inner, arity);
        match (forward, reversed) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Penalty of one orientation: sum of per-end substitution penalties with
    /// the inner-atom multiplier applied. `None` if any end is incompatible.
    fn oriented_penalty<'a, I>(
        &self,
        pattern: I,
        query: &[&String],
        inner: &[bool],
        arity: &str,
    ) -> Option<f64>
    where
        I: Iterator<Item = &'a String>,
    {
        let mut total = 0.0;
        for ((pat, q), is_inner) in pattern.zip(query.iter()).zip(inner.iter()) {
            let base = self.end_penalty(pat, q, arity)?;
            let mult = if *is_inner {
                self.inner_multiplier(arity)
            } else {
                1.0
            };
            total += base * mult;
        }
        Some(total)
    }

    /// Substitution penalty for matching one bonded-pattern endpoint `pattern`
    /// (a class / type / wildcard name) against one query atom type `q`.
    ///
    /// - exact type match or class match → `0` (the chain-2 exact / class tier);
    /// - wildcard (`""`/`"*"`/`"X"`) → small wildcard penalty;
    /// - equivalent type (parmchk2 `EQUA`) → `0` + the equtype weight;
    /// - corresponding type (parmchk2 `CORR`) → tabulated penalty × weight;
    /// - same element, no tabulated correspondence → the arity default penalty;
    /// - different element, nothing tabulated → `None` (incompatible).
    fn end_penalty(&self, pattern: &str, q: &str, arity: &str) -> Option<f64> {
        if is_wildcard(pattern) {
            return Some(self.wildcard_penalty(arity));
        }
        // exact type match.
        if pattern == q {
            return Some(0.0);
        }
        // class match (OPLS bonded forces key on class).
        if let Some(cls) = self.context.class_of(q)
            && pattern == cls
        {
            return Some(0.0);
        }
        // parmchk2 substitution table (GAFF type names) — query type vs pattern.
        if let Some(p) = self.substitution_penalty(q, pattern, arity) {
            return Some(p);
        }
        // element-based compatibility: same element ⇒ default penalty.
        let (qe, pe) = (self.element_of(q), self.element_of(pattern));
        match (qe, pe) {
            (Some(a), Some(b)) if a == b => Some(self.default_penalty(arity)),
            _ => None,
        }
    }

    /// Look up the parmchk2 substitution penalty for replacing query type `from`
    /// with pattern type `to` (or vice versa). Equivalent (`EQUA`) types add the
    /// equtype weight on a 0 base; corresponding (`CORR`) types use the
    /// tabulated per-arity penalty × weight (falling back to the default when the
    /// table stores `-1`).
    fn substitution_penalty(&self, from: &str, to: &str, arity: &str) -> Option<f64> {
        let try_one = |a: &str, b: &str| -> Option<f64> {
            let entry = self.equiv.entry(a)?;
            if entry.equa.iter().any(|e| e == b) {
                return Some(self.equiv.weights.equtype);
            }
            let row = entry.corr.iter().find(|r| r.to == b)?;
            let (raw, default, weight) = match arity {
                "bond" => (row.bond, self.equiv.defaults.bond, self.equiv.weights.bond),
                "angle" => (
                    row.angle,
                    self.equiv.defaults.angle,
                    self.equiv.weights.angle,
                ),
                "torsion" => (
                    row.torsion,
                    self.equiv.defaults.torsion,
                    self.equiv.weights.torsion,
                ),
                _ => (row.bond, self.equiv.defaults.bond, self.equiv.weights.bond),
            };
            let raw = if raw < 0.0 { default } else { raw };
            Some(raw * weight)
        };
        try_one(from, to).or_else(|| try_one(to, from))
    }

    /// Inner-atom penalty multiplier (CGenFF / parmchk2 `WEIGHT_*_CTR`).
    fn inner_multiplier(&self, arity: &str) -> f64 {
        match arity {
            "angle" => self.equiv.weights.angle_center_mult,
            "torsion" => self.equiv.weights.torsion_center_mult,
            _ => 1.0,
        }
    }

    /// Default per-arity substitution penalty when no correspondence is
    /// tabulated but the elements match.
    fn default_penalty(&self, arity: &str) -> f64 {
        match arity {
            "angle" => self.equiv.defaults.angle,
            "torsion" => self.equiv.defaults.torsion,
            _ => self.equiv.defaults.bond,
        }
    }

    /// Small wildcard-substitution penalty (a wildcard end matches anything but
    /// is less specific). Uses the per-arity default scaled down.
    fn wildcard_penalty(&self, arity: &str) -> f64 {
        // A wildcard end is a weak (high-penalty) analog source for bonds/angles
        // but the *normal* path for dihedrals; keep it modest and arity-scaled.
        match arity {
            "torsion" => 0.0,
            "angle" => self.equiv.defaults.angle,
            _ => self.equiv.defaults.bond,
        }
    }

    // -- empirical fallbacks ------------------------------------------------

    /// Badger empirical bond `k` (Wang2004 Eq.3): `K = exp(ln_Kij) / r^m`, with
    /// `r` the reference bond length for the element pair (the formula is
    /// evaluated at the equilibrium length, which is also returned as `r0`).
    /// `None` if either element is unknown or the pair is not tabulated.
    fn empirical_bond(&self, types: &[String; 2]) -> Option<Params> {
        let e1 = self.element_of(&types[0])?;
        let e2 = self.element_of(&types[1])?;
        let ln_kij = self.empirical.bond_lnk(&e1, &e2)?;
        let rref = self
            .empirical
            .bond_lnk
            .iter()
            .find(|r| (r.e1 == e1 && r.e2 == e2) || (r.e1 == e2 && r.e2 == e1))
            .map(|r| r.rref)?;
        let k0 = empirical_bond_k(ln_kij, rref, self.empirical.bond_power_m);
        let mut params = Params::from_pairs(&[("k0", k0), ("r0", rref)]);
        // Empirical estimates carry a high penalty (last resort).
        write_provenance(&mut params, EstimateMethod::Empirical, 60.0, "");
        Some(params)
    }

    /// Empirical angle: θ₀ = mean of the existing `A-B-A` and `C-B-C` angles
    /// sharing the centre `B`; `K_θ` from the GAFF Eq.5 formula. `None` if the
    /// neighbour angles or the Z/C factors / bond lengths are unavailable.
    fn empirical_angle(&self, types: &[String; 3]) -> Option<Params> {
        let (a, b, c) = (&types[0], &types[1], &types[2]);
        // θ₀ = mean of A-B-A and C-B-C (Wang2004).
        let theta_aba = self.existing_angle_theta0(a, b, a)?;
        let theta_cbc = self.existing_angle_theta0(c, b, c)?;
        let theta0 = empirical_angle_theta0(theta_aba, theta_cbc);

        // K_θ from elements + bond lengths + θ₀ (Eq.5).
        let (ea, eb, ec) = (
            self.element_of(a)?,
            self.element_of(b)?,
            self.element_of(c)?,
        );
        let zi = self.empirical.angle_z(&ea)?;
        let cj = self.empirical.angle_c(&eb)?;
        let zk = self.empirical.angle_z(&ec)?;
        let r_ab = self.empirical.bond_lnk.iter().find_map(|r| {
            ((r.e1 == ea && r.e2 == eb) || (r.e1 == eb && r.e2 == ea)).then_some(r.rref)
        })?;
        let r_bc = self.empirical.bond_lnk.iter().find_map(|r| {
            ((r.e1 == eb && r.e2 == ec) || (r.e1 == ec && r.e2 == eb)).then_some(r.rref)
        })?;
        let k0 = empirical_angle_k(zi, cj, zk, r_ab, r_bc, theta0);
        let mut params = Params::from_pairs(&[("k0", k0), ("theta0", theta0)]);
        write_provenance(&mut params, EstimateMethod::Empirical, 60.0, "");
        Some(params)
    }

    /// θ₀ (radians) of an existing `i-j-k` angle in the candidate table, if one
    /// matches by class/type in either orientation.
    fn existing_angle_theta0(&self, i: &str, j: &str, k: &str) -> Option<f64> {
        let q = [&i.to_string(), &j.to_string(), &k.to_string()];
        for cand in &self.angles {
            if cand.pattern.len() != 3 {
                continue;
            }
            let inner = [false, true, false];
            if self
                .sequence_penalty(&cand.pattern, &q, &inner, "angle")
                .is_some()
                && let Some(t0) = cand.params.get("theta0")
            {
                return Some(t0);
            }
        }
        None
    }

    /// Generic dihedral fallback: copy the most generic existing wildcard term
    /// keyed on the two inner atoms (whole multi-periodicity group), else emit a
    /// near-zero barrier with a high penalty — **never a fabricated barrier**.
    fn generic_dihedral(&self, types: &[String; 4]) -> Option<Params> {
        // Prefer a wildcard-ended dihedral `X-b-c-X` matching the inner two atoms.
        let q = [&types[0], &types[1], &types[2], &types[3]];
        let inner = [false, true, true, false];
        let mut best: Option<(f64, &AnalogCandidate)> = None;
        for cand in &self.dihedrals {
            if cand.pattern.len() != 4 {
                continue;
            }
            // Require the candidate to be a *generic* term (≥1 wildcard end).
            let has_wildcard = is_wildcard(&cand.pattern[0]) || is_wildcard(&cand.pattern[3]);
            if !has_wildcard {
                continue;
            }
            if let Some(p) = self.sequence_penalty(&cand.pattern, &q, &inner, "torsion")
                && best.is_none_or(|(bp, _)| p < bp)
            {
                best = Some((p, cand));
            }
        }
        if let Some((penalty, cand)) = best {
            let mut params = cand.params.clone();
            write_provenance(
                &mut params,
                EstimateMethod::GenericWildcard,
                penalty,
                &cand.name,
            );
            return Some(params);
        }

        // No analog, no generic term → near-zero barrier + HIGH penalty.
        // Copy the OPLS 4-cosine param shape with all coefficients ~0.
        let mut params = Params::from_pairs(&[("f1", 0.0), ("f2", 0.0), ("f3", 0.0), ("f4", 0.0)]);
        write_provenance(&mut params, EstimateMethod::GenericWildcard, 99.0, "");
        Some(params)
    }

    // -- helpers ------------------------------------------------------------

    /// Element symbol for an atom type: tries the type→element map (mass
    /// inference), then treats the name itself as an element symbol (GAFF lower
    /// types like `c3` reduce to `C`).
    fn element_of(&self, name: &str) -> Option<String> {
        self.context.element_of(name)
    }
}

impl ParameterInterpolator for ParameterEstimator {
    type Term = BondedTerm;

    /// Chain-2 seam: dispatch a missing bonded term to the right `estimate_*`.
    fn interpolate(&self, term: &BondedTerm) -> Result<Option<Params>, String> {
        Ok(match term {
            BondedTerm::Bond(t) => self.estimate_bond(t),
            BondedTerm::Angle(t) => self.estimate_angle(t),
            BondedTerm::Dihedral(t) => self.estimate_dihedral(t),
        })
    }
}

// ---------------------------------------------------------------------------
// Free functions (pinned by unit tests)
// ---------------------------------------------------------------------------

/// Badger empirical bond force constant (Wang2004 Eq.3):
/// `K_r = exp(ln_Kij) / r^m` (kcal/mol/Å²), `r` in Å.
fn empirical_bond_k(ln_kij: f64, r: f64, m: f64) -> f64 {
    ln_kij.exp() / r.powf(m)
}

/// Empirical equilibrium angle (Wang2004): the mean of the two shared-centre
/// reference angles `θ(A-B-A)` and `θ(C-B-C)`.
fn empirical_angle_theta0(theta_aba: f64, theta_cbc: f64) -> f64 {
    0.5 * (theta_aba + theta_cbc)
}

/// GAFF empirical angle force constant (Wang2004 Eq.5, parmchk2 source form):
///
/// ```text
/// K_θ = 143.9 · Z_i · C_j · Z_k · exp(-2·D) / (r_ij + r_jk) / sqrt(θ₀)
/// D   = (r_ij − r_jk)² / (r_ij + r_jk)²
/// ```
///
/// `θ₀` is in **radians** (the `/ sqrt(θ)` matches parmchk2's
/// `sqrt(angle·π/180)`), bond lengths in Å; result in kcal/mol/rad².
fn empirical_angle_k(zi: f64, cj: f64, zk: f64, r_ij: f64, r_jk: f64, theta0_rad: f64) -> f64 {
    let sum = r_ij + r_jk;
    let d = (r_ij - r_jk).powi(2) / sum.powi(2);
    ANGLE_K_PREFACTOR * zi * cj * zk * (-2.0 * d).exp() / sum / theta0_rad.sqrt()
}

/// Whether a bonded-type endpoint name is a wildcard (parity with chain-2's
/// `is_wildcard`: `""`, `"*"`, `"X"`).
fn is_wildcard(pattern: &str) -> bool {
    pattern.is_empty() || pattern == "*" || pattern == "X"
}

/// Write the provenance convention onto an estimated term's params.
fn write_provenance(params: &mut Params, method: EstimateMethod, penalty: f64, analog: &str) {
    params.set("estimated", 1.0);
    params.set("estimate_penalty", penalty);
    params.set_str("estimate_method", method.as_str());
    params.set_str("estimate_analog", analog);
}

/// Extract a bonded-style's type entries into analog candidates (name +
/// endpoint pattern + params). Empty if the style is absent.
fn extract_candidates(ff: &ForceField, category: &str, style: &str) -> Vec<AnalogCandidate> {
    match ff.get_style(category, style).map(|s| &s.defs) {
        Some(StyleDefs::Bond(types)) => types
            .iter()
            .map(|t| AnalogCandidate {
                name: t.name.clone(),
                pattern: vec![t.itom.clone(), t.jtom.clone()],
                params: t.params.clone(),
            })
            .collect(),
        Some(StyleDefs::Angle(types)) => types
            .iter()
            .map(|t| AnalogCandidate {
                name: t.name.clone(),
                pattern: vec![t.itom.clone(), t.jtom.clone(), t.ktom.clone()],
                params: t.params.clone(),
            })
            .collect(),
        Some(StyleDefs::Dihedral(types)) => types
            .iter()
            .map(|t| AnalogCandidate {
                name: t.name.clone(),
                pattern: vec![
                    t.itom.clone(),
                    t.jtom.clone(),
                    t.ktom.clone(),
                    t.ltom.clone(),
                ],
                params: t.params.clone(),
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Build a `type name → element symbol` map by inferring each atom type's
/// element from its tabulated mass (nearest standard atomic mass). FF-agnostic.
fn build_type_to_element(ff: &ForceField) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for at in ff.get_atomtypes() {
        if let Some(mass) = at.params.get("mass")
            && let Some(sym) = element_from_mass(mass)
        {
            out.insert(at.name.clone(), sym);
        }
    }
    out
}

/// Nearest standard-atomic-mass element symbol for a mass (amu). Returns `None`
/// for non-physical masses (≤ 0).
fn element_from_mass(mass: f64) -> Option<String> {
    if mass <= 0.0 {
        return None;
    }
    let mut best: Option<(f64, &'static str)> = None;
    for e in Element::ALL {
        let diff = (e.atomic_mass() as f64 - mass).abs();
        if best.is_none_or(|(d, _)| diff < d) {
            best = Some((diff, e.symbol()));
        }
    }
    best.map(|(_, s)| s.to_string())
}

/// Element symbol from an atom-type token (e.g. `c3` → `C`, `cl` → `Cl`).
///
/// GAFF lowercase atom types encode the element as the **leading letter**
/// (`c3`/`ca`/`cc` → C, `os`/`oh` → O), with only the genuine two-letter
/// halogens written two-letter (`cl` → Cl, `br` → Br). So the single leading
/// letter is tried first (correctly mapping `os` → O, not Osmium); the
/// two-letter form is the fallback for tokens whose single letter is not an
/// element. Type names that are real element symbols (`Cl`, `Br`) still resolve.
fn element_from_token(token: &str) -> Option<String> {
    let base: String = token
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if base.is_empty() {
        return None;
    }
    let title = |s: &str| -> String {
        let mut c = s.chars();
        let first = c.next().unwrap().to_ascii_uppercase();
        let rest: String = c.flat_map(|ch| ch.to_lowercase()).collect();
        format!("{first}{rest}")
    };
    // An explicitly title-cased multi-letter token (`Cl`, `Br`) is a real
    // element symbol — honour it before the GAFF leading-letter convention.
    if base.len() >= 2
        && base.chars().nth(1).is_some_and(|c| c.is_ascii_uppercase())
        && Element::from_str(&base).is_ok()
    {
        return Some(base);
    }
    // GAFF writes the genuine two-letter halogens lowercase (`cl`/`br`); these
    // must win over the leading-letter rule (which would read `cl` as carbon).
    let lower = base.to_ascii_lowercase();
    if lower == "cl" {
        return Some("Cl".to_string());
    }
    if lower == "br" {
        return Some("Br".to_string());
    }
    // GAFF convention: the leading letter is the element (`c3`/`os`/`hc`).
    let one = title(&base[..1]);
    if Element::from_str(&one).is_ok() {
        return Some(one);
    }
    // Fallback: any other genuine lowercase two-letter element.
    if base.len() >= 2 {
        let two = title(&base[..2]);
        if Element::from_str(&two).is_ok() {
            return Some(two);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- empirical formulas pinned against GAFF reference values -----------

    #[test]
    fn empirical_bond_k_matches_gaff_reference() {
        // ac-001: K = exp(ln_Kij)/r^4.5 reproduces gaff.dat force constants.
        // C-C: ln_Kij 7.643, r 1.5375 → ~300.9 (gaff.dat c3-c3 K=300.9).
        let cc = empirical_bond_k(7.643, 1.5375, 4.5);
        assert!((cc - 300.9).abs() / 300.9 < 1e-3, "C-C k {cc}");
        // C-H: ln_Kij 6.217, r 1.0969 → ~330.6 (gaff.dat c3-hc K=330.6).
        let ch = empirical_bond_k(6.217, 1.0969, 4.5);
        assert!((ch - 330.6).abs() / 330.6 < 1e-3, "C-H k {ch}");
    }

    #[test]
    fn empirical_angle_theta0_is_mean() {
        // ac-002: θ₀(A-B-C) = (θ(A-B-A) + θ(C-B-C)) / 2.
        let t = empirical_angle_theta0(1.90, 2.00);
        assert!((t - 1.95).abs() < 1e-6);
    }

    #[test]
    fn empirical_angle_k_matches_gaff_reference() {
        // ac-003: GAFF Eq.5 reproduces gaff.dat angle force constants.
        // c3-c3-c3: Z_C=1.183, C_C=1.339, Z_C=1.183, r=1.5375 (both), θ=111.51°.
        let theta = 111.51_f64.to_radians();
        let k = empirical_angle_k(1.183, 1.339, 1.183, 1.5375, 1.5375, theta);
        assert!((k - 62.9).abs() / 62.9 < 1e-3, "c3-c3-c3 K_θ {k}");
        // hc-c3-hc: Z_H=0.784, C_C=1.339, Z_H=0.784, r=1.0969 (both), θ=107.58°.
        let theta2 = 107.58_f64.to_radians();
        let k2 = empirical_angle_k(0.784, 1.339, 0.784, 1.0969, 1.0969, theta2);
        assert!((k2 - 39.4).abs() / 39.4 < 1e-3, "hc-c3-hc K_θ {k2}");
    }

    // --- penalty tiers (ac-006) --------------------------------------------

    #[test]
    fn penalty_tiers_classify_at_boundaries() {
        assert_eq!(PenaltyTier::of(0.0), PenaltyTier::Reliable);
        assert_eq!(PenaltyTier::of(9.999), PenaltyTier::Reliable);
        assert_eq!(PenaltyTier::of(10.0), PenaltyTier::Caution);
        assert_eq!(PenaltyTier::of(50.0), PenaltyTier::Caution);
        assert_eq!(PenaltyTier::of(50.0001), PenaltyTier::Poor);
        assert_eq!(PenaltyTier::of(100.0), PenaltyTier::Poor);
    }

    // --- element inference (element-sourcing decision) ---------------------

    #[test]
    fn element_from_mass_picks_nearest() {
        assert_eq!(element_from_mass(12.011).as_deref(), Some("C"));
        assert_eq!(element_from_mass(1.008).as_deref(), Some("H"));
        assert_eq!(element_from_mass(15.999).as_deref(), Some("O"));
        assert_eq!(element_from_mass(0.0), None);
    }

    #[test]
    fn element_from_token_reduces_gaff_types() {
        assert_eq!(element_from_token("c3").as_deref(), Some("C"));
        assert_eq!(element_from_token("hc").as_deref(), Some("H"));
        assert_eq!(element_from_token("cl").as_deref(), Some("Cl"));
        assert_eq!(element_from_token("os").as_deref(), Some("O"));
    }
}
