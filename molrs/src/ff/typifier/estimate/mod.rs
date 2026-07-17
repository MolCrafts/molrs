//! The missing-parameter estimator: one cascade, one seam, every force field.
//!
//! A force field's tables never cover every term of every molecule. What fills the
//! gap is [`Parmchk2Estimator`] — the parmchk2 analogy cascade (exact →
//! equivalent → **wildcard row** → corresponding, plus a CGenFF-style additive
//! penalty with inner atoms weighted ×10) backed by the GAFF empirical formulas
//! (Badger bond `k`, mean-of-neighbours θ₀, the Wang2004 Eq. 5 angle `K_θ`, and a
//! never-fabricate rule for torsions). **No ab-initio / QM fitting is ever
//! performed.**
//!
//! It reaches its callers two ways, and they are the same object:
//!
//! * **as an interpolation seam** — it implements [`ParameterInterpolator`] for
//!   [`BondedTerm`] and is injected into the OPLS bonded matcher via
//!   [`typify_bonded_with`](super::opls::assign::typify_bonded_with). Exact matches
//!   always win first; with `strict=true` the interpolator is never consulted; with
//!   none attached the assign path is byte-identical to pre-interpolator behaviour.
//! * **as a table reader** — [`forcefield::gaff`](crate::ff::forcefield::gaff)
//!   builds it over `gaff.dat` / `gaff2.dat` and asks it for every term the tables'
//!   wildcard-free rows do not cover. That path needs the
//!   [`Covered`](Estimate::Covered) / [`Estimated`](Estimate::Estimated)
//!   distinction, so it consumes [`Estimate`] directly rather than through the seam.
//!
//! # The tables are GAFF's. That is a limitation, and it is deliberate.
//!
//! Every constant the cascade scores with comes from AmberTools' GAFF data: the
//! atom-type equivalences and correspondences (`PARMCHK.DAT`), the penalty weights
//! and defaults (its `WEIGHT_*` / `DEFAULT_*` block), and the empirical bond /
//! angle constants (`PARM_BLBA_GAFF*.DAT`). They are keyed by **GAFF atom-type
//! names** — `c3`, `os`, `ca`.
//!
//! A GAFF-typed molecule therefore gets the full cascade. **Any other force field
//! borrows it and degrades**: `opls_135` appears in no row of the substitution
//! table, so no equivalence and no correspondence can ever be found for it, and the
//! estimator falls back on what it *can* still say — a type-name / class-name match
//! (penalty 0), element compatibility at the arity's default penalty, and the
//! element-keyed empirical formulas. That is a real floor, and an honest one: an
//! OPLS estimate is a coarser thing than a GAFF estimate, and its penalty says so.
//!
//! # Provenance
//!
//! Every estimated term carries the four provenance keys of
//! [`Provenance::write_onto`] (`estimated`, `estimate_penalty`, `estimate_method`,
//! `estimate_analog`) so a consumer can audit and tier it. A term a wildcard row
//! *covers* carries none of them — it is a parameter, not an estimate.
//!
//! # Units
//!
//! Angles (θ₀, phases) are **radians**, lengths Å — molrs's own conventions, and
//! what a candidate table must present. Force constants are copied **verbatim** in
//! the candidate table's own convention, and the empirical formulas produce the
//! `E = K·(x − x₀)²` constant that `gaff.dat` itself tabulates: see
//! [`empirical`] for why the ½ belongs to the consumer, not to the estimate.

pub mod candidate;
mod cascade;
pub mod empirical;
pub mod provenance;
pub mod tables;
pub mod term;

use std::collections::HashMap;
use std::str::FromStr;

use molrs::Element;

use crate::ff::forcefield::{ForceField, Params};
use crate::ff::params::{EmpiricalTable, ParmchkTable};

use super::opls::meta::OplsTypingMeta;

pub use candidate::{Candidate, CandidateSet};
pub use cascade::DEFAULT_IMPROPER;
pub use provenance::{Estimate, EstimateMethod, PenaltyTier, Provenance};
pub use tables::EmpiricalSet;
pub use term::BondedTerm;

use cascade::Arity;

/// Generic interpolation seam for typifier parameter families.
///
/// `Term` is intentionally an associated type: bonded parameters use
/// [`BondedTerm`], while future typifiers can introduce their own query structs
/// for atom, pair, stretch-bend, or charge-correction parameters without changing
/// this trait.
pub trait ParameterInterpolator {
    /// Query type understood by this interpolator.
    type Term;

    /// Interpolate parameters for `term`, or return `Ok(None)` to decline.
    fn interpolate(&self, term: &Self::Term) -> Result<Option<Params>, String>;
}

/// Reusable atom-type metadata the cascade needs from a typifier.
///
/// Two pieces, both force-field agnostic: a type-to-class map for class-keyed
/// force fields (OPLS bonded forces key on `CT`, not on `opls_135`), and a
/// type-to-element map for the empirical fallbacks. Keeping them here is what lets
/// a non-OPLS typifier build the same estimator without pretending to be OPLS.
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

    /// Add element inference from a force field's atom masses.
    ///
    /// The empirical formulas need a per-atom **element**, but a force-field reader
    /// keeps only `name` + `mass` per type. Rather than plumb a new element channel
    /// through every reader, each type's element is inferred from its tabulated mass
    /// by nearest standard-atomic-mass match ([`molrs::Element`]). This is
    /// force-field agnostic, and the analogy cascade itself never needs an element —
    /// it works on type / class names.
    pub fn with_forcefield_elements(mut self, ff: &ForceField) -> Self {
        for at in ff.get_atomtypes() {
            if let Some(mass) = at.params.get("mass")
                && let Some(sym) = element_from_mass(mass)
            {
                self.type_to_element.insert(at.name.clone(), sym);
            }
        }
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

/// The missing-parameter estimator: parmchk2's cascade over one candidate table.
///
/// Built once from a force field plus typifier metadata, it caches the candidate
/// rows and the per-type class / element maps, so each
/// [`interpolate`](ParameterInterpolator::interpolate) call is a scan and nothing
/// more.
///
/// # It is named after parmchk2 because it *is* parmchk2's algorithm
///
/// The tiers, the penalty weights, the equivalence and correspondence tables and
/// the empirical formulas are all AmberTools' GAFF data and AmberTools' ordering,
/// and `parmchk2` is the oracle every one of them is checked against
/// (`tests/ff/typifier/parmchk2_oracle.rs`: 37 molecules × {gaff, gaff2} — the same
/// estimated-term set, the same values, the same confidence bands). A force field
/// that is not GAFF may still use it, and the OPLS typifier does, but it borrows
/// GAFF's tables to do so and degrades where they cannot speak its type names. See
/// the [module docs](self#the-tables-are-gaffs-that-is-a-limitation-and-it-is-deliberate).
pub struct Parmchk2Estimator {
    /// The rows the cascade scans, by arity.
    candidates: CandidateSet,
    /// Typifier-side type metadata (class + element).
    context: TypifierParameterContext,
    /// `PARMCHK.DAT`: equivalences, correspondences, penalty weights, and the
    /// improper-centre column.
    substitutions: ParmchkTable,
    /// The Badger / angle empirical constants of one force field.
    empirical: EmpiricalTable,
}

impl Parmchk2Estimator {
    /// Build an estimator from a force field + OPLS typing metadata.
    ///
    /// The `type → class` map comes from `meta`; the `type → element` map is
    /// inferred from each atom type's tabulated mass.
    pub fn new(ff: &ForceField, meta: &OplsTypingMeta) -> Self {
        let context = TypifierParameterContext::from_type_classes(
            meta.iter()
                .map(|(name, row)| (name.clone(), row.class.clone())),
        )
        .with_forcefield_elements(ff);

        Self::with_context(ff, context)
    }

    /// Build an estimator from a force field and an explicit interpolation context.
    ///
    /// This is the constructor a non-OPLS typifier uses — it is how
    /// [`forcefield::gaff`](crate::ff::forcefield::gaff) builds the estimator over
    /// `gaff.dat`.
    ///
    /// The candidate rows are flattened out of every bonded style the force field
    /// declares, by style **kind** and never by style *name*: GAFF's dihedral style
    /// is `periodic` and OPLS's is `opls`, and an extractor that asks for one by
    /// name is an extractor with an empty table for the other.
    pub fn with_context(ff: &ForceField, context: TypifierParameterContext) -> Self {
        Self {
            candidates: CandidateSet::from_forcefield(ff),
            context,
            substitutions: tables::substitution_table(),
            empirical: EmpiricalSet::Gaff.table(),
        }
    }

    /// Choose the empirical constant set (`PARM_BLBA_GAFF.DAT` vs `…GAFF2.DAT`).
    ///
    /// The two files differ, so a GAFF2 force field must say so; everything else
    /// keeps the GAFF set, whose constants are element-keyed and generic.
    pub fn with_empirical(mut self, set: EmpiricalSet) -> Self {
        self.empirical = set.table();
        self
    }

    /// May an atom of this type be the CENTRE of an improper at all?
    ///
    /// `PARMCHK.DAT`'s `improper_flag` column: `ca` and `c` and `na` carry a
    /// planarity term, `c3` and `n3` do not. That is upstream **data**, not a
    /// hybridisation the engine re-derives — and it is why benzene gets its
    /// ring-planarity improper while methylamine's sp3 nitrogen gets none.
    pub fn is_improper_centre(&self, atom_type: &str) -> bool {
        self.substitutions.is_improper_centre(atom_type)
    }

    // -- the cascade, as a table reader -------------------------------------

    /// Run the cascade for one term.
    ///
    /// The primitive both callers share, and the one that keeps the distinction
    /// that matters: [`Estimate::Covered`] means a row of the table covers the term
    /// outright (nothing estimated, nothing charged), [`Estimate::Estimated`] means
    /// it had to be reached by analogy or by formula. `None` means nothing could
    /// produce it — **no barrier is ever fabricated here**.
    pub fn estimate(&self, term: &BondedTerm) -> Option<Estimate> {
        match term {
            BondedTerm::Bond(types) => self.bond(types),
            BondedTerm::Angle(types) => self.angle(types),
            BondedTerm::Dihedral(types) => self.torsion(refs(types)),
            BondedTerm::Improper(types) => {
                Some(self.improper(&types[2], [&types[0], &types[1], &types[3]]))
            }
        }
    }

    fn bond(&self, types: &[String; 2]) -> Option<Estimate> {
        self.analogy(
            &self.candidates.bonds,
            &[&types[0], &types[1]],
            &[false, false],
            Arity::Bond,
        )
        .or_else(|| self.empirical_bond(types))
    }

    fn angle(&self, types: &[String; 3]) -> Option<Estimate> {
        // The vertex (index 1) is the inner atom → ×10 weighting.
        self.analogy(
            &self.candidates.angles,
            &[&types[0], &types[1], &types[2]],
            &[false, true, false],
            Arity::Angle,
        )
        .or_else(|| self.empirical_angle(types))
    }

    // -- the cascade, as an interpolation seam ------------------------------

    /// Estimate bond parameters (`k0` / `r0`) for an uncovered bond, or `None`.
    pub fn estimate_bond(&self, types: &[String; 2]) -> Option<Params> {
        Some(self.bond(types)?.into_params())
    }

    /// Estimate angle parameters (`k0` / `theta0`, radians) for an uncovered angle,
    /// or `None`.
    pub fn estimate_angle(&self, types: &[String; 3]) -> Option<Params> {
        Some(self.angle(types)?.into_params())
    }

    /// Estimate dihedral parameters for an uncovered dihedral.
    ///
    /// Never fabricates a rigid barrier: an analog (the whole multi-periodicity
    /// group, copied as one), else a generic wildcard term, else a **near-zero
    /// barrier** carrying a poor-tier penalty that says there is no torsion here.
    pub fn estimate_dihedral(&self, types: &[String; 4]) -> Option<Params> {
        match self.torsion(refs(types)) {
            Some(estimate) => Some(estimate.into_params()),
            None => {
                let mut params = self.no_torsion();
                Provenance::wildcard(self.no_torsion_penalty(), "").write_onto(&mut params);
                Some(params)
            }
        }
    }

    /// Estimate improper parameters (`k` / `n` / `d`) for a planar centre, given the
    /// term in AMBER slot order (**centre third**).
    pub fn estimate_improper(&self, types: &[String; 4]) -> Option<Params> {
        Some(
            self.improper(&types[2], [&types[0], &types[1], &types[3]])
                .into_params(),
        )
    }

    /// Element symbol for an atom type: the typifier's map (mass inference), then
    /// the type name read as an element token (`c3` → C), then `PARMCHK.DAT`'s own
    /// atomic-number column.
    pub(crate) fn element_of(&self, name: &str) -> Option<String> {
        self.context
            .element_of(name)
            .or_else(|| self.substitutions.element(name).map(str::to_owned))
    }
}

impl ParameterInterpolator for Parmchk2Estimator {
    type Term = BondedTerm;

    /// The seam: dispatch a missing bonded term to the right estimate, with the
    /// provenance convention written onto the params.
    fn interpolate(&self, term: &BondedTerm) -> Result<Option<Params>, String> {
        Ok(match term {
            BondedTerm::Bond(t) => self.estimate_bond(t),
            BondedTerm::Angle(t) => self.estimate_angle(t),
            BondedTerm::Dihedral(t) => self.estimate_dihedral(t),
            BondedTerm::Improper(t) => self.estimate_improper(t),
        })
    }
}

/// Borrow a quartet of owned names as a quartet of `&str`.
fn refs(types: &[String; 4]) -> [&str; 4] {
    [&types[0], &types[1], &types[2], &types[3]]
}

/// Nearest standard-atomic-mass element symbol for a mass (amu). `None` for a
/// non-physical mass (≤ 0).
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
/// (`c3`/`ca`/`cc` → C, `os`/`oh` → O), with only the genuine two-letter halogens
/// written two-letter (`cl` → Cl, `br` → Br). So the single leading letter is tried
/// first (correctly mapping `os` → O, not Osmium); the two-letter form is the
/// fallback for tokens whose single letter is not an element. Type names that are
/// real element symbols (`Cl`, `Br`) still resolve.
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
        let first = c.next().unwrap_or_default().to_ascii_uppercase();
        let rest: String = c.flat_map(|ch| ch.to_lowercase()).collect();
        format!("{first}{rest}")
    };
    // An explicitly title-cased multi-letter token (`Cl`, `Br`) is a real element
    // symbol — honour it before the GAFF leading-letter convention.
    if base.len() >= 2
        && base.chars().nth(1).is_some_and(|c| c.is_ascii_uppercase())
        && Element::from_str(&base).is_ok()
    {
        return Some(base);
    }
    // GAFF writes the genuine two-letter halogens lowercase (`cl` / `br`); these
    // must win over the leading-letter rule (which would read `cl` as carbon).
    let lower = base.to_ascii_lowercase();
    if lower == "cl" {
        return Some("Cl".to_string());
    }
    if lower == "br" {
        return Some("Br".to_string());
    }
    // GAFF convention: the leading letter is the element (`c3` / `os` / `hc`).
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
    use crate::ff::forcefield::gaff::{GaffParameterSet, gaff_estimator};

    /// The estimator over `gaff.dat`, exactly as the force-field builder makes it.
    fn gaff() -> &'static Parmchk2Estimator {
        gaff_estimator(GaffParameterSet::Gaff)
    }

    fn types<const N: usize>(names: [&str; N]) -> [String; N] {
        names.map(str::to_owned)
    }

    // --- the tiers, in order (ac-004) --------------------------------------

    #[test]
    fn tier_wildcard_row_is_a_parameter_not_an_estimate() {
        // `X -c3-c3-X ` covers ethane's H-C-C-H: LEaP finds it, parmchk2 is silent.
        let estimate = gaff()
            .estimate(&BondedTerm::Dihedral(types(["hc", "c3", "c3", "hc"])))
            .expect("a match");
        assert!(
            estimate.provenance().is_none(),
            "a term the table covers is not an estimate"
        );
        assert!(estimate.params().get("k1").is_some());
    }

    #[test]
    fn tier_equivalent_type_substitutes_free_of_charge() {
        // gaff2's `ns` is `n`: the specific row `o-c-n -hn` covers `o-c-ns-hn`.
        let estimate = gaff_estimator(GaffParameterSet::Gaff2)
            .estimate(&BondedTerm::Dihedral(types(["o", "c", "ns", "hn"])))
            .expect("an equivalent-type match");
        let provenance = estimate.provenance().expect("estimated");
        assert_eq!(provenance.analog, "o-c-n-hn");
        assert_eq!(provenance.method, EstimateMethod::Analogy);
        assert!(
            provenance.penalty.abs() < 1e-12,
            "EQUA costs nothing, got {}",
            provenance.penalty
        );
    }

    #[test]
    fn tier_corresponding_type_substitution_is_scored() {
        // Thiophene's `cc-cd-ss-cd`: no row covers it, so `cd` stands in for `c2`
        // on the inner atom of the wildcard row `X -c2-ss-X `.
        let estimate = gaff()
            .estimate(&BondedTerm::Dihedral(types(["cc", "cd", "ss", "cd"])))
            .expect("a match");
        let provenance = estimate.provenance().expect("estimated");
        assert_eq!(provenance.analog, "X-c2-ss-X");
        assert!(
            (provenance.penalty - 232.0).abs() < 0.05,
            "parmchk2 charges 232.0, got {}",
            provenance.penalty
        );
        assert_eq!(provenance.tier(), PenaltyTier::Poor);
    }

    /// The empirical tier — the one the parmchk2 oracle **cannot** reach.
    ///
    /// Every bond and angle of all 37 oracle molecules is an exact hit in both
    /// tables, so nothing there exercises the Badger / Eq. 5 formulas: the oracle's
    /// green says nothing whatsoever about them, and no oracle case may be
    /// fabricated to pretend otherwise (`tests/ff/typifier/parmchk2_oracle.rs` says
    /// so in its own module docs). So the tier is driven here, directly.
    ///
    /// **H–Br is the hole.** `gaff.dat` has a bond row for `br` to every heavy atom
    /// it can reach and for `br-br` itself, but none for hydrogen bonded to bromine
    /// — no GAFF-typed molecule has one — so no row exists whose two ends are even
    /// the right *elements*, and the analogy tier cannot reach one by substitution.
    /// `PARM_BLBA_GAFF.DAT`, keyed by element rather than by atom type, does carry
    /// the H–Br pair. That is precisely the gap the empirical tier exists to fill.
    #[test]
    fn tier_empirical_formula_is_the_last_resort() {
        let estimator = gaff();

        let bond = estimator
            .estimate(&BondedTerm::Bond(types(["hc", "br"])))
            .expect("the empirical formula produces a bond");
        let provenance = bond.provenance().expect("estimated");
        assert_eq!(
            provenance.method,
            EstimateMethod::Empirical,
            "no row and no analog: the tier below analogy is the only one left"
        );
        assert_eq!(provenance.analog, "", "a formula has no row to name");
        assert_eq!(
            provenance.tier(),
            PenaltyTier::Caution,
            "an empirical bond is charged DEFAULT_BL — read it with care"
        );

        // The estimate must be Badger's rule (Wang2004 Eq. 3) evaluated on exactly
        // the H–Br row of the empirical table, at the equilibrium length it gives.
        let ln_k = estimator.empirical.bond_ln_k("H", "Br").expect("tabulated");
        let want_r = estimator
            .empirical
            .bond_length("H", "Br")
            .expect("tabulated");
        let want_k = empirical::bond_k(ln_k, want_r, estimator.empirical.bond_power);
        let r0 = bond.params().get("r0").expect("r0");
        let k0 = bond.params().get("k0").expect("k0");
        assert!((r0 - want_r).abs() < 1e-12, "r₀ is the reference length");
        assert!((k0 - want_k).abs() < 1e-9, "K = exp(ln Kij) / r^m");
        assert!(k0 > 0.0, "an empirical force constant is positive");
    }

    // --- impropers ---------------------------------------------------------

    #[test]
    fn benzene_gets_the_ring_planarity_improper_off_a_wildcard_row() {
        let estimate = gaff().improper("ca", ["ca", "ca", "ha"]);
        let provenance = estimate.provenance().expect("estimated");
        assert_eq!(provenance.analog, "X-X-ca-ha");
        assert_eq!(provenance.method, EstimateMethod::GenericWildcard);
        assert!((estimate.params().get("k").expect("k") - 1.1).abs() < 1e-12);
        assert!(
            (provenance.penalty - 6.0).abs() < 0.05,
            "two wildcards, 3.0 each"
        );
    }

    #[test]
    fn the_amide_improper_needs_a_planar_neighbour() {
        // N-methylacetamide's carbonyl: `n` is planar, so the 10.5 amide term applies.
        let amide = gaff().improper("c", ["c3", "o", "n"]);
        assert!((amide.params().get("k").expect("k") - 10.5).abs() < 1e-12);

        // Acetone's carbonyl has the same shape and NO planar neighbour, so
        // parmchk2 falls back on the default 1.1 rather than calling it an amide.
        let ketone = gaff().improper("c", ["c3", "c3", "o"]);
        let (barrier, ..) = DEFAULT_IMPROPER;
        assert!((ketone.params().get("k").expect("k") - barrier).abs() < 1e-12);
        assert!(
            ketone.provenance().is_some(),
            "a default is still an estimate"
        );
    }

    #[test]
    fn an_sp3_centre_carries_no_improper_at_all() {
        let estimator = gaff();
        assert!(estimator.is_improper_centre("ca"));
        assert!(estimator.is_improper_centre("c"));
        assert!(!estimator.is_improper_centre("c3"));
        assert!(!estimator.is_improper_centre("n3"), "methylamine's amine N");
    }

    // --- element inference -------------------------------------------------

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
