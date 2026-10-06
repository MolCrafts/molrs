//! OpenMM force-field XML writer — the inverse of
//! [`OplsXmlReader`](crate::ff::forcefield::readers::opls::OplsXmlReader).
//!
//! The schema is OpenMM's, and so are the units and factors: lengths in
//! **nm**, energies in **kJ/mol**, angles and phases in **radians**, harmonic
//! bonds and angles as `½k(x − x0)²`. The force-field IR follows LAMMPS's
//! definitions (`real`: Å, kcal/mol, degrees, un-halved `K`), so every value
//! is converted at this boundary, the exact inverse of the reader's table:
//!
//! | IR | OpenMM |
//! |---|---|
//! | `bond harmonic` | `<HarmonicBondForce>`, `k` × 2 × 418.4, `r0` ÷ 10 |
//! | `angle harmonic` | `<HarmonicAngleForce>`, `k` × 2 × 4.184, degrees → radians |
//! | `angle charmm` | the angle row above plus `<AmoebaUreyBradleyForce><UreyBradley k d>`, `k = k_ub` × 418.4 (OpenMM doubles it into a bond), `d = r_ub` ÷ 10 |
//! | `dihedral periodic`, `charmm` (`w = 0`), `harmonic` (`k[1 + d cos nφ]`: phase 0° / 180°), `class2` (`k[1 − cos(nφ − φₙ)]`: phase φₙ + 180°) | `<PeriodicTorsionForce><Proper>`, one term per term, `k` × 4.184 |
//! | `dihedral multi/harmonic`, `nharmonic` (N ≤ 6), `opls` | `<RBTorsionForce><Proper c0..c5>`, `Cₙ = (−1)ⁿ Aₙ₊₁` × 4.184 (OPLS through its series), constant included |
//! | `improper periodic` | `<PeriodicTorsionForce><Improper>`, stored `(i, j, k, l)` (centre `k`) written `class1 = k, class2 = i, class3 = j, class4 = l`: OpenMM prices `(c2, c3, c1, c4)` |
//! | `improper harmonic` | `<CustomTorsionForce energy="k*(theta-theta0)^2">` (every `chi0 = 0`, CHARMM's form) or `"k*(abs(theta)-theta0)^2"`, `<Improper>` in the stored order (`charmm` ordering prices it as written), `k` × 4.184 |
//! | `cmap charmm` | `<CMAPTorsionForce>`, OpenMM `(i, j)` = molrs `[(i + N/2) mod N][(j + N/2) mod N]` × 4.184; identical grids share one `<Map>` |
//! | `pair lj/cut` (no cross rows) + a Coulomb style | `<NonbondedForce>` (`combining_rule` on the root when not `arithmetic`) |
//! | `pair lj/charmm`, or `lj/cut` (`arithmetic`) with cross rows | `<LennardJonesForce>` (`sigma14` / `epsilon14`, cross rows as `<NBFixPair>`) beside a `<NonbondedForce>` holding the charges at `epsilon = 0` |
//!
//! `special_bonds` `[0, 0, s]` is `coulomb14scale` / `lj14scale`; charges are
//! the atom types' `charge` (none at all: `<UseAttributeFromResidue
//! name="charge"/>`). The Coulomb constant is OpenMM's own, as LAMMPS's is
//! LAMMPS's, and is not written. Placeholder atom types (`type_ = "*"`, the
//! reader's class stand-ins) are not written; the reader makes them again.
//! An endpoint is written `class{n}` when it is an atom class (or no atom
//! type names it), `type{n}` when it is the name of a type of another class,
//! and `""` is OpenMM's wildcard.
//!
//! # Refusals
//!
//! What OpenMM's tags cannot hold is an `Err` naming it, never a silent
//! approximation or a skipped style:
//!
//! - a style of any category outside the table (`bond morse`, `angle
//!   class2`, `dihedral mmff_torsion`, `improper cvff` — LAMMPS's `cvff`
//!   prices the dihedral with the centre first, which no OpenMM ordering
//!   does —, `pair buck`, …);
//! - a `dihedral charmm` type with `w ≠ 0` (OpenMM has no per-dihedral 1-4
//!   weight), an `nharmonic` past N = 6, a harmonic improper with a wildcard
//!   endpoint (OpenMM then re-orders it), a CMAP of odd size;
//! - `special_bonds` other than `[0, 0, s]`; a `lj/cut` `mixing` of
//!   `sixthpower`, or `geometric` with cross rows; a `lj/charmm` mixing other
//!   than `arithmetic`; a cross row with 1-4 parameters of its own (OpenMM
//!   prices an NBFIX 1-4 pair with the NBFIX row); a `lj/charmm` with
//!   `one_four = "regular"` (LAMMPS's semantics) whose types have their own
//!   1-4 parameters at a non-zero 1-4 weight (OpenMM would use them);
//! - a Coulomb style with `dielectric ≠ 1` or `delta ≠ 0`; charges beside no
//!   Coulomb style; charges on some types and not others.
//!
//! # Whole-FF serialization, not coefficient writing
//!
//! molrs has two kinds of force-field writer. This one is **whole-FF
//! serialization**: it writes every type the [`ForceField`] holds, as a
//! force-field file, and takes no type labels. **Coefficient writing**
//! ([`super::lammps::LammpsFfWriter`], LAMMPS only) answers "which coefficients
//! does this system's data file need" and is keyed by the system's
//! `TypeLabels`.

use std::collections::{BTreeSet, HashMap};

use super::ForceFieldWriter;
use crate::ff::forcefield::mixing::Mixing;
use crate::ff::forcefield::one_four::{OneFour, has_own_one_four};
use crate::ff::forcefield::readers::opls::{HARMONIC_IMPROPER_ABS, HARMONIC_IMPROPER_SIGNED};
use crate::ff::forcefield::torsion::{RyckaertBellemans, TorsionForm};
use crate::ff::forcefield::{ForceField, Params, Style, StyleDefs};
use crate::ff::potential::cmap::charmm::GRID;

/// kcal/mol → kJ/mol.
const KJ_PER_KCAL: f64 = 4.184;
/// Å → nm.
const NM_PER_ANGSTROM: f64 = 0.1;

/// Writer for OpenMM `<ForceField>` XML.
///
/// `precision`: decimals per number; `None` (the default) writes each number
/// in the shortest form that reads back to the same `f64`.
#[derive(Debug, Clone, Default)]
pub struct XmlForceFieldWriter {
    pub precision: Option<usize>,
}

impl XmlForceFieldWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_precision(mut self, precision: Option<usize>) -> Self {
        self.precision = precision;
        self
    }

    fn fmt_f(&self, v: f64) -> String {
        match self.precision {
            Some(p) => format!("{:.*}", p, v),
            None => format!("{v:?}"),
        }
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// How each endpoint label is spelled: `class{n}` or `type{n}`.
struct Endpoints {
    /// Real atom types whose class is another label, and that no type has
    /// as its class.
    typed: BTreeSet<String>,
}

impl Endpoints {
    fn new(ff: &ForceField) -> Self {
        let mut classes = BTreeSet::new();
        let mut names = Vec::new();
        for t in ff.get_atomtypes() {
            if t.params.get_str("type_") == Some("*") {
                continue;
            }
            let class = t.params.get_str("class");
            if let Some(c) = class {
                classes.insert(c.to_owned());
            }
            if class != Some(t.name.as_str()) {
                names.push(t.name.clone());
            }
        }
        let typed = names.into_iter().filter(|n| !classes.contains(n)).collect();
        Self { typed }
    }

    /// ` class1="…" class2="…"` (or `type{n}` per endpoint).
    fn attrs(&self, labels: &[&str]) -> String {
        let mut out = String::new();
        for (n, label) in labels.iter().enumerate() {
            let key = if !label.is_empty() && self.typed.contains(*label) {
                "type"
            } else {
                "class"
            };
            out.push_str(&format!(" {key}{}=\"{}\"", n + 1, esc(label)));
        }
        out
    }
}

/// `Err` naming a style of `category` this writer has no OpenMM form for.
fn refuse(style: &Style, why: &str) -> String {
    format!(
        "{} style `{}` has no OpenMM ForceField XML form{}",
        style.category(),
        style.name(),
        if why.is_empty() {
            String::new()
        } else {
            format!(": {why}")
        }
    )
}

fn need(p: &Params, key: &str, what: &str) -> Result<f64, String> {
    p.get(key).ok_or_else(|| format!("{what}: missing `{key}`"))
}

/// Sections of the output, in OpenMM's usual order.
#[derive(Default)]
struct Out {
    bonds: String,
    angles: String,
    urey_bradley: String,
    periodic: String,
    rb: String,
    custom: Vec<(f64, String)>,
    cmap_maps: Vec<Vec<f64>>,
    cmap_rows: String,
}

impl XmlForceFieldWriter {
    fn bonded(&self, ff: &ForceField, ends: &Endpoints, out: &mut Out) -> Result<(), String> {
        for style in ff.styles() {
            match (style.category(), &style.defs) {
                ("bond", StyleDefs::Bond(types)) => {
                    if style.name() != "harmonic" {
                        return Err(refuse(style, ""));
                    }
                    for t in types {
                        let what = format!("bond harmonic {}", t.name);
                        // LAMMPS K (kcal/mol/Å²) → OpenMM ½k, k = 2K (kJ/mol/nm²).
                        let k = 2.0 * need(&t.params, "k", &what)? * KJ_PER_KCAL
                            / (NM_PER_ANGSTROM * NM_PER_ANGSTROM);
                        let r0 = need(&t.params, "r0", &what)? * NM_PER_ANGSTROM;
                        out.bonds.push_str(&format!(
                            "    <Bond{} length=\"{}\" k=\"{}\"/>\n",
                            ends.attrs(&[&t.itom, &t.jtom]),
                            self.fmt_f(r0),
                            self.fmt_f(k)
                        ));
                    }
                }
                ("angle", StyleDefs::Angle(types)) => {
                    let ub = match style.name() {
                        "harmonic" => false,
                        "charmm" => true,
                        _ => return Err(refuse(style, "")),
                    };
                    for t in types {
                        let what = format!("angle {} {}", style.name(), t.name);
                        let labels = [t.itom.as_str(), &t.jtom, &t.ktom];
                        let k = 2.0 * need(&t.params, "k", &what)? * KJ_PER_KCAL;
                        let theta0 = need(&t.params, "theta0", &what)?.to_radians();
                        out.angles.push_str(&format!(
                            "    <Angle{} angle=\"{}\" k=\"{}\"/>\n",
                            ends.attrs(&labels),
                            self.fmt_f(theta0),
                            self.fmt_f(k)
                        ));
                        if ub {
                            if labels.iter().any(|l| l.is_empty()) {
                                return Err(format!(
                                    "{what}: OpenMM's Urey-Bradley row matches no wildcard"
                                ));
                            }
                            // OpenMM adds a bond of force constant 2k: k = K_ub.
                            let k_ub = need(&t.params, "k_ub", &what)? * KJ_PER_KCAL
                                / (NM_PER_ANGSTROM * NM_PER_ANGSTROM);
                            let d = need(&t.params, "r_ub", &what)? * NM_PER_ANGSTROM;
                            out.urey_bradley.push_str(&format!(
                                "    <UreyBradley{} k=\"{}\" d=\"{}\"/>\n",
                                ends.attrs(&labels),
                                self.fmt_f(k_ub),
                                self.fmt_f(d)
                            ));
                        }
                    }
                }
                ("dihedral", StyleDefs::Dihedral(types)) => {
                    for t in types {
                        let labels = [t.itom.as_str(), &t.jtom, &t.ktom, &t.ltom];
                        self.dihedral(style, &t.name, &t.params, &ends.attrs(&labels), out)?;
                    }
                }
                ("improper", StyleDefs::Improper(types)) => {
                    for t in types {
                        let what = format!("improper {} {}", style.name(), t.name);
                        match style.name() {
                            "periodic" => {
                                let (k, n, phase) = (
                                    need(&t.params, "k", &what)?,
                                    need(&t.params, "periodicity", &what)?,
                                    t.params.get("phase").unwrap_or(0.0),
                                );
                                // OpenMM prices (c2, c3, c1, c4); stored (i, j, k, l).
                                out.periodic.push_str(&format!(
                                    "    <Improper{}{}/>\n",
                                    ends.attrs(&[&t.ktom, &t.itom, &t.jtom, &t.ltom]),
                                    self.periodic_term(1, k, n, phase)
                                ));
                            }
                            "harmonic" => {
                                let labels = [t.itom.as_str(), &t.jtom, &t.ktom, &t.ltom];
                                if labels.iter().any(|l| l.is_empty()) {
                                    return Err(format!(
                                        "{what}: a wildcard endpoint makes OpenMM re-order the \
                                         improper, so it would price another dihedral"
                                    ));
                                }
                                let k = need(&t.params, "k", &what)? * KJ_PER_KCAL;
                                let chi0 = t.params.get("chi0").unwrap_or(0.0).to_radians();
                                out.custom.push((
                                    chi0,
                                    format!(
                                        "    <Improper{} k=\"{}\" theta0=\"{}\"/>\n",
                                        ends.attrs(&labels),
                                        self.fmt_f(k),
                                        self.fmt_f(chi0)
                                    ),
                                ));
                            }
                            "cvff" => {
                                return Err(refuse(
                                    style,
                                    "LAMMPS's cvff prices the dihedral I-J-K-L with I the \
                                     centre; OpenMM lists an improper's centre first and \
                                     prices (c2, c3, c1, c4), the AMBER periodic improper",
                                ));
                            }
                            _ => return Err(refuse(style, "")),
                        }
                    }
                }
                ("cmap", StyleDefs::Cmap(types)) => {
                    if style.name() != "charmm" {
                        return Err(refuse(style, ""));
                    }
                    for t in types {
                        let grid = t.params.get_array(GRID).ok_or_else(|| {
                            format!("cmap charmm {}: missing its `{GRID}` array", t.name)
                        })?;
                        let shape = grid.shape();
                        let n = shape[0];
                        if shape.len() != 2 || shape[1] != n || !n.is_multiple_of(2) || n < 2 {
                            return Err(format!(
                                "cmap charmm {}: grid of shape {shape:?}; OpenMM's map (origin \
                                 0) needs an even N×N grid to hold molrs's (origin −180°)",
                                t.name
                            ));
                        }
                        let mut values = vec![0.0; n * n];
                        for j in 0..n {
                            for i in 0..n {
                                values[i + n * j] =
                                    grid[[(i + n / 2) % n, (j + n / 2) % n]] * KJ_PER_KCAL;
                            }
                        }
                        let index = match out.cmap_maps.iter().position(|m| *m == values) {
                            Some(i) => i,
                            None => {
                                out.cmap_maps.push(values);
                                out.cmap_maps.len() - 1
                            }
                        };
                        out.cmap_rows.push_str(&format!(
                            "    <Torsion map=\"{index}\"{}/>\n",
                            ends.attrs(&[&t.itom, &t.jtom, &t.ktom, &t.ltom, &t.mtom])
                        ));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn periodic_term(&self, m: usize, k: f64, n: f64, phase_deg: f64) -> String {
        format!(
            " periodicity{m}=\"{}\" phase{m}=\"{}\" k{m}=\"{}\"",
            n,
            self.fmt_f(phase_deg.to_radians()),
            self.fmt_f(k * KJ_PER_KCAL)
        )
    }

    /// One dihedral type: OpenMM's periodic terms when the style is a sum of
    /// `k[1 + cos(nφ − γ)]` terms (constant included), RB otherwise.
    fn dihedral(
        &self,
        style: &Style,
        name: &str,
        p: &Params,
        attrs: &str,
        out: &mut Out,
    ) -> Result<(), String> {
        let what = format!("dihedral {} {name}", style.name());
        let form = TorsionForm::from_params("dihedral", style.name(), p)
            .map_err(|e| format!("{what}: {e}"))?;
        let terms: Option<Vec<(f64, f64, f64)>> = match &form {
            TorsionForm::Periodic(f) => Some(
                f.terms
                    .iter()
                    .map(|t| (t.k, t.periodicity, t.phase))
                    .collect(),
            ),
            TorsionForm::Charmm(f) => {
                if f.w != 0.0 {
                    return Err(format!(
                        "{what}: the 1-4 weight w = {} has no OpenMM form",
                        f.w
                    ));
                }
                Some(vec![(f.term.k, f.term.periodicity, f.term.phase)])
            }
            // k[1 + d cos nφ] = k[1 + cos(nφ − γ)], γ = 0° (d = 1) or 180° (d = −1).
            TorsionForm::Harmonic(f) => {
                let phase = match f.sign {
                    1.0 => 0.0,
                    -1.0 => 180.0,
                    d => return Err(format!("{what}: sign d = {d} is not ±1")),
                };
                Some(vec![(f.k, f.periodicity, phase)])
            }
            // k[1 − cos(nφ − φₙ)] = k[1 + cos(nφ − φₙ − 180°)].
            TorsionForm::Class2(f) => Some(
                f.k.iter()
                    .zip(&f.phi)
                    .enumerate()
                    .filter(|(_, (k, _))| **k != 0.0)
                    .map(|(i, (&k, &phi))| (k, (i + 1) as f64, phi + 180.0))
                    .collect(),
            ),
            _ => None,
        };
        if let Some(terms) = terms {
            let mut row = format!("    <Proper{attrs}");
            if terms.is_empty() {
                row.push_str(&self.periodic_term(1, 0.0, 1.0, 0.0));
            }
            for (m, (k, n, phase)) in terms.iter().enumerate() {
                row.push_str(&self.periodic_term(m + 1, *k, *n, *phase));
            }
            row.push_str("/>\n");
            out.periodic.push_str(&row);
            return Ok(());
        }
        // The polynomial forms: RB holds Σₙ₌₀⁵ Cₙ cosⁿ(φ − 180°), constant included.
        let series = form.to_series().map_err(|e| format!("{what}: {e}"))?;
        let rb = RyckaertBellemans::from_series(&series)
            .map_err(|e| format!("{what}: no RBTorsionForce form: {e}"))?;
        let mut row = format!("    <Proper{attrs}");
        for (n, c) in rb.c.iter().enumerate() {
            row.push_str(&format!(" c{n}=\"{}\"", self.fmt_f(c * KJ_PER_KCAL)));
        }
        row.push_str("/>\n");
        out.rb.push_str(&row);
        Ok(())
    }

    /// `<NonbondedForce>` and, when the van der Waals needs it,
    /// `<LennardJonesForce>`; also the root's `combining_rule`.
    fn nonbonded(&self, ff: &ForceField) -> Result<(String, Option<&'static str>), String> {
        let mut lj_style: Option<&Style> = None;
        let mut coul_style: Option<&Style> = None;
        for style in ff.get_styles("pair") {
            match style.name() {
                "lj/cut" | "lj/charmm" => {
                    if let Some(other) = lj_style {
                        return Err(format!(
                            "pair styles `{}` and `{}`: OpenMM holds one Lennard-Jones table",
                            other.name(),
                            style.name()
                        ));
                    }
                    lj_style = Some(style);
                }
                "coul/cut" | "coul/charmm" => {
                    if coul_style.is_some() {
                        return Err("two Coulomb pair styles".into());
                    }
                    let p = style.params();
                    if p.get("dielectric").is_some_and(|d| d != 1.0) {
                        return Err(refuse(style, "OpenMM's Coulomb has no dielectric"));
                    }
                    if p.get("delta").is_some_and(|d| d != 0.0) {
                        return Err(refuse(style, "a buffered Coulomb (delta ≠ 0)"));
                    }
                    coul_style = Some(style);
                }
                _ => return Err(refuse(style, "")),
            }
        }

        // Charges, per real atom type.
        let atoms: Vec<_> = ff
            .get_atomtypes()
            .into_iter()
            .filter(|t| t.params.get_str("type_") != Some("*"))
            .collect();
        let charged = atoms
            .iter()
            .filter(|t| t.params.get("charge").is_some())
            .count();
        if charged != 0 && charged != atoms.len() {
            return Err(
                "some atom types carry a charge and some do not: OpenMM takes every charge \
                 from <NonbondedForce> or every one from the residue templates"
                    .into(),
            );
        }
        let charge_of: HashMap<&str, f64> = atoms
            .iter()
            .filter_map(|t| t.params.get("charge").map(|q| (t.name.as_str(), q)))
            .collect();
        if coul_style.is_none() && charge_of.values().any(|&q| q != 0.0) {
            return Err(
                "atom types carry charges and the force field has no Coulomb pair style; \
                 OpenMM's NonbondedForce always prices charges"
                    .into(),
            );
        }
        if lj_style.is_none() && coul_style.is_none() {
            return Ok((String::new(), None));
        }

        let sb = ff.special_bonds();
        if sb.lj[0] != 0.0 || sb.lj[1] != 0.0 || sb.coul[0] != 0.0 || sb.coul[1] != 0.0 {
            return Err(format!(
                "special_bonds lj {:?} coul {:?}: OpenMM excludes 1-2 and 1-3 pairs and \
                 scales only 1-4",
                sb.lj, sb.coul
            ));
        }
        let (lj14, coul14) = (sb.lj_14(), sb.coul_14());

        // The van der Waals: per-type rows and cross rows.
        let mut own: Vec<(&str, &Params)> = Vec::new();
        let mut cross: Vec<(&str, &str, &Params)> = Vec::new();
        if let Some(style) = lj_style {
            let StyleDefs::Pair(types) = style.defs() else {
                unreachable!("a pair style holds pair types")
            };
            for t in types {
                if t.itom == t.jtom {
                    own.push((&t.itom, &t.params));
                } else {
                    cross.push((&t.itom, &t.jtom, &t.params));
                }
            }
        }
        let mixing = |style: &Style| -> Result<Mixing, String> {
            match style.params().get_str("mixing") {
                Some(m) => Mixing::parse(m),
                None => Ok(Mixing::UNDECLARED),
            }
        };
        let (use_lj_force, combining_rule) = match lj_style {
            None => (false, None),
            Some(style) if style.name() == "lj/charmm" => {
                if mixing(style)? != Mixing::Arithmetic {
                    return Err(refuse(style, "OpenMM's LennardJonesForce mixes arithmetic"));
                }
                if OneFour::of(style.params())? == OneFour::Regular
                    && lj14 != 0.0
                    && own.iter().any(|(_, p)| has_own_one_four(p))
                {
                    return Err(refuse(
                        style,
                        "its types carry epsilon14/sigma14 under one_four = \"regular\" \
                         (1-4 pairs at the regular epsilon/sigma, LAMMPS's semantics), which \
                         OpenMM's LennardJonesForce would replace by the 1-4 parameters",
                    ));
                }
                (true, None)
            }
            Some(style) => match (mixing(style)?, cross.is_empty()) {
                (Mixing::Arithmetic, true) => (false, None),
                (Mixing::Arithmetic, false) => (true, None),
                (Mixing::Geometric, true) => (false, Some("geometric")),
                (rule, _) => {
                    return Err(refuse(
                        style,
                        &format!(
                            "mixing {rule:?}{}: OpenMM mixes arithmetic (foyer's \
                             combining_rule adds geometric, without cross rows)",
                            if cross.is_empty() {
                                ""
                            } else {
                                " with cross rows"
                            }
                        ),
                    ));
                }
            },
        };

        let lj_of = |p: &Params, what: &str| -> Result<(f64, f64), String> {
            Ok((
                need(p, "sigma", what)? * NM_PER_ANGSTROM,
                need(p, "epsilon", what)? * KJ_PER_KCAL,
            ))
        };
        // Every type the NonbondedForce needs a row for.
        let mut nb_types: Vec<&str> = own.iter().map(|(t, _)| *t).collect();
        if lj_style.is_none() {
            nb_types = atoms.iter().map(|t| t.name.as_str()).collect();
        }

        let mut out = format!(
            "  <NonbondedForce coulomb14scale=\"{}\" lj14scale=\"{}\" \
             useDispersionCorrection=\"False\">\n",
            self.fmt_f(coul14),
            self.fmt_f(lj14)
        );
        if charge_of.is_empty() {
            out.push_str("    <UseAttributeFromResidue name=\"charge\"/>\n");
        }
        for ty in &nb_types {
            let charge = charge_of
                .get(ty)
                .map(|q| format!(" charge=\"{}\"", self.fmt_f(*q)))
                .unwrap_or_default();
            let (sigma, epsilon) = if use_lj_force || lj_style.is_none() {
                (1.0, 0.0)
            } else {
                let p = own.iter().find(|(t, _)| t == ty).unwrap().1;
                lj_of(p, &format!("pair {ty}"))?
            };
            out.push_str(&format!(
                "    <Atom type=\"{}\"{charge} sigma=\"{}\" epsilon=\"{}\"/>\n",
                esc(ty),
                self.fmt_f(sigma),
                self.fmt_f(epsilon)
            ));
        }
        out.push_str("  </NonbondedForce>\n");

        if use_lj_force {
            out.push_str(&format!(
                "  <LennardJonesForce lj14scale=\"{}\" useDispersionCorrection=\"False\">\n",
                self.fmt_f(lj14)
            ));
            for (ty, p) in &own {
                let what = format!("pair {ty}");
                let (sigma, epsilon) = lj_of(p, &what)?;
                let mut row = format!(
                    "    <Atom type=\"{}\" sigma=\"{}\" epsilon=\"{}\"",
                    esc(ty),
                    self.fmt_f(sigma),
                    self.fmt_f(epsilon)
                );
                if has_own_one_four(p) {
                    let s14 = p.get("sigma14").unwrap_or(sigma / NM_PER_ANGSTROM) * NM_PER_ANGSTROM;
                    let e14 = p.get("epsilon14").unwrap_or(epsilon / KJ_PER_KCAL) * KJ_PER_KCAL;
                    row.push_str(&format!(
                        " sigma14=\"{}\" epsilon14=\"{}\"",
                        self.fmt_f(s14),
                        self.fmt_f(e14)
                    ));
                }
                row.push_str("/>\n");
                out.push_str(&row);
            }
            for (a, b, p) in &cross {
                let what = format!("pair {a}-{b}");
                if has_own_one_four(p) {
                    return Err(format!(
                        "{what}: a cross row with 1-4 parameters of its own; OpenMM prices an \
                         NBFIX 1-4 pair with the NBFIX row"
                    ));
                }
                let (sigma, epsilon) = lj_of(p, &what)?;
                out.push_str(&format!(
                    "    <NBFixPair type1=\"{}\" type2=\"{}\" sigma=\"{}\" epsilon=\"{}\"/>\n",
                    esc(a),
                    esc(b),
                    self.fmt_f(sigma),
                    self.fmt_f(epsilon)
                ));
            }
            out.push_str("  </LennardJonesForce>\n");
        }
        Ok((out, combining_rule))
    }
}

impl ForceFieldWriter for XmlForceFieldWriter {
    fn write_str(&self, ff: &ForceField) -> Result<String, String> {
        let ends = Endpoints::new(ff);
        let mut sections = Out::default();
        self.bonded(ff, &ends, &mut sections)?;
        let (nonbonded, combining_rule) = self.nonbonded(ff)?;

        let mut out = String::from("<?xml version='1.0' encoding='utf-8'?>\n");
        out.push_str(&format!(
            "<ForceField name=\"{}\"{}>\n",
            esc(if ff.name.is_empty() {
                "MolPy"
            } else {
                &ff.name
            }),
            combining_rule
                .map(|r| format!(" combining_rule=\"{r}\""))
                .unwrap_or_default()
        ));

        // AtomTypes (the reader's class placeholders are not types).
        let mut atoms_xml = String::new();
        let mut sorted: Vec<_> = ff
            .get_atomtypes()
            .into_iter()
            .filter(|t| t.params.get_str("type_") != Some("*"))
            .collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        for t in sorted {
            let mut attrs = vec![format!("name=\"{}\"", esc(&t.name))];
            for (xml_key, key) in [
                ("class", "class"),
                ("element", "element"),
                ("mass", "mass"),
                ("def", "smarts"),
                ("desc", "desc"),
                ("doi", "doi"),
                ("overrides", "overrides"),
            ] {
                if let Some(v) = t.params.get_str(key) {
                    attrs.push(format!("{xml_key}=\"{}\"", esc(v)));
                } else if let Some(v) = t.params.get(key) {
                    attrs.push(format!("{xml_key}=\"{}\"", self.fmt_f(v)));
                }
            }
            atoms_xml.push_str(&format!("    <Type {}/>\n", attrs.join(" ")));
        }
        let mut section = |tag: &str, head: &str, body: &str| {
            if !body.is_empty() {
                out.push_str(&format!("  <{tag}{head}>\n{body}  </{tag}>\n"));
            }
        };
        section("AtomTypes", "", &atoms_xml);
        section("HarmonicBondForce", "", &sections.bonds);
        section("HarmonicAngleForce", "", &sections.angles);
        section("AmoebaUreyBradleyForce", "", &sections.urey_bradley);
        section("PeriodicTorsionForce", "", &sections.periodic);
        section("RBTorsionForce", "", &sections.rb);
        if !sections.custom.is_empty() {
            let energy = if sections.custom.iter().all(|(chi0, _)| *chi0 == 0.0) {
                HARMONIC_IMPROPER_SIGNED
            } else {
                HARMONIC_IMPROPER_ABS
            };
            let mut body = String::from(
                "    <PerTorsionParameter name=\"k\"/>\n    <PerTorsionParameter name=\"theta0\"/>\n",
            );
            for (_, row) in &sections.custom {
                body.push_str(row);
            }
            section(
                "CustomTorsionForce",
                &format!(" energy=\"{energy}\""),
                &body,
            );
        }
        if !sections.cmap_rows.is_empty() {
            let mut body = String::new();
            for map in &sections.cmap_maps {
                let values: Vec<String> = map.iter().map(|v| self.fmt_f(*v)).collect();
                body.push_str(&format!("    <Map>{}</Map>\n", values.join(" ")));
            }
            body.push_str(&sections.cmap_rows);
            section("CMAPTorsionForce", "", &body);
        }
        out.push_str(&nonbonded);
        out.push_str("</ForceField>\n");
        Ok(out)
    }
}

pub fn write_forcefield_xml(
    path: &str,
    ff: &ForceField,
    precision: Option<usize>,
) -> Result<(), String> {
    XmlForceFieldWriter::new()
        .with_precision(precision)
        .write(ff, path)
}

pub fn write_forcefield_xml_str(
    ff: &ForceField,
    precision: Option<usize>,
) -> Result<String, String> {
    XmlForceFieldWriter::new()
        .with_precision(precision)
        .write_str(ff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::readers::ForceFieldReader;
    use crate::ff::forcefield::readers::opls::OplsXmlReader;
    use crate::ff::forcefield::xml::read_forcefield_xml_str;
    use crate::ff::forcefield::{ForceField, Params, Style};

    fn style<'a>(ff: &'a ForceField, category: &str) -> &'a Style {
        ff.styles()
            .iter()
            .find(|s| s.category() == category)
            .expect(category)
    }

    fn type_params(style: &Style, name: &str) -> Params {
        style
            .defs
            .collect_type_params()
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p)
            .expect(name)
    }

    fn write(ff: &ForceField) -> String {
        write_forcefield_xml_str(ff, None).unwrap()
    }

    fn small_ff() -> ForceField {
        let mut ff = ForceField::new("tiny");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_pairs(&[("mass", 12.011)]))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_pairs(&[("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT-CT",
                &["CT", "CT", "CT"],
                Params::from_pairs(&[("k", 58.35), ("theta0", 112.7)]),
            )
            .unwrap();
        ff
    }

    fn close(got: f64, want: f64, what: &str) {
        assert!(
            (got - want).abs() <= 1e-13 * want.abs().max(1.0),
            "{what}: {got} vs {want}"
        );
    }

    #[test]
    fn what_is_written_reads_back_with_the_same_styles_and_parameters() {
        let ff = small_ff();
        let back = read_forcefield_xml_str(&write(&ff)).unwrap();
        assert_eq!(back.name, "tiny");
        let bond = style(&back, "bond");
        assert_eq!(bond.name, "harmonic");
        let bt = type_params(bond, "CT-CT");
        close(bt.get("k").unwrap(), 268.0, "k");
        close(bt.get("r0").unwrap(), 1.529, "r0");
        let at = type_params(style(&back, "angle"), "CT-CT-CT");
        close(at.get("theta0").unwrap(), 112.7, "theta0");
        close(at.get("k").unwrap(), 58.35, "k");
    }

    /// The torsion series of the only dihedral type of `ff`.
    fn series(ff: &ForceField) -> crate::ff::forcefield::torsion::FourierSeries {
        let s = style(ff, "dihedral");
        let p = &s.defs.collect_type_params()[0].1;
        TorsionForm::from_params("dihedral", s.name(), p)
            .unwrap()
            .to_series()
            .unwrap()
    }

    /// OPLS goes out as RB and reads back as `multi/harmonic`: the same
    /// function of φ, constant included.
    #[test]
    fn opls_torsions_and_pairs_round_trip_through_the_openmm_units() {
        let mut ff = ForceField::new("opls");
        ff.def_style("dihedral", "opls", Params::new())
            .unwrap()
            .def_type(
                "CT-CT-CT-CT",
                &["CT", "CT", "CT", "CT"],
                Params::from_pairs(&[("k1", 1.3), ("k2", -0.05), ("k3", 0.2), ("k4", 0.0)]),
            )
            .unwrap();
        let mut lj = Params::new();
        lj.set_str("mixing", "geometric");
        ff.def_style("pair", "lj/cut", lj)
            .unwrap()
            .def_type(
                "opls_135",
                &["opls_135"],
                Params::from_pairs(&[("sigma", 3.5), ("epsilon", 0.066)]),
            )
            .unwrap();
        let xml = write(&ff);
        assert!(xml.contains(r#"combining_rule="geometric""#), "{xml}");
        let back = read_forcefield_xml_str(&xml).unwrap();
        assert_eq!(style(&back, "dihedral").name(), "multi/harmonic");
        for phi in [-2.0, 0.3, 1.9] {
            close(series(&back).energy(phi), series(&ff).energy(phi), "E(φ)");
        }
        let pt = type_params(back.get_style("pair", "lj/cut").unwrap(), "opls_135");
        close(pt.get("sigma").unwrap(), 3.5, "sigma");
        close(pt.get("epsilon").unwrap(), 0.066, "epsilon");
        assert_eq!(
            back.get_style("pair", "lj/cut")
                .unwrap()
                .params()
                .get_str("mixing"),
            Some("geometric")
        );
    }

    /// The `c0..c5` attributes of the only `<Proper>` under `<RBTorsionForce>`.
    fn rb_coefficients(xml: &str) -> [f64; 6] {
        let doc = roxmltree::Document::parse(xml).expect("well-formed XML");
        let propers: Vec<_> = doc
            .descendants()
            .filter(|n| n.has_tag_name("Proper"))
            .filter(|n| n.parent().is_some_and(|p| p.has_tag_name("RBTorsionForce")))
            .collect();
        assert_eq!(propers.len(), 1, "{xml}");
        let mut c = [0.0; 6];
        for (i, ci) in c.iter_mut().enumerate() {
            let raw = propers[0]
                .attribute(format!("c{i}").as_str())
                .unwrap_or_else(|| panic!("missing c{i}: {xml}"));
            *ci = raw.parse().expect("numeric coefficient");
        }
        c
    }

    fn one_torsion(style_name: &str, params: Params) -> ForceField {
        let mut ff = ForceField::new("t");
        ff.def_style("dihedral", style_name, Params::new())
            .unwrap()
            .def_type("HC-CT-CT-HC", &["HC", "CT", "CT", "HC"], params)
            .unwrap();
        ff
    }

    /// k3 = 0.3 kcal/mol → F3 = 1.2552 kJ/mol: C0 = ½F3 = 0.6276,
    /// C1 = 1.5·F3 = 1.8828, C3 = −2·F3 = −2.5104, C2 = C4 = C5 = 0 (kJ/mol).
    #[test]
    fn opls_fourier_torsion_is_written_as_rb_in_kj() {
        let ff = one_torsion(
            "opls",
            Params::from_pairs(&[("k1", 0.0), ("k2", 0.0), ("k3", 0.3), ("k4", 0.0)]),
        );
        let c = rb_coefficients(&write(&ff));
        let want = [0.6276, 1.8828, 0.0, -2.5104, 0.0, 0.0];
        for (i, (got, want)) in c.iter().zip(want).enumerate() {
            assert!((got - want).abs() < 1e-12, "c{i}: got {got}, want {want}");
        }
    }

    /// `multi/harmonic` and `nharmonic` (N = 6) are RB with `Cₙ = (−1)ⁿ Aₙ₊₁`,
    /// and read back as the same style, parameter for parameter.
    #[test]
    fn polynomial_torsions_round_trip_through_rb() {
        for (name, a) in [
            ("multi/harmonic", vec![0.18, -0.54, 0.1, 0.72, -0.3]),
            ("nharmonic", vec![0.18, -0.54, 0.1, 0.72, -0.3, 0.25]),
        ] {
            let mut p = Params::new();
            for (i, v) in a.iter().enumerate() {
                p.set(&format!("a{}", i + 1), *v);
            }
            let ff = one_torsion(name, p);
            let xml = write(&ff);
            let c = rb_coefficients(&xml);
            close(c[1], 0.54 * 4.184, "C1 = −A2");
            let back = read_forcefield_xml_str(&xml).unwrap();
            let bp = type_params(back.get_style("dihedral", name).unwrap(), "HC-CT-CT-HC");
            for (i, v) in a.iter().enumerate() {
                close(bp.get(&format!("a{}", i + 1)).unwrap(), *v, name);
            }
        }
    }

    /// `harmonic` (`k[1 + d cos nφ]`), `class2` and `charmm` (w = 0) are
    /// periodic terms, constant included, so the energy at every φ is kept.
    #[test]
    fn cosine_torsions_are_written_as_periodic_terms() {
        for (name, p) in [
            (
                "harmonic",
                Params::from_pairs(&[("k", 0.7), ("sign", -1.0), ("periodicity", 2.0)]),
            ),
            (
                "class2",
                Params::from_pairs(&[("k1", 0.4), ("phi1", 30.0), ("k3", -0.2), ("phi3", 0.0)]),
            ),
            (
                "charmm",
                Params::from_pairs(&[
                    ("k", 0.2),
                    ("periodicity", 3.0),
                    ("phase", 180.0),
                    ("w", 0.0),
                ]),
            ),
        ] {
            let ff = one_torsion(name, p);
            let back = read_forcefield_xml_str(&write(&ff)).unwrap();
            assert_eq!(style(&back, "dihedral").name(), "periodic");
            for phi in [-2.5, -0.4, 0.0, 1.1, 3.0] {
                close(series(&back).energy(phi), series(&ff).energy(phi), name);
            }
        }
    }

    fn improper_params(ff: &ForceField) -> Vec<(String, Params)> {
        ff.styles()
            .iter()
            .filter(|s| s.category() == "improper")
            .flat_map(|s| match &s.defs {
                StyleDefs::Improper(v) => v
                    .iter()
                    .map(|t| (format!("{}:{}", s.name, t.name), t.params.clone()))
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect()
    }

    /// A multi-term periodic proper is written in OpenMM's spelling and reads
    /// back term for term.
    #[test]
    fn periodic_propers_round_trip_term_for_term() {
        let terms = [
            ("k1", 0.15),
            ("periodicity1", 3.0),
            ("phase1", 0.0),
            ("k2", 0.25),
            ("periodicity2", 1.0),
            ("phase2", 180.0),
        ];
        let ff = one_torsion("periodic", Params::from_pairs(&terms));
        let back = read_forcefield_xml_str(&write(&ff)).unwrap();
        let p = type_params(style(&back, "dihedral"), "HC-CT-CT-HC");
        for (key, want) in terms {
            close(p.get(key).unwrap(), want, key);
        }
    }

    /// Impropers go out as OpenMM `<Improper>` rows under
    /// `<PeriodicTorsionForce>`: the stored AMBER order `C-O-N-CT` (centre `N`
    /// third) is written centre first, `class1="N" class2="C" class3="O"
    /// class4="CT"`, whose dihedral OpenMM prices as `(c2, c3, c1, c4)` =
    /// `C-O-N-CT`, and reads back as stored.
    #[test]
    fn a_periodic_improper_round_trips_through_openmm_order() {
        let mut ff = ForceField::new("amber");
        ff.def_style("improper", "periodic", Params::new())
            .unwrap()
            .def_type(
                "C-O-N-CT",
                &["C", "O", "N", "CT"],
                Params::from_pairs(&[("k", 10.5), ("periodicity", 2.0), ("phase", 180.0)]),
            )
            .unwrap();
        let xml = write(&ff);
        assert!(!xml.contains("PeriodicImproperForce"), "{xml}");
        assert!(
            xml.contains(r#"<Improper class1="N" class2="C" class3="O" class4="CT""#),
            "{xml}"
        );
        let back = read_forcefield_xml_str(&xml).unwrap();
        let got = improper_params(&back);
        assert_eq!(got.len(), 1, "{got:?}");
        let (name, p) = &got[0];
        assert_eq!(name, "periodic:C-O-N-CT");
        close(p.get("k").unwrap(), 10.5, "k");
        assert_eq!(p.get("periodicity"), Some(2.0));
        close(p.get("phase").unwrap(), 180.0, "phase");
    }

    /// LAMMPS's `cvff` prices the dihedral I-J-K-L with I the centre; OpenMM
    /// lists the centre first and prices `(c2, c3, c1, c4)`, so no OpenMM row
    /// prices a cvff improper.
    #[test]
    fn a_cvff_improper_is_refused() {
        let mut ff = ForceField::new("cvff");
        ff.def_style("improper", "cvff", Params::new())
            .unwrap()
            .def_type(
                "CA-CA-CA-HA",
                &["CA", "CA", "CA", "HA"],
                Params::from_pairs(&[("k", 1.1), ("sign", -1.0), ("periodicity", 2.0)]),
            )
            .unwrap();
        let err = write_forcefield_xml_str(&ff, None).unwrap_err();
        assert!(err.contains("cvff"), "{err}");
    }

    /// A `dihedral charmm` term with a 1-4 weight has no OpenMM form.
    #[test]
    fn a_charmm_dihedral_with_a_weight_is_refused() {
        let ff = one_torsion(
            "charmm",
            Params::from_pairs(&[("k", 0.2), ("periodicity", 3.0), ("phase", 0.0), ("w", 1.0)]),
        );
        let err = write_forcefield_xml_str(&ff, None).unwrap_err();
        assert!(err.contains("w = 1"), "{err}");
    }

    /// Styles with no OpenMM form are refused by name, never skipped.
    #[test]
    fn styles_without_an_openmm_form_are_refused_by_name() {
        let ff = one_torsion("mmff_torsion", Params::from_pairs(&[("v1", 1.0)]));
        assert!(
            write_forcefield_xml_str(&ff, None)
                .unwrap_err()
                .contains("mmff_torsion")
        );
        let mut ff = ForceField::new("x");
        ff.def_style("bond", "morse", Params::new())
            .unwrap()
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_pairs(&[("d0", 1.0), ("alpha", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        assert!(
            write_forcefield_xml_str(&ff, None)
                .unwrap_err()
                .contains("morse")
        );
        let mut ff = ForceField::new("x");
        ff.def_style("pair", "buck", Params::new())
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_pairs(&[("a", 1.0), ("rho", 1.0), ("c", 1.0)]),
            )
            .unwrap();
        assert!(
            write_forcefield_xml_str(&ff, None)
                .unwrap_err()
                .contains("buck")
        );
    }

    /// A harmonic improper is OpenMM's CHARMM `<CustomTorsionForce>`, in the
    /// stored order; `chi0 ≠ 0` takes the `abs(theta)` form, which LAMMPS's
    /// χ = |φ| is.
    #[test]
    fn harmonic_improper_round_trips_through_custom_torsion() {
        for (chi0, energy) in [
            (0.0, HARMONIC_IMPROPER_SIGNED),
            (12.0, HARMONIC_IMPROPER_ABS),
        ] {
            let mut ff = ForceField::new("charmm");
            ff.def_style("improper", "harmonic", Params::new())
                .unwrap()
                .def_type(
                    "C-O-N-CT",
                    &["C", "O", "N", "CT"],
                    Params::from_pairs(&[("k", 120.0), ("chi0", chi0)]),
                )
                .unwrap();
            let xml = write(&ff);
            assert!(xml.contains(&format!("energy=\"{energy}\"")), "{xml}");
            let got = improper_params(&read_forcefield_xml_str(&xml).unwrap());
            assert_eq!(got[0].0, "harmonic:C-O-N-CT");
            close(got[0].1.get("k").unwrap(), 120.0, "k");
            close(got[0].1.get("chi0").unwrap(), chi0, "chi0");
        }
    }

    /// An explicit LJ cross row is an `<NBFixPair>` of a `<LennardJonesForce>`
    /// (`arithmetic` mixing), beside a `<NonbondedForce>` at ε = 0; geometric
    /// mixing with a cross row has no OpenMM form.
    #[test]
    fn an_lj_cross_row_is_an_nbfix_pair() {
        let mut ff = ForceField::new("charmm");
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_pairs(&[("sigma", 3.0), ("epsilon", 0.1)]),
            )
            .unwrap()
            .def_type(
                "B",
                &["B"],
                Params::from_pairs(&[("sigma", 2.0), ("epsilon", 0.3)]),
            )
            .unwrap()
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_pairs(&[("sigma", 2.0), ("epsilon", 0.9)]),
            )
            .unwrap();
        let xml = write(&ff);
        assert!(xml.contains("<NBFixPair type1=\"A\" type2=\"B\""), "{xml}");
        let back = OplsXmlReader::new().read_str(&xml).unwrap();
        let lj = back.get_style("pair", "lj/charmm").expect("lj/charmm");
        let fix = type_params(lj, "A-B");
        close(fix.get("epsilon").unwrap(), 0.9, "epsilon");

        let mut geo = ff.clone();
        geo.get_style_mut("pair", "lj/cut")
            .unwrap()
            .set_str_param("mixing", "geometric");
        let err = write_forcefield_xml_str(&geo, None).unwrap_err();
        assert!(
            err.contains("Geometric") && err.contains("cross rows"),
            "{err}"
        );
    }

    /// `lj/charmm` under LAMMPS's semantics (`one_four` regular) with its own
    /// 1-4 parameters at a non-zero 1-4 weight: OpenMM would price the 1-4
    /// pairs with them, so it is refused.
    #[test]
    fn regular_one_four_with_own_14_parameters_is_refused() {
        let mut ff = ForceField::new("x");
        let mut sp = Params::new();
        sp.set_str("mixing", "arithmetic");
        ff.def_style("pair", "lj/charmm", sp)
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_pairs(&[
                    ("sigma", 3.0),
                    ("epsilon", 0.1),
                    ("sigma14", 2.5),
                    ("epsilon14", 0.05),
                ]),
            )
            .unwrap();
        ff.set_special_bonds(crate::ff::forcefield::SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.5],
        });
        let err = write_forcefield_xml_str(&ff, None).unwrap_err();
        assert!(err.contains("one_four"), "{err}");
        ff.get_style_mut("pair", "lj/charmm")
            .unwrap()
            .set_str_param("one_four", "epsilon14");
        let xml = write(&ff);
        assert!(xml.contains("sigma14=\"0.25\""), "{xml}");
    }

    #[test]
    fn the_precision_bounds_the_written_decimals() {
        let ff = small_ff();
        // r0 = 1.529 Å is written as 0.1529 nm.
        let coarse = write_forcefield_xml_str(&ff, Some(2)).unwrap();
        assert!(coarse.contains("length=\"0.15\""), "{coarse}");
        let fine = write_forcefield_xml_str(&ff, Some(4)).unwrap();
        assert!(fine.contains("length=\"0.1529\""), "{fine}");
    }
}
