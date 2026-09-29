//! GROMACS force-field directive writer.
//!
//! The inverse of
//! [`GromacsTopFfReader`](crate::ff::forcefield::readers::gromacs::GromacsTopFfReader):
//! it writes a [`ForceField`] as GROMACS force-field **directives**, converting
//! molrs units (Å, kcal/mol, rad, e) to file units (nm, kJ/mol, degrees, e)
//! at this boundary only. No molecule section (`[ atoms ]`, `[ bonds ]`,
//! `[ angles ]`, `[ dihedrals ]`, `[ pairs ]`, …) is written: a force field
//! holds no molecule.
//!
//! # Output
//!
//! - **`[ defaults ]`** `1 <comb> yes <fudgeLJ> <fudgeQQ>`. comb is the
//!   `pair/lj/cut` style's `mixing` — `arithmetic` → 2, `geometric` → 3 — or,
//!   when none is declared, the rule an undeclared `lj/cut` is evaluated under
//!   (arithmetic, 2). fudgeLJ / fudgeQQ are the 1-4 special-bond weights.
//! - **`[ atomtypes ]`** `name [bond_type] [at.num] mass charge ptype V W`, one
//!   row per `atom/full` type: `mass` (amu), `charge` (e), `bond_type` and
//!   `atomic_number` from the atom type (choosing the 6-, 7- or 8-column form),
//!   `ptype` as declared or `A` (an `atom/full` type is a real atom), and
//!   V = σ/10 (nm), W = ε·4.184 (kJ/mol) from the type's `pair/lj/cut` self
//!   row.
//! - **`[ bondtypes ]` / `[ angletypes ]` / `[ dihedraltypes ]`**, the inverse
//!   of the reader's function-code map:
//!
//! | molrs style | Directive, funct | Columns (file units) |
//! |---|---|---|
//! | `bond/harmonic` | bondtypes 1 | b₀ = r0/10 nm; k_b = k·418.4 kJ/mol/nm² |
//! | `bond/morse` | bondtypes 3 | b₀ = r0/10 nm; D·4.184 kJ/mol; β = alpha·10 nm⁻¹ |
//! | `angle/harmonic` | angletypes 1 | θ₀ in degrees; k·4.184 kJ/mol/rad² |
//! | `dihedral/periodic` | dihedraltypes 1 | φ_s in degrees; k·4.184 kJ/mol; n |
//! | `improper/harmonic` | dihedraltypes 2 | ξ₀ = 0; k_ξ = 2·k·4.184 kJ/mol/rad² |
//! | `dihedral/opls` | dihedraltypes 3 | C₀..C₅ (kJ/mol) by the exact Fourier → Ryckaert–Bellemans relation |
//! | `improper/periodic` | dihedraltypes 4 | as `dihedral/periodic` |
//!
//! The empty-endpoint wildcard is written as `X`.
//!
//! A pair style's `cutoff` is a run setting (the .mdp's `rvdw` / `rcoulomb`),
//! not force-field data, so it is not written.
//!
//! `pair/coul/cut` has no directive: GROMACS takes Coulomb constants from the
//! run parameters, so only the constants the reader declares (the real-units
//! Coulomb constant, dielectric 1) are accepted, and nothing is written.
//!
//! # Refusals
//!
//! What GROMACS force-field directives cannot express is an `Err` naming it,
//! never a silent drop or an invented value:
//!
//! - any other style (e.g. `dihedral/charmm`, `dihedral/multi/harmonic`), and a
//!   multi-term `dihedral/periodic` (code 9 is not modelled);
//! - `sixthpower` mixing; a non-zero 1-2 or 1-3 special-bond weight;
//! - an atom type lacking `mass`, `charge` or its `lj/cut` self row; an
//!   explicit `lj/cut` cross row (that needs `[ nonbond_params ]`); an
//!   `lj/cut` self row whose type is not an `atom/full` type;
//! - a bonded type missing a parameter, carrying one with no column, or with
//!   an endpoint label that is neither an atom-type name nor a `bond_type`;
//!   `improper/harmonic` with `chi0 ≠ 0`.
//!
//! # Whole-FF serialization, not coefficient writing
//!
//! molrs has two kinds of force-field writer. This one is **whole-FF
//! serialization**: it writes every type the [`ForceField`] holds, as a
//! force-field file, and takes no type labels. **Coefficient writing**
//! ([`super::lammps::LammpsFfWriter`], LAMMPS only) answers "which coefficients
//! does this system's data file need" and is keyed by the system's
//! `TypeLabels`.

use std::collections::HashSet;

use super::ForceFieldWriter;
use crate::ff::constants::VACUUM_DIELECTRIC;
use crate::ff::forcefield::mixing::Mixing;
use crate::ff::forcefield::torsion::opls_to_rb;
use crate::ff::forcefield::{AtomType, ForceField, Params, Style};
use molrs::units::constants::COULOMB_REAL;

const KJ_PER_KCAL: f64 = 4.184;
const NM_TO_ANGSTROM: f64 = 10.0;

/// Writer for GROMACS force-field directives.
///
/// `GromacsTopFfWriter::new().with_precision(p)`, then
/// [`ForceFieldWriter::write`] / [`ForceFieldWriter::write_str`]. The layout,
/// conversions and refusals are listed in the module documentation.
#[derive(Debug, Clone)]
pub struct GromacsTopFfWriter {
    /// Decimal places for floating coefficients.
    pub precision: usize,
}

impl Default for GromacsTopFfWriter {
    fn default() -> Self {
        Self { precision: 6 }
    }
}

impl GromacsTopFfWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }

    fn fmt_f(&self, v: f64) -> String {
        format!("{:.*}", self.precision, v)
    }

    /// The `[ defaults ]` row: `1 comb yes fudgeLJ fudgeQQ`.
    fn defaults_row(&self, ff: &ForceField) -> Result<String, String> {
        let mixing = match ff
            .get_style("pair", "lj/cut")
            .and_then(|s| s.params().get_str("mixing"))
        {
            Some(name) => Mixing::parse(name).map_err(|e| format!("pair/lj/cut: {e}"))?,
            None => Mixing::UNDECLARED,
        };
        let comb = match mixing {
            Mixing::Arithmetic => 2,
            Mixing::Geometric => 3,
            Mixing::SixthPower => {
                return Err(format!(
                    "pair/lj/cut mixing '{}' has no GROMACS comb-rule (2 is arithmetic, 3 \
                     geometric)",
                    mixing.name()
                ));
            }
        };
        let sb = ff.special_bonds();
        for (idx, order) in [(0, "1-2"), (1, "1-3")] {
            if sb.lj[idx] != 0.0 || sb.coul[idx] != 0.0 {
                return Err(format!(
                    "special-bond {order} weights (lj {}, coul {}) are not zero: GROMACS \
                     excludes 1-2 and 1-3 pairs, and gen-pairs scales only 1-4 pairs",
                    sb.lj[idx], sb.coul[idx]
                ));
            }
        }
        Ok(format!(
            "  1  {comb}  yes  {}  {}\n",
            self.fmt_f(sb.lj[2]),
            self.fmt_f(sb.coul[2]),
        ))
    }

    /// The `[ atomtypes ]` row of `t`, with σ/ε from its `lj/cut` self row.
    fn atomtypes_row(&self, t: &AtomType, lj: Option<&Style>) -> Result<String, String> {
        let p = &t.params;
        let name = &t.name;
        let need = |key: &str| {
            p.get(key)
                .ok_or_else(|| format!("atom type '{name}' has no {key}"))
        };
        let (mass, charge) = (need("mass")?, need("charge")?);
        let lj_row = lj.and_then(|s| s.get_pairtype(name, None)).ok_or_else(|| {
            format!("atom type '{name}' has no pair/lj/cut self row (sigma, epsilon)")
        })?;
        let lj_need = |key: &str| {
            lj_row
                .params
                .get(key)
                .ok_or_else(|| format!("pair/lj/cut self row '{name}' has no {key}"))
        };
        let (sigma, epsilon) = (lj_need("sigma")?, lj_need("epsilon")?);

        let mut cols = vec![name.clone()];
        if let Some(bond_type) = p.get_str("bond_type") {
            cols.push(bond_type.to_owned());
        }
        if let Some(z) = p.get("atomic_number") {
            if z.fract() != 0.0 || !z.is_finite() {
                return Err(format!(
                    "atom type '{name}': atomic_number {z} is not an integer"
                ));
            }
            cols.push(format!("{}", z as i64));
        }
        cols.extend([
            self.fmt_f(mass),
            self.fmt_f(charge),
            p.get_str("ptype").unwrap_or("A").to_owned(),
            self.fmt_f(sigma / NM_TO_ANGSTROM),
            self.fmt_f(epsilon * KJ_PER_KCAL),
        ]);
        Ok(format!("  {}\n", cols.join("  ")))
    }

    /// The function code and file-unit columns of one bonded type of `style`.
    fn bonded_columns(&self, style: &Style, name: &str, p: &Params) -> Result<String, String> {
        let what = format!("{}/{} type '{name}'", style.category(), style.name());
        if p.get("k1").is_some() && style.name() == "periodic" {
            return Err(format!(
                "{what} has several periodic terms: that needs dihedraltypes code 9, which \
                 is not modelled"
            ));
        }
        let allowed: &[&str] = match (style.category(), style.name()) {
            ("bond", "harmonic") => &["r0", "k"],
            ("bond", "morse") => &["D", "alpha", "r0"],
            ("angle", "harmonic") => &["theta0", "k"],
            ("dihedral" | "improper", "periodic") => &["k", "periodicity", "phase"],
            ("dihedral", "opls") => &["k1", "k2", "k3", "k4"],
            ("improper", "harmonic") => &["k", "chi0"],
            (category, style) => {
                return Err(format!("{category}/{style} has no GROMACS directive"));
            }
        };
        if let Some((key, _)) = p.iter().find(|(key, _)| !allowed.contains(key)) {
            return Err(format!("{what}: parameter '{key}' has no GROMACS column"));
        }
        let need = |key: &str| p.get(key).ok_or_else(|| format!("{what} has no {key}"));
        let cols: Vec<f64> = match (style.category(), style.name()) {
            ("bond", "harmonic") => vec![
                1.0,
                need("r0")? / NM_TO_ANGSTROM,
                need("k")? * KJ_PER_KCAL * NM_TO_ANGSTROM * NM_TO_ANGSTROM,
            ],
            ("bond", "morse") => vec![
                3.0,
                need("r0")? / NM_TO_ANGSTROM,
                need("D")? * KJ_PER_KCAL,
                need("alpha")? * NM_TO_ANGSTROM,
            ],
            ("angle", "harmonic") => {
                vec![1.0, need("theta0")?.to_degrees(), need("k")? * KJ_PER_KCAL]
            }
            ("dihedral" | "improper", "periodic") => {
                let n = need("periodicity")?;
                if n.fract() != 0.0 || !n.is_finite() {
                    return Err(format!("{what}: periodicity {n} is not an integer"));
                }
                let code = if style.category() == "dihedral" {
                    1.0
                } else {
                    4.0
                };
                vec![
                    code,
                    need("phase")?.to_degrees(),
                    need("k")? * KJ_PER_KCAL,
                    n,
                ]
            }
            ("improper", "harmonic") => {
                let chi0 = p.get("chi0").unwrap_or(0.0);
                if chi0 != 0.0 {
                    return Err(format!(
                        "{what}: chi0 = {chi0} rad; dihedraltypes code 2 is signed and agrees \
                         with K(|phi| - chi0)^2 only at chi0 = 0"
                    ));
                }
                vec![2.0, 0.0, 2.0 * need("k")? * KJ_PER_KCAL]
            }
            _ => {
                // dihedral/opls: an absent k_n is a zero term, as the kernel reads it.
                let f = |key: &str| p.get(key).unwrap_or(0.0) * KJ_PER_KCAL;
                let rb = opls_to_rb([f("k1"), f("k2"), f("k3"), f("k4")]);
                std::iter::once(3.0).chain(rb).collect()
            }
        };
        let (code, values) = cols.split_first().expect("a function code");
        let mut out = format!("{}", *code as i64);
        for (i, v) in values.iter().enumerate() {
            let is_periodicity = style.name() == "periodic" && i == 2;
            out.push_str("  ");
            out.push_str(&if is_periodicity {
                format!("{}", *v as i64)
            } else {
                self.fmt_f(*v)
            });
        }
        Ok(out)
    }
}

impl ForceFieldWriter for GromacsTopFfWriter {
    fn write_str(&self, ff: &ForceField) -> Result<String, String> {
        for style in ff.styles() {
            match (style.category(), style.name()) {
                ("atom", "full")
                | ("bond", "harmonic" | "morse")
                | ("angle", "harmonic")
                | ("dihedral", "periodic" | "opls")
                | ("improper", "periodic" | "harmonic") => {}
                // A pair `cutoff` is a run setting (GROMACS keeps it in the
                // .mdp), not force-field data: it is not written.
                ("pair", "lj/cut") => {
                    if let Some((key, _)) = style.params().iter().find(|(k, _)| *k != "cutoff") {
                        return Err(format!(
                            "pair/lj/cut style param '{key}' has no GROMACS directive"
                        ));
                    }
                }
                ("pair", "coul/cut") => {
                    let declared =
                        |key: &str, value: f64| style.params().get(key).is_none_or(|v| v == value);
                    let extra = style
                        .params()
                        .iter()
                        .any(|(key, _)| !matches!(key, "coulomb" | "dielectric" | "cutoff"));
                    if extra
                        || !declared("coulomb", COULOMB_REAL)
                        || !declared("dielectric", VACUUM_DIELECTRIC)
                        || !style.type_rows().is_empty()
                    {
                        return Err(format!(
                            "pair/coul/cut {:?} has no GROMACS directive: only coulomb = \
                             {COULOMB_REAL} and dielectric = {VACUUM_DIELECTRIC}, with no \
                             types, are implied by the directives",
                            style.params()
                        ));
                    }
                }
                (category, name) => {
                    return Err(format!("{category}/{name} has no GROMACS directive"));
                }
            }
        }

        let mut out = String::from("; Generated by molrs\n\n");
        out.push_str("[ defaults ]\n");
        out.push_str("; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ\n");
        out.push_str(&self.defaults_row(ff)?);
        out.push('\n');

        let atom_types: Vec<&AtomType> = ff.get_atomtypes();
        let lj = ff.get_style("pair", "lj/cut");
        let type_names: HashSet<&str> = atom_types.iter().map(|t| t.name.as_str()).collect();
        if let Some(lj) = lj {
            for (name, ends, _) in lj.type_rows() {
                if ends[0] != ends[1] {
                    return Err(format!(
                        "explicit pair/lj/cut cross row '{name}' ({} with {}) needs \
                         [ nonbond_params ], which is not modelled",
                        ends[0], ends[1]
                    ));
                }
                if !type_names.contains(ends[0]) {
                    return Err(format!(
                        "pair/lj/cut self row '{name}' names no atom/full type"
                    ));
                }
            }
        }
        if !atom_types.is_empty() {
            out.push_str("[ atomtypes ]\n");
            out.push_str("; name  [bond_type]  [at.num]  mass  charge  ptype  sigma  epsilon\n");
            for t in &atom_types {
                out.push_str(&self.atomtypes_row(t, lj)?);
            }
            out.push('\n');
        }

        // A bonded endpoint is the wildcard, an atom-type name or a bond_type.
        let mut labels = type_names;
        labels.extend(
            atom_types
                .iter()
                .filter_map(|t| t.params.get_str("bond_type")),
        );
        for (directive, categories, header) in [
            ("bondtypes", &["bond"][..], "; i  j  func  b0  kb"),
            ("angletypes", &["angle"][..], "; i  j  k  func  th0  cth"),
            (
                "dihedraltypes",
                &["dihedral", "improper"][..],
                "; i  j  k  l  func  params",
            ),
        ] {
            let mut rows = String::new();
            for style in ff
                .styles()
                .iter()
                .filter(|s| categories.contains(&s.category()))
            {
                for (name, ends, params) in style.type_rows() {
                    let mut cols = Vec::with_capacity(ends.len());
                    for end in ends {
                        if end.is_empty() {
                            cols.push("X");
                        } else if labels.contains(end) {
                            cols.push(end);
                        } else {
                            return Err(format!(
                                "{}/{} type '{name}': endpoint '{end}' is neither an atom-type \
                                 name nor a bond_type",
                                style.category(),
                                style.name()
                            ));
                        }
                    }
                    let values = self.bonded_columns(style, name, params)?;
                    rows.push_str(&format!("  {}  {values}\n", cols.join("  ")));
                }
            }
            if !rows.is_empty() {
                out.push_str(&format!("[ {directive} ]\n{header}\n{rows}\n"));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::constants::VACUUM_DIELECTRIC;
    use crate::ff::forcefield::readers::{ForceFieldReader, gromacs::GromacsTopFfReader};
    use crate::ff::forcefield::writers::ForceFieldWriter;
    use crate::ff::forcefield::{ForceField, Params, SpecialBonds, Style};
    use molrs::units::constants::COULOMB_REAL;
    use std::f64::consts::PI;

    // -- fixtures (molrs units: Å, kcal/mol, rad) --------------------------------

    fn atom_params(mass: f64, charge: f64, z: f64, bond_type: &str) -> Params {
        let mut p = Params::from_pairs(&[("mass", mass), ("charge", charge), ("atomic_number", z)]);
        p.set_str("bond_type", bond_type);
        p.set_str("ptype", "A");
        p
    }

    /// opls_135 (CT) and opls_140 (HC), each with its `lj/cut` self row;
    /// `mixing` declared when `Some`. 1-4 weights 0.5 / 0.5.
    fn opls_ff(mixing: Option<&str>) -> ForceField {
        let mut ff = ForceField::new("gmx");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("opls_135", &[], atom_params(12.011, -0.18, 6.0, "CT"))
            .unwrap()
            .def_type("opls_140", &[], atom_params(1.008, 0.06, 1.0, "HC"))
            .unwrap();
        let mut lj_params = Params::new();
        if let Some(rule) = mixing {
            lj_params.set_str("mixing", rule);
        }
        ff.def_style("pair", "lj/cut", lj_params)
            .unwrap()
            .def_type(
                "opls_135",
                &["opls_135"],
                Params::from_pairs(&[("sigma", 3.5), ("epsilon", 0.066)]),
            )
            .unwrap()
            .def_type(
                "opls_140",
                &["opls_140"],
                Params::from_pairs(&[("sigma", 2.5), ("epsilon", 0.03)]),
            )
            .unwrap();
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.5],
        });
        ff
    }

    /// `opls_ff(Some("geometric"))` plus one `category/name` type.
    fn with_type(
        category: &str,
        name: &str,
        type_name: &str,
        endpoints: &[&str],
        params: Params,
    ) -> ForceField {
        let mut ff = opls_ff(Some("geometric"));
        ff.def_style(category, name, Params::new())
            .unwrap()
            .def_type(type_name, endpoints, params)
            .unwrap();
        ff
    }

    fn write(ff: &ForceField) -> String {
        GromacsTopFfWriter::new()
            .write_str(ff)
            .unwrap_or_else(|e| panic!("write_str: {e}"))
    }

    fn write_err(ff: &ForceField) -> String {
        GromacsTopFfWriter::new()
            .write_str(ff)
            .expect_err("expected Err from write_str")
    }

    /// The whitespace-split data rows of `[ section ]` (comments skipped).
    fn section_rows<'a>(text: &'a str, section: &str) -> Vec<Vec<&'a str>> {
        let header = format!("[ {section} ]");
        let mut rows = Vec::new();
        let mut inside = false;
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with('[') {
                inside = t == header;
                continue;
            }
            if inside && !t.is_empty() && !t.starts_with(';') {
                rows.push(t.split_whitespace().collect());
            }
        }
        rows
    }

    /// The only row of `[ section ]` whose leading tokens are `labels`.
    fn row<'a>(text: &'a str, section: &str, labels: &[&str]) -> Vec<&'a str> {
        let rows = section_rows(text, section);
        let mut found: Vec<Vec<&str>> = rows
            .into_iter()
            .filter(|r| r.len() >= labels.len() && r[..labels.len()] == *labels)
            .collect();
        assert_eq!(
            found.len(),
            1,
            "[ {section} ] rows for {labels:?} in:\n{text}"
        );
        found.pop().expect("one row")
    }

    fn number(token: &str) -> f64 {
        token
            .parse()
            .unwrap_or_else(|e| panic!("{token:?} is not a number: {e}"))
    }

    /// `tokens` (after the labels) are `code` then `values`, numerically.
    fn assert_row_values(tokens: &[&str], code: &str, values: &[f64]) {
        assert_eq!(tokens[0], code, "function code in {tokens:?}");
        assert_eq!(tokens.len(), 1 + values.len(), "{tokens:?}");
        for (tok, want) in tokens[1..].iter().zip(values) {
            assert!(
                (number(tok) - want).abs() < 1e-9,
                "{tok} != {want} in {tokens:?}"
            );
        }
    }

    fn assert_names(err: &str, needles: &[&str]) {
        for needle in needles {
            assert!(err.contains(needle), "error should name `{needle}`: {err}");
        }
    }

    // -- [ defaults ] ------------------------------------------------------------

    #[test]
    fn geometric_mixing_writes_comb_rule_3() {
        let text = write(&opls_ff(Some("geometric")));
        let rows = section_rows(&text, "defaults");
        assert_eq!(rows.len(), 1, "{text}");
        assert_eq!(rows[0][..3], ["1", "3", "yes"]);
        assert!((number(rows[0][3]) - 0.5).abs() < 1e-12);
        assert!((number(rows[0][4]) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn arithmetic_mixing_writes_comb_rule_2() {
        let text = write(&opls_ff(Some("arithmetic")));
        assert_eq!(section_rows(&text, "defaults")[0][..3], ["1", "2", "yes"]);
    }

    /// No declared rule is `Mixing::UNDECLARED`, Lorentz-Berthelot: comb-rule 2.
    #[test]
    fn undeclared_mixing_writes_comb_rule_2() {
        let text = write(&opls_ff(None));
        assert_eq!(section_rows(&text, "defaults")[0][..3], ["1", "2", "yes"]);
    }

    #[test]
    fn empty_force_field_writes_only_defaults() {
        let text = write(&ForceField::new("x"));
        assert_eq!(section_rows(&text, "defaults")[0][..3], ["1", "2", "yes"]);
        assert!(section_rows(&text, "atomtypes").is_empty(), "{text}");
    }

    /// GROMACS has no comb-rule for sixth-power mixing.
    #[test]
    fn sixthpower_mixing_is_an_error() {
        let err = write_err(&opls_ff(Some("sixthpower")));
        assert_names(&err, &["sixthpower"]);
    }

    /// gen-pairs 1-4 scaling cannot say "keep 1-2 neighbours".
    #[test]
    fn nonzero_1_2_special_bond_weight_is_an_error() {
        let mut ff = opls_ff(Some("geometric"));
        ff.set_special_bonds(SpecialBonds {
            lj: [0.5, 0.0, 0.5],
            coul: [0.0, 0.0, 0.5],
        });
        let err = write_err(&ff);
        assert_names(&err, &["1-2"]);
    }

    #[test]
    fn nonzero_1_3_special_bond_weight_is_an_error() {
        let mut ff = opls_ff(Some("geometric"));
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.5, 0.5],
        });
        let err = write_err(&ff);
        assert_names(&err, &["1-3"]);
    }

    // -- [ atomtypes ] -----------------------------------------------------------

    /// opls_135 in file units: V = 3.5 Å ÷ 10 = 0.35 nm; W = 0.066 kcal/mol ×
    /// 4.184 = 0.276144 kJ/mol, from the `lj/cut` self row.
    #[test]
    fn atomtypes_row_joins_atom_full_and_the_lj_cut_self_row() {
        let text = write(&opls_ff(Some("geometric")));
        let r = row(&text, "atomtypes", &["opls_135"]);
        assert_eq!(r.len(), 8, "{r:?}");
        assert_eq!(r[1], "CT");
        assert_eq!(r[2], "6");
        assert!((number(r[3]) - 12.011).abs() < 1e-9);
        assert!((number(r[4]) - -0.18).abs() < 1e-9);
        assert_eq!(r[5], "A");
        assert!((number(r[6]) - 0.35).abs() < 1e-9);
        assert!((number(r[7]) - 0.276144).abs() < 1e-9);
    }

    /// Neither `bond_type` nor `atomic_number`: the 6-column form.
    #[test]
    fn atomtypes_row_without_bond_type_or_atomic_number_has_six_columns() {
        let mut ff = opls_ff(Some("geometric"));
        let atoms = ff.get_style_mut("atom", "full").unwrap();
        atoms.remove_type("opls_135");
        let mut p = Params::from_pairs(&[("mass", 12.011), ("charge", -0.18)]);
        p.set_str("ptype", "A");
        atoms.def_type("opls_135", &[], p).unwrap();
        let text = write(&ff);
        let r = row(&text, "atomtypes", &["opls_135"]);
        assert_eq!(r.len(), 6, "{r:?}");
        assert!((number(r[1]) - 12.011).abs() < 1e-9);
        assert_eq!(r[3], "A");
        assert!((number(r[4]) - 0.35).abs() < 1e-9);
    }

    /// molrs `atom/full` types are real atoms: no `ptype` is written `A`.
    #[test]
    fn missing_ptype_is_written_a() {
        let mut ff = opls_ff(Some("geometric"));
        let atoms = ff.get_style_mut("atom", "full").unwrap();
        atoms.remove_type("opls_135");
        let mut p =
            Params::from_pairs(&[("mass", 12.011), ("charge", -0.18), ("atomic_number", 6.0)]);
        p.set_str("bond_type", "CT");
        atoms.def_type("opls_135", &[], p).unwrap();
        let text = write(&ff);
        assert_eq!(row(&text, "atomtypes", &["opls_135"])[5], "A");
    }

    #[test]
    fn declared_ptype_is_written_as_declared() {
        let mut ff = opls_ff(Some("geometric"));
        assert!(
            ff.get_style_mut("atom", "full")
                .unwrap()
                .set_type_str_param("opls_135", "ptype", "S")
        );
        let text = write(&ff);
        assert_eq!(row(&text, "atomtypes", &["opls_135"])[5], "S");
    }

    fn opls_135_without(key: &str) -> ForceField {
        let mut ff = opls_ff(Some("geometric"));
        let atoms = ff.get_style_mut("atom", "full").unwrap();
        atoms.remove_type("opls_135");
        let mut p = Params::new();
        for (k, v) in [("mass", 12.011), ("charge", -0.18), ("atomic_number", 6.0)] {
            if k != key {
                p.set(k, v);
            }
        }
        p.set_str("bond_type", "CT");
        p.set_str("ptype", "A");
        atoms.def_type("opls_135", &[], p).unwrap();
        ff
    }

    #[test]
    fn atom_type_without_mass_is_an_error() {
        let err = write_err(&opls_135_without("mass"));
        assert_names(&err, &["opls_135", "mass"]);
    }

    #[test]
    fn atom_type_without_charge_is_an_error() {
        let err = write_err(&opls_135_without("charge"));
        assert_names(&err, &["opls_135", "charge"]);
    }

    #[test]
    fn atom_type_without_an_lj_cut_self_row_is_an_error() {
        let mut ff = opls_ff(Some("geometric"));
        ff.get_style_mut("pair", "lj/cut")
            .unwrap()
            .remove_type("opls_135");
        let err = write_err(&ff);
        assert_names(&err, &["opls_135", "lj/cut"]);
    }

    /// `[ atomtypes ]` holds self rows only; a cross row needs
    /// `[ nonbond_params ]`, which is not modelled.
    #[test]
    fn explicit_lj_cut_cross_row_is_an_error() {
        let mut ff = opls_ff(Some("geometric"));
        ff.get_style_mut("pair", "lj/cut")
            .unwrap()
            .def_type(
                "opls_135-opls_140",
                &["opls_135", "opls_140"],
                Params::from_pairs(&[("sigma", 3.0), ("epsilon", 0.05)]),
            )
            .unwrap();
        let err = write_err(&ff);
        assert_names(&err, &["opls_135", "opls_140"]);
    }

    // -- bonded directives -------------------------------------------------------

    /// r0 = 1.09 Å → 0.109 nm; k = 680 kcal/mol/Å² × 418.4 = 284512 kJ/mol/nm².
    #[test]
    fn bond_harmonic_is_bondtypes_code_1() {
        let ff = with_type(
            "bond",
            "harmonic",
            "CT-HC",
            &["CT", "HC"],
            Params::from_pairs(&[("r0", 1.09), ("k", 680.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "bondtypes", &["CT", "HC"]);
        assert_row_values(&r[2..], "1", &[0.109, 284512.0]);
    }

    /// b₀ = 0.1529 nm; D = 95.602294455066… × 4.184 = 400 kJ/mol; β = 2 × 10 =
    /// 20 nm⁻¹.
    #[test]
    fn bond_morse_is_bondtypes_code_3() {
        let ff = with_type(
            "bond",
            "morse",
            "CT-CT",
            &["CT", "CT"],
            Params::from_pairs(&[("D", 95.602_294_455_066_9), ("alpha", 2.0), ("r0", 1.529)]),
        );
        let text = write(&ff);
        let r = row(&text, "bondtypes", &["CT", "CT"]);
        assert_row_values(&r[2..], "3", &[0.1529, 400.0, 20.0]);
    }

    /// θ₀ = 107.8·π/180 rad → 107.8°; k = 66 × 4.184 = 276.144 kJ/mol/rad².
    #[test]
    fn angle_harmonic_is_angletypes_code_1() {
        let ff = with_type(
            "angle",
            "harmonic",
            "HC-CT-HC",
            &["HC", "CT", "HC"],
            Params::from_pairs(&[("theta0", 107.8 * PI / 180.0), ("k", 66.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "angletypes", &["HC", "CT", "HC"]);
        assert_row_values(&r[3..], "1", &[107.8, 276.144]);
    }

    /// φ_s = 0°; k = 1 × 4.184 = 4.184 kJ/mol; n = 3.
    #[test]
    fn dihedral_periodic_is_dihedraltypes_code_1() {
        let ff = with_type(
            "dihedral",
            "periodic",
            "CT-CT-CT-CT",
            &["CT", "CT", "CT", "CT"],
            Params::from_pairs(&[("k", 1.0), ("periodicity", 3.0), ("phase", 0.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["CT", "CT", "CT", "CT"]);
        assert_row_values(&r[4..], "1", &[0.0, 4.184, 3.0]);
    }

    /// F3 = 0.3 kcal/mol × 4.184 = 1.2552 kJ/mol: C0 = ½·1.2552 = 0.6276,
    /// C1 = 1.5·1.2552 = 1.8828, C3 = −2·1.2552 = −2.5104, C2 = C4 = C5 = 0.
    #[test]
    fn dihedral_opls_is_dihedraltypes_code_3_in_rb_form() {
        let ff = with_type(
            "dihedral",
            "opls",
            "HC-CT-CT-HC",
            &["HC", "CT", "CT", "HC"],
            Params::from_pairs(&[("k1", 0.0), ("k2", 0.0), ("k3", 0.3), ("k4", 0.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["HC", "CT", "CT", "HC"]);
        assert_row_values(&r[4..], "3", &[0.6276, 1.8828, 0.0, -2.5104, 0.0, 0.0]);
    }

    #[test]
    fn a_pair_cutoff_is_a_run_setting_and_not_written() {
        let mut ff = ForceField::new("t");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        ff.def_style("pair", "coul/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        let text = write(&ff);
        assert!(!text.contains("10"), "{text}");
    }

    /// The empty endpoint wildcard is written as GROMACS `X`.
    #[test]
    fn empty_endpoint_is_written_x() {
        let ff = with_type(
            "dihedral",
            "opls",
            "-CT-CT-",
            &["", "CT", "CT", ""],
            Params::from_pairs(&[("k1", 0.0), ("k2", 0.0), ("k3", 0.3), ("k4", 0.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["X", "CT", "CT", "X"]);
        assert_eq!(r[4], "3");
    }

    /// k = 2.5 × 4.184 = 10.46 kJ/mol; φ_s = π → 180°; n = 2.
    #[test]
    fn improper_periodic_is_dihedraltypes_code_4() {
        let ff = with_type(
            "improper",
            "periodic",
            "--CT-HC",
            &["", "", "CT", "HC"],
            Params::from_pairs(&[("k", 2.5), ("periodicity", 2.0), ("phase", PI)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["X", "X", "CT", "HC"]);
        assert_row_values(&r[4..], "4", &[180.0, 10.46, 2.0]);
    }

    /// K(χ)² = ½k_ξ(ξ)²: k_ξ = 2 · 20 · 4.184 = 167.36 kJ/mol/rad²; ξ₀ = 0.
    #[test]
    fn improper_harmonic_is_dihedraltypes_code_2() {
        let ff = with_type(
            "improper",
            "harmonic",
            "--CT-HC",
            &["", "", "CT", "HC"],
            Params::from_pairs(&[("k", 20.0), ("chi0", 0.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["X", "X", "CT", "HC"]);
        assert_row_values(&r[4..], "2", &[0.0, 167.36]);
    }

    /// Directives only: no molecule section is written.
    #[test]
    fn no_molecule_section_is_written() {
        let mut ff = with_type(
            "bond",
            "harmonic",
            "CT-HC",
            &["CT", "HC"],
            Params::from_pairs(&[("r0", 1.09), ("k", 680.0)]),
        );
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "HC-CT-HC",
                &["HC", "CT", "HC"],
                Params::from_pairs(&[("theta0", 1.9), ("k", 66.0)]),
            )
            .unwrap();
        let text = write(&ff);
        for section in [
            "atoms",
            "bonds",
            "angles",
            "dihedrals",
            "pairs",
            "moleculetype",
        ] {
            assert!(
                !text.contains(&format!("[ {section} ]")),
                "[ {section} ] written:\n{text}"
            );
        }
    }

    // -- refusals ----------------------------------------------------------------

    #[test]
    fn dihedral_charmm_is_an_error() {
        let ff = with_type(
            "dihedral",
            "charmm",
            "CT-CT-CT-CT",
            &["CT", "CT", "CT", "CT"],
            Params::from_pairs(&[("k", 1.0), ("periodicity", 3.0), ("phase", 0.0), ("w", 0.5)]),
        );
        let err = write_err(&ff);
        assert_names(&err, &["charmm"]);
    }

    #[test]
    fn dihedral_multi_harmonic_is_an_error() {
        let ff = with_type(
            "dihedral",
            "multi/harmonic",
            "CT-CT-CT-CT",
            &["CT", "CT", "CT", "CT"],
            Params::from_pairs(&[
                ("a1", 1.0),
                ("a2", 0.0),
                ("a3", 0.0),
                ("a4", 0.0),
                ("a5", 0.0),
            ]),
        );
        let err = write_err(&ff);
        assert_names(&err, &["multi/harmonic"]);
    }

    /// Several periodic terms on one type need code 9, which is not modelled.
    #[test]
    fn multi_term_dihedral_periodic_is_an_error() {
        let ff = with_type(
            "dihedral",
            "periodic",
            "CT-CT-CT-CT",
            &["CT", "CT", "CT", "CT"],
            Params::from_pairs(&[
                ("k1", 1.0),
                ("periodicity1", 1.0),
                ("phase1", 0.0),
                ("k2", 0.5),
                ("periodicity2", 3.0),
                ("phase2", 0.0),
            ]),
        );
        let err = write_err(&ff);
        assert_names(&err, &["periodic", "CT-CT-CT-CT"]);
    }

    /// `ZZ` is neither an atom-type name nor any type's `bond_type`.
    #[test]
    fn unresolvable_bonded_endpoint_is_an_error() {
        let ff = with_type(
            "bond",
            "harmonic",
            "ZZ-CT",
            &["ZZ", "CT"],
            Params::from_pairs(&[("r0", 1.09), ("k", 680.0)]),
        );
        let err = write_err(&ff);
        assert_names(&err, &["ZZ"]);
    }

    /// An atom-type name is a resolvable endpoint too.
    #[test]
    fn atom_type_name_endpoint_is_written() {
        let ff = with_type(
            "bond",
            "harmonic",
            "opls_135-opls_140",
            &["opls_135", "opls_140"],
            Params::from_pairs(&[("r0", 1.09), ("k", 680.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "bondtypes", &["opls_135", "opls_140"]);
        assert_row_values(&r[2..], "1", &[0.109, 284512.0]);
    }

    // -- read(write(ff)) == ff -----------------------------------------------------

    /// A force field holding every style the GROMACS directives express.
    fn every_supported_style() -> ForceField {
        let mut ff = opls_ff(Some("geometric"));
        ff.def_style(
            "pair",
            "coul/cut",
            Params::from_pairs(&[("coulomb", COULOMB_REAL), ("dielectric", VACUUM_DIELECTRIC)]),
        )
        .unwrap();
        // (category, style, type name, endpoints, params) of one type definition.
        type TypeDef<'a> = (
            &'a str,
            &'a str,
            &'a str,
            &'a [&'a str],
            &'a [(&'a str, f64)],
        );
        let defs: [TypeDef; 9] = [
            (
                "bond",
                "harmonic",
                "CT-HC",
                &["CT", "HC"],
                &[("r0", 1.09), ("k", 680.0)],
            ),
            (
                "bond",
                "morse",
                "CT-CT",
                &["CT", "CT"],
                &[("D", 95.602_294_455_066_9), ("alpha", 2.0), ("r0", 1.529)],
            ),
            (
                "angle",
                "harmonic",
                "HC-CT-HC",
                &["HC", "CT", "HC"],
                &[("theta0", 107.8 * PI / 180.0), ("k", 66.0)],
            ),
            (
                "dihedral",
                "periodic",
                "CT-CT-CT-CT",
                &["CT", "CT", "CT", "CT"],
                &[("k", 1.0), ("periodicity", 3.0), ("phase", 0.0)],
            ),
            (
                "dihedral",
                "opls",
                "HC-CT-CT-HC",
                &["HC", "CT", "CT", "HC"],
                &[("k1", 0.0), ("k2", 0.0), ("k3", 0.3), ("k4", 0.0)],
            ),
            (
                "dihedral",
                "opls",
                "-CT-CT-",
                &["", "CT", "CT", ""],
                &[("k1", 1.3), ("k2", -0.05), ("k3", 0.2), ("k4", 0.1)],
            ),
            (
                "improper",
                "periodic",
                "--CT-HC",
                &["", "", "CT", "HC"],
                &[("k", 2.5), ("periodicity", 2.0), ("phase", PI)],
            ),
            (
                "improper",
                "harmonic",
                "--HC-CT",
                &["", "", "HC", "CT"],
                &[("k", 20.0), ("chi0", 0.0)],
            ),
            (
                "angle",
                "harmonic",
                "CT-CT-HC",
                &["CT", "CT", "HC"],
                &[("theta0", 1.9), ("k", 37.5)],
            ),
        ];
        for (category, style, name, endpoints, params) in defs {
            ff.def_style(category, style, Params::new())
                .unwrap()
                .def_type(name, endpoints, Params::from_pairs(params))
                .unwrap();
        }
        ff
    }

    type TypeRow = (String, Vec<String>, Params);

    /// `(category, name)` → (style params, types sorted by name).
    fn snapshot(ff: &ForceField) -> Vec<((String, String), Params, Vec<TypeRow>)> {
        let mut out: Vec<((String, String), Params, Vec<TypeRow>)> = ff
            .styles()
            .iter()
            .map(|s: &Style| {
                let mut types: Vec<TypeRow> = s
                    .defs()
                    .collect_type_params()
                    .into_iter()
                    .map(|(name, params)| {
                        let ends = s.type_endpoints(&name).expect("type has endpoints");
                        (name, ends, params)
                    })
                    .collect();
                types.sort_by(|a, b| a.0.cmp(&b.0));
                (
                    (s.category().to_owned(), s.name().to_owned()),
                    s.params().clone(),
                    types,
                )
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn assert_params_close(what: &str, got: &Params, want: &Params) {
        let mut got_keys: Vec<&str> = got.iter().map(|(k, _)| k).collect();
        let mut want_keys: Vec<&str> = want.iter().map(|(k, _)| k).collect();
        got_keys.sort_unstable();
        want_keys.sort_unstable();
        assert_eq!(got_keys, want_keys, "{what}: numeric keys");
        for (k, w) in want.iter() {
            let g = got.get(k).expect("key present");
            assert!((g - w).abs() < 1e-9, "{what}.{k}: got {g}, want {w}");
        }
        let mut got_strs: Vec<(&str, &str)> = got.iter_strings().collect();
        let mut want_strs: Vec<(&str, &str)> = want.iter_strings().collect();
        got_strs.sort_unstable();
        want_strs.sort_unstable();
        assert_eq!(got_strs, want_strs, "{what}: string params");
    }

    #[test]
    fn reading_what_is_written_gives_back_the_force_field() {
        let ff = every_supported_style();
        let text = GromacsTopFfWriter::new()
            .with_precision(10)
            .write_str(&ff)
            .unwrap_or_else(|e| panic!("write_str: {e}"));
        let back = GromacsTopFfReader::new()
            .read_str(&text)
            .unwrap_or_else(|e| panic!("read_str: {e}\n{text}"));

        assert_eq!(back.special_bonds(), ff.special_bonds());
        let (want, got) = (snapshot(&ff), snapshot(&back));
        let keys = |s: &[((String, String), Params, Vec<TypeRow>)]| {
            s.iter().map(|(k, _, _)| k.clone()).collect::<Vec<_>>()
        };
        assert_eq!(keys(&got), keys(&want), "styles");
        for ((key, want_params, want_types), (_, got_params, got_types)) in want.iter().zip(&got) {
            let what = format!("{}/{}", key.0, key.1);
            assert_params_close(&what, got_params, want_params);
            let names = |t: &[TypeRow]| t.iter().map(|r| r.0.clone()).collect::<Vec<_>>();
            assert_eq!(names(got_types), names(want_types), "{what}: type names");
            for ((name, want_ends, want_p), (_, got_ends, got_p)) in
                want_types.iter().zip(got_types)
            {
                assert_eq!(got_ends, want_ends, "{what} {name}: endpoints");
                assert_params_close(&format!("{what} {name}"), got_p, want_p);
            }
        }
    }
}
