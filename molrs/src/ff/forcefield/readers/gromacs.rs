//! GROMACS force-field directive reader.
//!
//! Reads the force-field **directives** of a GROMACS topology
//! (`forcefield.itp` with its `ffnonbonded.itp` / `ffbonded.itp` includes, or a
//! `.top` whose molecule sections are skipped) into a molrs [`ForceField`].
//! The file speaks nm, kJ/mol, degrees and e; the force field is in molrs units
//! (Å, kcal/mol, rad, e). Every conversion happens here, at the boundary.
//!
//! # Directives read
//!
//! - **`[ defaults ]`** `nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ`. Requires
//!   nbfunc 1 (Lennard-Jones) and gen-pairs `yes`. comb-rule 2
//!   (σᵢⱼ = ½(σᵢ+σⱼ), εᵢⱼ = √(εᵢεⱼ)) sets the `pair/lj/cut` string param
//!   `mixing` to `arithmetic`; comb-rule 3 (σᵢⱼ = √(σᵢσⱼ), εᵢⱼ = √(εᵢεⱼ)) sets
//!   it to `geometric`. The special-bond weights are `lj [0, 0, fudgeLJ]` and
//!   `coul [0, 0, fudgeQQ]`.
//! - **`[ atomtypes ]`** `name [bond_type] [at.num] mass charge ptype V W`.
//!   Columns resolve from the right: the last five are `mass charge ptype V W`
//!   (`ptype` one of `A S V D B`); the one to three leading tokens are `name`,
//!   an optional `bond_type` and an optional integer `at.num`. Each row defines
//!   - an `atom/full` type with `mass` (amu), `charge` (e), `atomic_number`
//!     (when present) and string `ptype` and `class` (= `bond_type`, when
//!     present);
//!   - a `pair/lj/cut` self row with `sigma` = V·10 (Å) and
//!     `epsilon` = W/4.184 (kcal/mol) — V/W are σ/ε under comb-rules 2 and 3,
//!     so `[ atomtypes ]` requires `[ defaults ]`;
//!   - the `pair/coul/cut` style, with `coulomb` = `COULOMB_REAL` and
//!     `dielectric` = 1 (vacuum).
//! - **`[ bondtypes ]` / `[ angletypes ]` / `[ dihedraltypes ]`**
//!   `labels… funct params…`, keyed by the row's labels. GROMACS `X` becomes
//!   the empty-endpoint wildcard, so `X CT CT X` is the type `-CT-CT-`.
//!
//! | Directive, funct | GROMACS form | molrs style | Conversion |
//! |---|---|---|---|
//! | bondtypes 1 | ½k_b(r−b₀)² | `bond/harmonic` {`r0`, `k`} | r0 = b₀·10 Å; k = k_b/418.4 kcal/mol/Å² |
//! | bondtypes 3 | D[1−e^{−β(r−b₀)}]² | `bond/morse` {`D`, `alpha`, `r0`} | D/4.184 kcal/mol; alpha = β/10 Å⁻¹; r0 = b₀·10 Å |
//! | angletypes 1 | ½k_θ(θ−θ₀)² | `angle/harmonic` {`theta0`, `k`} | θ₀ deg → rad; k/4.184 kcal/mol/rad² |
//! | dihedraltypes 1 | k_φ[1+cos(nφ−φ_s)] | `dihedral/periodic` {`k`, `periodicity`, `phase`} | φ_s deg → rad; k/4.184 kcal/mol |
//! | dihedraltypes 2 | ½k_ξ(ξ−ξ₀)² | `improper/harmonic` {`k`, `chi0` = 0} | k = k_ξ/(2·4.184) kcal/mol/rad²; only ξ₀ = 0 |
//! | dihedraltypes 3 | Σₙ Cₙ cosⁿψ (Ryckaert–Bellemans) | `dihedral/opls` {`k1`..`k4`} | exact RB → Fourier inversion, then /4.184 kcal/mol |
//! | dihedraltypes 4 | k_φ[1+cos(nφ−φ_s)] | `improper/periodic` {`k`, `periodicity`, `phase`} | as funct 1 |
//!
//! ½k_ξ(ξ−ξ₀)² is signed and molrs's `improper/harmonic` is `K(χ−χ₀)²` with
//! χ = |φ|; the two agree only at ξ₀ = 0. An RB row converts only when
//! `C₅ = 0` and `ΣCₙ = 0` (see `ff::forcefield::torsion`). Rows repeated
//! across sections or includes follow the conflict rule of
//! [`Style::def_type`](crate::ff::forcefield::Style::def_type): equal
//! parameters are one type, different ones an error.
//!
//! # Refusals
//!
//! Anything this reader does not model is an `Err` naming it, never a silent
//! drop:
//!
//! - function codes other than those in the table (bond 2, 4, 5+; angle 2+;
//!   dihedral 5, 8, 9, 10+), dihedral 2 with ξ₀ ≠ 0, an RB row with `C₅ ≠ 0` or
//!   `ΣCₙ ≠ 0`, a row with the wrong parameter count, and the 2-name
//!   dihedraltypes form;
//! - comb-rule 1 (V/W are C6/C12), nbfunc 2 (Buckingham), gen-pairs `no`;
//! - the sections `[ pairtypes ]`, `[ nonbond_params ]`, `[ constrainttypes ]`,
//!   `[ cmaptypes ]`, `[ implicit_genborn_params ]`, and any unknown section;
//! - every molecule section (`[ moleculetype ]`, `[ atoms ]`, `[ bonds ]`,
//!   `[ pairs ]`, `[ angles ]`, `[ dihedrals ]`, `[ exclusions ]`,
//!   `[ settles ]`, `[ system ]`, `[ molecules ]`, …): that is topology, read
//!   with io::data::top::read_top.
//!
//! A section the caller names with
//! [`GromacsTopFfReader::with_skipped_directive`] is read past instead, rows
//! and all.
//!
//! # Preprocessor
//!
//! - `#include "file"` is followed when [`GromacsTopFfReader::with_include`]
//!   is `true` (and ignored otherwise). It resolves relative to the including
//!   file; a file already read is not read again.
//! - `#define NAME [body]` and `#undef NAME` maintain a define set. Bodies are
//!   never expanded.
//! - `#ifdef` / `#ifndef` / `#else` / `#endif` select lines against that set,
//!   and nest.
//! - `#if`, `#elif` and any other directive are an `Err`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::ForceFieldReader;
use crate::ff::constants::VACUUM_DIELECTRIC;
use crate::ff::forcefield::mixing::Mixing;
use crate::ff::forcefield::torsion::rb_to_opls;
use crate::ff::forcefield::{ForceField, Params, SpecialBonds};
use molrs::store::type_labels::TypeName;
use molrs::units::constants::COULOMB_REAL;

const KJ_PER_KCAL: f64 = 4.184;
const NM_TO_ANGSTROM: f64 = 10.0;

/// Sections that describe a molecule — topology, not force-field directives.
const MOLECULE_SECTIONS: &[&str] = &[
    "moleculetype",
    "atoms",
    "bonds",
    "pairs",
    "pairs_nb",
    "angles",
    "dihedrals",
    "exclusions",
    "constraints",
    "settles",
    "cmap",
    "virtual_sites1",
    "virtual_sites2",
    "virtual_sites3",
    "virtual_sites4",
    "virtual_sitesn",
    "position_restraints",
    "distance_restraints",
    "dihedral_restraints",
    "orientation_restraints",
    "angle_restraints",
    "angle_restraints_z",
    "polarization",
    "water_polarization",
    "thole_polarization",
    "intermolecular_interactions",
    "system",
    "molecules",
];

/// Force-field directives GROMACS defines that this reader does not model.
const UNMODELLED_SECTIONS: &[&str] = &[
    "pairtypes",
    "nonbond_params",
    "constrainttypes",
    "cmaptypes",
    "implicit_genborn_params",
];

/// Reader for GROMACS force-field directives.
///
/// Configure with the builders, then read with
/// [`ForceFieldReader::read`] / [`ForceFieldReader::read_str`]. The supported
/// directives, conversions and refusals are listed in the module
/// documentation.
///
/// # Examples
///
/// ```
/// use molrs::ff::{ForceFieldReader, GromacsTopFfReader};
///
/// let text = "\
/// [ defaults ]
/// 1  3  yes  0.5  0.5
/// [ atomtypes ]
/// opls_135  CT  6  12.011  -0.18  A  0.35  0.276144
/// [ bondtypes ]
/// CT  HC  1  0.10900  284512.0
/// ";
/// let ff = GromacsTopFfReader::new().read_str(text)?;
///
/// // comb-rule 3 is geometric mixing, declared on the lj/cut style.
/// let lj = ff.get_style("pair", "lj/cut").expect("lj/cut style");
/// assert_eq!(lj.params().get_str("mixing"), Some("geometric"));
///
/// // b₀ = 0.109 nm is r0 = 1.09 Å.
/// let bonds = ff.get_bondtypes();
/// let r0 = bonds[0].params.get("r0").expect("r0");
/// assert!((r0 - 1.09).abs() < 1e-12);
/// # Ok::<(), String>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct GromacsTopFfReader {
    include: bool,
    skipped: HashSet<String>,
}

impl GromacsTopFfReader {
    /// A reader that ignores `#include` and skips no section.
    pub fn new() -> Self {
        Self::default()
    }

    /// Follow `#include` directives, resolved relative to the including file
    /// (default `false`: they are ignored).
    pub fn with_include(mut self, include: bool) -> Self {
        self.include = include;
        self
    }

    /// Read past every `[ name ]` section instead of refusing it. `name` is the
    /// directive name without brackets (`"constrainttypes"`, `"atoms"`);
    /// section names are case-insensitive.
    pub fn with_skipped_directive(mut self, name: &str) -> Self {
        self.skipped.insert(name.trim().to_ascii_lowercase());
        self
    }

    /// Whether the rows of `section` are read (`Ok(true)`) or skipped
    /// (`Ok(false)`); `Err` naming the section when it is refused.
    fn admits(&self, section: &str) -> Result<bool, String> {
        if self.skipped.contains(section) {
            return Ok(false);
        }
        match section {
            "defaults" | "atomtypes" | "bondtypes" | "angletypes" | "dihedraltypes" => Ok(true),
            s if MOLECULE_SECTIONS.contains(&s) => Err(format!(
                "[ {s} ] is topology, not a force-field directive: read it with \
                 io::data::top::read_top, or skip it with with_skipped_directive(\"{s}\")"
            )),
            s if UNMODELLED_SECTIONS.contains(&s) => Err(format!(
                "[ {s} ] is not modelled by the GROMACS force-field reader; skip it with \
                 with_skipped_directive(\"{s}\")"
            )),
            s => Err(format!(
                "[ {s} ] is not a GROMACS force-field directive this reader knows; skip it \
                 with with_skipped_directive(\"{s}\")"
            )),
        }
    }

    /// Preprocess `text` (from `origin`, in directory `dir`) into `scan`.
    fn scan(
        &self,
        text: &str,
        origin: &str,
        dir: Option<&Path>,
        scan: &mut Scan,
    ) -> Result<(), String> {
        let mut conds: Vec<Cond> = Vec::new();
        for (idx, raw) in text.lines().enumerate() {
            let at = format!("{origin}:{}", idx + 1);
            let line = raw.split(';').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let active = conds.last().is_none_or(Cond::active);
            if let Some(rest) = line.strip_prefix('#') {
                let rest = rest.trim_start();
                let (name, arg) = rest
                    .split_once(char::is_whitespace)
                    .map_or((rest, ""), |(n, a)| (n, a.trim()));
                let symbol = arg.split_whitespace().next();
                match name {
                    "ifdef" | "ifndef" => {
                        let symbol = symbol.ok_or_else(|| format!("{at}: #{name} needs a name"))?;
                        let defined = scan.defines.contains(symbol);
                        conds.push(Cond {
                            parent: active,
                            taken: if name == "ifdef" { defined } else { !defined },
                            seen_else: false,
                        });
                    }
                    "else" => {
                        let cond = conds
                            .last_mut()
                            .ok_or_else(|| format!("{at}: #else without #ifdef / #ifndef"))?;
                        if cond.seen_else {
                            return Err(format!("{at}: second #else in one conditional"));
                        }
                        cond.taken = !cond.taken;
                        cond.seen_else = true;
                    }
                    "endif" => {
                        conds
                            .pop()
                            .ok_or_else(|| format!("{at}: #endif without #ifdef / #ifndef"))?;
                    }
                    "if" | "elif" => {
                        return Err(format!(
                            "{at}: #{name} is not supported (only #ifdef / #ifndef / #else / \
                             #endif are evaluated)"
                        ));
                    }
                    _ if !active => {}
                    "define" => {
                        let symbol = symbol.ok_or_else(|| format!("{at}: #define needs a name"))?;
                        scan.defines.insert(symbol.to_owned());
                    }
                    "undef" => {
                        let symbol = symbol.ok_or_else(|| format!("{at}: #undef needs a name"))?;
                        scan.defines.remove(symbol);
                    }
                    "include" => {
                        if self.include {
                            self.include_file(arg, &at, dir, scan)?;
                        }
                    }
                    other => {
                        return Err(format!("{at}: #{other} is not a supported directive"));
                    }
                }
                continue;
            }
            if !active {
                continue;
            }
            if let Some(header) = line.strip_prefix('[') {
                let name = header
                    .strip_suffix(']')
                    .ok_or_else(|| format!("{at}: unterminated section header '{line}'"))?
                    .trim()
                    .to_ascii_lowercase();
                let admitted = self.admits(&name).map_err(|e| format!("{at}: {e}"))?;
                scan.section = Some((name, admitted));
                continue;
            }
            match &scan.section {
                None => return Err(format!("{at}: row '{line}' is outside any [ section ]")),
                Some((_, false)) => {}
                Some((section, true)) => scan.rows.push(Row {
                    at,
                    section: section.clone(),
                    text: line.to_owned(),
                }),
            }
        }
        if !conds.is_empty() {
            return Err(format!(
                "{origin}: {} #ifdef / #ifndef block(s) without #endif",
                conds.len()
            ));
        }
        Ok(())
    }

    /// Follow `#include <arg>` from the file at `at`, relative to `dir`.
    fn include_file(
        &self,
        arg: &str,
        at: &str,
        dir: Option<&Path>,
        scan: &mut Scan,
    ) -> Result<(), String> {
        let name = arg.trim_matches(|c| c == '"' || c == '<' || c == '>');
        if name.is_empty() {
            return Err(format!("{at}: #include names no file"));
        }
        let given = Path::new(name);
        let path = if given.is_absolute() {
            given.to_path_buf()
        } else {
            dir.ok_or_else(|| {
                format!(
                    "{at}: #include \"{name}\" is relative, and text read from a string has \
                     no directory to resolve it against"
                )
            })?
            .join(given)
        };
        if !path.is_file() {
            return Err(format!(
                "{at}: could not resolve #include \"{name}\" (tried {})",
                path.display()
            ));
        }
        if !scan.visited.insert(path.clone()) {
            return Ok(());
        }
        let body = std::fs::read_to_string(&path)
            .map_err(|e| format!("{at}: #include {}: {e}", path.display()))?;
        self.scan(&body, &path.display().to_string(), path.parent(), scan)
    }
}

impl ForceFieldReader for GromacsTopFfReader {
    fn read_str(&self, text: &str) -> Result<ForceField, String> {
        let mut scan = Scan::default();
        self.scan(text, "<string>", None, &mut scan)?;
        scan.build()
    }

    fn read(&self, path: &str) -> Result<ForceField, String> {
        let p = Path::new(path);
        if !p.is_file() {
            return Err(format!("file not found: {path}"));
        }
        let text = std::fs::read_to_string(p).map_err(|e| format!("read {path}: {e}"))?;
        let mut scan = Scan::default();
        scan.visited.insert(p.to_path_buf());
        self.scan(
            &text,
            path,
            Some(p.parent().unwrap_or(Path::new("."))),
            &mut scan,
        )?;
        scan.build()
    }
}

// ---------------------------------------------------------------------------
// Preprocessed text
// ---------------------------------------------------------------------------

/// One open `#ifdef` / `#ifndef` block.
struct Cond {
    /// Whether the enclosing text is active.
    parent: bool,
    /// Whether the current branch of this block is taken.
    taken: bool,
    seen_else: bool,
}

impl Cond {
    fn active(&self) -> bool {
        self.parent && self.taken
    }
}

/// The preprocessor's state and output: the define set, the files read, the
/// current section, and every admitted data row in file order.
#[derive(Default)]
struct Scan {
    defines: HashSet<String>,
    visited: HashSet<PathBuf>,
    /// The current section and whether its rows are read.
    section: Option<(String, bool)>,
    rows: Vec<Row>,
}

impl Scan {
    /// The force field the admitted rows define.
    fn build(&self) -> Result<ForceField, String> {
        let mut ff = ForceField::new("GROMACS");
        let mut defaults = self.rows.iter().filter(|r| r.section == "defaults");
        let mixing = match defaults.next() {
            Some(row) => {
                if let Some(extra) = defaults.next() {
                    return Err(extra.err("a second [ defaults ] row"));
                }
                let (special_bonds, mixing) = row.defaults()?;
                ff.set_special_bonds(special_bonds);
                Some(mixing)
            }
            None => None,
        };
        for row in &self.rows {
            match row.section.as_str() {
                "defaults" => {}
                "atomtypes" => {
                    let mixing = mixing.ok_or_else(|| {
                        row.err(
                            "[ atomtypes ] needs [ defaults ]: V/W are sigma/epsilon only under \
                             the comb-rule it declares",
                        )
                    })?;
                    let (name, atom, lj) = row.atomtype()?;
                    ff.def_style("atom", "full", Params::new())
                        .and_then(|s| s.def_type(name, &[], atom))
                        .map_err(|e| row.err(&e.to_string()))?;
                    let mut lj_style = Params::new();
                    lj_style.set_str("mixing", mixing.name());
                    ff.def_style("pair", "lj/cut", lj_style)
                        .and_then(|s| s.def_type(name, &[name], lj))
                        .map_err(|e| row.err(&e.to_string()))?;
                    ff.def_style(
                        "pair",
                        "coul/cut",
                        Params::from_pairs(&[
                            ("coulomb", COULOMB_REAL),
                            ("dielectric", VACUUM_DIELECTRIC),
                        ]),
                    )
                    .map_err(|e| row.err(&e.to_string()))?;
                }
                _ => {
                    let (category, style, labels, params) = row.bonded()?;
                    let name = TypeName::join(&labels).map_err(|e| row.err(&e))?;
                    ff.def_style(category, style, Params::new())
                        .and_then(|s| s.def_type(name.as_str(), &labels, params))
                        .map_err(|e| row.err(&e.to_string()))?;
                }
            }
        }
        Ok(ff)
    }
}

/// One admitted data row, with the section it sits in and where it came from.
struct Row {
    /// `file:line`.
    at: String,
    section: String,
    text: String,
}

impl Row {
    /// An error about this row, naming its place, section and text.
    fn err(&self, why: &str) -> String {
        format!(
            "{}: [ {} ] row '{}': {why}",
            self.at, self.section, self.text
        )
    }

    /// `[ defaults ]` → the special-bond weights and the mixing rule.
    fn defaults(&self) -> Result<(SpecialBonds, Mixing), String> {
        let cols: Vec<&str> = self.text.split_whitespace().collect();
        let [nbfunc, comb, gen_pairs, fudge_lj, fudge_qq] = cols[..] else {
            return Err(self.err("expected `nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ`"));
        };
        if nbfunc != "1" {
            return Err(self.err(&format!(
                "nbfunc {nbfunc} is not supported: only 1 (Lennard-Jones) is modelled"
            )));
        }
        let mixing = match comb {
            "2" => Mixing::Arithmetic,
            "3" => Mixing::Geometric,
            other => {
                return Err(self.err(&format!(
                    "comb-rule {other} is not supported: only 2 (arithmetic) and 3 \
                     (geometric), under which V/W are sigma/epsilon"
                )));
            }
        };
        if !gen_pairs.eq_ignore_ascii_case("yes") {
            return Err(self.err(&format!(
                "gen-pairs {gen_pairs} is not supported: 1-4 pairs are generated from \
                 [ atomtypes ] (gen-pairs yes) and [ pairtypes ] is not modelled"
            )));
        }
        let number = |tok: &str, what: &str| {
            tok.parse::<f64>()
                .map_err(|_| self.err(&format!("{what} is not a number: {tok}")))
        };
        let special_bonds = SpecialBonds {
            lj: [0.0, 0.0, number(fudge_lj, "fudgeLJ")?],
            coul: [0.0, 0.0, number(fudge_qq, "fudgeQQ")?],
        };
        Ok((special_bonds, mixing))
    }

    /// `[ atomtypes ]` → `(name, atom/full params, lj/cut self-row params)` in
    /// molrs units.
    ///
    /// Columns resolve from the right: the last five are `mass charge ptype V
    /// W`; the one to three leading tokens are `name`, then an optional
    /// `bond_type` and an optional integer `at.num`.
    fn atomtype(&self) -> Result<(&str, Params, Params), String> {
        let cols: Vec<&str> = self.text.split_whitespace().collect();
        if cols.len() < 6 {
            return Err(self.err("expected `name [bond_type] [at.num] mass charge ptype V W`"));
        }
        let (lead, tail) = cols.split_at(cols.len() - 5);
        let is_int = |tok: &str| tok.parse::<i64>().is_ok();
        let (name, bond_type, at_num) = match *lead {
            [name] => (name, None, None),
            [name, second] if is_int(second) => (name, None, Some(second)),
            [name, second] => (name, Some(second), None),
            [name, bond_type, at_num] if is_int(at_num) => (name, Some(bond_type), Some(at_num)),
            [_, _, _] => return Err(self.err("at.num is not an integer")),
            _ => {
                return Err(self.err("more than three tokens before `mass charge ptype V W`"));
            }
        };
        let number = |tok: &str, what: &str| {
            tok.parse::<f64>()
                .map_err(|_| self.err(&format!("{what} is not a number: {tok}")))
        };
        let ptype = tail[2];
        if !matches!(ptype, "A" | "S" | "V" | "D" | "B") {
            return Err(self.err(&format!("ptype '{ptype}' is not one of A S V D B")));
        }

        let mut atom = Params::from_pairs(&[
            ("mass", number(tail[0], "mass")?),
            ("charge", number(tail[1], "charge")?),
        ]);
        if let Some(tok) = at_num {
            atom.set("atomic_number", number(tok, "at.num")?);
        }
        atom.set_str("ptype", ptype);
        if let Some(bond_type) = bond_type {
            atom.set_str("class", bond_type);
        }
        let lj = Params::from_pairs(&[
            ("sigma", number(tail[3], "V (sigma)")? * NM_TO_ANGSTROM),
            ("epsilon", number(tail[4], "W (epsilon)")? / KJ_PER_KCAL),
        ]);
        Ok((name, atom, lj))
    }

    /// `[ bondtypes ]` / `[ angletypes ]` / `[ dihedraltypes ]` →
    /// `(category, style, endpoint labels, params)` in molrs units, `X` mapped
    /// to the empty wildcard.
    fn bonded(&self) -> Result<(&'static str, &'static str, Vec<&str>, Params), String> {
        let n = match self.section.as_str() {
            "bondtypes" => 2,
            "angletypes" => 3,
            "dihedraltypes" => 4,
            other => return Err(self.err(&format!("[ {other} ] is not a bonded directive"))),
        };
        let cols: Vec<&str> = self.text.split_whitespace().collect();
        if cols.len() <= n {
            return Err(self.err(&format!(
                "expected {n} type names, a function code and its parameters"
            )));
        }
        let funct: u32 = cols[n].parse().map_err(|_| {
            if n == 4 {
                self.err(
                    "expected `i j k l funct params`; the 2-name `j k funct params` form is \
                     not supported",
                )
            } else {
                self.err(&format!("function code '{}' is not an integer", cols[n]))
            }
        })?;
        let values = cols[n + 1..]
            .iter()
            .map(|tok| {
                tok.parse::<f64>()
                    .map_err(|_| self.err(&format!("parameter '{tok}' is not a number")))
            })
            .collect::<Result<Vec<f64>, String>>()?;
        let labels: Vec<&str> = cols[..n]
            .iter()
            .map(|&label| if label == "X" { "" } else { label })
            .collect();

        let (category, style, params) = match (n, funct) {
            (2, 1) => {
                let [b0, kb] = self.exactly(funct, &values)?;
                (
                    "bond",
                    "harmonic",
                    Params::from_pairs(&[
                        ("r0", b0 * NM_TO_ANGSTROM),
                        ("k", kb / (KJ_PER_KCAL * NM_TO_ANGSTROM * NM_TO_ANGSTROM)),
                    ]),
                )
            }
            (2, 3) => {
                let [b0, d, beta] = self.exactly(funct, &values)?;
                (
                    "bond",
                    "morse",
                    Params::from_pairs(&[
                        ("D", d / KJ_PER_KCAL),
                        ("alpha", beta / NM_TO_ANGSTROM),
                        ("r0", b0 * NM_TO_ANGSTROM),
                    ]),
                )
            }
            (3, 1) => {
                let [theta0, k] = self.exactly(funct, &values)?;
                (
                    "angle",
                    "harmonic",
                    Params::from_pairs(&[("theta0", theta0.to_radians()), ("k", k / KJ_PER_KCAL)]),
                )
            }
            (4, 1 | 4) => {
                let [phase, k, mult] = self.exactly(funct, &values)?;
                let category = if funct == 1 { "dihedral" } else { "improper" };
                (
                    category,
                    "periodic",
                    Params::from_pairs(&[
                        ("k", k / KJ_PER_KCAL),
                        ("periodicity", mult),
                        ("phase", phase.to_radians()),
                    ]),
                )
            }
            (4, 2) => {
                let [xi0, k] = self.exactly(funct, &values)?;
                if xi0 != 0.0 {
                    return Err(self.err(&format!(
                        "function code 2 with xi0 = {xi0} deg is not supported: the signed \
                         GROMACS form and molrs's improper/harmonic (chi = |phi|) agree only \
                         at xi0 = 0"
                    )));
                }
                (
                    "improper",
                    "harmonic",
                    Params::from_pairs(&[("k", k / (2.0 * KJ_PER_KCAL)), ("chi0", 0.0)]),
                )
            }
            (4, 3) => {
                let c: [f64; 6] = self.exactly(funct, &values)?;
                let [k1, k2, k3, k4] = rb_to_opls(c)
                    .map_err(|e| self.err(&e))?
                    .map(|f| f / KJ_PER_KCAL);
                (
                    "dihedral",
                    "opls",
                    Params::from_pairs(&[("k1", k1), ("k2", k2), ("k3", k3), ("k4", k4)]),
                )
            }
            _ => {
                let supported = match n {
                    2 => "1 (harmonic), 3 (Morse)",
                    3 => "1 (harmonic)",
                    _ => {
                        "1 (periodic), 2 (harmonic improper), 3 (Ryckaert-Bellemans), \
                          4 (periodic improper)"
                    }
                };
                return Err(self.err(&format!(
                    "function code {funct} is not supported (supported: {supported})"
                )));
            }
        };
        Ok((category, style, labels, params))
    }

    /// `values` as exactly `N` parameters of function code `funct`.
    fn exactly<const N: usize>(&self, funct: u32, values: &[f64]) -> Result<[f64; N], String> {
        <[f64; N]>::try_from(values).map_err(|_| {
            self.err(&format!(
                "function code {funct} takes {N} parameters, got {}",
                values.len()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::constants::VACUUM_DIELECTRIC;
    use crate::ff::forcefield::{AtomType, ForceField, PairType, Params, Style, StyleDefs};
    use molrs::units::constants::COULOMB_REAL;
    use std::f64::consts::PI;

    /// `nbfunc 1`, comb-rule 3 (OPLS-AA: geometric σ and ε), `gen-pairs yes`.
    const DEFAULTS: &str = "[ defaults ]\n1  3  yes  0.5  0.5\n";

    /// GROMACS OPLS-AA `ffnonbonded.itp` opls_135 (8-column form).
    const OPLS_135: &str = "opls_135  CT  6  12.011  -0.18  A  0.35  0.276144";

    fn read(text: &str) -> ForceField {
        GromacsTopFfReader::new()
            .read_str(text)
            .unwrap_or_else(|e| panic!("read_str: {e}"))
    }

    fn read_err(text: &str) -> String {
        GromacsTopFfReader::new()
            .read_str(text)
            .expect_err("expected Err from read_str")
    }

    /// `[ defaults ]` (comb-rule 3) then one `[ section ]` holding `rows`.
    fn with_section(section: &str, rows: &str) -> String {
        format!("{DEFAULTS}[ {section} ]\n{rows}\n")
    }

    fn style<'a>(ff: &'a ForceField, category: &str, name: &str) -> &'a Style {
        ff.get_style(category, name)
            .unwrap_or_else(|| panic!("no {category}/{name} style in {:?}", style_keys(ff)))
    }

    fn style_keys(ff: &ForceField) -> Vec<(String, String)> {
        ff.styles()
            .iter()
            .map(|s| (s.category().to_owned(), s.name().to_owned()))
            .collect()
    }

    fn atom_type<'a>(ff: &'a ForceField, name: &str) -> &'a AtomType {
        style(ff, "atom", "full")
            .get_atomtype(name)
            .unwrap_or_else(|| panic!("no atom type {name}"))
    }

    fn lj_self_row<'a>(ff: &'a ForceField, name: &str) -> &'a PairType {
        style(ff, "pair", "lj/cut")
            .get_pairtype(name, None)
            .unwrap_or_else(|| panic!("no lj/cut self row for {name}"))
    }

    /// Every bonded type of `s` as `(endpoints, params)`.
    fn bonded(s: &Style) -> Vec<(Vec<&str>, &Params)> {
        match s.defs() {
            StyleDefs::Bond(v) => v
                .iter()
                .map(|t| (vec![t.itom.as_str(), t.jtom.as_str()], &t.params))
                .collect(),
            StyleDefs::Angle(v) => v
                .iter()
                .map(|t| {
                    (
                        vec![t.itom.as_str(), t.jtom.as_str(), t.ktom.as_str()],
                        &t.params,
                    )
                })
                .collect(),
            StyleDefs::Dihedral(v) => v
                .iter()
                .map(|t| {
                    (
                        vec![
                            t.itom.as_str(),
                            t.jtom.as_str(),
                            t.ktom.as_str(),
                            t.ltom.as_str(),
                        ],
                        &t.params,
                    )
                })
                .collect(),
            StyleDefs::Improper(v) => v
                .iter()
                .map(|t| {
                    (
                        vec![
                            t.itom.as_str(),
                            t.jtom.as_str(),
                            t.ktom.as_str(),
                            t.ltom.as_str(),
                        ],
                        &t.params,
                    )
                })
                .collect(),
            other => panic!("{} is not a bonded style", other.category()),
        }
    }

    /// The params of the only type of `category/name`, with its endpoints.
    fn only_type<'a>(ff: &'a ForceField, category: &str, name: &str) -> (Vec<&'a str>, &'a Params) {
        let mut types = bonded(style(ff, category, name));
        assert_eq!(types.len(), 1, "{category}/{name}: {types:?}");
        types.pop().expect("one type")
    }

    fn assert_param(p: &Params, key: &str, want: f64, tol: f64) {
        let got = p
            .get(key)
            .unwrap_or_else(|| panic!("missing `{key}` in {p:?}"));
        assert!((got - want).abs() < tol, "`{key}`: got {got}, want {want}");
    }

    fn assert_names(err: &str, needles: &[&str]) {
        for needle in needles {
            assert!(err.contains(needle), "error should name `{needle}`: {err}");
        }
    }

    // -- [ defaults ] ----------------------------------------------------------

    #[test]
    fn comb_rule_3_declares_geometric_mixing_on_lj_cut() {
        let ff = read(&with_section("atomtypes", OPLS_135));
        assert_eq!(
            style(&ff, "pair", "lj/cut").params().get_str("mixing"),
            Some("geometric")
        );
    }

    #[test]
    fn comb_rule_2_declares_arithmetic_mixing_on_lj_cut() {
        let text = format!("[ defaults ]\n1  2  yes  0.5  0.5\n[ atomtypes ]\n{OPLS_135}\n");
        let ff = read(&text);
        assert_eq!(
            style(&ff, "pair", "lj/cut").params().get_str("mixing"),
            Some("arithmetic")
        );
    }

    /// 1-2 and 1-3 are excluded; fudgeLJ / fudgeQQ are the 1-4 weights.
    #[test]
    fn fudge_factors_are_the_1_4_special_bond_weights() {
        let ff = read("[ defaults ]\n1  3  yes  0.5  0.8333\n");
        let sb = ff.special_bonds();
        assert_eq!(sb.lj, [0.0, 0.0, 0.5]);
        assert_eq!(sb.coul, [0.0, 0.0, 0.8333]);
    }

    /// comb-rule 1 makes V/W C6/C12, which `lj/cut` does not take.
    #[test]
    fn comb_rule_1_is_an_error_naming_the_value() {
        let err = read_err("[ defaults ]\n1  1  yes  0.5  0.5\n");
        assert_names(&err, &["comb-rule", "1"]);
    }

    /// nbfunc 2 is Buckingham.
    #[test]
    fn nbfunc_2_is_an_error_naming_the_value() {
        let err = read_err("[ defaults ]\n2  3  yes  0.5  0.5\n");
        assert_names(&err, &["nbfunc", "2"]);
    }

    #[test]
    fn gen_pairs_no_is_an_error() {
        let err = read_err("[ defaults ]\n1  3  no  0.5  0.5\n");
        assert_names(&err, &["gen-pairs"]);
    }

    // -- [ atomtypes ] ---------------------------------------------------------

    /// opls_135: σ = 0.35 nm × 10 = 3.5 Å; ε = 0.276144 kJ/mol ÷ 4.184 =
    /// 0.066 kcal/mol, on the `lj/cut` self row, not on the atom type.
    #[test]
    fn atomtypes_row_splits_into_atom_full_and_an_lj_cut_self_row() {
        let ff = read(&with_section("atomtypes", OPLS_135));

        let a = &atom_type(&ff, "opls_135").params;
        assert_eq!(a.get("mass"), Some(12.011));
        assert_eq!(a.get("charge"), Some(-0.18));
        assert_eq!(a.get("atomic_number"), Some(6.0));
        assert_eq!(a.get_str("class"), Some("CT"));
        assert_eq!(a.get_str("ptype"), Some("A"));
        assert_eq!(a.get("sigma"), None, "σ belongs on lj/cut");
        assert_eq!(a.get("epsilon"), None, "ε belongs on lj/cut");

        let lj = lj_self_row(&ff, "opls_135");
        assert_eq!(
            (lj.itom.as_str(), lj.jtom.as_str()),
            ("opls_135", "opls_135")
        );
        assert_param(&lj.params, "sigma", 3.5, 1e-12);
        assert_param(&lj.params, "epsilon", 0.066, 1e-12);
    }

    /// Atom types imply the Coulomb pair style with its constants declared.
    #[test]
    fn atomtypes_declare_coul_cut_with_its_constants() {
        let ff = read(&with_section("atomtypes", OPLS_135));
        let p = style(&ff, "pair", "coul/cut").params();
        assert_eq!(p.get("coulomb"), Some(COULOMB_REAL));
        assert_eq!(p.get("dielectric"), Some(VACUUM_DIELECTRIC));
    }

    #[test]
    fn atomtypes_six_column_row_has_no_bond_type_or_atomic_number() {
        let ff = read(&with_section(
            "atomtypes",
            "opls_135  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert_eq!(p.get_str("class"), None);
        assert_eq!(p.get("atomic_number"), None);
        assert_param(&lj_self_row(&ff, "opls_135").params, "sigma", 3.5, 1e-12);
    }

    /// Seven columns with an integer second token: `name at.num mass …`.
    #[test]
    fn atomtypes_seven_column_row_with_an_integer_reads_the_atomic_number() {
        let ff = read(&with_section(
            "atomtypes",
            "opls_135  6  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert_eq!(p.get("atomic_number"), Some(6.0));
        assert_eq!(p.get_str("class"), None);
        assert_eq!(p.get("mass"), Some(12.011));
    }

    /// Seven columns with a non-integer second token: `name bond_type mass …`.
    #[test]
    fn atomtypes_seven_column_row_with_a_label_reads_the_bond_type() {
        let ff = read(&with_section(
            "atomtypes",
            "opls_135  CT  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert_eq!(p.get_str("class"), Some("CT"));
        assert_eq!(p.get("atomic_number"), None);
        assert_eq!(p.get("mass"), Some(12.011));
    }

    #[test]
    fn atomtypes_eight_column_row_with_a_non_integer_atomic_number_is_an_error() {
        let err = read_err(&with_section(
            "atomtypes",
            "opls_135  CT  C  12.011  -0.18  A  0.35  0.276144",
        ));
        assert_names(&err, &["opls_135"]);
    }

    #[test]
    fn atomtypes_row_with_four_leading_tokens_is_an_error() {
        let err = read_err(&with_section(
            "atomtypes",
            "opls_135  CT  6  extra  12.011  -0.18  A  0.35  0.276144",
        ));
        assert_names(&err, &["opls_135"]);
    }

    #[test]
    fn atomtypes_row_with_an_unknown_ptype_is_an_error() {
        let err = read_err(&with_section(
            "atomtypes",
            "opls_135  12.011  -0.18  X  0.35  0.276144",
        ));
        assert_names(&err, &["opls_135"]);
    }

    /// V/W mean σ/ε only under the comb-rule `[ defaults ]` declares.
    #[test]
    fn atomtypes_without_defaults_is_an_error() {
        let err = read_err(&format!("[ atomtypes ]\n{OPLS_135}\n"));
        assert_names(&err, &["defaults"]);
    }

    // -- [ bondtypes ] ---------------------------------------------------------

    /// b₀ = 0.109 nm × 10 = 1.09 Å; k = 284512 kJ/mol/nm² ÷ 418.4 = 680
    /// kcal/mol/Å² (both ½k forms).
    #[test]
    fn bondtypes_funct_1_is_bond_harmonic_in_molrs_units() {
        let ff = read(&with_section("bondtypes", "CT  HC  1  0.10900  284512.0"));
        let (ends, p) = only_type(&ff, "bond", "harmonic");
        assert_eq!(ends, ["CT", "HC"]);
        assert_param(p, "r0", 1.09, 1e-12);
        assert_param(p, "k", 680.0, 1e-9);
    }

    /// D = 400 kJ/mol ÷ 4.184 = 95.602294455066… kcal/mol; α = 20 nm⁻¹ ÷ 10 =
    /// 2 Å⁻¹; b₀ = 0.1529 nm × 10 = 1.529 Å.
    #[test]
    fn bondtypes_funct_3_is_bond_morse_in_molrs_units() {
        let ff = read(&with_section("bondtypes", "CT  CT  3  0.1529  400.0  20.0"));
        let (ends, p) = only_type(&ff, "bond", "morse");
        assert_eq!(ends, ["CT", "CT"]);
        assert_param(p, "D", 95.602_294_455_066_9, 1e-9);
        assert_param(p, "alpha", 2.0, 1e-12);
        assert_param(p, "r0", 1.529, 1e-12);
    }

    #[test]
    fn bondtypes_funct_2_is_an_error_naming_section_and_code() {
        let err = read_err(&with_section("bondtypes", "CT  HC  2  0.10900  284512.0"));
        assert_names(&err, &["bondtypes", "2", "CT"]);
    }

    /// Two identical rows of one label pair are one type.
    #[test]
    fn two_equal_bondtypes_rows_define_one_type() {
        let ff = read(&with_section(
            "bondtypes",
            "CT  HC  1  0.10900  284512.0\nCT  HC  1  0.10900  284512.0",
        ));
        assert_eq!(ff.get_bondtypes().len(), 1);
    }

    /// Two rows of one label pair with different k: a conflict, not
    /// last-writer-wins.
    #[test]
    fn two_bondtypes_rows_of_one_pair_with_different_k_are_an_error() {
        let err = read_err(&with_section(
            "bondtypes",
            "CT  HC  1  0.10900  284512.0\nCT  HC  1  0.10900  300000.0",
        ));
        assert_names(&err, &["CT-HC"]);
    }

    // -- [ angletypes ] --------------------------------------------------------

    /// θ₀ = 107.8° → 107.8·π/180 rad; k = 276.144 kJ/mol/rad² ÷ 4.184 = 66.
    #[test]
    fn angletypes_funct_1_is_angle_harmonic_in_molrs_units() {
        let ff = read(&with_section(
            "angletypes",
            "HC  CT  HC  1  107.800  276.144",
        ));
        let (ends, p) = only_type(&ff, "angle", "harmonic");
        assert_eq!(ends, ["HC", "CT", "HC"]);
        assert_param(p, "theta0", 107.8 * PI / 180.0, 1e-12);
        assert_param(p, "k", 66.0, 1e-9);
    }

    #[test]
    fn angletypes_funct_5_is_an_error_naming_section_and_code() {
        let err = read_err(&with_section(
            "angletypes",
            "HC  CT  HC  5  107.800  276.144  0.0  0.0",
        ));
        assert_names(&err, &["angletypes", "5", "HC"]);
    }

    // -- [ dihedraltypes ] -----------------------------------------------------

    /// GROMACS OPLS-AA HC-CT-CT-HC, RB in kJ/mol. F3 = −C3/2 = 1.2552 kJ/mol
    /// ÷ 4.184 = 0.3 kcal/mol; F1 = −2·1.8828 + 1.5·2.5104 = 0; F2 = F4 = 0.
    #[test]
    fn dihedraltypes_funct_3_is_dihedral_opls_in_kcal() {
        let ff = read(&with_section(
            "dihedraltypes",
            "HC  CT  CT  HC  3  0.62760  1.88280  0.00000  -2.51040  0.00000  0.00000",
        ));
        let (ends, p) = only_type(&ff, "dihedral", "opls");
        assert_eq!(ends, ["HC", "CT", "CT", "HC"]);
        for (key, want) in [("k1", 0.0), ("k2", 0.0), ("k3", 0.3), ("k4", 0.0)] {
            assert_param(p, key, want, 1e-12);
        }
    }

    /// GROMACS `X` is the canonical empty-endpoint wildcard.
    #[test]
    fn x_endpoint_is_the_empty_wildcard() {
        let ff = read(&with_section(
            "dihedraltypes",
            "X  CT  CT  X  3  0.62760  1.88280  0.00000  -2.51040  0.00000  0.00000",
        ));
        let (ends, _) = only_type(&ff, "dihedral", "opls");
        assert_eq!(ends, ["", "CT", "CT", ""]);
    }

    /// k = 4.184 kJ/mol ÷ 4.184 = 1 kcal/mol; n = 3; φ_s = 0.
    #[test]
    fn dihedraltypes_funct_1_is_dihedral_periodic() {
        let ff = read(&with_section(
            "dihedraltypes",
            "CT  CT  CT  CT  1  0.0  4.184  3",
        ));
        let (ends, p) = only_type(&ff, "dihedral", "periodic");
        assert_eq!(ends, ["CT", "CT", "CT", "CT"]);
        assert_param(p, "k", 1.0, 1e-12);
        assert_param(p, "periodicity", 3.0, 1e-12);
        assert_param(p, "phase", 0.0, 1e-12);
    }

    /// k = 10.46 kJ/mol ÷ 4.184 = 2.5 kcal/mol; n = 2; φ_s = 180° = π.
    #[test]
    fn dihedraltypes_funct_4_is_improper_periodic() {
        let ff = read(&with_section(
            "dihedraltypes",
            "X  X  N  H  4  180.0  10.46  2",
        ));
        let (ends, p) = only_type(&ff, "improper", "periodic");
        assert_eq!(ends, ["", "", "N", "H"]);
        assert_param(p, "k", 2.5, 1e-12);
        assert_param(p, "periodicity", 2.0, 1e-12);
        assert_param(p, "phase", PI, 1e-12);
    }

    /// ½k_ξ(ξ−0)² = K(χ−0)² with K = 167.36 ÷ (2·4.184) = 20 kcal/mol/rad²;
    /// χ₀ = 0.
    #[test]
    fn dihedraltypes_funct_2_at_zero_is_improper_harmonic() {
        let ff = read(&with_section("dihedraltypes", "X  X  C  O  2  0.0  167.36"));
        let (ends, p) = only_type(&ff, "improper", "harmonic");
        assert_eq!(ends, ["", "", "C", "O"]);
        assert_param(p, "k", 20.0, 1e-12);
        assert_eq!(p.get("chi0"), Some(0.0));
    }

    /// Signed ½k(ξ−ξ₀)² and unsigned K(|φ|−χ₀)² agree only at ξ₀ = 0.
    #[test]
    fn dihedraltypes_funct_2_off_zero_is_an_error() {
        let err = read_err(&with_section(
            "dihedraltypes",
            "X  X  C  O  2  10.0  167.36",
        ));
        assert_names(&err, &["dihedraltypes", "2", "10"]);
    }

    /// Multi-term periodic (funct 9) is not modelled.
    #[test]
    fn dihedraltypes_funct_9_is_an_error_naming_section_and_code() {
        let err = read_err(&with_section(
            "dihedraltypes",
            "CT  CT  CT  CT  9  0.0  4.184  3",
        ));
        assert_names(&err, &["dihedraltypes", "9"]);
    }

    /// ΣC = 1 kJ/mol is a constant offset the OPLS form cannot hold.
    #[test]
    fn dihedraltypes_funct_3_with_a_nonzero_sum_is_an_error() {
        let err = read_err(&with_section(
            "dihedraltypes",
            "HC  CT  CT  HC  3  1.00000  0.00000  0.00000  0.00000  0.00000  0.00000",
        ));
        assert_names(&err, &["dihedraltypes", "HC"]);
    }

    /// The 2-name (`j k`) dihedraltypes form is refused.
    #[test]
    fn two_name_dihedraltypes_row_is_an_error() {
        let err = read_err(&with_section("dihedraltypes", "CT  CT  1  0.0  4.184  3"));
        assert_names(&err, &["dihedraltypes"]);
    }

    // -- refused sections --------------------------------------------------------

    #[test]
    fn pairtypes_is_an_error_naming_the_section() {
        let err = read_err(&with_section(
            "pairtypes",
            "opls_135  opls_135  1  0.35  0.276144",
        ));
        assert_names(&err, &["pairtypes"]);
    }

    #[test]
    fn nonbond_params_is_an_error_naming_the_section() {
        let err = read_err(&with_section(
            "nonbond_params",
            "opls_135  opls_140  1  0.3  0.2",
        ));
        assert_names(&err, &["nonbond_params"]);
    }

    #[test]
    fn constrainttypes_is_an_error_naming_the_section() {
        let err = read_err(&with_section("constrainttypes", "CT  HC  1  0.109"));
        assert_names(&err, &["constrainttypes"]);
    }

    #[test]
    fn skipped_constrainttypes_is_read_past() {
        let text = format!(
            "{DEFAULTS}[ constrainttypes ]\nCT  HC  1  0.109\n\
             [ bondtypes ]\nCT  HC  1  0.10900  284512.0\n"
        );
        let ff = GromacsTopFfReader::new()
            .with_skipped_directive("constrainttypes")
            .read_str(&text)
            .unwrap_or_else(|e| panic!("skipped section still refused: {e}"));
        assert_eq!(ff.get_bondtypes().len(), 1);
    }

    /// Skipping one section does not skip another.
    #[test]
    fn skipping_one_section_still_refuses_another() {
        let text = format!(
            "{DEFAULTS}[ constrainttypes ]\nCT  HC  1  0.109\n\
             [ pairtypes ]\nopls_135  opls_135  1  0.35  0.276144\n"
        );
        let err = GromacsTopFfReader::new()
            .with_skipped_directive("constrainttypes")
            .read_str(&text)
            .expect_err("pairtypes is not skipped");
        assert_names(&err, &["pairtypes"]);
    }

    #[test]
    fn unknown_section_is_an_error_naming_it() {
        let err = read_err(&with_section("mystery_section", "a  b  c"));
        assert_names(&err, &["mystery_section"]);
    }

    /// `[ atoms ]` is topology: refused with the plain-text hint.
    #[test]
    fn atoms_section_is_an_error_naming_read_top() {
        let err = read_err(&with_section(
            "atoms",
            "1  opls_135  1  LIG  C1  1  -0.18  12.011",
        ));
        assert_names(&err, &["atoms", "read_top"]);
    }

    #[test]
    fn every_molecule_section_is_an_error_naming_read_top() {
        for (section, row) in [
            ("moleculetype", "LIG  3"),
            ("bonds", "1  2  1"),
            ("pairs", "1  4  1"),
            ("angles", "1  2  3  1"),
            ("dihedrals", "1  2  3  4  3"),
            ("exclusions", "1  2"),
            ("settles", "1  1  0.1  0.1633"),
            ("system", "test"),
            ("molecules", "LIG  1"),
        ] {
            let err = read_err(&with_section(section, row));
            assert_names(&err, &[section, "read_top"]);
        }
    }

    #[test]
    fn skipped_molecule_sections_are_read_past() {
        let text = format!(
            "{DEFAULTS}[ bondtypes ]\nCT  HC  1  0.10900  284512.0\n\
             [ moleculetype ]\nLIG  3\n\
             [ atoms ]\n1  opls_135  1  LIG  C1  1  -0.18  12.011\n"
        );
        let ff = GromacsTopFfReader::new()
            .with_skipped_directive("moleculetype")
            .with_skipped_directive("atoms")
            .read_str(&text)
            .unwrap_or_else(|e| panic!("skipped sections still refused: {e}"));
        assert_eq!(ff.get_bondtypes().len(), 1);
        assert!(ff.get_atomtypes().is_empty(), "[ atoms ] defined a type");
    }

    // -- preprocessor ------------------------------------------------------------

    const ROW_A: &str = "CT  HC  1  0.10900  284512.0";
    const ROW_B: &str = "CT  CT  1  0.15290  224262.4";

    /// The bond label pairs read from `body`, which sits in `[ bondtypes ]`.
    fn bond_pairs(body: &str) -> Vec<(String, String)> {
        let ff = read(&format!("{DEFAULTS}[ bondtypes ]\n{body}\n"));
        ff.get_bondtypes()
            .iter()
            .map(|t| (t.itom.clone(), t.jtom.clone()))
            .collect()
    }

    fn ct_hc() -> (String, String) {
        ("CT".to_owned(), "HC".to_owned())
    }

    fn ct_ct() -> (String, String) {
        ("CT".to_owned(), "CT".to_owned())
    }

    #[test]
    fn ifdef_block_is_read_when_the_define_comes_first() {
        let got = bond_pairs(&format!("#define FOO\n#ifdef FOO\n{ROW_A}\n#endif"));
        assert_eq!(got, [ct_hc()]);
    }

    #[test]
    fn ifdef_block_is_skipped_without_the_define() {
        let got = bond_pairs(&format!("#ifdef FOO\n{ROW_A}\n#endif\n{ROW_B}"));
        assert_eq!(got, [ct_ct()]);
    }

    #[test]
    fn ifdef_block_is_skipped_when_the_define_comes_after() {
        let got = bond_pairs(&format!(
            "#ifdef FOO\n{ROW_A}\n#endif\n#define FOO\n{ROW_B}"
        ));
        assert_eq!(got, [ct_ct()]);
    }

    #[test]
    fn ifndef_block_is_read_without_the_define() {
        let got = bond_pairs(&format!("#ifndef FOO\n{ROW_A}\n#endif"));
        assert_eq!(got, [ct_hc()]);
    }

    #[test]
    fn ifndef_block_is_skipped_with_the_define() {
        let got = bond_pairs(&format!(
            "#define FOO\n#ifndef FOO\n{ROW_A}\n#endif\n{ROW_B}"
        ));
        assert_eq!(got, [ct_ct()]);
    }

    #[test]
    fn else_takes_the_other_branch() {
        let block = format!("#ifdef FOO\n{ROW_A}\n#else\n{ROW_B}\n#endif");
        assert_eq!(bond_pairs(&block), [ct_ct()]);
        assert_eq!(bond_pairs(&format!("#define FOO\n{block}")), [ct_hc()]);
    }

    /// A false outer block hides a true inner one; a true outer block
    /// evaluates its inner one.
    #[test]
    fn conditionals_nest() {
        let got = bond_pairs(&format!(
            "#define FOO\n\
             #ifdef BAR\n#ifdef FOO\n{ROW_A}\n#endif\n#endif\n\
             #ifdef FOO\n#ifndef BAR\n{ROW_B}\n#endif\n#endif"
        ));
        assert_eq!(got, [ct_ct()]);
    }

    #[test]
    fn undef_removes_a_define() {
        let got = bond_pairs(&format!(
            "#define FOO\n#undef FOO\n#ifdef FOO\n{ROW_A}\n#endif\n{ROW_B}"
        ));
        assert_eq!(got, [ct_ct()]);
    }

    /// A macro with a body is recorded, never expanded or refused.
    #[test]
    fn define_with_a_body_is_accepted() {
        let got = bond_pairs(&format!(
            "#define improper_Z_N_X_Y 180.0 4.6024 2\n#ifdef improper_Z_N_X_Y\n{ROW_A}\n#endif"
        ));
        assert_eq!(got, [ct_hc()]);
    }

    #[test]
    fn if_directive_is_an_error() {
        let err = read_err(&format!("{DEFAULTS}#if FOO\n#endif\n"));
        assert_names(&err, &["#if"]);
    }

    #[test]
    fn elif_directive_is_an_error() {
        let err = read_err(&format!("{DEFAULTS}#ifdef FOO\n#elif BAR\n#endif\n"));
        assert_names(&err, &["#elif"]);
    }

    /// `#include` resolves relative to the including file, at every depth.
    #[test]
    fn include_resolves_relative_to_the_including_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sub = dir.path().join("ff");
        std::fs::create_dir(&sub).expect("mkdir");
        std::fs::write(
            dir.path().join("forcefield.itp"),
            format!("{DEFAULTS}#include \"ff/ffbonded.itp\"\n"),
        )
        .expect("write top");
        std::fs::write(sub.join("ffbonded.itp"), "#include \"more.itp\"\n").expect("write mid");
        std::fs::write(sub.join("more.itp"), format!("[ bondtypes ]\n{ROW_A}\n"))
            .expect("write leaf");

        let path = dir.path().join("forcefield.itp");
        let ff = GromacsTopFfReader::new()
            .with_include(true)
            .read(path.to_str().expect("utf-8 path"))
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(ff.get_bondtypes().len(), 1);
    }
}
