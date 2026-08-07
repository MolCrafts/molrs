//! LAMMPS force-field writer (the `*.ff` include next to a data file).
//!
//! Inverse of [`super::super::readers::lammps::LammpsFfReader`]: takes a molrs
//! [`ForceField`] in molrs units (Å, kcal/mol, **radians**, e; harmonic
//! stiffness in the `½k(x−x₀)²` form) and emits an AMBER/GAFF-flavour LAMMPS
//! include in LAMMPS `real` units:
//!
//! ```text
//! pair_style lj/cut/coul/cut 10.0 10.0
//! pair_coeff c3 c3 0.107800 3.397710          # epsilon(kcal/mol) sigma(Å)
//! bond_style harmonic
//! bond_coeff c3-c3 228.890000 1.535400        # K(kcal/mol/Å²) r0(Å)  — K = k/2
//! angle_style harmonic
//! angle_coeff c3-c3-oh 76.790000 109.660000   # K  theta0(deg)
//! dihedral_style fourier
//! dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.0 # m  K1 n1 d1(deg) ...
//! ```
//!
//! # Units (molrs store → LAMMPS file)
//!
//! Store is **real** (Å, kcal/mol, rad; `½k` harmonic form) for force fields
//! read from physical styles, or **lj** pass-through when the file was already
//! reduced. Writing always goes through
//! [`LammpsUnitSystem`](crate::ff::forcefield::lammps_units::LammpsUnitSystem)
//! (`store → lj hub → target`) — never ad-hoc eV/kcal factors.
//!
//! Form map (independent of unit style):
//! - harmonic bond/angle/improper: `K = k/2` (molrs `½k` → LAMMPS `K`);
//! - angle-valued params (`theta0`, dihedral phase, improper `chi0`) are stored
//!   in **radians** and written in **degrees** (LAMMPS file convention for all
//!   of real/metal/lj).
//!
//! Default write target is LAMMPS **`real`**. Set
//! [`LammpsWriteOptions::units`] for `metal` or `lj`.
//!
//! # Pair style layout
//!
//! The reader splits a combined `lj/cut/coul/*` kernel into `lj/cut` + `coul/cut`
//! styles. This writer recombines that pair into one `pair_style lj/cut/coul/cut`
//! line so LAMMPS keeps geometric mixing on LJ (writing them as `hybrid` with a
//! `pair_coeff * * coul/cut` wildcard marks every cross pair as explicit and
//! defeats mixing). A force field that already holds a single combined-style
//! name, or only one of the two halves, is written as-is.

use std::collections::{HashMap, HashSet};

use super::ForceFieldWriter;
use crate::ff::forcefield::lammps_units::{
    LammpsUnitSystem, LammpsUnits, molrs_half_k_to_lammps_k,
};
use crate::ff::forcefield::{
    AngleType, BondType, ForceField, ImproperType, PairType, Params, Style, StyleDefs,
};

/// Default pair cutoff (Å / reduced σ) when a style carries none — keeps the
/// written include a legal LAMMPS command rather than a bare `pair_style lj/cut`.
const DEFAULT_PAIR_CUTOFF: f64 = 10.0;

/// Optional filters / formatting for [`LammpsFfWriter`].
#[derive(Debug, Clone)]
pub struct LammpsWriteOptions {
    /// Decimal places for floating-point coefficients (default 6).
    pub precision: usize,
    /// When true, omit the `pair_style` line (caller sets it in the input script).
    pub skip_pair_style: bool,
    /// LAMMPS `units` style for the written include (default [`LammpsUnits::Real`]).
    pub units: LammpsUnits,
    /// Restrict pair coeffs to pairs whose atom types are a subset of this set.
    pub atom_types: Option<HashSet<String>>,
    pub bond_types: Option<HashSet<String>>,
    pub angle_types: Option<HashSet<String>>,
    pub dihedral_types: Option<HashSet<String>>,
    pub improper_types: Option<HashSet<String>>,
    /// Optional ForceField type-name → 1-based LAMMPS type id (from Frame).
    /// Used by [`LammpsFfWriter::write_data_coeffs_str`]. Integer-parseable
    /// names (`\"3\"`, `\"3-3\"`) resolve without this map.
    pub type_ids: Option<HashMap<String, u32>>,
}

impl Default for LammpsWriteOptions {
    fn default() -> Self {
        Self {
            precision: 6,
            skip_pair_style: false,
            units: LammpsUnits::Real,
            atom_types: None,
            bond_types: None,
            angle_types: None,
            dihedral_types: None,
            improper_types: None,
            type_ids: None,
        }
    }
}

/// Conversion context: store → file units via the lj hub + form maps.
struct WriteUnits {
    sys: LammpsUnitSystem,
    file: LammpsUnits,
}

impl WriteUnits {
    fn new(file: LammpsUnits) -> Result<Self, String> {
        Ok(Self {
            sys: LammpsUnitSystem::canonical().map_err(|e| format!("lammps unit system: {e}"))?,
            file,
        })
    }

    fn energy(&self, store: f64) -> Result<f64, String> {
        self.sys.from_store_energy(store, self.file)
    }

    fn length(&self, store: f64) -> Result<f64, String> {
        self.sys.from_store_length(store, self.file)
    }

    /// molrs `½k` bond stiffness → LAMMPS file `K` (form map + unit convert).
    fn bond_k(&self, k_molrs: f64) -> Result<f64, String> {
        let k_lammps_store = molrs_half_k_to_lammps_k(k_molrs);
        self.sys.from_store_bond_k_lammps(k_lammps_store, self.file)
    }

    /// molrs `½k` angle/improper stiffness → LAMMPS file `K`.
    fn angle_k(&self, k_molrs: f64) -> Result<f64, String> {
        let k_lammps_store = molrs_half_k_to_lammps_k(k_molrs);
        self.sys
            .from_store_angle_k_lammps(k_lammps_store, self.file)
    }
}

/// Writer for a LAMMPS force-field include (`*.ff`), AMBER/GAFF flavour.
#[derive(Debug, Clone)]
pub struct LammpsFfWriter {
    options: LammpsWriteOptions,
}

impl LammpsFfWriter {
    /// Writer with default options (6 decimal places, no type filters).
    pub fn new() -> Self {
        Self {
            options: LammpsWriteOptions::default(),
        }
    }

    /// Writer with explicit options (filters, precision, skip pair style).
    pub fn with_options(options: LammpsWriteOptions) -> Self {
        Self { options }
    }

    /// Emit data-file `* Coeffs` sections only (no `units` / `*_style` lines).
    ///
    /// Numbers use the same form map and [`LammpsWriteOptions::units`] as
    /// [`ForceFieldWriter::write_str`]. Type ids come from integer-parseable
    /// type names or [`LammpsWriteOptions::type_ids`] (Frame-derived).
    pub fn write_data_coeffs_str(&self, ff: &ForceField) -> Result<String, String> {
        let units = WriteUnits::new(self.options.units)?;
        let mut lines: Vec<String> = Vec::new();
        write_data_pair_coeffs(&mut lines, ff, &self.options, &units)?;
        write_data_bond_coeffs(&mut lines, ff, &self.options, &units)?;
        write_data_angle_coeffs(&mut lines, ff, &self.options, &units)?;
        write_data_dihedral_coeffs(&mut lines, ff, &self.options, &units)?;
        write_data_improper_coeffs(&mut lines, ff, &self.options, &units)?;
        Ok(lines.concat())
    }
}

impl Default for LammpsFfWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ForceFieldWriter for LammpsFfWriter {
    fn write_str(&self, ff: &ForceField) -> Result<String, String> {
        let units = WriteUnits::new(self.options.units)?;
        let mut lines: Vec<String> = Vec::new();
        lines.push("# LAMMPS force field generated by molrs\n".to_owned());
        lines.push(format!("units {}\n", self.options.units.as_str()));
        lines.push("\n".to_owned());

        write_pair_section(&mut lines, ff, &self.options, &units)?;
        write_bond_section(&mut lines, ff, &self.options, &units)?;
        write_angle_section(&mut lines, ff, &self.options, &units)?;
        write_dihedral_section(&mut lines, ff, &self.options, &units)?;
        write_improper_section(&mut lines, ff, &self.options, &units)?;

        Ok(lines.concat())
    }
}

/// Resolve a ForceField type name to a 1-based LAMMPS data type id.
fn data_type_id(name: &str, map: Option<&HashMap<String, u32>>) -> Result<u32, String> {
    if let Some(m) = map
        && let Some(&id) = m.get(name)
    {
        return Ok(id);
    }
    if let Ok(id) = name.parse::<u32>()
        && id >= 1
    {
        return Ok(id);
    }
    let parts: Vec<&str> = name.split('-').filter(|p| !p.is_empty()).collect();
    if !parts.is_empty() && parts.iter().all(|p| p.parse::<u32>().is_ok()) {
        let first: u32 = parts[0].parse().unwrap();
        if first >= 1 && parts.iter().all(|p| p.parse::<u32>().ok() == Some(first)) {
            return Ok(first);
        }
        // Distinct numeric segments (rare as a type *name*): prefer map, else first.
        if first >= 1 {
            return Ok(first);
        }
    }
    if let Some(m) = map {
        let head = name.split('-').next().unwrap_or(name);
        if let Some(&id) = m.get(head) {
            return Ok(id);
        }
    }
    Err(format!(
        "cannot resolve LAMMPS type id for `{name}`: need an integer type name \
         (or repeated `N-N`) or a Frame-derived entry in type_ids"
    ))
}

fn push_data_section(lines: &mut Vec<String>, heading: &str, rows: &mut [(u32, String)]) {
    if rows.is_empty() {
        return;
    }
    rows.sort_by_key(|(id, _)| *id);
    lines.push(format!("{heading}\n\n"));
    for (id, body) in rows.iter() {
        lines.push(format!("{id} {body}\n"));
    }
    lines.push("\n".to_owned());
}

fn write_data_pair_coeffs(
    lines: &mut Vec<String>,
    ff: &ForceField,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let mut rows: Vec<(u32, String)> = Vec::new();
    let mut seen = HashSet::new();
    for style in ff.get_styles("pair") {
        let StyleDefs::Pair(types) = &style.defs else {
            continue;
        };
        for t in types {
            if !pair_included(t, opts) {
                continue;
            }
            let Some(eps_store) = t.params.get("epsilon") else {
                continue;
            };
            let Some(sigma_store) = t.params.get("sigma") else {
                continue;
            };
            if !seen.insert(t.name.clone()) {
                continue;
            }
            let id = data_type_id(&t.name, opts.type_ids.as_ref())?;
            let eps = units.energy(eps_store)?;
            let sigma = units.length(sigma_store)?;
            rows.push((
                id,
                format!(
                    "{} {}",
                    fmt_num(eps, opts.precision),
                    fmt_num(sigma, opts.precision)
                ),
            ));
        }
    }
    push_data_section(lines, "Pair Coeffs", &mut rows);
    Ok(())
}

fn write_data_bond_coeffs(
    lines: &mut Vec<String>,
    ff: &ForceField,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let mut rows: Vec<(u32, String)> = Vec::new();
    let mut seen = HashSet::new();
    for style in ff.get_styles("bond") {
        if style.name != "harmonic" {
            return Err(format!(
                "unsupported bond_style `{}` for data coeffs (expected `harmonic`)",
                style.name
            ));
        }
        let StyleDefs::Bond(types) = &style.defs else {
            continue;
        };
        for t in types {
            if !name_included(&t.name, &opts.bond_types) {
                continue;
            }
            if !seen.insert(t.name.clone()) {
                continue;
            }
            let id = data_type_id(&t.name, opts.type_ids.as_ref())?;
            let (k, r0) = bond_lammps_params(t, units)?;
            rows.push((
                id,
                format!(
                    "{} {}",
                    fmt_num(k, opts.precision),
                    fmt_num(r0, opts.precision)
                ),
            ));
        }
    }
    push_data_section(lines, "Bond Coeffs", &mut rows);
    Ok(())
}

fn write_data_angle_coeffs(
    lines: &mut Vec<String>,
    ff: &ForceField,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let mut rows: Vec<(u32, String)> = Vec::new();
    let mut seen = HashSet::new();
    for style in ff.get_styles("angle") {
        if style.name != "harmonic" {
            return Err(format!(
                "unsupported angle_style `{}` for data coeffs (expected `harmonic`)",
                style.name
            ));
        }
        let StyleDefs::Angle(types) = &style.defs else {
            continue;
        };
        for t in types {
            if !name_included(&t.name, &opts.angle_types) {
                continue;
            }
            if !seen.insert(t.name.clone()) {
                continue;
            }
            let id = data_type_id(&t.name, opts.type_ids.as_ref())?;
            let (k, theta0_deg) = angle_lammps_params(t, units)?;
            rows.push((
                id,
                format!(
                    "{} {}",
                    fmt_num(k, opts.precision),
                    fmt_num(theta0_deg, opts.precision)
                ),
            ));
        }
    }
    push_data_section(lines, "Angle Coeffs", &mut rows);
    Ok(())
}

fn write_data_dihedral_coeffs(
    lines: &mut Vec<String>,
    ff: &ForceField,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let mut rows: Vec<(u32, String)> = Vec::new();
    let mut seen = HashSet::new();
    for style in ff.get_styles("dihedral") {
        let StyleDefs::Dihedral(types) = &style.defs else {
            continue;
        };
        match style.name.as_str() {
            "harmonic" => {
                for t in types {
                    if !name_included(&t.name, &opts.dihedral_types) {
                        continue;
                    }
                    if !seen.insert(t.name.clone()) {
                        continue;
                    }
                    let id = data_type_id(&t.name, opts.type_ids.as_ref())?;
                    let k_store = t
                        .params
                        .get("k")
                        .ok_or_else(|| format!("dihedral type `{}` missing `k`", t.name))?;
                    let d_rad = t.params.get("d").unwrap_or(0.0);
                    let n = t.params.get("n").unwrap_or(1.0);
                    let k = units.energy(k_store)?;
                    rows.push((
                        id,
                        format!(
                            "{} {} {}",
                            fmt_num(k, opts.precision),
                            fmt_num(d_rad.to_degrees(), opts.precision),
                            n.round() as i64
                        ),
                    ));
                }
            }
            "opls" => {
                for t in types {
                    if !name_included(&t.name, &opts.dihedral_types) {
                        continue;
                    }
                    if !seen.insert(t.name.clone()) {
                        continue;
                    }
                    let id = data_type_id(&t.name, opts.type_ids.as_ref())?;
                    let f1 = units.energy(
                        t.params
                            .get("f1")
                            .or_else(|| t.params.get("c1"))
                            .unwrap_or(0.0),
                    )?;
                    let f2 = units.energy(
                        t.params
                            .get("f2")
                            .or_else(|| t.params.get("c2"))
                            .unwrap_or(0.0),
                    )?;
                    let f3 = units.energy(
                        t.params
                            .get("f3")
                            .or_else(|| t.params.get("c3"))
                            .unwrap_or(0.0),
                    )?;
                    let f4 = units.energy(
                        t.params
                            .get("f4")
                            .or_else(|| t.params.get("c4"))
                            .unwrap_or(0.0),
                    )?;
                    rows.push((
                        id,
                        format!(
                            "{} {} {} {}",
                            fmt_num(f1, opts.precision),
                            fmt_num(f2, opts.precision),
                            fmt_num(f3, opts.precision),
                            fmt_num(f4, opts.precision)
                        ),
                    ));
                }
            }
            "fourier" => {
                for t in types {
                    if !name_included(&t.name, &opts.dihedral_types) {
                        continue;
                    }
                    if !seen.insert(t.name.clone()) {
                        continue;
                    }
                    let id = data_type_id(&t.name, opts.type_ids.as_ref())?;
                    let terms = fourier_terms(&t.params, &t.name, units)?;
                    let m = terms.len();
                    let mut parts = vec![format!("{m}")];
                    for (k, n, d_deg) in terms {
                        parts.push(fmt_num(k, opts.precision));
                        parts.push(format!("{}", n.round() as i64));
                        parts.push(fmt_num(d_deg, opts.precision));
                    }
                    rows.push((id, parts.join(" ")));
                }
            }
            other => {
                return Err(format!(
                    "unsupported dihedral_style `{other}` for data coeffs"
                ));
            }
        }
    }
    push_data_section(lines, "Dihedral Coeffs", &mut rows);
    Ok(())
}

fn write_data_improper_coeffs(
    lines: &mut Vec<String>,
    ff: &ForceField,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let mut rows: Vec<(u32, String)> = Vec::new();
    let mut seen = HashSet::new();
    for style in ff.get_styles("improper") {
        if style.name != "harmonic" {
            return Err(format!(
                "unsupported improper_style `{}` for data coeffs (expected `harmonic`)",
                style.name
            ));
        }
        let StyleDefs::Improper(types) = &style.defs else {
            continue;
        };
        for t in types {
            if !name_included(&t.name, &opts.improper_types) {
                continue;
            }
            if !seen.insert(t.name.clone()) {
                continue;
            }
            let id = data_type_id(&t.name, opts.type_ids.as_ref())?;
            let (k, chi0_deg) = improper_lammps_params(t, units)?;
            rows.push((
                id,
                format!(
                    "{} {}",
                    fmt_num(k, opts.precision),
                    fmt_num(chi0_deg, opts.precision)
                ),
            ));
        }
    }
    push_data_section(lines, "Improper Coeffs", &mut rows);
    Ok(())
}

// ── pair ─────────────────────────────────────────────────────────────────────

fn write_pair_section(
    lines: &mut Vec<String>,
    ff: &ForceField,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let styles = ff.get_styles("pair");
    if styles.is_empty() {
        return Ok(());
    }

    // Reader always builds lj/cut + coul/cut; recombine for a correct write-back.
    if is_split_lj_coulomb(&styles) {
        write_combined_lj_coulomb(lines, &styles, opts, units)?;
        return Ok(());
    }

    if styles.len() == 1 {
        let s = styles[0];
        if !opts.skip_pair_style {
            let params = pair_style_cutoffs(s, units)?;
            lines.push(format!(
                "pair_style {} {}\n",
                s.name,
                format_nums(&params, opts.precision)
            ));
            lines.push("\n".to_owned());
        }
        write_pair_coeffs(lines, s, /*hybrid_substyle=*/ None, opts, units)?;
        return Ok(());
    }

    // Genuinely independent sub-styles → hybrid with per-substyle cutoffs.
    if !opts.skip_pair_style {
        let mut sub = Vec::new();
        for s in &styles {
            let cuts = pair_style_cutoffs(s, units)?;
            if cuts.is_empty() {
                sub.push(s.name.clone());
            } else {
                sub.push(format!("{} {}", s.name, format_nums(&cuts, opts.precision)));
            }
        }
        lines.push(format!("pair_style hybrid {}\n", sub.join(" ")));
        lines.push("\n".to_owned());
    }
    let mut seen = HashSet::new();
    for s in &styles {
        write_pair_coeffs_dedup(lines, s, Some(s.name.as_str()), opts, units, &mut seen)?;
    }
    lines.push("\n".to_owned());
    Ok(())
}

fn is_split_lj_coulomb(styles: &[&Style]) -> bool {
    if styles.len() != 2 {
        return false;
    }
    let names: HashSet<&str> = styles.iter().map(|s| s.name.as_str()).collect();
    names == HashSet::from(["lj/cut", "coul/cut"])
        || names == HashSet::from(["lj/cut", "coul/long"])
}

fn write_combined_lj_coulomb(
    lines: &mut Vec<String>,
    styles: &[&Style],
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let lj = styles
        .iter()
        .find(|s| s.name == "lj/cut")
        .ok_or_else(|| "split pair styles missing lj/cut".to_owned())?;
    let coul = styles
        .iter()
        .find(|s| s.name == "coul/cut" || s.name == "coul/long")
        .ok_or_else(|| "split pair styles missing coul/*".to_owned())?;

    let lj_cut = units.length(style_cutoff(lj).unwrap_or(DEFAULT_PAIR_CUTOFF))?;
    let coul_cut = units.length(style_cutoff(coul).unwrap_or(DEFAULT_PAIR_CUTOFF))?;

    if !opts.skip_pair_style {
        lines.push(format!(
            "pair_style lj/cut/coul/cut {} {}\n",
            fmt_num(lj_cut, opts.precision),
            fmt_num(coul_cut, opts.precision)
        ));
        lines.push("\n".to_owned());
    }
    // Only LJ carries per-type ε/σ; Coulomb charges live on the atoms.
    write_pair_coeffs(lines, lj, None, opts, units)?;
    Ok(())
}

fn pair_style_cutoffs(style: &Style, units: &WriteUnits) -> Result<Vec<f64>, String> {
    // Combined names want two cutoffs; simple kernels one. Fall back to default
    // so the line is a legal LAMMPS command.
    let convert = |c: f64| units.length(c);
    match style.name.as_str() {
        "lj/cut/coul/cut" | "lj/cut/coul/long" => {
            let c = convert(style_cutoff(style).unwrap_or(DEFAULT_PAIR_CUTOFF))?;
            Ok(vec![c, c])
        }
        "lj/cut" | "lj126" | "coul/cut" | "coul/long" => Ok(vec![convert(
            style_cutoff(style).unwrap_or(DEFAULT_PAIR_CUTOFF),
        )?]),
        _ => match style_cutoff(style) {
            Some(c) => Ok(vec![convert(c)?]),
            None => Ok(vec![]),
        },
    }
}

fn style_cutoff(style: &Style) -> Option<f64> {
    style.params.get("cutoff")
}

fn write_pair_coeffs(
    lines: &mut Vec<String>,
    style: &Style,
    hybrid_substyle: Option<&str>,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let mut seen = HashSet::new();
    write_pair_coeffs_dedup(lines, style, hybrid_substyle, opts, units, &mut seen)?;
    if !seen.is_empty() {
        lines.push("\n".to_owned());
    }
    Ok(())
}

fn write_pair_coeffs_dedup(
    lines: &mut Vec<String>,
    style: &Style,
    hybrid_substyle: Option<&str>,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
    seen: &mut HashSet<String>,
) -> Result<(), String> {
    let StyleDefs::Pair(types) = &style.defs else {
        return Ok(());
    };
    for t in types {
        if !pair_included(t, opts) {
            continue;
        }
        // Only emit self-pairs with ε/σ (cross terms are combining-rule products).
        // Coulomb-only styles have no ε/σ — skip empty coeffs.
        let Some(eps_store) = t.params.get("epsilon") else {
            continue;
        };
        let Some(sigma_store) = t.params.get("sigma") else {
            continue;
        };
        let id = format!("{} {}", t.itom, t.jtom);
        if !seen.insert(id.clone()) {
            continue;
        }
        let eps = units.energy(eps_store)?;
        let sigma = units.length(sigma_store)?;
        let nums = format_nums(&[eps, sigma], opts.precision);
        match hybrid_substyle {
            Some(sub) => lines.push(format!("pair_coeff {id} {sub} {nums}\n")),
            None => lines.push(format!("pair_coeff {id} {nums}\n")),
        }
    }
    Ok(())
}

fn pair_included(t: &PairType, opts: &LammpsWriteOptions) -> bool {
    match &opts.atom_types {
        None => true,
        Some(allow) => allow.contains(&t.itom) && allow.contains(&t.jtom),
    }
}

// ── bonded ───────────────────────────────────────────────────────────────────

fn write_bond_section(
    lines: &mut Vec<String>,
    ff: &ForceField,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let styles = ff.get_styles("bond");
    if styles.is_empty() {
        return Ok(());
    }
    // AMBER/GAFF include is single-style harmonic.
    for style in styles {
        if style.name != "harmonic" {
            return Err(format!(
                "unsupported bond_style `{}` for LAMMPS *.ff writer (expected `harmonic`)",
                style.name
            ));
        }
        lines.push("bond_style harmonic\n".to_owned());
        let StyleDefs::Bond(types) = &style.defs else {
            continue;
        };
        let mut seen = HashSet::new();
        let mut any = false;
        for t in types {
            if !name_included(&t.name, &opts.bond_types) {
                continue;
            }
            if !seen.insert(t.name.clone()) {
                continue;
            }
            let (k, r0) = bond_lammps_params(t, units)?;
            lines.push(format!(
                "bond_coeff {} {} {}\n",
                t.name,
                fmt_num(k, opts.precision),
                fmt_num(r0, opts.precision)
            ));
            any = true;
        }
        if any {
            lines.push("\n".to_owned());
        }
    }
    Ok(())
}

fn bond_lammps_params(t: &BondType, units: &WriteUnits) -> Result<(f64, f64), String> {
    let k = t
        .params
        .get("k")
        .ok_or_else(|| format!("bond type `{}` missing param `k`", t.name))?;
    let r0 = t
        .params
        .get("r0")
        .ok_or_else(|| format!("bond type `{}` missing param `r0`", t.name))?;
    Ok((units.bond_k(k)?, units.length(r0)?))
}

fn write_angle_section(
    lines: &mut Vec<String>,
    ff: &ForceField,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let styles = ff.get_styles("angle");
    if styles.is_empty() {
        return Ok(());
    }
    for style in styles {
        if style.name != "harmonic" {
            return Err(format!(
                "unsupported angle_style `{}` for LAMMPS *.ff writer (expected `harmonic`)",
                style.name
            ));
        }
        lines.push("angle_style harmonic\n".to_owned());
        let StyleDefs::Angle(types) = &style.defs else {
            continue;
        };
        let mut seen = HashSet::new();
        let mut any = false;
        for t in types {
            if !name_included(&t.name, &opts.angle_types) {
                continue;
            }
            if !seen.insert(t.name.clone()) {
                continue;
            }
            let (k, theta0_deg) = angle_lammps_params(t, units)?;
            lines.push(format!(
                "angle_coeff {} {} {}\n",
                t.name,
                fmt_num(k, opts.precision),
                fmt_num(theta0_deg, opts.precision)
            ));
            any = true;
        }
        if any {
            lines.push("\n".to_owned());
        }
    }
    Ok(())
}

fn angle_lammps_params(t: &AngleType, units: &WriteUnits) -> Result<(f64, f64), String> {
    let k = t
        .params
        .get("k")
        .ok_or_else(|| format!("angle type `{}` missing param `k`", t.name))?;
    let theta0 = t
        .params
        .get("theta0")
        .ok_or_else(|| format!("angle type `{}` missing param `theta0`", t.name))?;
    // Equilibrium angle is always degrees in the LAMMPS file (all unit styles).
    Ok((units.angle_k(k)?, theta0.to_degrees()))
}

fn write_dihedral_section(
    lines: &mut Vec<String>,
    ff: &ForceField,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let styles = ff.get_styles("dihedral");
    if styles.is_empty() {
        return Ok(());
    }
    for style in styles {
        match style.name.as_str() {
            "fourier" => write_dihedral_fourier(lines, style, opts, units)?,
            "opls" => write_dihedral_opls(lines, style, opts, units)?,
            "harmonic" => write_dihedral_harmonic(lines, style, opts, units)?,
            other => {
                return Err(format!(
                    "unsupported dihedral_style `{other}` for LAMMPS *.ff writer \
                     (expected `fourier`, `opls`, or `harmonic`)"
                ));
            }
        }
    }
    Ok(())
}

fn write_dihedral_fourier(
    lines: &mut Vec<String>,
    style: &Style,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    lines.push("dihedral_style fourier\n".to_owned());
    let StyleDefs::Dihedral(types) = &style.defs else {
        return Ok(());
    };
    let mut seen = HashSet::new();
    let mut any = false;
    for t in types {
        if !name_included(&t.name, &opts.dihedral_types) {
            continue;
        }
        if !seen.insert(t.name.clone()) {
            continue;
        }
        let terms = fourier_terms(&t.params, &t.name, units)?;
        let m = terms.len();
        let mut parts = vec![format!("dihedral_coeff {}", t.name), format!("{m}")];
        for (k, n, d_deg) in terms {
            parts.push(fmt_num(k, opts.precision));
            // LAMMPS EXTRA-MOLECULE dihedral_fourier requires integer n.
            parts.push(format!("{}", n.round() as i64));
            parts.push(fmt_num(d_deg, opts.precision));
        }
        lines.push(parts.join(" ") + "\n");
        any = true;
    }
    if any {
        lines.push("\n".to_owned());
    }
    Ok(())
}

/// Collect fourier terms `(K_file, n, phase_deg)` from molrs keys `k{i}/n{i}/d{i}`.
fn fourier_terms(
    params: &Params,
    name: &str,
    units: &WriteUnits,
) -> Result<Vec<(f64, f64, f64)>, String> {
    let mut terms = Vec::new();
    let mut i = 1usize;
    while let Some(k_store) = params.get(&format!("k{i}")) {
        let n = params
            .get(&format!("n{i}"))
            .ok_or_else(|| format!("dihedral type `{name}` has k{i} but missing n{i}"))?;
        let d_rad = params.get(&format!("d{i}")).unwrap_or(0.0);
        terms.push((units.energy(k_store)?, n, d_rad.to_degrees()));
        i += 1;
    }
    if terms.is_empty() {
        return Err(format!(
            "dihedral type `{name}` has no fourier terms (expected k1/n1/d1…)"
        ));
    }
    Ok(terms)
}

fn write_dihedral_opls(
    lines: &mut Vec<String>,
    style: &Style,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    lines.push("dihedral_style opls\n".to_owned());
    let StyleDefs::Dihedral(types) = &style.defs else {
        return Ok(());
    };
    let mut seen = HashSet::new();
    let mut any = false;
    for t in types {
        if !name_included(&t.name, &opts.dihedral_types) {
            continue;
        }
        if !seen.insert(t.name.clone()) {
            continue;
        }
        // molrs OPLS kernel keys are f1–f4 (energy); accept legacy c1–c4 aliases.
        let f1 = units.energy(
            t.params
                .get("f1")
                .or_else(|| t.params.get("c1"))
                .unwrap_or(0.0),
        )?;
        let f2 = units.energy(
            t.params
                .get("f2")
                .or_else(|| t.params.get("c2"))
                .unwrap_or(0.0),
        )?;
        let f3 = units.energy(
            t.params
                .get("f3")
                .or_else(|| t.params.get("c3"))
                .unwrap_or(0.0),
        )?;
        let f4 = units.energy(
            t.params
                .get("f4")
                .or_else(|| t.params.get("c4"))
                .unwrap_or(0.0),
        )?;
        lines.push(format!(
            "dihedral_coeff {} {} {} {} {}\n",
            t.name,
            fmt_num(f1, opts.precision),
            fmt_num(f2, opts.precision),
            fmt_num(f3, opts.precision),
            fmt_num(f4, opts.precision)
        ));
        any = true;
    }
    if any {
        lines.push("\n".to_owned());
    }
    Ok(())
}

fn write_dihedral_harmonic(
    lines: &mut Vec<String>,
    style: &Style,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    // dihedral_style harmonic: K d n  (d phase degrees, n multiplicity)
    lines.push("dihedral_style harmonic\n".to_owned());
    let StyleDefs::Dihedral(types) = &style.defs else {
        return Ok(());
    };
    let mut seen = HashSet::new();
    let mut any = false;
    for t in types {
        if !name_included(&t.name, &opts.dihedral_types) {
            continue;
        }
        if !seen.insert(t.name.clone()) {
            continue;
        }
        let k_store = t
            .params
            .get("k")
            .ok_or_else(|| format!("dihedral type `{}` missing param `k`", t.name))?;
        let d_rad = t.params.get("d").unwrap_or(0.0);
        let n = t.params.get("n").unwrap_or(1.0);
        let k = units.energy(k_store)?;
        lines.push(format!(
            "dihedral_coeff {} {} {} {}\n",
            t.name,
            fmt_num(k, opts.precision),
            fmt_num(d_rad.to_degrees(), opts.precision),
            // multiplicity is an integer in LAMMPS
            n.round() as i64
        ));
        any = true;
    }
    if any {
        lines.push("\n".to_owned());
    }
    Ok(())
}

fn write_improper_section(
    lines: &mut Vec<String>,
    ff: &ForceField,
    opts: &LammpsWriteOptions,
    units: &WriteUnits,
) -> Result<(), String> {
    let styles = ff.get_styles("improper");
    if styles.is_empty() {
        return Ok(());
    }
    for style in styles {
        if style.name != "harmonic" {
            return Err(format!(
                "unsupported improper_style `{}` for LAMMPS *.ff writer (expected `harmonic`)",
                style.name
            ));
        }
        lines.push("improper_style harmonic\n".to_owned());
        let StyleDefs::Improper(types) = &style.defs else {
            continue;
        };
        let mut seen = HashSet::new();
        let mut any = false;
        for t in types {
            if !name_included(&t.name, &opts.improper_types) {
                continue;
            }
            if !seen.insert(t.name.clone()) {
                continue;
            }
            let (k, chi0_deg) = improper_lammps_params(t, units)?;
            lines.push(format!(
                "improper_coeff {} {} {}\n",
                t.name,
                fmt_num(k, opts.precision),
                fmt_num(chi0_deg, opts.precision)
            ));
            any = true;
        }
        if any {
            lines.push("\n".to_owned());
        }
    }
    Ok(())
}

fn improper_lammps_params(t: &ImproperType, units: &WriteUnits) -> Result<(f64, f64), String> {
    let k = t
        .params
        .get("k")
        .ok_or_else(|| format!("improper type `{}` missing param `k`", t.name))?;
    let chi0 = t
        .params
        .get("chi0")
        .ok_or_else(|| format!("improper type `{}` missing param `chi0`", t.name))?;
    Ok((units.angle_k(k)?, chi0.to_degrees()))
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn name_included(name: &str, allow: &Option<HashSet<String>>) -> bool {
    match allow {
        None => true,
        Some(set) => set.contains(name),
    }
}

fn fmt_num(v: f64, precision: usize) -> String {
    format!("{v:.precision$}")
}

fn format_nums(vals: &[f64], precision: usize) -> String {
    vals.iter()
        .map(|v| fmt_num(*v, precision))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::readers::{ForceFieldReader, lammps::LammpsFfReader};

    /// Same GAFF2-shaped mini include the reader tests pin.
    const MINI: &str = r#"
# LAMMPS force field generated by molrs
pair_style lj/cut/coul/long 10.0 10.0
pair_coeff c3 c3 0.107800 3.397710
pair_coeff oh oh 0.093000 3.242871
pair_coeff c3 c3 0.107800 3.397710

bond_style harmonic
bond_coeff c3-c3 228.890000 1.535400

angle_style harmonic
angle_coeff c3-c3-oh 76.790000 109.660000

dihedral_style fourier
dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.000000
"#;

    #[test]
    fn writes_lammps_units_inverse_of_reader() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let text = LammpsFfWriter::new().write_str(&ff).unwrap();

        // Combined pair style (not hybrid) so mixing stays intact.
        assert!(
            text.contains("pair_style lj/cut/coul/cut 10.000000 10.000000"),
            "combined pair_style:\n{text}"
        );
        assert!(!text.contains("hybrid"), "must not emit hybrid:\n{text}");
        assert!(
            text.contains("pair_coeff c3 c3 0.107800 3.397710"),
            "pair eps/sigma:\n{text}"
        );

        // K = k/2: reader stored k=457.78 → write K=228.89
        assert!(
            text.contains("bond_coeff c3-c3 228.890000 1.535400"),
            "bond K=k/2:\n{text}"
        );
        // angle: K=76.79, theta0 in degrees
        assert!(
            text.contains("angle_coeff c3-c3-oh 76.790000 109.660000"),
            "angle K + deg:\n{text}"
        );
        // fourier: m K n phase_deg. `n` is the cos(n*phi) multiplicity and
        // LAMMPS reads it with `inumeric()`, so it is written as an integer —
        // `3.000000` would be rejected by LAMMPS at parse time. It must also
        // stay in the `n` slot rather than sliding into the phase.
        assert!(
            text.contains("dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.000000"),
            "dihedral fourier:\n{text}"
        );
    }

    #[test]
    fn round_trip_preserves_molrs_params() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let text = LammpsFfWriter::new().write_str(&ff).unwrap();
        let ff2 = LammpsFfReader::new().read_str(&text).unwrap();

        let bt = ff2
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("c3", "c3")
            .unwrap();
        assert!((bt.params.get("k").unwrap() - 457.78).abs() < 1e-6);
        assert!((bt.params.get("r0").unwrap() - 1.5354).abs() < 1e-9);

        let angle = ff2.get_style("angle", "harmonic").unwrap();
        let StyleDefs::Angle(atypes) = &angle.defs else {
            panic!("not angle");
        };
        let at = &atypes[0];
        assert!((at.params.get("k").unwrap() - 153.58).abs() < 1e-6);
        assert!((at.params.get("theta0").unwrap() - 109.66_f64.to_radians()).abs() < 1e-9);

        let dih = ff2.get_style("dihedral", "fourier").unwrap();
        let StyleDefs::Dihedral(dtypes) = &dih.defs else {
            panic!("not dihedral");
        };
        let dt = &dtypes[0];
        assert!((dt.params.get("k1").unwrap() - 0.06).abs() < 1e-12);
        assert!((dt.params.get("n1").unwrap() - 3.0).abs() < 1e-12);
        assert!((dt.params.get("d1").unwrap() - 0.0).abs() < 1e-12);

        let lj = ff2.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("c3", None).unwrap();
        assert!((pt.params.get("epsilon").unwrap() - 0.1078).abs() < 1e-9);
        assert!((pt.params.get("sigma").unwrap() - 3.39771).abs() < 1e-9);
        assert!((lj.params.get("cutoff").unwrap_or(0.0) - 10.0).abs() < 1e-12);
        let coul = ff2.get_style("pair", "coul/cut").unwrap();
        assert!((coul.params.get("cutoff").unwrap_or(0.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn skip_pair_style_omits_header() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let opts = LammpsWriteOptions {
            skip_pair_style: true,
            ..Default::default()
        };
        let text = LammpsFfWriter::with_options(opts).write_str(&ff).unwrap();
        assert!(!text.contains("pair_style"), "no pair_style:\n{text}");
        assert!(text.contains("pair_coeff c3 c3"), "coeffs remain:\n{text}");
    }

    #[test]
    fn writes_units_real_line_by_default() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let text = LammpsFfWriter::new().write_str(&ff).unwrap();
        assert!(text.contains("units real\n"), "default units real:\n{text}");
    }

    #[test]
    fn write_data_coeffs_matches_command_form_numbers() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        // Labelled types need an id map (Frame-derived in production).
        let mut ids = HashMap::new();
        ids.insert("c3".into(), 1u32);
        ids.insert("oh".into(), 2u32);
        ids.insert("c3-c3".into(), 1u32);
        ids.insert("c3-c3-oh".into(), 1u32);
        ids.insert("c3-c3-oh-ho".into(), 1u32);
        let opts = LammpsWriteOptions {
            type_ids: Some(ids),
            ..Default::default()
        };
        let data = LammpsFfWriter::with_options(opts)
            .write_data_coeffs_str(&ff)
            .unwrap();
        assert!(
            !data.contains("pair_style") && !data.contains("units "),
            "sections only:\n{data}"
        );
        assert!(data.contains("Pair Coeffs\n"), "{data}");
        assert!(data.contains("Bond Coeffs\n"), "{data}");
        assert!(
            data.contains("1 0.107800 3.397710") || data.contains("1 0.107800"),
            "pair row:\n{data}"
        );
        assert!(
            data.contains("1 228.890000 1.535400"),
            "bond K=k/2:\n{data}"
        );

        let cmd = LammpsFfWriter::new().write_str(&ff).unwrap();
        // Same bond K appears in both layouts.
        assert!(cmd.contains("bond_coeff c3-c3 228.890000 1.535400"));
        assert!(data.contains("228.890000 1.535400"));
    }

    #[test]
    fn metal_write_converts_energy_via_lj_hub() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let opts = LammpsWriteOptions {
            units: LammpsUnits::Metal,
            ..Default::default()
        };
        let text = LammpsFfWriter::with_options(opts).write_str(&ff).unwrap();
        assert!(text.contains("units metal\n"), "metal header:\n{text}");

        // 0.1078 kcal/mol → eV through lj hub
        let sys = crate::ff::forcefield::lammps_units::LammpsUnitSystem::canonical().unwrap();
        let eps_ev = sys.from_store_energy(0.1078, LammpsUnits::Metal).unwrap();
        let expected = format!("pair_coeff c3 c3 {:.6}", eps_ev);
        assert!(text.contains(&expected), "expected {expected} in:\n{text}");

        // Length unchanged (Å in both real and metal).
        assert!(
            text.contains(&format!("{:.6}", 3.39771)),
            "sigma stays Å:\n{text}"
        );

        // Round-trip metal → store recovers original real values within the
        // printed precision (default 6 decimals on file numbers).
        let ff2 = LammpsFfReader::new().read_str(&text).unwrap();
        let pt = ff2
            .get_style("pair", "lj/cut")
            .unwrap()
            .get_pairtype("c3", None)
            .unwrap();
        assert!(
            (pt.params.get("epsilon").unwrap() - 0.1078).abs() < 5e-5,
            "eps store {}",
            pt.params.get("epsilon").unwrap()
        );
        let bt = ff2
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("c3", "c3")
            .unwrap();
        assert!(
            (bt.params.get("k").unwrap() - 457.78).abs() < 1e-3,
            "bond k store {}",
            bt.params.get("k").unwrap()
        );
    }

    #[test]
    fn atom_type_filter_restricts_pair_coeffs() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let opts = LammpsWriteOptions {
            atom_types: Some(HashSet::from(["c3".to_owned()])),
            ..Default::default()
        };
        let text = LammpsFfWriter::with_options(opts).write_str(&ff).unwrap();
        assert!(text.contains("pair_coeff c3 c3"));
        assert!(
            !text.contains("pair_coeff oh oh"),
            "oh filtered out:\n{text}"
        );
    }
}
