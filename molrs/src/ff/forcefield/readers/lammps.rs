//! LAMMPS force-field reader (the `*.ff` include next to a data file).
//!
//! Parses a LAMMPS force-field include — `pair_style`/`pair_coeff`,
//! `bond_style harmonic`, `angle_style harmonic`, `dihedral_style fourier`
//! (+ optional `improper_style harmonic`) with **type-label** coefficients — into
//! a molrs [`ForceField`] in molrs units (Å, kcal/mol, radians, e). Inverse of
//! [`LammpsFfWriter`](crate::ff::forcefield::writers::lammps::LammpsFfWriter), e.g.:
//!
//! ```text
//! pair_style lj/cut/coul/long 10.0 10.0
//! pair_coeff c3 c3 0.107800 3.397710          # epsilon(kcal/mol) sigma(Å)
//! bond_style harmonic
//! bond_coeff c3-c3 228.890000 1.535400        # K(kcal/mol/Å²) r0(Å)
//! angle_style harmonic
//! angle_coeff c3-c3-oh 76.790000 109.660000   # K(kcal/mol/rad²) theta0(deg)
//! dihedral_style fourier
//! dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.0 # m  K1 n1 d1(deg) [K2 n2 d2 ...]
//! ```
//!
//! # Units (LAMMPS style → molrs store)
//!
//! File-side `units real|metal|lj` is respected (default **real** for bare
//! molecular includes). Conversions use [`crate::ff::forcefield::lammps_units`]
//! — always **source → lj reduced → store** via `UnitRegistry`/`Quantity`, never
//! ad-hoc factors. Store for physical styles is **real** (Å, kcal/mol); `lj`
//! files stay reduced.
//!
//! **Form map** (independent of unit style): molrs harmonic bond/angle kernels
//! use `½·k·(x−x₀)²`, LAMMPS uses `K(x−x₀)²` → stored `k = 2·K`. Angle/phase
//! values in real/metal files are **degrees** and become **radians** at this
//! boundary. The `fourier` dihedral maps to molrs's `periodic` kernel.
//!
//! # Charges and masses
//!
//! Per-atom charge and mass live in the LAMMPS **data** file, not this include,
//! so they are not read here: the `coul/cut` style draws charges from the
//! [`Frame`](molrs::store::frame::Frame) at evaluation time (as for OPLS), and
//! masses are irrelevant to geometry relaxation.
//!
//! 1-4 scaling follows the AMBER/GAFF convention this format targets
//! (`special_bonds amber`): LJ ×0.5, Coulomb ×0.8333.

use super::ForceFieldReader;
use crate::ff::constants::VACUUM_DIELECTRIC;
use crate::ff::forcefield::lammps_units::{
    LammpsUnitSystem, LammpsUnits, lammps_k_to_molrs_half_k,
};
use crate::ff::forcefield::{ForceField, SpecialBonds};
use molrs::units::constants::COULOMB_REAL;
use std::collections::BTreeMap;

/// AMBER/GAFF 1-4 Lennard-Jones scale (`special_bonds amber`).
const AMBER_LJ14: f64 = 0.5;
/// AMBER/GAFF 1-4 Coulomb scale (`special_bonds amber`, = 1/1.2).
const AMBER_COUL14: f64 = 1.0 / 1.2;

/// Optional id→label maps (from a data-file Type Labels section).
#[derive(Debug, Clone, Default)]
pub struct LammpsTypeLabelMaps {
    pub atom: BTreeMap<u32, String>,
    pub bond: BTreeMap<u32, String>,
    pub angle: BTreeMap<u32, String>,
    pub dihedral: BTreeMap<u32, String>,
    pub improper: BTreeMap<u32, String>,
}

/// Reader for a LAMMPS force-field include (`*.ff`), AMBER/GAFF flavour.
#[derive(Debug, Clone)]
pub struct LammpsFfReader {
    /// Used when the file has no `units` line. Molecular includes default to
    /// **real** (LAMMPS bare-script default is `lj` — pass `default_units: Lj`
    /// or write an explicit `units` line when that matters).
    pub default_units: LammpsUnits,
}

impl Default for LammpsFfReader {
    fn default() -> Self {
        Self {
            default_units: LammpsUnits::Real,
        }
    }
}

impl LammpsFfReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default_units(default_units: LammpsUnits) -> Self {
        Self { default_units }
    }

    /// Parse data-file `* Coeffs` sections with optional Type Labels maps.
    ///
    /// `coeffs_text` is a fragment containing `Pair Coeffs` / `Bond Coeffs` / …
    /// (and optional `units` line). Style defaults to harmonic / lj when the
    /// data file does not declare `*_style`.
    pub fn read_data_coeffs(
        &self,
        coeffs_text: &str,
        labels: &LammpsTypeLabelMaps,
        units: LammpsUnits,
    ) -> Result<ForceField, String> {
        // Synthesize style lines so the shared dispatcher can run, then parse
        // section-form coeff lines rewritten as command-form.
        let mut synthetic = String::new();
        synthetic.push_str(&format!("units {}\n", units.as_str()));
        // Default styles for data-file coeffs (no style line in the data file).
        synthetic.push_str("pair_style lj/cut 10.0\n");
        synthetic.push_str("bond_style harmonic\n");
        synthetic.push_str("angle_style harmonic\n");
        synthetic.push_str("dihedral_style harmonic\n");
        synthetic.push_str("improper_style harmonic\n");
        synthetic.push_str(&data_sections_to_commands(coeffs_text, labels)?);
        self.read_str_with_labels(&synthetic, labels)
    }

    fn read_str_with_labels(
        &self,
        text: &str,
        labels: &LammpsTypeLabelMaps,
    ) -> Result<ForceField, String> {
        let unit_sys =
            LammpsUnitSystem::canonical().map_err(|e| format!("lammps unit system: {e}"))?;
        let mut file_units = self.default_units;
        let mut ff = ForceField::new("LAMMPS");
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, AMBER_LJ14],
            coul: [0.0, 0.0, AMBER_COUL14],
        });
        let mut pair_rows: Vec<(String, f64, f64)> = Vec::new();
        let mut cutoffs: (Option<f64>, Option<f64>) = (None, None);
        let mut dihedral_style_name: Option<String> = None;

        for (lineno, raw) in text.lines().enumerate() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            let mut tok = line.split_whitespace();
            let kw = tok.next().unwrap();
            let rest: Vec<&str> = tok.collect();
            let where_ = || format!("line {}", lineno + 1);

            match kw {
                "units" => {
                    let name = rest
                        .first()
                        .ok_or_else(|| format!("{}: units missing style name", where_()))?;
                    file_units =
                        LammpsUnits::parse(name).map_err(|e| format!("{}: {e}", where_()))?;
                }
                "pair_style" => cutoffs = require_pair_style(&rest, &where_)?,
                "bond_style" => {
                    require_kernel("bond_style", &rest, "harmonic", &where_)?;
                    ff.def_bondstyle("harmonic");
                }
                "angle_style" => {
                    require_kernel("angle_style", &rest, "harmonic", &where_)?;
                    ff.def_anglestyle("harmonic");
                }
                "dihedral_style" => {
                    let name = rest
                        .first()
                        .ok_or_else(|| format!("{}: dihedral_style missing name", where_()))?;
                    let allowed = ["fourier", "opls", "harmonic", "multi/harmonic", "charmm"];
                    if !allowed.contains(name) {
                        return Err(format!(
                            "{}: unsupported dihedral_style `{name}` (expected one of {})",
                            where_(),
                            allowed.join(", ")
                        ));
                    }
                    // charmm multi-term shares the fourier/periodic param layout.
                    let style_name = if *name == "charmm" || *name == "multi/harmonic" {
                        "fourier"
                    } else {
                        *name
                    };
                    dihedral_style_name = Some(style_name.to_owned());
                    ff.def_dihedralstyle(style_name);
                }
                "improper_style" => {
                    require_kernel("improper_style", &rest, "harmonic", &where_)?;
                    ff.def_improperstyle("harmonic");
                }
                "pair_coeff" => collect_pair(
                    &rest,
                    &mut pair_rows,
                    &where_,
                    &unit_sys,
                    file_units,
                    labels,
                )?,
                "bond_coeff" => add_bond(&mut ff, &rest, &where_, &unit_sys, file_units, labels)?,
                "angle_coeff" => add_angle(&mut ff, &rest, &where_, &unit_sys, file_units, labels)?,
                "dihedral_coeff" => {
                    let dname = dihedral_style_name.as_deref().ok_or_else(|| {
                        format!("{}: coeff before its `dihedral_style`", where_())
                    })?;
                    add_dihedral(
                        &mut ff, &rest, &where_, &unit_sys, file_units, labels, dname,
                    )?
                }
                "improper_coeff" => {
                    add_improper(&mut ff, &rest, &where_, &unit_sys, file_units, labels)?
                }
                "pair_modify" | "special_bonds" | "atom_style" | "kspace_style" => {}
                other => return Err(format!("{}: unknown LAMMPS keyword `{other}`", where_())),
            }
        }

        // Cutoffs are lengths in the file unit system.
        let cutoffs = (
            cutoffs
                .0
                .map(|c| unit_sys.to_store_length(c, file_units))
                .transpose()?,
            cutoffs
                .1
                .map(|c| unit_sys.to_store_length(c, file_units))
                .transpose()?,
        );
        build_pairs(&mut ff, &pair_rows, cutoffs);
        let _ = file_units; // store units stamped on Python side; name stays LAMMPS
        Ok(ff)
    }
}

impl ForceFieldReader for LammpsFfReader {
    fn read_str(&self, text: &str) -> Result<ForceField, String> {
        self.read_str_with_labels(text, &LammpsTypeLabelMaps::default())
    }
}

/// Rewrite data-file section blocks into `*_coeff` command lines.
fn data_sections_to_commands(text: &str, labels: &LammpsTypeLabelMaps) -> Result<String, String> {
    let mut out = String::new();
    let mut section: Option<&str> = None;
    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("pair coeffs") {
            section = Some("pair");
            continue;
        }
        if lower.starts_with("bond coeffs") {
            section = Some("bond");
            continue;
        }
        if lower.starts_with("angle coeffs") {
            section = Some("angle");
            continue;
        }
        if lower.starts_with("dihedral coeffs") {
            section = Some("dihedral");
            continue;
        }
        if lower.starts_with("improper coeffs") {
            section = Some("improper");
            continue;
        }
        // New uppercase section ends coeffs.
        if line.chars().next().is_some_and(|c| c.is_uppercase())
            && !line.chars().next().unwrap().is_ascii_digit()
        {
            section = None;
            continue;
        }
        let Some(kind) = section else {
            continue;
        };
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let id: u32 = parts[0].parse().map_err(|_| {
            format!(
                "line {}: expected integer type id in {kind} coeffs, got {}",
                lineno + 1,
                parts[0]
            )
        })?;
        let type_tok = match kind {
            "pair" => labels
                .atom
                .get(&id)
                .cloned()
                .unwrap_or_else(|| id.to_string()),
            "bond" => labels
                .bond
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("{id}-{id}")),
            "angle" => labels
                .angle
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("{id}-{id}-{id}")),
            "dihedral" => labels
                .dihedral
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("{id}-{id}-{id}-{id}")),
            "improper" => labels
                .improper
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("{id}-{id}-{id}-{id}")),
            _ => unreachable!(),
        };
        match kind {
            "pair" => {
                // Pair Coeffs: id ε σ  →  pair_coeff T T ε σ
                if parts.len() < 3 {
                    return Err(format!(
                        "line {}: Pair Coeffs needs `id epsilon sigma`",
                        lineno + 1
                    ));
                }
                out.push_str(&format!(
                    "pair_coeff {type_tok} {type_tok} {} {}\n",
                    parts[1], parts[2]
                ));
            }
            other => {
                // bond/angle/dihedral/improper: id params… → *_coeff TYPE params…
                out.push_str(&format!("{other}_coeff {type_tok}"));
                for p in &parts[1..] {
                    out.push(' ');
                    out.push_str(p);
                }
                out.push('\n');
            }
        }
    }
    Ok(out)
}

// ── pair ──────────────────────────────────────────────────────────────────────

/// Validate the pair kernel and return its `(lj, coulomb)` cutoffs in Å.
///
/// Three spellings all map to the reader's lj/cut + coul/cut pair:
///
/// - the combined kernel — `pair_style lj/cut/coul/cut 10.0 [12.0]`, Coulomb
///   cutoff defaulting to the LJ one when omitted;
/// - `hybrid lj/cut 10.0 coul/cut 10.0`, one pair per sub-style;
/// - `hybrid/overlay lj/cut 10.0 coul/cut 10.0`, both on every pair — what this
///   reader's own force field means, so its writer emits it.
///
/// The cutoffs are part of the force field, not a rendering detail: a reader
/// that keeps only the kernel name cannot write a runnable input back out.
fn require_pair_style(
    rest: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<(Option<f64>, Option<f64>), String> {
    let name = rest
        .first()
        .ok_or_else(|| format!("{}: pair_style missing kernel name", where_()))?;
    if *name == "hybrid" || *name == "hybrid/overlay" {
        return hybrid_cutoffs(&rest[1..], where_);
    }
    // Any LJ-12-6 + Coulomb variant maps to lj/cut + coul/cut for the relaxer.
    if !name.starts_with("lj/cut") {
        return Err(format!(
            "{}: unsupported pair_style `{name}` (expected an `lj/cut...`, \
             `hybrid`, or `hybrid/overlay` variant)",
            where_()
        ));
    }
    let mut cutoffs = rest[1..]
        .iter()
        .map(|t| parse_f64(t, "pair_style cutoff", where_));
    let lj = cutoffs.next().transpose()?;
    let coul = cutoffs.next().transpose()?.or(lj);
    Ok((lj, coul))
}

/// Cutoffs from a `hybrid` / `hybrid/overlay` pair line, e.g.
/// `lj/cut 10.0 coul/cut 10.0`: read each `lj/cut` and `coul/cut` sub-style's
/// first numeric argument. Other sub-styles are rejected, matching the combined
/// form's `lj/cut` requirement.
fn hybrid_cutoffs(
    rest: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<(Option<f64>, Option<f64>), String> {
    let (mut lj, mut coul) = (None, None);
    let mut i = 0;
    while i < rest.len() {
        let sub = rest[i];
        // A sub-style's cutoff is optional: the next token is a cutoff only if
        // it parses as a number, otherwise it is the next sub-style name (and
        // this sub-style falls back to LAMMPS's global default).
        let cut = rest.get(i + 1).and_then(|t| t.parse::<f64>().ok());
        match sub {
            "lj/cut" => lj = cut,
            "coul/cut" | "coul/long" => coul = cut,
            other => {
                return Err(format!(
                    "{}: unsupported hybrid pair sub-style `{other}` \
                     (expected `lj/cut` or `coul/cut`)",
                    where_()
                ));
            }
        }
        // Step over the sub-style name and its cutoff argument, if any.
        i += if cut.is_some() { 2 } else { 1 };
    }
    Ok((lj, coul.or(lj)))
}

fn collect_pair(
    rest: &[&str],
    rows: &mut Vec<(String, f64, f64)>,
    where_: &dyn Fn() -> String,
    unit_sys: &LammpsUnitSystem,
    file_units: LammpsUnits,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    // pair_coeff <i> <j> [sub-style] <epsilon> <sigma>. Only self-pairs i==j
    // are transcribed; cross terms come from the combining rule in
    // `to_potentials`.
    if rest.len() < 2 {
        return Err(format!("{}: pair_coeff needs `<i> <j> ...`", where_()));
    }
    let ti = resolve_atom_type(rest[0], labels, where_)?;
    let tj = resolve_atom_type(rest[1], labels, where_)?;
    // A hybrid line names its sub-style before the numbers: `c3 c3 lj/cut …`.
    // Only lj/cut carries eps/sigma; the `* * coul/cut` wildcard (charges come
    // from the frame) has nothing to transcribe.
    let mut args = &rest[2..];
    // Optional hybrid sub-style token (must be a known name, not any non-float —
    // otherwise `notanumber` would be silently skipped as an unknown sub-style).
    if let Some(&first) = args.first()
        && first.parse::<f64>().is_err()
    {
        match first {
            "lj/cut" => args = &args[1..],
            "coul/cut" | "coul/long" => return Ok(()), // charges from frame
            other => {
                return Err(format!(
                    "{}: pair_coeff unexpected token `{other}` (expected \
                     [lj/cut] eps sigma)",
                    where_()
                ));
            }
        }
    }
    if ti != tj {
        return Ok(());
    }
    if args.len() < 2 {
        return Err(format!(
            "{}: pair_coeff needs `<i> <j> [style] eps sigma`",
            where_()
        ));
    }
    let eps = unit_sys.to_store_energy(parse_f64(args[0], "pair epsilon", where_)?, file_units)?;
    let sigma = unit_sys.to_store_length(parse_f64(args[1], "pair sigma", where_)?, file_units)?;
    if !rows.iter().any(|(t, _, _)| t == &ti) {
        rows.push((ti, eps, sigma));
    }
    Ok(())
}

/// Emit the collected LJ self-params as a `lj/cut` style plus a `coul/cut` style
/// (charges resolved from the frame), with AMBER 1-4 scales.
///
/// `coul/cut` is the **buffered** Coulomb `E = k·qᵢqⱼ/(D·(r + δ))`; a LAMMPS `real`
/// -units force field is the unbuffered case (δ = 0) in vacuum with CODATA's `k`.
/// The force field states the constant explicitly: the kernel has no default, because
/// MMFF's `k` is a different number and both are correct.
fn build_pairs(
    ff: &mut ForceField,
    rows: &[(String, f64, f64)],
    cutoffs: (Option<f64>, Option<f64>),
) {
    if rows.is_empty() {
        return;
    }
    // 1-4 scaling lives on the ForceField's `special_bonds` (set in `read_str`),
    // not on the pair styles — `to_potentials` projects it into the kernels.
    let (cut_lj, cut_coul) = cutoffs;
    let lj_params: Vec<(&str, f64)> = cut_lj.map(|c| vec![("cutoff", c)]).unwrap_or_default();
    let lj = ff.def_pairstyle("lj/cut", &lj_params);
    for (ty, eps, sigma) in rows {
        lj.def_pairtype(ty, None, &[("epsilon", *eps), ("sigma", *sigma)]);
    }
    let mut coul_params = vec![("coulomb", COULOMB_REAL), ("dielectric", VACUUM_DIELECTRIC)];
    if let Some(c) = cut_coul {
        coul_params.push(("cutoff", c));
    }
    ff.def_pairstyle("coul/cut", &coul_params);
}

// ── bonded ──────────────────────────────────────────────────────────────────

fn add_bond(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
    unit_sys: &LammpsUnitSystem,
    file_units: LammpsUnits,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    // bond_coeff <type> K r0  — type is label `a-b` or numeric id
    let [a, b] = split_types::<2>(rest.first(), "bond", where_, Some(&labels.bond))?;
    let k_file = parse_f64(get(rest, 1, "bond K", where_)?, "bond K", where_)?;
    let r0_file = parse_f64(get(rest, 2, "bond r0", where_)?, "bond r0", where_)?;
    let k_lammps = unit_sys.to_store_bond_k_lammps(k_file, file_units)?;
    let r0 = unit_sys.to_store_length(r0_file, file_units)?;
    let k = lammps_k_to_molrs_half_k(k_lammps);
    style_mut(ff, "bond", "harmonic", "bond_style harmonic", where_)?.def_bondtype(
        &a,
        &b,
        &[("k", k), ("r0", r0)],
    );
    Ok(())
}

fn add_angle(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
    unit_sys: &LammpsUnitSystem,
    file_units: LammpsUnits,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    let [a, b, c] = split_types::<3>(rest.first(), "angle", where_, Some(&labels.angle))?;
    let k_file = parse_f64(get(rest, 1, "angle K", where_)?, "angle K", where_)?;
    let theta0_deg = parse_f64(
        get(rest, 2, "angle theta0", where_)?,
        "angle theta0",
        where_,
    )?;
    let k_lammps = unit_sys.to_store_angle_k_lammps(k_file, file_units)?;
    let k = lammps_k_to_molrs_half_k(k_lammps);
    style_mut(ff, "angle", "harmonic", "angle_style harmonic", where_)?.def_angletype(
        &a,
        &b,
        &c,
        &[("k", k), ("theta0", theta0_deg.to_radians())],
    );
    Ok(())
}

fn add_dihedral(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
    unit_sys: &LammpsUnitSystem,
    file_units: LammpsUnits,
    labels: &LammpsTypeLabelMaps,
    style_name: &str,
) -> Result<(), String> {
    let [a, b, c, d] = split_types::<4>(rest.first(), "dihedral", where_, Some(&labels.dihedral))?;
    match style_name {
        "opls" => {
            // dihedral_coeff a-b-c-d K1 K2 K3 K4
            let mut ks = [0.0_f64; 4];
            for (i, slot) in ks.iter_mut().enumerate() {
                let raw = parse_f64(
                    get(rest, 1 + i, "dihedral K", where_)?,
                    "dihedral K",
                    where_,
                )?;
                *slot = unit_sys.to_store_energy(raw, file_units)?;
            }
            style_mut(ff, "dihedral", "opls", "dihedral_style opls", where_)?.def_dihedraltype(
                &a,
                &b,
                &c,
                &d,
                &[("f1", ks[0]), ("f2", ks[1]), ("f3", ks[2]), ("f4", ks[3])],
            );
        }
        "harmonic" => {
            // dihedral_coeff a-b-c-d K d n
            let k_raw = parse_f64(get(rest, 1, "dihedral K", where_)?, "dihedral K", where_)?;
            let phase = parse_f64(get(rest, 2, "dihedral d", where_)?, "dihedral d", where_)?;
            let n = parse_f64(get(rest, 3, "dihedral n", where_)?, "dihedral n", where_)?;
            let k = unit_sys.to_store_energy(k_raw, file_units)?;
            style_mut(
                ff,
                "dihedral",
                "harmonic",
                "dihedral_style harmonic",
                where_,
            )?
            .def_dihedraltype(
                &a,
                &b,
                &c,
                &d,
                &[("k", k), ("d", phase.to_radians()), ("n", n)],
            );
        }
        _ => {
            // fourier / multi/harmonic / charmm layout:
            // dihedral_coeff a-b-c-d m  K1 n1 d1  [K2 n2 d2 ...]
            let m: usize = get(rest, 1, "dihedral m", where_)?
                .parse()
                .map_err(|_| format!("{}: dihedral m is not an integer", where_()))?;
            let mut owned: Vec<(String, f64)> = Vec::with_capacity(3 * m);
            for term in 0..m {
                let base = 2 + 3 * term;
                let k_raw =
                    parse_f64(get(rest, base, "dihedral K", where_)?, "dihedral K", where_)?;
                let n = parse_f64(
                    get(rest, base + 1, "dihedral n", where_)?,
                    "dihedral n",
                    where_,
                )?;
                let phase = parse_f64(
                    get(rest, base + 2, "dihedral d", where_)?,
                    "dihedral d",
                    where_,
                )?;
                let k = unit_sys.to_store_energy(k_raw, file_units)?;
                owned.push((format!("k{}", term + 1), k));
                owned.push((format!("n{}", term + 1), n));
                owned.push((format!("d{}", term + 1), phase.to_radians()));
            }
            let params: Vec<(&str, f64)> = owned.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            style_mut(ff, "dihedral", "fourier", "dihedral_style fourier", where_)?
                .def_dihedraltype(&a, &b, &c, &d, &params);
        }
    }
    Ok(())
}

fn add_improper(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
    unit_sys: &LammpsUnitSystem,
    file_units: LammpsUnits,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    let [a, b, c, d] = split_types::<4>(rest.first(), "improper", where_, Some(&labels.improper))?;
    let k_file = parse_f64(get(rest, 1, "improper K", where_)?, "improper K", where_)?;
    let chi0_deg = parse_f64(
        get(rest, 2, "improper chi0", where_)?,
        "improper chi0",
        where_,
    )?;
    let k_lammps = unit_sys.to_store_angle_k_lammps(k_file, file_units)?;
    let k = lammps_k_to_molrs_half_k(k_lammps);
    style_mut(
        ff,
        "improper",
        "harmonic",
        "improper_style harmonic",
        where_,
    )?
    .def_impropertype(&a, &b, &c, &d, &[("k", k), ("chi0", chi0_deg.to_radians())]);
    Ok(())
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Fetch the style created by the matching `*_style` directive; a coeff line
/// before its style is an error, not a silently-dropped parameter.
fn style_mut<'a>(
    ff: &'a mut ForceField,
    category: &str,
    name: &str,
    directive: &str,
    where_: &dyn Fn() -> String,
) -> Result<&'a mut crate::ff::forcefield::Style, String> {
    ff.get_style_mut(category, name)
        .ok_or_else(|| format!("{}: coeff before its `{directive}` declaration", where_()))
}

/// Drop a trailing `#` comment from a LAMMPS line.
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

fn require_kernel(
    directive: &str,
    rest: &[&str],
    expect: &str,
    where_: &dyn Fn() -> String,
) -> Result<(), String> {
    match rest.first() {
        Some(&name) if name == expect => Ok(()),
        Some(&name) => Err(format!(
            "{}: unsupported {directive} `{name}` (expected `{expect}`)",
            where_()
        )),
        None => Err(format!("{}: {directive} missing kernel name", where_())),
    }
}

/// Resolve one atom-type token: numeric id via label map, or bare label / `*`.
fn resolve_atom_type(
    raw: &str,
    labels: &LammpsTypeLabelMaps,
    where_: &dyn Fn() -> String,
) -> Result<String, String> {
    if raw == "*" {
        return Ok("*".into());
    }
    if let Ok(id) = raw.parse::<u32>() {
        return Ok(labels
            .atom
            .get(&id)
            .cloned()
            .unwrap_or_else(|| id.to_string()));
    }
    let _ = where_;
    Ok(raw.to_owned())
}

/// Split a type key into `N` endpoint names.
///
/// Accepts:
/// - hyphen form `a-b` / `a-b-c` (or `::` when labels contain `-`);
/// - a single numeric id, expanded via `label_map` or synthetic `id-id-…`;
/// - a full label from the map when `raw` is a numeric id whose map value
///   already encodes endpoints.
fn split_types<const N: usize>(
    label: Option<&&str>,
    kind: &str,
    where_: impl Fn() -> String,
    label_map: Option<&BTreeMap<u32, String>>,
) -> Result<[String; N], String> {
    let raw = label.ok_or_else(|| format!("{}: {kind}_coeff missing type label", where_()))?;

    // Numeric type id → map or synthetic.
    if let Ok(id) = raw.parse::<u32>() {
        let expanded = label_map
            .and_then(|m| m.get(&id).cloned())
            .unwrap_or_else(|| {
                std::iter::repeat_n(id.to_string(), N)
                    .collect::<Vec<_>>()
                    .join("-")
            });
        return split_types_str::<N>(&expanded, kind, &where_);
    }
    split_types_str::<N>(raw, kind, &where_)
}

fn split_types_str<const N: usize>(
    label: &str,
    kind: &str,
    where_: &dyn Fn() -> String,
) -> Result<[String; N], String> {
    // Prefer :: when present (labels with embedded '-').
    let parts: Vec<&str> = if label.contains("::") {
        label.split("::").collect()
    } else {
        label.split('-').collect()
    };
    if parts.len() != N {
        return Err(format!(
            "{}: {kind} type `{label}` has {} atoms, expected {N}",
            where_(),
            parts.len()
        ));
    }
    Ok(std::array::from_fn(|i| parts[i].to_owned()))
}

fn get<'a>(
    rest: &'a [&'a str],
    idx: usize,
    what: &str,
    where_: &dyn Fn() -> String,
) -> Result<&'a str, String> {
    rest.get(idx)
        .copied()
        .ok_or_else(|| format!("{}: missing {what}", where_()))
}

fn parse_f64(raw: &str, what: &str, where_: &dyn Fn() -> String) -> Result<f64, String> {
    raw.parse::<f64>()
        .map_err(|_| format!("{}: {what} is not a number: {raw:?}", where_()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::{AngleType, DihedralType, Style, StyleDefs};

    /// A LAMMPS include covering every style, with values copied from a real
    /// GAFF2 PEO `.ff` (figure5).
    const MINI: &str = r#"
# LAMMPS force field generated by molrs
pair_style lj/cut/coul/long 10.0 10.0
pair_coeff c3 c3 0.107800 3.397710
pair_coeff oh oh 0.093000 3.242871
pair_coeff c3 c3 0.107800 3.397710   # duplicate, ignored

bond_style harmonic
bond_coeff c3-c3 228.890000 1.535400

angle_style harmonic
angle_coeff c3-c3-oh 76.790000 109.660000

dihedral_style fourier
dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.000000
"#;

    fn angle_types(s: &Style) -> &[AngleType] {
        match &s.defs {
            StyleDefs::Angle(v) => v,
            _ => unreachable!(),
        }
    }
    fn dihedral_types(s: &Style) -> &[DihedralType] {
        match &s.defs {
            StyleDefs::Dihedral(v) => v,
            _ => unreachable!(),
        }
    }

    #[test]
    fn reads_lammps_units() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();

        // bond: K 228.89 → k = 2K = 457.78 (param key "k") ; r0 unchanged.
        let bond = ff.get_style("bond", "harmonic").unwrap();
        let bt = bond.get_bondtype("c3", "c3").unwrap();
        assert!((bt.params.get("k").unwrap() - 457.78).abs() < 1e-6, "k");
        assert!((bt.params.get("r0").unwrap() - 1.5354).abs() < 1e-9, "r0");

        // angle: K 76.79 → k = 153.58 ; theta0 normalized to radians at read.
        let angle = ff.get_style("angle", "harmonic").unwrap();
        let at = &angle_types(angle)[0];
        assert!((at.params.get("k").unwrap() - 153.58).abs() < 1e-6, "ak");
        assert!(
            (at.params.get("theta0").unwrap() - 109.66_f64.to_radians()).abs() < 1e-12,
            "theta0"
        );

        // dihedral fourier → periodic keys k1/n1/d1 (phase d normalized to radians;
        // 0° → 0 rad).
        let dih = ff.get_style("dihedral", "fourier").unwrap();
        let dt = &dihedral_types(dih)[0];
        assert!((dt.params.get("k1").unwrap() - 0.06).abs() < 1e-12, "k1");
        assert!((dt.params.get("n1").unwrap() - 3.0).abs() < 1e-12, "n1");
        assert!((dt.params.get("d1").unwrap() - 0.0).abs() < 1e-12, "d1");

        // pair: ε/σ pass through; the duplicate c3 row is ignored.
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("c3", None).unwrap();
        assert!(
            (pt.params.get("epsilon").unwrap() - 0.1078).abs() < 1e-9,
            "eps"
        );
        assert!(
            (pt.params.get("sigma").unwrap() - 3.39771).abs() < 1e-9,
            "sig"
        );
        assert!(ff.get_style("pair", "coul/cut").is_some(), "coul style");

        // The cutoffs on the `pair_style` line belong to the force field: without
        // them a written-back include is not a runnable LAMMPS input.
        assert!(
            (lj.params.get("cutoff").unwrap_or(0.0) - 10.0).abs() < 1e-12,
            "lj cutoff"
        );
        let coul = ff.get_style("pair", "coul/cut").unwrap();
        assert!(
            (coul.params.get("cutoff").unwrap_or(0.0) - 10.0).abs() < 1e-12,
            "coulomb cutoff"
        );

        // AMBER/GAFF 1-4 scaling is recorded on the ForceField's special_bonds
        // (1-2/1-3 excluded), the source the pair kernels consume.
        let sb = ff.special_bonds();
        assert_eq!(sb.lj, [0.0, 0.0, 0.5]);
        assert!((sb.coul_14() - 1.0 / 1.2).abs() < 1e-12);
        assert_eq!(sb.coul[0], 0.0);
        assert_eq!(sb.coul[1], 0.0);
    }

    #[test]
    fn unknown_keyword_errors() {
        let err = LammpsFfReader::new()
            .read_str("mystery_style foo\n")
            .unwrap_err();
        assert!(err.contains("unknown LAMMPS keyword"), "err: {err}");
    }

    #[test]
    fn coeff_before_style_errors() {
        let err = LammpsFfReader::new()
            .read_str("bond_coeff c3-c3 1.0 1.5\n")
            .unwrap_err();
        assert!(err.contains("before its"), "err: {err}");
    }

    #[test]
    fn wrong_arity_type_label_errors() {
        let err = LammpsFfReader::new()
            .read_str("bond_style harmonic\nbond_coeff c3-c3-oh 1.0 1.5\n")
            .unwrap_err();
        assert!(err.contains("expected 2"), "err: {err}");
    }

    /// The `hybrid/overlay` pair form round-trips: both cutoffs come back and the
    /// wildcard `* * coul/cut` line is skipped rather than mistaken for LJ coefficients.
    #[test]
    fn reads_hybrid_overlay_pair_style() {
        let text = "\
pair_style hybrid/overlay lj/cut 10.0 coul/cut 12.0
pair_coeff * * coul/cut
pair_coeff c3 c3 lj/cut 0.1078 3.39771
";
        let ff = LammpsFfReader::new().read_str(text).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        assert!((lj.params.get("cutoff").unwrap_or(0.0) - 10.0).abs() < 1e-12);
        let pt = lj.get_pairtype("c3", None).unwrap();
        assert!((pt.params.get("epsilon").unwrap() - 0.1078).abs() < 1e-9);
        assert!((pt.params.get("sigma").unwrap() - 3.39771).abs() < 1e-9);
        let coul = ff.get_style("pair", "coul/cut").unwrap();
        assert!((coul.params.get("cutoff").unwrap_or(0.0) - 12.0).abs() < 1e-12);
    }

    /// A hybrid line whose sub-styles carry no cutoff (`hybrid lj/cut coul/cut`)
    /// must not read the following sub-style name as a cutoff number — both fall
    /// back with no recorded cutoff.
    #[test]
    fn reads_hybrid_pair_style_without_cutoffs() {
        let text = "pair_style hybrid lj/cut coul/cut
pair_coeff c3 c3 lj/cut 0.1078 3.39771
";
        let ff = LammpsFfReader::new().read_str(text).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        assert!(lj.params.get("cutoff").is_none(), "no lj cutoff recorded");
        assert!(
            (lj.get_pairtype("c3", None)
                .unwrap()
                .params
                .get("sigma")
                .unwrap()
                - 3.39771)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn metal_units_convert_energy_via_lj_hub() {
        // 1 eV in metal → ~23.06 kcal/mol in store (real).
        let text = "\
units metal
pair_style lj/cut 10.0
pair_coeff c3 c3 1.0 3.4
bond_style harmonic
bond_coeff c3-c3 1.0 1.5
";
        let ff = LammpsFfReader::new().read_str(text).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("c3", None).unwrap();
        let eps = pt.params.get("epsilon").unwrap();
        // 1 eV → kcal/mol through units component
        let sys = crate::ff::forcefield::lammps_units::LammpsUnitSystem::canonical().unwrap();
        let expect = sys
            .energy(
                1.0,
                crate::ff::forcefield::lammps_units::LammpsUnits::Metal,
                crate::ff::forcefield::lammps_units::LammpsUnits::Real,
            )
            .unwrap();
        assert!((eps - expect).abs() < 1e-9, "eps {eps} vs {expect}");
        // bond K=1 eV/Å² → store k = 2 * K_real
        let bond = ff.get_style("bond", "harmonic").unwrap();
        let bt = bond.get_bondtype("c3", "c3").unwrap();
        let k_lammps_real = sys
            .bond_k_lammps(
                1.0,
                crate::ff::forcefield::lammps_units::LammpsUnits::Metal,
                crate::ff::forcefield::lammps_units::LammpsUnits::Real,
            )
            .unwrap();
        let expect_k = crate::ff::forcefield::lammps_units::lammps_k_to_molrs_half_k(k_lammps_real);
        assert!((bt.params.get("k").unwrap() - expect_k).abs() < 1e-9);
        assert!((bt.params.get("r0").unwrap() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn numeric_bond_id_with_type_labels_map() {
        let mut labels = LammpsTypeLabelMaps::default();
        labels.bond.insert(1, "CT-HC".into());
        labels.atom.insert(1, "CT".into());
        labels.atom.insert(2, "HC".into());
        let text = "\
units real
bond_style harmonic
bond_coeff 1 100.0 1.09
pair_style lj/cut 10.0
pair_coeff 1 1 0.1 3.5
";
        let ff = LammpsFfReader::new()
            .read_str_with_labels(text, &labels)
            .unwrap();
        let bond = ff.get_style("bond", "harmonic").unwrap();
        assert!(bond.get_bondtype("CT", "HC").is_some());
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        assert!(lj.get_pairtype("CT", None).is_some());
    }

    #[test]
    fn data_coeffs_section_with_labels() {
        let mut labels = LammpsTypeLabelMaps::default();
        labels.bond.insert(1, "OW-HW".into());
        labels.atom.insert(1, "OW".into());
        let coeffs = "\
Bond Coeffs

1 450.0 0.9572

Pair Coeffs

1 0.1521 3.1507
";
        let ff = LammpsFfReader::new()
            .read_data_coeffs(coeffs, &labels, LammpsUnits::Real)
            .unwrap();
        let bond = ff.get_style("bond", "harmonic").unwrap();
        let bt = bond.get_bondtype("OW", "HW").unwrap();
        assert!((bt.params.get("k").unwrap() - 900.0).abs() < 1e-9); // 2*450
        assert!((bt.params.get("r0").unwrap() - 0.9572).abs() < 1e-9);
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("OW", None).unwrap();
        assert!((pt.params.get("epsilon").unwrap() - 0.1521).abs() < 1e-9);
    }

    #[test]
    fn dihedral_opls_four_coeffs() {
        let text = "\
dihedral_style opls
dihedral_coeff CT-CT-CT-CT 1.0 2.0 3.0 4.0
";
        let ff = LammpsFfReader::new().read_str(text).unwrap();
        let d = ff.get_style("dihedral", "opls").unwrap();
        let dt = &dihedral_types(d)[0];
        assert!((dt.params.get("f1").unwrap() - 1.0).abs() < 1e-12);
        assert!((dt.params.get("f4").unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn numeric_pair_types_stable_with_bonded_coeffs() {
        // Regression: bonded types whose endpoints look like "10" must not
        // scramble pair self-types named "10".
        let text = "\
units real
pair_style lj/cut 10.0
pair_coeff 1 1 0.11 3.5
pair_coeff 2 2 0.08 3.6
pair_coeff 10 10 0.046 0.4
bond_style harmonic
bond_coeff 1-1 100.0 1.5
bond_coeff 10-10 200.0 1.2
dihedral_style harmonic
dihedral_coeff 10-10-10-10 0.2 1.0 180.0
";
        let ff = LammpsFfReader::new().read_str(text).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let p1 = lj.get_pairtype("1", None).unwrap();
        let p2 = lj.get_pairtype("2", None).unwrap();
        let p10 = lj.get_pairtype("10", None).unwrap();
        assert!((p1.params.get("epsilon").unwrap() - 0.11).abs() < 1e-12);
        assert!((p2.params.get("epsilon").unwrap() - 0.08).abs() < 1e-12);
        assert!((p10.params.get("epsilon").unwrap() - 0.046).abs() < 1e-12);
        assert!((p10.params.get("sigma").unwrap() - 0.4).abs() < 1e-12);
    }
}
