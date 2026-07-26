//! LAMMPS force-field reader (the `*.ff` include molpy writes alongside a data file).
//!
//! Parses a LAMMPS force-field include — `pair_style`/`pair_coeff`,
//! `bond_style harmonic`, `angle_style harmonic`, `dihedral_style fourier`
//! (+ optional `improper_style harmonic`) with **type-label** coefficients — into
//! a molrs [`ForceField`] in molrs units
//! (Å, kcal/mol, radians, e). This is the format the molpy
//! `LAMMPSForceFieldWriter` emits, e.g.:
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
//! # Units (LAMMPS `real` → molrs)
//!
//! molrs harmonic bond/angle kernels use the `½·k·(x−x₀)²` form, while LAMMPS
//! `harmonic` uses `K(x−x₀)²` with **no ½** — so the stored stiffness is `k = 2·K`
//! (the molrs ctor param key is `"k"`). Every angle-valued parameter — angle
//! `theta0`, dihedral phase `d`, improper `chi0` — is read in **degrees** and
//! **normalized to radians at this boundary** (`.to_radians()`), matching molrs's
//! internal-radians convention: the kernels consume radians directly and do no
//! unit conversion of their own. The `fourier` dihedral maps to molrs's `periodic` kernel
//! (`E = Σ Kₘ[1+cos(nₘφ−dₘ)]`). ε, σ, r0 and the dihedral K/n are already in
//! molrs units.
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
use crate::ff::forcefield::{ForceField, SpecialBonds};
use molrs::units::constants::COULOMB_REAL;

/// AMBER/GAFF 1-4 Lennard-Jones scale (`special_bonds amber`).
const AMBER_LJ14: f64 = 0.5;
/// AMBER/GAFF 1-4 Coulomb scale (`special_bonds amber`, = 1/1.2).
const AMBER_COUL14: f64 = 1.0 / 1.2;

/// Reader for a LAMMPS force-field include (`*.ff`), AMBER/GAFF flavour.
#[derive(Debug, Default, Clone, Copy)]
pub struct LammpsFfReader;

impl LammpsFfReader {
    pub fn new() -> Self {
        Self
    }
}

impl ForceFieldReader for LammpsFfReader {
    fn read_str(&self, text: &str) -> Result<ForceField, String> {
        let mut ff = ForceField::new("LAMMPS");
        // AMBER/GAFF 1-4 scaling (special_bonds amber): LJ ×0.5, Coulomb ×1/1.2;
        // 1-2 and 1-3 are fully excluded by the neighbour list. Owned by the
        // ForceField, applied to flagged 1-4 pairs by the pair kernels.
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, AMBER_LJ14],
            coul: [0.0, 0.0, AMBER_COUL14],
        });
        // Pair self-params are collected first, then emitted once as a lj/cut +
        // coul/cut pair (mirroring the OPLS reader): the coul charges come from
        // the frame, so only the LJ self-terms are transcribed here.
        let mut pair_rows: Vec<(String, f64, f64)> = Vec::new();
        // Cutoffs declared on the `pair_style` line, kept for `build_pairs`.
        let mut cutoffs: (Option<f64>, Option<f64>) = (None, None);

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
                // Style declarations: validate the kernel is one we translate,
                // then create the (empty) style its coeff lines append to.
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
                    require_kernel("dihedral_style", &rest, "fourier", &where_)?;
                    ff.def_dihedralstyle("fourier");
                }
                "improper_style" => {
                    require_kernel("improper_style", &rest, "harmonic", &where_)?;
                    ff.def_improperstyle("harmonic");
                }
                // Coefficient lines append a typed entry to their style.
                "pair_coeff" => collect_pair(&rest, &mut pair_rows, &where_)?,
                "bond_coeff" => add_bond(&mut ff, &rest, &where_)?,
                "angle_coeff" => add_angle(&mut ff, &rest, &where_)?,
                "dihedral_coeff" => add_dihedral(&mut ff, &rest, &where_)?,
                "improper_coeff" => add_improper(&mut ff, &rest, &where_)?,
                // LAMMPS settings that may ride along in an include but carry no
                // force-field parameters for the relaxer.
                "pair_modify" | "special_bonds" | "units" | "atom_style" | "kspace_style" => {}
                other => return Err(format!("{}: unknown LAMMPS keyword `{other}`", where_())),
            }
        }

        build_pairs(&mut ff, &pair_rows, cutoffs);
        Ok(ff)
    }
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
) -> Result<(), String> {
    // pair_coeff <i> <j> [sub-style] <epsilon> <sigma>. Only self-pairs i==j
    // are transcribed; cross terms come from the combining rule in
    // `to_potentials`.
    if rest.len() < 2 {
        return Err(format!("{}: pair_coeff needs `<i> <j> ...`", where_()));
    }
    let (ti, tj) = (rest[0], rest[1]);
    // A hybrid line names its sub-style before the numbers: `c3 c3 lj/cut …`.
    // Only lj/cut carries eps/sigma; the `* * coul/cut` wildcard (charges come
    // from the frame) has nothing to transcribe.
    let mut args = &rest[2..];
    if let Some(&first) = args.first()
        && first.parse::<f64>().is_err()
    {
        if first != "lj/cut" {
            return Ok(());
        }
        args = &args[1..];
    }
    if ti != tj {
        // Explicit cross terms are not part of the GAFF include; skip rather than
        // invent a type, leaving combining to `to_potentials`.
        return Ok(());
    }
    if args.len() < 2 {
        return Err(format!(
            "{}: pair_coeff needs `<i> <j> [style] eps sigma`",
            where_()
        ));
    }
    let eps = parse_f64(args[0], "pair epsilon", where_)?;
    let sigma = parse_f64(args[1], "pair sigma", where_)?;
    if !rows.iter().any(|(t, _, _)| t == ti) {
        rows.push((ti.to_owned(), eps, sigma));
    }
    Ok(())
}

/// Emit the collected LJ self-params as a `lj/cut` style plus a `coul/cut` style
/// (charges resolved from the frame), with AMBER 1-4 scales.
///
/// `coul/cut` is the **buffered** Coulomb `E = k·qᵢqⱼ/(D·(r + δ))`; a LAMMPS `real`
/// -units force field is the unbuffered case (δ = 0, the semantic default) in vacuum,
/// with CODATA's `k`. It used to be defined with EMPTY params and merely happened to
/// agree with the constant the kernel held privately — the right number for the wrong
/// reason. The force field states it now: the kernel has no default, because MMFF's
/// `k` is a different number and both are correct.
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

fn add_bond(ff: &mut ForceField, rest: &[&str], where_: &dyn Fn() -> String) -> Result<(), String> {
    // bond_coeff <a>-<b> K r0      (LAMMPS K(r−r0)² → molrs ½k0(r−r0)², k0 = 2K)
    let [a, b] = split_types::<2>(rest.first(), "bond", where_)?;
    let k = parse_f64(get(rest, 1, "bond K", where_)?, "bond K", where_)?;
    let r0 = parse_f64(get(rest, 2, "bond r0", where_)?, "bond r0", where_)?;
    style_mut(ff, "bond", "harmonic", "bond_style harmonic", where_)?.def_bondtype(
        &a,
        &b,
        &[("k", 2.0 * k), ("r0", r0)],
    );
    Ok(())
}

fn add_angle(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<(), String> {
    // angle_coeff <a>-<b>-<c> K theta0(deg)   (k = 2K; theta0 deg→rad at read)
    let [a, b, c] = split_types::<3>(rest.first(), "angle", where_)?;
    let k = parse_f64(get(rest, 1, "angle K", where_)?, "angle K", where_)?;
    let theta0_deg = parse_f64(
        get(rest, 2, "angle theta0", where_)?,
        "angle theta0",
        where_,
    )?;
    style_mut(ff, "angle", "harmonic", "angle_style harmonic", where_)?.def_angletype(
        &a,
        &b,
        &c,
        &[("k", 2.0 * k), ("theta0", theta0_deg.to_radians())],
    );
    Ok(())
}

fn add_dihedral(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<(), String> {
    // dihedral_coeff <a>-<b>-<c>-<d> m  K1 n1 d1  [K2 n2 d2 ...]
    // → periodic kernel keys k{m}/n{m}/d{m} (phase d deg→rad at read).
    let [a, b, c, d] = split_types::<4>(rest.first(), "dihedral", where_)?;
    let m: usize = get(rest, 1, "dihedral m", where_)?
        .parse()
        .map_err(|_| format!("{}: dihedral m is not an integer", where_()))?;
    let mut owned: Vec<(String, f64)> = Vec::with_capacity(3 * m);
    for term in 0..m {
        let base = 2 + 3 * term; // first triple starts at index 2
        let k = parse_f64(get(rest, base, "dihedral K", where_)?, "dihedral K", where_)?;
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
        owned.push((format!("k{}", term + 1), k));
        owned.push((format!("n{}", term + 1), n));
        owned.push((format!("d{}", term + 1), phase.to_radians())); // deg→rad at read
    }
    let params: Vec<(&str, f64)> = owned.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    style_mut(ff, "dihedral", "fourier", "dihedral_style fourier", where_)?
        .def_dihedraltype(&a, &b, &c, &d, &params);
    Ok(())
}

fn add_improper(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<(), String> {
    // improper_coeff <a>-<b>-<c>-<d> K chi0(deg)   (k = 2K; chi0 deg→rad at read)
    let [a, b, c, d] = split_types::<4>(rest.first(), "improper", where_)?;
    let k = parse_f64(get(rest, 1, "improper K", where_)?, "improper K", where_)?;
    let chi0_deg = parse_f64(
        get(rest, 2, "improper chi0", where_)?,
        "improper chi0",
        where_,
    )?;
    style_mut(
        ff,
        "improper",
        "harmonic",
        "improper_style harmonic",
        where_,
    )?
    .def_impropertype(
        &a,
        &b,
        &c,
        &d,
        &[("k", 2.0 * k), ("chi0", chi0_deg.to_radians())],
    );
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

/// Split a hyphen-joined type label (`c3-c3`, `hc-c3-oh`) into exactly `N` tokens.
fn split_types<const N: usize>(
    label: Option<&&str>,
    kind: &str,
    where_: &dyn Fn() -> String,
) -> Result<[String; N], String> {
    let label = label.ok_or_else(|| format!("{}: {kind}_coeff missing type label", where_()))?;
    let parts: Vec<&str> = label.split('-').collect();
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
# LAMMPS force field generated by molpy
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

    /// The `hybrid/overlay` pair form the molpy writer emits round-trips: both
    /// cutoffs come back and the wildcard `* * coul/cut` line is skipped rather
    /// than mistaken for LJ coefficients.
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

    /// A hybrid line whose sub-styles carry no cutoff (the older writer form
    /// `hybrid lj/cut coul/cut`) must not read the following sub-style name as a
    /// cutoff number -- both simply fall back with no recorded cutoff.
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
}
