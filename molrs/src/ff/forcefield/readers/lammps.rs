//! LAMMPS force-field reader (the `*.ff` include next to a data file).
//!
//! Parses a LAMMPS force-field include — `pair_style`/`pair_coeff`,
//! `bond_style harmonic`, `angle_style harmonic`, `dihedral_style`
//! `fourier` / `opls` / `harmonic` / `charmm` / `multi/harmonic` (+ optional
//! `improper_style harmonic`) with **type-label** coefficients — into
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
//! use `½·k·(x−x₀)²`, LAMMPS uses `K(x−x₀)²` → stored `k = 2·K`. The harmonic
//! improper kernel is LAMMPS's own `K(χ−χ₀)²` → stored `k = K`. Angle/phase
//! values in real/metal files are **degrees** and become **radians** at this
//! boundary. The `fourier` dihedral maps to molrs's `periodic` kernel. The
//! token → params conversion for one coefficient line is
//! [`lammps_coeff_params`], the single place it happens.
//!
//! # Charges and masses
//!
//! Per-atom charge and mass live in the LAMMPS **data** file, not this include,
//! so they are not read here: the `coul/cut` style draws charges from the
//! [`Frame`](molrs::store::frame::Frame) at evaluation time (as for OPLS), and
//! masses are irrelevant to geometry relaxation.
//!
//! 1-4 weights are **declared** on a `special_bonds` line and stored on
//! [`ForceField::special_bonds`](crate::ff::forcefield::ForceField::special_bonds)
//! (dimensionless `[1-2, 1-3, 1-4]`). An include that omits the line is an
//! error — LAMMPS's own default (`0 0 0`) is not AMBER's weights, so this
//! reader will not invent either. Data-file `read_data_coeffs` synthesizes
//! an explicit AMBER-like line so those reads keep the 0.5 / 5/6 they have
//! always produced.

use super::ForceFieldReader;
use crate::ff::constants::VACUUM_DIELECTRIC;
use crate::ff::forcefield::lammps_units::{LammpsFfUnits, lammps_k_to_molrs_half_k, parse_style};
use crate::ff::forcefield::mixing::Mixing;
use crate::ff::forcefield::{ForceField, Params, SpecialBonds};
use crate::ff::params::amber::{AMBER_SCEE, AMBER_SCNB};
use molrs::store::type_labels::TypeName;
use molrs::units::constants::COULOMB_REAL;
use std::collections::BTreeMap;

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
    pub default_units: &'static str,
}

impl Default for LammpsFfReader {
    fn default() -> Self {
        Self {
            default_units: "real",
        }
    }
}

impl LammpsFfReader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse data-file `* Coeffs` sections with optional Type Labels maps.
    ///
    /// `coeffs_text` is a fragment containing `Pair Coeffs` / `Bond Coeffs` / …
    /// (and optional `units` line).
    ///
    /// # Styles
    ///
    /// A data file has no `*_style` lines; `write_data` records each style as
    /// the section header's comment, e.g. `Bond Coeffs # harmonic/kk`. That
    /// hint selects the category's style, with an accelerator suffix (`/kk`,
    /// `/gpu`, `/omp`, `/intel`, `/opt`) removed. A hinted style this reader
    /// has no kernel for (`fene`, `cosine`, …) is an error naming the section
    /// and the style — the numbers are never read under another kernel.
    ///
    /// A section **without** a hint, and a category with no section, fall back
    /// to `harmonic` (bond, angle, dihedral, improper) and `lj/cut` (pair).
    /// The pair cutoff is always 10.0 in the file's units: a data file does not
    /// carry one.
    ///
    /// # Errors
    ///
    /// An unsupported hinted style, plus every error of the `*_coeff` parse.
    pub fn read_data_coeffs(
        &self,
        coeffs_text: &str,
        labels: &LammpsTypeLabelMaps,
        units: &str,
    ) -> Result<ForceField, String> {
        // Synthesize style lines so the shared dispatcher can run, then parse
        // section-form coeff lines rewritten as command-form.
        let mut synthetic = String::new();
        synthetic.push_str(&format!("units {units}\n"));
        // Data files carry no special_bonds; keep the AMBER weights this path
        // has always assumed, as an explicit declaration. Built from the same
        // constants as the `amber` preset so the two cannot drift apart.
        synthetic.push_str(&format!(
            "special_bonds lj 0.0 0.0 {} coul 0.0 0.0 {}\n",
            1.0 / AMBER_SCNB,
            1.0 / AMBER_SCEE
        ));
        let (hints, commands) = data_sections_to_commands(coeffs_text, labels)?;
        for category in ["pair", "bond", "angle", "dihedral", "improper"] {
            let style = match hints.get(category) {
                Some(hint) => {
                    hint.require_supported(category)?;
                    hint.style.as_str()
                }
                None if category == "pair" => "lj/cut",
                None => "harmonic",
            };
            if category == "pair" {
                synthetic.push_str(&format!("pair_style {style} {DATA_PAIR_CUTOFF}\n"));
            } else {
                synthetic.push_str(&format!("{category}_style {style}\n"));
            }
        }
        synthetic.push_str(&commands);
        self.read_str_with_labels(&synthetic, labels)
    }

    fn read_str_with_labels(
        &self,
        text: &str,
        labels: &LammpsTypeLabelMaps,
    ) -> Result<ForceField, String> {
        let unit_sys =
            LammpsFfUnits::canonical().map_err(|e| format!("lammps unit system: {e}"))?;
        let mut file_units = self.default_units;
        let mut ff = ForceField::new("LAMMPS");
        let mut pair_rows: Vec<(String, Params)> = Vec::new();
        let mut cutoffs: (Option<f64>, Option<f64>) = (None, None);
        let mut pair_mix: Option<String> = None;
        let mut dihedral_style_name: Option<String> = None;
        let mut saw_special_bonds = false;

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
                    file_units = parse_style(name).map_err(|e| format!("{}: {e}", where_()))?;
                }
                "pair_style" => cutoffs = require_pair_style(&rest, &where_)?,
                "bond_style" => {
                    require_kernel("bond_style", &rest, "harmonic", &where_)?;
                    ff.def_style("bond", "harmonic", Params::new())
                        .map_err(|e| e.to_string())?;
                }
                "angle_style" => {
                    require_kernel("angle_style", &rest, "harmonic", &where_)?;
                    ff.def_style("angle", "harmonic", Params::new())
                        .map_err(|e| e.to_string())?;
                }
                "dihedral_style" => {
                    let name = rest
                        .first()
                        .ok_or_else(|| format!("{}: dihedral_style missing name", where_()))?;
                    require_dihedral_style(name, &where_)?;
                    // Each LAMMPS style has its own molrs kernel and coefficient
                    // layout (see `coeff_params`); the name is kept as written.
                    dihedral_style_name = Some((*name).to_owned());
                    ff.def_style("dihedral", name, Params::new())
                        .map_err(|e| e.to_string())?;
                }
                "improper_style" => {
                    require_kernel("improper_style", &rest, "harmonic", &where_)?;
                    ff.def_style("improper", "harmonic", Params::new())
                        .map_err(|e| e.to_string())?;
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
                "special_bonds" => {
                    ff.set_special_bonds(parse_special_bonds(&rest, &where_)?);
                    saw_special_bonds = true;
                }
                "pair_modify" => {
                    if let Some(i) = rest.iter().position(|t| *t == "mix") {
                        let rule = rest.get(i + 1).ok_or_else(|| {
                            format!("{}: pair_modify mix missing a rule", where_())
                        })?;
                        Mixing::parse(rule).map_err(|e| format!("{}: {e}", where_()))?;
                        pair_mix = Some((*rule).to_owned());
                    }
                }
                "atom_style" | "kspace_style" => {}
                other => return Err(format!("{}: unknown LAMMPS keyword `{other}`", where_())),
            }
        }
        if !saw_special_bonds {
            return Err(
                "special_bonds declaration is missing; 1-4 weights cannot be invented".into(),
            );
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
        build_pairs(&mut ff, &pair_rows, cutoffs, pair_mix.as_deref())?;
        // The parameters are now in store units: `lj` files stay reduced,
        // every physical style was converted to `real`.
        ff.set_units(if file_units == "lj" { "lj" } else { "real" });
        Ok(ff)
    }
}

impl ForceFieldReader for LammpsFfReader {
    fn read_str(&self, text: &str) -> Result<ForceField, String> {
        self.read_str_with_labels(text, &LammpsTypeLabelMaps::default())
    }
}

/// Pair cutoff `read_data_coeffs` declares: a data file carries none.
const DATA_PAIR_CUTOFF: f64 = 10.0;

/// A data-file `* Coeffs` section header's `# <style>` comment.
#[derive(Debug)]
struct SectionStyleHint {
    /// The header line as written, e.g. `Bond Coeffs # fene/kk`.
    header: String,
    /// The style with any accelerator suffix removed, e.g. `fene`.
    style: String,
}

impl SectionStyleHint {
    /// The hint on `header`, or `None` when it has no `# <style>` comment.
    fn parse(header: &str) -> Option<Self> {
        let (_, comment) = header.split_once('#')?;
        let raw = comment.split_whitespace().next()?;
        let style = ["/kk", "/gpu", "/omp", "/intel", "/opt"]
            .iter()
            .find_map(|suffix| raw.strip_suffix(suffix))
            .unwrap_or(raw);
        Some(Self {
            header: header.trim().to_owned(),
            style: style.to_owned(),
        })
    }

    /// Refuse a style the `category`'s `*_style` directive would refuse, with
    /// the error naming this section.
    fn require_supported(&self, category: &str) -> Result<(), String> {
        let where_ = || format!("`{}` section", self.header);
        let style = self.style.as_str();
        match category {
            "pair" => {
                require_pair_style(&[style, &DATA_PAIR_CUTOFF.to_string()], &where_).map(|_| ())
            }
            "dihedral" => require_dihedral_style(style, &where_),
            _ => require_kernel(&format!("{category}_style"), &[style], "harmonic", &where_),
        }
    }
}

/// Rewrite data-file section blocks into `*_coeff` command lines, and collect
/// each section header's `# <style>` hint by category.
fn data_sections_to_commands(
    text: &str,
    labels: &LammpsTypeLabelMaps,
) -> Result<(BTreeMap<&'static str, SectionStyleHint>, String), String> {
    let mut out = String::new();
    let mut hints = BTreeMap::new();
    let mut section: Option<&str> = None;
    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        let opened = [
            ("pair coeffs", "pair"),
            ("bond coeffs", "bond"),
            ("angle coeffs", "angle"),
            ("dihedral coeffs", "dihedral"),
            ("improper coeffs", "improper"),
        ]
        .into_iter()
        .find(|(name, _)| lower.starts_with(name));
        if let Some((_, kind)) = opened {
            section = Some(kind);
            match SectionStyleHint::parse(raw) {
                Some(hint) => hints.insert(kind, hint),
                None => hints.remove(kind),
            };
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
    Ok((hints, out))
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
    rows: &mut Vec<(String, Params)>,
    where_: &dyn Fn() -> String,
    unit_sys: &LammpsFfUnits,
    file_units: &str,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    // pair_coeff <i> <j> [sub-style] <epsilon> <sigma>. Only self-pairs i==j
    // are transcribed; cross terms come from the combining rule in
    // `PotentialCompiler::compile`.
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
    let params = coeff_params(unit_sys, file_units, "pair", "lj/cut", args, where_)?;
    if !rows.iter().any(|(t, _)| t == &ti) {
        rows.push((ti, params));
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
    rows: &[(String, Params)],
    cutoffs: (Option<f64>, Option<f64>),
    mix: Option<&str>,
) -> Result<(), String> {
    if rows.is_empty() {
        return Ok(());
    }
    // 1-4 scaling lives on the ForceField's `special_bonds` (set in `read_str`),
    // not on the pair styles — `PotentialCompiler::compile` projects it into the kernels.
    let (cut_lj, cut_coul) = cutoffs;
    let lj_pairs: Vec<(&str, f64)> = cut_lj.map(|c| vec![("cutoff", c)]).unwrap_or_default();
    let mut lj_params = Params::from_pairs(&lj_pairs);
    // LAMMPS mixes `lj/cut` **geometrically** unless `pair_modify mix` says
    // otherwise; record it explicitly rather than inherit the kernel's
    // Lorentz-Berthelot default, which would shift every \u03c3 silently.
    lj_params.set_str("mixing", mix.unwrap_or("geometric"));
    let lj = ff
        .def_style("pair", "lj/cut", lj_params)
        .map_err(|e| e.to_string())?;
    for (ty, params) in rows {
        lj.def_type(ty, &[ty], params.clone())
            .map_err(|e| e.to_string())?;
    }
    let mut coul_params = vec![("coulomb", COULOMB_REAL), ("dielectric", VACUUM_DIELECTRIC)];
    if let Some(c) = cut_coul {
        coul_params.push(("cutoff", c));
    }
    ff.def_style("pair", "coul/cut", Params::from_pairs(&coul_params))
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ── bonded ──────────────────────────────────────────────────────────────────

fn add_bond(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
    unit_sys: &LammpsFfUnits,
    file_units: &str,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    // bond_coeff <type> K r0  — type is label `a-b` or numeric id
    let (name, [a, b]) = label_type::<2>(rest.first(), "bond", where_, Some(&labels.bond))?;
    let params = coeff_params(unit_sys, file_units, "bond", "harmonic", &rest[1..], where_)?;
    style_mut(ff, "bond", "harmonic", "bond_style harmonic", where_)?
        .def_type(&name, &[&a, &b], params)
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn add_angle(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
    unit_sys: &LammpsFfUnits,
    file_units: &str,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    let (name, [a, b, c]) = label_type::<3>(rest.first(), "angle", where_, Some(&labels.angle))?;
    let params = coeff_params(
        unit_sys,
        file_units,
        "angle",
        "harmonic",
        &rest[1..],
        where_,
    )?;
    style_mut(ff, "angle", "harmonic", "angle_style harmonic", where_)?
        .def_type(&name, &[&a, &b, &c], params)
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn add_dihedral(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
    unit_sys: &LammpsFfUnits,
    file_units: &str,
    labels: &LammpsTypeLabelMaps,
    style_name: &str,
) -> Result<(), String> {
    let (name, [a, b, c, d]) =
        label_type::<4>(rest.first(), "dihedral", where_, Some(&labels.dihedral))?;
    let params = coeff_params(
        unit_sys,
        file_units,
        "dihedral",
        style_name,
        &rest[1..],
        where_,
    )?;
    let directive = format!("dihedral_style {style_name}");
    style_mut(ff, "dihedral", style_name, &directive, where_)?
        .def_type(&name, &[&a, &b, &c, &d], params)
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn add_improper(
    ff: &mut ForceField,
    rest: &[&str],
    where_: &dyn Fn() -> String,
    unit_sys: &LammpsFfUnits,
    file_units: &str,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    let (name, [a, b, c, d]) =
        label_type::<4>(rest.first(), "improper", where_, Some(&labels.improper))?;
    let params = coeff_params(
        unit_sys,
        file_units,
        "improper",
        "harmonic",
        &rest[1..],
        where_,
    )?;
    style_mut(
        ff,
        "improper",
        "harmonic",
        "improper_style harmonic",
        where_,
    )?
    .def_type(&name, &[&a, &b, &c, &d], params)
    .map_err(|e| e.to_string())?;
    Ok(())
}

// ── coefficient conversion ──────────────────────────────────────────────────

/// Convert one LAMMPS coefficient line into the molrs params the reader stores.
///
/// `values` are the coefficient tokens **after** the type field(s) — for
/// `bond_coeff c3-c3 228.89 1.5354` that is `["228.89", "1.5354"]`, for
/// `pair_coeff c3 c3 0.1078 3.3977` it is `["0.1078", "3.3977"]`. `units` is the
/// LAMMPS `units` keyword the numbers are written in (`real`, `metal`, `lj`).
///
/// The result is in molrs store units — Å, kcal/mol, radians for `real` and
/// `metal`; `lj` stays reduced — with the LAMMPS → molrs form map applied:
///
/// | category / style        | LAMMPS tokens        | stored params |
/// |-------------------------|----------------------|---------------|
/// | `bond harmonic`         | `K r0`               | `k = 2K`, `r0` |
/// | `angle harmonic`        | `K theta0(deg)`      | `k = 2K`, `theta0` (rad) |
/// | `improper harmonic`     | `K chi0(deg)`        | `k = K`, `chi0` (rad) |
/// | `dihedral opls`         | `K1 K2 K3 K4`        | `k1..k4` |
/// | `dihedral harmonic`     | `K d n`              | `k`, `sign = d` (±1), `periodicity = n` |
/// | `dihedral fourier`      | `m K1 n1 d1(deg) …`  | `k<i>`, `periodicity<i>`, `phase<i>` (rad) |
/// | `dihedral charmm`       | `K n d(deg) w`       | `k`, `periodicity`, `phase` (rad), `w` |
/// | `dihedral multi/harmonic` | `A1 A2 A3 A4 A5`   | `a1..a5` |
/// | `pair lj/cut…`          | `epsilon sigma`      | `epsilon`, `sigma` |
///
/// The bond and angle `k = 2K` factor exists because molrs's harmonic bond and
/// angle kernels are `½·k·(x−x₀)²` and LAMMPS's are `K·(x−x₀)²`; molrs's
/// improper kernel is LAMMPS's `K·(χ−χ₀)²`, so the improper `k` is `K`. Any `pair` style spelled `lj/cut…`
/// (`lj/cut/coul/long`, …) carries the same `epsilon sigma` pair.
///
/// # Errors
///
/// A `(category, style)` the LAMMPS reader has no kernel for, an unknown
/// `units` keyword, a missing coefficient, or a non-numeric token.
///
/// ```
/// use molrs::ff::forcefield::readers::lammps::lammps_coeff_params;
///
/// let p = lammps_coeff_params("bond", "harmonic", &["450", "0.9572"], "real").unwrap();
/// assert_eq!(p.get("k"), Some(900.0));
/// assert!(lammps_coeff_params("bond", "morse", &["1", "2", "3"], "real").is_err());
/// ```
pub fn lammps_coeff_params(
    category: &str,
    style: &str,
    values: &[&str],
    units: &str,
) -> Result<Params, String> {
    let file_units = parse_style(units)?;
    let unit_sys = LammpsFfUnits::canonical().map_err(|e| format!("lammps unit system: {e}"))?;
    coeff_params(&unit_sys, file_units, category, style, values, &|| {
        format!("{category} {style}")
    })
}

/// The one LAMMPS-coefficient → molrs-params conversion, shared by the reader
/// (which already holds a unit system) and [`lammps_coeff_params`].
fn coeff_params(
    unit_sys: &LammpsFfUnits,
    file_units: &str,
    category: &str,
    style: &str,
    values: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<Params, String> {
    let num = |idx: usize, what: &str| -> Result<f64, String> {
        parse_f64(get(values, idx, what, where_)?, what, where_)
    };
    let energy = |idx: usize, what: &str| -> Result<f64, String> {
        unit_sys.to_store_energy(num(idx, what)?, file_units)
    };
    match (category, style) {
        ("bond", "harmonic") => {
            let k_lammps = unit_sys.to_store_bond_k_lammps(num(0, "bond K")?, file_units)?;
            let r0 = unit_sys.to_store_length(num(1, "bond r0")?, file_units)?;
            Ok(Params::from_pairs(&[
                ("k", lammps_k_to_molrs_half_k(k_lammps)),
                ("r0", r0),
            ]))
        }
        ("angle", "harmonic") => {
            let k_lammps = unit_sys.to_store_angle_k_lammps(num(0, "angle K")?, file_units)?;
            let theta0_deg = num(1, "angle theta0")?;
            Ok(Params::from_pairs(&[
                ("k", lammps_k_to_molrs_half_k(k_lammps)),
                ("theta0", theta0_deg.to_radians()),
            ]))
        }
        // The improper kernel is LAMMPS's own form, K·(χ − χ₀)², so its `k`
        // is the file's `K`: no ½ factor, unlike bond and angle.
        ("improper", "harmonic") => {
            let k = unit_sys.to_store_angle_k_lammps(num(0, "improper K")?, file_units)?;
            let chi0_deg = num(1, "improper chi0")?;
            Ok(Params::from_pairs(&[
                ("k", k),
                ("chi0", chi0_deg.to_radians()),
            ]))
        }
        ("dihedral", "opls") => Ok(Params::from_pairs(&[
            ("k1", energy(0, "dihedral K1")?),
            ("k2", energy(1, "dihedral K2")?),
            ("k3", energy(2, "dihedral K3")?),
            ("k4", energy(3, "dihedral K4")?),
        ])),
        // E = K[1 + d·cos(nφ)]: LAMMPS `d` is a SIGN (±1), not a phase angle,
        // so it is stored verbatim as `sign`.
        ("dihedral", "harmonic") => Ok(Params::from_pairs(&[
            ("k", energy(0, "dihedral K")?),
            ("sign", num(1, "dihedral d")?),
            ("periodicity", num(2, "dihedral n")?),
        ])),
        // E = K[1 + cos(nφ − d)]; `w` is the 1-4 pair weight, kept for the pair
        // term (the torsion kernel does not read it).
        ("dihedral", "charmm") => Ok(Params::from_pairs(&[
            ("k", energy(0, "dihedral K")?),
            ("periodicity", num(1, "dihedral n")?),
            ("phase", num(2, "dihedral d")?.to_radians()),
            ("w", num(3, "dihedral w")?),
        ])),
        ("dihedral", "multi/harmonic") => Ok(Params::from_pairs(&[
            ("a1", energy(0, "dihedral A1")?),
            ("a2", energy(1, "dihedral A2")?),
            ("a3", energy(2, "dihedral A3")?),
            ("a4", energy(3, "dihedral A4")?),
            ("a5", energy(4, "dihedral A5")?),
        ])),
        // m  K1 n1 d1  [K2 n2 d2 ...]
        ("dihedral", "fourier") => {
            let m: usize = get(values, 0, "dihedral m", where_)?
                .parse()
                .map_err(|_| format!("{}: dihedral m is not an integer", where_()))?;
            let mut params = Params::new();
            for term in 0..m {
                let base = 1 + 3 * term;
                let i = term + 1;
                params.set(&format!("k{i}"), energy(base, "dihedral K")?);
                params.set(&format!("periodicity{i}"), num(base + 1, "dihedral n")?);
                params.set(
                    &format!("phase{i}"),
                    num(base + 2, "dihedral d")?.to_radians(),
                );
            }
            Ok(params)
        }
        ("pair", s) if s.starts_with("lj/cut") => Ok(Params::from_pairs(&[
            ("epsilon", energy(0, "pair epsilon")?),
            (
                "sigma",
                unit_sys.to_store_length(num(1, "pair sigma")?, file_units)?,
            ),
        ])),
        _ => Err(format!(
            "{}: unsupported LAMMPS {category} style `{style}`",
            where_()
        )),
    }
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

/// Refuse a `dihedral_style` this reader has no kernel for.
fn require_dihedral_style(name: &str, where_: &dyn Fn() -> String) -> Result<(), String> {
    let allowed = ["fourier", "opls", "harmonic", "multi/harmonic", "charmm"];
    if allowed.contains(&name) {
        return Ok(());
    }
    Err(format!(
        "{}: unsupported dihedral_style `{name}` (expected one of {})",
        where_(),
        allowed.join(", ")
    ))
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

/// A label-only type key: the type name, and the `N` endpoints it means.
///
/// LAMMPS coefficient lines carry a type label and nothing else, so this is
/// the one place molrs infers endpoints from a name
/// ([`TypeName::infer_endpoints`]). The name is the label verbatim; a numeric
/// id is first expanded through `label_map`, or to the synthetic `id-id-…`.
fn label_type<const N: usize>(
    label: Option<&&str>,
    kind: &str,
    where_: impl Fn() -> String,
    label_map: Option<&BTreeMap<u32, String>>,
) -> Result<(String, [String; N]), String> {
    let raw = label.ok_or_else(|| format!("{}: {kind}_coeff missing type label", where_()))?;
    let name = match raw.parse::<u32>() {
        Ok(id) => label_map
            .and_then(|m| m.get(&id).cloned())
            .unwrap_or_else(|| {
                std::iter::repeat_n(id.to_string(), N)
                    .collect::<Vec<_>>()
                    .join("-")
            }),
        Err(_) => (*raw).to_owned(),
    };
    let parts = TypeName::infer_endpoints(&name, N)
        .map_err(|e| format!("{}: {kind} type: {e}", where_()))?;
    let endpoints = std::array::from_fn(|i| parts[i].to_owned());
    Ok((name, endpoints))
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

fn parse_triple(
    toks: &[&str],
    start: usize,
    what: &str,
    where_: &dyn Fn() -> String,
) -> Result<[f64; 3], String> {
    Ok([
        parse_f64(get(toks, start, what, where_)?, what, where_)?,
        parse_f64(get(toks, start + 1, what, where_)?, what, where_)?,
        parse_f64(get(toks, start + 2, what, where_)?, what, where_)?,
    ])
}

fn parse_special_bonds(rest: &[&str], where_: &dyn Fn() -> String) -> Result<SpecialBonds, String> {
    let mut toks: Vec<&str> = rest.to_vec();
    while toks.len() >= 2 {
        let kind = toks[toks.len() - 2];
        let val = toks[toks.len() - 1];
        if (kind == "angle" || kind == "dihedral") && (val == "yes" || val == "no") {
            toks.truncate(toks.len() - 2);
            continue;
        }
        break;
    }
    if toks.is_empty() {
        return Err(format!("{}: special_bonds missing weights", where_()));
    }
    match toks[0] {
        // The preset is AMBER's own pair of divisors, spelled as the weights
        // LAMMPS wants: 1/SCNB and 1/SCEE.
        "amber" if toks.len() == 1 => Ok(SpecialBonds {
            lj: [0.0, 0.0, 1.0 / AMBER_SCNB],
            coul: [0.0, 0.0, 1.0 / AMBER_SCEE],
        }),
        "charmm" if toks.len() == 1 => Ok(SpecialBonds {
            lj: [0.0, 0.0, 0.0],
            coul: [0.0, 0.0, 0.0],
        }),
        "dreiding" if toks.len() == 1 => Ok(SpecialBonds {
            lj: [0.0, 0.0, 1.0],
            coul: [0.0, 0.0, 1.0],
        }),
        "fene" if toks.len() == 1 => Ok(SpecialBonds {
            lj: [0.0, 1.0, 1.0],
            coul: [0.0, 1.0, 1.0],
        }),
        "lj" | "coul" => {
            let mut lj = [0.0, 0.0, 0.0];
            let mut coul = [0.0, 0.0, 0.0];
            let mut i = 0;
            while i < toks.len() {
                match toks[i] {
                    "lj" => {
                        lj = parse_triple(&toks, i + 1, "lj weight", where_)?;
                        i += 4;
                    }
                    "coul" => {
                        coul = parse_triple(&toks, i + 1, "coul weight", where_)?;
                        i += 4;
                    }
                    other => {
                        return Err(format!(
                            "{}: unknown special_bonds token `{other}`",
                            where_()
                        ));
                    }
                }
            }
            Ok(SpecialBonds { lj, coul })
        }
        first if first.parse::<f64>().is_ok() => {
            if toks.len() != 3 {
                return Err(format!(
                    "{}: special_bonds bare weights need three numbers",
                    where_()
                ));
            }
            let w = parse_triple(&toks, 0, "weight", where_)?;
            Ok(SpecialBonds { lj: w, coul: w })
        }
        other => Err(format!(
            "{}: unknown special_bonds token `{other}`",
            where_()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::{AngleType, DihedralType, Style, StyleDefs};

    /// A LAMMPS include covering every style, with values copied from a real
    /// GAFF2 PEO `.ff` (figure5).
    const MINI: &str = r#"
# LAMMPS force field generated by molrs
special_bonds amber
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
        assert!(
            (dt.params.get("periodicity1").unwrap() - 3.0).abs() < 1e-12,
            "periodicity1"
        );
        assert!(
            (dt.params.get("phase1").unwrap() - 0.0).abs() < 1e-12,
            "phase1"
        );

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

    /// Regression: a LAMMPS `improper_style harmonic` term evaluates at the
    /// energy LAMMPS gives it. The kernel is `k·(χ − χ₀)²`, LAMMPS's own
    /// `K·(χ − χ₀)²`, so `k` must be `K`; the reader once stored `k = 2K` (the
    /// bond/angle form map) and every LAMMPS improper came out twice too high.
    #[test]
    fn a_lammps_improper_evaluates_at_the_lammps_energy() {
        use crate::ff::potential::PotentialCompiler;
        use molrs::store::block::Block;
        use molrs::store::frame::Frame;
        use molrs::types::Idx;
        use ndarray::Array1;

        let text = "special_bonds amber\n\
                    improper_style harmonic\n\
                    improper_coeff a-b-c-d 10.0 0.0\n";
        let ff = LammpsFfReader::new().read_str(text).unwrap();

        let mut impropers = Block::new();
        for (key, atom) in [("atomi", 0), ("atomj", 1), ("atomk", 2), ("atoml", 3)] {
            impropers
                .insert(key, Array1::from_vec(vec![atom as Idx]).into_dyn())
                .unwrap();
        }
        impropers
            .insert(
                "type",
                Array1::from_vec(vec!["a-b-c-d".to_string()]).into_dyn(),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("impropers", impropers);
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();

        // i = (0,1,0), j = origin, k = (1,0,0), l = (1, cos χ, sin χ): the
        // I-J-K-L dihedral is χ = 30°.
        let chi = 30.0_f64.to_radians();
        let (sin, cos) = chi.sin_cos();
        let coords = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, cos, sin];
        // LAMMPS: E = K·(χ − χ₀)² = 10 · (π/6)² kcal/mol.
        let lammps = 10.0 * (std::f64::consts::PI / 6.0).powi(2);
        let e = pots.calc_energy(&coords);
        assert!((e - lammps).abs() < 1e-10, "E = {e}, LAMMPS gives {lammps}");
    }

    #[test]
    fn special_bonds_presets_and_absent_line() {
        let amber = LammpsFfReader::new()
            .read_str("special_bonds amber\npair_style lj/cut 10.0\npair_coeff c3 c3 0.1 3.4\n")
            .unwrap();
        assert!((amber.special_bonds().coul_14() - 5.0 / 6.0).abs() < 1e-12);
        assert_eq!(amber.special_bonds().lj_14(), 0.5);

        let err = LammpsFfReader::new()
            .read_str("pair_style lj/cut 10.0\npair_coeff c3 c3 0.1 3.4\n")
            .unwrap_err();
        assert!(err.contains("special_bonds"), "{err}");
    }

    /// Reduced units stay reduced: an `lj` include declares `lj`, it is not
    /// relabelled as the store's `real`.
    #[test]
    fn units_lj_include_declares_lj_units() {
        let ff = LammpsFfReader::new()
            .read_str(
                "units lj\nspecial_bonds lj 0.0 0.0 0.0 coul 0.0 0.0 0.0\n\
                 pair_style lj/cut 2.5\npair_coeff A A 1.0 1.0\n",
            )
            .unwrap();
        assert_eq!(ff.units(), "lj");
        assert_eq!(ff.declared_units(), Some("lj"));
    }

    #[test]
    fn include_without_a_units_line_reads_as_real() {
        let ff = LammpsFfReader::new()
            .read_str("special_bonds amber\npair_style lj/cut 10.0\npair_coeff c3 c3 0.1 3.4\n")
            .unwrap();
        assert_eq!(ff.units(), "real");
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
special_bonds amber
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
        let text = "special_bonds amber
pair_style hybrid lj/cut coul/cut
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
special_bonds amber
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
        let sys = crate::ff::forcefield::lammps_units::LammpsFfUnits::canonical().unwrap();
        let expect = sys.energy(1.0, "metal", "real").unwrap();
        assert!((eps - expect).abs() < 1e-9, "eps {eps} vs {expect}");
        // bond K=1 eV/Å² → store k = 2 * K_real
        let bond = ff.get_style("bond", "harmonic").unwrap();
        let bt = bond.get_bondtype("c3", "c3").unwrap();
        let k_lammps_real = sys.bond_k_lammps(1.0, "metal", "real").unwrap();
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
special_bonds amber
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
            .read_data_coeffs(coeffs, &labels, "real")
            .unwrap();
        let bond = ff.get_style("bond", "harmonic").unwrap();
        let bt = bond.get_bondtype("OW", "HW").unwrap();
        assert!((bt.params.get("k").unwrap() - 900.0).abs() < 1e-9); // 2*450
        assert!((bt.params.get("r0").unwrap() - 0.9572).abs() < 1e-9);
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("OW", None).unwrap();
        assert!((pt.params.get("epsilon").unwrap() - 0.1521).abs() < 1e-9);
    }

    fn read_data(coeffs: &str) -> Result<ForceField, String> {
        LammpsFfReader::new().read_data_coeffs(coeffs, &LammpsTypeLabelMaps::default(), "real")
    }

    /// `Bond Coeffs # fene/kk` must not be read as harmonic: the reader has no
    /// FENE kernel, so it refuses, naming the section and the style.
    #[test]
    fn data_coeffs_unsupported_bond_hint_is_an_error() {
        let err = read_data("Bond Coeffs # fene/kk\n\n1 30 1.5 1 1\n").unwrap_err();
        assert!(err.contains("Bond Coeffs"), "{err}");
        assert!(
            err.contains("`fene`"),
            "stripped style must be named: {err}"
        );
    }

    #[test]
    fn data_coeffs_unsupported_angle_hint_is_an_error() {
        let err = read_data("Angle Coeffs # cosine/kk\n\n1 2.156\n").unwrap_err();
        assert!(err.contains("cosine"), "{err}");
        assert!(err.contains("Angle Coeffs"), "{err}");
    }

    #[test]
    fn data_coeffs_unsupported_pair_hint_is_an_error() {
        let err = read_data("Pair Coeffs # morse\n\n1 1.0 2.0 3.0\n").unwrap_err();
        assert!(err.contains("morse"), "{err}");
        assert!(err.contains("Pair Coeffs"), "{err}");
    }

    #[test]
    fn data_coeffs_angle_harmonic_hint_reads() {
        let ff = read_data("Angle Coeffs # harmonic\n\n1 50.0 109.5\n").unwrap();
        let a = ff.get_style("angle", "harmonic").unwrap();
        let at = &angle_types(a)[0];
        assert!((at.params.get("k").unwrap() - 100.0).abs() < 1e-12); // 2*50
        assert!((at.params.get("theta0").unwrap() - 109.5_f64.to_radians()).abs() < 1e-12);
    }

    #[test]
    fn data_coeffs_accelerator_suffix_is_stripped() {
        let ff = read_data("Bond Coeffs # harmonic/kk\n\n1 450.0 0.9572\n").unwrap();
        let bt = ff
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("1", "1")
            .unwrap();
        assert!((bt.params.get("k").unwrap() - 900.0).abs() < 1e-9);
        assert!((bt.params.get("r0").unwrap() - 0.9572).abs() < 1e-12);
    }

    #[test]
    fn data_coeffs_pair_lj_cut_coul_long_hint_reads() {
        let ff = read_data("Pair Coeffs # lj/cut/coul/long/kk\n\n1 0.1521 3.1507\n").unwrap();
        let pt = ff
            .get_style("pair", "lj/cut")
            .unwrap()
            .get_pairtype("1", None)
            .unwrap();
        assert!((pt.params.get("epsilon").unwrap() - 0.1521).abs() < 1e-12);
        assert!((pt.params.get("sigma").unwrap() - 3.1507).abs() < 1e-12);
    }

    /// The hint selects the dihedral kernel and its coefficient layout: four
    /// OPLS coefficients, not the default harmonic `K d n`.
    #[test]
    fn data_coeffs_dihedral_hint_selects_kernel() {
        let ff = read_data("Dihedral Coeffs # opls/omp\n\n1 1.0 2.0 3.0 4.0\n").unwrap();
        let d = ff.get_style("dihedral", "opls").unwrap();
        let dt = &dihedral_types(d)[0];
        assert!((dt.params.get("k4").unwrap() - 4.0).abs() < 1e-12);
    }

    /// A section without a `# style` hint keeps the documented default.
    #[test]
    fn data_coeffs_without_hint_defaults_to_harmonic() {
        let ff = read_data("Bond Coeffs\n\n1 450.0 0.9572\n").unwrap();
        let bt = ff
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("1", "1")
            .unwrap();
        assert!((bt.params.get("k").unwrap() - 900.0).abs() < 1e-9);
    }

    #[test]
    fn dihedral_opls_four_coeffs() {
        let text = "\
special_bonds amber
dihedral_style opls
dihedral_coeff CT-CT-CT-CT 1.0 2.0 3.0 4.0
";
        let ff = LammpsFfReader::new().read_str(text).unwrap();
        let d = ff.get_style("dihedral", "opls").unwrap();
        let dt = &dihedral_types(d)[0];
        assert!((dt.params.get("k1").unwrap() - 1.0).abs() < 1e-12);
        assert!((dt.params.get("k4").unwrap() - 4.0).abs() < 1e-12);
    }

    /// `lammps_coeff_params` returns the params the reader stores, for each kernel:
    /// harmonic `K` → `k = 2K`, degrees → radians, energies in `real` pass through.
    #[test]
    fn lammps_coeff_params_converts_each_kernel() {
        let close = |p: &Params, key: &str, want: f64| {
            let got = p.get(key).unwrap_or_else(|| panic!("missing `{key}`"));
            assert!((got - want).abs() < 1e-12, "{key}: {got} != {want}");
        };
        let keys = |p: &Params| {
            let mut k: Vec<String> = p.iter().map(|(k, _)| k.to_owned()).collect();
            k.sort();
            k
        };

        let p = lammps_coeff_params("bond", "harmonic", &["450", "0.9572"], "real").unwrap();
        close(&p, "k", 900.0);
        close(&p, "r0", 0.9572);
        assert_eq!(keys(&p), ["k", "r0"]);

        let p = lammps_coeff_params("angle", "harmonic", &["55", "104.52"], "real").unwrap();
        close(&p, "k", 110.0);
        close(&p, "theta0", 1.824_218_134_184_473_2);
        assert_eq!(keys(&p), ["k", "theta0"]);

        // The improper kernel is K·(χ − χ₀)², LAMMPS's own form: k = K.
        let p = lammps_coeff_params("improper", "harmonic", &["10", "180"], "real").unwrap();
        close(&p, "k", 10.0);
        close(&p, "chi0", std::f64::consts::PI);
        assert_eq!(keys(&p), ["chi0", "k"]);

        let p = lammps_coeff_params("dihedral", "opls", &["1", "2", "3", "4"], "real").unwrap();
        for (key, want) in [("k1", 1.0), ("k2", 2.0), ("k3", 3.0), ("k4", 4.0)] {
            close(&p, key, want);
        }
        assert_eq!(keys(&p), ["k1", "k2", "k3", "k4"]);

        let p = lammps_coeff_params("dihedral", "harmonic", &["2", "-1", "3"], "real").unwrap();
        close(&p, "k", 2.0);
        close(&p, "sign", -1.0);
        close(&p, "periodicity", 3.0);
        assert_eq!(keys(&p), ["k", "periodicity", "sign"]);

        // fourier: m, then (K n d) per term; d degrees → radians.
        let p = lammps_coeff_params(
            "dihedral",
            "fourier",
            &["2", "0.5", "1", "180", "0.25", "3", "0"],
            "real",
        )
        .unwrap();
        close(&p, "k1", 0.5);
        close(&p, "periodicity1", 1.0);
        close(&p, "phase1", std::f64::consts::PI);
        close(&p, "k2", 0.25);
        close(&p, "periodicity2", 3.0);
        close(&p, "phase2", 0.0);

        // charmm: K n d w — its own layout, not fourier's `m K n d`.
        let p =
            lammps_coeff_params("dihedral", "charmm", &["0.2", "3", "180", "0.5"], "real").unwrap();
        close(&p, "k", 0.2);
        close(&p, "periodicity", 3.0);
        close(&p, "phase", std::f64::consts::PI);
        close(&p, "w", 0.5);
        assert_eq!(keys(&p), ["k", "periodicity", "phase", "w"]);

        // multi/harmonic: A1..A5, energies.
        let p = lammps_coeff_params(
            "dihedral",
            "multi/harmonic",
            &["1", "2", "3", "4", "5"],
            "real",
        )
        .unwrap();
        for (key, want) in [
            ("a1", 1.0),
            ("a2", 2.0),
            ("a3", 3.0),
            ("a4", 4.0),
            ("a5", 5.0),
        ] {
            close(&p, key, want);
        }

        let p = lammps_coeff_params("pair", "lj/cut", &["0.066", "3.5"], "real").unwrap();
        close(&p, "epsilon", 0.066);
        close(&p, "sigma", 3.5);
        assert_eq!(keys(&p), ["epsilon", "sigma"]);
    }

    #[test]
    fn lammps_coeff_params_rejects_unsupported_kernel_and_bad_tokens() {
        let err = lammps_coeff_params("bond", "morse", &["1", "2", "3"], "real").unwrap_err();
        assert!(err.contains("bond") && err.contains("morse"), "{err}");

        let err = lammps_coeff_params("bond", "harmonic", &["450"], "real").unwrap_err();
        assert!(err.contains("r0"), "{err}");

        let err = lammps_coeff_params("bond", "harmonic", &["x", "1"], "real").unwrap_err();
        assert!(err.contains("not a number"), "{err}");

        let err = lammps_coeff_params("bond", "harmonic", &["1", "1"], "si").unwrap_err();
        assert!(err.contains("si"), "{err}");
    }

    /// The reader stores `dihedral_style charmm` under its own kernel with the
    /// `K n d w` layout, not as a fourier term.
    #[test]
    fn dihedral_charmm_reads_its_own_layout() {
        let text = "\
special_bonds charmm
dihedral_style charmm
dihedral_coeff A-B-C-D 0.2 3 180 1.0
";
        let ff = LammpsFfReader::new().read_str(text).unwrap();
        let d = ff.get_style("dihedral", "charmm").unwrap();
        let dt = &dihedral_types(d)[0];
        assert!((dt.params.get("k").unwrap() - 0.2).abs() < 1e-12);
        assert!((dt.params.get("periodicity").unwrap() - 3.0).abs() < 1e-12);
        assert!((dt.params.get("phase").unwrap() - std::f64::consts::PI).abs() < 1e-12);
        assert!(ff.get_style("dihedral", "fourier").is_none());
    }

    #[test]
    fn numeric_pair_types_stable_with_bonded_coeffs() {
        // Regression: bonded types whose endpoints look like "10" must not
        // scramble pair self-types named "10".
        let text = "\
units real
special_bonds amber
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
