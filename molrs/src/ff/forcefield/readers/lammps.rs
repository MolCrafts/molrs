//! LAMMPS force-field reader (the `*.ff` include next to a data file).
//!
//! Parses a LAMMPS force-field include — `pair_style`/`pair_coeff`,
//! `bond_style harmonic|morse`, `angle_style harmonic`, `dihedral_style`
//! `fourier` / `opls` / `harmonic` / `charmm` / `multi/harmonic`,
//! `improper_style harmonic|cvff` — with **type-label** coefficients into a
//! molrs [`ForceField`]. Inverse of
//! [`LammpsFfWriter`](crate::ff::forcefield::writers::lammps::LammpsFfWriter), e.g.:
//!
//! ```text
//! pair_style lj/cut/coul/long 10.0 10.0
//! pair_coeff c3 c3 0.107800 3.397710          # epsilon sigma
//! bond_style harmonic
//! bond_coeff c3-c3 228.890000 1.535400        # K r0
//! angle_style harmonic
//! angle_coeff c3-c3-oh 76.790000 109.660000   # K theta0(deg)
//! dihedral_style fourier
//! dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.0 # m  K1 n1 d1(deg) [K2 n2 d2 ...]
//! ```
//!
//! # The identity on coefficients
//!
//! molrs's convention **is** LAMMPS's (molrs-python docs, "Force-field
//! conventions"): every style's energy expression, factors and parameter
//! units are the LAMMPS style's, with angle-valued parameters in degrees. So
//! this reader converts nothing: each coefficient is stored as written, under
//! the molrs name of its slot, and the force field declares the file's `units`
//! (`real` when the file has no `units` line). [`lammps_coeff_params`] is that
//! one token → params map. The only renames are of style names molrs spells
//! differently: `dihedral_style fourier` is molrs's `dihedral periodic`.
//!
//! # Pair rows
//!
//! `pair_coeff i i ε σ` is atom type `i`'s own `lj/cut` row. A cross
//! `pair_coeff i j ε σ` (`i ≠ j`; CHARMM NBFIX and the like) is an explicit
//! pair type between `i` and `j`, which the `lj/cut` kernel uses for that pair
//! in place of the mixing rule. A later line for the same pair, in either
//! order, replaces an earlier one, as in LAMMPS. A cross line with a wildcard
//! (`pair_coeff c3 * …`) is refused rather than expanded.
//!
//! # Charges and masses
//!
//! Per-atom charge and mass live in the LAMMPS **data** file, not this include,
//! so they are not read here: the `coul/cut` style draws charges from the
//! [`Frame`](molrs::store::frame::Frame) at evaluation time, with LAMMPS's own
//! Coulomb constant (`qqr2e`) for the file's units.
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
use crate::ff::forcefield::lammps_units::parse_style;
use crate::ff::forcefield::mixing::Mixing;
use crate::ff::forcefield::{ForceField, Params, SpecialBonds};
use crate::ff::params::amber::{AMBER_SCEE, AMBER_SCNB};
use molrs::store::type_labels::TypeName;
use molrs::units::constants::{COULOMB_METAL, COULOMB_REAL};
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
    /// `coeffs_text` is a fragment containing `Pair Coeffs` / `PairIJ Coeffs` /
    /// `Bond Coeffs` / … (and optional `units` line). A `PairIJ Coeffs` row
    /// `i j ε σ` is the `pair_coeff i j ε σ` line: with `i ≠ j` an explicit
    /// cross pair that replaces the mixing rule for that type pair.
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
    /// carry one. The coefficients are stored as written, in `units`.
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
        let mut file_units = self.default_units;
        let mut ff = ForceField::new("LAMMPS");
        let mut pair_rows: Vec<PairRow> = Vec::new();
        let mut cutoffs: (Option<f64>, Option<f64>) = (None, None);
        let mut pair_mix: Option<String> = None;
        // The LAMMPS style each category's coefficient lines are read under.
        let mut styles: BTreeMap<&'static str, String> = BTreeMap::new();
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
                "bond_style" | "angle_style" | "dihedral_style" | "improper_style" => {
                    let category = kw.trim_end_matches("_style");
                    let category = BONDED.iter().copied().find(|c| *c == category).unwrap();
                    let name = rest
                        .first()
                        .ok_or_else(|| format!("{}: {kw} missing name", where_()))?;
                    require_bonded_style(category, name, &where_)?;
                    ff.def_style(category, molrs_style_name(category, name), Params::new())
                        .map_err(|e| e.to_string())?;
                    styles.insert(category, (*name).to_owned());
                }
                "pair_coeff" => collect_pair(&rest, &mut pair_rows, &where_, labels)?,
                "bond_coeff" | "angle_coeff" | "dihedral_coeff" | "improper_coeff" => {
                    let category = kw.trim_end_matches("_coeff");
                    let category = BONDED.iter().copied().find(|c| *c == category).unwrap();
                    let lammps_style = styles.get(category).ok_or_else(|| {
                        format!("{}: coeff before its `{category}_style`", where_())
                    })?;
                    add_bonded(&mut ff, category, lammps_style, &rest, &where_, labels)?;
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

        build_pairs(
            &mut ff,
            &pair_rows,
            cutoffs,
            pair_mix.as_deref(),
            coulomb_constant(file_units),
        )?;
        // Every number is as the file wrote it, so the force field is in the
        // file's units.
        ff.set_units(file_units);
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

/// The bonded categories, in the order a `*_style` / `*_coeff` keyword names
/// them.
const BONDED: [&str; 4] = ["bond", "angle", "dihedral", "improper"];

/// LAMMPS's Coulomb constant `qqr2e` for a `units` style, which the
/// `coul/cut` style states.
fn coulomb_constant(units: &str) -> f64 {
    match units {
        "metal" => COULOMB_METAL,
        "lj" => 1.0,
        _ => COULOMB_REAL,
    }
}

/// The molrs name of a LAMMPS style: the same, except `dihedral fourier`,
/// which is molrs's canonical multi-term `dihedral periodic`.
fn molrs_style_name<'a>(category: &str, lammps: &'a str) -> &'a str {
    match (category, lammps) {
        ("dihedral", "fourier") => "periodic",
        _ => lammps,
    }
}

/// The LAMMPS styles this reader stores, per bonded category: each has a
/// molrs kernel of the same expression.
fn supported_styles(category: &str) -> &'static [&'static str] {
    match category {
        "bond" => &["harmonic", "morse"],
        "angle" => &["harmonic"],
        "dihedral" => &["fourier", "opls", "harmonic", "multi/harmonic", "charmm"],
        "improper" => &["harmonic", "cvff"],
        _ => &[],
    }
}

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
            _ => require_bonded_style(category, style, &where_),
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
            ("pairij coeffs", "pairij"),
            ("bond coeffs", "bond"),
            ("angle coeffs", "angle"),
            ("dihedral coeffs", "dihedral"),
            ("improper coeffs", "improper"),
        ]
        .into_iter()
        .find(|(name, _)| lower.starts_with(name));
        if let Some((_, kind)) = opened {
            section = Some(kind);
            // `PairIJ Coeffs` is the pair category: one hint for both forms.
            let category = if kind == "pairij" { "pair" } else { kind };
            match SectionStyleHint::parse(raw) {
                Some(hint) => hints.insert(category, hint),
                None => hints.remove(category),
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
        let atom_label = |id: u32| {
            labels
                .atom
                .get(&id)
                .cloned()
                .unwrap_or_else(|| id.to_string())
        };
        let type_tok = match kind {
            "pair" | "pairij" => atom_label(id),
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
            "pairij" => {
                // PairIJ Coeffs: i j ε σ  →  pair_coeff Ti Tj ε σ. A row with
                // i ≠ j is an explicit cross pair, kept as a pair type of its
                // own (it used to end the coefficient sections, unread).
                if parts.len() < 4 {
                    return Err(format!(
                        "line {}: PairIJ Coeffs needs `i j epsilon sigma`",
                        lineno + 1
                    ));
                }
                let j: u32 = parts[1].parse().map_err(|_| {
                    format!(
                        "line {}: expected integer type id in PairIJ Coeffs, got {}",
                        lineno + 1,
                        parts[1]
                    )
                })?;
                out.push_str(&format!(
                    "pair_coeff {type_tok} {} {} {}\n",
                    atom_label(j),
                    parts[2],
                    parts[3]
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

/// One `pair_coeff` row: the two atom types (equal for a self pair) and its
/// `lj/cut` parameters.
type PairRow = (String, String, Params);

fn collect_pair(
    rest: &[&str],
    rows: &mut Vec<PairRow>,
    where_: &dyn Fn() -> String,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    // pair_coeff <i> <j> [sub-style] <epsilon> <sigma>. A self pair i==j is an
    // atom type's own row; a cross pair i!=j is an explicit override (NBFIX) that
    // the kernel uses in place of the combining rule. A later line for the same
    // pair replaces an earlier one, as in LAMMPS.
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
    if ti != tj && (ti.contains('*') || tj.contains('*')) {
        return Err(format!(
            "{}: pair_coeff `{ti} {tj}` is a wildcard cross pair; expand it to explicit \
             type pairs",
            where_()
        ));
    }
    let params = coeff_params("pair", "lj/cut", args, where_)?;
    let same = |(a, b, _): &PairRow| (a == &ti && b == &tj) || (a == &tj && b == &ti);
    match rows.iter_mut().find(|row| same(row)) {
        Some(row) => *row = (ti, tj, params),
        None => rows.push((ti, tj, params)),
    }
    Ok(())
}

/// Emit the collected LJ rows (self pairs and explicit cross pairs) as a `lj/cut` style plus a `coul/cut` style
/// (charges resolved from the frame).
///
/// `coul/cut` is the **buffered** Coulomb `E = k·qᵢqⱼ/(D·(r + δ))`; a LAMMPS
/// force field is the unbuffered case (δ = 0) in vacuum with LAMMPS's `k`
/// (`qqr2e`) for its units. The force field states the constant explicitly:
/// the kernel has no default, because MMFF's `k` is a different number and
/// both are correct.
fn build_pairs(
    ff: &mut ForceField,
    rows: &[PairRow],
    cutoffs: (Option<f64>, Option<f64>),
    mix: Option<&str>,
    coulomb: f64,
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
    for (ti, tj, params) in rows {
        if ti == tj {
            lj.def_type(ti, &[ti], params.clone())
        } else {
            let name = TypeName::pair(ti, tj)?;
            lj.def_type(name.as_str(), &[ti, tj], params.clone())
        }
        .map_err(|e| e.to_string())?;
    }
    let mut coul_params = vec![("coulomb", coulomb), ("dielectric", VACUUM_DIELECTRIC)];
    if let Some(c) = cut_coul {
        coul_params.push(("cutoff", c));
    }
    ff.def_style("pair", "coul/cut", Params::from_pairs(&coul_params))
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ── bonded ──────────────────────────────────────────────────────────────────

/// One `<category>_coeff <type> <values…>` line under its LAMMPS style.
fn add_bonded(
    ff: &mut ForceField,
    category: &'static str,
    lammps_style: &str,
    rest: &[&str],
    where_: &dyn Fn() -> String,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    let (name, endpoints) = match category {
        "bond" => {
            let (name, e) = label_type::<2>(rest.first(), category, where_, Some(&labels.bond))?;
            (name, e.to_vec())
        }
        "angle" => {
            let (name, e) = label_type::<3>(rest.first(), category, where_, Some(&labels.angle))?;
            (name, e.to_vec())
        }
        "dihedral" => {
            let map = Some(&labels.dihedral);
            let (name, e) = label_type::<4>(rest.first(), category, where_, map)?;
            (name, e.to_vec())
        }
        _ => {
            let map = Some(&labels.improper);
            let (name, e) = label_type::<4>(rest.first(), category, where_, map)?;
            (name, e.to_vec())
        }
    };
    let params = coeff_params(category, lammps_style, &rest[1..], where_)?;
    let ends: Vec<&str> = endpoints.iter().map(String::as_str).collect();
    let directive = format!("{category}_style {lammps_style}");
    style_mut(
        ff,
        category,
        molrs_style_name(category, lammps_style),
        &directive,
        where_,
    )?
    .def_type(&name, &ends, params)
    .map_err(|e| e.to_string())?;
    Ok(())
}

// ── coefficient conversion ──────────────────────────────────────────────────

/// One LAMMPS coefficient line as the molrs params the reader stores.
///
/// `values` are the coefficient tokens **after** the type field(s) — for
/// `bond_coeff c3-c3 228.89 1.5354` that is `["228.89", "1.5354"]`, for
/// `pair_coeff c3 c3 0.1078 3.3977` it is `["0.1078", "3.3977"]`. `units` is the
/// LAMMPS `units` keyword the numbers are written in (`real`, `metal`, `lj`),
/// which the params are in too: molrs's convention is LAMMPS's, so every
/// value is stored as written — the ½ inside LAMMPS's `K`, degrees for every
/// angle-valued slot — under the molrs name of its slot:
///
/// | category / style        | LAMMPS tokens        | stored params |
/// |-------------------------|----------------------|---------------|
/// | `bond harmonic`         | `K r0`               | `k`, `r0` |
/// | `bond morse`            | `D0 alpha r0`        | `d0`, `alpha`, `r0` |
/// | `angle harmonic`        | `K theta0`           | `k`, `theta0` (deg) |
/// | `improper harmonic`     | `K chi0`             | `k`, `chi0` (deg) |
/// | `improper cvff`         | `K d n`              | `k`, `sign = d` (±1), `periodicity = n` |
/// | `dihedral opls`         | `K1 K2 K3 K4`        | `k1..k4` |
/// | `dihedral harmonic`     | `K d n`              | `k`, `sign = d` (±1), `periodicity = n` |
/// | `dihedral fourier`      | `m K1 n1 d1 …`       | molrs `periodic`: `k<i>`, `periodicity<i>`, `phase<i>` (deg) |
/// | `dihedral charmm`       | `K n d w`            | `k`, `periodicity`, `phase` (deg), `w` |
/// | `dihedral multi/harmonic` | `A1 A2 A3 A4 A5`   | `a1..a5` |
/// | `pair lj/cut…`          | `epsilon sigma`      | `epsilon`, `sigma` |
///
/// LAMMPS's single-letter `n` and `d` take molrs's descriptive names because
/// LAMMPS spells two different things `d`: a phase (`charmm`, `fourier`) and a
/// sign (`harmonic`, `cvff`). Any `pair` style spelled `lj/cut…`
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
/// assert_eq!(p.get("k"), Some(450.0));
/// let p = lammps_coeff_params("angle", "harmonic", &["55", "104.52"], "real").unwrap();
/// assert_eq!(p.get("theta0"), Some(104.52));
/// assert!(lammps_coeff_params("bond", "fene", &["1", "2", "3", "4"], "real").is_err());
/// ```
pub fn lammps_coeff_params(
    category: &str,
    style: &str,
    values: &[&str],
    units: &str,
) -> Result<Params, String> {
    parse_style(units)?;
    coeff_params(category, style, values, &|| format!("{category} {style}"))
}

/// The one LAMMPS-coefficient → molrs-params map, shared by the reader and
/// [`lammps_coeff_params`]. It renames slots and converts nothing.
fn coeff_params(
    category: &str,
    style: &str,
    values: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<Params, String> {
    let num = |idx: usize, what: &str| -> Result<f64, String> {
        parse_f64(get(values, idx, what, where_)?, what, where_)
    };
    let slots = |names: &[(&str, &str)]| -> Result<Params, String> {
        let mut params = Params::new();
        for (idx, (key, what)) in names.iter().enumerate() {
            params.set(key, num(idx, what)?);
        }
        Ok(params)
    };
    match (category, style) {
        ("bond", "harmonic") => slots(&[("k", "bond K"), ("r0", "bond r0")]),
        ("bond", "morse") => slots(&[
            ("d0", "bond D0"),
            ("alpha", "bond alpha"),
            ("r0", "bond r0"),
        ]),
        ("angle", "harmonic") => slots(&[("k", "angle K"), ("theta0", "angle theta0")]),
        ("improper", "harmonic") => slots(&[("k", "improper K"), ("chi0", "improper chi0")]),
        // E = K[1 + d·cos(nφ)]: `d` is a SIGN (±1), not a phase angle.
        ("improper", "cvff") | ("dihedral", "harmonic") => {
            slots(&[("k", "K"), ("sign", "d"), ("periodicity", "n")])
        }
        ("dihedral", "opls") => slots(&[
            ("k1", "dihedral K1"),
            ("k2", "dihedral K2"),
            ("k3", "dihedral K3"),
            ("k4", "dihedral K4"),
        ]),
        // E = K[1 + cos(nφ − d)]; `w` is the 1-4 weight of the dihedral's own
        // 1-4 pair (see `dihedral::charmm`).
        ("dihedral", "charmm") => slots(&[
            ("k", "dihedral K"),
            ("periodicity", "dihedral n"),
            ("phase", "dihedral d"),
            ("w", "dihedral w"),
        ]),
        ("dihedral", "multi/harmonic") => slots(&[
            ("a1", "dihedral A1"),
            ("a2", "dihedral A2"),
            ("a3", "dihedral A3"),
            ("a4", "dihedral A4"),
            ("a5", "dihedral A5"),
        ]),
        // m  K1 n1 d1  [K2 n2 d2 ...]
        ("dihedral", "fourier") => {
            let m: usize = get(values, 0, "dihedral m", where_)?
                .parse()
                .map_err(|_| format!("{}: dihedral m is not an integer", where_()))?;
            let mut params = Params::new();
            for term in 0..m {
                let base = 1 + 3 * term;
                let i = term + 1;
                params.set(&format!("k{i}"), num(base, "dihedral K")?);
                params.set(&format!("periodicity{i}"), num(base + 1, "dihedral n")?);
                params.set(&format!("phase{i}"), num(base + 2, "dihedral d")?);
            }
            Ok(params)
        }
        ("pair", s) if s.starts_with("lj/cut") => {
            slots(&[("epsilon", "pair epsilon"), ("sigma", "pair sigma")])
        }
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

/// Refuse a bonded `*_style` this reader has no kernel for.
fn require_bonded_style(
    category: &str,
    name: &str,
    where_: &dyn Fn() -> String,
) -> Result<(), String> {
    let allowed = supported_styles(category);
    if allowed.contains(&name) {
        return Ok(());
    }
    Err(format!(
        "{}: unsupported {category}_style `{name}` (expected one of {})",
        where_(),
        allowed.join(", ")
    ))
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
pair_coeff c3 c3 0.107800 3.397710   # duplicate; the last one is used

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

        // Every coefficient is stored as written: bond K, r0.
        let bond = ff.get_style("bond", "harmonic").unwrap();
        let bt = bond.get_bondtype("c3", "c3").unwrap();
        assert_eq!(bt.params.get("k"), Some(228.89), "k");
        assert_eq!(bt.params.get("r0"), Some(1.5354), "r0");

        // angle: K, theta0 in degrees.
        let angle = ff.get_style("angle", "harmonic").unwrap();
        let at = &angle_types(angle)[0];
        assert_eq!(at.params.get("k"), Some(76.79), "ak");
        assert_eq!(at.params.get("theta0"), Some(109.66), "theta0");

        // dihedral fourier is molrs's `periodic`: k1/periodicity1/phase1 (deg).
        assert!(ff.get_style("dihedral", "fourier").is_none());
        let dih = ff.get_style("dihedral", "periodic").unwrap();
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
    /// energy LAMMPS gives it, `K·(χ − χ₀)²` with `χ₀` in degrees; the reader
    /// once stored `k = 2K` (the old bond/angle form map) and every LAMMPS
    /// improper came out twice too high.
    #[test]
    fn a_lammps_improper_evaluates_at_the_lammps_energy() {
        use crate::ff::potential::PotentialCompiler;
        use molrs::store::block::Block;
        use molrs::store::frame::Frame;
        use molrs::types::Idx;
        use ndarray::Array1;

        let text = "special_bonds amber\n\
                    improper_style harmonic\n\
                    improper_coeff a-b-c-d 10.0 10.0\n";
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
        // LAMMPS: E = K·(χ − χ₀)² = 10 · (20°)² kcal/mol, in radians².
        let lammps = 10.0 * 20.0_f64.to_radians().powi(2);
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

    /// A cross `pair_coeff i j` (NBFIX) is a pair type of its own. The reader used
    /// to drop it silently, so the pair was priced by the mixing rule instead.
    #[test]
    fn a_cross_pair_coeff_is_kept_as_a_pair_type() {
        let text = "\
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 0.1078 3.39771
pair_coeff oh oh 0.0930 3.24287
pair_coeff c3 oh 0.2500 3.10000
";
        let ff = LammpsFfReader::new().read_str(text).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("c3", Some("oh")).expect("cross row c3-oh");
        assert!((pt.params.get("epsilon").unwrap() - 0.25).abs() < 1e-12);
        assert!((pt.params.get("sigma").unwrap() - 3.1).abs() < 1e-12);
        assert!(lj.get_pairtype("c3", None).is_some());
    }

    /// LAMMPS uses the last `pair_coeff` given for a pair.
    #[test]
    fn a_repeated_pair_coeff_replaces_the_earlier_one() {
        let text = "\
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 0.1078 3.39771
pair_coeff c3 c3 0.2000 3.00000
pair_coeff c3 oh 0.2500 3.10000
pair_coeff oh c3 0.3000 3.20000
";
        let ff = LammpsFfReader::new().read_str(text).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let own = lj.get_pairtype("c3", None).unwrap();
        assert!((own.params.get("epsilon").unwrap() - 0.2).abs() < 1e-12);
        let cross = lj.get_pairtype("c3", Some("oh")).expect("cross row");
        assert!((cross.params.get("epsilon").unwrap() - 0.3).abs() < 1e-12);
    }

    /// `pair_coeff 1 *` names a cross pair against every type; expanding it is
    /// not implemented, so it is refused rather than dropped.
    #[test]
    fn a_wildcard_cross_pair_coeff_is_an_error() {
        let text = "\
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 * 0.1078 3.39771
";
        assert!(LammpsFfReader::new().read_str(text).is_err());
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

    /// A `metal` file is read in `metal`: the numbers are not converted, the
    /// force field declares `metal`, and its Coulomb constant is LAMMPS's
    /// `metal` `qqr2e`.
    #[test]
    fn metal_units_are_kept_and_declared() {
        let text = "\
units metal
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 1.0 3.4
bond_style harmonic
bond_coeff c3-c3 1.0 1.5
";
        let ff = LammpsFfReader::new().read_str(text).unwrap();
        assert_eq!(ff.units(), "metal");
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            lj.get_pairtype("c3", None).unwrap().params.get("epsilon"),
            Some(1.0)
        );
        assert_eq!(lj.params().get("cutoff"), Some(10.0));
        let bt = ff
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("c3", "c3")
            .unwrap();
        assert_eq!(bt.params.get("k"), Some(1.0));
        assert_eq!(bt.params.get("r0"), Some(1.5));
        let coul = ff.get_style("pair", "coul/cut").unwrap();
        assert_eq!(coul.params().get("coulomb"), Some(COULOMB_METAL));
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
        assert_eq!(bt.params.get("k"), Some(450.0));
        assert!((bt.params.get("r0").unwrap() - 0.9572).abs() < 1e-9);
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("OW", None).unwrap();
        assert!((pt.params.get("epsilon").unwrap() - 0.1521).abs() < 1e-9);
    }

    /// `PairIJ Coeffs` rows are pair_coeff lines: the self rows are the types'
    /// own, `1 2` an explicit cross pair. The section used to be skipped.
    #[test]
    fn data_coeffs_pairij_rows_keep_their_cross_pairs() {
        let mut labels = LammpsTypeLabelMaps::default();
        labels.atom.insert(1, "c3".into());
        labels.atom.insert(2, "oh".into());
        let coeffs = "\
PairIJ Coeffs # lj/cut

1 1 0.1078 3.39771
1 2 0.2500 3.10000
2 2 0.0930 3.24287
";
        let ff = LammpsFfReader::new()
            .read_data_coeffs(coeffs, &labels, "real")
            .unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let cross = lj.get_pairtype("c3", Some("oh")).expect("cross row c3-oh");
        assert!((cross.params.get("epsilon").unwrap() - 0.25).abs() < 1e-12);
        assert!((cross.params.get("sigma").unwrap() - 3.1).abs() < 1e-12);
        let own = lj.get_pairtype("oh", None).expect("self row oh");
        assert!((own.params.get("epsilon").unwrap() - 0.093).abs() < 1e-12);
        assert_eq!(lj.type_rows().len(), 3);
    }

    #[test]
    fn data_coeffs_pairij_row_needs_both_ids() {
        let err = read_data("PairIJ Coeffs\n\n1 0.1 3.0\n").unwrap_err();
        assert!(err.contains("PairIJ Coeffs"), "{err}");
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
        assert_eq!(at.params.get("k"), Some(50.0));
        assert_eq!(at.params.get("theta0"), Some(109.5));
    }

    #[test]
    fn data_coeffs_accelerator_suffix_is_stripped() {
        let ff = read_data("Bond Coeffs # harmonic/kk\n\n1 450.0 0.9572\n").unwrap();
        let bt = ff
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("1", "1")
            .unwrap();
        assert_eq!(bt.params.get("k"), Some(450.0));
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
        assert_eq!(bt.params.get("k"), Some(450.0));
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

    /// `lammps_coeff_params` returns the params the reader stores, for each
    /// kernel: every value as written, under its molrs slot name.
    #[test]
    fn lammps_coeff_params_is_the_identity_on_values() {
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
        close(&p, "k", 450.0);
        close(&p, "r0", 0.9572);
        assert_eq!(keys(&p), ["k", "r0"]);

        let p = lammps_coeff_params("bond", "morse", &["95.6", "2.0", "1.53"], "metal").unwrap();
        close(&p, "d0", 95.6);
        close(&p, "alpha", 2.0);
        close(&p, "r0", 1.53);
        assert_eq!(keys(&p), ["alpha", "d0", "r0"]);

        let p = lammps_coeff_params("angle", "harmonic", &["55", "104.52"], "real").unwrap();
        close(&p, "k", 55.0);
        close(&p, "theta0", 104.52);
        assert_eq!(keys(&p), ["k", "theta0"]);

        let p = lammps_coeff_params("improper", "harmonic", &["10", "180"], "real").unwrap();
        close(&p, "k", 10.0);
        close(&p, "chi0", 180.0);
        assert_eq!(keys(&p), ["chi0", "k"]);

        let p = lammps_coeff_params("improper", "cvff", &["1.1", "-1", "2"], "real").unwrap();
        close(&p, "k", 1.1);
        close(&p, "sign", -1.0);
        close(&p, "periodicity", 2.0);
        assert_eq!(keys(&p), ["k", "periodicity", "sign"]);

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

        // fourier: m, then (K n d) per term; d stays degrees.
        let p = lammps_coeff_params(
            "dihedral",
            "fourier",
            &["2", "0.5", "1", "180", "0.25", "3", "0"],
            "real",
        )
        .unwrap();
        close(&p, "k1", 0.5);
        close(&p, "periodicity1", 1.0);
        close(&p, "phase1", 180.0);
        close(&p, "k2", 0.25);
        close(&p, "periodicity2", 3.0);
        close(&p, "phase2", 0.0);

        // charmm: K n d w — its own layout, not fourier's `m K n d`.
        let p =
            lammps_coeff_params("dihedral", "charmm", &["0.2", "3", "180", "0.5"], "real").unwrap();
        close(&p, "k", 0.2);
        close(&p, "periodicity", 3.0);
        close(&p, "phase", 180.0);
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
        let err = lammps_coeff_params("bond", "fene", &["1", "2", "3", "4"], "real").unwrap_err();
        assert!(err.contains("bond") && err.contains("fene"), "{err}");

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
        assert_eq!(dt.params.get("phase"), Some(180.0));
        assert!(ff.get_style("dihedral", "periodic").is_none());
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
