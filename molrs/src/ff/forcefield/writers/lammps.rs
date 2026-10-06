//! LAMMPS force-field coefficient writer: the `*.ff` include next to a data
//! file, and the data file's `* Coeffs` sections.
//!
//! # Coefficient writing, not whole-FF serialization
//!
//! molrs has two kinds of force-field writer:
//!
//! - **Coefficient writing** (this module, LAMMPS only) answers "which
//!   coefficients does this system's data file need". It is keyed by the
//!   system's type labels ([`TypeLabels`]): one coefficient per label, in label
//!   id order. A label the [`ForceField`] does not define is an error naming the
//!   block and the label; a `ForceField` type no label uses is not written
//!   (assembly retyping legitimately leaves stale types behind).
//! - **Whole-FF serialization** ([`super::gromacs`], [`super::xml`]) writes
//!   every type the `ForceField` holds, as a force-field file, and takes no
//!   labels.
//!
//! The data-file writer and this writer are composed by the caller; neither
//! calls the other.
//!
//! # Label matching
//!
//! - `atoms` labels select pair coefficients: the self pair named by each label
//!   (missing → error), plus explicit cross pairs whose two atom types are both
//!   labels.
//! - `bonds`, `angles`, `dihedrals` and `impropers` labels match a
//!   `ForceField` type name exactly: a label is the name of the type it
//!   stands for, and `h1-c3` does not find a type named `c3-h1`.
//! - A block whose types carry no labels (pure-integer types) is matched by
//!   its ids, `"1"`, `"2"`, ….
//! - A style is written only when it holds a used type; an unsupported style
//!   is an error only then. Type-less pair styles (`coul/cut`) apply to every
//!   atom and are always in play.
//!
//! # The identity on coefficients
//!
//! Inverse of [`super::super::readers::lammps::LammpsFfReader`]. The
//! force-field IR follows the LAMMPS standard — every style's expression,
//! factors and parameter units, with angle-valued parameters in degrees — so a
//! coefficient is
//! written as it is stored:
//!
//! ```text
//! pair_style lj/cut/coul/cut 10.0 10.0
//! pair_coeff c3 c3 0.107800 3.397710          # epsilon sigma
//! bond_style harmonic
//! bond_coeff c3-c3 228.890000 1.535400        # k r0
//! angle_style harmonic
//! angle_coeff c3-c3-oh 76.790000 109.660000   # k theta0(deg)
//! dihedral_style fourier
//! dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.0 # m  k1 periodicity1 phase1 ...
//! ```
//!
//! The file is written in [`LammpsWriteOptions::units`] (default `real`). A
//! force field declared in those units ([`ForceField::units`]) is written
//! number for number; one declared in another LAMMPS unit style (`real`,
//! `metal`, `lj`) has its energies and lengths converted through
//! [`LammpsFfUnits`] (`from → lj hub → to`) — never ad-hoc eV/kcal factors.
//! Angles need no conversion in any unit style.
//!
//! Two styles are written under another LAMMPS name: molrs's `dihedral
//! periodic` is LAMMPS's `fourier`, term for term, and AMBER's `improper
//! periodic` with one term at phase 0° or 180° is LAMMPS's `cvff`
//! (`d = cos phase`) — the atom order needs no change, because both price the
//! dihedral I-J-K-L of the stored order (see `improper::periodic`).
//!
//! A category whose used types span several LAMMPS styles (say `angle
//! harmonic` and `angle charmm`) is written as one `angle_style hybrid
//! harmonic charmm` line, each `angle_coeff` naming its sub-style; a data
//! file's section is `Angle Coeffs # hybrid` with the sub-style on each row.
//! Every data-file section names its style in the header comment, as LAMMPS's
//! `write_data` does.
//!
//! # CMAP crossterms (`fix cmap`)
//!
//! A `cmaps` block's labels select `cmap charmm` rows the same way, in id
//! order: [`LammpsFfWriter::write_cmap_str`] writes their grids as the
//! `fix cmap` file (map `t` is crossterm type `t`, the id the data writer
//! gives the `CMAP` section), and the include names that file
//! ([`LammpsWriteOptions::cmap_file`]) on a `fix cmap all cmap <file>` line,
//! with `fix_modify cmap energy yes` so the crossterms count in `pe`. LAMMPS
//! reads the crossterms with the data file, so the fix must precede
//! `read_data <data> fix cmap crossterm CMAP` (LAMMPS takes it before the box
//! exists); the include writes it first, beside `units`.
//!
//! # Pair style layout
//!
//! The reader splits a combined `lj/cut/coul/*` kernel into `lj/cut` + `coul/cut`
//! styles. This writer recombines that pair into one `pair_style lj/cut/coul/cut`
//! line so LAMMPS keeps geometric mixing on LJ (writing them as `hybrid` with a
//! `pair_coeff * * coul/cut` wildcard marks every cross pair as explicit and
//! defeats mixing). A force field that already holds a single combined-style
//! name, or only one of the two halves, is written as-is. `lj/charmm` +
//! `coul/charmm` is `pair_style lj/charmm/coul/charmm`, its only LAMMPS
//! spelling, `pair_coeff i j epsilon sigma epsilon14 sigma14`.
//!
//! # Per-pair 1-4 overrides
//!
//! LAMMPS has no per-pair exception. A frame whose `pairs` block carries an
//! override column ([`PAIR_OVERRIDE_COLUMNS`](molrs::store::schema::PAIR_OVERRIDE_COLUMNS))
//! is refused by name — by the data-file writer, and by
//! [`refuse_pair_overrides`] for a caller writing a force field for it.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::ForceFieldWriter;
use crate::ff::forcefield::lammps_units::{LammpsFfUnits, parse_style};
use crate::ff::forcefield::mixing::Mixing;
use crate::ff::forcefield::one_four::OneFour;
use crate::ff::forcefield::readers::lammps::{LAMMPS_CMAP_DIM, LAMMPS_CMAP_MAX};
use crate::ff::forcefield::torsion::nharmonic_coefficients;
use crate::ff::forcefield::{
    AngleType, BondType, CmapType, DihedralType, ForceField, ImproperType, PairType, Params, Style,
    StyleDefs,
};
use molrs::store::type_labels::{TypeLabels, TypeName};
use ndarray::ArrayD;

/// The cutoff a `pair_style` line needs. The writer never invents one
/// (operator 2026-09-28: the user provides the cutoff); a style without one is
/// an error unless the caller skips the `pair_style` line.
fn required_cutoff(style: &Style) -> Result<f64, String> {
    style_cutoff(style).ok_or_else(|| {
        format!(
            "pair style '{}' has no cutoff: declare one on the style or write with \
             skip_pair_style",
            style.name()
        )
    })
}

/// Formatting options for [`LammpsFfWriter`].
#[derive(Debug, Clone)]
pub struct LammpsWriteOptions {
    /// Decimal places for floating-point coefficients (default 6).
    pub precision: usize,
    /// When true, omit protocol commands the caller sets in the input script:
    /// `pair_style` **and** `special_bonds`. A coeff-only include that still
    /// emits Amber `special_bonds` (coul 1-4 = 1/1.2) silently overrides a
    /// later `special_bonds` in `in.lammps` if the include is sourced first —
    /// or, if the input never restates 1-4, leaves Amber weights in effect.
    pub skip_pair_style: bool,
    /// When true, omit the `units` line so the include can follow `units` /
    /// `atom_style` / `pair_style` in the input (LAMMPS rejects `units` after
    /// the box exists, and a second `units` is redundant).
    pub skip_units: bool,
    /// LAMMPS `units` style for the written include (default `"real"`).
    pub units: &'static str,
    /// The `fix cmap` file the include names, as the input script will find
    /// it — where [`LammpsFfWriter::write_cmap_str`]'s text is saved. Needed
    /// exactly when the system has CMAP crossterms (default `None`).
    pub cmap_file: Option<String>,
}

impl Default for LammpsWriteOptions {
    fn default() -> Self {
        Self {
            precision: 6,
            skip_pair_style: false,
            skip_units: false,
            units: "real",
            cmap_file: None,
        }
    }
}

/// The fix id the include gives `fix cmap`.
const CMAP_FIX_ID: &str = "cmap";

/// Conversion context: the force field's units → the file's, through the lj
/// hub. The identity when the two are the same unit style.
struct WriteUnits {
    sys: LammpsFfUnits,
    from: &'static str,
    to: &'static str,
}

impl WriteUnits {
    fn new(from: &'static str, to: &'static str) -> Result<Self, String> {
        Ok(Self {
            sys: LammpsFfUnits::canonical().map_err(|e| format!("lammps unit system: {e}"))?,
            from,
            to,
        })
    }

    /// The units of `ff`, written as `to`. A force field in a unit system
    /// LAMMPS has no `units` style for is refused only when it would need
    /// converting.
    fn of(ff: &ForceField, to: &'static str) -> Result<Self, String> {
        let from = match parse_style(ff.units()) {
            Ok(from) => from,
            Err(_) if ff.units() == to => to,
            Err(e) => return Err(format!("force field units: {e}")),
        };
        Self::new(from, to)
    }

    fn energy(&self, value: f64) -> Result<f64, String> {
        self.sys.energy(value, self.from, self.to)
    }

    fn length(&self, value: f64) -> Result<f64, String> {
        self.sys.length(value, self.from, self.to)
    }

    /// Bond stiffness, energy/length².
    fn bond_k(&self, value: f64) -> Result<f64, String> {
        self.sys.bond_k(value, self.from, self.to)
    }

    /// Angle-like stiffness, energy/rad²: radians are pure numbers, so it
    /// converts as an energy.
    fn angle_k(&self, value: f64) -> Result<f64, String> {
        self.energy(value)
    }

    /// Inverse length (Morse `alpha`).
    fn inverse_length(&self, value: f64) -> Result<f64, String> {
        Ok(1.0 / self.length(1.0 / value)?)
    }
}

/// One field of a LAMMPS `*_coeff` line. LAMMPS parses a dihedral
/// multiplicity and the fourier term count as integers, so they are written
/// without decimals.
#[derive(Debug, Clone, Copy)]
enum Coeff {
    Real(f64),
    Int(i64),
}

impl Coeff {
    fn value(self) -> f64 {
        match self {
            Self::Real(v) => v,
            Self::Int(n) => n as f64,
        }
    }

    fn render(self, precision: usize) -> String {
        match self {
            Self::Real(v) => fmt_num(v, precision),
            Self::Int(n) => n.to_string(),
        }
    }
}

/// Render one type's molrs params as the numbers of its LAMMPS coefficient
/// line — the inverse of
/// [`lammps_coeff_params`](crate::ff::forcefield::readers::lammps::lammps_coeff_params)
/// and the conversion every `*_coeff` line and `* Coeffs` row this writer
/// emits goes through.
///
/// The result is the coefficients **after** the type field(s). `params` and
/// the result are both in the LAMMPS `units` style `units` (`real`, `metal`,
/// `lj`): the map is the identity on values, slot for slot.
///
/// | category / style          | stored params                          | LAMMPS values |
/// |---------------------------|----------------------------------------|---------------|
/// | `bond harmonic`           | `k`, `r0`                              | `K r0` |
/// | `bond morse`              | `d0`, `alpha`, `r0`                    | `D0 alpha r0` |
/// | `angle harmonic`          | `k`, `theta0` (deg)                    | `K theta0` |
/// | `angle charmm`            | `k`, `theta0` (deg), `k_ub`, `r_ub`    | `K theta0 K_ub r_ub` |
/// | `improper harmonic`       | `k`, `chi0` (deg)                      | `K chi0` |
/// | `improper cvff`           | `k`, `sign` (±1), `periodicity`        | `K d n` |
/// | `improper periodic`       | `k`, `periodicity`, `phase` (deg, 0 or 180) | `K d n` (LAMMPS `cvff`, `d = cos phase`) |
/// | `dihedral opls`           | `k1..k4` (absent → 0)                  | `K1 K2 K3 K4` |
/// | `dihedral harmonic`       | `k`, `sign` (±1), `periodicity`        | `K d n` |
/// | `dihedral periodic`       | `k<i>`, `periodicity<i>`, `phase<i>` (deg, absent → 0), or one term as `k`, `periodicity`, `phase` | LAMMPS `fourier`: `m K1 n1 d1 …` |
/// | `dihedral charmm`         | `k`, `periodicity`, `phase` (deg, absent → 0), `w` | `K n d w` |
/// | `dihedral multi/harmonic` | `a1..a5` (absent → 0)                  | `A1 A2 A3 A4 A5` |
/// | `dihedral nharmonic`      | `a1..aN` (contiguous, N ≥ 1)           | `N A1 … AN` |
/// | `pair lj/cut…`            | `epsilon`, `sigma`                     | `epsilon sigma` |
///
/// An absent param falls back only where the molrs kernel reads the same
/// default. Multiplicities (`n`, fourier `m`) are integral.
///
/// # Errors
///
/// A `(category, style)` LAMMPS output has no form for, an unknown `units`
/// keyword, a missing param (named), or a non-integral multiplicity.
///
/// ```
/// use molrs::ff::forcefield::Params;
/// use molrs::ff::forcefield::writers::lammps::lammps_coeff_values;
///
/// let p = Params::from_pairs(&[("k", 450.0), ("r0", 0.9572)]);
/// assert_eq!(lammps_coeff_values("bond", "harmonic", &p, "real").unwrap(), [450.0, 0.9572]);
/// assert!(lammps_coeff_values("bond", "fene", &p, "real").is_err());
/// ```
pub fn lammps_coeff_values(
    category: &str,
    style: &str,
    params: &Params,
    units: &str,
) -> Result<Vec<f64>, String> {
    let units = parse_style(units)?;
    let units = WriteUnits::new(units, units)?;
    Ok(coeff_fields(&units, category, style, params)?
        .into_iter()
        .map(Coeff::value)
        .collect())
}

/// `fields` joined as the tail of a `*_coeff` line or `* Coeffs` row.
fn render_coeffs(fields: &[Coeff], precision: usize) -> String {
    fields
        .iter()
        .map(|c| c.render(precision))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The LAMMPS spelling of a molrs style: the same name, except AMBER's
/// `improper periodic`, which LAMMPS calls `cvff` (see [`coeff_fields`]).
fn lammps_style_name<'a>(category: &str, style: &'a str) -> &'a str {
    match (category, style) {
        ("improper", "periodic") => "cvff",
        // The canonical multi-term torsion is LAMMPS's `fourier`, term for term.
        ("dihedral", "periodic") => "fourier",
        _ => style,
    }
}

/// The one molrs-params → LAMMPS-coefficient conversion, shared by the writer
/// (which already holds its [`WriteUnits`]) and [`lammps_coeff_values`].
fn coeff_fields(
    units: &WriteUnits,
    category: &str,
    style: &str,
    params: &Params,
) -> Result<Vec<Coeff>, String> {
    let need = |key: &str| {
        params
            .get(key)
            .ok_or_else(|| format!("{category} {style}: missing param `{key}`"))
    };
    let multiplicity = |key: &str, n: f64| {
        if n.fract() == 0.0 {
            Ok(Coeff::Int(n as i64))
        } else {
            Err(format!(
                "{category} {style}: param `{key}` = {n} is not an integer multiplicity"
            ))
        }
    };
    let sign = |d: f64| {
        if d == 1.0 || d == -1.0 {
            Ok(d as i64)
        } else {
            Err(format!("{category} {style}: param `sign` = {d} is not ±1"))
        }
    };
    let energies = |keys: &[&str]| -> Result<Vec<Coeff>, String> {
        keys.iter()
            .map(|key| Ok(Coeff::Real(units.energy(params.get(key).unwrap_or(0.0))?)))
            .collect()
    };
    use Coeff::Real;
    match (category, style) {
        ("bond", "harmonic") => Ok(vec![
            Real(units.bond_k(need("k")?)?),
            Real(units.length(need("r0")?)?),
        ]),
        ("bond", "morse") => Ok(vec![
            Real(units.energy(need("d0")?)?),
            Real(units.inverse_length(need("alpha")?)?),
            Real(units.length(need("r0")?)?),
        ]),
        ("angle", "harmonic") => Ok(vec![
            Real(units.angle_k(need("k")?)?),
            Real(need("theta0")?),
        ]),
        // Harmonic plus Urey–Bradley: `K_ub` is a bond stiffness.
        ("angle", "charmm") => Ok(vec![
            Real(units.angle_k(need("k")?)?),
            Real(need("theta0")?),
            Real(units.bond_k(need("k_ub")?)?),
            Real(units.length(need("r_ub")?)?),
        ]),
        ("improper", "harmonic") => Ok(vec![Real(units.angle_k(need("k")?)?), Real(need("chi0")?)]),
        // E = K[1 + d·cos(nφ)]: `d` is the stored sign (±1), not a phase.
        ("improper", "cvff") => Ok(vec![
            Real(units.energy(need("k")?)?),
            Coeff::Int(sign(need("sign")?)?),
            multiplicity("periodicity", need("periodicity")?)?,
        ]),
        // AMBER `improper periodic`, E = K[1 + cos(nφ − φ0)], is LAMMPS
        // `improper_style cvff`, E = K[1 + d·cos(nφ)], when φ0 is 0° (d = +1) or
        // 180° (d = −1) — every GAFF improper. AMBER writes π as 3.1416, which a
        // reader in degrees stores as 180.0004, hence the tolerance (1e-3 rad).
        // Any other phase has no cvff form and is refused, not rounded.
        ("improper", "periodic") => {
            let phase = params.get("phase").unwrap_or(0.0).rem_euclid(360.0);
            let near = |x: f64| (phase - x).abs() < 1e-3_f64.to_degrees();
            let d = if near(0.0) || near(360.0) {
                1
            } else if near(180.0) {
                -1
            } else {
                return Err(format!(
                    "{category} {style}: phase {phase}° has no LAMMPS `cvff` form \
                     (needs 0° or 180°)"
                ));
            };
            Ok(vec![
                Real(units.energy(need("k")?)?),
                Coeff::Int(d),
                multiplicity("periodicity", need("periodicity")?)?,
            ])
        }
        ("dihedral", "opls") => energies(&["k1", "k2", "k3", "k4"]),
        ("dihedral", "harmonic") => Ok(vec![
            Real(units.energy(need("k")?)?),
            Coeff::Int(sign(need("sign")?)?),
            multiplicity("periodicity", need("periodicity")?)?,
        ]),
        // m  K1 n1 d1  [K2 n2 d2 ...] from `k<i>` / `periodicity<i>` / `phase<i>`:
        // molrs's `periodic` is LAMMPS's `fourier` term for term; its unindexed
        // `k` / `periodicity` / `phase` is the one-term case.
        ("dihedral", "periodic") => {
            let mut terms = Vec::new();
            if params.get("k1").is_none()
                && let Some(k) = params.get("k")
            {
                terms.push(Real(units.energy(k)?));
                terms.push(multiplicity("periodicity", need("periodicity")?)?);
                terms.push(Real(params.get("phase").unwrap_or(0.0)));
            }
            let mut i = 1usize;
            while let Some(k) = params.get(&format!("k{i}")) {
                let n_key = format!("periodicity{i}");
                terms.push(Real(units.energy(k)?));
                terms.push(multiplicity(&n_key, need(&n_key)?)?);
                terms.push(Real(params.get(&format!("phase{i}")).unwrap_or(0.0)));
                i += 1;
            }
            if terms.is_empty() {
                return Err(format!("{category} {style}: missing param `k1`"));
            }
            let mut fields = vec![Coeff::Int(terms.len() as i64 / 3)];
            fields.extend(terms);
            Ok(fields)
        }
        // E = K[1 + cos(nφ − d)]; `w` is the 1-4 pair weight. LAMMPS reads
        // `d` as an integer number of degrees (`dihedral_charmm.cpp`).
        ("dihedral", "charmm") => {
            let phase = params.get("phase").unwrap_or(0.0);
            if phase.fract() != 0.0 {
                return Err(format!(
                    "dihedral charmm: phase = {phase}° is not an integer number of \
                     degrees, which LAMMPS's dihedral_style charmm requires"
                ));
            }
            Ok(vec![
                Real(units.energy(need("k")?)?),
                multiplicity("periodicity", need("periodicity")?)?,
                Coeff::Int(phase as i64),
                Real(need("w")?),
            ])
        }
        ("dihedral", "multi/harmonic") => energies(&["a1", "a2", "a3", "a4", "a5"]),
        ("dihedral", "nharmonic") => {
            let a = nharmonic_coefficients(params)?;
            let mut fields = vec![Coeff::Int(a.len() as i64)];
            for v in a {
                fields.push(Real(units.energy(v)?));
            }
            Ok(fields)
        }
        ("pair", s) if s.starts_with("lj/cut") => Ok(vec![
            Real(units.energy(need("epsilon")?)?),
            Real(units.length(need("sigma")?)?),
        ]),
        // `epsilon sigma epsilon14 sigma14`, the 1-4 pair always written.
        ("pair", "lj/charmm") => {
            let (eps, sigma) = (need("epsilon")?, need("sigma")?);
            Ok(vec![
                Real(units.energy(eps)?),
                Real(units.length(sigma)?),
                Real(units.energy(params.get("epsilon14").unwrap_or(eps))?),
                Real(units.length(params.get("sigma14").unwrap_or(sigma))?),
            ])
        }
        _ => Err(format!(
            "unsupported LAMMPS {category} style `{style}` for coefficient output"
        )),
    }
}

/// A bonded type category the writer emits coefficients for.
trait BondedCoeff: Sized {
    /// Style category, and the LAMMPS command prefix (`bond_style`, `bond_coeff`).
    const CATEGORY: &'static str;
    /// The [`TypeLabels`] block holding this category's labels.
    const BLOCK: &'static str;
    /// Data-file section heading.
    const HEADING: &'static str;

    /// The types of this category `defs` holds, or `None` for another category.
    fn types_of(defs: &StyleDefs) -> Option<&[Self]>;

    /// The stored type name.
    fn name(&self) -> &str;

    /// The stored endpoint atom types, in slot order.
    fn endpoints(&self) -> Vec<&str>;

    /// The stored params, rendered by [`lammps_coeff_values`]'s conversion.
    fn params(&self) -> &Params;
}

impl BondedCoeff for BondType {
    const CATEGORY: &'static str = "bond";
    const BLOCK: &'static str = "bonds";
    const HEADING: &'static str = "Bond Coeffs";

    fn types_of(defs: &StyleDefs) -> Option<&[Self]> {
        match defs {
            StyleDefs::Bond(types) => Some(types),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn endpoints(&self) -> Vec<&str> {
        vec![self.itom.as_str(), self.jtom.as_str()]
    }

    fn params(&self) -> &Params {
        &self.params
    }
}

impl BondedCoeff for AngleType {
    const CATEGORY: &'static str = "angle";
    const BLOCK: &'static str = "angles";
    const HEADING: &'static str = "Angle Coeffs";

    fn types_of(defs: &StyleDefs) -> Option<&[Self]> {
        match defs {
            StyleDefs::Angle(types) => Some(types),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn endpoints(&self) -> Vec<&str> {
        vec![self.itom.as_str(), self.jtom.as_str(), self.ktom.as_str()]
    }

    fn params(&self) -> &Params {
        &self.params
    }
}

impl BondedCoeff for DihedralType {
    const CATEGORY: &'static str = "dihedral";
    const BLOCK: &'static str = "dihedrals";
    const HEADING: &'static str = "Dihedral Coeffs";

    fn types_of(defs: &StyleDefs) -> Option<&[Self]> {
        match defs {
            StyleDefs::Dihedral(types) => Some(types),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn endpoints(&self) -> Vec<&str> {
        vec![
            self.itom.as_str(),
            self.jtom.as_str(),
            self.ktom.as_str(),
            self.ltom.as_str(),
        ]
    }

    fn params(&self) -> &Params {
        &self.params
    }
}

impl BondedCoeff for ImproperType {
    const CATEGORY: &'static str = "improper";
    const BLOCK: &'static str = "impropers";
    const HEADING: &'static str = "Improper Coeffs";

    fn types_of(defs: &StyleDefs) -> Option<&[Self]> {
        match defs {
            StyleDefs::Improper(types) => Some(types),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn endpoints(&self) -> Vec<&str> {
        vec![
            self.itom.as_str(),
            self.jtom.as_str(),
            self.ktom.as_str(),
            self.ltom.as_str(),
        ]
    }

    fn params(&self) -> &Params {
        &self.params
    }
}

impl BondedCoeff for CmapType {
    const CATEGORY: &'static str = "cmap";
    const BLOCK: &'static str = "cmaps";
    const HEADING: &'static str = "CMAP";

    fn types_of(defs: &StyleDefs) -> Option<&[Self]> {
        match defs {
            StyleDefs::Cmap(types) => Some(types),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn endpoints(&self) -> Vec<&str> {
        vec![
            self.itom.as_str(),
            self.jtom.as_str(),
            self.ktom.as_str(),
            self.ltom.as_str(),
            self.mtom.as_str(),
        ]
    }

    fn params(&self) -> &Params {
        &self.params
    }
}

/// A bonded label resolved to the style and type that define it.
struct Resolved<'f, T> {
    /// 1-based type id (label order).
    id: usize,
    /// The label as written: the type's name.
    label: String,
    style: &'f Style,
    ty: &'f T,
}

impl<T: BondedCoeff> Resolved<'_, T> {
    /// The coefficients of the resolved type; an error names the label.
    fn coeffs(&self, units: &WriteUnits, precision: usize) -> Result<String, String> {
        coeff_fields(units, T::CATEGORY, self.style.name(), self.ty.params())
            .map(|fields| render_coeffs(&fields, precision))
            .map_err(|e| format!("{} label `{}`: {e}", T::BLOCK, self.label))
    }
}

/// A pair coefficient row: the ids of its two atom labels (lower first) and
/// the pair type that defines it.
struct PairRow<'f> {
    ids: (usize, usize),
    style: &'f Style,
    ty: &'f PairType,
}

impl PairRow<'_> {
    /// The pair coefficients in file units. A style with no `pair_coeff` form
    /// here (`thole`, `coul/tt`, `buck`, …) is an error naming the style and
    /// type: skipping it would leave the `pair_style` line without its
    /// `pair_coeff` rows. Type-less styles (`coul/cut`) produce no rows and
    /// never reach here.
    fn coeffs(&self, units: &WriteUnits, precision: usize) -> Result<String, String> {
        coeff_fields(units, "pair", self.style.name(), &self.ty.params)
            .map(|fields| render_coeffs(&fields, precision))
            .map_err(|e| format!("pair type `{}`: {e}", self.ty.name))
    }
}

/// Label-driven LAMMPS coefficient writer (AMBER/GAFF flavour).
///
/// Holds the system's [`TypeLabels`]; see the module docs for the matching
/// rules. [`ForceFieldWriter::write_str`] emits the `*.ff` include,
/// [`LammpsFfWriter::write_data_coeffs_str`] the data-file `* Coeffs`
/// sections; both write the same labels with the same numbers.
#[derive(Debug, Clone)]
pub struct LammpsFfWriter<'a> {
    labels: &'a TypeLabels,
    options: LammpsWriteOptions,
}

impl<'a> LammpsFfWriter<'a> {
    /// Writer for `labels` with default options (6 decimal places, `real`).
    pub fn new(labels: &'a TypeLabels) -> Self {
        Self::with_options(labels, LammpsWriteOptions::default())
    }

    /// Writer for `labels` with explicit options.
    pub fn with_options(labels: &'a TypeLabels, options: LammpsWriteOptions) -> Self {
        Self { labels, options }
    }

    /// Emit data-file `* Coeffs` sections only (no `units` / `*_style` lines).
    ///
    /// Rows are the labels' 1-based ids in [`TypeLabels`] order, with the same
    /// numbers, form map and [`LammpsWriteOptions::units`] as
    /// [`ForceFieldWriter::write_str`]. `Pair Coeffs` holds self pairs only, so
    /// a used explicit cross pair is an error here (write it through the
    /// include).
    pub fn write_data_coeffs_str(&self, ff: &ForceField) -> Result<String, String> {
        let units = WriteUnits::of(ff, self.options.units)?;
        let mut lines: Vec<String> = Vec::new();
        self.write_data_pair_coeffs(&mut lines, ff, &units)?;
        self.write_data_section::<BondType>(&mut lines, ff, &units)?;
        self.write_data_section::<AngleType>(&mut lines, ff, &units)?;
        self.write_data_section::<DihedralType>(&mut lines, ff, &units)?;
        self.write_data_section::<ImproperType>(&mut lines, ff, &units)?;
        Ok(lines.concat())
    }

    /// The LAMMPS `fix cmap` file of the `cmaps` labels: the grid of the
    /// `cmap charmm` row each label names, in label id order, so map `t` is
    /// the crossterm type `t` the data writer gives the `CMAP` section
    /// ([`lammps_cmap_str`] is the layout). Energies are converted to
    /// [`LammpsWriteOptions::units`] as every coefficient is.
    ///
    /// # Errors
    ///
    /// No `cmaps` label, a label with no cmap type, a style other than
    /// `charmm`, a row without a `grid`, a grid that is not 24×24, or more
    /// than six maps (LAMMPS's `CMAPDIM`, `CMAPMAX`).
    pub fn write_cmap_str(&self, ff: &ForceField) -> Result<String, String> {
        let units = WriteUnits::of(ff, self.options.units)?;
        let rows = self.resolve::<CmapType>(ff)?;
        if rows.is_empty() {
            return Err("cmaps: the system has no CMAP crossterm labels".into());
        }
        if rows.len() > LAMMPS_CMAP_MAX {
            return Err(format!(
                "cmaps: {} CMAP types, fix cmap reads at most {LAMMPS_CMAP_MAX}",
                rows.len()
            ));
        }
        let mut maps = Vec::with_capacity(rows.len());
        for r in &rows {
            let what = || format!("cmaps label `{}`", r.label);
            if r.style.name() != "charmm" {
                return Err(format!(
                    "{}: cmap style `{}` has no fix cmap form (only `charmm`)",
                    what(),
                    r.style.name()
                ));
            }
            let grid =
                r.ty.params
                    .get_array("grid")
                    .ok_or_else(|| format!("{}: no `grid`", what()))?;
            if grid.shape() != [LAMMPS_CMAP_DIM, LAMMPS_CMAP_DIM] {
                return Err(format!(
                    "{}: a {:?} grid; fix cmap reads {LAMMPS_CMAP_DIM}×{LAMMPS_CMAP_DIM}",
                    what(),
                    grid.shape()
                ));
            }
            let converted = grid
                .iter()
                .map(|&v| units.energy(v))
                .collect::<Result<Vec<f64>, String>>()?;
            let converted =
                ArrayD::from_shape_vec(grid.shape(), converted).map_err(|e| e.to_string())?;
            maps.push((r.label.clone(), converted));
        }
        let titled: Vec<(&str, &ArrayD<f64>)> =
            maps.iter().map(|(label, g)| (label.as_str(), g)).collect();
        lammps_cmap_str(&titled, self.options.units, self.options.precision)
    }

    /// The include's `fix cmap` lines when the system has CMAP crossterms,
    /// nothing otherwise.
    fn write_cmap_fix(&self, lines: &mut Vec<String>, ff: &ForceField) -> Result<(), String> {
        if self.resolve::<CmapType>(ff)?.is_empty() {
            return Ok(());
        }
        let file = self.options.cmap_file.as_deref().ok_or(
            "the system has CMAP crossterms: set LammpsWriteOptions::cmap_file to the file \
             write_cmap_str's text is saved as",
        )?;
        if file.is_empty() || file.chars().any(char::is_whitespace) {
            return Err(format!(
                "cmap_file {file:?}: LAMMPS reads one word as the fix cmap file"
            ));
        }
        lines.push(
            "# CMAP crossterms: this fix must precede `read_data <data> fix cmap crossterm CMAP`\n"
                .to_owned(),
        );
        lines.push(format!("fix {CMAP_FIX_ID} all cmap {file}\n"));
        lines.push(format!("fix_modify {CMAP_FIX_ID} energy yes\n"));
        lines.push("\n".to_owned());
        Ok(())
    }

    /// Labels of `block` in id order; the ids themselves (`"1"`, `"2"`, …)
    /// when the block's types carry no labels.
    fn block_labels(&self, block: &str) -> Vec<String> {
        match self.labels.block(block) {
            None => Vec::new(),
            Some(types) => match types.labels() {
                Some(labels) => labels.to_vec(),
                None => (1..=types.n_types()).map(|id| id.to_string()).collect(),
            },
        }
    }

    /// Every label of `T::BLOCK` resolved to its type, in id order. The first
    /// style (in force-field order) defining a name wins.
    fn resolve<'f, T: BondedCoeff>(
        &self,
        ff: &'f ForceField,
    ) -> Result<Vec<Resolved<'f, T>>, String> {
        let labels = self.block_labels(T::BLOCK);
        if labels.is_empty() {
            return Ok(Vec::new());
        }
        let mut by_name: HashMap<&'f str, (&'f Style, &'f T)> = HashMap::new();
        for style in ff.get_styles(T::CATEGORY) {
            for ty in T::types_of(style.defs()).unwrap_or_default() {
                by_name.entry(ty.name()).or_insert((style, ty));
            }
        }
        labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let &(style, ty) = by_name.get(label.as_str()).ok_or_else(|| {
                    let mut message = format!(
                        "{}: type label `{label}` has no {} type in the force field",
                        T::BLOCK,
                        T::CATEGORY
                    );
                    // A label matches a name exactly, orientation included; the
                    // likeliest cause is the other spelling, so name it when it
                    // is the one defined.
                    let other_way = by_name.values().find(|(_, ty)| {
                        let reversed: Vec<&str> = ty.endpoints().into_iter().rev().collect();
                        TypeName::join(&reversed).is_ok_and(|n| n.as_str() == label)
                    });
                    if let Some((_, defined)) = other_way {
                        message.push_str(&format!(
                            " (`{}` is defined on the same atom types read the other way; a \
                             label matches a type name exactly, orientation included)",
                            defined.name()
                        ));
                    }
                    message
                })?;
                Ok(Resolved {
                    id: i + 1,
                    label: label.clone(),
                    style,
                    ty,
                })
            })
            .collect()
    }

    /// Pair rows for the `atoms` labels, ordered by id pair: every label's
    /// self pair (missing → error) and every explicit cross pair of two used
    /// labels. The first style (in force-field order) defining a pair wins.
    ///
    /// A force field with no typed pair style (bonded-only, or only type-less
    /// styles such as Coulomb) has no pair rows to write, so it yields none
    /// rather than demanding a self pair per label.
    fn resolve_pairs<'f>(&self, ff: &'f ForceField) -> Result<Vec<PairRow<'f>>, String> {
        let typed = ff
            .get_styles("pair")
            .iter()
            .any(|s| matches!(s.defs(), StyleDefs::Pair(types) if !types.is_empty()));
        if !typed {
            return Ok(Vec::new());
        }
        let labels = self.block_labels("atoms");
        let ids: HashMap<&str, usize> = labels
            .iter()
            .enumerate()
            .map(|(i, label)| (label.as_str(), i + 1))
            .collect();
        let mut rows: BTreeMap<(usize, usize), PairRow<'f>> = BTreeMap::new();
        for style in ff.get_styles("pair") {
            let StyleDefs::Pair(types) = style.defs() else {
                continue;
            };
            for ty in types {
                let (Some(&i), Some(&j)) = (ids.get(ty.itom.as_str()), ids.get(ty.jtom.as_str()))
                else {
                    continue;
                };
                let key = (i.min(j), i.max(j));
                rows.entry(key).or_insert(PairRow {
                    ids: key,
                    style,
                    ty,
                });
            }
        }
        if let Some((_, label)) = labels
            .iter()
            .enumerate()
            .find(|(i, _)| !rows.contains_key(&(i + 1, i + 1)))
        {
            return Err(format!(
                "atoms: type label `{label}` has no pair type in the force field"
            ));
        }
        Ok(rows.into_values().collect())
    }

    fn write_pair_section(
        &self,
        lines: &mut Vec<String>,
        ff: &ForceField,
        units: &WriteUnits,
    ) -> Result<(), String> {
        let rows = self.resolve_pairs(ff)?;
        if rows.is_empty() {
            return Ok(());
        }
        // Styles in play: those defining a used pair, plus type-less ones
        // (Coulomb), which apply to every atom.
        let styles: Vec<&Style> = ff
            .get_styles("pair")
            .into_iter()
            .filter(|s| {
                matches!(s.defs(), StyleDefs::Pair(types) if types.is_empty())
                    || rows.iter().any(|r| std::ptr::eq(r.style, *s))
            })
            .collect();
        let opts = &self.options;
        for style in &styles {
            refuse_unexpressible_coulomb(style)?;
        }

        // lj/charmm + coul/charmm is LAMMPS's one `lj/charmm/coul/charmm`; it
        // has no other spelling (no `hybrid` sub-style is either half).
        if styles
            .iter()
            .any(|s| matches!(s.name(), "lj/charmm" | "coul/charmm"))
        {
            let names: HashSet<&str> = styles.iter().map(|s| s.name()).collect();
            if names != HashSet::from(["lj/charmm", "coul/charmm"]) {
                let mut names: Vec<&str> = names.into_iter().collect();
                names.sort_unstable();
                return Err(format!(
                    "pair styles {names:?}: LAMMPS has lj/charmm only as \
                     `lj/charmm/coul/charmm`, the pair lj/charmm + coul/charmm and nothing \
                     beside it"
                ));
            }
            if let Some(lj) = styles.iter().find(|s| s.name() == "lj/charmm")
                && OneFour::of(lj.params())? == OneFour::Epsilon14
            {
                return Err(
                    "pair lj/charmm has one_four = \"epsilon14\" (its special_bonds 1-4 pairs \
                     at epsilon14/sigma14), which LAMMPS's lj/charmm/coul/charmm prices at the \
                     regular epsilon/sigma; LAMMPS reaches epsilon14/sigma14 only through \
                     dihedral charmm w"
                        .into(),
                );
            }
            if !opts.skip_pair_style {
                let lj = styles.iter().find(|s| s.name() == "lj/charmm").unwrap();
                let coul = styles.iter().find(|s| s.name() == "coul/charmm").unwrap();
                let (lj_cuts, coul_cuts) =
                    (charmm_cutoffs(lj, units)?, charmm_cutoffs(coul, units)?);
                let mut cuts = lj_cuts.to_vec();
                if coul_cuts != lj_cuts {
                    cuts.extend(coul_cuts);
                }
                lines.push(format!(
                    "pair_style lj/charmm/coul/charmm {}\n",
                    format_nums(&cuts, opts.precision)
                ));
                lines.extend(pair_modify_line(lj));
                lines.push("\n".to_owned());
            }
            return self.push_pair_coeffs(lines, &rows, false, units);
        }

        // Reader always builds lj/cut + coul/cut; recombine for a correct write-back.
        if is_split_lj_coulomb(&styles) {
            if !opts.skip_pair_style {
                let lj = styles
                    .iter()
                    .find(|s| s.name() == "lj/cut")
                    .ok_or_else(|| "split pair styles missing lj/cut".to_owned())?;
                let coul = styles
                    .iter()
                    .find(|s| s.name() == "coul/cut" || s.name() == "coul/long")
                    .ok_or_else(|| "split pair styles missing coul/*".to_owned())?;
                let lj_cut = units.length(required_cutoff(lj)?)?;
                let coul_cut = units.length(required_cutoff(coul)?)?;
                lines.push(format!(
                    "pair_style lj/cut/coul/cut {} {}\n",
                    fmt_num(lj_cut, opts.precision),
                    fmt_num(coul_cut, opts.precision)
                ));
                lines.extend(pair_modify_line(lj));
                lines.push("\n".to_owned());
            }
            // Only LJ carries per-type ε/σ; Coulomb charges live on the atoms.
            return self.push_pair_coeffs(lines, &rows, false, units);
        }

        if let [style] = styles.as_slice() {
            if !opts.skip_pair_style {
                let params = pair_style_cutoffs(style, units)?;
                lines.push(format!(
                    "pair_style {} {}\n",
                    style.name(),
                    format_nums(&params, opts.precision)
                ));
                lines.extend(pair_modify_line(style));
                lines.push("\n".to_owned());
            }
            return self.push_pair_coeffs(lines, &rows, false, units);
        }

        // Genuinely independent sub-styles → hybrid with per-substyle cutoffs.
        if !opts.skip_pair_style {
            let mut sub = Vec::new();
            for s in &styles {
                let cuts = pair_style_cutoffs(s, units)?;
                if cuts.is_empty() {
                    sub.push(s.name().to_owned());
                } else {
                    sub.push(format!(
                        "{} {}",
                        s.name(),
                        format_nums(&cuts, opts.precision)
                    ));
                }
            }
            lines.push(format!("pair_style hybrid {}\n", sub.join(" ")));
            // A hybrid needs `pair_modify pair <substyle> mix <rule>` per sub-style;
            // no in-tree force field carries a non-default `mixing` on a hybrid, so
            // emitting it is deferred rather than guessed.
            lines.push("\n".to_owned());
        }
        self.push_pair_coeffs(lines, &rows, true, units)?;
        lines.push("\n".to_owned());
        Ok(())
    }

    /// `pair_coeff` lines for the rows, naming the sub-style when `hybrid`. A
    /// single-style block ends with a blank line when non-empty.
    fn push_pair_coeffs(
        &self,
        lines: &mut Vec<String>,
        rows: &[PairRow<'_>],
        hybrid: bool,
        units: &WriteUnits,
    ) -> Result<(), String> {
        for row in rows {
            let nums = row.coeffs(units, self.options.precision)?;
            let (i, j) = (&row.ty.itom, &row.ty.jtom);
            if hybrid {
                lines.push(format!("pair_coeff {i} {j} {} {nums}\n", row.style.name()));
            } else {
                lines.push(format!("pair_coeff {i} {j} {nums}\n"));
            }
        }
        if !rows.is_empty() && !hybrid {
            lines.push("\n".to_owned());
        }
        Ok(())
    }

    /// The `T_style` line and `T_coeff` lines for the used labels, in label id
    /// order. When the labels' types belong to more than one LAMMPS style the
    /// line is `T_style hybrid <sub-style>…` and each coefficient line names
    /// its sub-style, as LAMMPS reads them.
    fn write_section<T: BondedCoeff>(
        &self,
        lines: &mut Vec<String>,
        ff: &ForceField,
        units: &WriteUnits,
    ) -> Result<(), String> {
        let used = self.resolve::<T>(ff)?;
        if used.is_empty() {
            return Ok(());
        }
        let subs = lammps_styles_of(&used);
        let hybrid = subs.len() > 1;
        lines.push(format!("{}_style {}\n", T::CATEGORY, style_line(&subs)));
        for r in &used {
            lines.push(format!(
                "{}_coeff {} {}{}\n",
                T::CATEGORY,
                r.label,
                sub_style_field(hybrid, r),
                r.coeffs(units, self.options.precision)?
            ));
        }
        lines.push("\n".to_owned());
        Ok(())
    }

    fn write_data_pair_coeffs(
        &self,
        lines: &mut Vec<String>,
        ff: &ForceField,
        units: &WriteUnits,
    ) -> Result<(), String> {
        let mut section = Vec::new();
        for row in self.resolve_pairs(ff)? {
            if row.ids.0 != row.ids.1 {
                return Err(format!(
                    "pair type `{}` is an explicit cross pair; a data-file Pair Coeffs \
                     section holds self pairs only (write it through the *.ff include)",
                    row.ty.name
                ));
            }
            let nums = row.coeffs(units, self.options.precision)?;
            section.push(format!("{} {nums}\n", row.ids.0));
        }
        push_data_section(lines, "Pair Coeffs", section);
        Ok(())
    }

    fn write_data_section<T: BondedCoeff>(
        &self,
        lines: &mut Vec<String>,
        ff: &ForceField,
        units: &WriteUnits,
    ) -> Result<(), String> {
        let used = self.resolve::<T>(ff)?;
        let subs = lammps_styles_of(&used);
        let mut section = Vec::new();
        for r in &used {
            section.push(format!(
                "{} {}{}\n",
                r.id,
                sub_style_field(subs.len() > 1, r),
                r.coeffs(units, self.options.precision)?
            ));
        }
        // The `# style` comment is how `read_data_coeffs` (and LAMMPS's own
        // `write_data`) knows which style the numbers are; without it a reader
        // falls back to `harmonic`.
        let heading = format!("{} # {}", T::HEADING, style_hint(&subs));
        push_data_section(lines, &heading, section);
        Ok(())
    }
}

impl ForceFieldWriter for LammpsFfWriter<'_> {
    fn write_str(&self, ff: &ForceField) -> Result<String, String> {
        let units = WriteUnits::of(ff, self.options.units)?;
        let mut lines: Vec<String> = Vec::new();
        lines.push("# LAMMPS force field generated by molrs\n".to_owned());
        if !self.options.skip_units {
            lines.push(format!("units {}\n", self.options.units));
            lines.push("\n".to_owned());
        }
        self.write_cmap_fix(&mut lines, ff)?;
        // `special_bonds` is protocol (like `pair_style`), not a coefficient.
        // skip_pair_style means the include is coeff-only — the input script
        // owns 1-4 weights. Always emitting Amber 1/SCEE here is what made
        // PEO-Tg jobs run coul 1-4 = 0.8333 when the input asked for 0.5.
        if !self.options.skip_pair_style {
            let sb = ff.special_bonds();
            let p = self.options.precision;
            lines.push(format!(
                "special_bonds lj {} {} {} coul {} {} {}\n",
                fmt_num(sb.lj[0], p),
                fmt_num(sb.lj[1], p),
                fmt_num(sb.lj[2], p),
                fmt_num(sb.coul[0], p),
                fmt_num(sb.coul[1], p),
                fmt_num(sb.coul[2], p),
            ));
            lines.push("\n".to_owned());
        }

        self.write_pair_section(&mut lines, ff, &units)?;
        self.write_section::<BondType>(&mut lines, ff, &units)?;
        self.write_section::<AngleType>(&mut lines, ff, &units)?;
        self.write_section::<DihedralType>(&mut lines, ff, &units)?;
        self.write_section::<ImproperType>(&mut lines, ff, &units)?;

        Ok(lines.concat())
    }
}

/// The distinct LAMMPS styles of `used`, in first-use order. Two molrs styles
/// LAMMPS spells alike (`improper cvff` and `improper periodic`, both `cvff`)
/// are one LAMMPS style with one coefficient form.
fn lammps_styles_of<'f, T: BondedCoeff>(used: &[Resolved<'f, T>]) -> Vec<&'f str> {
    let mut subs: Vec<&str> = Vec::new();
    for r in used {
        let name = lammps_style_name(T::CATEGORY, r.style.name());
        if !subs.contains(&name) {
            subs.push(name);
        }
    }
    subs
}

/// The argument of a `*_style` line for `subs`: the one style, or `hybrid`
/// and every sub-style.
fn style_line(subs: &[&str]) -> String {
    match subs {
        [one] => (*one).to_owned(),
        _ => format!("hybrid {}", subs.join(" ")),
    }
}

/// The style a data-file `* Coeffs` header names: the one style, or `hybrid`.
fn style_hint<'s>(subs: &[&'s str]) -> &'s str {
    match subs {
        [one] => one,
        _ => "hybrid",
    }
}

/// The sub-style token (and its separating space) a hybrid coefficient line
/// carries before its numbers; empty otherwise.
fn sub_style_field<T: BondedCoeff>(hybrid: bool, r: &Resolved<'_, T>) -> String {
    if hybrid {
        format!("{} ", lammps_style_name(T::CATEGORY, r.style.name()))
    } else {
        String::new()
    }
}

/// `heading`, a blank line, the rows and a blank line; nothing when empty.
fn push_data_section(lines: &mut Vec<String>, heading: &str, rows: Vec<String>) {
    if rows.is_empty() {
        return;
    }
    lines.push(format!("{heading}\n\n"));
    lines.extend(rows);
    lines.push("\n".to_owned());
}

// ── pair styles ──────────────────────────────────────────────────────────────

/// `pair_modify mix <rule>`: the rule the style declares, or — for an `lj/cut`
/// that declares none — the rule the kernel evaluates it under,
/// [`Mixing::UNDECLARED`]. LAMMPS' own default for `lj/cut` is `geometric`, so
/// an export that stays silent mixes differently from molrs whenever the rule
/// is anything else, and the run silently uses the wrong cross terms.
fn pair_modify_line(style: &Style) -> Option<String> {
    let rule = match style.params().get_str("mixing") {
        Some(rule) => rule,
        None if style.name() == "lj/cut" => Mixing::UNDECLARED.name(),
        None => return None,
    };
    Some(format!("pair_modify mix {rule}\n"))
}

/// A `coul/cut` LAMMPS cannot price as molrs does: a buffer `delta ≠ 0`
/// (MMFF's `qᵢqⱼ/(r + δ)`; LAMMPS's Coulomb styles have none) or a
/// `dielectric ≠ 1` (an input-script `dielectric` command, not a coefficient).
/// The Coulomb constant is LAMMPS's own `qqr2e` for the `units` and is not
/// written; a field stating another (AMBER's 332.0522173) is priced by LAMMPS
/// at LAMMPS's, a documented difference of ~3e-5 relative.
fn refuse_unexpressible_coulomb(style: &Style) -> Result<(), String> {
    if style.name() == "coul/charmm" {
        if let Some(d) = style.params().get("dielectric").filter(|d| *d != 1.0) {
            return Err(format!(
                "pair coul/charmm: dielectric = {d} is a LAMMPS input-script `dielectric` \
                 command, not a coefficient this include can carry"
            ));
        }
        return Ok(());
    }
    if style.name() != "coul/cut" {
        return Ok(());
    }
    let p = style.params();
    if let Some(delta) = p.get("delta").filter(|d| *d != 0.0) {
        return Err(format!(
            "pair coul/cut: the buffer delta = {delta} (E = k·qq/(D·(r + delta))) has no \
             LAMMPS Coulomb style"
        ));
    }
    if let Some(d) = p.get("dielectric").filter(|d| *d != 1.0) {
        return Err(format!(
            "pair coul/cut: dielectric = {d} is a LAMMPS input-script `dielectric` \
             command, not a coefficient this include can carry"
        ));
    }
    Ok(())
}

fn is_split_lj_coulomb(styles: &[&Style]) -> bool {
    if styles.len() != 2 {
        return false;
    }
    let names: HashSet<&str> = styles.iter().map(|s| s.name()).collect();
    names == HashSet::from(["lj/cut", "coul/cut"])
        || names == HashSet::from(["lj/cut", "coul/long"])
}

fn pair_style_cutoffs(style: &Style, units: &WriteUnits) -> Result<Vec<f64>, String> {
    // Combined names want two cutoffs; simple kernels one, which they must carry.
    let convert = |c: f64| units.length(c);
    match style.name() {
        "lj/cut/coul/cut" | "lj/cut/coul/long" => {
            let c = convert(required_cutoff(style)?)?;
            Ok(vec![c, c])
        }
        "lj/cut" | "lj126" | "coul/cut" | "coul/long" => {
            Ok(vec![convert(required_cutoff(style)?)?])
        }
        _ => match style_cutoff(style) {
            Some(c) => Ok(vec![convert(c)?]),
            None => Ok(vec![]),
        },
    }
}

/// `(inner, cutoff)` of a CHARMM style, in file units; both required.
fn charmm_cutoffs(style: &Style, units: &WriteUnits) -> Result<[f64; 2], String> {
    let get = |key: &str| {
        style.params().get(key).ok_or_else(|| {
            format!(
                "pair style '{}' has no '{key}': lj/charmm/coul/charmm needs its inner and \
                 outer switching cutoffs",
                style.name()
            )
        })
    };
    Ok([units.length(get("inner")?)?, units.length(get("cutoff")?)?])
}

/// The per-pair override columns of `frame`'s `pairs` block, refused by name:
/// LAMMPS has no per-pair exception (see the conventions guide, "1-4
/// interactions"). The data-file writer refuses them too; a caller writing a
/// force field for a frame checks the frame here.
pub fn refuse_pair_overrides(frame: &molrs::store::frame::Frame) -> Result<(), String> {
    let Some(pairs) = frame.get("pairs") else {
        return Ok(());
    };
    let present: Vec<&str> = molrs::store::schema::PAIR_OVERRIDE_COLUMNS
        .iter()
        .copied()
        .filter(|k| pairs.get(k).is_some())
        .collect();
    if present.is_empty() {
        return Ok(());
    }
    Err(format!(
        "pairs: the per-pair override columns {present:?} have no LAMMPS form — LAMMPS \
         prices a 1-4 pair by special_bonds, lj/charmm's epsilon14/sigma14 and dihedral \
         charmm's w, never per pair"
    ))
}

fn style_cutoff(style: &Style) -> Option<f64> {
    style.params().get("cutoff")
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// A LAMMPS `fix cmap` file of `maps` (`(title, grid)`, each grid N×N,
/// φ-major), in CHARMM's layout: a first line with the `UNITS:` tag LAMMPS
/// checks, then per map a `# <title>, type <t>` comment and, per φ row, a
/// `# <φ>` comment over its N values, five a line, each `precision` decimals
/// wide as CHARMM's `%13.6f` is — so a map of six-decimal values (CHARMM's
/// own files) is written back as the very lines it was read from (less the
/// blank CHARMM ends a short line with).
///
/// [`read_lammps_cmap_str`](crate::ff::forcefield::readers::lammps::read_lammps_cmap_str)
/// reads it back; a value survives bit for bit once `precision` decimals
/// reach past its 17th significant digit.
///
/// # Errors
///
/// A grid that is not square, or a title holding a line break.
pub fn lammps_cmap_str(
    maps: &[(&str, &ArrayD<f64>)],
    units: &str,
    precision: usize,
) -> Result<String, String> {
    let width = precision + 7;
    let mut out = format!("# UNITS: {units} CMAP correction maps written by molrs\n");
    for (t, (title, grid)) in maps.iter().enumerate() {
        let n = match grid.shape() {
            [a, b] if a == b => *a,
            shape => return Err(format!("map `{title}`: a {shape:?} grid is not square")),
        };
        if title.contains(['\n', '\r']) {
            return Err(format!("map {}: its title holds a line break", t + 1));
        }
        out.push_str(&format!("\n# {title}, type {}\n", t + 1));
        let values: Vec<f64> = grid.iter().copied().collect();
        for (i, row) in values.chunks(n).enumerate() {
            let phi = -180.0 + 360.0 * i as f64 / n as f64;
            out.push_str(&format!("\n# {phi:.1}\n"));
            for line in row.chunks(5) {
                for (k, v) in line.iter().enumerate() {
                    if k > 0 {
                        out.push(' ');
                    }
                    out.push_str(&format!("{v:>width$.precision$}"));
                }
                out.push('\n');
            }
        }
    }
    Ok(out)
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
special_bonds amber
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

    /// Labels of a system using every type in [`MINI`].
    fn mini_labels() -> TypeLabels {
        labels_of(&[
            ("atoms", &["c3", "c3", "oh"]),
            ("bonds", &["c3-c3"]),
            ("angles", &["c3-c3-oh"]),
            ("dihedrals", &["c3-c3-oh-ho"]),
        ])
    }

    /// A bonded-only field has no pair rows: the writer used to demand a self
    /// pair for every atom label anyway, so such a field could not be written.
    #[test]
    fn a_bonded_only_field_writes_without_pair_rows() {
        let mut ff = ForceField::new("bonded");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "c3-c3",
                &["c3", "c3"],
                Params::from_pairs(&[("k", 457.78), ("r0", 1.5354)]),
            )
            .unwrap();
        let labels = labels_of(&[("atoms", &["c3", "c3"]), ("bonds", &["c3-c3"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert!(text.contains("bond_coeff c3-c3"), "{text}");
        assert!(!text.contains("pair_coeff"), "{text}");
    }

    #[test]
    fn writes_lammps_units_inverse_of_reader() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let labels = mini_labels();
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();

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

        // The identity: the reader stored K = 228.89, the writer writes it.
        assert!(
            text.contains("bond_coeff c3-c3 228.890000 1.535400"),
            "bond K:\n{text}"
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
        let labels = mini_labels();
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        let ff2 = LammpsFfReader::new().read_str(&text).unwrap();

        let bt = ff2
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("c3", "c3")
            .unwrap();
        assert_eq!(bt.params.get("k"), Some(228.89));
        assert!((bt.params.get("r0").unwrap() - 1.5354).abs() < 1e-9);

        let angle = ff2.get_style("angle", "harmonic").unwrap();
        let StyleDefs::Angle(atypes) = &angle.defs else {
            panic!("not angle");
        };
        let at = &atypes[0];
        assert_eq!(at.params.get("k"), Some(76.79));
        assert_eq!(at.params.get("theta0"), Some(109.66));

        let dih = ff2.get_style("dihedral", "periodic").unwrap();
        let StyleDefs::Dihedral(dtypes) = &dih.defs else {
            panic!("not dihedral");
        };
        let dt = &dtypes[0];
        assert!((dt.params.get("k1").unwrap() - 0.06).abs() < 1e-12);
        assert!((dt.params.get("periodicity1").unwrap() - 3.0).abs() < 1e-12);
        assert!((dt.params.get("phase1").unwrap() - 0.0).abs() < 1e-12);

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
        let labels = mini_labels();
        let text = LammpsFfWriter::with_options(&labels, opts)
            .write_str(&ff)
            .unwrap();
        assert!(!text.contains("pair_style"), "no pair_style:\n{text}");
        assert!(
            !text.contains("special_bonds"),
            "coeff include must not inject Amber 1-4:\n{text}"
        );
        assert!(text.contains("pair_coeff c3 c3"), "coeffs remain:\n{text}");
    }

    #[test]
    fn default_write_keeps_special_bonds() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let labels = mini_labels();
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert!(
            text.contains("special_bonds lj"),
            "full include declares 1-4:\n{text}"
        );
    }

    #[test]
    fn skip_units_omits_units_line() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let opts = LammpsWriteOptions {
            skip_units: true,
            skip_pair_style: true,
            ..Default::default()
        };
        let labels = mini_labels();
        let text = LammpsFfWriter::with_options(&labels, opts)
            .write_str(&ff)
            .unwrap();
        assert!(
            !text.contains("units "),
            "include after input units:\n{text}"
        );
        assert!(text.contains("bond_style harmonic"), "{text}");
        assert!(text.contains("bond_coeff c3-c3"), "{text}");
    }

    /// A dihedral label is its type's name: the reader defines `h1-c3-os-c3`
    /// and `c3-os-c3-h1` as two types, and the writer emits one row per label
    /// used, each under its own name.
    #[test]
    fn write_ff_writes_each_dihedral_label_under_its_own_name() {
        const SRC: &str = r#"
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 0.107800 3.397710
dihedral_style fourier
dihedral_coeff h1-c3-os-c3 1 0.337000 3 0.000000
dihedral_coeff c3-os-c3-h1 1 0.337000 3 0.000000
"#;
        let ff = LammpsFfReader::new().read_str(SRC).unwrap();
        let labels = labels_of(&[
            ("atoms", &["c3"]),
            ("dihedrals", &["h1-c3-os-c3", "c3-os-c3-h1"]),
        ]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "dihedral_coeff"),
            vec![
                "dihedral_coeff c3-os-c3-h1 1 0.337000 3 0.000000",
                "dihedral_coeff h1-c3-os-c3 1 0.337000 3 0.000000",
            ],
            "{text}"
        );
    }

    #[test]
    fn writes_units_real_line_by_default() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let labels = mini_labels();
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert!(text.contains("units real\n"), "default units real:\n{text}");
    }

    #[test]
    fn write_data_coeffs_matches_command_form_numbers() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let labels = mini_labels();
        let writer = LammpsFfWriter::new(&labels);
        let data = writer.write_data_coeffs_str(&ff).unwrap();
        assert!(
            !data.contains("pair_style") && !data.contains("units "),
            "sections only:\n{data}"
        );
        assert!(data.contains("Pair Coeffs\n"), "{data}");
        assert!(data.contains("Bond Coeffs # harmonic\n"), "{data}");
        assert!(
            data.contains("1 0.107800 3.397710") || data.contains("1 0.107800"),
            "pair row:\n{data}"
        );
        assert!(
            data.contains("1 228.890000 1.535400"),
            "bond K=k/2:\n{data}"
        );

        let cmd = writer.write_str(&ff).unwrap();
        // Same bond K appears in both layouts.
        assert!(cmd.contains("bond_coeff c3-c3 228.890000 1.535400"));
        assert!(data.contains("228.890000 1.535400"));
    }

    fn coeff_ids(text: &str, heading: &str) -> Vec<u32> {
        let Some(rest) = text.split(heading).nth(1) else {
            return vec![];
        };
        // Past the rest of the header line (its `# style` comment).
        rest.lines()
            .skip(1)
            .skip_while(|l| l.trim().is_empty())
            .take_while(|l| !l.trim().is_empty())
            .filter_map(|l| l.split_whitespace().next()?.parse().ok())
            .collect()
    }

    /// A label is its type's name, so labels in both orientations are two
    /// types: each block gets two ids and one coeff row per id (never two rows
    /// under one id — LAMMPS would read the extra line as an unknown
    /// identifier).
    #[test]
    fn write_data_coeffs_writes_one_row_per_label_in_either_orientation() {
        const SRC: &str = r#"
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 0.107800 3.397710

bond_style harmonic
bond_coeff c3-h1 340.000000 1.090000
bond_coeff h1-c3 340.000000 1.090000

angle_style harmonic
angle_coeff c3-c3-h1 37.500000 110.000000
angle_coeff h1-c3-c3 37.500000 110.000000

dihedral_style fourier
dihedral_coeff h1-c3-c3-os 2 0.250000 1 0.000000 0.000000 3 0.000000
dihedral_coeff os-c3-c3-h1 2 0.250000 1 0.000000 0.000000 3 0.000000
"#;
        let ff = LammpsFfReader::new().read_str(SRC).unwrap();
        let labels = labels_of(&[
            ("atoms", &["c3"]),
            ("bonds", &["c3-h1", "h1-c3"]),
            ("angles", &["c3-c3-h1", "h1-c3-c3"]),
            ("dihedrals", &["h1-c3-c3-os", "os-c3-c3-h1"]),
        ]);
        let data = LammpsFfWriter::new(&labels)
            .write_data_coeffs_str(&ff)
            .unwrap();
        assert_eq!(coeff_ids(&data, "Bond Coeffs"), vec![1, 2], "{data}");
        assert_eq!(coeff_ids(&data, "Angle Coeffs"), vec![1, 2], "{data}");
        assert_eq!(coeff_ids(&data, "Dihedral Coeffs"), vec![1, 2], "{data}");
    }

    #[test]
    fn write_data_coeffs_does_not_steal_atom_type_id_from_hyphen_head() {
        const SRC: &str = r#"
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 0.107800 3.397710
"#;
        // The dihedral label's head `c3` is an atom label with a pair type;
        // the dihedral must still resolve against dihedral types only.
        let ff = LammpsFfReader::new().read_str(SRC).unwrap();
        let labels = labels_of(&[("atoms", &["c3"]), ("dihedrals", &["c3-os-c3-h1"])]);
        let err = LammpsFfWriter::new(&labels)
            .write_data_coeffs_str(&ff)
            .unwrap_err();
        assert!(
            err.contains("c3-os-c3-h1"),
            "unmapped dihedral must not resolve as atom type c3: {err}"
        );
    }

    /// A `real` force field written as `metal` has its energies converted
    /// through the lj hub; reading the `metal` file back gives a `metal`
    /// force field, and writing that back as `metal` is the identity.
    #[test]
    fn metal_write_converts_energy_via_lj_hub() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        assert_eq!(ff.units(), "real");
        let opts = || LammpsWriteOptions {
            units: "metal",
            ..Default::default()
        };
        let labels = mini_labels();
        let text = LammpsFfWriter::with_options(&labels, opts())
            .write_str(&ff)
            .unwrap();
        assert!(text.contains("units metal\n"), "metal header:\n{text}");

        // 0.1078 kcal/mol → eV through lj hub
        let sys = crate::ff::forcefield::lammps_units::LammpsFfUnits::canonical().unwrap();
        let eps_ev = sys.energy(0.1078, "real", "metal").unwrap();
        let expected = format!("pair_coeff c3 c3 {:.6}", eps_ev);
        assert!(text.contains(&expected), "expected {expected} in:\n{text}");

        // Length unchanged (Å in both real and metal).
        assert!(
            text.contains(&format!("{:.6}", 3.39771)),
            "sigma stays Å:\n{text}"
        );

        // The metal file reads as a metal force field, number for number, and
        // writes back unchanged.
        let ff2 = LammpsFfReader::new().read_str(&text).unwrap();
        assert_eq!(ff2.units(), "metal");
        let again = LammpsFfWriter::with_options(&labels, opts())
            .write_str(&ff2)
            .unwrap();
        assert_eq!(again, text);
    }

    #[test]
    fn atom_type_filter_restricts_pair_coeffs() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        // Only `c3` is used; `oh` is a ForceField type nobody labels.
        let labels = labels_of(&[("atoms", &["c3"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert!(text.contains("pair_coeff c3 c3"));
        assert!(
            !text.contains("pair_coeff oh oh"),
            "oh filtered out:\n{text}"
        );
    }

    // ------------------------------------------------------------------
    // Label-driven writing (system-forcefield-06): a hand-built ForceField
    // plus TypeLabels from a hand-written Frame, one stage per test.
    // ------------------------------------------------------------------

    use crate::core::store::block::Block;
    use crate::core::store::frame::Frame;
    use crate::core::store::keys;
    use crate::core::store::type_labels::TypeLabels;
    use ndarray::{ArrayD, IxDyn};

    /// A block whose only column is the string `type` label per row.
    fn label_type_block(types: &[&str]) -> Block {
        let mut block = Block::new();
        block
            .insert(
                keys::TYPE,
                ArrayD::from_shape_vec(
                    IxDyn(&[types.len()]),
                    types.iter().map(|t| t.to_string()).collect::<Vec<String>>(),
                )
                .unwrap(),
            )
            .unwrap();
        block
    }

    /// `TypeLabels` of a Frame holding one labelled block per `(name, types)`.
    fn labels_of(blocks: &[(&str, &[&str])]) -> TypeLabels {
        let mut frame = Frame::new();
        for (name, types) in blocks {
            frame.insert(*name, label_type_block(types));
        }
        TypeLabels::from_frame(&frame).unwrap()
    }

    fn lj(eps: f64, sigma: f64) -> Params {
        Params::from_pairs(&[("epsilon", eps), ("sigma", sigma)])
    }

    /// Harmonic bond: molrs's `k` is LAMMPS's `K`.
    fn bond(k: f64, r0: f64) -> Params {
        Params::from_pairs(&[("k", k), ("r0", r0)])
    }

    /// Split `lj/cut` (cutoff 9) + `coul/cut` (cutoff 10) with hand-written
    /// ε/σ for `c3`, `hc` and `oh`, and harmonic bonds `c3-hc`, `c3-oh`.
    fn split_pair_ff() -> ForceField {
        let mut ff = ForceField::new("hand");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 9.0)]))
            .unwrap()
            .def_type("c3", &["c3"], lj(0.1078, 3.39771))
            .unwrap()
            .def_type("hc", &["hc"], lj(0.0157, 2.64953))
            .unwrap()
            .def_type("oh", &["oh"], lj(0.093, 3.242871))
            .unwrap();
        ff.def_style("pair", "coul/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-hc", &["c3", "hc"], bond(340.0, 1.09))
            .unwrap()
            .def_type("c3-oh", &["c3", "oh"], bond(320.0, 1.41))
            .unwrap();
        ff
    }

    /// Rows of a data-file `heading` section (between its blank lines), past
    /// the header line and its `# style` comment.
    fn data_section_rows(text: &str, heading: &str) -> Vec<String> {
        let Some(rest) = text.split(heading).nth(1) else {
            return vec![];
        };
        rest.lines()
            .skip(1)
            .skip_while(|l| l.trim().is_empty())
            .take_while(|l| !l.trim().is_empty())
            .map(str::to_owned)
            .collect()
    }

    fn lines_starting_with(text: &str, prefix: &str) -> Vec<String> {
        text.lines()
            .filter(|l| l.starts_with(prefix))
            .map(str::to_owned)
            .collect()
    }

    /// `lj/cut` (cutoff 9) with `c3` / `hc` self rows and, when given, the
    /// style-level string param `mixing`.
    fn lj_only_ff(mixing: Option<&str>) -> ForceField {
        let mut params = Params::from_pairs(&[("cutoff", 9.0)]);
        if let Some(rule) = mixing {
            params.set_str("mixing", rule);
        }
        let mut ff = ForceField::new("hand");
        ff.def_style("pair", "lj/cut", params)
            .unwrap()
            .def_type("c3", &["c3"], lj(0.1078, 3.39771))
            .unwrap()
            .def_type("hc", &["hc"], lj(0.0157, 2.64953))
            .unwrap();
        ff
    }

    /// [`lj_only_ff`] plus a type-less `coul/cut` (cutoff 10): the split pair
    /// the writer recombines into `lj/cut/coul/cut`.
    fn split_pair_ff_mixing(rule: &str) -> ForceField {
        let mut ff = lj_only_ff(Some(rule));
        ff.def_style("pair", "coul/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        ff
    }

    /// Combined branch (`lj/cut` + `coul/cut` → `lj/cut/coul/cut`): an
    /// `lj/cut` declaring no rule is evaluated arithmetically by the kernel,
    /// so the export must say so — LAMMPS' own default is geometric.
    #[test]
    fn combined_undeclared_lj_cut_writes_pair_modify_mix_arithmetic() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "pair_modify"),
            vec!["pair_modify mix arithmetic".to_owned()],
            "{text}"
        );
    }

    /// Single-style branch: the same rule for a lone undeclared `lj/cut`.
    #[test]
    fn single_undeclared_lj_cut_writes_pair_modify_mix_arithmetic() {
        let ff = lj_only_ff(None);
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "pair_modify"),
            vec!["pair_modify mix arithmetic".to_owned()],
            "{text}"
        );
    }

    /// A declared rule is written as declared, in both branches.
    #[test]
    fn declared_geometric_lj_cut_writes_pair_modify_mix_geometric() {
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        for ff in [
            split_pair_ff_mixing("geometric"),
            lj_only_ff(Some("geometric")),
        ] {
            let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
            assert_eq!(
                lines_starting_with(&text, "pair_modify"),
                vec!["pair_modify mix geometric".to_owned()],
                "{text}"
            );
        }
    }

    #[test]
    fn label_writer_skips_unused_forcefield_types_in_include() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc", "hc"]), ("bonds", &["c3-hc"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert!(!text.contains("pair_coeff oh"), "unused oh:\n{text}");
        assert!(!text.contains("c3-oh"), "unused c3-oh:\n{text}");
        assert!(text.contains("pair_coeff c3 c3"), "{text}");
        assert!(text.contains("bond_coeff c3-hc"), "{text}");
    }

    #[test]
    fn label_writer_skips_unused_forcefield_types_in_data_coeffs() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc", "hc"]), ("bonds", &["c3-hc"])]);
        let data = LammpsFfWriter::new(&labels)
            .write_data_coeffs_str(&ff)
            .unwrap();
        assert_eq!(
            data_section_rows(&data, "Pair Coeffs"),
            vec!["1 0.107800 3.397710", "2 0.015700 2.649530"],
            "{data}"
        );
        assert_eq!(
            data_section_rows(&data, "Bond Coeffs"),
            vec!["1 340.000000 1.090000"],
            "{data}"
        );
    }

    #[test]
    fn label_writer_missing_bond_label_is_err_naming_block_and_label() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc"]), ("bonds", &["c3-n"])]);
        let err = LammpsFfWriter::new(&labels).write_str(&ff).unwrap_err();
        assert!(err.contains("bonds"), "names the block: {err}");
        assert!(err.contains("c3-n"), "names the label: {err}");
    }

    #[test]
    fn label_writer_missing_atom_label_in_data_coeffs_is_err_naming_block_and_label() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "os"])]);
        let err = LammpsFfWriter::new(&labels)
            .write_data_coeffs_str(&ff)
            .unwrap_err();
        assert!(err.contains("atoms"), "names the block: {err}");
        assert!(err.contains("os"), "names the label: {err}");
    }

    /// Improper `k` of the kernel `k·(χ − χ₀)²` (LAMMPS's `K`), `chi0` in
    /// degrees.
    fn improper_ff() -> ForceField {
        let mut ff = ForceField::new("hand");
        ff.def_style("improper", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "c3-n-c-o",
                &["c3", "n", "c", "o"],
                Params::from_pairs(&[("k", 2.2), ("chi0", 180.0)]),
            )
            .unwrap();
        ff
    }

    #[test]
    fn label_writer_matches_improper_label_exactly() {
        let ff = improper_ff();
        let labels = labels_of(&[("impropers", &["c3-n-c-o"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "improper_coeff"),
            vec!["improper_coeff c3-n-c-o 2.200000 180.000000"],
            "{text}"
        );
    }

    #[test]
    fn label_writer_does_not_resolve_reversed_improper_label() {
        let ff = improper_ff();
        let labels = labels_of(&[("impropers", &["o-c-n-c3"])]);
        let err = LammpsFfWriter::new(&labels).write_str(&ff).unwrap_err();
        assert!(err.contains("impropers"), "names the block: {err}");
        assert!(err.contains("o-c-n-c3"), "names the label: {err}");
    }

    /// AMBER `improper periodic`: `k` kcal/mol, integer `periodicity`, `phase` deg.
    fn periodic_improper_ff(phase: f64) -> ForceField {
        let mut ff = ForceField::new("hand");
        ff.def_style("improper", "periodic", Params::new())
            .unwrap()
            .def_type(
                "c3-o-c-os",
                &["c3", "o", "c", "os"],
                Params::from_pairs(&[("k", 1.1), ("periodicity", 2.0), ("phase", phase)]),
            )
            .unwrap();
        ff
    }

    #[test]
    fn periodic_improper_is_written_as_cvff() {
        // AMBER stores π rounded to 3.1416 in a prmtop, 7.3e-6 rad above the
        // true value, which a prmtop reader stores as 180.0004°.
        let ff = periodic_improper_ff((std::f64::consts::PI + 7.3e-6).to_degrees());
        let labels = labels_of(&[("impropers", &["c3-o-c-os"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "improper_style"),
            vec!["improper_style cvff"],
            "{text}"
        );
        assert_eq!(
            lines_starting_with(&text, "improper_coeff"),
            vec!["improper_coeff c3-o-c-os 1.100000 -1 2"],
            "{text}"
        );
    }

    #[test]
    fn periodic_improper_phase_sets_the_cvff_sign() {
        let zero = Params::from_pairs(&[("k", 1.1), ("periodicity", 2.0), ("phase", 0.0)]);
        assert_eq!(
            lammps_coeff_values("improper", "periodic", &zero, "real").unwrap(),
            [1.1, 1.0, 2.0]
        );
        let pi = Params::from_pairs(&[("k", 1.1), ("periodicity", 2.0), ("phase", 180.0)]);
        assert_eq!(
            lammps_coeff_values("improper", "periodic", &pi, "real").unwrap(),
            [1.1, -1.0, 2.0]
        );
    }

    #[test]
    fn periodic_improper_with_other_phase_is_refused() {
        let p = Params::from_pairs(&[("k", 1.1), ("periodicity", 2.0), ("phase", 57.3)]);
        let err = lammps_coeff_values("improper", "periodic", &p, "real").unwrap_err();
        assert!(err.contains("cvff"), "{err}");
    }

    #[test]
    fn label_data_coeff_ids_follow_type_labels_not_forcefield_order() {
        // ForceField order: oh, hc, c3 / c3-oh, c3-hc. Label order (sorted):
        // c3=1, hc=2, oh=3 / c3-hc=1, c3-oh=2.
        let mut ff = ForceField::new("hand");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 9.0)]))
            .unwrap()
            .def_type("oh", &["oh"], lj(0.093, 3.242871))
            .unwrap()
            .def_type("hc", &["hc"], lj(0.0157, 2.64953))
            .unwrap()
            .def_type("c3", &["c3"], lj(0.1078, 3.39771))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-oh", &["c3", "oh"], bond(320.0, 1.41))
            .unwrap()
            .def_type("c3-hc", &["c3", "hc"], bond(340.0, 1.09))
            .unwrap();
        let labels = labels_of(&[
            ("atoms", &["hc", "oh", "c3", "hc"]),
            ("bonds", &["c3-oh", "c3-hc"]),
        ]);
        let data = LammpsFfWriter::new(&labels)
            .write_data_coeffs_str(&ff)
            .unwrap();
        assert_eq!(
            data_section_rows(&data, "Pair Coeffs"),
            vec![
                "1 0.107800 3.397710",
                "2 0.015700 2.649530",
                "3 0.093000 3.242871"
            ],
            "{data}"
        );
        assert_eq!(
            data_section_rows(&data, "Bond Coeffs"),
            vec!["1 340.000000 1.090000", "2 320.000000 1.410000"],
            "{data}"
        );
    }

    #[test]
    fn label_writer_combines_split_lj_coul_into_one_pair_style_line() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "pair_style"),
            vec!["pair_style lj/cut/coul/cut 9.000000 10.000000"],
            "{text}"
        );
    }

    #[test]
    fn a_pair_style_without_a_cutoff_is_refused_not_defaulted() {
        let mut ff = ForceField::new("hand");
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type("c3", &["c3"], lj(0.1078, 3.39771))
            .unwrap();
        ff.def_style("pair", "coul/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        let labels = labels_of(&[("atoms", &["c3"])]);
        let err = LammpsFfWriter::new(&labels).write_str(&ff).unwrap_err();
        assert!(err.contains("'lj/cut' has no cutoff"), "{err}");
    }

    #[test]
    fn label_writer_pair_coeff_lines_follow_hand_written_eps_sigma() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["hc", "c3"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "pair_coeff"),
            vec![
                "pair_coeff c3 c3 0.107800 3.397710",
                "pair_coeff hc hc 0.015700 2.649530"
            ],
            "{text}"
        );
    }

    #[test]
    fn label_writer_writes_cross_pair_only_when_both_endpoints_used() {
        let mut ff = split_pair_ff();
        ff.get_style_mut("pair", "lj/cut")
            .unwrap()
            .def_type("c3-hc", &["c3", "hc"], lj(0.05, 3.0))
            .unwrap()
            .def_type("c3-oh", &["c3", "oh"], lj(0.07, 3.3))
            .unwrap();
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert!(
            text.contains("pair_coeff c3 hc 0.050000 3.000000\n"),
            "{text}"
        );
        assert!(!text.contains("pair_coeff c3 oh"), "{text}");
    }

    #[test]
    fn label_writer_skip_pair_style_omits_pair_style_and_special_bonds() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let opts = LammpsWriteOptions {
            skip_pair_style: true,
            ..Default::default()
        };
        let text = LammpsFfWriter::with_options(&labels, opts)
            .write_str(&ff)
            .unwrap();
        assert!(!text.contains("pair_style"), "{text}");
        assert!(!text.contains("special_bonds"), "{text}");
        assert!(
            text.contains("pair_coeff c3 c3 0.107800 3.397710"),
            "{text}"
        );
    }

    /// Harmonic `c3-hc` plus an unsupported `fene` style holding the bond
    /// `c3-oh` on `c3`, `oh`.
    fn ff_with_fene() -> ForceField {
        let mut ff = ForceField::new("hand");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-hc", &["c3", "hc"], bond(340.0, 1.09))
            .unwrap();
        ff.def_style("bond", "fene", Params::new())
            .unwrap()
            .def_type(
                "c3-oh",
                &["c3", "oh"],
                Params::from_pairs(&[("k", 30.0), ("r0", 1.5), ("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap();
        ff
    }

    #[test]
    fn label_writer_tolerates_unsupported_style_holding_only_unused_types() {
        let ff = ff_with_fene();
        let labels = labels_of(&[("bonds", &["c3-hc"])]);
        let writer = LammpsFfWriter::new(&labels);
        let text = writer.write_str(&ff).unwrap();
        assert!(!text.contains("fene"), "{text}");
        assert!(text.contains("bond_coeff c3-hc"), "{text}");
        let data = writer.write_data_coeffs_str(&ff).unwrap();
        assert_eq!(
            data_section_rows(&data, "Bond Coeffs"),
            vec!["1 340.000000 1.090000"],
            "{data}"
        );
    }

    /// An orientation mismatch reads differently from a missing type: the
    /// error names the spelling that is defined.
    #[test]
    fn label_writer_names_the_reversed_spelling_that_is_defined() {
        let ff = ff_with_fene();
        let labels = labels_of(&[("bonds", &["hc-c3"])]);
        let err = LammpsFfWriter::new(&labels).write_str(&ff).unwrap_err();
        assert!(err.contains("`hc-c3` has no bond type"), "{err}");
        assert!(err.contains("`c3-hc` is defined"), "{err}");
    }

    #[test]
    fn label_writer_rejects_unsupported_style_holding_a_used_type() {
        let ff = ff_with_fene();
        let labels = labels_of(&[("bonds", &["c3-hc", "c3-oh"])]);
        let writer = LammpsFfWriter::new(&labels);
        let err = writer.write_str(&ff).unwrap_err();
        assert!(err.contains("fene"), "names the style: {err}");
        let err = writer.write_data_coeffs_str(&ff).unwrap_err();
        assert!(err.contains("fene"), "names the style: {err}");
    }

    /// A label is matched to a type name exactly. The force field defines the
    /// bond `c3-h1` (K = 340) and the qualified angle `C_3-C_R-O_2@1_1.5_2`
    /// (K = 60, theta0 = 120 degrees);
    /// labels spelled the same are written, and the reversed spellings
    /// `h1-c3` / `O_2-C_R-C_3@1.5_1_2` find no type — an error naming the
    /// block and the label.
    #[test]
    fn label_writer_matches_labels_to_type_names_exactly() {
        let mut ff = ForceField::new("hand");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-h1", &["c3", "h1"], bond(340.0, 1.09))
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "C_3-C_R-O_2@1_1.5_2",
                &["C_3", "C_R", "O_2"],
                Params::from_pairs(&[("k", 60.0), ("theta0", 120.0)]),
            )
            .unwrap();

        let exact = labels_of(&[("bonds", &["c3-h1"]), ("angles", &["C_3-C_R-O_2@1_1.5_2"])]);
        let text = LammpsFfWriter::new(&exact)
            .write_str(&ff)
            .expect("exact labels resolve");
        assert_eq!(
            lines_starting_with(&text, "bond_coeff"),
            vec!["bond_coeff c3-h1 340.000000 1.090000"],
            "{text}"
        );
        assert_eq!(
            lines_starting_with(&text, "angle_coeff"),
            vec!["angle_coeff C_3-C_R-O_2@1_1.5_2 60.000000 120.000000"],
            "{text}"
        );

        let reversed_bond = labels_of(&[("bonds", &["h1-c3"])]);
        let err = LammpsFfWriter::new(&reversed_bond)
            .write_str(&ff)
            .unwrap_err();
        assert!(err.contains("bonds") && err.contains("h1-c3"), "{err}");

        let reversed_angle = labels_of(&[("angles", &["O_2-C_R-C_3@1.5_1_2"])]);
        let err = LammpsFfWriter::new(&reversed_angle)
            .write_str(&ff)
            .unwrap_err();
        assert!(
            err.contains("angles") && err.contains("O_2-C_R-C_3@1.5_1_2"),
            "{err}"
        );
    }

    /// MMFF's buffered Coulomb (`delta = 0.05`) has no LAMMPS style: refused.
    #[test]
    fn a_buffered_coulomb_is_refused() {
        let mut ff = split_pair_ff();
        ff.get_style_mut("pair", "coul/cut")
            .unwrap()
            .set_param("delta", 0.05);
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let err = LammpsFfWriter::new(&labels).write_str(&ff).unwrap_err();
        assert!(err.contains("delta"), "{err}");
    }

    /// A pair style whose types carry no ε/σ (`thole`: per-type `charge`,
    /// `alpha`, `damp`) has no `pair_coeff` form here. Writing the
    /// `pair_style` line without its coefficients is an incomplete include, so
    /// both writers refuse and name the category and the style.
    #[test]
    fn pair_style_without_writable_coeffs_is_err_naming_style() {
        let mut ff = ForceField::new("hand");
        ff.def_style("pair", "thole", Params::from_pairs(&[("cutoff", 12.0)]))
            .unwrap()
            .def_type(
                "c3",
                &["c3"],
                Params::from_pairs(&[("charge", -0.2), ("alpha", 1.1), ("damp", 2.6)]),
            )
            .unwrap();
        let labels = labels_of(&[("atoms", &["c3"])]);
        let writer = LammpsFfWriter::new(&labels);

        for err in [
            writer
                .write_str(&ff)
                .expect_err("thole has no pair_coeff form"),
            writer
                .write_data_coeffs_str(&ff)
                .expect_err("thole has no Pair Coeffs form"),
        ] {
            assert!(err.contains("pair") && err.contains("thole"), "{err}");
        }
    }

    fn assert_values(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{got:?} != {want:?}");
        for (g, w) in got.iter().zip(want) {
            let tol = 1e-12 * w.abs().max(1.0);
            assert!((g - w).abs() <= tol, "{got:?} != {want:?}");
        }
    }

    /// `(category, style, stored params, expected LAMMPS values)`.
    type ValuesCase<'a> = (&'a str, &'a str, &'a [(&'a str, f64)], &'a [f64]);

    /// Hand-derived goldens for every kernel, in `real`: the identity on
    /// values, slot for slot — LAMMPS's `K`, degrees, energies as stored.
    #[test]
    fn lammps_coeff_values_renders_each_kernel() {
        let cases: &[ValuesCase<'_>] = &[
            (
                "bond",
                "harmonic",
                &[("k", 450.0), ("r0", 0.9572)],
                &[450.0, 0.9572],
            ),
            (
                "bond",
                "morse",
                &[("d0", 95.6), ("alpha", 2.0), ("r0", 1.53)],
                &[95.6, 2.0, 1.53],
            ),
            (
                "angle",
                "harmonic",
                &[("k", 55.0), ("theta0", 104.52)],
                &[55.0, 104.52],
            ),
            (
                "improper",
                "harmonic",
                &[("k", 10.0), ("chi0", 180.0)],
                &[10.0, 180.0],
            ),
            (
                "improper",
                "cvff",
                &[("k", 1.1), ("sign", -1.0), ("periodicity", 2.0)],
                &[1.1, -1.0, 2.0],
            ),
            (
                "dihedral",
                "opls",
                &[("k1", 1.0), ("k2", 2.0), ("k3", 3.0), ("k4", 4.0)],
                &[1.0, 2.0, 3.0, 4.0],
            ),
            (
                "dihedral",
                "harmonic",
                &[("k", 2.0), ("sign", -1.0), ("periodicity", 3.0)],
                &[2.0, -1.0, 3.0],
            ),
            (
                "dihedral",
                "periodic",
                &[
                    ("k1", 0.5),
                    ("periodicity1", 1.0),
                    ("phase1", 180.0),
                    ("k2", 0.25),
                    ("periodicity2", 3.0),
                    ("phase2", 0.0),
                ],
                &[2.0, 0.5, 1.0, 180.0, 0.25, 3.0, 0.0],
            ),
            (
                "dihedral",
                "charmm",
                &[
                    ("k", 0.2),
                    ("periodicity", 3.0),
                    ("phase", 180.0),
                    ("w", 0.5),
                ],
                &[0.2, 3.0, 180.0, 0.5],
            ),
            (
                "dihedral",
                "multi/harmonic",
                &[
                    ("a1", 1.0),
                    ("a2", 2.0),
                    ("a3", 3.0),
                    ("a4", 4.0),
                    ("a5", 5.0),
                ],
                &[1.0, 2.0, 3.0, 4.0, 5.0],
            ),
            (
                "pair",
                "lj/cut",
                &[("epsilon", 0.066), ("sigma", 3.5)],
                &[0.066, 3.5],
            ),
        ];
        for (category, style, params, want) in cases {
            let got = lammps_coeff_values(category, style, &Params::from_pairs(params), "real")
                .unwrap_or_else(|e| panic!("{category} {style}: {e}"));
            assert_values(&got, want);
        }
    }

    #[test]
    fn lammps_coeff_values_rejects_unsupported_kernel_and_missing_param() {
        let err = lammps_coeff_values("bond", "fene", &Params::from_pairs(&[("k", 1.0)]), "real")
            .unwrap_err();
        assert!(err.contains("bond") && err.contains("fene"), "{err}");

        let err = lammps_coeff_values(
            "bond",
            "harmonic",
            &Params::from_pairs(&[("k", 450.0)]),
            "real",
        )
        .unwrap_err();
        assert!(err.contains("r0"), "{err}");

        let bond = Params::from_pairs(&[("k", 450.0), ("r0", 0.9572)]);
        let err = lammps_coeff_values("bond", "harmonic", &bond, "si").unwrap_err();
        assert!(err.contains("si"), "{err}");
    }

    /// `dihedral periodic` is molrs's name for LAMMPS `dihedral_style
    /// fourier`, term for term; the unindexed one-term spelling the kernel
    /// accepts is written as one fourier term.
    #[test]
    fn periodic_dihedral_is_written_as_fourier() {
        assert_eq!(lammps_style_name("dihedral", "periodic"), "fourier");
        let single = Params::from_pairs(&[("k", 0.3), ("periodicity", 2.0), ("phase", 180.0)]);
        assert_eq!(
            lammps_coeff_values("dihedral", "periodic", &single, "real").unwrap(),
            vec![1.0, 0.3, 2.0, 180.0]
        );
    }

    /// Writer and reader are inverse through their two public homes: the
    /// values rendered for a kernel read back, as tokens, to the same params,
    /// and the tokens come back as written.
    #[test]
    fn lammps_coeff_values_round_trips_through_lammps_coeff_params() {
        use crate::ff::forcefield::readers::lammps::lammps_coeff_params;
        // (category, LAMMPS style, molrs style, tokens)
        let cases: &[(&str, &str, &str, &[&str])] = &[
            ("bond", "harmonic", "harmonic", &["450", "0.9572"]),
            ("bond", "morse", "morse", &["95.6", "2", "1.53"]),
            ("angle", "harmonic", "harmonic", &["55", "104.52"]),
            (
                "angle",
                "charmm",
                "charmm",
                &["33.43", "110.1", "22.53", "2.179"],
            ),
            ("improper", "harmonic", "harmonic", &["10", "180"]),
            ("improper", "cvff", "cvff", &["1.1", "-1", "2"]),
            ("dihedral", "opls", "opls", &["1", "2", "3", "4"]),
            ("dihedral", "harmonic", "harmonic", &["2", "-1", "3"]),
            (
                "dihedral",
                "fourier",
                "periodic",
                &["2", "0.5", "1", "180", "0.25", "3", "0"],
            ),
            ("dihedral", "charmm", "charmm", &["0.2", "3", "180", "0.5"]),
            (
                "dihedral",
                "multi/harmonic",
                "multi/harmonic",
                &["1", "2", "3", "4", "5"],
            ),
            (
                "dihedral",
                "nharmonic",
                "nharmonic",
                &["3", "1", "-2", "0.5"],
            ),
            ("pair", "lj/cut", "lj/cut", &["0.066", "3.5"]),
        ];
        for (category, lammps, molrs, tokens) in cases {
            let params = lammps_coeff_params(category, lammps, tokens, "real").unwrap();
            let values = lammps_coeff_values(category, molrs, &params, "real")
                .unwrap_or_else(|e| panic!("{category} {molrs}: {e}"));
            let strings: Vec<String> = values.iter().map(f64::to_string).collect();
            let strs: Vec<&str> = strings.iter().map(String::as_str).collect();
            assert_eq!(&strs, tokens, "{category} {lammps}: the identity on tokens");
            let back = lammps_coeff_params(category, lammps, &strs, "real").unwrap();
            assert_eq!(back, params, "{category} {lammps}: {strs:?}");
        }
    }

    /// A CHARMM include with Urey–Bradley angles beside plain ones: one
    /// `angle_style hybrid`, each `angle_coeff` naming its sub-style.
    const UB_HYBRID: &str = "\
# LAMMPS force field generated by molrs
units real

special_bonds lj 0.000000 0.000000 0.000000 coul 0.000000 0.000000 0.000000

angle_style hybrid harmonic charmm
angle_coeff CT-CT-CT harmonic 58.350000 113.600000
angle_coeff HA-CT-CT charmm 33.430000 110.100000 22.530000 2.179000
angle_coeff HA-CT-HA charmm 35.500000 108.400000 5.400000 1.802000

";

    /// `angle charmm` reads `K theta0 K_ub r_ub` as written and writes it back
    /// the same: the LAMMPS read → write identity, hybrid included. (The
    /// writer orders rows by label id, and `TypeLabels` sorts labels.)
    #[test]
    fn angle_charmm_hybrid_reads_and_writes_back_identically() {
        let ff = LammpsFfReader::new().read_str(UB_HYBRID).unwrap();
        let ub = ff.get_style("angle", "charmm").unwrap();
        let StyleDefs::Angle(types) = ub.defs() else {
            panic!("an angle style");
        };
        let t = types.iter().find(|t| t.name == "HA-CT-CT").unwrap();
        for (key, want) in [
            ("k", 33.43),
            ("theta0", 110.1),
            ("k_ub", 22.53),
            ("r_ub", 2.179),
        ] {
            assert_eq!(t.params.get(key), Some(want), "{key}");
        }
        assert!(ff.get_style("angle", "harmonic").is_some());

        let labels = labels_of(&[("angles", &["CT-CT-CT", "HA-CT-CT", "HA-CT-HA"])]);
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
        assert_eq!(text, UB_HYBRID);
        let again = LammpsFfReader::new().read_str(&text).unwrap();
        assert_eq!(
            LammpsFfWriter::new(&labels).write_str(&again).unwrap(),
            text
        );
    }

    /// One style needs no `hybrid`; the data-file section names it.
    #[test]
    fn angle_charmm_alone_is_a_plain_style_and_its_data_section_says_so() {
        let ff = LammpsFfReader::new().read_str(UB_HYBRID).unwrap();
        let labels = labels_of(&[("angles", &["HA-CT-CT"])]);
        let writer = LammpsFfWriter::new(&labels);
        let text = writer.write_str(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "angle_"),
            vec![
                "angle_style charmm",
                "angle_coeff HA-CT-CT 33.430000 110.100000 22.530000 2.179000",
            ],
            "{text}"
        );
        let data = writer.write_data_coeffs_str(&ff).unwrap();
        assert!(data.contains("Angle Coeffs # charmm\n"), "{data}");
        assert_eq!(
            data_section_rows(&data, "Angle Coeffs"),
            vec!["1 33.430000 110.100000 22.530000 2.179000"]
        );
    }

    /// A data file's hybrid section carries the sub-style on each row, and
    /// `read_data_coeffs` reads it back to the same force field.
    #[test]
    fn angle_charmm_hybrid_data_coeffs_round_trip() {
        use crate::ff::forcefield::readers::lammps::LammpsTypeLabelMaps;
        let ff = LammpsFfReader::new().read_str(UB_HYBRID).unwrap();
        let names = ["CT-CT-CT", "HA-CT-CT", "HA-CT-HA"];
        let labels = labels_of(&[("angles", &names)]);
        let data = LammpsFfWriter::new(&labels)
            .write_data_coeffs_str(&ff)
            .unwrap();
        assert!(data.contains("Angle Coeffs # hybrid\n"), "{data}");
        assert_eq!(
            data_section_rows(&data, "Angle Coeffs"),
            vec![
                "1 harmonic 58.350000 113.600000",
                "2 charmm 33.430000 110.100000 22.530000 2.179000",
                "3 charmm 35.500000 108.400000 5.400000 1.802000",
            ]
        );
        let maps = LammpsTypeLabelMaps {
            angle: (1..).zip(names.iter().map(|n| n.to_string())).collect(),
            ..Default::default()
        };
        let back = LammpsFfReader::new()
            .read_data_coeffs(&data, &maps, "real")
            .unwrap();
        let angles = |ff: &ForceField, style: &str| -> Vec<AngleType> {
            match ff.get_style("angle", style).unwrap().defs() {
                StyleDefs::Angle(types) => types.clone(),
                _ => panic!("an angle style"),
            }
        };
        for style in ["charmm", "harmonic"] {
            assert_eq!(angles(&back, style), angles(&ff, style), "{style}");
        }
    }

    /// Under `angle_style harmonic`, a `K theta0 K_ub r_ub` line is refused —
    /// read as harmonic it would silently drop its Urey–Bradley term — and a
    /// hybrid line must name a declared sub-style.
    #[test]
    fn a_urey_bradley_line_under_the_wrong_style_is_refused() {
        let err = LammpsFfReader::new()
            .read_str(
                "special_bonds charmm\nangle_style harmonic\n\
                 angle_coeff A-B-A 33.43 110.1 22.53 2.179\n",
            )
            .unwrap_err();
        assert!(err.contains("takes 2 coefficients, got 4"), "{err}");
        let err = LammpsFfReader::new()
            .read_str(
                "special_bonds charmm\nangle_style hybrid harmonic\n\
                 angle_coeff A-B-A charmm 33.43 110.1 22.53 2.179\n",
            )
            .unwrap_err();
        assert!(err.contains("not one of"), "{err}");
    }

    // ── fix cmap ────────────────────────────────────────────────────────────

    const ALANINE: &str = include_str!("../../potential/cmap/testdata/charmm36_alanine.cmap");

    /// The numbers of a CHARMM cmap file, as its lines: comments dropped.
    fn number_lines(text: &str) -> Vec<&str> {
        text.lines()
            .map(str::trim_end)
            .filter(|l| !l.is_empty() && !l.trim_start().starts_with('#'))
            .collect()
    }

    fn cmap_ff(grids: &[(&str, ArrayD<f64>)]) -> ForceField {
        let mut ff = ForceField::new("charmm");
        let style = ff.def_style("cmap", "charmm", Params::new()).unwrap();
        for (name, grid) in grids {
            let mut params = Params::new();
            params.set_array("grid", grid.clone());
            style.def_type(name, &[*name; 5], params).unwrap();
        }
        ff
    }

    /// Read CHARMM's file, write it: every number line comes back as the very
    /// line it was read from, and the text reads back to the same bits.
    #[test]
    fn a_charmm_cmap_file_is_written_back_line_for_line() {
        use crate::ff::forcefield::readers::lammps::read_lammps_cmap_str;
        let map = read_lammps_cmap_str(ALANINE).unwrap().maps.remove(0);
        let ff = cmap_ff(&[("ala", map.clone())]);
        let labels = labels_of(&[("cmaps", &["ala", "ala"])]);
        let text = LammpsFfWriter::new(&labels).write_cmap_str(&ff).unwrap();
        assert_eq!(number_lines(&text), number_lines(ALANINE));
        assert!(text.starts_with("# UNITS: real "), "{text}");
        assert!(text.contains("\n# ala, type 1\n"), "{text}");
        let back = read_lammps_cmap_str(&text).unwrap();
        assert_eq!(back.units.as_deref(), Some("real"));
        assert_eq!(back.maps, vec![map.clone()]);

        // Any value survives once the decimals reach its 17th digit.
        let odd = map.mapv(|v| v / 3.0);
        let text = lammps_cmap_str(&[("odd", &odd)], "real", 25).unwrap();
        assert_eq!(read_lammps_cmap_str(&text).unwrap().maps, vec![odd]);
    }

    /// Maps go out in the `cmaps` labels' id order (the data file's crossterm
    /// types), converted to the file's units; the include names the file on
    /// a `fix cmap` line beside `units`, and reads back.
    #[test]
    fn the_cmap_file_and_fix_line_follow_the_labels() {
        let a = ArrayD::from_shape_fn(vec![24, 24], |ix| (ix[0] * 24 + ix[1]) as f64 / 100.0);
        let b = a.mapv(|v| -v);
        let ff = cmap_ff(&[("b", b.clone()), ("a", a.clone())]);
        let labels = labels_of(&[("cmaps", &["b", "a", "b"])]);
        let options = LammpsWriteOptions {
            cmap_file: Some("sys.cmap".into()),
            skip_pair_style: true,
            ..LammpsWriteOptions::default()
        };
        let writer = LammpsFfWriter::with_options(&labels, options);
        let text = writer.write_cmap_str(&ff).unwrap();
        let maps = crate::ff::forcefield::readers::lammps::read_lammps_cmap_str(&text)
            .unwrap()
            .maps;
        assert_eq!(maps, vec![a.clone(), b]);

        let include = writer.write_str(&ff).unwrap();
        let fix = include.find("fix cmap all cmap sys.cmap\nfix_modify cmap energy yes\n");
        let units = include.find("units real");
        assert!(fix.is_some() && units < fix, "{include}");

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("sys.cmap"), &text).unwrap();
        let path = dir.path().join("sys.ff");
        std::fs::write(&path, include.replace("\n\n", "\nspecial_bonds charmm\n\n")).unwrap();
        let back = LammpsFfReader::new().read(path.to_str().unwrap()).unwrap();
        let rows = back.get_cmaptypes();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].params.get_array("grid"), Some(&a));

        // In metal units the energies are converted.
        let metal = LammpsWriteOptions {
            units: "metal",
            precision: 12,
            ..LammpsWriteOptions::default()
        };
        let text = LammpsFfWriter::with_options(&labels, metal)
            .write_cmap_str(&ff)
            .unwrap();
        let file = crate::ff::forcefield::readers::lammps::read_lammps_cmap_str(&text).unwrap();
        assert_eq!(file.units.as_deref(), Some("metal"));
        let ev = file.maps[0][[0, 1]];
        assert!((ev - 0.01 * 0.0433641).abs() < 1e-8, "{ev}");
    }

    #[test]
    fn what_fix_cmap_cannot_read_is_refused() {
        let grid = ArrayD::zeros(vec![24, 24]);
        let labels = labels_of(&[("cmaps", &["a"])]);
        // The include of a system with crossterms needs the file name.
        let err = LammpsFfWriter::new(&labels)
            .write_str(&cmap_ff(&[("a", grid.clone())]))
            .unwrap_err();
        assert!(err.contains("cmap_file"), "{err}");
        // fix cmap reads 24×24 maps, at most six.
        let err = LammpsFfWriter::new(&labels)
            .write_cmap_str(&cmap_ff(&[("a", ArrayD::zeros(vec![12, 12]))]))
            .unwrap_err();
        assert!(err.contains("24×24"), "{err}");
        let names = ["a", "b", "c", "d", "e", "f", "g"];
        let many: Vec<(&str, ArrayD<f64>)> = names.iter().map(|n| (*n, grid.clone())).collect();
        let err = LammpsFfWriter::new(&labels_of(&[("cmaps", &names)]))
            .write_cmap_str(&cmap_ff(&many))
            .unwrap_err();
        assert!(err.contains("at most 6"), "{err}");
        // A label without a row.
        let err = LammpsFfWriter::new(&labels_of(&[("cmaps", &["zz"])]))
            .write_cmap_str(&cmap_ff(&[("a", grid)]))
            .unwrap_err();
        assert!(err.contains("`zz`"), "{err}");
    }
}
