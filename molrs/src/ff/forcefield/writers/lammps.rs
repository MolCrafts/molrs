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
//! - `bonds`, `angles` and `dihedrals` labels match a `ForceField` type in
//!   either orientation ([`TypeName::canonical`]); the canonical label is what
//!   is written.
//! - `impropers` labels match exactly: reversing an improper moves its centre
//!   and names a different term.
//! - A block whose types carry no labels (pure-integer types) is matched by
//!   its ids, `"1"`, `"2"`, ….
//! - A style is written only when it holds a used type; an unsupported style
//!   is an error only then. Type-less pair styles (`coul/cut`) apply to every
//!   atom and are always in play.
//!
//! # Units (molrs store → LAMMPS file)
//!
//! Inverse of [`super::super::readers::lammps::LammpsFfReader`]: takes a molrs
//! [`ForceField`] in molrs units (Å, kcal/mol, **radians**, e; harmonic
//! stiffness in the `½k(x−x₀)²` form) and emits AMBER/GAFF-flavour LAMMPS
//! coefficients in LAMMPS `real` units:
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
//! Store is **real** (Å, kcal/mol, rad; `½k` harmonic form) for force fields
//! read from physical styles, or **lj** pass-through when the file was already
//! reduced. Writing always goes through
//! [`LammpsFfUnits`]
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

use std::collections::{BTreeMap, HashMap, HashSet};

use super::ForceFieldWriter;
use crate::ff::forcefield::lammps_units::{LammpsFfUnits, molrs_half_k_to_lammps_k, parse_style};
use crate::ff::forcefield::mixing::Mixing;
use crate::ff::forcefield::{
    AngleType, BondType, DihedralType, ForceField, ImproperType, PairType, Params, Style, StyleDefs,
};
use molrs::store::type_labels::{TypeLabels, TypeName};

/// Default pair cutoff (Å / reduced σ) when a style carries none — keeps the
/// written include a legal LAMMPS command rather than a bare `pair_style lj/cut`.
const DEFAULT_PAIR_CUTOFF: f64 = 10.0;

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
}

impl Default for LammpsWriteOptions {
    fn default() -> Self {
        Self {
            precision: 6,
            skip_pair_style: false,
            skip_units: false,
            units: "real",
        }
    }
}

/// Conversion context: store → file units via the lj hub + form maps.
struct WriteUnits {
    sys: LammpsFfUnits,
    file: &'static str,
}

impl WriteUnits {
    fn new(file: &'static str) -> Result<Self, String> {
        Ok(Self {
            sys: LammpsFfUnits::canonical().map_err(|e| format!("lammps unit system: {e}"))?,
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
/// The result is the coefficients **after** the type field(s), in the LAMMPS
/// `units` style `units` (`real`, `metal`, `lj`); `params` are in molrs store
/// units — Å, kcal/mol, radians (`lj` stays reduced):
///
/// | category / style          | stored params                          | LAMMPS values |
/// |---------------------------|----------------------------------------|---------------|
/// | `bond harmonic`           | `k`, `r0`                              | `K = k/2`, `r0` |
/// | `angle harmonic`          | `k`, `theta0` (rad)                    | `K = k/2`, `theta0` (deg) |
/// | `improper harmonic`       | `k`, `chi0` (rad)                      | `K = k/2`, `chi0` (deg) |
/// | `dihedral opls`           | `k1..k4` (absent → 0)                  | `K1 K2 K3 K4` |
/// | `dihedral harmonic`       | `k`, `sign` (±1), `periodicity`        | `K d n` |
/// | `dihedral fourier`        | `k<i>`, `periodicity<i>`, `phase<i>` (rad, absent → 0) | `m K1 n1 d1(deg) …` |
/// | `dihedral charmm`         | `k`, `periodicity`, `phase` (rad, absent → 0), `w` | `K n d(deg) w` |
/// | `dihedral multi/harmonic` | `a1..a5` (absent → 0)                  | `A1 A2 A3 A4 A5` |
/// | `pair lj/cut…`            | `epsilon`, `sigma`                     | `epsilon sigma` |
///
/// `K = k/2` because molrs's harmonic kernels are `½·k·(x−x₀)²` and LAMMPS's
/// are `K·(x−x₀)²`. An absent param falls back only where the molrs kernel
/// reads the same default. Multiplicities (`n`, fourier `m`) are integral.
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
/// let p = Params::from_pairs(&[("k", 900.0), ("r0", 0.9572)]);
/// assert_eq!(lammps_coeff_values("bond", "harmonic", &p, "real").unwrap(), [450.0, 0.9572]);
/// assert!(lammps_coeff_values("bond", "morse", &p, "real").is_err());
/// ```
pub fn lammps_coeff_values(
    category: &str,
    style: &str,
    params: &Params,
    units: &str,
) -> Result<Vec<f64>, String> {
    let units = WriteUnits::new(parse_style(units)?)?;
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
        // Equilibrium angles are degrees in the LAMMPS file for every unit style.
        ("angle", "harmonic") => Ok(vec![
            Real(units.angle_k(need("k")?)?),
            Real(need("theta0")?.to_degrees()),
        ]),
        ("improper", "harmonic") => Ok(vec![
            Real(units.angle_k(need("k")?)?),
            Real(need("chi0")?.to_degrees()),
        ]),
        ("dihedral", "opls") => energies(&["k1", "k2", "k3", "k4"]),
        // E = K[1 + d·cos(nφ)]: `d` is the stored sign (±1), not a phase.
        ("dihedral", "harmonic") => Ok(vec![
            Real(units.energy(need("k")?)?),
            Real(need("sign")?),
            multiplicity("periodicity", need("periodicity")?)?,
        ]),
        // m  K1 n1 d1  [K2 n2 d2 ...] from `k<i>` / `periodicity<i>` / `phase<i>`.
        ("dihedral", "fourier") => {
            let mut terms = Vec::new();
            let mut i = 1usize;
            while let Some(k) = params.get(&format!("k{i}")) {
                let n_key = format!("periodicity{i}");
                terms.push(Real(units.energy(k)?));
                terms.push(multiplicity(&n_key, need(&n_key)?)?);
                let phase = params.get(&format!("phase{i}")).unwrap_or(0.0);
                terms.push(Real(phase.to_degrees()));
                i += 1;
            }
            if terms.is_empty() {
                return Err(format!("{category} {style}: missing param `k1`"));
            }
            let mut fields = vec![Coeff::Int(terms.len() as i64 / 3)];
            fields.extend(terms);
            Ok(fields)
        }
        // E = K[1 + cos(nφ − d)]; `w` is the 1-4 pair weight.
        ("dihedral", "charmm") => Ok(vec![
            Real(units.energy(need("k")?)?),
            multiplicity("periodicity", need("periodicity")?)?,
            Real(params.get("phase").unwrap_or(0.0).to_degrees()),
            Real(need("w")?),
        ]),
        ("dihedral", "multi/harmonic") => energies(&["a1", "a2", "a3", "a4", "a5"]),
        ("pair", s) if s.starts_with("lj/cut") => Ok(vec![
            Real(units.energy(need("epsilon")?)?),
            Real(units.length(need("sigma")?)?),
        ]),
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
    /// Whether a label and its reverse name one term (all but impropers).
    const UNDIRECTED: bool;

    /// The types of this category `defs` holds, or `None` for another category.
    fn types_of(defs: &StyleDefs) -> Option<&[Self]>;

    /// The stored type name.
    fn name(&self) -> &str;

    /// The stored params, rendered by [`lammps_coeff_values`]'s conversion.
    fn params(&self) -> &Params;

    /// Lookup key of a name: its canonical orientation when undirected.
    fn key(name: &str) -> String {
        if Self::UNDIRECTED {
            TypeName::from(name.to_owned()).canonical().to_string()
        } else {
            name.to_owned()
        }
    }
}

impl BondedCoeff for BondType {
    const CATEGORY: &'static str = "bond";
    const BLOCK: &'static str = "bonds";
    const HEADING: &'static str = "Bond Coeffs";
    const UNDIRECTED: bool = true;

    fn types_of(defs: &StyleDefs) -> Option<&[Self]> {
        match defs {
            StyleDefs::Bond(types) => Some(types),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn params(&self) -> &Params {
        &self.params
    }
}

impl BondedCoeff for AngleType {
    const CATEGORY: &'static str = "angle";
    const BLOCK: &'static str = "angles";
    const HEADING: &'static str = "Angle Coeffs";
    const UNDIRECTED: bool = true;

    fn types_of(defs: &StyleDefs) -> Option<&[Self]> {
        match defs {
            StyleDefs::Angle(types) => Some(types),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn params(&self) -> &Params {
        &self.params
    }
}

impl BondedCoeff for DihedralType {
    const CATEGORY: &'static str = "dihedral";
    const BLOCK: &'static str = "dihedrals";
    const HEADING: &'static str = "Dihedral Coeffs";
    const UNDIRECTED: bool = true;

    fn types_of(defs: &StyleDefs) -> Option<&[Self]> {
        match defs {
            StyleDefs::Dihedral(types) => Some(types),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn params(&self) -> &Params {
        &self.params
    }
}

impl BondedCoeff for ImproperType {
    const CATEGORY: &'static str = "improper";
    const BLOCK: &'static str = "impropers";
    const HEADING: &'static str = "Improper Coeffs";
    const UNDIRECTED: bool = false;

    fn types_of(defs: &StyleDefs) -> Option<&[Self]> {
        match defs {
            StyleDefs::Improper(types) => Some(types),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn params(&self) -> &Params {
        &self.params
    }
}

/// A bonded label resolved to the style and type that define it.
struct Resolved<'f, T> {
    /// 1-based type id (label order).
    id: usize,
    /// The label as written: canonical orientation when undirected.
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
        let units = WriteUnits::new(self.options.units)?;
        let mut lines: Vec<String> = Vec::new();
        self.write_data_pair_coeffs(&mut lines, ff, &units)?;
        self.write_data_section::<BondType>(&mut lines, ff, &units)?;
        self.write_data_section::<AngleType>(&mut lines, ff, &units)?;
        self.write_data_section::<DihedralType>(&mut lines, ff, &units)?;
        self.write_data_section::<ImproperType>(&mut lines, ff, &units)?;
        Ok(lines.concat())
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
        let mut by_key: HashMap<String, (&'f Style, &'f T)> = HashMap::new();
        for style in ff.get_styles(T::CATEGORY) {
            for ty in T::types_of(style.defs()).unwrap_or_default() {
                by_key.entry(T::key(ty.name())).or_insert((style, ty));
            }
        }
        labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let label = T::key(label);
                let &(style, ty) = by_key.get(&label).ok_or_else(|| {
                    format!(
                        "{}: type label `{label}` has no {} type in the force field",
                        T::BLOCK,
                        T::CATEGORY
                    )
                })?;
                Ok(Resolved {
                    id: i + 1,
                    label,
                    style,
                    ty,
                })
            })
            .collect()
    }

    /// Pair rows for the `atoms` labels, ordered by id pair: every label's
    /// self pair (missing → error) and every explicit cross pair of two used
    /// labels. The first style (in force-field order) defining a pair wins.
    fn resolve_pairs<'f>(&self, ff: &'f ForceField) -> Result<Vec<PairRow<'f>>, String> {
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
                let lj_cut = units.length(style_cutoff(lj).unwrap_or(DEFAULT_PAIR_CUTOFF))?;
                let coul_cut = units.length(style_cutoff(coul).unwrap_or(DEFAULT_PAIR_CUTOFF))?;
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

    /// `T_style` and `T_coeff` lines for every style holding a used label, the
    /// coefficients in label id order.
    fn write_section<T: BondedCoeff>(
        &self,
        lines: &mut Vec<String>,
        ff: &ForceField,
        units: &WriteUnits,
    ) -> Result<(), String> {
        let used = self.resolve::<T>(ff)?;
        for style in ff.get_styles(T::CATEGORY) {
            let rows: Vec<&Resolved<'_, T>> = used
                .iter()
                .filter(|r| std::ptr::eq(r.style, style))
                .collect();
            if rows.is_empty() {
                continue;
            }
            let mut section = vec![format!("{}_style {}\n", T::CATEGORY, style.name())];
            for r in rows {
                section.push(format!(
                    "{}_coeff {} {}\n",
                    T::CATEGORY,
                    r.label,
                    r.coeffs(units, self.options.precision)?
                ));
            }
            section.push("\n".to_owned());
            lines.extend(section);
        }
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
        let mut section = Vec::new();
        for r in self.resolve::<T>(ff)? {
            section.push(format!(
                "{} {}\n",
                r.id,
                r.coeffs(units, self.options.precision)?
            ));
        }
        push_data_section(lines, T::HEADING, section);
        Ok(())
    }
}

impl ForceFieldWriter for LammpsFfWriter<'_> {
    fn write_str(&self, ff: &ForceField) -> Result<String, String> {
        let units = WriteUnits::new(self.options.units)?;
        let mut lines: Vec<String> = Vec::new();
        lines.push("# LAMMPS force field generated by molrs\n".to_owned());
        if !self.options.skip_units {
            lines.push(format!("units {}\n", self.options.units));
            lines.push("\n".to_owned());
        }
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

fn is_split_lj_coulomb(styles: &[&Style]) -> bool {
    if styles.len() != 2 {
        return false;
    }
    let names: HashSet<&str> = styles.iter().map(|s| s.name()).collect();
    names == HashSet::from(["lj/cut", "coul/cut"])
        || names == HashSet::from(["lj/cut", "coul/long"])
}

fn pair_style_cutoffs(style: &Style, units: &WriteUnits) -> Result<Vec<f64>, String> {
    // Combined names want two cutoffs; simple kernels one. Fall back to default
    // so the line is a legal LAMMPS command.
    let convert = |c: f64| units.length(c);
    match style.name() {
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
    style.params().get("cutoff")
}

// ── helpers ──────────────────────────────────────────────────────────────────

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
        let labels = mini_labels();
        let text = LammpsFfWriter::new(&labels).write_str(&ff).unwrap();
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

    #[test]
    fn write_ff_emits_canonical_dihedral_name_once() {
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
        let n = text
            .lines()
            .filter(|l| l.starts_with("dihedral_coeff "))
            .count();
        assert_eq!(n, 1, "{text}");
        assert!(text.contains("dihedral_coeff c3-os-c3-h1"), "{text}");
        assert!(!text.contains("dihedral_coeff h1-c3-os-c3"), "{text}");
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
        assert!(data.contains("Bond Coeffs\n"), "{data}");
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
        rest.lines()
            .skip_while(|l| l.trim().is_empty())
            .take_while(|l| !l.trim().is_empty())
            .filter_map(|l| l.split_whitespace().next()?.parse().ok())
            .collect()
    }

    /// Bonds / angles / dihedrals are undirected: reverse hyphen names share
    /// one LAMMPS type id and must emit one coeff row (not two rows with the
    /// same id — LAMMPS then treats the extra line as an unknown identifier).
    #[test]
    fn write_data_coeffs_collapses_reverse_bonded_names() {
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
        // Rows in both orientations collapse to one label (id 1) per block.
        let labels = labels_of(&[
            ("atoms", &["c3"]),
            ("bonds", &["c3-h1", "h1-c3"]),
            ("angles", &["c3-c3-h1", "h1-c3-c3"]),
            ("dihedrals", &["h1-c3-c3-os", "os-c3-c3-h1"]),
        ]);
        let data = LammpsFfWriter::new(&labels)
            .write_data_coeffs_str(&ff)
            .unwrap();
        assert_eq!(coeff_ids(&data, "Bond Coeffs"), vec![1], "{data}");
        assert_eq!(coeff_ids(&data, "Angle Coeffs"), vec![1], "{data}");
        assert_eq!(coeff_ids(&data, "Dihedral Coeffs"), vec![1], "{data}");
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

    #[test]
    fn metal_write_converts_energy_via_lj_hub() {
        let ff = LammpsFfReader::new().read_str(MINI).unwrap();
        let opts = LammpsWriteOptions {
            units: "metal",
            ..Default::default()
        };
        let labels = mini_labels();
        let text = LammpsFfWriter::with_options(&labels, opts)
            .write_str(&ff)
            .unwrap();
        assert!(text.contains("units metal\n"), "metal header:\n{text}");

        // 0.1078 kcal/mol → eV through lj hub
        let sys = crate::ff::forcefield::lammps_units::LammpsFfUnits::canonical().unwrap();
        let eps_ev = sys.from_store_energy(0.1078, "metal").unwrap();
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

    /// Harmonic bond in molrs `½k` form: stored `k` is twice the LAMMPS `K`.
    fn bond(k_lammps: f64, r0: f64) -> Params {
        Params::from_pairs(&[("k", 2.0 * k_lammps), ("r0", r0)])
    }

    /// Split `lj/cut` (cutoff 9) + `coul/cut` (cutoff 10) with hand-written
    /// ε/σ for `c3`, `hc` and `oh`, and harmonic bonds `c3-hc`, `c3-oh`.
    fn split_pair_ff() -> ForceField {
        let mut ff = ForceField::new("hand");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 9.0)]))
            .unwrap()
            .def_type("c3", lj(0.1078, 3.39771))
            .unwrap()
            .def_type("hc", lj(0.0157, 2.64953))
            .unwrap()
            .def_type("oh", lj(0.093, 3.242871))
            .unwrap();
        ff.def_style("pair", "coul/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-hc", bond(340.0, 1.09))
            .unwrap()
            .def_type("c3-oh", bond(320.0, 1.41))
            .unwrap();
        ff
    }

    /// Rows of a data-file `heading` section (between its blank lines).
    fn data_section_rows(text: &str, heading: &str) -> Vec<String> {
        let Some(rest) = text.split(&format!("{heading}\n")).nth(1) else {
            return vec![];
        };
        rest.lines()
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
            .def_type("c3", lj(0.1078, 3.39771))
            .unwrap()
            .def_type("hc", lj(0.0157, 2.64953))
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

    #[test]
    fn label_writer_resolves_reversed_bond_label_to_stored_type() {
        let mut ff = ForceField::new("hand");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("h1-c3", bond(340.0, 1.09))
            .unwrap();
        // TypeLabels stores the canonical orientation `c3-h1`.
        let labels = labels_of(&[("bonds", &["c3-h1"])]);
        let writer = LammpsFfWriter::new(&labels);

        let text = writer.write_str(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "bond_coeff"),
            vec!["bond_coeff c3-h1 340.000000 1.090000"],
            "{text}"
        );
        let data = writer.write_data_coeffs_str(&ff).unwrap();
        assert_eq!(
            data_section_rows(&data, "Bond Coeffs"),
            vec!["1 340.000000 1.090000"],
            "{data}"
        );
    }

    /// Improper `k` in molrs `½k` form, `chi0` in radians.
    fn improper_ff() -> ForceField {
        let mut ff = ForceField::new("hand");
        ff.def_style("improper", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "c3-n-c-o",
                Params::from_pairs(&[("k", 2.2), ("chi0", std::f64::consts::PI)]),
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
            vec!["improper_coeff c3-n-c-o 1.100000 180.000000"],
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

    #[test]
    fn label_data_coeff_ids_follow_type_labels_not_forcefield_order() {
        // ForceField order: oh, hc, c3 / c3-oh, c3-hc. Label order (sorted):
        // c3=1, hc=2, oh=3 / c3-hc=1, c3-oh=2.
        let mut ff = ForceField::new("hand");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 9.0)]))
            .unwrap()
            .def_type("oh", lj(0.093, 3.242871))
            .unwrap()
            .def_type("hc", lj(0.0157, 2.64953))
            .unwrap()
            .def_type("c3", lj(0.1078, 3.39771))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-oh", bond(320.0, 1.41))
            .unwrap()
            .def_type("c3-hc", bond(340.0, 1.09))
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
            .def_type("c3-hc", lj(0.05, 3.0))
            .unwrap()
            .def_type("c3-oh", lj(0.07, 3.3))
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

    /// Harmonic `c3-hc` plus an unsupported `morse` style holding `morse_type`.
    fn ff_with_morse(morse_type: &str) -> ForceField {
        let mut ff = ForceField::new("hand");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-hc", bond(340.0, 1.09))
            .unwrap();
        ff.def_style("bond", "morse", Params::new())
            .unwrap()
            .def_type(
                morse_type,
                Params::from_pairs(&[("d0", 90.0), ("alpha", 2.0), ("r0", 1.4)]),
            )
            .unwrap();
        ff
    }

    #[test]
    fn label_writer_tolerates_unsupported_style_holding_only_unused_types() {
        let ff = ff_with_morse("c3-oh");
        let labels = labels_of(&[("bonds", &["c3-hc"])]);
        let writer = LammpsFfWriter::new(&labels);
        let text = writer.write_str(&ff).unwrap();
        assert!(!text.contains("morse"), "{text}");
        assert!(text.contains("bond_coeff c3-hc"), "{text}");
        let data = writer.write_data_coeffs_str(&ff).unwrap();
        assert_eq!(
            data_section_rows(&data, "Bond Coeffs"),
            vec!["1 340.000000 1.090000"],
            "{data}"
        );
    }

    #[test]
    fn label_writer_rejects_unsupported_style_holding_a_used_type() {
        let ff = ff_with_morse("c3-oh");
        let labels = labels_of(&[("bonds", &["c3-hc", "c3-oh"])]);
        let writer = LammpsFfWriter::new(&labels);
        let err = writer.write_str(&ff).unwrap_err();
        assert!(err.contains("morse"), "names the style: {err}");
        let err = writer.write_data_coeffs_str(&ff).unwrap_err();
        assert!(err.contains("morse"), "names the style: {err}");
    }

    /// UFF-style qualified labels (system-forcefield-07's grammar on 05's `@`
    /// qualifier) resolve through the reversed orientation: the frame labels
    /// the bond `O_R-C_3@1.5` and the angle `O_2-C_R-C_3@1.5_1_2` (the angle's
    /// two bond-order fields swapped, as `TypeName::reversed` does), the force
    /// field defines `C_3-O_R@1.5` and `C_3-C_R-O_2@1_1.5_2`. Hand-written
    /// numbers: bond K = 350 (stored `k` = 700, molrs ½k form), r0 = 1.40;
    /// angle K = 60 (stored `k` = 120), theta0 = 120 degrees.
    #[test]
    fn label_writer_writes_reversed_uff_qualified_labels_once_each() {
        let mut ff = ForceField::new("hand");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("C_3-O_R@1.5", bond(350.0, 1.40))
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "C_3-C_R-O_2@1_1.5_2",
                Params::from_pairs(&[("k", 120.0), ("theta0", 120.0_f64.to_radians())]),
            )
            .unwrap();
        let labels = labels_of(&[
            ("bonds", &["O_R-C_3@1.5"]),
            ("angles", &["O_2-C_R-C_3@1.5_1_2"]),
        ]);

        let text = LammpsFfWriter::new(&labels)
            .write_str(&ff)
            .expect("both reversed labels resolve");

        assert_eq!(
            lines_starting_with(&text, "bond_coeff"),
            vec!["bond_coeff C_3-O_R@1.5 350.000000 1.400000"],
            "{text}"
        );
        assert_eq!(
            lines_starting_with(&text, "angle_coeff"),
            vec!["angle_coeff C_3-C_R-O_2@1_1.5_2 60.000000 120.000000"],
            "{text}"
        );
    }

    /// A pair style whose types carry no ε/σ (`thole`: per-type `charge`,
    /// `alpha`, `a_thole`) has no `pair_coeff` form here. Writing the
    /// `pair_style` line without its coefficients is an incomplete include, so
    /// both writers refuse and name the category and the style.
    #[test]
    fn pair_style_without_writable_coeffs_is_err_naming_style() {
        let mut ff = ForceField::new("hand");
        ff.def_style("pair", "thole", Params::from_pairs(&[("cutoff", 12.0)]))
            .unwrap()
            .def_type(
                "c3",
                Params::from_pairs(&[("charge", -0.2), ("alpha", 1.1), ("a_thole", 2.6)]),
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

    /// Hand-derived goldens for every kernel, in `real`: harmonic `K = k/2`,
    /// radians → degrees, energies pass through.
    #[test]
    fn lammps_coeff_values_renders_each_kernel() {
        let pi = std::f64::consts::PI;
        let cases: &[ValuesCase<'_>] = &[
            (
                "bond",
                "harmonic",
                &[("k", 900.0), ("r0", 0.9572)],
                &[450.0, 0.9572],
            ),
            (
                "angle",
                "harmonic",
                &[("k", 110.0), ("theta0", 1.824_218_134_184_473_2)],
                &[55.0, 104.52],
            ),
            (
                "improper",
                "harmonic",
                &[("k", 20.0), ("chi0", pi)],
                &[10.0, 180.0],
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
                "fourier",
                &[
                    ("k1", 0.5),
                    ("periodicity1", 1.0),
                    ("phase1", pi),
                    ("k2", 0.25),
                    ("periodicity2", 3.0),
                    ("phase2", 0.0),
                ],
                &[2.0, 0.5, 1.0, 180.0, 0.25, 3.0, 0.0],
            ),
            (
                "dihedral",
                "charmm",
                &[("k", 0.2), ("periodicity", 3.0), ("phase", pi), ("w", 0.5)],
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
        let err = lammps_coeff_values("bond", "morse", &Params::from_pairs(&[("k", 1.0)]), "real")
            .unwrap_err();
        assert!(err.contains("bond") && err.contains("morse"), "{err}");

        let err = lammps_coeff_values(
            "bond",
            "harmonic",
            &Params::from_pairs(&[("k", 900.0)]),
            "real",
        )
        .unwrap_err();
        assert!(err.contains("r0"), "{err}");

        let bond = Params::from_pairs(&[("k", 900.0), ("r0", 0.9572)]);
        let err = lammps_coeff_values("bond", "harmonic", &bond, "si").unwrap_err();
        assert!(err.contains("si"), "{err}");
    }

    /// Writer and reader are inverse through their two public homes: the
    /// values rendered for a kernel read back, as tokens, to the same params.
    #[test]
    fn lammps_coeff_values_round_trips_through_lammps_coeff_params() {
        use crate::ff::forcefield::readers::lammps::lammps_coeff_params;
        let cases: &[(&str, &str, &[&str])] = &[
            ("bond", "harmonic", &["450", "0.9572"]),
            ("angle", "harmonic", &["55", "104.52"]),
            ("improper", "harmonic", &["10", "180"]),
            ("dihedral", "opls", &["1", "2", "3", "4"]),
            ("dihedral", "harmonic", &["2", "-1", "3"]),
            (
                "dihedral",
                "fourier",
                &["2", "0.5", "1", "180", "0.25", "3", "0"],
            ),
            ("dihedral", "charmm", &["0.2", "3", "180", "0.5"]),
            ("dihedral", "multi/harmonic", &["1", "2", "3", "4", "5"]),
            ("pair", "lj/cut", &["0.066", "3.5"]),
        ];
        for (category, style, tokens) in cases {
            let params = lammps_coeff_params(category, style, tokens, "real").unwrap();
            let values = lammps_coeff_values(category, style, &params, "real")
                .unwrap_or_else(|e| panic!("{category} {style}: {e}"));
            let strings: Vec<String> = values.iter().map(f64::to_string).collect();
            let strs: Vec<&str> = strings.iter().map(String::as_str).collect();
            let back = lammps_coeff_params(category, style, &strs, "real").unwrap();
            assert_eq!(back, params, "{category} {style}: {strs:?}");
        }
    }
}
