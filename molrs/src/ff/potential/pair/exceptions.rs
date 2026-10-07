//! The 1-4 exceptions kernel: every non-bonded pair whose pricing is not the
//! pair styles' own, as one list.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use ndarray::{Array2, ArrayView2};

use crate::ff::forcefield::combining_rule::CombiningRule;
use crate::ff::forcefield::one_four::check_materialized;
use crate::ff::forcefield::{ForceField, Params};
use crate::ff::potential::compile::gathered;
use crate::ff::potential::end_pairs;
use crate::ff::potential::flat_coords::{term_table, validate_coords};
use crate::ff::potential::pair::charmm::{charmm_mixing, charmm_pair_params, lj_coeffs};
use crate::ff::potential::pair::lj_cut::{lj_pair_params, mixing_of};
use crate::ff::potential::param_reads;
use crate::ff::potential::{IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::core::keys::{ATOMI, ATOMJ, ATOML};
use molrs::core::schema::PAIR_OVERRIDE_COLUMNS;
use molrs::core::schema::block_names::{ATOMS, BONDS, DIHEDRALS, PAIRS};
use molrs::op::F;

const MIN_R2: F = 1e-24;

/// The 1-4 exceptions of one molecule: LJ 12-6 plus Coulomb, no cutoff.
///
/// LAMMPS has three ways to price a close (1-4) pair, and molrs represents
/// each with LAMMPS's parameters (the conventions guide, "1-4 interactions"):
///
/// 1. **`special_bonds`** — the pair styles price the pair at their own
///    parameters, scaled by the force field's 1-4 weights. Nothing here.
/// 2. **`dihedral_style charmm` `w`** — each dihedral prices the pair of its
///    end atoms, `w·[LJ(ε₁₄, σ₁₄) + C qᵢqⱼ/r]` (`dihedral_charmm.cpp`), with
///    the `epsilon14` / `sigma14` of the `lj/charmm` pair style mixed as
///    LAMMPS's `init_one` mixes them, no cutoff and no switch. LAMMPS refuses
///    `w > 0` beside non-zero `special_bonds` 1-4 weights, and so does molrs.
///    A pair at the ends of several dihedrals takes the sum of their `w`.
/// 3. **Per-pair overrides** — columns on the frame's `pairs` block for what
///    LAMMPS cannot express (a GROMACS `[ pairs ]` row with parameters, an
///    OpenMM exception, an AMBER dihedral's own SCEE / SCNB):
///    [`PAIR_OVERRIDE_COLUMNS`] = `epsilon`, `sigma`, `charge_product`,
///    `lj_scale`, `coul_scale`.
///
/// **Precedence**, per pair and per quantity: a per-pair override cell is
/// final; a null cell takes what the pair would have without the row — the
/// dihedral's `w` pricing when its ends carry `w > 0`, the pair style's
/// parameters at the `special_bonds` weight of the pair's bond-distance class
/// otherwise. So `lj_scale` / `coul_scale` replace `w` or the global weight,
/// and `epsilon` / `sigma` / `charge_product` replace the style's (or the
/// dihedral's 1-4) values. A cell is priced only by the style it belongs to:
/// `epsilon` / `sigma` / `lj_scale` under a Lennard-Jones style,
/// `charge_product` / `coul_scale` under a Coulomb style. A field without
/// that style ignores them, so a bonded-only field on a frame with
/// materialized 1-4 cells prices no pair at all.
///
/// Every such pair is priced here, once:
///
/// ```text
/// E = lj_w · 4ε[(σ/r)¹² − (σ/r)⁶]  +  coul_w · C qᵢqⱼ / r
/// ```
///
/// with no cutoff (as LAMMPS's dihedral 1-4 term, and an OpenMM exception),
/// `C` the Coulomb style's `coulomb / dielectric`. The regular pair kernels
/// price an **override** pair at weight 0 — the compiled door drops its
/// `pairs` row, the neighbour-driven door zeroes its weight
/// ([`PairWeights`](crate::ff::potential::PairWeights)). A `w` pair needs no
/// such step: `special_bonds` 1-4 is 0 for it, as LAMMPS requires.
///
/// The kernel is a fixed list of atom pairs, so it is an indexed (bond-like)
/// member at both compile doors: a periodic régime rebinds it like a bond.
pub struct PairExceptions {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    /// LAMMPS's `lj1..lj4` of each pair.
    lj: Vec<[F; 4]>,
    lj_w: Vec<F>,
    /// `C·qᵢqⱼ`, LAMMPS's `qqrd2e·qᵢ·qⱼ`.
    qq: Vec<F>,
    coul_w: Vec<F>,
}

impl PairExceptions {
    /// One pair per row: `(ε, σ)` with its weight and `C·qᵢqⱼ` with its
    /// weight.
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        eps_sigma: &[(F, F)],
        lj_w: Vec<F>,
        qq: Vec<F>,
        coul_w: Vec<F>,
    ) -> Self {
        let n = atom_i.len();
        assert!(
            [
                atom_j.len(),
                eps_sigma.len(),
                lj_w.len(),
                qq.len(),
                coul_w.len()
            ]
            .iter()
            .all(|&m| m == n)
        );
        Self {
            atom_i,
            atom_j,
            lj: eps_sigma.iter().map(|&(e, s)| lj_coeffs(e, s)).collect(),
            lj_w,
            qq,
            coul_w,
        }
    }

    /// How many pairs this kernel prices.
    pub fn len(&self) -> usize {
        self.atom_i.len()
    }

    /// Whether it prices none.
    pub fn is_empty(&self) -> bool {
        self.atom_i.is_empty()
    }

    /// The van-der-Waals and Coulomb parts of the energy, apart — LAMMPS
    /// tallies the dihedral's 1-4 pair into `evdwl` and `ecoul`.
    pub fn energy_terms(&self, coords: &[F]) -> (F, F) {
        let mut out = vec![0.0; coords.len()];
        self.fold(coords, &mut out, self.len(), |t| {
            (self.atom_i[t], self.atom_j[t])
        })
    }

    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n: usize,
        atoms: impl Fn(usize) -> (usize, usize),
    ) -> (F, F) {
        validate_coords(coords);
        let (mut evdwl, mut ecoul) = (0.0, 0.0);
        for t in 0..n {
            let (i, j) = atoms(t);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            let rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if rsq < MIN_R2 {
                continue;
            }
            // dihedral_charmm.cpp, with the one weight split in two.
            let r2inv = 1.0 / rsq;
            let r6inv = r2inv * r2inv * r2inv;
            let c = &self.lj[t];
            let forcecoul = self.qq[t] * r2inv.sqrt();
            let forcelj = r6inv * (c[0] * r6inv - c[1]);
            let fpair = (self.lj_w[t] * forcelj + self.coul_w[t] * forcecoul) * r2inv;
            evdwl += self.lj_w[t] * (r6inv * (c[2] * r6inv - c[3]));
            ecoul += self.coul_w[t] * forcecoul;
            for k in 0..3 {
                out[j * 3 + k] += fpair * d[k];
                out[i * 3 + k] -= fpair * d[k];
            }
        }
        (evdwl, ecoul)
    }
}

impl Potential for PairExceptions {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let mut out = vec![0.0; coords.len()];
        let e = self.accumulate(coords, &mut out);
        (e, out)
    }

    fn accumulate(&self, coords: &[F], out: &mut [F]) -> F {
        let (v, c) = self.fold(coords, out, self.len(), |t| {
            (self.atom_i[t], self.atom_j[t])
        });
        v + c
    }
}

impl IndexedTerms for PairExceptions {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j])
    }

    fn calc_energy_forces_with_terms(
        &self,
        coords: &[F],
        terms: ArrayView2<'_, u32>,
    ) -> (F, Vec<F>) {
        let mut out = vec![0.0; coords.len()];
        let e = self.accumulate_with_terms(coords, terms, &mut out);
        (e, out)
    }

    fn accumulate_with_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, out: &mut [F]) -> F {
        debug_assert_eq!(terms.nrows(), self.len());
        let (v, c) = self.fold(coords, out, terms.nrows(), |t| {
            (terms[[t, 0]] as usize, terms[[t, 1]] as usize)
        });
        v + c
    }
}

/// What the compiler routes through the exceptions kernel for one frame.
pub(crate) struct Exceptions {
    /// The kernel, when any pair is an exception.
    pub(crate) kernel: Option<PairExceptions>,
    /// Rows of the frame's `pairs` block that carry an override cell: the
    /// regular pair kernels must not price them.
    pub(crate) override_rows: Vec<usize>,
    /// Those rows' atom pairs, `(lo, hi)`.
    pub(crate) replaced: Vec<(usize, usize)>,
}

/// One `pairs` row's override cells, in [`PAIR_OVERRIDE_COLUMNS`] order.
type Cells = [Option<F>; 5];
const EPS: usize = 0;
const SIGMA: usize = 1;
const QQ: usize = 2;
const LJ_SCALE: usize = 3;
const COUL_SCALE: usize = 4;

/// The van-der-Waals style an exception's default LJ comes from.
enum Vdw<'f> {
    None,
    LjCut(HashMap<String, Params>, CombiningRule),
    Charmm(HashMap<String, Params>, CombiningRule),
    /// A style with no LJ 12-6 form (`buck`, `morse`, Mie `lj/cut`, …).
    Other(&'f str),
}

/// How the compiler prices the frame's 1-4 exceptions under `ff`.
///
/// # Errors
///
/// A `dihedral charmm` `w` outside `[0, 1]`; `w > 0` beside `special_bonds`
/// 1-4 weights other than 0, or without a `lj/charmm` pair style and a
/// Coulomb style (all three LAMMPS's own errors); an override column that is
/// not float, holds a non-finite value, or names one pair twice; and, when
/// any exception exists, a pair style the exception cannot stand in for.
pub(crate) fn plan(ff: &ForceField, frame: &Frame) -> Result<Exceptions, String> {
    let wsum = dihedral_weights(ff, frame)?;
    let (cells, override_rows) = override_cells(frame)?;
    let rows_ij = pair_ends(frame);
    check_one_four(ff, frame, &wsum, &cells, &rows_ij)?;

    let mut entries: BTreeMap<(usize, usize), (F, Option<usize>)> = BTreeMap::new();
    for (&key, &w) in &wsum {
        entries.insert(key, (w, None));
    }
    let mut replaced = Vec::with_capacity(override_rows.len());
    for &r in &override_rows {
        let key = rows_ij[r];
        let e = entries.entry(key).or_insert((0.0, None));
        if e.1.is_some() {
            return Err(format!(
                "pairs: atoms {} and {} carry per-pair overrides on two rows; a pair \
                 has one set of exception parameters",
                key.0, key.1
            ));
        }
        e.1 = Some(r);
        replaced.push(key);
    }
    if entries.is_empty() {
        return Ok(Exceptions {
            kernel: None,
            override_rows,
            replaced,
        });
    }

    let (vdw, coul) = pair_styles(ff)?;
    if let Vdw::Other(name) = vdw {
        return Err(format!(
            "pair style '{name}': a per-pair 1-4 exception is priced as Lennard-Jones 12-6 \
             plus Coulomb, and '{name}' has no such form to stand in for"
        ));
    }
    let types = frame
        .get(ATOMS)
        .and_then(|b| b.get("type"))
        .and_then(|c| c.as_string());
    let charges = frame
        .get(ATOMS)
        .and_then(|b| b.get("charge"))
        .and_then(|c| c.as_float());
    let is_14 = frame
        .get(PAIRS)
        .and_then(|b| b.get("is_14"))
        .and_then(|c| c.as_bool());
    let classes = BondClasses::new(frame);
    let sb = ff.special_bonds();
    let lj_rows = match &vdw {
        Vdw::Charmm(rows, _) | Vdw::LjCut(rows, _) => as_refs(rows),
        _ => HashMap::new(),
    };

    let n = entries.len();
    let (mut ai, mut aj) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut es, mut lj_w, mut qq, mut coul_w) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (&(i, j), &(w, row)) in &entries {
        let cell = row.map(|r| cells[r]).unwrap_or([None; 5]);
        // The class weight is needed only for a pair the dihedral does not
        // price, and only where an override leaves a scale null.
        let class = || match &classes {
            Some(c) => c.distance(i, j),
            None => {
                let flagged = row.is_some_and(|r| is_14.is_some_and(|f| f[r]));
                if flagged { 3 } else { 0 }
            }
        };
        let weight = |table: [f64; 3]| match class() {
            d @ 1..=3 => table[d - 1],
            _ => 1.0,
        };

        // Van der Waals: the cells, else the dihedral's 1-4 parameters, else
        // the style's own. A field with no Lennard-Jones style prices none of
        // the LJ cells: they belong to a style it does not have.
        let base = |mixing: CombiningRule, charmm: bool| -> Result<(F, F), String> {
            let (a, b) = atom_type_pair(types, i, j)?;
            if charmm {
                let (regular, one_four) =
                    charmm_pair_params(&lj_rows, mixing, a, b).map_err(|e| e.to_string())?;
                Ok(if w > 0.0 { one_four } else { regular })
            } else {
                lj_pair_params("lj/cut", &lj_rows, mixing, a, b).map_err(|e| e.to_string())
            }
        };
        let (eps, sigma, w_lj) = match &vdw {
            Vdw::Charmm(_, mixing) | Vdw::LjCut(_, mixing) => {
                let (e, s) = match (cell[EPS], cell[SIGMA]) {
                    (Some(e), Some(s)) => (e, s),
                    (e, s) => {
                        let (be, bs) = base(*mixing, matches!(vdw, Vdw::Charmm(..)))?;
                        (e.unwrap_or(be), s.unwrap_or(bs))
                    }
                };
                let w_lj =
                    cell[LJ_SCALE].unwrap_or_else(|| if w > 0.0 { w } else { weight(sb.lj) });
                (e, s, w_lj)
            }
            _ => (0.0, 0.0, 0.0),
        };

        // Coulomb, likewise only under a Coulomb style.
        let (k, w_c, qq_ij) = match coul {
            None => (0.0, 0.0, 0.0),
            Some(k) => {
                let w_c =
                    cell[COUL_SCALE].unwrap_or_else(|| if w > 0.0 { w } else { weight(sb.coul) });
                let qq_ij = match cell[QQ] {
                    Some(v) => v,
                    None if w_c == 0.0 => 0.0,
                    None => {
                        let q = charges.ok_or_else(|| {
                            "1-4 exceptions: atoms block missing \"charge\" column".to_string()
                        })?;
                        q[i] * q[j]
                    }
                };
                (k, w_c, qq_ij)
            }
        };
        ai.push(i);
        aj.push(j);
        es.push((eps, sigma));
        lj_w.push(w_lj);
        qq.push(k * qq_ij);
        coul_w.push(w_c);
    }
    Ok(Exceptions {
        kernel: Some(PairExceptions::new(ai, aj, &es, lj_w, qq, coul_w)),
        override_rows,
        replaced,
    })
}

fn as_refs(rows: &HashMap<String, Params>) -> HashMap<&str, &Params> {
    rows.iter().map(|(k, v)| (k.as_str(), v)).collect()
}

fn atom_type_pair(
    types: Option<&ndarray::ArrayD<String>>,
    i: usize,
    j: usize,
) -> Result<(&str, &str), String> {
    let t = types.ok_or("1-4 exceptions: atoms block missing \"type\" column")?;
    Ok((t[i].as_str(), t[j].as_str()))
}

/// The sum of `w` over the `dihedral charmm` rows ending at each atom pair,
/// after LAMMPS's checks of the weights.
pub(crate) fn dihedral_weights(
    ff: &ForceField,
    frame: &Frame,
) -> Result<BTreeMap<(usize, usize), F>, String> {
    let mut wsum = BTreeMap::new();
    for style in ff.get_styles("dihedral") {
        if style.name() != "charmm" {
            continue;
        }
        let mut by_type: HashMap<String, F> = HashMap::new();
        let (_, rows) = gathered(style).map_err(|e| e.to_string())?;
        for (name, p) in rows {
            let w = param_reads::type_num("charmm", &name, &p, "w").map_err(|e| e.to_string())?;
            // LAMMPS: "Incorrect weight arg for dihedral coefficients".
            if !(0.0..=1.0).contains(&w) {
                return Err(format!(
                    "dihedral charmm type '{name}': the 1-4 weight w = {w} is outside [0, 1]"
                ));
            }
            by_type.insert(name, w);
        }
        if !by_type.values().any(|&w| w > 0.0) {
            continue;
        }
        check_weightflag(ff)?;
        let Some(block) = frame.get(DIHEDRALS) else {
            continue;
        };
        let Some(labels) = block.get("type").and_then(|c| c.as_string()) else {
            continue;
        };
        for (r, (i, l)) in end_pairs(frame, DIHEDRALS, ATOMI, ATOML)
            .into_iter()
            .enumerate()
        {
            if let Some(&w) = by_type.get(labels[r].as_str())
                && w > 0.0
            {
                *wsum.entry((i, l)).or_insert(0.0) += w;
            }
        }
    }
    Ok(wsum)
}

/// A `lj/charmm` field with `one_four = "epsilon14"` must have its 1-4 pairs
/// priced by an override (`epsilon` and `sigma` cells) or a `w` dihedral:
/// [`check_materialized`]. The 1-4 pairs are the `pairs` rows flagged `is_14`,
/// or, for a frame without a `pairs` block (the neighbour-driven door), the
/// bond-graph 1-4 pairs.
fn check_one_four(
    ff: &ForceField,
    frame: &Frame,
    wsum: &BTreeMap<(usize, usize), F>,
    cells: &[Cells],
    rows_ij: &[(usize, usize)],
) -> Result<(), String> {
    let mut overridden: HashSet<(usize, usize)> = HashSet::new();
    for (r, c) in cells.iter().enumerate() {
        if c[EPS].is_some() && c[SIGMA].is_some() {
            overridden.insert(rows_ij[r]);
        }
    }
    let pairs_14: Vec<(usize, usize)> = match frame
        .get(PAIRS)
        .and_then(|b| b.get("is_14"))
        .and_then(|c| c.as_bool())
    {
        Some(flags) => rows_ij
            .iter()
            .zip(flags.iter())
            .filter(|(_, f)| **f)
            .map(|(k, _)| *k)
            .collect(),
        None => BondClasses::new(frame)
            .map(|c| c.pairs_at_three())
            .unwrap_or_default(),
    };
    let covered = |i: usize, j: usize| {
        overridden.contains(&(i, j)) || wsum.get(&(i, j)).is_some_and(|&w| w > 0.0)
    };
    check_materialized(ff, frame, &pairs_14, &covered)
}

/// LAMMPS's `DihedralCharmm::init_style` checks for a field with `w > 0`.
fn check_weightflag(ff: &ForceField) -> Result<(), String> {
    let sb = ff.special_bonds();
    if sb.lj[2] != 0.0 || sb.coul[2] != 0.0 {
        return Err(format!(
            "dihedral charmm has a 1-4 weight w > 0 beside special_bonds 1-4 weights \
             lj {} coul {}: the pair would be priced twice. LAMMPS refuses it too \
             (\"Must use 'special_bonds charmm' with dihedral style charmm for use with \
             CHARMM pair styles\"): set the 1-4 weights to 0, or w to 0",
            sb.lj[2], sb.coul[2]
        ));
    }
    let names: Vec<&str> = ff.get_styles("pair").iter().map(|s| s.name()).collect();
    if !names.contains(&"lj/charmm") {
        return Err(
            "dihedral charmm has a 1-4 weight w > 0, which prices the 1-4 pair with the \
             epsilon14/sigma14 of a lj/charmm pair style, and the force field has none \
             (LAMMPS: \"Dihedral charmm is incompatible with Pair style\")"
                .into(),
        );
    }
    if !names
        .iter()
        .any(|n| matches!(*n, "coul/charmm" | "coul/cut"))
    {
        return Err(
            "dihedral charmm has a 1-4 weight w > 0, which prices the 1-4 Coulomb pair with \
             the Coulomb pair style's constant, and the force field has none (coul/charmm)"
                .into(),
        );
    }
    Ok(())
}

/// Each `pairs` row's override cells, and the rows that carry any.
fn override_cells(frame: &Frame) -> Result<(Vec<Cells>, Vec<usize>), String> {
    let Some(block) = frame.get(PAIRS) else {
        return Ok((Vec::new(), Vec::new()));
    };
    let n = block.nrows().unwrap_or(0);
    let mut cells = vec![[None; 5]; n];
    for (c, &key) in PAIR_OVERRIDE_COLUMNS.iter().enumerate() {
        let Some(col) = block.get(key) else {
            continue;
        };
        let values = col
            .as_float()
            .ok_or_else(|| format!("pairs: the per-pair override column '{key}' must be float"))?;
        let valid = block.validity(key);
        for r in 0..n {
            if valid.is_some_and(|m| !m[r]) {
                continue;
            }
            let v = values[r];
            if !v.is_finite() {
                return Err(format!(
                    "pairs row {r}: '{key}' = {v}; a null cell is a validity mask, not a \
                     non-finite value"
                ));
            }
            cells[r][c] = Some(v);
        }
    }
    let rows = (0..n)
        .filter(|&r| cells[r].iter().any(Option::is_some))
        .collect();
    Ok((cells, rows))
}

/// `(lo, hi)` of each `pairs` row.
fn pair_ends(frame: &Frame) -> Vec<(usize, usize)> {
    end_pairs(frame, PAIRS, ATOMI, ATOMJ)
}

/// The force field's van-der-Waals style and Coulomb constant, as far as an
/// exception needs them.
fn pair_styles(ff: &ForceField) -> Result<(Vdw<'_>, Option<F>), String> {
    let (mut vdw, mut coul) = (Vdw::None, None);
    let mut seen_vdw = None::<&str>;
    for style in ff.get_styles("pair") {
        let name = style.name();
        let found = match name {
            "lj/cut" => {
                let (p, rows) = gathered(style).map_err(|e| e.to_string())?;
                let num = |k: &str| param_reads::style_num(name, &p, k).map_err(|e| e.to_string());
                if num("n")? == 12.0 && num("m")? == 6.0 {
                    let mixing = mixing_of(name, &p).map_err(|e| e.to_string())?;
                    Some(Vdw::LjCut(rows.into_iter().collect(), mixing))
                } else {
                    Some(Vdw::Other(name))
                }
            }
            "lj/charmm" => {
                let (p, rows) = gathered(style).map_err(|e| e.to_string())?;
                let mixing = charmm_mixing(&p).map_err(|e| e.to_string())?;
                Some(Vdw::Charmm(rows.into_iter().collect(), mixing))
            }
            "coul/cut" | "coul/charmm" => {
                let (p, _) = gathered(style).map_err(|e| e.to_string())?;
                if p.get("delta").is_some_and(|d| d != 0.0) {
                    return Err(format!(
                        "pair style '{name}': a buffered Coulomb (delta ≠ 0) has no 1-4 \
                         exception form"
                    ));
                }
                if coul.is_some() {
                    return Err("1-4 exceptions: the force field has two Coulomb styles".into());
                }
                let num = |k: &str| param_reads::style_num(name, &p, k).map_err(|e| e.to_string());
                coul = Some(num("coulomb")? / num("dielectric")?);
                None
            }
            _ => Some(Vdw::Other(name)),
        };
        if let Some(v) = found {
            if let Some(prev) = seen_vdw {
                return Err(format!(
                    "1-4 exceptions: pair styles '{prev}' and '{name}' are both non-Coulomb; \
                     an exception stands in for one Lennard-Jones style"
                ));
            }
            seen_vdw = Some(name);
            vdw = v;
        }
    }
    Ok((vdw, coul))
}

/// Bond-graph distances up to 3, from the frame's `bonds`.
struct BondClasses {
    adjacency: Vec<Vec<usize>>,
}

impl BondClasses {
    fn new(frame: &Frame) -> Option<Self> {
        frame.get(BONDS)?;
        let n = frame.get(ATOMS).and_then(|b| b.nrows()).unwrap_or(0);
        let mut adjacency = vec![Vec::new(); n];
        for (a, b) in end_pairs(frame, BONDS, ATOMI, ATOMJ) {
            if a < n && b < n && a != b {
                adjacency[a].push(b);
                adjacency[b].push(a);
            }
        }
        Some(Self { adjacency })
    }

    /// Every `(lo, hi)` pair at bond distance exactly 3.
    fn pairs_at_three(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for i in 0..self.adjacency.len() {
            let mut depth = HashMap::from([(i, 0usize)]);
            let mut queue = VecDeque::from([i]);
            while let Some(a) = queue.pop_front() {
                let d = depth[&a];
                if d == 3 {
                    if a > i {
                        out.push((i, a));
                    }
                    continue;
                }
                for &b in &self.adjacency[a] {
                    if let std::collections::hash_map::Entry::Vacant(e) = depth.entry(b) {
                        e.insert(d + 1);
                        queue.push_back(b);
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// The bond distance of `(i, j)` when it is 1, 2 or 3; 0 beyond.
    fn distance(&self, i: usize, j: usize) -> usize {
        let mut seen = HashSet::from([i]);
        let mut queue = VecDeque::from([(i, 0usize)]);
        while let Some((a, d)) = queue.pop_front() {
            if a == j {
                return d;
            }
            if d == 3 {
                continue;
            }
            for &b in self.adjacency.get(a).map(Vec::as_slice).unwrap_or(&[]) {
                if seen.insert(b) {
                    queue.push_back((b, d + 1));
                }
            }
        }
        0
    }
}
