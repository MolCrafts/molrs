//! MMFF94's Van der Waals kernel — the buffered 14-7 potential.
//!
//! The style is **per-type**: its 95 rows (one per MMFF atom type) carry the
//! polarizabilities the combining rules need, and the style-level block
//! (`<VdWParams B= Beta= DARAD= DAEPS=>`) shapes those rules.
//!
//! # MMFF's electrostatics is not here, and owns no kernel
//!
//! It is a **buffered Coulomb** — `E = k·qᵢqⱼ / (D·(r + δ))` — which is the *generic*
//! [`PairCoulCut`](super::coul_cut::PairCoulCut) kernel at `k = 332.0716`, `D = 1.0`,
//! `δ = 0.05 Å`: a parameterization, not a kernel of its own. The numbers live in
//! [`MMFF_ELE_STYLE`](crate::ff::params::mmff::MMFF_ELE_STYLE) and reach the kernel
//! through the style. vdW is the opposite case — a genuine per-type table — which is
//! why it stays.
//!
//! # 1-4 scaling (house convention)
//!
//! MMFF scales the 1-4 **electrostatic** interaction by 0.75 and does **not**
//! scale the 1-4 **van der Waals** interaction at all (Halgren 1996; RDKit
//! `Nonbonded.cpp`). Both weights arrive through [`SpecialBonds`], which
//! [`Style::to_potential`] projects into the pair params as `coulomb14scale` /
//! `lj14scale` — so neither kernel hardcodes a scale factor.
//!
//! [`SpecialBonds`]: crate::ff::forcefield::SpecialBonds
//! [`Style::to_potential`]: crate::ff::forcefield::Style::to_potential

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::{mag3, sub3, validate_coords};
use molrs::store::frame::Frame;
use molrs::types::F;

// ---------------------------------------------------------------------------
// Donor / acceptor encoding (MMFF94.PAR's `DA` column)
// ---------------------------------------------------------------------------

/// Neither hydrogen-bond donor nor acceptor (`DA` = `"-"`).
pub(crate) const DA_NEITHER: u8 = 0;
/// Hydrogen-bond **donor** (`DA` = `"D"`) — polar hydrogen.
pub(crate) const DA_DONOR: u8 = 1;
/// Hydrogen-bond **acceptor** (`DA` = `"A"`).
pub(crate) const DA_ACCEPTOR: u8 = 2;

/// Encode MMFF's `DA` column (`"D"` / `"A"` / `"-"`) as the small integer code
/// stored in a `mmff_vdw` [`PairType`](crate::ff::forcefield::PairType) row's
/// `da` param.
///
/// [`Params`] is a numeric bag, so a reader transcribes the letter here and
/// [`vdw_combining`] decodes it — one encoding, owned by its consumer. Anything
/// unrecognised is [`DA_NEITHER`], which is also MMFF's own default for the
/// 80-odd non-hydrogen-bonding types.
pub(crate) fn encode_da(raw: &str) -> f64 {
    match raw {
        "D" => DA_DONOR,
        "A" => DA_ACCEPTOR,
        _ => DA_NEITHER,
    }
    .into()
}

/// [`encode_da`] for the compiled table, whose `da` column is the letter's ASCII
/// byte ([`MmffVdW::da`](crate::ff::params::mmff::MmffVdW::da), as RDKit stores
/// it) rather than a string. Same three codes, same default — one mapping, two
/// spellings of the letter.
pub(crate) fn encode_da_byte(code: u8) -> f64 {
    match code {
        b'D' => DA_DONOR,
        b'A' => DA_ACCEPTOR,
        _ => DA_NEITHER,
    }
    .into()
}

// ---------------------------------------------------------------------------
// MMFFVdW: Buffered 14-7 potential
// ---------------------------------------------------------------------------

/// Buffered 14-7 van der Waals (MMFF.I eq. 8), one row per non-excluded pair.
///
/// `r_star` is in Å and `epsilon` in kcal·mol⁻¹; both are the *combined* pair
/// values produced by [`mmff_vdw_ctor`]'s combining rules (including the
/// donor/acceptor corrections), with the 1-4 weight (`lj14scale`, 1.0 for MMFF)
/// already folded into `epsilon`.
pub struct MMFFVdW {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    r_star: Vec<F>,
    epsilon: Vec<F>,
}

impl Potential for MMFFVdW {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0 as F; coords.len()];

        for idx in 0..self.atom_i.len() {
            let (i, j) = (self.atom_i[idx], self.atom_j[idx]);
            let d = sub3(coords, j, coords, i);
            let r = mag3(d);
            if r < 1e-12 as F {
                continue;
            }
            let rs = self.r_star[idx];
            let eps = self.epsilon[idx];

            let rho = r + 0.07 * rs;
            let u = 1.07 * rs / rho;
            let u7 = u * u * u * u * u * u * u;
            let r7 = r * r * r * r * r * r * r;
            let rs7 = rs * rs * rs * rs * rs * rs * rs;
            let v = 1.12 * rs7 / (r7 + 0.12 * rs7);

            energy += eps * u7 * (v - 2.0);

            let du_dr = -u / rho;
            let dv_dr = -7.0 * r * r * r * r * r * r * v / (r7 + 0.12 * rs7);
            let de_dr = eps * (7.0 * u7 / u * du_dr * (v - 2.0) + u7 * dv_dr);

            let factor = -de_dr / r;
            for dim in 0..3 {
                forces[j * 3 + dim] += factor * d[dim];
                forces[i * 3 + dim] -= factor * d[dim];
            }
        }
        (energy, forces)
    }
}

/// Per-atom VdW parameters for combining rules.
///
/// `alpha` is the atomic polarizability α (Å³), `n_eff` the Slater-Kirkwood
/// effective electron number N, `a_i` / `g_i` the MMFF scale factors A and G, and
/// `da` the hydrogen-bond role ([`DA_NEITHER`] / [`DA_DONOR`] / [`DA_ACCEPTOR`]).
struct VdwAtomParams {
    alpha: f64,
    n_eff: f64,
    a_i: f64,
    g_i: f64,
    da: u8,
}

/// Style-level VdW constants (MMFF94.PAR's global vdW line, `<VdWParams>`).
///
/// `B` and `Beta` shape the R* combining rule; `DARAD` and `DAEPS` are the
/// donor-acceptor corrections. They are **data**, not constants of this kernel —
/// the defaults below are MMFF94's own values and exist only so a hand-built
/// style without the section still evaluates the standard force field.
struct VdwStyleParams {
    b: f64,
    beta: f64,
    darad: f64,
    daeps: f64,
}

impl VdwStyleParams {
    fn from_style(sp: &Params) -> Self {
        Self {
            b: sp.get("B").unwrap_or(0.2),
            beta: sp.get("Beta").unwrap_or(12.0),
            darad: sp.get("DARAD").unwrap_or(0.8),
            daeps: sp.get("DAEPS").unwrap_or(0.5),
        }
    }
}

/// MMFF's van der Waals combining rules (Halgren, *MMFF.II*, J. Comput. Chem.
/// **17**, 520–552 (1996), eqs. 5–7), returning `(R*_ij [Å], eps_ij [kcal/mol])`.
///
/// ```text
/// R*_ii = A_i * alpha_i^(1/4)
/// gamma = (R*_ii - R*_jj) / (R*_ii + R*_jj)
/// R*_ij = 0.5 * (R*_ii + R*_jj) * (1 + B * (1 - exp(-Beta * gamma^2)))
/// eps_ij = 181.16 * G_i * G_j * alpha_i * alpha_j
///          / ((sqrt(alpha_i/N_i) + sqrt(alpha_j/N_j)) * R*_ij^6)
/// ```
///
/// # The two donor/acceptor rules
///
/// Both are part of the *definition* of MMFF's vdW term, not an optional
/// refinement, and omitting them costs (measured, vs RDKit): urea +0.588, e_big
/// +1.261, N-methylacetamide +0.116 kcal/mol.
///
/// 1. **Donor R\* suppression** — if either partner is a donor, the `B` shape
///    term is dropped (`R*_ij` is the plain arithmetic mean). A polar hydrogen
///    approaches its acceptor far closer than the empirical expansion allows.
/// 2. **Donor-acceptor scaling** — for a donor *paired with* an acceptor,
///    `R*_ij *= DARAD` (0.8) and `eps_ij *= DAEPS` (0.5): the hydrogen bond is
///    shorter and softer than dispersion alone would give.
///
/// `eps_ij` is evaluated at the **unscaled** `R*_ij`, then scaled — the order is
/// load-bearing (`R*^-6` would otherwise pick up a spurious `DARAD^-6` = 3.8x).
fn vdw_combining(pi: &VdwAtomParams, pj: &VdwAtomParams, sp: &VdwStyleParams) -> (F, F) {
    let rs_i = pi.a_i * pi.alpha.powf(0.25);
    let rs_j = pj.a_i * pj.alpha.powf(0.25);
    let gamma = (rs_i - rs_j) / (rs_i + rs_j);

    // Rule 1: a donor suppresses the B shape term entirely.
    let donor = pi.da == DA_DONOR || pj.da == DA_DONOR;
    let shape = if donor {
        0.0
    } else {
        sp.b * (1.0 - (-sp.beta * gamma * gamma).exp())
    };
    let mut r_star = 0.5 * (rs_i + rs_j) * (1.0 + shape);

    let denom = (pi.alpha / pi.n_eff).sqrt() + (pj.alpha / pj.n_eff).sqrt();
    let mut epsilon = 181.16 * pi.g_i * pj.g_i * pi.alpha * pj.alpha / (denom * r_star.powi(6));

    // Rule 2: donor-acceptor pairs get a shorter, softer minimum.
    let da_pair =
        (pi.da == DA_DONOR && pj.da == DA_ACCEPTOR) || (pi.da == DA_ACCEPTOR && pj.da == DA_DONOR);
    if da_pair {
        r_star *= sp.darad;
        epsilon *= sp.daeps;
    }
    (r_star as F, epsilon as F)
}

/// Build the buffered-14-7 van der Waals potential.
///
/// Style params (`sp`, from `<VdWParams>`): `B`, `Beta`, `DARAD`, `DAEPS`, plus
/// the `lj14scale` weight [`Style::to_potential`] projects out of the force
/// field's [`SpecialBonds`]. Type params (`tp`, from `<VdW>`): `alpha`, `n_eff`,
/// `a_i`, `g_i`, `da`.
///
/// # MMFF does not scale 1-4 van der Waals
///
/// `lj14scale` is **1.0** for MMFF (set by the reader) and applying it is a
/// deliberate no-op: MMFF's 1-4 rule touches electrostatics only (0.75), and its
/// torsion parameters were fitted against *unscaled* 1-4 vdW. Do not "fix" this
/// to 0.5 by analogy with Amber — it would corrupt every MMFF vdW energy with a
/// 1-4 pair. The weight is read (rather than ignored) so that a force field which
/// *does* scale can reuse this kernel, and so the 1.0 is visible as a choice.
///
/// [`SpecialBonds`]: crate::ff::forcefield::SpecialBonds
/// [`Style::to_potential`]: crate::ff::forcefield::Style::to_potential
pub fn mmff_vdw_ctor(
    sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let style = VdwStyleParams::from_style(sp);
    let lj_14 = sp.get("lj14scale").unwrap_or(1.0) as F;
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let atoms = frame.get("atoms").ok_or("mmff_vdw: missing \"atoms\"")?;
    let atom_types = atoms
        .get_string("type")
        .ok_or("mmff_vdw: missing atom \"type\"")?;
    let pairs = frame.get("pairs").ok_or("mmff_vdw: missing \"pairs\"")?;
    let ic = pairs.get_uint("atomi").ok_or("missing atomi")?;
    let jc = pairs.get_uint("atomj").ok_or("missing atomj")?;
    let is_14 = pairs.get_bool("is_14");

    let n = ic.len();
    let (mut ai, mut aj, mut rs_vec, mut eps_vec) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let ti = &atom_types[ic[idx] as usize];
        let tj = &atom_types[jc[idx] as usize];
        let pi = type_map
            .get(ti.as_str())
            .ok_or_else(|| format!("mmff_vdw: unknown atom type '{}'", ti))?;
        let pj = type_map
            .get(tj.as_str())
            .ok_or_else(|| format!("mmff_vdw: unknown atom type '{}'", tj))?;

        let to_vdw = |p: &Params, label: &str| -> Result<VdwAtomParams, String> {
            let get = |k: &str| {
                p.get(k)
                    .ok_or_else(|| format!("mmff_vdw type '{}': missing '{}'", label, k))
            };
            Ok(VdwAtomParams {
                alpha: get("alpha")?,
                n_eff: get("n_eff")?,
                a_i: get("a_i")?,
                g_i: get("g_i")?,
                da: p.get("da").unwrap_or(f64::from(DA_NEITHER)) as u8,
            })
        };
        let (rs, eps) = vdw_combining(&to_vdw(pi, ti)?, &to_vdw(pj, tj)?, &style);
        // The 1-4 weight scales the well depth, hence the whole pair energy
        // (E is linear in eps). For MMFF this multiplies by exactly 1.0.
        let scale = if is_14.is_some_and(|b| b[idx]) {
            lj_14
        } else {
            1.0
        };
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        rs_vec.push(rs);
        eps_vec.push(eps * scale);
    }
    Ok(Box::new(MMFFVdW {
        atom_i: ai,
        atom_j: aj,
        r_star: rs_vec,
        epsilon: eps_vec,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MMFF94's global vdW line (`<VdWParams B="0.2" Beta="12.0" DARAD="0.8"
    /// DAEPS="0.5">`), as the reader hands it to the kernel.
    fn mmff94_vdw_style() -> VdwStyleParams {
        VdwStyleParams::from_style(&Params::from_pairs(&[
            ("B", 0.2),
            ("Beta", 12.0),
            ("DARAD", 0.8),
            ("DAEPS", 0.5),
        ]))
    }

    #[test]
    fn test_mmff_vdw_combining() {
        let p = VdwAtomParams {
            alpha: 1.050,
            n_eff: 2.490,
            a_i: 3.890,
            g_i: 1.282,
            da: DA_NEITHER,
        };
        let (rs, eps) = vdw_combining(&p, &p, &mmff94_vdw_style());
        assert!(rs > 0.0, "r_star should be positive");
        assert!(eps > 0.0, "epsilon should be positive");
    }

    #[test]
    fn test_mmff_vdw_energy() {
        let pot = MMFFVdW {
            atom_i: vec![0],
            atom_j: vec![1],
            r_star: vec![1.94],
            epsilon: vec![0.02],
        };
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (e, forces) = pot.calc_energy_forces(&coords);
        assert!(e.is_finite());
        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-4, "dim {}: sum = {}", dim, sum);
        }
    }
}
