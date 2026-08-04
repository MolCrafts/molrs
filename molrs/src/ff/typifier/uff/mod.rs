//! Universal Force Field typifier.
//!
//! Assigns RDKit-style UFF atom labels, generates bond/angle/dihedral topology,
//! and bakes per-instance force constants onto the graph so
//! [`ForceField::to_potentials`](crate::ff::forcefield::ForceField::to_potentials)
//! can compile `uff_bond` / `uff_angle` / `uff_torsion` / `uff_lj` kernels.
//!
//! # Route
//!
//! ```ignore
//! let t = UFFTypifier::new();
//! let mut frame = t.typify(&mol)?.to_frame();
//! frame.insert("pairs", intramolecular_pairs(&frame));
//! let pots = t.ff().to_potentials(&frame)?;
//! ```
//!
//! Organic / main-group subset only (see [`crate::ff::params::uff`]). No GFN-FF.

use std::collections::HashMap;

use molrs::system::molgraph::PropValue;
use molrs::{AtomId, Atomistic, Element};

use crate::ff::forcefield::{ForceField, SpecialBonds};
use crate::ff::params::uff::{AMIDE_BOND_ORDER, AtomicParams, G, LAMBDA, params_for_label};
use crate::ff::typifier::Typifier;
use crate::perceive::rings::find_rings;

/// Universal Force Field typifier (Rappé 1992, RDKit-aligned).
pub struct UFFTypifier {
    ff: ForceField,
}

impl Default for UFFTypifier {
    fn default() -> Self {
        Self::new()
    }
}

impl UFFTypifier {
    /// Infallible: parameters are compile-time constants.
    pub fn new() -> Self {
        let mut ff = ForceField::new("UFF");
        ff.def_bondstyle("uff_bond");
        ff.def_anglestyle("uff_angle");
        ff.def_dihedralstyle("uff_torsion");
        ff.def_improperstyle("uff_inversion");
        ff.def_pairstyle("uff_lj", &[]);
        // UFF has no electrostatics; 1-4 LJ at full strength (RDKit).
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 1.0],
            coul: [0.0, 0.0, 0.0],
        });
        Self { ff }
    }

    pub fn ff(&self) -> &ForceField {
        &self.ff
    }

    /// Label + bake per-instance UFF parameters.
    pub fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        let mut out = mol.clone();
        out.generate_topology(true, true, true)
            .map_err(|e| e.to_string())?;

        let atom_ids: Vec<AtomId> = out.atoms().map(|(id, _)| id).collect();
        let id_to_idx: HashMap<AtomId, usize> = atom_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        let n = atom_ids.len();

        // Neighbours + bond orders
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        let mut bond_order: HashMap<(usize, usize), f64> = HashMap::new();
        for (i, &aid) in atom_ids.iter().enumerate() {
            for (nid, bid) in out.neighbor_bonds(aid) {
                if let Some(&j) = id_to_idx.get(&nid) {
                    adj[i].push(j);
                    // UFF's bond-order term is the localized count; an
                    // aromatic bond is conventionally 1.5 *here*, as a UFF
                    // parameter, not as a molrs bond order.
                    let ord = if out.bond_type(bid).is_aromatic() {
                        1.5
                    } else {
                        out.bond_number(bid).count().max(1) as f64
                    };
                    bond_order.insert(if i < j { (i, j) } else { (j, i) }, ord);
                }
            }
            adj[i].sort_unstable();
            adj[i].dedup();
        }

        let rings = find_rings(&out);
        let mut aromatic_atom = vec![false; n];
        for ring in rings.rings() {
            if ring.len() == 5 || ring.len() == 6 {
                // crude: if all C/N/O and max order suggests aromaticity from 1.5 bonds
                let idxs: Vec<usize> = ring
                    .iter()
                    .filter_map(|id| id_to_idx.get(id).copied())
                    .collect();
                let mut aromatic_like = true;
                for &i in &idxs {
                    let el = element_of(&out, atom_ids[i]);
                    if !matches!(el.symbol(), "C" | "N" | "O" | "S" | "B") {
                        aromatic_like = false;
                        break;
                    }
                }
                if aromatic_like {
                    for &i in &idxs {
                        // mark if any bond order ≈ 1.5 in ring
                        for &j in &adj[i] {
                            if idxs.contains(&j) {
                                let o = bond_order
                                    .get(&(i.min(j), i.max(j)))
                                    .copied()
                                    .unwrap_or(1.0);
                                if (o - 1.5).abs() < 0.1 {
                                    aromatic_atom[i] = true;
                                    aromatic_atom[j] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Also honor is_aromatic property if present
        for (i, &aid) in atom_ids.iter().enumerate() {
            if let Ok(atom) = out.get_atom(aid)
                && atom.get_f64("is_aromatic").unwrap_or(0.0) > 0.5
            {
                aromatic_atom[i] = true;
            }
        }

        // Hybridization + labels
        let mut labels: Vec<String> = Vec::with_capacity(n);
        let mut params: Vec<&'static AtomicParams> = Vec::with_capacity(n);
        let mut hyb: Vec<Hyb> = Vec::with_capacity(n);
        let mut z: Vec<u8> = Vec::with_capacity(n);

        for (i, &aid) in atom_ids.iter().enumerate() {
            let el = element_of(&out, aid);
            z.push(el.z());
            let degree = adj[i].len();
            let mut max_order = 1.0_f64;
            let mut valence = 0.0_f64;
            for &j in &adj[i] {
                let o = bond_order
                    .get(&(i.min(j), i.max(j)))
                    .copied()
                    .unwrap_or(1.0);
                max_order = max_order.max(o);
                valence += o;
            }
            let h = classify_hyb(el.symbol(), degree, max_order, aromatic_atom[i]);
            hyb.push(h);
            let label = atom_label(el.symbol(), h, aromatic_atom[i], valence).ok_or_else(|| {
                format!("UFF: no atom type for {} (degree={degree})", el.symbol())
            })?;
            let p = params_for_label(&label)
                .ok_or_else(|| format!("UFF: parameters missing for label '{label}'"))?;
            labels.push(label.clone());
            params.push(p);
            out.set_atom(aid, "type", PropValue::Str(label))
                .map_err(|e| e.to_string())?;
            out.set_atom(aid, "x1", PropValue::F64(p.x1))
                .map_err(|e| e.to_string())?;
            out.set_atom(aid, "D1", PropValue::F64(p.d1))
                .map_err(|e| e.to_string())?;
        }

        // Bonds
        for (bid, bond) in out.bonds().collect::<Vec<_>>() {
            let (a, b) = (bond.nodes[0], bond.nodes[1]);
            let i = id_to_idx[&a];
            let j = id_to_idx[&b];
            let mut bo = bond_order
                .get(&(i.min(j), i.max(j)))
                .copied()
                .unwrap_or(1.0);
            if aromatic_atom[i] && aromatic_atom[j] && (bo - 1.5).abs() < 0.2 {
                bo = 1.5;
            }
            // amide C-N: N next to carbonyl-ish C_R/C_2 with O
            if is_amide_cn(&out, &atom_ids, &id_to_idx, &adj, &labels, i, j) {
                bo = AMIDE_BOND_ORDER;
            }
            let (r0, kb) = bond_rest_and_k(params[i], params[j], bo);
            out.set_bond_prop(bid, "kb", PropValue::F64(kb))
                .map_err(|e| e.to_string())?;
            out.set_bond_prop(bid, "r0", PropValue::F64(r0))
                .map_err(|e| e.to_string())?;
            out.set_bond_prop(
                bid,
                "type",
                PropValue::Str(format!("{}-{}", labels[i], labels[j])),
            )
            .map_err(|e| e.to_string())?;
        }

        // Angles
        for (aid, angle) in out.angles().collect::<Vec<_>>() {
            let (ai, aj, ak) = (angle.nodes[0], angle.nodes[1], angle.nodes[2]);
            let (i, j, k) = (id_to_idx[&ai], id_to_idx[&aj], id_to_idx[&ak]);
            let bo_ij = bond_order
                .get(&(i.min(j), i.max(j)))
                .copied()
                .unwrap_or(1.0);
            let bo_jk = bond_order
                .get(&(j.min(k), j.max(k)))
                .copied()
                .unwrap_or(1.0);
            let mut order = match hyb[j] {
                Hyb::Sp => 1u8,
                Hyb::Sp2 => 3,
                Hyb::Sp3D2 => 4,
                _ => 0,
            };
            // Ring hacks for sp2 (RDKit Builder) — simplified
            let in_ring3 = |aid: AtomId| rings.rings_of_size(3).iter().any(|r| r.contains(&aid));
            let in_ring4 = |aid: AtomId| rings.rings_of_size(4).iter().any(|r| r.contains(&aid));
            if hyb[j] == Hyb::Sp2 {
                if in_ring3(atom_ids[j]) {
                    order = if in_ring3(atom_ids[i]) && in_ring3(atom_ids[k]) {
                        35
                    } else {
                        30
                    };
                } else if in_ring4(atom_ids[j]) {
                    order = if in_ring4(atom_ids[i]) && in_ring4(atom_ids[k]) {
                        45
                    } else {
                        40
                    };
                }
            }
            let (theta0, order_eff) = match order {
                30 => (150.0_f64.to_radians(), 0u8),
                35 => (60.0_f64.to_radians(), 0),
                40 => (135.0_f64.to_radians(), 0),
                45 => (90.0_f64.to_radians(), 0),
                o => (params[j].theta0_rad(), o),
            };
            let ka = angle_force_constant(theta0, bo_ij, bo_jk, params[i], params[j], params[k]);
            let (c0, c1, c2) = if order_eff == 0 {
                fourier_coeffs(theta0)
            } else {
                (0.0, 0.0, 0.0)
            };
            out.set_angle_prop(aid, "ka", PropValue::F64(ka))
                .map_err(|e| e.to_string())?;
            out.set_angle_prop(aid, "order", PropValue::F64(f64::from(order_eff)))
                .map_err(|e| e.to_string())?;
            out.set_angle_prop(aid, "c0", PropValue::F64(c0))
                .map_err(|e| e.to_string())?;
            out.set_angle_prop(aid, "c1", PropValue::F64(c1))
                .map_err(|e| e.to_string())?;
            out.set_angle_prop(aid, "c2", PropValue::F64(c2))
                .map_err(|e| e.to_string())?;
            out.set_angle_prop(aid, "theta0", PropValue::F64(theta0))
                .map_err(|e| e.to_string())?;
        }

        // Dihedrals — scale V by multiplicity about the central bond
        let mut dih_list: Vec<(molrs::system::atomistic::DihedralId, [usize; 4])> = Vec::new();
        for (did, dih) in out.dihedrals() {
            let idxs = [
                id_to_idx[&dih.nodes[0]],
                id_to_idx[&dih.nodes[1]],
                id_to_idx[&dih.nodes[2]],
                id_to_idx[&dih.nodes[3]],
            ];
            dih_list.push((did, idxs));
        }
        let mut bond_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for &(_, idx) in &dih_list {
            let key = (idx[1].min(idx[2]), idx[1].max(idx[2]));
            *bond_counts.entry(key).or_insert(0) += 1;
        }
        for (did, idx) in dih_list {
            let (i, j, k, l) = (idx[0], idx[1], idx[2], idx[3]);
            // Only SP2/SP3 central atoms (RDKit)
            if !matches!(hyb[j], Hyb::Sp2 | Hyb::Sp3) || !matches!(hyb[k], Hyb::Sp2 | Hyb::Sp3) {
                // leave zeros — energy 0
                out.set_dihedral_prop(did, "V", PropValue::F64(0.0))
                    .map_err(|e| e.to_string())?;
                out.set_dihedral_prop(did, "order", PropValue::F64(3.0))
                    .map_err(|e| e.to_string())?;
                out.set_dihedral_prop(did, "cosTerm", PropValue::F64(-1.0))
                    .map_err(|e| e.to_string())?;
                continue;
            }
            let bo = bond_order
                .get(&(j.min(k), j.max(k)))
                .copied()
                .unwrap_or(1.0);
            let end_sp2 = hyb[i] == Hyb::Sp2 || hyb[l] == Hyb::Sp2;
            let (mut v, order, cos_term) = torsion_params(
                bo, z[j], z[k], hyb[j], hyb[k], params[j], params[k], end_sp2,
            );
            let mult = bond_counts
                .get(&(j.min(k), j.max(k)))
                .copied()
                .unwrap_or(1)
                .max(1) as f64;
            v /= mult;
            out.set_dihedral_prop(did, "V", PropValue::F64(v))
                .map_err(|e| e.to_string())?;
            out.set_dihedral_prop(did, "order", PropValue::F64(f64::from(order)))
                .map_err(|e| e.to_string())?;
            out.set_dihedral_prop(did, "cosTerm", PropValue::F64(cos_term))
                .map_err(|e| e.to_string())?;
        }

        // Inversions (RDKit Tools::addInversions) — three Wilson rows per centre.
        for j in 0..n {
            if adj[j].len() != 3 {
                continue;
            }
            let zc = z[j];
            let eligible = matches!(zc, 6 | 7 | 8 | 15 | 33 | 51 | 83);
            if !eligible {
                continue;
            }
            if matches!(zc, 6..=8) && hyb[j] != Hyb::Sp2 {
                continue;
            }
            let nbrs = [adj[j][0], adj[j][1], adj[j][2]];
            let is_c_bound_to_sp2_o =
                zc == 6 && nbrs.iter().any(|&o| z[o] == 8 && hyb[o] == Hyb::Sp2);
            let (k_inv, c0, c1, c2) = inversion_coeffs(zc, is_c_bound_to_sp2_o);
            // three permutations: centre j, outer atoms in RDKit order
            let perms = [
                (nbrs[0], nbrs[1], nbrs[2]),
                (nbrs[0], nbrs[2], nbrs[1]),
                (nbrs[1], nbrs[2], nbrs[0]),
            ];
            for (a, b, c) in perms {
                let iid = out
                    .add_improper(atom_ids[a], atom_ids[j], atom_ids[b], atom_ids[c])
                    .map_err(|e| e.to_string())?;
                out.set_improper_prop(iid, "K", PropValue::F64(k_inv))
                    .map_err(|e| e.to_string())?;
                out.set_improper_prop(iid, "c0", PropValue::F64(c0))
                    .map_err(|e| e.to_string())?;
                out.set_improper_prop(iid, "c1", PropValue::F64(c1))
                    .map_err(|e| e.to_string())?;
                out.set_improper_prop(iid, "c2", PropValue::F64(c2))
                    .map_err(|e| e.to_string())?;
            }
        }

        Ok(out)
    }
}

impl Typifier for UFFTypifier {
    type Mol = Atomistic;
    fn typify(&self, mol: &Self::Mol) -> Result<Self::Mol, String> {
        self.typify(mol)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Hyb {
    Sp,
    Sp2,
    Sp3,
    Sp3D2,
    Other,
}

fn element_of(mol: &Atomistic, id: AtomId) -> Element {
    mol.get_atom(id)
        .ok()
        .and_then(|a| a.get_str("element").and_then(Element::by_symbol))
        .unwrap_or(Element::C)
}

fn classify_hyb(sym: &str, degree: usize, max_order: f64, aromatic: bool) -> Hyb {
    // Halogens / alkali-like terminals: no hybridization suffix in RDKit.
    if matches!(sym, "H" | "F" | "Cl" | "Br" | "I" | "At") {
        return Hyb::Other;
    }
    if aromatic {
        return Hyb::Sp2;
    }
    // Hypervalent / high coordination (SP3D / SP3D2 heuristics).
    if degree >= 6 {
        return Hyb::Sp3D2;
    }
    if degree == 5 {
        return Hyb::Sp3D2; // treated as '5'/'6' by label builder
    }
    if max_order >= 2.9 {
        return Hyb::Sp;
    }
    if max_order >= 1.9 || (degree == 3 && max_order > 1.4) {
        return Hyb::Sp2;
    }
    if degree <= 4 {
        return Hyb::Sp3;
    }
    Hyb::Other
}

/// RDKit `Tools::getAtomLabel` + `addAtomChargeFlags` (full default UFF set).
fn atom_label(sym: &str, h: Hyb, aromatic: bool, valence: f64) -> Option<String> {
    let z = Element::by_symbol(sym)?.z();
    let mut key = sym.to_string();
    if key.len() == 1 {
        key.push('_');
    }

    // No hybridization on alkali metals (group 1) or halogens (group 7).
    let n_outer = match z {
        1 | 3 | 11 | 19 | 37 | 55 | 87 => 1u8, // alkali
        9 | 17 | 35 | 53 | 85 => 7u8,          // halogen
        _ => 0u8,
    };
    let skip_hyb = n_outer == 1 || n_outer == 7 || z == 0;

    if !skip_hyb {
        // Main-group force SP3 suffix (RDKit cases 12–15, 50–52, 81–84).
        if matches!(z, 12 | 13 | 14 | 15 | 50 | 51 | 52 | 81 | 82 | 83 | 84) {
            key.push('3');
        } else if z == 80 {
            // Hg → Hg1
            key.push('1');
        } else {
            match h {
                Hyb::Sp => key.push('1'),
                Hyb::Sp2 => {
                    if aromatic && matches!(z, 6 | 7 | 8 | 16) {
                        key.push('R');
                    } else {
                        key.push('2');
                    }
                }
                Hyb::Sp3 => key.push('3'),
                Hyb::Sp3D2 if adj_degree_hint(valence) >= 6 => key.push('6'),
                Hyb::Sp3D2 => key.push('5'),
                Hyb::Other => {}
            }
        }
    }

    add_charge_flags(&mut key, z, valence, h);
    Some(key)
}

fn adj_degree_hint(valence: f64) -> usize {
    valence.round().max(0.0) as usize
}

/// RDKit `addAtomChargeFlags` (tolerateChargeMismatch = true for robustness).
fn add_charge_flags(key: &mut String, z: u8, valence: f64, hyb: Hyb) {
    let v = valence.round() as i32;
    let push = |k: &mut String, s: &str| k.push_str(s);

    match z {
        // only +1
        29 | 47 => push(key, "+1"),
        // only +2
        4 | 20 | 25 | 26 | 28 | 46 | 78 => push(key, "+2"),
        // only +3
        21 | 24 | 27 | 79 | 89 | 96..=103 => push(key, "+3"),
        // only +4
        2 | 18 | 22 | 36 | 54 | 90..=95 => push(key, "+4"),
        // only +5
        23 | 41 | 43 | 73 => push(key, "+5"),
        // only +6
        42 => push(key, "+6"),
        12 => push(key, "+2"), // Mg
        15 => {
            // P
            if v <= 3 {
                push(key, "+3");
            } else {
                push(key, "+5");
            }
        }
        16 => {
            // S — skip charge flag for SP2 (S_2 / S_R)
            if hyb != Hyb::Sp2 {
                match v {
                    2 => push(key, "+2"),
                    4 => push(key, "+4"),
                    _ => push(key, "+6"),
                }
            }
        }
        30 => push(key, "+2"), // Zn
        31 => push(key, "+3"), // Ga
        33 => push(key, "+3"), // As
        34 => push(key, "+2"), // Se
        48 => push(key, "+2"), // Cd
        49 => push(key, "+3"), // In
        51 => push(key, "+3"), // Sb
        52 => push(key, "+2"), // Te
        75 => {
            // Re special
            if key.starts_with("Re6") {
                *key = "Re6+5".into();
            } else if key.starts_with("Re3") {
                *key = "Re3+7".into();
            }
        }
        80 => push(key, "+2"),      // Hg
        81 => push(key, "+3"),      // Tl
        82 => push(key, "+3"),      // Pb — RDKit uses +3 default
        83 => push(key, "+3"),      // Bi
        84 => push(key, "+2"),      // Po
        57..=71 => push(key, "+3"), // lanthanides
        _ => {}
    }
}

fn inversion_coeffs(at2_z: u8, is_c_bound_to_o: bool) -> (f64, f64, f64, f64) {
    // RDKit Utils::calcInversionCoefficientsAndForceConstant
    if matches!(at2_z, 6..=8) {
        let res = if is_c_bound_to_o { 50.0 } else { 6.0 } / 3.0;
        return (res, 1.0, -1.0, 0.0);
    }
    let w0_deg: f64 = match at2_z {
        15 => 84.4339,
        33 => 86.9735,
        51 => 87.7047,
        83 => 90.0,
        _ => 90.0,
    };
    let w0 = w0_deg.to_radians();
    let c2: f64 = 1.0;
    let c1 = -4.0 * w0.cos();
    let c0 = -(c1 * w0.cos() + c2 * (2.0 * w0).cos());
    let res = 22.0 / (c0 + c1 + c2) / 3.0;
    (res, c0, c1, c2)
}

fn bond_rest_and_k(p1: &AtomicParams, p2: &AtomicParams, bond_order: f64) -> (f64, f64) {
    let bo = bond_order.max(1e-6);
    let (ri, rj) = (p1.r1, p2.r1);
    let r_bo = -LAMBDA * (ri + rj) * bo.ln();
    let (xi, xj) = (p1.xi, p2.xi);
    let dx = xi.sqrt() - xj.sqrt();
    let r_en = ri * rj * dx * dx / (xi * ri + xj * rj);
    let r0 = ri + rj + r_bo - r_en;
    let kb = 2.0 * G * p1.z1 * p2.z1 / (r0 * r0 * r0);
    (r0, kb)
}

fn angle_force_constant(
    theta0: f64,
    bo12: f64,
    bo23: f64,
    p1: &AtomicParams,
    p2: &AtomicParams,
    p3: &AtomicParams,
) -> f64 {
    let cos0 = theta0.cos();
    let r12 = bond_rest_and_k(p1, p2, bo12).0;
    let r23 = bond_rest_and_k(p2, p3, bo23).0;
    let r13 = (r12 * r12 + r23 * r23 - 2.0 * r12 * r23 * cos0).sqrt();
    let beta = 2.0 * G / (r12 * r23);
    let pref = beta * p1.z1 * p3.z1 / r13.powi(5);
    let r_term = r12 * r23;
    let inner = 3.0 * r_term * (1.0 - cos0 * cos0) - r13 * r13 * cos0;
    pref * r_term * inner
}

fn fourier_coeffs(theta0: f64) -> (f64, f64, f64) {
    let sin0 = theta0.sin();
    let cos0 = theta0.cos();
    let c2 = 1.0 / (4.0 * (sin0 * sin0).max(1e-8));
    let c1 = -4.0 * c2 * cos0;
    let c0 = c2 * (2.0 * cos0 * cos0 + 1.0);
    (c0, c1, c2)
}

fn is_group6(z: u8) -> bool {
    matches!(z, 8 | 16 | 34 | 52 | 84)
}

#[allow(clippy::too_many_arguments)]
fn torsion_params(
    bo23: f64,
    z2: u8,
    z3: u8,
    h2: Hyb,
    h3: Hyb,
    p2: &AtomicParams,
    p3: &AtomicParams,
    end_sp2: bool,
) -> (f64, u8, f64) {
    if h2 == Hyb::Sp3 && h3 == Hyb::Sp3 {
        let mut v = (p2.v1 * p3.v1).sqrt();
        let mut order = 3u8;
        let mut cos_term = -1.0;
        if (bo23 - 1.0).abs() < 1e-6 && is_group6(z2) && is_group6(z3) {
            let v2: f64 = if z2 == 8 { 2.0 } else { 6.8 };
            let v3: f64 = if z3 == 8 { 2.0 } else { 6.8 };
            v = (v2 * v3).sqrt();
            order = 2;
            cos_term = -1.0;
        }
        return (v, order, cos_term);
    }
    if h2 == Hyb::Sp2 && h3 == Hyb::Sp2 {
        let v = 5.0 * (p2.u1 * p3.u1).sqrt() * (1.0 + 4.18 * bo23.ln());
        return (v, 2, 1.0);
    }
    // SP2-SP3
    let mut v = 1.0;
    let mut order = 6u8;
    let mut cos_term = 1.0;
    if (bo23 - 1.0).abs() < 1e-6 {
        if (h2 == Hyb::Sp3 && is_group6(z2) && !is_group6(z3))
            || (h3 == Hyb::Sp3 && is_group6(z3) && !is_group6(z2))
        {
            v = 5.0 * (p2.u1 * p3.u1).sqrt() * (1.0 + 4.18 * bo23.ln());
            order = 2;
            cos_term = -1.0;
        } else if end_sp2 {
            v = 2.0;
            order = 3;
            cos_term = -1.0;
        }
    }
    (v, order, cos_term)
}

fn is_amide_cn(
    mol: &Atomistic,
    atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
    adj: &[Vec<usize>],
    labels: &[String],
    i: usize,
    j: usize,
) -> bool {
    let c_idx = if labels[i].starts_with('C') && labels[j].starts_with('N') {
        i
    } else if labels[j].starts_with('C') && labels[i].starts_with('N') {
        j
    } else {
        return false;
    };
    let _ = (mol, atom_ids, id_to_idx);
    // C attached to carbonyl O
    for &o in &adj[c_idx] {
        if labels[o] == "O_2" || labels[o] == "O_1" {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::extract_coords;
    use crate::ff::potential::intramolecular_pairs;
    use molrs::system::atomistic::Atomistic;

    fn ethanol() -> Atomistic {
        // Build C-C-O + hydrogens with rough coords
        let mut m = Atomistic::new();
        let c0 = m.add_atom_xyz("C", 0.9, 0.0, 0.0);
        let c1 = m.add_atom_xyz("C", -0.5, 0.0, 0.0);
        let o = m.add_atom_xyz("O", -1.2, 0.9, 0.0);
        let _ = m.add_bond(c0, c1);
        let _ = m.add_bond(c1, o);
        // add H
        let h1 = m.add_atom_xyz("H", 1.3, 0.9, 0.0);
        let h2 = m.add_atom_xyz("H", 1.3, -0.5, 0.9);
        let h3 = m.add_atom_xyz("H", 1.3, -0.5, -0.9);
        let h4 = m.add_atom_xyz("H", -0.9, -0.9, 0.5);
        let h5 = m.add_atom_xyz("H", -0.9, -0.5, -0.9);
        let h6 = m.add_atom_xyz("H", -1.1, 1.5, -0.5);
        for h in [h1, h2, h3] {
            let _ = m.add_bond(c0, h);
        }
        for h in [h4, h5] {
            let _ = m.add_bond(c1, h);
        }
        let _ = m.add_bond(o, h6);
        m
    }

    #[test]
    fn uff_types_ethanol() {
        let t = UFFTypifier::new();
        let typed = t.typify(&ethanol()).unwrap();
        let mut c3 = 0;
        let mut o3 = 0;
        for (_, a) in typed.atoms() {
            match a.get_str("type") {
                Some("C_3") => c3 += 1,
                Some("O_3") => o3 += 1,
                _ => {}
            }
        }
        assert_eq!(c3, 2);
        assert_eq!(o3, 1);
    }

    #[test]
    fn uff_energy_finite() {
        let t = UFFTypifier::new();
        let typed = t.typify(&ethanol()).unwrap();
        let mut frame = typed.to_frame();
        frame.insert("pairs", intramolecular_pairs(&frame));
        let pots = t.ff().to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();
        let (e, f) = pots.calc_energy_forces(&coords);
        assert!(e.is_finite(), "energy={e}");
        assert!(f.iter().all(|x| x.is_finite()));
        // Distorted ethanol should have non-trivial energy
        assert!(e.abs() > 0.01);
    }

    /// Every UFF force must be −∂E/∂x, checked against the kernel's own energy.
    ///
    /// `uff_energy_finite` above asserts only that the forces are finite, which
    /// a wrong gradient passes trivially — and two did: the inversion term's
    /// normal had the wrong orientation (E is even in cosY, so the energy hid
    /// it and the force came out inverted), and the torsion projected
    /// un-normalized plane normals with a flipped sign.
    #[test]
    fn uff_forces_are_the_negative_energy_gradient() {
        let t = UFFTypifier::new();
        // Ethanol exercises torsions; formaldehyde exercises the inversion term.
        for (name, mol) in [("ethanol", ethanol()), ("formaldehyde", formaldehyde())] {
            let typed = t.typify(&mol).unwrap();
            let mut frame = typed.to_frame();
            frame.insert("pairs", intramolecular_pairs(&frame));
            let pots = t.ff().to_potentials(&frame).unwrap();
            let coords = extract_coords(&frame).unwrap();
            let (_, f) = pots.calc_energy_forces(&coords);

            let h = 1e-6;
            let mut worst = 0.0_f64;
            for i in 0..coords.len() {
                let mut plus = coords.clone();
                let mut minus = coords.clone();
                plus[i] += h;
                minus[i] -= h;
                let numeric = -(pots.calc_energy(&plus) - pots.calc_energy(&minus)) / (2.0 * h);
                worst = worst.max((f[i] - numeric).abs());
            }
            assert!(
                worst < 1e-4,
                "{name}: max|F + dE/dx| = {worst:.3e}; forces are not the energy's gradient"
            );
        }
    }

    /// Planar H2C=O — three Wilson inversion rows on the carbon, no torsions.
    fn formaldehyde() -> Atomistic {
        let mut mol = Atomistic::new();
        mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        mol.add_atom_xyz("O", 0.0, 1.22, 0.0);
        mol.add_atom_xyz("H", 0.94, -0.54, 0.0);
        mol.add_atom_xyz("H", -0.94, -0.54, 0.03);
        let ids: Vec<_> = mol.atoms().map(|(id, _)| id).collect();
        let b = mol.add_bond(ids[0], ids[1]).unwrap();
        mol.set_bond_type(b, crate::system::bond::BondType::Double)
            .unwrap();
        mol.add_bond(ids[0], ids[2]).unwrap();
        mol.add_bond(ids[0], ids[3]).unwrap();
        mol
    }

    #[test]
    fn uff_lbfgs_reduces_energy() {
        use crate::optimize::{LBFGS, Optimizer};
        use std::sync::Arc;

        let t = UFFTypifier::new();
        let mut mol = ethanol();
        // Stretch C–C
        let ids: Vec<_> = mol.atoms().map(|(id, _)| id).collect();
        mol.set_atom(ids[0], "x", PropValue::F64(1.5)).unwrap();

        let typed = t.typify(&mol).unwrap();
        let mut frame = typed.to_frame();
        frame.insert("pairs", intramolecular_pairs(&frame));
        let pots = t.ff().to_potentials(&frame).unwrap();
        let coords0 = extract_coords(&frame).unwrap();
        let (e0, _) = pots.calc_energy_forces(&coords0);

        let mut opt = LBFGS::new(Arc::new(pots), 0.5, 200, 0.2, 8);
        let report = opt.run(&mut frame).unwrap();
        eprintln!(
            "UFF minimize: e0={e0:.3} e1={:.3} steps={} fmax={:.3} conv={}",
            report.final_energy, report.n_steps, report.final_fmax, report.converged
        );
        assert!(report.final_energy < e0, "energy should drop");
        assert!(report.n_steps > 0);
    }

    #[test]
    fn bond_params_match_rdkit_ethanol() {
        let c = params_for_label("C_3").unwrap();
        let h = params_for_label("H_").unwrap();
        let o = params_for_label("O_3").unwrap();
        let (r, k) = bond_rest_and_k(c, c, 1.0);
        assert!((r - 1.514).abs() < 1e-6);
        assert!((k - 699.591798712679).abs() < 1e-6);
        let (r, k) = bond_rest_and_k(c, h, 1.0);
        assert!((r - 1.109400794877744).abs() < 1e-6);
        assert!((k - 662.1387775328197).abs() < 1e-6);
        let (r, k) = bond_rest_and_k(c, o, 1.0);
        assert!((r - 1.3938448452526835).abs() < 1e-6);
        assert!((k - 1078.4971040429152).abs() < 1e-6);
    }
}
