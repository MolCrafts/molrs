//! MMFF94 frame builder — assembles a typed [`Frame`] from a [`MolGraph`].

use std::collections::{HashMap, HashSet};

use ndarray::Array1;

use crate::block::Block;
use crate::frame::Frame;
use crate::gasteiger::compute_gasteiger_charges;
use crate::molgraph::{AtomId, MolGraph, PropValue};
use crate::rings::find_rings;
use crate::topology::Topology;
use crate::types::{F, U};

use super::atom_typing::assign_atom_types;
use super::classify::{classify_angle_type, classify_bond_type, classify_torsion_type};
use super::params::MMFFParams;

/// Build a [`Frame`] with MMFF94 type labels from a molecular graph.
///
/// The resulting frame contains:
/// - `atoms`: x, y, z, type (MMFF type as string), charge
/// - `bonds`: atomi, atomj, type (e.g. "0_1_5")
/// - `angles`: atomi, atomj, atomk, type (e.g. "0_1_2_1"), stbn_type
/// - `dihedrals`: atomi, atomj, atomk, atoml, type (e.g. "0_5_1_1_5")
/// - `impropers`: atomi, atomj, atomk, atoml, type (e.g. "1_2_3_4")
/// - `pairs`: atomi, atomj, is_14
pub(crate) fn build_mmff_frame(mol: &MolGraph, params: &MMFFParams) -> Result<Frame, String> {
    // Step 1: Ring detection
    let ring_info = find_rings(mol);

    // Step 2: Atom type assignment
    let atom_types = assign_atom_types(mol, &ring_info, params);

    // Step 3: Build stable atom ordering (AtomId → index)
    let atom_vec: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    let atom_to_idx: HashMap<AtomId, usize> = atom_vec
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let n_atoms = atom_vec.len();

    // Step 4: Build topology from edges
    let edges: Vec<[usize; 2]> = mol
        .bonds()
        .map(|(_, bond)| [atom_to_idx[&bond.atoms[0]], atom_to_idx[&bond.atoms[1]]])
        .collect();
    let topo = Topology::from_edges(n_atoms, &edges);

    // Build bond order lookup: (min_idx, max_idx) → bond_order
    let mut bond_order_map: HashMap<(usize, usize), f64> = HashMap::new();
    for (_, bond) in mol.bonds() {
        let i = atom_to_idx[&bond.atoms[0]];
        let j = atom_to_idx[&bond.atoms[1]];
        let order = match bond.props.get("order") {
            Some(PropValue::F64(v)) => *v,
            _ => 1.0,
        };
        let key = if i < j { (i, j) } else { (j, i) };
        bond_order_map.insert(key, order);
    }

    // Step 5: Build atoms block
    let mut atoms_block = Block::new();
    let mut xs = Vec::with_capacity(n_atoms);
    let mut ys = Vec::with_capacity(n_atoms);
    let mut zs = Vec::with_capacity(n_atoms);
    let mut type_labels = Vec::with_capacity(n_atoms);

    for &aid in &atom_vec {
        let atom = mol.get_atom(aid).map_err(|e| e.to_string())?;
        xs.push(atom.get_f64("x").unwrap_or(0.0) as F);
        ys.push(atom.get_f64("y").unwrap_or(0.0) as F);
        zs.push(atom.get_f64("z").unwrap_or(0.0) as F);
        let t = atom_types.get(&aid).copied().unwrap_or(0);
        type_labels.push(format!("{}", t));
    }

    atoms_block
        .insert("x", Array1::from_vec(xs).into_dyn())
        .map_err(|e| e.to_string())?;
    atoms_block
        .insert("y", Array1::from_vec(ys).into_dyn())
        .map_err(|e| e.to_string())?;
    atoms_block
        .insert("z", Array1::from_vec(zs).into_dyn())
        .map_err(|e| e.to_string())?;
    atoms_block
        .insert("type", Array1::from_vec(type_labels).into_dyn())
        .map_err(|e| e.to_string())?;

    // Gasteiger charges
    let charges_result = compute_gasteiger_charges(mol, 12);
    let charge_map: HashMap<AtomId, f64> = charges_result
        .into_iter()
        .map(|(id, gc)| (id, gc.charge))
        .collect();
    let charge_vec: Vec<F> = atom_vec
        .iter()
        .map(|aid| charge_map.get(aid).copied().unwrap_or(0.0) as F)
        .collect();
    atoms_block
        .insert("charge", Array1::from_vec(charge_vec).into_dyn())
        .map_err(|e| e.to_string())?;

    // Helper: get MMFF type for atom index
    let type_of = |idx: usize| -> u32 {
        let aid = atom_vec[idx];
        atom_types.get(&aid).copied().unwrap_or(0)
    };

    // Step 6: Build bonds block
    let topo_bonds = topo.bonds();
    let n_bonds = topo_bonds.len();
    let mut bi: Vec<U> = Vec::with_capacity(n_bonds);
    let mut bj: Vec<U> = Vec::with_capacity(n_bonds);
    let mut bond_types = Vec::with_capacity(n_bonds);
    // Cache bond_type for each bond (indexed by (min,max) atom index)
    let mut bond_type_map: HashMap<(usize, usize), u32> = HashMap::new();

    for &[a, b] in &topo_bonds {
        let t1 = type_of(a);
        let t2 = type_of(b);
        let key = if a < b { (a, b) } else { (b, a) };
        let order = bond_order_map.get(&key).copied().unwrap_or(1.0);
        let bt = classify_bond_type(t1, t2, order, params);
        bond_type_map.insert(key, bt);

        let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        bi.push(a as U);
        bj.push(b as U);
        bond_types.push(format!("{}_{}_{}", bt, lo, hi));
    }

    let mut bonds_block = Block::new();
    bonds_block
        .insert("atomi", Array1::from_vec(bi).into_dyn())
        .map_err(|e| e.to_string())?;
    bonds_block
        .insert("atomj", Array1::from_vec(bj).into_dyn())
        .map_err(|e| e.to_string())?;
    bonds_block
        .insert("type", Array1::from_vec(bond_types).into_dyn())
        .map_err(|e| e.to_string())?;

    // Helper to look up bond type between two atom indices
    let get_bt = |a: usize, b: usize| -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        bond_type_map.get(&key).copied().unwrap_or(0)
    };

    // Step 7: Build angles block
    let topo_angles = topo.angles();
    let n_angles = topo_angles.len();
    let mut ai: Vec<U> = Vec::with_capacity(n_angles);
    let mut aj: Vec<U> = Vec::with_capacity(n_angles);
    let mut ak: Vec<U> = Vec::with_capacity(n_angles);
    let mut angle_types = Vec::with_capacity(n_angles);
    let mut stbn_types = Vec::with_capacity(n_angles);

    for &[a, b, c] in &topo_angles {
        let t1 = type_of(a);
        let t2 = type_of(b);
        let t3 = type_of(c);
        let bt_ab = get_bt(a, b);
        let bt_bc = get_bt(b, c);
        let at = classify_angle_type(bt_ab, bt_bc);

        ai.push(a as U);
        aj.push(b as U);
        ak.push(c as U);
        angle_types.push(format!("{}_{}_{}_{}", at, t1, t2, t3));
        stbn_types.push(format!("{}_{}_{}_{}", at, t1, t2, t3));
    }

    let mut angles_block = Block::new();
    angles_block
        .insert("atomi", Array1::from_vec(ai).into_dyn())
        .map_err(|e| e.to_string())?;
    angles_block
        .insert("atomj", Array1::from_vec(aj).into_dyn())
        .map_err(|e| e.to_string())?;
    angles_block
        .insert("atomk", Array1::from_vec(ak).into_dyn())
        .map_err(|e| e.to_string())?;
    angles_block
        .insert("type", Array1::from_vec(angle_types).into_dyn())
        .map_err(|e| e.to_string())?;
    angles_block
        .insert("stbn_type", Array1::from_vec(stbn_types).into_dyn())
        .map_err(|e| e.to_string())?;

    // Step 8: Build dihedrals block
    let topo_dihedrals = topo.dihedrals();
    let n_dihedrals = topo_dihedrals.len();
    let mut di: Vec<U> = Vec::with_capacity(n_dihedrals);
    let mut dj: Vec<U> = Vec::with_capacity(n_dihedrals);
    let mut dk: Vec<U> = Vec::with_capacity(n_dihedrals);
    let mut dl: Vec<U> = Vec::with_capacity(n_dihedrals);
    let mut dih_types = Vec::with_capacity(n_dihedrals);

    for &[a, b, c, d] in &topo_dihedrals {
        let t1 = type_of(a);
        let t2 = type_of(b);
        let t3 = type_of(c);
        let t4 = type_of(d);
        let bt_ab = get_bt(a, b);
        let bt_bc = get_bt(b, c);
        let bt_cd = get_bt(c, d);
        let tt = classify_torsion_type(bt_ab, bt_bc, bt_cd);

        di.push(a as U);
        dj.push(b as U);
        dk.push(c as U);
        dl.push(d as U);
        dih_types.push(format!("{}_{}_{}_{}_{}", tt, t1, t2, t3, t4));
    }

    let mut dihedrals_block = Block::new();
    dihedrals_block
        .insert("atomi", Array1::from_vec(di).into_dyn())
        .map_err(|e| e.to_string())?;
    dihedrals_block
        .insert("atomj", Array1::from_vec(dj).into_dyn())
        .map_err(|e| e.to_string())?;
    dihedrals_block
        .insert("atomk", Array1::from_vec(dk).into_dyn())
        .map_err(|e| e.to_string())?;
    dihedrals_block
        .insert("atoml", Array1::from_vec(dl).into_dyn())
        .map_err(|e| e.to_string())?;
    dihedrals_block
        .insert("type", Array1::from_vec(dih_types).into_dyn())
        .map_err(|e| e.to_string())?;

    // Step 9: Build impropers block
    let topo_impropers = topo.impropers();
    let n_impropers = topo_impropers.len();
    let mut ii: Vec<U> = Vec::with_capacity(n_impropers);
    let mut ij: Vec<U> = Vec::with_capacity(n_impropers);
    let mut ik: Vec<U> = Vec::with_capacity(n_impropers);
    let mut il: Vec<U> = Vec::with_capacity(n_impropers);
    let mut imp_types = Vec::with_capacity(n_impropers);

    for &[center, a, b, c] in &topo_impropers {
        let tc = type_of(center);
        let ta = type_of(a);
        let tb = type_of(b);
        let tcc = type_of(c);

        ii.push(center as U);
        ij.push(a as U);
        ik.push(b as U);
        il.push(c as U);
        imp_types.push(format!("{}_{}_{}_{}", tc, ta, tb, tcc));
    }

    let mut impropers_block = Block::new();
    if n_impropers > 0 {
        impropers_block
            .insert("atomi", Array1::from_vec(ii).into_dyn())
            .map_err(|e| e.to_string())?;
        impropers_block
            .insert("atomj", Array1::from_vec(ij).into_dyn())
            .map_err(|e| e.to_string())?;
        impropers_block
            .insert("atomk", Array1::from_vec(ik).into_dyn())
            .map_err(|e| e.to_string())?;
        impropers_block
            .insert("atoml", Array1::from_vec(il).into_dyn())
            .map_err(|e| e.to_string())?;
        impropers_block
            .insert("type", Array1::from_vec(imp_types).into_dyn())
            .map_err(|e| e.to_string())?;
    }

    // Step 10: Build non-bonded pairs (exclude 1-2 and 1-3, flag 1-4)
    let excluded_12: HashSet<(usize, usize)> = topo_bonds
        .iter()
        .flat_map(|&[a, b]| {
            let key = if a < b { (a, b) } else { (b, a) };
            std::iter::once(key)
        })
        .collect();

    let excluded_13: HashSet<(usize, usize)> = topo_angles
        .iter()
        .flat_map(|&[a, _, c]| {
            let key = if a < c { (a, c) } else { (c, a) };
            std::iter::once(key)
        })
        .collect();

    let set_14: HashSet<(usize, usize)> = topo_dihedrals
        .iter()
        .flat_map(|&[a, _, _, d]| {
            let key = if a < d { (a, d) } else { (d, a) };
            std::iter::once(key)
        })
        .collect();

    let mut pi: Vec<U> = Vec::new();
    let mut pj: Vec<U> = Vec::new();
    let mut p14 = Vec::new();

    for a in 0..n_atoms {
        for b in (a + 1)..n_atoms {
            let key = (a, b);
            if excluded_12.contains(&key) || excluded_13.contains(&key) {
                continue;
            }
            pi.push(a as U);
            pj.push(b as U);
            p14.push(set_14.contains(&key));
        }
    }

    let mut pairs_block = Block::new();
    if !pi.is_empty() {
        pairs_block
            .insert("atomi", Array1::from_vec(pi).into_dyn())
            .map_err(|e| e.to_string())?;
        pairs_block
            .insert("atomj", Array1::from_vec(pj).into_dyn())
            .map_err(|e| e.to_string())?;
        pairs_block
            .insert("is_14", Array1::from_vec(p14).into_dyn())
            .map_err(|e| e.to_string())?;
    }

    // Assemble frame
    let mut frame = Frame::new();
    frame.insert("atoms", atoms_block);
    frame.insert("bonds", bonds_block);
    frame.insert("angles", angles_block);
    frame.insert("dihedrals", dihedrals_block);
    if n_impropers > 0 {
        frame.insert("impropers", impropers_block);
    }
    if !pairs_block.is_empty() {
        frame.insert("pairs", pairs_block);
    }

    Ok(frame)
}
