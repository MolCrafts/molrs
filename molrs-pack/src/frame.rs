//! Helpers for converting between `molrs_core::Frame` and packing inputs.

use molrs::block::Block;
use molrs::types::{F, I};
use ndarray::Array1;
use std::str::FromStr;

use crate::context::PackContext;

/// Extract atom positions, VdW radii, and element symbols from a `molrs::Frame`.
///
/// Reads the `"atoms"` block, expecting `"x"`, `"y"`, `"z"` (f32) and `"element"` (String)
/// columns. VdW radii are looked up from `molrs::Element::vdw_radius()`.
/// Unknown elements fall back to 1.5 Å with symbol `"X"`.
///
/// # Panics
/// Panics if the frame has no `"atoms"` block or no `"x"` / `"y"` / `"z"` columns.
pub fn frame_to_coords(frame: &molrs::Frame) -> (Vec<[F; 3]>, Vec<F>) {
    let (positions, radii, _) = frame_to_coords_and_elements(frame);
    (positions, radii)
}

/// Like [`frame_to_coords`] but also returns element symbols.
pub fn frame_to_coords_and_elements(frame: &molrs::Frame) -> (Vec<[F; 3]>, Vec<F>, Vec<String>) {
    let atoms = frame.get("atoms").expect("frame has no 'atoms' block");

    let positions: Vec<[F; 3]> = {
        let x = atoms.get_float("x").expect("atoms block has no 'x' column");
        let y = atoms.get_float("y").expect("atoms block has no 'y' column");
        let z = atoms.get_float("z").expect("atoms block has no 'z' column");
        x.iter()
            .zip(y.iter())
            .zip(z.iter())
            .map(|((&xi, &yi), &zi)| [xi, yi, zi])
            .collect()
    };

    let n = positions.len();

    let (radii, elements): (Vec<F>, Vec<String>) = if let Some(elems) = atoms.get_string("element")
    {
        elems
            .iter()
            .map(|sym| {
                let s = sym.trim();
                let r = molrs::Element::from_str(s)
                    .map(|e| e.vdw_radius() as F)
                    .unwrap_or(1.5);
                (r, s.to_string())
            })
            .unzip()
    } else {
        (vec![1.5; n], vec!["X".to_string(); n])
    };

    (positions, radii, elements)
}

/// Compute per-atom global molecule IDs from the structural layout.
///
/// The atom layout is: for each type, for each molecule copy, for each atom.
/// Global mol ID = cumulative molecule count of preceding types + mol index.
pub fn compute_mol_ids(sys: &PackContext) -> Vec<usize> {
    let mut ids = vec![0usize; sys.ntotat];
    let mut icart = 0usize;
    let mut mol_offset = 0usize;
    for itype in 0..sys.ntype_with_fixed {
        let nmol = sys.nmols[itype];
        let nat = sys.natoms[itype];
        for imol in 0..nmol {
            for _iatom in 0..nat {
                ids[icart] = mol_offset + imol;
                icart += 1;
            }
        }
        mol_offset += nmol;
    }
    ids
}

/// Write constant columns (element, mol_id) into `sys.frame`.
///
/// Called once after elements and topology are fully initialized.
/// The hot loop never touches the frame — only `xcart` is updated.
#[inline(never)]
pub fn init_frame_constants(sys: &mut PackContext) {
    let elem_strs: Vec<String> = sys
        .elements
        .iter()
        .map(|e| {
            e.map(|el| el.symbol().to_string())
                .unwrap_or_else(|| "X".to_string())
        })
        .collect();

    let mol_ids = compute_mol_ids(sys);
    let mol_id_int: Vec<I> = mol_ids.iter().map(|&id| id as I).collect();

    let mut atoms = Block::new();
    atoms
        .insert("element", Array1::from_vec(elem_strs).into_dyn())
        .expect("element insert");
    atoms
        .insert("mol_id", Array1::from_vec(mol_id_int).into_dyn())
        .expect("mol_id insert");

    sys.frame.insert("atoms", atoms);
}

/// Finalize the frame by writing position columns from `xcart`,
/// then move the frame out of the context (zero-copy ownership transfer).
///
/// After this call `sys.frame` is empty and `sys.xcart` is drained.
#[inline(never)]
pub fn finalize_frame(sys: &mut PackContext) -> molrs::Frame {
    let xcart = std::mem::take(&mut sys.xcart);

    let x_vals: Vec<F> = xcart.iter().map(|p| p[0]).collect();
    let y_vals: Vec<F> = xcart.iter().map(|p| p[1]).collect();
    let z_vals: Vec<F> = xcart.iter().map(|p| p[2]).collect();
    let n = x_vals.len();

    // Get the existing atoms block (has element + mol_id), add position columns
    let atoms = sys
        .frame
        .get_mut("atoms")
        .expect("frame missing atoms block");
    atoms
        .insert("x", Array1::from_vec(x_vals).into_dyn())
        .expect("x insert");
    atoms
        .insert("y", Array1::from_vec(y_vals).into_dyn())
        .expect("y insert");
    atoms
        .insert("z", Array1::from_vec(z_vals).into_dyn())
        .expect("z insert");

    let mut frame = std::mem::take(&mut sys.frame);
    frame.meta.insert("natoms".to_string(), n.to_string());
    frame
}

/// Build a `molrs::Frame` from a [`PackContext`] (one-shot convenience wrapper).
pub fn context_to_frame(sys: &PackContext) -> molrs::Frame {
    let n = sys.xcart.len();

    let x_vals: Vec<F> = sys.xcart.iter().map(|p| p[0]).collect();
    let y_vals: Vec<F> = sys.xcart.iter().map(|p| p[1]).collect();
    let z_vals: Vec<F> = sys.xcart.iter().map(|p| p[2]).collect();

    let elem_strs: Vec<String> = sys
        .elements
        .iter()
        .map(|e| {
            e.map(|el| el.symbol().to_string())
                .unwrap_or_else(|| "X".to_string())
        })
        .collect();

    let mol_ids = compute_mol_ids(sys);
    let mol_id_int: Vec<I> = mol_ids.iter().map(|&id| id as I).collect();

    let mut atoms = Block::new();
    atoms
        .insert("x", Array1::from_vec(x_vals).into_dyn())
        .expect("x insert");
    atoms
        .insert("y", Array1::from_vec(y_vals).into_dyn())
        .expect("y insert");
    atoms
        .insert("z", Array1::from_vec(z_vals).into_dyn())
        .expect("z insert");
    atoms
        .insert("element", Array1::from_vec(elem_strs).into_dyn())
        .expect("element insert");
    atoms
        .insert("mol_id", Array1::from_vec(mol_id_int).into_dyn())
        .expect("mol_id insert");

    let mut frame = molrs::Frame::new();
    frame.insert("atoms", atoms);
    frame.meta.insert("natoms".to_string(), n.to_string());
    frame
}
