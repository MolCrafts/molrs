//! Partial charges for the C++ engine — the CXX face of `molrs::ff::charge`
//! (AM1-BCC corrections over engine-supplied AM1 base charges).

use molrs::core::keys;
use molrs::ff::charge::{BccModel, BccParameterSet};
use ndarray::Array1;

use crate::frame::{FrameRef, with_block_inserted_res};

/// Apply AM1-BCC corrections to a molrs frame using AM1 base
/// charges supplied by Atomiverse.
///
/// All BCC atom typing and correction lookup stays in molrs. The bridge hands the
/// engine's AM1 base charges to `BccModel::correct(&mol, &am1)` — a pure function
/// that returns the corrected charges without touching the molecule — and writes
/// them into the frame's `atoms.charge` column. Other frame blocks and non-charge
/// atom columns are left untouched; in particular the caller's `atoms.type` column
/// survives, because the model keeps its BCC codes to itself.
///
/// The correction family is chosen by name (`parameter_set`), because a name is
/// what this bridge already carries for every other choice it offers (`block`,
/// `col`, `path`) and because an unrecognized one then has somewhere to go: the
/// `Err` arm. Both families molrs ships are reachable — see
/// [`BccParameterSet::from_name`].
///
/// # Errors
///
/// Every error here is the *caller's chemistry*, not a programmer bug: a molecule
/// with no BCC correction row (BF4⁻ — boron has none), a missing atom type, a
/// missing bond order, or a parameter-set name molrs does not know. They are
/// returned rather than panicked precisely so that cxx renders them as a
/// `rust::Error` C++ can catch. A panic here would cross an `extern "C"` shim and
/// abort the engine that asked.
///
/// There is no total-charge argument. AM1-BCC does not renormalize to a target:
/// the reference AM1-BCC algorithm ends at the increment loop; spreading the AM1
/// rounding residual over the atoms would make molrs *diverge* from the reference
/// while hiding a non-converged AM1 behind a plausible-looking answer.
///
/// @param fref          frame handle; `atoms.charge` is written on success
/// @param am1_charges   AM1 base charges from the engine, one per atom
/// @param parameter_set `"bcc"` or `"abcg2"`
/// @return the corrected charges, one per atom
pub(crate) fn assign_am1_bcc_charges(
    fref: &mut FrameRef,
    am1_charges: &[f64],
    parameter_set: &str,
) -> Result<Vec<f64>, String> {
    let set = BccParameterSet::from_name(parameter_set)?;

    fref.0
        .with_mut(|frame| -> Result<Vec<f64>, String> {
            let mol = molrs::core::Atomistic::from_frame(frame).map_err(|e| e.to_string())?;
            let charges = BccModel::new(set)
                .correct(&mol, am1_charges)
                .map_err(|e| e.to_string())?;

            with_block_inserted_res(frame, "atoms", |blk| {
                blk.insert(keys::CHARGE, Array1::from_vec(charges.clone()).into_dyn())
                    .map_err(|e| format!("assign_am1_bcc_charges: insert charge: {e}"))
            })?;
            Ok(charges)
        })
        .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{frame_column_f64, frame_new};

    #[test]
    fn am1_bcc_bridge_applies_molrs_typifier_to_frame_from_base_charges() {
        let mut mol = molrs::core::Atomistic::new();
        let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let mut hydrogens = Vec::new();
        for [x, y, z] in [
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ] {
            let h = mol.add_atom_xyz("H", x, y, z);
            mol.add_bond(c, h).expect("add C-H");
            hydrogens.push(h);
        }

        let mut fref = frame_new();
        fref.0
            .with_mut(|frame| *frame = mol.to_frame().expect("methane converts"))
            .unwrap();
        let charges = assign_am1_bcc_charges(
            &mut fref,
            &[-0.266000, 0.066000, 0.066000, 0.066000, 0.066000],
            "bcc",
        )
        .expect("methane is covered by the BCC table");
        let expected = [-0.1088, 0.0267, 0.0267, 0.0267, 0.0267];
        for (actual, expected) in charges.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-12,
                "{actual} != {expected}"
            );
        }

        let written = frame_column_f64(&fref, "atoms", keys::CHARGE);
        assert_eq!(written.len(), charges.len());
        for (actual, expected) in written.iter().zip(charges) {
            assert!(
                (actual - expected).abs() <= 1.0e-12,
                "{actual} != {expected}"
            );
        }
    }
}
