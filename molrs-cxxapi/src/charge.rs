//! Partial charges for the C++ engine — the CXX face of `molrs::ff::charge`
//! (AM1-BCC corrections over engine-supplied AM1 base charges).

use molrs::core::keys;
use molrs::ff::charge::{BccParameterSet, ChargeModel};
use ndarray::Array1;

use crate::frame::{FrameRef, with_block_inserted_res};

/// `molrs::ff::charge::BccModel` across the bridge: AM1-BCC / ABCG2 bond-charge
/// corrections over AM1 base charges the engine supplies.
///
/// All BCC atom typing and correction lookup stays in molrs. Both methods
/// write the result into the frame's `atoms.charge` column and return it;
/// other frame blocks and non-charge atom columns are left untouched — in
/// particular the caller's `atoms.type` column survives, because the model
/// keeps its BCC codes to itself.
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
pub struct BccModel(molrs::ff::charge::BccModel);

/// `BccModel::new` for the correction family named `parameter_set`, `"bcc"`
/// or `"abcg2"` (see [`BccParameterSet::from_name`]). A name molrs does not
/// know is an error, never a default.
pub(crate) fn bcc_model_new(parameter_set: &str) -> Result<Box<BccModel>, String> {
    let set = BccParameterSet::from_name(parameter_set)?;
    Ok(Box::new(BccModel(molrs::ff::charge::BccModel::new(set))))
}

impl BccModel {
    /// `BccModel::assign`: the whole model — the AM1 charges are averaged over
    /// the topological-equivalence classes, then corrected.
    pub(crate) fn assign(
        &self,
        fref: &mut FrameRef,
        am1_charges: &[f64],
    ) -> Result<Vec<f64>, String> {
        self.write_charges(fref, |mol| self.0.assign(mol, Some(am1_charges)))
    }

    /// `BccModel::correct`: the correction stage alone, for base charges that
    /// are already equivalenced.
    pub(crate) fn correct(
        &self,
        fref: &mut FrameRef,
        am1_charges: &[f64],
    ) -> Result<Vec<f64>, String> {
        self.write_charges(fref, |mol| self.0.correct(mol, am1_charges))
    }

    fn write_charges(
        &self,
        fref: &mut FrameRef,
        charges_of: impl FnOnce(
            &molrs::core::Atomistic,
        ) -> Result<Vec<f64>, molrs::ff::charge::ChargeError>,
    ) -> Result<Vec<f64>, String> {
        fref.0
            .with_mut(|frame| -> Result<Vec<f64>, String> {
                let mol = molrs::core::Atomistic::from_frame(frame).map_err(|e| e.to_string())?;
                let charges = charges_of(&mol).map_err(|e| e.to_string())?;
                with_block_inserted_res(frame, "atoms", |blk| {
                    blk.insert(keys::CHARGE, Array1::from_vec(charges.clone()).into_dyn())
                        .map_err(|e| format!("BccModel: insert charge: {e}"))
                })?;
                Ok(charges)
            })
            .map_err(|e| e.to_string())?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{frame_column_f64, frame_new};

    #[test]
    fn bcc_model_correct_writes_the_corrected_charges_into_the_frame() {
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
        let charges = bcc_model_new("bcc")
            .expect("bcc is a parameter set")
            .correct(
                &mut fref,
                &[-0.266000, 0.066000, 0.066000, 0.066000, 0.066000],
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

    #[test]
    fn bcc_model_assign_equivalences_then_corrects() {
        let mut mol = molrs::core::Atomistic::new();
        let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        for [x, y, z] in [
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ] {
            let h = mol.add_atom_xyz("H", x, y, z);
            mol.add_bond(c, h).expect("add C-H");
        }
        let mut fref = frame_new();
        fref.0
            .with_mut(|frame| *frame = mol.to_frame().expect("methane converts"))
            .unwrap();
        // Unequal hydrogens average to 0.066 before the correction, so the
        // result is the one `correct` gives on equivalenced input.
        let charges = bcc_model_new("bcc")
            .unwrap()
            .assign(&mut fref, &[-0.266, 0.056, 0.076, 0.066, 0.066])
            .expect("methane is covered by the BCC table");
        let expected = [-0.1088, 0.0267, 0.0267, 0.0267, 0.0267];
        for (actual, expected) in charges.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-12,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn bcc_model_new_refuses_an_unknown_parameter_set() {
        assert!(bcc_model_new("am1").is_err());
    }
}
