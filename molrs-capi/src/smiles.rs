//! `extern "C"` door for SMILES text: the C face of
//! `molrs::io::read_smiles_str`.

use std::ffi::{CStr, c_char};

use crate::error::{self, MolrsStatus, ffi_err_to_status};
use crate::handle::{MolrsFrameHandle, frame_id_to_handle};
use crate::handle_registry::lock_registry;
use crate::{ffi_try, null_check};

/// Read one molecule from SMILES text into a new frame
/// (`molrs::io::read_smiles_str`, then `Atomistic::to_frame`).
///
/// Connectivity only: hydrogens implicit in the SMILES are **not** added and
/// **no coordinates** are generated, so the frame has no `x`/`y`/`z`
/// columns. The `"atoms"` block carries `element` (string) and `mass`, plus
/// `h_count`, `formal_charge`, `isotope`, `atom_class` and `stereo` where a
/// bracket atom states them and `is_aromatic` for an aromatic atom;
/// the `"bonds"` block carries `atomi` / `atomj` (`uint64_t`, 0-based),
/// `bond_type` (0 unknown, 1 single, 2 double, 3 triple, 4 aromatic) and
/// `bond_number` (the localized Lewis order).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_read_smiles_str(const char* smiles,
///                                   MolrsFrameHandle* out);
/// ```
///
/// # Arguments
///
/// * `smiles` -- Null-terminated SMILES string (e.g. `"CCO"`). One
///   molecule: a `.`-separated set of components is refused.
/// * `out` -- On success, receives the new frame handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if either pointer is null.
/// * `MolrsStatus::Utf8Error` if `smiles` is not valid UTF-8.
/// * `MolrsStatus::ParseError` if the SMILES string is malformed or names
///   more than one molecule.
/// * `MolrsStatus::InternalError` if the parsed molecule cannot be expressed
///   as a frame (a property contradicting the Frame schema).
///
/// # Safety
///
/// * `smiles` must be a valid, null-terminated UTF-8 C string.
/// * `out` must point to a writable `MolrsFrameHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_read_smiles_str(
    smiles: *const c_char,
    out: *mut MolrsFrameHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(smiles);
        null_check!(out);
        let Ok(smiles_str) = (unsafe { CStr::from_ptr(smiles) }).to_str() else {
            error::set_last_error("SMILES string is not valid UTF-8");
            return MolrsStatus::Utf8Error;
        };
        let mol = match molrs::io::read_smiles_str(smiles_str) {
            Ok(m) => m,
            Err(e) => {
                error::set_last_error(format!("{e}"));
                return MolrsStatus::ParseError;
            }
        };
        let frame = match mol.to_frame() {
            Ok(f) => f,
            Err(e) => {
                error::set_last_error(format!("{e}"));
                return MolrsStatus::InternalError;
            }
        };

        let mut registry = lock_registry();
        let id = registry.frames.frame_new();
        if let Err(e) = registry.frames.set_frame(id, frame) {
            return ffi_err_to_status(&e);
        }
        unsafe { *out = frame_id_to_handle(id) };
        MolrsStatus::Ok
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::handle_to_frame_id;

    #[test]
    fn ethanol_has_the_documented_columns() {
        let mut out = MolrsFrameHandle { idx: 0, version: 0 };
        let status = unsafe { molrs_read_smiles_str(c"CCO".as_ptr(), &mut out) };
        assert_eq!(status, MolrsStatus::Ok);
        let frame = lock_registry()
            .frames
            .clone_frame(handle_to_frame_id(out))
            .unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(3));
        for key in ["element", "mass"] {
            assert!(atoms.contains_key(key), "atoms.{key}");
        }
        assert!(!atoms.contains_key("x"), "SMILES carries no coordinates");
        let bonds = frame.get("bonds").unwrap();
        for key in ["atomi", "atomj", "bond_type", "bond_number"] {
            assert!(bonds.contains_key(key), "bonds.{key}");
        }
    }

    #[test]
    fn two_molecules_are_a_parse_error() {
        let mut out = MolrsFrameHandle { idx: 0, version: 0 };
        let status = unsafe { molrs_read_smiles_str(c"CCO.O".as_ptr(), &mut out) };
        assert_eq!(status, MolrsStatus::ParseError);
    }
}
