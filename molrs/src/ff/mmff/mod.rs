//! MMFF94 force field support.
//!
//! In addition to the parameter [`tables`] (ported from RDKit
//! `Code/ForceField/MMFF/Params.cpp`), this module ports the MMFF94 atom
//! typing, aromaticity perception, and partial-charge model from RDKit
//! `Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp` and
//! `Code/GraphMol/Aromaticity.cpp` (BSD-3, Paolo Tosco / RDKit
//! contributors).
//!
//! Pipeline (mirrors `MMFFMolProperties`'s constructor):
//! `aromaticity::set_mmff_aromaticity` → `atomtype::assign_atom_types`
//! → `charges::compute_partial_charges`.
//!
//! ```no_run
//! use molrs::ff::mmff::{MmffMolProperties, MmffVariant};
//! # fn run(mol: &molrs::Atomistic) -> Result<(), molrs::error::MolRsError> {
//! let props = MmffMolProperties::compute(mol, MmffVariant::Mmff94)?;
//! let t = props.atom_type(0);
//! let q = props.partial_charge(0);
//! # let _ = (t, q); Ok(())
//! # }
//! ```

pub(crate) mod aromaticity;
pub(crate) mod atomtype;
pub(crate) mod charges;
pub mod energy;
mod hybrid;
pub mod tables;
pub(crate) mod topo;

pub use energy::{MmffEnergyBreakdown, MmffForceField};

use molrs::Atomistic;
use molrs::error::MolRsError;

use topo::Topo;

/// MMFF parameterization variant.
///
/// `Mmff94s` is the "static" variant (Halgren 1999). Atom typing and partial
/// charges are **identical** for both variants — MMFF94 and MMFF94s share all 95
/// atom types and every bond / angle / stretch-bend / vdW / charge parameter. The
/// two differ only in the **out-of-plane and torsion** tables, and only on
/// delocalized trivalent nitrogen (MMFF numeric types 10 `NC=O` and 40 `NC=C`):
/// 11 Oop rows and 42 Torsion rows.
///
/// Both `_S` tables ARE shipped in [`tables`] — [`tables::MMFF_OOP_S`] (117 rows)
/// and [`tables::MMFF_TOR_S`] (926 rows) — and the parameter layer
/// (`energy::params`) dispatches on this variant to read them, falling back to the
/// base table for keys the `_S` table does not re-parameterise.
///
/// The physics of the difference is the out-of-plane force constant `koop`
/// (md·Å·rad⁻²), which the improper kernel evaluates as
/// `E_oop = 0.5 · 143.9325 · koop · χ²` with χ the Wilson out-of-plane angle in
/// **radians**. `koop > 0` makes the planar centre (χ = 0) an energy *minimum*;
/// `koop < 0` makes it a *maximum*. MMFF94s raises `koop` on those nitrogens to a
/// flat `+0.015` (type 10) / `+0.030` (type 40) — i.e. it **flattens** them, which
/// is what "static" means: geometries minimized under MMFF94s match the planar
/// nitrogen seen in crystal structures, where MMFF94 reproduces the dynamic,
/// time-averaged (pyramidal) picture.
///
/// Pick a variant by picking a typifier —
/// [`MMFF94Typifier`](crate::ff::typifier::mmff::MMFF94Typifier) or
/// [`MMFF94STypifier`](crate::ff::typifier::mmff::MMFF94STypifier) — not by passing
/// this enum: it is their private field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmffVariant {
    Mmff94,
    Mmff94s,
}

/// Per-atom MMFF properties (numeric atom types + partial charges) for a
/// molecule, computed once.
#[derive(Debug, Clone)]
pub struct MmffMolProperties {
    variant: MmffVariant,
    atom_types: Vec<u8>,
    partial_charges: Vec<f64>,
    valid: bool,
}

impl MmffMolProperties {
    /// Run the full MMFF setup (aromaticity → typing → charges).
    ///
    /// Returns `Err` if any atom could not be assigned an MMFF type
    /// (e.g. an unsupported element / transition metal with no MMFF type).
    pub fn compute(mol: &Atomistic, variant: MmffVariant) -> Result<Self, MolRsError> {
        let base = Topo::build(mol).map_err(|sym| {
            MolRsError::validation(format!("MMFF: unsupported element symbol '{sym}'"))
        })?;
        let topo = aromaticity::set_mmff_aromaticity(&base);
        let atom_types = atomtype::assign_atom_types(&topo);

        // Locate the first untyped atom for a useful error message.
        if let Some(bad) = atom_types.iter().position(|&t| t == 0) {
            let z = topo.atno[bad];
            return Err(MolRsError::validation(format!(
                "MMFF: could not assign an atom type to atom index {bad} (Z={z})"
            )));
        }

        let partial_charges = charges::compute_partial_charges(&topo, &atom_types);

        Ok(Self {
            variant,
            atom_types,
            partial_charges,
            valid: true,
        })
    }

    /// The variant this was computed for.
    pub fn variant(&self) -> MmffVariant {
        self.variant
    }

    /// MMFF numeric atom type (1..=99) for atom index `i`
    /// (the index is the molecule's atom iteration order).
    pub fn atom_type(&self, i: usize) -> u8 {
        self.atom_types[i]
    }

    /// MMFF partial charge for atom index `i`.
    pub fn partial_charge(&self, i: usize) -> f64 {
        self.partial_charges[i]
    }

    /// Whether every atom received a valid MMFF type.
    pub fn is_setup_complete(&self) -> bool {
        self.valid
    }

    /// Number of atoms.
    pub fn len(&self) -> usize {
        self.atom_types.len()
    }

    /// Whether the molecule was empty.
    pub fn is_empty(&self) -> bool {
        self.atom_types.is_empty()
    }
}
