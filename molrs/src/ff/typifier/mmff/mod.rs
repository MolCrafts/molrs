//! MMFF atom/bond/angle/torsion/improper typifiers — two named front doors.
//!
//! Annotates an [`Atomistic`] with MMFF type labels and partial charges. That is
//! the typifier's contract, and all of it: MMFF is a parameter set plus a topology
//! labeler, and it computes energies the way every other force field in molrs does
//! — through [`ForceField::to_potentials`](crate::ff::forcefield::ForceField).
//! **MMFF is not a special case.**
//!
//! # Which door?
//!
//! | Type | Parameter set | Delocalised trivalent N |
//! |---|---|---|
//! | [`MMFF94Typifier`] | `MMFF94` | pyramidal (dynamic / time-averaged picture) |
//! | [`MMFF94STypifier`] | `MMFF94s` | **planar** (static, Halgren 1999) |
//!
//! There is **one** engine behind both; the variant is its private field. Users
//! choose a parameter set by choosing a type, never by passing a flag.
//!
//! The two parameter sets share all 95 atom types and every bond / angle /
//! stretch-bend / vdW / charge parameter. They differ in **11 out-of-plane rows**
//! and **42 torsion rows**, every one of them centred on MMFF numeric type 10
//! (`NC=O`, amide N) or 40 (`NC=C`, enamine-type N). MMFF94s ("s" = *static*)
//! raises the out-of-plane force constant `koop` on those centres to a flat
//! `+0.015` (type 10) / `+0.030` (type 40) md·Å·rad⁻², which — see
//! [`MMFF94STypifier`] — makes the planar nitrogen an energy *minimum*.
//!
//! # Example — the one route
//!
//! ```no_run
//! use molrs::Atomistic;
//! use molrs::ff::potential::intramolecular_pairs;
//! use molrs::ff::typifier::mmff::MMFF94Typifier;
//! # fn main() -> Result<(), String> {
//! let mol = Atomistic::new();                             // build or load your molecule
//! let typifier = MMFF94Typifier::new();
//!
//! let mut frame = typifier.typify(&mol)?.to_frame();      // labels + charges
//! frame.insert("pairs", intramolecular_pairs(&frame));    // the consumer's neighbour list
//! let potentials = typifier.ff().to_potentials(&frame)?;  // the standard compile path
//!
//! let coords: Vec<f64> = Vec::new();                      // flat [x,y,z, ...]
//! let (energy, _forces) = potentials.calc_energy_forces(&coords);
//! println!("MMFF94 energy = {energy} kcal/mol");
//! # Ok(())
//! # }
//! ```
//!
//! The `pairs` block is the caller's because the neighbour list is the caller's:
//! a minimizer that moves atoms decides when to rebuild it. See `docs/interop.md`.

#![allow(clippy::type_complexity)]

use crate::ff::forcefield::ForceField;
use crate::ff::mmff::MmffVariant;
use molrs::Atomistic;

use super::Typifier;
use engine::MmffEngine;

mod embedded;
mod engine;
pub(crate) mod frame_builder;
pub mod params;

#[cfg(test)]
mod tests;

// Re-exports
pub use params::{MMFFAtomProp, MMFFParams};

/// Declare a front door: a newtype over the one [`MmffEngine`], with its variant
/// and its force-field name pinned by the type itself.
///
/// The forwarding is written once here rather than twice by hand, so the two doors
/// cannot drift apart — but each door is still a distinct concrete type with a
/// distinct parameter set, which is the whole public contract.
macro_rules! mmff_front_door {
    (
        $(#[$doc:meta])*
        $name:ident, $variant:expr, $set:literal
    ) => {
        $(#[$doc])*
        pub struct $name(MmffEngine);

        impl $name {
            #[doc = concat!("Create a typifier over the shipped `", $set, "` parameter set.")]
            ///
            /// Infallible: the parameters are compiled-in typed Rust
            /// ([`ff::params::mmff`](crate::ff::params::mmff)), so there is
            /// nothing that can fail to parse at runtime.
            pub fn new() -> Self {
                Self(MmffEngine::embedded($variant, $set))
            }

            #[doc = concat!("Create a typifier from a caller-supplied `", $set, "` XML string.")]
            ///
            /// Parses both typing metadata ([`MMFFParams`]) and potential parameters
            /// ([`ForceField`]) in one call. The variant is pinned by *this type* —
            /// it is never an argument.
            pub fn from_xml_str(xml: &str) -> Result<Self, String> {
                MmffEngine::from_xml_str($variant, xml).map(Self)
            }

            /// The MMFF typing metadata (atom-type properties, equivalences).
            pub fn params(&self) -> &MMFFParams {
                self.0.params()
            }

            /// The force field this door compiles potentials from.
            pub fn ff(&self) -> &ForceField {
                self.0.ff()
            }

            #[doc = concat!("Assign `", $set, "` labels and parameters to an all-atom graph.")]
            ///
            /// Atoms get their MMFF numeric `type` and partial `charge`; bonds,
            /// angles, dihedrals and impropers get their type labels **and** the
            /// per-instance numbers the kernels read — including `koop` on every
            /// improper and `(v1, v2, v3)` on every dihedral, resolved from *this*
            /// door's parameter set.
            ///
            /// This is the whole contract. To evaluate an energy, materialize the
            /// result ([`Atomistic::to_frame`]), add the neighbour list
            /// ([`intramolecular_pairs`](crate::ff::potential::intramolecular_pairs)),
            /// and compile it with [`ff()`](Self::ff)`.to_potentials(&frame)` — see
            /// the module example.
            pub fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
                self.0.typify(mol)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Typifier for $name {
            type Mol = Atomistic;

            fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
                self.0.typify(mol)
            }
        }
    };
}

mmff_front_door! {
    /// MMFF94 typifier (Halgren 1996) — the standard parameterization.
    ///
    /// Owns the MMFF94 typing metadata and force-field parameters, both read from
    /// the compiled table [`crate::ff::params::mmff`].
    ///
    /// ```no_run
    /// use molrs::ff::typifier::mmff::MMFF94Typifier;
    /// # fn main() -> Result<(), String> {
    /// # let mol = molrs::Atomistic::new();
    /// let typifier = MMFF94Typifier::new();
    /// assert_eq!(typifier.ff().name, "MMFF94");
    /// let typed = typifier.typify(&mol)?;
    /// # let _ = typed;
    /// # Ok(())
    /// # }
    /// ```
    MMFF94Typifier, MmffVariant::Mmff94, "MMFF94"
}

mmff_front_door! {
    /// MMFF94s typifier (Halgren 1999) — the **static** variant, for energy
    /// minimization.
    ///
    /// Identical to [`MMFF94Typifier`] except on delocalised trivalent nitrogen
    /// (MMFF numeric types 10 `NC=O` and 40 `NC=C`), where it re-parameterises 11
    /// out-of-plane rows and 42 torsion rows so that the nitrogen minimises to a
    /// **planar** geometry — the picture seen in crystal structures, rather than
    /// MMFF94's dynamic / time-averaged pyramidal one.
    ///
    /// The mechanism is the sign and size of the out-of-plane force constant. The
    /// kernel (`ff::potential::improper::mmff`) evaluates
    ///
    /// ```text
    /// E_oop = 0.5 · 143.9325 · koop · χ²
    /// ```
    ///
    /// with χ the Wilson out-of-plane angle in **radians** and `koop` in
    /// md·Å·rad⁻² (143.9325 converts md·Å → kcal·mol⁻¹). So `koop > 0` makes the
    /// planar geometry (χ = 0) an energy **minimum** and `koop < 0` makes it a
    /// **maximum**. MMFF94s sets `koop` on every such centre to a flat `+0.015`
    /// (type 10) / `+0.030` (type 40); under MMFF94 those rows range over
    /// `−0.033 … +0.004`.
    ///
    /// Parameters come from the same compiled table as [`MMFF94Typifier`]
    /// ([`crate::ff::params::mmff`]) — the two front doors differ by this
    /// force field's **name** and by the variant, which selects
    /// [`MMFF_OOP_S`](crate::ff::params::mmff::MMFF_OOP_S) /
    /// [`MMFF_TOR_S`](crate::ff::params::mmff::MMFF_TOR_S) and drives the
    /// per-instance `koop` / `(v1, v2, v3)` baked onto the typed graph.
    ///
    /// ```no_run
    /// use molrs::ff::typifier::mmff::MMFF94STypifier;
    /// # fn main() -> Result<(), String> {
    /// # let mol = molrs::Atomistic::new();
    /// let typifier = MMFF94STypifier::new();
    /// assert_eq!(typifier.ff().name, "MMFF94s");
    /// let typed = typifier.typify(&mol)?;
    /// # let _ = typed;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # References
    ///
    /// - T. A. Halgren, *MMFF VI. MMFF94s option for energy minimization studies*,
    ///   J. Comput. Chem. **20**, 720–729 (1999).
    MMFF94STypifier, MmffVariant::Mmff94s, "MMFF94s"
}
