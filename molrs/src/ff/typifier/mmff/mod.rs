//! MMFF atom/bond/angle/torsion/improper typifiers — two named front doors.
//!
//! Annotates an [`Atomistic`] with MMFF type labels and partial charges (the
//! typifier's job). MMFF carries no bespoke energy path here — it is a parameter
//! set plus a topology labeler — so the `build` convenience just materializes the
//! labeled graph to a [`Frame`](molrs::store::frame::Frame) and routes it through
//! the generic [`ForceField::to_potentials`](crate::ff::potential) compile path.
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
//! stretch-bend / vdW / charge row. They differ in **11 out-of-plane rows** and
//! **42 torsion rows**, every one of them centred on MMFF numeric type 10 (`NC=O`,
//! amide N) or 40 (`NC=C`, enamine-type N). MMFF94s ("s" = *static*) raises the
//! out-of-plane force constant `koop` on those centres to a flat `+0.015`
//! (type 10) / `+0.030` (type 40) md·Å·rad⁻², which — see
//! [`MMFF94STypifier`] — makes the planar nitrogen an energy *minimum*.
//!
//! # Example
//!
//! ```no_run
//! use molrs::ff::typifier::mmff::MMFF94Typifier;
//! use molrs::Atomistic;
//! # fn main() -> Result<(), String> {
//! let mol = Atomistic::new();                          // build or load your molecule
//! let potentials = MMFF94Typifier::new().build(&mol)?; // typify → to_frame → to_potentials
//! let coords: Vec<f64> = Vec::new();                   // flat [x,y,z, ...]
//! let (energy, _forces) = potentials.calc_energy_forces(&coords);
//! println!("MMFF94 energy = {energy} kcal/mol");
//! # Ok(())
//! # }
//! ```

#![allow(clippy::type_complexity)]

use crate::ff::forcefield::ForceField;
use crate::ff::mmff::MmffVariant;
use crate::ff::potential::Potentials;
use molrs::Atomistic;

use super::Typifier;
use engine::MmffEngine;

pub(crate) mod classify;
mod engine;
pub(crate) mod frame_builder;
pub mod params;

#[cfg(test)]
mod tests;

// Re-exports
pub use params::{MMFFAtomProp, MMFFParams};

/// Declare a front door: a newtype over the one [`MmffEngine`], with its variant
/// and its embedded XML pinned by the type itself.
///
/// The forwarding is written once here rather than twice by hand, so the two doors
/// cannot drift apart — but each door is still a distinct concrete type with a
/// distinct parameter set, which is the whole public contract.
macro_rules! mmff_front_door {
    (
        $(#[$doc:meta])*
        $name:ident, $variant:expr, $xml:expr, $set:literal
    ) => {
        $(#[$doc])*
        pub struct $name(MmffEngine);

        impl $name {
            #[doc = concat!("Create a typifier over the embedded `", $set, "` parameter set.")]
            ///
            /// Infallible: the parameter set is a `&'static str` compiled into the
            /// binary, so it cannot be malformed at runtime.
            pub fn new() -> Self {
                Self(MmffEngine::embedded($variant, $xml))
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
            pub fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
                self.0.typify(mol)
            }

            /// Typify a molecule and compile potentials in one step.
            ///
            /// `mol → Frame → Potentials`. The intermediate `Frame` is not retained.
            ///
            /// Requires [`Atomistic`] because MMFF typing depends on element
            /// symbols, bond orders, and ring membership.
            pub fn build(&self, mol: &Atomistic) -> Result<Potentials, String> {
                self.0.build(mol)
            }

            /// Typify an MMFF bond: 0=normal, 1=delocalized/aromatic.
            pub fn typify_bond(&self, t1: u32, t2: u32, bond_order: f64) -> u32 {
                self.0.typify_bond(t1, t2, bond_order)
            }

            /// Typify an MMFF angle from the two MMFF bond classes forming the angle.
            pub fn typify_angle(&self, bt_ij: u32, bt_jk: u32) -> u32 {
                self.0.typify_angle(bt_ij, bt_jk)
            }

            /// Typify an MMFF dihedral from the three MMFF bond classes in the dihedral.
            pub fn typify_dihedral(&self, bt_ij: u32, bt_jk: u32, bt_kl: u32) -> u32 {
                self.0.typify_dihedral(bt_ij, bt_jk, bt_kl)
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
    /// [`molrs::data::MMFF94_XML`].
    ///
    /// ```no_run
    /// use molrs::ff::typifier::mmff::MMFF94Typifier;
    /// # fn main() -> Result<(), String> {
    /// # let mol = molrs::Atomistic::new();
    /// let typifier = MMFF94Typifier::new();
    /// assert_eq!(typifier.ff().name, "MMFF94");
    /// let potentials = typifier.build(&mol)?;
    /// # let _ = potentials;
    /// # Ok(())
    /// # }
    /// ```
    MMFF94Typifier, MmffVariant::Mmff94, molrs::data::MMFF94_XML, "MMFF94"
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
    /// Parameters are read from [`molrs::data::MMFF94S_XML`], and the same variant
    /// drives the per-instance `koop` / `(v1, v2, v3)` baked onto the typed graph.
    ///
    /// ```no_run
    /// use molrs::ff::typifier::mmff::MMFF94STypifier;
    /// # fn main() -> Result<(), String> {
    /// # let mol = molrs::Atomistic::new();
    /// let typifier = MMFF94STypifier::new();
    /// assert_eq!(typifier.ff().name, "MMFF94s");
    /// let potentials = typifier.build(&mol)?;
    /// # let _ = potentials;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # References
    ///
    /// - T. A. Halgren, *MMFF VI. MMFF94s option for energy minimization studies*,
    ///   J. Comput. Chem. **20**, 720–729 (1999).
    MMFF94STypifier, MmffVariant::Mmff94s, molrs::data::MMFF94S_XML, "MMFF94s"
}
