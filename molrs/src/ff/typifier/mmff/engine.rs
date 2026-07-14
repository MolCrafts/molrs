//! The **one** MMFF typifier engine.
//!
//! MMFF94 and MMFF94s are the same typing pipeline over two parameter sets, so
//! there is exactly one implementation here and the public surface
//! ([`MMFF94Typifier`](super::MMFF94Typifier) /
//! [`MMFF94STypifier`](super::MMFF94STypifier)) is two newtypes over it. The
//! [`MmffVariant`] is a **private field** of this engine: users pick a variant by
//! picking a front door, never by passing a flag.
//!
//! The variant reaches the parameters by two independent paths, and both must be
//! fed or MMFF94s is only half-applied:
//!
//! 1. **Frame annotation** — `frame_builder::annotate_mmff` bakes the resolved
//!    per-instance numbers onto the labeled graph: `koop` on impropers and
//!    `(v1, v2, v3)` on dihedrals. These are exactly the columns the `mmff_oop` /
//!    `mmff_torsion` kernels read, so this is where the 94/94s numerical
//!    difference physically enters an energy.
//! 2. **The [`ForceField`] tree** — parsed from the front door's own XML
//!    (`MMFF94_XML` / `MMFF94S_XML`) and compiled by
//!    [`ForceField::to_potentials`]. It carries the force-field name and the
//!    Oop/Torsion type rows.
//!
//! Feeding only path 1 leaves a tree that still calls itself `MMFF94`; feeding
//! only path 2 leaves a Frame whose baked `koop` is still MMFF94's — the potentials
//! would then be bit-identical to MMFF94 while claiming to be MMFF94s.

use crate::ff::forcefield::ForceField;
use crate::ff::mmff::MmffVariant;
use crate::ff::potential::{Potentials, intramolecular_pairs};
use molrs::Atomistic;

use super::classify;
use super::frame_builder;
use super::params::MMFFParams;

/// Typing metadata + potential parameters + the variant they were read for.
///
/// Crate-private by construction: it is the implementation the two named front
/// doors share, not an API. Nothing outside this module may name it, so nothing
/// outside this module can construct an MMFF typifier with an arbitrary variant.
pub(super) struct MmffEngine {
    variant: MmffVariant,
    params: MMFFParams,
    ff: ForceField,
}

impl MmffEngine {
    /// Parse both halves (typing metadata + force field) out of one XML string.
    ///
    /// `variant` is supplied by the front door, never by a user.
    pub(super) fn from_xml_str(variant: MmffVariant, xml: &str) -> Result<Self, String> {
        let params = crate::ff::forcefield::xml::read_mmff_params_xml_str(xml)?;
        let ff = crate::ff::forcefield::xml::read_forcefield_xml_str(xml)?;
        Ok(Self {
            variant,
            params,
            ff,
        })
    }

    /// Parse one of the **embedded** parameter sets, which cannot fail at runtime.
    ///
    /// `xml` is a `&'static str` compiled into the binary and covered by this
    /// crate's own tests, so a parse error here is a build-time defect, not a
    /// caller error — which is why the front doors' `new()` is infallible.
    pub(super) fn embedded(variant: MmffVariant, xml: &'static str) -> Self {
        Self::from_xml_str(variant, xml)
            .unwrap_or_else(|e| panic!("embedded {variant:?} parameter set is malformed: {e}"))
    }

    pub(super) fn params(&self) -> &MMFFParams {
        &self.params
    }

    pub(super) fn ff(&self) -> &ForceField {
        &self.ff
    }

    /// Path 1: label the graph and bake this variant's per-instance parameters.
    pub(super) fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        frame_builder::annotate_mmff(mol, &self.params, self.variant)
    }

    /// Path 1 + path 2: `mol → Frame → Potentials`.
    pub(super) fn build(&self, mol: &Atomistic) -> Result<Potentials, String> {
        let mut frame = self.typify(mol)?.to_frame();
        let pairs = intramolecular_pairs(&frame);
        frame.insert("pairs", pairs);
        self.ff.to_potentials(&frame)
    }

    pub(super) fn typify_bond(&self, t1: u32, t2: u32, bond_order: f64) -> u32 {
        classify::typify_bond(t1, t2, bond_order, &self.params)
    }

    pub(super) fn typify_angle(&self, bt_ij: u32, bt_jk: u32) -> u32 {
        classify::typify_angle(bt_ij, bt_jk)
    }

    pub(super) fn typify_dihedral(&self, bt_ij: u32, bt_jk: u32, bt_kl: u32) -> u32 {
        classify::typify_dihedral(bt_ij, bt_jk, bt_kl)
    }
}
