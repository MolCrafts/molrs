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
//! 2. **The [`ForceField`] tree** — assembled from the compiled table under the
//!    front door's own name ([`embedded`](super::embedded)) and compiled by
//!    [`ForceField::to_potentials`]. It carries the force-field name and the
//!    style skeleton.
//!
//! Feeding only path 1 leaves a tree that still calls itself `MMFF94`; feeding
//! only path 2 leaves a Frame whose baked `koop` is still MMFF94's — the potentials
//! would then be bit-identical to MMFF94 while claiming to be MMFF94s.
//!
//! # The engine compiles nothing
//!
//! It labels a graph and it owns a [`ForceField`]. Turning the two into
//! [`Potentials`](crate::ff::potential::Potentials) is
//! `ForceField::to_potentials(&frame)` — the same call every other force field in
//! molrs goes through. There used to be a `build(mol)` convenience that did
//! typify → `to_frame` → `intramolecular_pairs` → `to_potentials` behind one
//! method name, which made MMFF the only typifier in the crate that could also
//! compile; it is gone. A typifier's contract is `typify`.

use crate::ff::forcefield::ForceField;
use crate::ff::mmff::MmffVariant;
use molrs::Atomistic;

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

    /// Assemble one of the **shipped** parameter sets from the compiled table.
    ///
    /// Infallible by construction: there is nothing to parse. `name` and
    /// `variant` are supplied by the front door and are the only things the two
    /// doors disagree about — both read the same
    /// [`ff::params::mmff`](crate::ff::params::mmff) rows.
    pub(super) fn embedded(variant: MmffVariant, name: &str) -> Self {
        Self {
            variant,
            params: super::embedded::typing_params(),
            ff: super::embedded::force_field(name),
        }
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
}
