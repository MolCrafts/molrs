//! Turning a [`ForceField`]'s declarations into evaluable potentials.
//!
//! Two doors, and the difference between them is what carries the non-bonded
//! pair list:
//!
//! * [`ForceField::to_potentials`] resolves every pair style against the
//!   frame's `pairs` block — a fixed list, finite by construction and with no
//!   spatial cutoff. Right for a molecule in free space, and what the geometry
//!   optimizer and the conformer pipeline use.
//! * [`ForceField::to_typed_potentials`] resolves them against the **atoms**,
//!   so the kernels answer for whatever pairs a neighbour search turns up.
//!   Right for a periodic box, and what MD uses.
//!
//! The direction is one-way: a force field **declares** styles, types and
//! constants, and this module reads them to build kernels. Nothing here is
//! imported back by [`crate::ff::forcefield`] — the gate
//! `forcefield_never_names_potential` keeps it that way, after the two
//! submodules spent a while naming each other.

use std::borrow::Cow;

use crate::ff::forcefield::{ForceField, Params, SpecialBonds};
use crate::ff::potential::registry::{self, ParamSource};
use crate::ff::potential::{Member, Potentials, TypedKernel, TypedMember};
use molrs::store::frame::Frame;

impl crate::ff::forcefield::Style {
    /// Build this style's kernel for a **neighbour-driven** evaluation, and
    /// say which special-bonds weights scale it.
    ///
    /// The counterpart of [`to_potential`](Self::to_potential). A bonded style
    /// is built identically — it reads indices, and a neighbour table does not
    /// concern it. A pair style is built in its typed form, which reads no
    /// `pairs` block: there is none to read when the list is rebuilt every few
    /// steps, and a kernel whose parameters were resolved against an older one
    /// would be naming different atoms.
    ///
    /// A pair style with no typed form is an [`Err`], not a fallback to the
    /// compiled one. Falling back would hand back a kernel whose parameters
    /// belong to a pair list nobody is evaluating, and it would answer.
    pub fn to_typed_potential(&self, frame: &Frame) -> Result<Option<TypedKernel>, String> {
        let category = self.category();
        if category == "atom" {
            return Ok(None);
        }
        if category != "pair" {
            // Bonded styles are unchanged, and take no special-bonds weight:
            // the term *is* the bonded interaction, not a scaled copy of it.
            // `special_bonds` is irrelevant to them, so a default is honest.
            return Ok(self
                .to_potential(frame, &SpecialBonds::default())?
                .map(|p| (p, None)));
        }
        // No `pairs` gate: a typed pair kernel is built from the atoms, and a
        // frame with atoms always has those.
        let type_params = self.defs.collect_type_params();
        let param_source =
            registry::lookup_param_source(category, &self.name).unwrap_or(ParamSource::TypeRows);
        if type_params.is_empty() && param_source == ParamSource::TypeRows {
            return Err(format!(
                "Style '{}' ({}) has no type definitions",
                self.name, category
            ));
        }
        let type_refs: Vec<(&str, &Params)> = type_params
            .iter()
            .map(|(name, params)| (name.as_str(), params))
            .collect();
        let (ctor, special) =
            registry::lookup_typed_kernel(category, &self.name).ok_or_else(|| {
                format!(
                    "pair style '{}' has no neighbour-driven form, so it cannot be \
                     evaluated over a neighbour list; its compiled form answers only \
                     for the pair list it was built from",
                    self.name
                )
            })?;
        let pot = ctor(&self.params, &type_refs, frame)?;
        Ok(Some((pot, Some(special))))
    }

    /// Build this style's molecule-bound [`Member`] by **expanding** its type
    /// parameters against `frame`'s topology — each bond/angle/… row's string
    /// type label is resolved to its parameters and stored as per-element
    /// arrays, so the resulting potential evaluates from coordinates alone.
    ///
    /// Returns `Ok(None)` for a style that carries no pairwise kernel (an atom
    /// style — types/charges only), `Err` for an unknown `(category, name)`.
    ///
    /// The `(category, name)` → constructor mapping lives in the [`registry`]; a
    /// new potential is added by registering its kernel, not by editing this
    /// dispatch.
    pub fn to_potential(
        &self,
        frame: &Frame,
        special_bonds: &SpecialBonds,
    ) -> Result<Option<Member>, String> {
        let category = self.category();
        if category == "atom" {
            return Ok(None);
        }
        // A style contributes nothing when the molecule carries no topology of its
        // kind: a bonded style with no bonds/angles/dihedrals/impropers, or a pair
        // style when the neighbour list is empty (e.g. methane, whose every atom
        // pair is 1-2 or 1-3 excluded). Skip it rather than letting the kernel ctor
        // fault on the absent/empty block.
        // Which block gates a style is the *kernel's* property, not the
        // category's: PME is a `pair` style that reads charges and
        // `exclusions` and never looks at `pairs`, so gating it on `pairs`
        // deleted a system's whole long-range electrostatics whenever the
        // caller had not built a pair list.
        let gated = registry::lookup_row_source(category, &self.name)
            .unwrap_or(registry::RowSource::CategoryBlock)
            == registry::RowSource::CategoryBlock;
        let topo_block = match category {
            _ if !gated => None,
            "bond" => Some("bonds"),
            "angle" => Some("angles"),
            "dihedral" => Some("dihedrals"),
            "improper" => Some("impropers"),
            "pair" => Some("pairs"),
            _ => None,
        };
        if let Some(block_name) = topo_block {
            let rows = frame.get(block_name).and_then(|b| b.nrows()).unwrap_or(0);
            if rows == 0 {
                return Ok(None);
            }
        }
        let type_params = self.defs.collect_type_params();
        // A style whose kernel resolves its parameters from type rows
        // (`ParamSource::TypeRows`) can resolve nothing without them — so no rows
        // is an error, not a silently-zero potential. A `PerInstance` style
        // (MMFF's bonded terms, `coul/cut`, `pme`) reads its numbers from Frame
        // columns the typifier baked and ignores `tp` entirely, so zero rows is
        // its *normal* state. Asking the registry which one this is replaces the
        // old blanket `category != "pair"` escape hatch — the hatch that let MMFF
        // register as table-driven and then be fed 4,065 rows of XML no code reads.
        //
        // The registry is the authority because it is where the kernel is declared;
        // an unregistered style falls through to `TypeRows` here and then fails on
        // the kernel lookup below with a more specific message.
        let param_source =
            registry::lookup_param_source(category, &self.name).unwrap_or(ParamSource::TypeRows);
        if type_params.is_empty() && param_source == ParamSource::TypeRows {
            return Err(format!(
                "Style '{}' ({}) has no type definitions",
                self.name, category
            ));
        }
        let type_refs: Vec<(&str, &Params)> = type_params
            .iter()
            .map(|(name, params)| (name.as_str(), params))
            .collect();
        let ctor = registry::lookup_kernel(category, &self.name).ok_or_else(|| {
            format!(
                "no kernel for style category '{}' name '{}'",
                category, self.name
            )
        })?;
        // Project the ForceField's `special_bonds` 1-4 weights into the params the
        // pair kernel reads (`lj14scale` / `coulomb14scale`), so the kernel scales
        // 1-4-flagged pairs without the registry signature carrying special_bonds.
        // Bonded kernels see their params unchanged.
        let params: Cow<Params> = if category == "pair" {
            let mut p = self.params.clone();
            p.set("lj14scale", special_bonds.lj_14());
            p.set("coulomb14scale", special_bonds.coul_14());
            Cow::Owned(p)
        } else {
            Cow::Borrowed(&self.params)
        };
        let pot = ctor(&params, &type_refs, frame)?;
        Ok(Some(pot))
    }
}

impl ForceField {
    /// Build the members of a **neighbour-driven** force evaluation, each with
    /// the bond-distance weights its non-bonded term takes.
    ///
    /// The counterpart of [`to_potentials`](Self::to_potentials), and what
    /// periodic MD needs. That one resolves every pair style against the
    /// frame's `pairs` block — a fixed list, finite by construction and with no
    /// spatial cutoff, which is right for a free-boundary molecule and wrong
    /// for a periodic system. This one resolves them against the **atoms**, so
    /// the kernels can answer for whatever pairs a neighbour search turns up,
    /// and reads no `pairs` block at all.
    ///
    /// The weights come back per member rather than once, because a force field
    /// may scale close van-der-Waals and electrostatic neighbours differently —
    /// Amber uses `1/2` and `1/1.2` — and in molrs those are separate kernels.
    /// A bonded member takes `None`: it *is* the bonded interaction, not a
    /// scaled copy of it.
    ///
    /// # Why the weights matter here and not there
    ///
    /// A compiled list carries the exclusions by leaving the excluded rows out
    /// and baking the 1-4 factor into the parameters. A neighbour table has no
    /// such memory — it finds every pair inside the cutoff, bonded or not — so
    /// without the weights a bonded pair is counted twice: once by the bond
    /// term and once at full non-bonded strength, at bond length.
    pub fn to_typed_potentials(&self, frame: &Frame) -> Result<Vec<TypedMember>, String> {
        let mut out = Vec::new();
        for style in self.styles() {
            // A bonded style contributes nothing when the molecule carries no
            // topology of its kind. A pair style is never skipped: which pairs
            // exist is the neighbour search's answer, not the frame's.
            let block = match style.category() {
                "bond" => Some("bonds"),
                "angle" => Some("angles"),
                "dihedral" => Some("dihedrals"),
                "improper" => Some("impropers"),
                _ => None,
            };
            if let Some(b) = block
                && frame.get(b).is_none()
            {
                continue;
            }
            if let Some((pot, special)) = style.to_typed_potential(frame)? {
                let weights = special.map(|c| match c {
                    registry::SpecialClass::Vdw => self.special_bonds().lj_weights(),
                    registry::SpecialClass::Coulomb => self.special_bonds().coul_weights(),
                });
                out.push((pot, weights));
            }
        }
        Ok(out)
    }

    /// Build evaluable [`Potentials`] by expanding every style against a
    /// typed [`Frame`].
    ///
    /// Each style's `to_potential` resolves its string type labels to per-element
    /// parameter arrays (see [`Style::to_potential`](crate::ff::forcefield::Style::to_potential)),
    /// so the resulting potentials are **molecule-bound**: they retain no Frame
    /// and evaluate from coordinates alone. Styles with no kernel (atom styles)
    /// are skipped. This is the molpy-style `ForceField → Potentials` conversion;
    /// there is no separate "compile" step.
    ///
    /// The `pairs` block carries the 1-2 / 1-3 weights by whether a row is
    /// there, so a force field that *scales* those classes rather than
    /// excluding them is an [`Err`] here — see
    /// [`SpecialBonds::compiled_inclusion`](crate::ff::forcefield::SpecialBonds::compiled_inclusion).
    /// [`to_typed_potentials`](Self::to_typed_potentials) carries a per-pair
    /// weight and takes every force field.
    pub fn to_potentials(&self, frame: &Frame) -> Result<Potentials, String> {
        // The `pairs` block need not have come from `intramolecular_pairs` — a
        // GROMACS `[ pairs ]` section is read straight off a file — so the
        // weights are checked here too, at the door that decides the physics,
        // and not only where the list happens to be built.
        self.special_bonds().compiled_inclusion()?;
        let mut pots = Potentials::new();
        for style in self.styles() {
            // A style whose topology block is entirely absent contributes nothing
            // (the molecule simply has no bonds/angles/… of that kind) — skip it,
            // rather than error. A *present* block with an unknown type label is a
            // real error and still propagates from the kernel constructor.
            let block = match style.category() {
                "bond" => Some("bonds"),
                "angle" => Some("angles"),
                "dihedral" => Some("dihedrals"),
                "improper" => Some("impropers"),
                "pair" => Some("pairs"),
                _ => None,
            };
            if let Some(b) = block
                && frame.get(b).is_none()
            {
                continue;
            }
            if let Some(pot) = style.to_potential(frame, self.special_bonds())? {
                pots.push(pot);
            }
        }
        // Record the atom count so callers (e.g. the geometry optimizer's batch
        // path) can validate coordinate shapes against this topology.
        pots.set_n_atoms(frame.get("atoms").and_then(|b| b.nrows()).unwrap_or(0));
        Ok(pots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::store::block::Block;
    use molrs::types::Idx;
    use ndarray::Array1;

    fn two_atoms() -> Frame {
        let mut atoms = Block::new();
        for (key, v) in [("x", [0.0, 1.6]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0])] {
            atoms
                .insert(key, Array1::from_vec(v.to_vec()).into_dyn())
                .unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    fn with_bond(mut frame: Frame, label: &str) -> Frame {
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("type", Array1::from_vec(vec![label.to_string()]).into_dyn())
            .unwrap();
        frame.insert("bonds", bonds);
        frame
    }

    fn bond_ff() -> ForceField {
        let mut ff = ForceField::new("t");
        ff.def_bondstyle("harmonic")
            .def_type("CT-CT", &[("k", 300.0), ("r0", 1.5)]);
        ff
    }

    #[test]
    fn an_atom_style_carries_no_kernel() {
        let mut ff = ForceField::new("t");
        ff.def_atomstyle("full").def_type("CT", &[("mass", 12.0)]);
        let style = &ff.styles()[0];
        let frame = two_atoms();
        assert!(
            style
                .to_potential(&frame, &SpecialBonds::default())
                .unwrap()
                .is_none()
        );
        assert!(style.to_typed_potential(&frame).unwrap().is_none());
    }

    #[test]
    fn a_bonded_style_is_skipped_without_its_block_and_built_with_it() {
        let ff = bond_ff();
        let style = &ff.styles()[0];
        let bare = two_atoms();
        assert!(
            style
                .to_potential(&bare, &SpecialBonds::default())
                .unwrap()
                .is_none()
        );
        assert!(ff.to_potentials(&bare).unwrap().members().is_empty());

        let bonded = with_bond(two_atoms(), "CT-CT");
        let pots = ff.to_potentials(&bonded).unwrap();
        assert_eq!(pots.members().len(), 1);
        // ½·300·(1.6 − 1.5)² = 1.5 kcal/mol
        let e = pots.calc_energy(&[0.0, 0.0, 0.0, 1.6, 0.0, 0.0]);
        assert!((e - 1.5).abs() < 1e-9);
    }

    #[test]
    fn an_unknown_type_label_in_a_present_block_is_an_error() {
        let ff = bond_ff();
        let frame = with_bond(two_atoms(), "XX-XX");
        assert!(ff.to_potentials(&frame).is_err());
    }

    #[test]
    fn a_pair_style_without_a_neighbour_driven_form_is_refused_by_the_typed_door() {
        let mut ff = ForceField::new("t");
        ff.def_pairstyle(
            "coul/long/pme",
            &[
                ("coulomb", 332.06371),
                ("dielectric", 1.0),
                ("coulomb14scale", 0.5),
            ],
        );
        let style = &ff.styles()[0];
        let err = style.to_typed_potential(&two_atoms()).unwrap_err();
        assert!(err.contains("neighbour-driven"), "{err}");
    }
}
