//! Turning a [`ForceField`]'s declarations into evaluable potentials.
//!
//! [`PotentialCompiler`] reads one force field and has two doors; the
//! difference between them is what carries the non-bonded pair list:
//!
//! * [`PotentialCompiler::compile`] resolves every pair style against the
//!   frame's `pairs` block — a fixed list, finite by construction and with no
//!   spatial cutoff. Right for a molecule in free space, and what the geometry
//!   optimizer and the conformer pipeline use.
//! * [`PotentialCompiler::compile_typed`] resolves them against the **atoms**,
//!   so the kernels answer for whatever pairs a neighbour search turns up.
//!   Right for a periodic box, and what MD uses.
//!
//! The direction is one-way: a force field **declares** styles, types and
//! constants, and this module reads them to build kernels. Nothing here is
//! imported back by [`crate::ff::forcefield`].

use std::borrow::Cow;

use crate::ff::forcefield::{ForceField, Params, SpecialBonds, Style};
use crate::ff::potential::registry::{self, ParamSource};
use crate::ff::potential::{Member, Potentials, TypedKernel, TypedMember};
use molrs::store::frame::Frame;
use molrs::store::schema::block_names::{ANGLES, ATOMS, BONDS, CMAPS, DIHEDRALS, IMPROPERS, PAIRS};

/// Compiles one [`ForceField`] against typed [`Frame`]s.
///
/// The force field declares; the compiler reads those declarations and builds
/// kernels. It borrows the force field, so it is cheap to make where the frame
/// is at hand:
///
/// ```
/// use molrs::ff::forcefield::{ForceField, Params};
/// use molrs::ff::potential::PotentialCompiler;
/// use molrs::Frame;
///
/// let mut ff = ForceField::new("example");
/// ff.def_style("bond", "harmonic", Params::new())
///     .unwrap()
///     .def_type("A-B", &["A", "B"], Params::from_pairs(&[("k", 300.0), ("r0", 1.5)]))
///     .unwrap();
///
/// // A frame without a `bonds` block has nothing for the bond style to read.
/// let pots = PotentialCompiler::new(&ff).compile(&Frame::new()).unwrap();
/// assert!(pots.members().is_empty());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PotentialCompiler<'a> {
    ff: &'a ForceField,
}

impl<'a> PotentialCompiler<'a> {
    /// A compiler that reads `ff`.
    pub fn new(ff: &'a ForceField) -> Self {
        Self { ff }
    }

    /// Build evaluable [`Potentials`] by expanding every style against a
    /// typed [`Frame`].
    ///
    /// Each style resolves its string type labels to per-element parameter
    /// arrays, so the resulting potentials are **molecule-bound**: they retain
    /// no Frame and evaluate from coordinates alone. Styles with no kernel
    /// (atom styles) are skipped.
    ///
    /// The `pairs` block carries the 1-2 / 1-3 weights by whether a row is
    /// there, so a force field that *scales* those classes rather than
    /// excluding them is an [`Err`] here — see
    /// [`SpecialBonds::compiled_inclusion`](crate::ff::forcefield::SpecialBonds::compiled_inclusion).
    /// [`compile_typed`](Self::compile_typed) carries a per-pair weight and
    /// takes every force field.
    pub fn compile(&self, frame: &Frame) -> Result<Potentials, String> {
        // The `pairs` block need not have come from `intramolecular_pairs` — a
        // GROMACS `[ pairs ]` section is read straight off a file — so the
        // weights are checked here too, at the door that decides the physics,
        // and not only where the list happens to be built.
        self.ff.special_bonds().compiled_inclusion()?;
        let mut pots = Potentials::new();
        for style in self.ff.styles() {
            // `member` skips a style whose topology block is absent or empty; a
            // *present* block with an unknown type label is a real error and
            // propagates from the kernel constructor.
            if let Some(pot) = self.member(style, frame, self.ff.special_bonds())? {
                pots.push(pot);
            }
        }
        // Record the atom count so callers (e.g. the geometry optimizer's batch
        // path) can validate coordinate shapes against this topology.
        pots.set_n_atoms(frame.get(ATOMS).and_then(|b| b.nrows()).unwrap_or(0));
        Ok(pots)
    }

    /// Build the members of a **neighbour-driven** force evaluation, each with
    /// the bond-distance weights its non-bonded term takes.
    ///
    /// The counterpart of [`compile`](Self::compile), and what periodic MD
    /// needs. That one resolves every pair style against the frame's `pairs`
    /// block — a fixed list, finite by construction and with no spatial
    /// cutoff, which is right for a free-boundary molecule and wrong for a
    /// periodic system. This one resolves them against the **atoms**, so the
    /// kernels can answer for whatever pairs a neighbour search turns up, and
    /// reads no `pairs` block at all.
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
    pub fn compile_typed(&self, frame: &Frame) -> Result<Vec<TypedMember>, String> {
        let mut out = Vec::new();
        for style in self.ff.styles() {
            // A bonded style contributes nothing when the molecule carries no
            // topology of its kind (`member` skips it). A pair style is never
            // skipped: which pairs exist is the neighbour search's answer, not
            // the frame's.
            if let Some((pot, special)) = self.typed_member(style, frame)? {
                let weights = special.map(|c| match c {
                    registry::SpecialClass::Vdw => self.ff.special_bonds().lj_weights(),
                    registry::SpecialClass::Coulomb => self.ff.special_bonds().coul_weights(),
                });
                out.push((pot, weights));
            }
        }
        Ok(out)
    }

    /// Build `style`'s kernel for a **neighbour-driven** evaluation, and say
    /// which special-bonds weights scale it.
    ///
    /// The counterpart of [`member`](Self::member). A bonded style is built
    /// identically — it reads indices, and a neighbour table does not concern
    /// it. A pair style is built in its typed form, which reads no `pairs`
    /// block: there is none to read when the list is rebuilt every few steps,
    /// and a kernel whose parameters were resolved against an older one would
    /// be naming different atoms.
    ///
    /// A pair style with no typed form is an [`Err`], not a fallback to the
    /// compiled one. Falling back would hand back a kernel whose parameters
    /// belong to a pair list nobody is evaluating, and it would answer.
    fn typed_member(&self, style: &Style, frame: &Frame) -> Result<Option<TypedKernel>, String> {
        let category = style.category();
        if category == "atom" {
            return Ok(None);
        }
        if category != "pair" {
            // Bonded styles are unchanged, and take no special-bonds weight:
            // the term *is* the bonded interaction, not a scaled copy of it.
            // `special_bonds` is irrelevant to them, so a default is honest.
            return Ok(self
                .member(style, frame, &SpecialBonds::default())?
                .map(|p| (p, None)));
        }
        // No `pairs` gate: a typed pair kernel is built from the atoms, and a
        // frame with atoms always has those.
        let type_params = style.defs().kernel_type_params()?;
        let param_source =
            registry::lookup_param_source(category, style.name()).unwrap_or(ParamSource::TypeRows);
        if type_params.is_empty() && param_source == ParamSource::TypeRows {
            return Err(format!(
                "Style '{}' ({}) has no type definitions",
                style.name(),
                category
            ));
        }
        let type_refs: Vec<(&str, &Params)> = type_params
            .iter()
            .map(|(name, params)| (name.as_str(), params))
            .collect();
        let (ctor, special) =
            registry::lookup_typed_kernel(category, style.name()).ok_or_else(|| {
                format!(
                    "pair style '{}' has no neighbour-driven form, so it cannot be \
                     evaluated over a neighbour list; its compiled form answers only \
                     for the pair list it was built from",
                    style.name()
                )
            })?;
        let pot = ctor(style.params(), &type_refs, frame)?;
        Ok(Some((pot, Some(special))))
    }

    /// Build `style`'s molecule-bound [`Member`] by **expanding** its type
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
    fn member(
        &self,
        style: &Style,
        frame: &Frame,
        special_bonds: &SpecialBonds,
    ) -> Result<Option<Member>, String> {
        let category = style.category();
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
        let gated = registry::lookup_row_source(category, style.name())
            .unwrap_or(registry::RowSource::CategoryBlock)
            == registry::RowSource::CategoryBlock;
        let topo_block = match category {
            _ if !gated => None,
            "bond" => Some(BONDS),
            "angle" => Some(ANGLES),
            "dihedral" => Some(DIHEDRALS),
            "improper" => Some(IMPROPERS),
            "cmap" => Some(CMAPS),
            "pair" => Some(PAIRS),
            _ => None,
        };
        if let Some(block_name) = topo_block {
            let rows = frame.get(block_name).and_then(|b| b.nrows()).unwrap_or(0);
            if rows == 0 {
                return Ok(None);
            }
        }
        let type_params = style.defs().kernel_type_params()?;
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
            registry::lookup_param_source(category, style.name()).unwrap_or(ParamSource::TypeRows);
        if type_params.is_empty() && param_source == ParamSource::TypeRows {
            return Err(format!(
                "Style '{}' ({}) has no type definitions",
                style.name(),
                category
            ));
        }
        let type_refs: Vec<(&str, &Params)> = type_params
            .iter()
            .map(|(name, params)| (name.as_str(), params))
            .collect();
        let ctor = registry::lookup_kernel(category, style.name()).ok_or_else(|| {
            format!(
                "no kernel for style category '{}' name '{}'",
                category,
                style.name()
            )
        })?;
        // Project the ForceField's `special_bonds` 1-4 weights into the params the
        // pair kernel reads (`lj14scale` / `coulomb14scale`), so the kernel scales
        // 1-4-flagged pairs without the registry signature carrying special_bonds.
        // Bonded kernels see their params unchanged.
        let params: Cow<Params> = if category == "pair" {
            let mut p = style.params().clone();
            p.set("lj14scale", special_bonds.lj_14());
            p.set("coulomb14scale", special_bonds.coul_14());
            Cow::Owned(p)
        } else {
            Cow::Borrowed(style.params())
        };
        let frame = self.own_rows(style, topo_block, param_source, frame)?;
        let pot = ctor(&params, &type_refs, &frame)?;
        Ok(Some(pot))
    }

    /// `frame` with `style`'s topology block cut to the rows whose type is one
    /// of `style`'s — when another table-driven style of the same category
    /// owns the rest (a LAMMPS `angle_style hybrid harmonic charmm`).
    ///
    /// A kernel resolves every row of its block against its own types, so
    /// without this a second style's rows are "unknown types" to the first.
    /// A row whose type no style of the category defines is still an error.
    /// Per-instance styles (MMFF's bend and stretch-bend) each price every
    /// row and are never cut; a category with one table-driven style is
    /// passed through unchanged.
    fn own_rows<'f>(
        &self,
        style: &Style,
        block_name: Option<&str>,
        source: ParamSource,
        frame: &'f Frame,
    ) -> Result<Cow<'f, Frame>, String> {
        let table_driven = |s: &&Style| {
            registry::lookup_param_source(s.category(), s.name()).unwrap_or(ParamSource::TypeRows)
                == ParamSource::TypeRows
        };
        let siblings: Vec<&Style> = self
            .ff
            .get_styles(style.category())
            .into_iter()
            .filter(table_driven)
            .collect();
        let (Some(block_name), ParamSource::TypeRows, true) =
            (block_name, source, siblings.len() > 1)
        else {
            return Ok(Cow::Borrowed(frame));
        };
        let Some(block) = frame.get(block_name) else {
            return Ok(Cow::Borrowed(frame));
        };
        let types = block
            .get("type")
            .and_then(|c| c.as_string())
            .ok_or_else(|| format!("{block_name} block missing \"type\" column"))?;
        let names = |s: &Style| -> Result<Vec<String>, String> {
            Ok(s.defs()
                .kernel_type_params()?
                .into_iter()
                .map(|(name, _)| name)
                .collect())
        };
        let own = names(style)?;
        let all: Vec<Vec<String>> = siblings
            .iter()
            .map(|s| names(s))
            .collect::<Result<_, _>>()?;
        let mut keep = Vec::new();
        for (row, label) in types.iter().enumerate() {
            if own.contains(label) {
                keep.push(row);
            } else if !all.iter().flatten().any(|n| n == label) {
                return Err(format!(
                    "{block_name} row {row}: type '{label}' is defined by no {} style",
                    style.category()
                ));
            }
        }
        let mut cut = frame.clone();
        cut.insert(
            block_name,
            block.select_rows(&keep).map_err(|e| e.to_string())?,
        );
        Ok(Cow::Owned(cut))
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
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_pairs(&[("k", 300.0), ("r0", 1.5)]),
            )
            .unwrap();
        ff
    }

    #[test]
    fn an_atom_style_carries_no_kernel() {
        let mut ff = ForceField::new("t");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_pairs(&[("mass", 12.0)]))
            .unwrap();
        let frame = two_atoms();
        let compiler = PotentialCompiler::new(&ff);
        assert!(compiler.compile(&frame).unwrap().members().is_empty());
        assert!(compiler.compile_typed(&frame).unwrap().is_empty());
    }

    #[test]
    fn a_bonded_style_is_skipped_without_its_block_and_built_with_it() {
        let ff = bond_ff();
        let compiler = PotentialCompiler::new(&ff);
        let bare = two_atoms();
        assert!(compiler.compile(&bare).unwrap().members().is_empty());
        assert!(compiler.compile_typed(&bare).unwrap().is_empty());

        let bonded = with_bond(two_atoms(), "CT-CT");
        let pots = compiler.compile(&bonded).unwrap();
        assert_eq!(pots.members().len(), 1);
        // k·(r − r0)² = 300·(1.6 − 1.5)² = 3 kcal/mol (LAMMPS bond harmonic)
        let e = pots.calc_energy(&[0.0, 0.0, 0.0, 1.6, 0.0, 0.0]);
        assert!((e - 3.0).abs() < 1e-10, "{e}");
        assert_eq!(compiler.compile_typed(&bonded).unwrap().len(), 1);
    }

    /// A cmap style is gated on the `cmaps` block like any bonded style, and
    /// a present block with no cmap kernel registered is the plain
    /// "no kernel" error, never a panic.
    #[test]
    fn a_cmap_style_is_gated_on_cmaps_and_needs_a_kernel() {
        let mut params = Params::new();
        params.set_array("grid", ndarray::ArrayD::zeros(vec![24, 24]));
        let mut ff = ForceField::new("t");
        ff.def_style("cmap", "charmm", Params::new())
            .unwrap()
            .def_type("c", &["C", "N", "CA", "C", "N"], params)
            .unwrap();
        let compiler = PotentialCompiler::new(&ff);
        assert!(compiler.compile(&two_atoms()).unwrap().members().is_empty());
        assert!(compiler.compile_typed(&two_atoms()).unwrap().is_empty());

        let mut frame = two_atoms();
        let mut cmaps = Block::new();
        for key in ["atomi", "atomj", "atomk", "atoml", "atomm"] {
            cmaps
                .insert(key, Array1::from_vec(vec![0 as Idx]).into_dyn())
                .unwrap();
        }
        cmaps
            .insert("type", Array1::from_vec(vec!["c".to_string()]).into_dyn())
            .unwrap();
        frame.insert("cmaps", cmaps);
        for err in [
            compiler.compile(&frame).map(|_| ()).unwrap_err(),
            compiler.compile_typed(&frame).map(|_| ()).unwrap_err(),
        ] {
            assert!(
                err.contains("no kernel for style category 'cmap' name 'charmm'"),
                "{err}"
            );
        }
    }

    #[test]
    fn an_unknown_type_label_in_a_present_block_is_an_error() {
        let ff = bond_ff();
        let frame = with_bond(two_atoms(), "XX-XX");
        assert!(PotentialCompiler::new(&ff).compile(&frame).is_err());
    }

    #[test]
    fn compile_typed_refuses_a_pair_style_without_a_neighbour_driven_form() {
        let mut ff = ForceField::new("t");
        ff.def_style(
            "pair",
            "coul/long/pme",
            Params::from_pairs(&[
                ("coulomb", 332.06371),
                ("dielectric", 1.0),
                ("coulomb14scale", 0.5),
            ]),
        )
        .unwrap();
        let err = PotentialCompiler::new(&ff)
            .compile_typed(&two_atoms())
            .unwrap_err();
        assert!(err.contains("neighbour-driven"), "{err}");
    }

    #[test]
    fn compile_refuses_a_1_2_or_1_3_weight_other_than_0_or_1() {
        let mut ff = bond_ff();
        // A fraction on 1-3: a compiled pairs list carries that class by row
        // presence, which cannot say "half".
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.5, 0.5],
            coul: [0.0, 0.5, 0.5],
        });
        let frame = with_bond(two_atoms(), "CT-CT");
        let err = PotentialCompiler::new(&ff).compile(&frame).unwrap_err();
        assert!(err.contains("1-3"), "{err}");

        // A fraction on 1-2 is refused the same way.
        let mut ff = bond_ff();
        ff.set_special_bonds(SpecialBonds {
            lj: [0.5, 0.0, 0.5],
            coul: [0.5, 0.0, 0.5],
        });
        let err = PotentialCompiler::new(&ff).compile(&frame).unwrap_err();
        assert!(err.contains("1-2"), "{err}");
    }
}
