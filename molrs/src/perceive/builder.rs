//! The [`Perceive`] builder — one uniform skin over the perception free
//! functions of this layer.
//!
//! The wrapped functions have four different shapes: a side table
//! ([`rings::find_rings`] → [`rings::RingInfo`]), an in-place mutation returning
//! a count ([`aromaticity::perceive_aromaticity`]), a graph-out transform
//! ([`hydrogens::add_hydrogens`]), and maps
//! ([`stereo::assign_stereo_from_3d`], [`rotatable::detect_rotatable_bonds`]).
//! `Perceive` normalises all four to a single contract:
//!
//! > **graph in / graph out, non-mutating** — each `find_*` clones `mol`, writes
//! > the perceived facts onto the clone as atom / bond props, and returns it.
//! > The input is never touched.
//!
//! Because the output is a graph, the finders compose: the result of one is a
//! legal input to the next, and earlier facts survive later stages.
//!
//! # Props written
//!
//! | Method | Atom props | Bond props |
//! |---|---|---|
//! | [`Perceive::find_rings`] | `is_in_ring` (0/1), `n_rings` | `is_in_ring` (0/1), `n_rings` |
//! | [`Perceive::find_aromaticity`] | `is_aromatic` (0/1) | `is_aromatic` (0/1), `order` (1.5 when aromatic), `kekule_order` |
//! | [`Perceive::find_hydrogens`] | — (adds H atoms) | — (adds H bonds) |
//! | [`Perceive::find_stereo`] | `stereo` (`"CW"` / `"CCW"`) | `stereo` (`"E"` / `"Z"` / `"either"`) |
//! | [`Perceive::find_rotatable`] | — | `is_rotatable` (0/1) |
//! | [`Perceive::find_bond_types`] | — | `type` (BCC bond type: 1/2/3/6/7/8/9) |

use std::collections::HashSet;

use super::stereo::{BondStereo, TetrahedralStereo};
use super::{aromaticity, bond_type, hydrogens, rings, rotatable, stereo};
use crate::system::atomistic::{AtomId, Atomistic, BondId};

/// Atom / bond prop: `1` when the atom / bond lies on at least one SSSR ring.
const IS_IN_RING: &str = "is_in_ring";
/// Atom / bond prop: number of SSSR rings the atom / bond belongs to.
const N_RINGS: &str = "n_rings";
/// Bond prop: `1` when the bond is a rotatable single bond.
const IS_ROTATABLE: &str = "is_rotatable";
/// Atom / bond prop: the perceived stereo descriptor.
const STEREO: &str = "stereo";

/// Chemical perception, as a builder.
///
/// Every `find_*` method is graph-in / graph-out and **non-mutating**: it clones
/// the input, annotates the clone, and returns it. See the [module
/// docs](self) for the props each method writes.
///
/// The builder currently carries no options; it exists to give the perception
/// functions one shape (and a place to hang options later).
///
/// # Examples
///
/// ```
/// use molrs::Atomistic;
/// use molrs::perceive::Perceive;
///
/// let mol = Atomistic::new();
/// let perceived = Perceive::new().find_rings(&mol);
/// assert_eq!(perceived.n_atoms(), mol.n_atoms());
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Perceive;

impl Perceive {
    /// Create a perception builder with default settings.
    ///
    /// # Returns
    ///
    /// A `Perceive` with no options set.
    pub fn new() -> Self {
        Self
    }

    /// Perceive rings (SSSR) and project them onto the graph.
    ///
    /// Wraps [`rings::find_rings`], whose [`rings::RingInfo`] side table is
    /// re-projected as props: every atom and every bond receives `is_in_ring`
    /// (`0` / `1`) and `n_rings` — including the acyclic ones, which are
    /// explicitly flagged `0` rather than left unset.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule to perceive; left untouched.
    ///
    /// # Returns
    ///
    /// A clone of `mol` carrying the ring props.
    pub fn find_rings(&self, mol: &Atomistic) -> Atomistic {
        let info = rings::find_rings(mol);
        let mut out = mol.clone();

        let atom_ids: Vec<AtomId> = out.atoms().map(|(id, _)| id).collect();
        for id in atom_ids {
            let _ = out.set_atom(id, IS_IN_RING, i32::from(info.is_atom_in_ring(id)));
            let _ = out.set_atom(id, N_RINGS, saturating_i32(info.num_atom_rings(id)));
        }

        let bond_ids: Vec<BondId> = out.bonds().map(|(id, _)| id).collect();
        for bid in bond_ids {
            let _ = out.set_bond_prop(bid, IS_IN_RING, i32::from(info.is_bond_in_ring(bid)));
            let _ = out.set_bond_prop(bid, N_RINGS, saturating_i32(info.num_bond_rings(bid)));
        }

        out
    }

    /// Perceive aromaticity (RDKit default model) on a clone of the graph.
    ///
    /// Wraps [`aromaticity::perceive_aromaticity`], which is `&mut`-shaped and
    /// rewrites aromatic bond `order` to `1.5` (stashing the Kekulé value under
    /// `kekule_order`). Cloning first is therefore load-bearing: it is what
    /// keeps the caller's Kekulé structure intact.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule to perceive; left untouched.
    ///
    /// # Returns
    ///
    /// A clone of `mol` with `is_aromatic` on every atom and bond, aromatic bond
    /// orders set to `1.5`, and `kekule_order` preserved on the clone's bonds.
    pub fn find_aromaticity(&self, mol: &Atomistic) -> Atomistic {
        let mut out = mol.clone();
        let _ = aromaticity::perceive_aromaticity(&mut out);
        out
    }

    /// Add the hydrogens implied by each heavy atom's open valence.
    ///
    /// Wraps [`hydrogens::add_hydrogens`], which is already graph-in / graph-out.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule to fill; left untouched.
    ///
    /// # Returns
    ///
    /// A new graph: the heavy-atom skeleton of `mol` plus the perceived
    /// hydrogens and their bonds.
    pub fn find_hydrogens(&self, mol: &Atomistic) -> Atomistic {
        hydrogens::add_hydrogens(mol)
    }

    /// Perceive stereochemistry from 3-D coordinates and project it onto the
    /// graph.
    ///
    /// Wraps [`stereo::assign_stereo_from_3d`] (atoms) and
    /// [`stereo::assign_bond_stereo_from_3d`] (bonds). Both return **maps**, and
    /// the bond map is *total*: it carries [`BondStereo::None`] for every bond
    /// that is not a double bond. That variant is a sentinel meaning "not a
    /// stereo bond", not a perceived fact, so it is **skipped** — as is
    /// [`TetrahedralStereo::Unspecified`]. A `stereo` prop appears only where a
    /// real descriptor was perceived: `"CW"` / `"CCW"` on atoms, `"E"` / `"Z"` /
    /// `"either"` on bonds.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule to perceive; left untouched.
    ///
    /// # Returns
    ///
    /// A clone of `mol` carrying a `stereo` prop on each perceived stereocentre
    /// and stereo bond, and none elsewhere.
    pub fn find_stereo(&self, mol: &Atomistic) -> Atomistic {
        let atom_stereo = stereo::assign_stereo_from_3d(mol);
        let bond_stereo = stereo::assign_bond_stereo_from_3d(mol);
        let mut out = mol.clone();

        for (id, s) in atom_stereo {
            if let Some(label) = tetrahedral_label(s) {
                let _ = out.set_atom(id, STEREO, label);
            }
        }
        for (bid, s) in bond_stereo {
            if let Some(label) = bond_label(s) {
                let _ = out.set_bond_prop(bid, STEREO, label);
            }
        }

        out
    }

    /// Perceive rotatable bonds and project them onto the graph.
    ///
    /// Wraps [`rotatable::detect_rotatable_bonds`], which returns atom **pairs**;
    /// they are resolved back to bond ids here. A bond is rotatable when it is a
    /// single, acyclic bond with two non-terminal endpoints. Every bond is
    /// flagged — non-rotatable ones explicitly with `0`.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule to perceive; left untouched.
    ///
    /// # Returns
    ///
    /// A clone of `mol` with `is_rotatable` (`0` / `1`) on every bond.
    pub fn find_rotatable(&self, mol: &Atomistic) -> Atomistic {
        let rotatable: HashSet<(AtomId, AtomId)> = rotatable::detect_rotatable_bonds(mol)
            .into_iter()
            .map(|(a, b)| unordered(a, b))
            .collect();
        let mut out = mol.clone();

        let bonds: Vec<(BondId, AtomId, AtomId)> = out
            .bonds()
            .map(|(bid, bond)| (bid, bond.nodes[0], bond.nodes[1]))
            .collect();
        for (bid, a, b) in bonds {
            let flag = i32::from(rotatable.contains(&unordered(a, b)));
            let _ = out.set_bond_prop(bid, IS_ROTATABLE, flag);
        }

        out
    }

    /// Perceive BCC bond types and project them onto the graph.
    ///
    /// Wraps [`bond_type::find_bond_types`], which is already graph-in /
    /// graph-out. Every bond receives a `type` prop in `{1, 2, 3, 6, 7, 8, 9}` —
    /// the alphabet AM1-BCC's atom-type rules and correction table are keyed on,
    /// which distinguishes aromatic bonds (7/8) and *delocalized* ones (9, e.g. a
    /// carboxylate's two equivalent C–O bonds) from plain orders. See that
    /// module's docs for the promotion boundary and the delocalization rules.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule to perceive; left untouched.
    ///
    /// # Returns
    ///
    /// A clone of `mol` with a BCC `type` on every bond.
    pub fn find_bond_types(&self, mol: &Atomistic) -> Atomistic {
        bond_type::find_bond_types(mol)
    }
}

/// Order an atom pair so that a bond can be looked up regardless of the
/// direction it was reported in.
fn unordered(a: AtomId, b: AtomId) -> (AtomId, AtomId) {
    if a <= b { (a, b) } else { (b, a) }
}

/// The label to persist for a tetrahedral descriptor, or `None` when the
/// descriptor carries no perceived fact.
fn tetrahedral_label(s: TetrahedralStereo) -> Option<&'static str> {
    match s {
        TetrahedralStereo::CW => Some("CW"),
        TetrahedralStereo::CCW => Some("CCW"),
        TetrahedralStereo::Unspecified => None,
    }
}

/// The label to persist for a bond descriptor, or `None` when the descriptor is
/// the "not a stereo bond" sentinel.
fn bond_label(s: BondStereo) -> Option<&'static str> {
    match s {
        BondStereo::E => Some("E"),
        BondStereo::Z => Some("Z"),
        BondStereo::Either => Some("either"),
        BondStereo::None => None,
    }
}

/// Narrow a count to the `i32` that [`crate::system::molgraph::PropValue`]
/// stores, saturating instead of wrapping (counts never approach the bound in
/// practice).
fn saturating_i32(n: usize) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}
