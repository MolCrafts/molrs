//! Fragment graph with named attachment points (ports).
//!
//! A *fragment* is a molecular graph that is deliberately incomplete: it
//! carries named valences — **ports** — that are satisfied only when the
//! fragment is joined to a neighbour. [`Fragment`] is the third newtype over
//! [`MolGraph`], peer to [`Atomistic`](crate::system::atomistic::Atomistic)
//! (every node is an atom) and
//! [`CoarseGrain`](crate::system::coarsegrain::CoarseGrain) (every node is a
//! bead): its nodes are atoms, and it owns two relation kinds, `bonds` and
//! `ports`.
//!
//! A port is an arity-2 relation `(anchor, handle)`: the **anchor** is the atom
//! that keeps its place in the product molecule, and the **handle** is a real
//! capping hydrogen bonded to it — which is what an unconsumed BigSMILES /
//! CGsmiles descriptor means, and what a paired descriptor's bond replaces.
//! Storing only the anchor would lose which valence of a multivalent anchor was
//! meant. The descriptor itself is a [`PortKind`] (the closed four-glyph
//! vocabulary), a free-form `label` and a [`BondNumber`] order.
//!
//! References: Lin, T.-S. et al., *BigSMILES: A Structurally-Based Line Notation
//! for Describing Macromolecules*, ACS Cent. Sci. **5**, 1523–1531 (2019),
//! DOI [10.1021/acscentsci.9b00476](https://doi.org/10.1021/acscentsci.9b00476);
//! the CGsmiles documentation, <https://cgsmiles.readthedocs.io>.
//!
//! # Reserved open properties
//!
//! This module reserves five names that no schema declares — they are **open**
//! props, unconstrained by [`check_schema`](crate::store::block::Block) and
//! round-tripping through [`Fragment::to_frame`] as plain columns:
//!
//! | Name | Where | Meaning |
//! |---|---|---|
//! | `frag_id` | node prop, `Int` | the fragment instance an atom came from |
//! | `port_kind` | `ports` relation prop, `Str` | the descriptor glyph |
//! | `port_label` | `ports` relation prop, `Str` | the descriptor label |
//! | `port_order` | `ports` relation prop, `Int` | the [`BondNumber`] code |
//! | `ports` | relation kind / frame block | the port table |
//!
//! The pending schema-vocabulary spec declares the fragment / port vocabulary
//! as a whole and reconciles `frag_id` with the biopolymer residue key; until
//! then nothing under `core/store/schema/` names them and no constant spells
//! them. The prefix on the port triple is deliberate: bare `order` is already
//! written as an `F64` relation prop by the UFF typifier and bare `label` by
//! the graph itself, and a frame schema binds a key across *every* block.
//!
//! # Two enums for four roles
//!
//! [`PortKind`] and the notation-side
//! [`DescriptorKind`](crate::io::smiles::DescriptorKind) name the same four
//! roles and are deliberately distinct types, mirroring the existing split
//! between the SMILES AST's `BondKind` and this layer's
//! [`BondType`] / [`BondNumber`]: the AST names what was *written*, `core`
//! names what is *stored*. `core` does not name `io`, and the single conversion
//! between the two lives on the notation side.
//!
//! # Examples
//!
//! ```
//! use molrs::system::bond::BondNumber;
//! use molrs::system::fragment::{Fragment, PortKind};
//!
//! // C–C with one capping hydrogen on the first carbon, and a symmetric
//! // descriptor `[$A]` sitting on that C–H valence.
//! let mut frag = Fragment::new();
//! let c0 = frag.add_atom_xyz("C", 0.0, 0.0, 0.0);
//! let c1 = frag.add_atom_xyz("C", 1.54, 0.0, 0.0);
//! let h = frag.add_atom_bare("H");
//! frag.add_bond(c0, c1)?;
//! frag.add_bond(c0, h)?;
//! let port = frag.add_port(c0, h, PortKind::Symmetric, "A", BondNumber::Single)?;
//!
//! for atom in [c0, c1, h] {
//!     frag.set_frag_id(atom, 1)?;
//! }
//!
//! assert_eq!(frag.n_atoms(), 3);
//! assert_eq!(frag.n_bonds(), 2);
//! assert_eq!(frag.n_ports(), 1);
//!
//! let port = frag.port(port)?;
//! assert_eq!(port.kind, PortKind::Symmetric);
//! assert_eq!(port.label, "A");
//! assert_eq!(port.order, BondNumber::Single);
//! assert_eq!(frag.frag_id(c0), Some(1));
//! # Ok::<(), molrs::MolRsError>(())
//! ```

use std::ops::{Deref, DerefMut};
use std::str::FromStr;

use crate::error::MolRsError;
use crate::store::frame::Frame;
use crate::store::keys;
use crate::system::atomistic::{AtomId, BondId};
use crate::system::bond::{BondNumber, BondType, write_bond_class};
use crate::system::molgraph::{Atom, KindId, MolGraph, PropValue, RelationId};

/// Handle to a port (a relation of the `ports` kind).
///
/// Mirrors [`BondId`]: distinct-key semantics
/// for a `HashSet<PortId>`.
pub type PortId = RelationId;

/// The closed descriptor vocabulary: what *role* a port plays when two ports
/// are paired.
///
/// The four glyphs are fixed by the BigSMILES / CGsmiles grammar, not an open
/// string: `<` bonds only `>`, `$` bonds any `$` of the same label, and `!` is
/// the CGsmiles shared descriptor. Inside the type system a bad descriptor is
/// unrepresentable, so validation lives at the one string boundary —
/// [`FromStr`].
///
/// **Storage form.** Unlike [`BondType`], whose stored form is a numeric
/// [`code`](BondType::code), a `PortKind` is stored as its glyph in a `Str`
/// column (see the module docs on `port_kind`): the glyph is the only spelling
/// a user ever writes or reads, and `port_kind` is an open column with no
/// declared dtype to pin a code table to.
///
/// This enum is the **stored** vocabulary; the notation-side
/// [`DescriptorKind`](crate::io::smiles::DescriptorKind) is the *written* one,
/// and the two are distinct types by design (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortKind {
    /// `$` — self-complementary: bonds any `$` carrying the same label.
    Symmetric,
    /// `<` — bonds only a `>` carrying the same label.
    Left,
    /// `>` — bonds only a `<` carrying the same label.
    Right,
    /// `!` — the CGsmiles shared descriptor.
    Shared,
}

impl PortKind {
    /// The grammar glyph, which is also this role's stored spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            PortKind::Symmetric => "$",
            PortKind::Left => "<",
            PortKind::Right => ">",
            PortKind::Shared => "!",
        }
    }
}

impl FromStr for PortKind {
    type Err = MolRsError;

    /// Read a glyph back into a role.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] for anything but the four grammar
    /// glyphs `$`, `<`, `>` and `!` — the vocabulary is closed, so an unknown
    /// spelling is a data error, never a fallback role.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "$" => Ok(PortKind::Symmetric),
            "<" => Ok(PortKind::Left),
            ">" => Ok(PortKind::Right),
            "!" => Ok(PortKind::Shared),
            other => Err(MolRsError::validation(format!(
                "'{other}' is not a port descriptor; expected one of $ < > !"
            ))),
        }
    }
}

/// A materialized port: the two atoms it spans and the descriptor it carries.
///
/// Read out of a [`Fragment`] by [`Fragment::port`]; the fragment stores the
/// three descriptor facts as relation props, never as a side table.
///
/// `label` is free-form text taken from the input, and `""` means **unnamed**
/// (a bare `$` / `<` / `>` with no label after it). `order` is the multiplicity
/// the bond formed from this port will have; it is recorded here and never
/// interpreted — forming the bond belongs to the caller that pairs two ports.
#[derive(Debug, Clone, PartialEq)]
pub struct Port {
    /// The fragment atom that keeps its place in the product molecule.
    pub anchor: AtomId,
    /// The capping hydrogen this descriptor sits on.
    pub handle: AtomId,
    /// The descriptor's role.
    pub kind: PortKind,
    /// The descriptor's label; `""` when unnamed.
    pub label: String,
    /// The multiplicity of the bond this port will form.
    pub order: BondNumber,
}

/// Molecular graph with named attachment points.
///
/// Invariant: every node carries the canonical [`keys::ELEMENT`] property, as
/// on [`Atomistic`](crate::system::atomistic::Atomistic) — a handle is a real
/// hydrogen, never a dummy atom, because a conformer needs an element.
///
/// Generic graph methods (`nodes`, `neighbors`, `remove_relation`, …) remain
/// available through `Deref` / `DerefMut`, with two consequences worth stating:
///
/// * The anchor–handle bond a port names is checked **once**, at
///   [`add_port`](Self::add_port). It is not an invariant maintained across
///   `DerefMut`: a caller that removes that bond through the inner graph leaves
///   a port whose handle is unbonded, and [`port`](Self::port) — which
///   validates the three descriptor props only — will still read it back.
///   Hydrogen-stripping passes are the practical hazard: removing a fragment's
///   hydrogens removes its handles, and the ports left behind are dangling.
/// * A `frag_id` an atom does not carry is emitted to a [`Frame`] as a null
///   cell, not as fragment instance zero (see [`to_frame`](Self::to_frame)).
///
/// The open props this type reserves — `frag_id`, `port_kind`, `port_label`,
/// `port_order` and the `ports` block — are listed in the module docs.
#[derive(Debug, Clone)]
pub struct Fragment {
    graph: MolGraph,
    bond: KindId,
    port: KindId,
}

impl Deref for Fragment {
    type Target = MolGraph;
    fn deref(&self) -> &MolGraph {
        &self.graph
    }
}

impl DerefMut for Fragment {
    fn deref_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }
}

impl Default for Fragment {
    fn default() -> Self {
        Self::new()
    }
}

impl Fragment {
    /// Create an empty fragment with the `bonds` and `ports` kinds registered,
    /// both arity 2.
    ///
    /// Registration order carries no meaning: every consumer — this type, the
    /// frame round trip, the sibling promotions — resolves a kind by name.
    pub fn new() -> Self {
        let mut graph = MolGraph::new();
        let bond = graph.register_kind("bonds", 2);
        let port = graph.register_kind("ports", 2);
        Self { graph, bond, port }
    }

    /// Promote from a [`MolGraph`], validating the element invariant.
    ///
    /// The `bonds` and `ports` kinds are resolved **by name**: an existing kind
    /// of arity 2 keeps its id, a missing one is registered fresh. Ports are
    /// not scanned, so this stays O(nodes) like its siblings.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when a node carries no
    /// [`keys::ELEMENT`] (naming the node), and when the graph already spells
    /// `bonds` or `ports` at an arity other than 2 (naming the kind and both
    /// arities).
    pub fn try_from_molgraph(mut mol: MolGraph) -> Result<Self, MolRsError> {
        Self::check_elements(&mol)?;
        let bond = mol.try_register_kind("bonds", 2)?;
        let port = mol.try_register_kind("ports", 2)?;
        Ok(Self {
            graph: mol,
            bond,
            port,
        })
    }

    /// Every node carries [`keys::ELEMENT`], or the first one that does not is
    /// named in the error.
    fn check_elements(mol: &MolGraph) -> Result<(), MolRsError> {
        for (id, atom) in mol.nodes() {
            if atom.get_str(keys::ELEMENT).is_none() {
                return Err(MolRsError::validation(format!(
                    "node {:?} missing '{}' property",
                    id,
                    keys::ELEMENT
                )));
            }
        }
        Ok(())
    }

    /// Unwrap to the inner [`MolGraph`] (zero cost).
    pub fn into_inner(self) -> MolGraph {
        self.graph
    }

    /// Borrow the inner [`MolGraph`].
    pub fn as_molgraph(&self) -> &MolGraph {
        &self.graph
    }

    /// Mutably borrow the inner [`MolGraph`].
    pub fn as_molgraph_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }

    /// Translate every atom that has coordinates by `delta`.
    pub fn translate(&mut self, delta: [f64; 3]) {
        crate::spatial::geometry::translate(self.as_molgraph_mut(), delta);
    }

    /// Scale every atom that has coordinates by a per-axis `factor` about
    /// `about` (the origin when `None`). Pass `[s, s, s]` for a uniform scale.
    pub fn scale(&mut self, factor: [f64; 3], about: Option<[f64; 3]>) {
        crate::spatial::geometry::scale(self.as_molgraph_mut(), factor, about);
    }

    /// Rotate every atom that has coordinates by `angle` radians about `axis`.
    /// `about` defaults to the origin when `None`.
    ///
    /// # Errors
    ///
    /// The error of [`crate::spatial::geometry::rotate`] — `axis` has no
    /// direction or `angle` is not finite; nothing moves then.
    pub fn rotate(
        &mut self,
        axis: [f64; 3],
        angle: f64,
        about: Option<[f64; 3]>,
    ) -> Result<(), crate::error::MolRsError> {
        crate::spatial::geometry::rotate(self.as_molgraph_mut(), axis, angle, about)
    }

    // ---- atoms (nodes) ----

    /// Add an atom with element symbol and 3D coordinates (Å).
    ///
    /// # Panics
    ///
    /// Panics when a value of the bag this builds contradicts the element
    /// type an existing atom column holds for that key (a string `x` where
    /// `x` is an `f64` column). The bag is built here out of typed arguments,
    /// so that is a defect in the graph's own vocabulary and not a data
    /// condition; the generic
    /// [`MolGraph::add_node_with`](crate::system::molgraph::MolGraph::add_node_with)
    /// returns the conflict for callers holding a foreign bag.
    pub fn add_atom_xyz(&mut self, symbol: &str, x: f64, y: f64, z: f64) -> AtomId {
        self.graph
            .add_node_with(Atom::xyz(symbol, x, y, z))
            .expect("caller-built atom bag contradicts an existing atom column")
    }

    /// Add an atom with element symbol only (no coordinates).
    ///
    /// Writes the chemical identity under the canonical [`keys::ELEMENT`] field
    /// (not a format alias such as `"symbol"`).
    ///
    /// # Panics
    ///
    /// Panics when the `element` column already holds a different element
    /// type — see [`add_atom_xyz`](Self::add_atom_xyz).
    pub fn add_atom_bare(&mut self, symbol: &str) -> AtomId {
        let mut atom = Atom::new();
        atom.set(keys::ELEMENT, symbol);
        self.graph
            .add_node_with(atom)
            .expect("caller-built atom bag contradicts an existing atom column")
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.graph.n_nodes()
    }

    // ---- bonds ----

    /// Add a bond between two existing atoms, classed
    /// [`BondType::Single`] / [`BondNumber::Single`].
    ///
    /// A hand-built template's bonds are classed like every other bond in the
    /// crate: an unclassed bond reads back as `Unknown`, which a valence count
    /// treats as zero.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::NotFound`] when either atom handle is stale or
    /// unknown, and propagates a store refusal of the class props.
    pub fn add_bond(&mut self, a: AtomId, b: AtomId) -> Result<BondId, MolRsError> {
        let bid = self.graph.add_relation(self.bond, &[a, b])?;
        write_bond_class(
            &mut self.graph,
            self.bond,
            bid,
            BondType::Single,
            BondNumber::Single,
        )?;
        Ok(bid)
    }

    /// Number of bonds. Ports are a separate kind and are not counted here.
    pub fn n_bonds(&self) -> usize {
        self.graph.n_relations(self.bond)
    }

    // ---- ports ----

    /// Record a descriptor on the `(anchor, handle)` valence.
    ///
    /// Endpoint order is load-bearing: `anchor` is the atom that stays in the
    /// product molecule, `handle` the capping hydrogen bonded to it that a
    /// paired descriptor's bond replaces. `label` is free-form; `""` means
    /// unnamed. `order` is the multiplicity the formed bond will carry.
    ///
    /// The three descriptor facts are written as the relation props `port_kind`
    /// (the glyph), `port_label` and `port_order`.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::NotFound`] when either handle is stale or unknown,
    /// and [`MolRsError::Validation`] when `handle` is not a hydrogen, when it
    /// is not bonded to `anchor` (checked here and only here — see the type
    /// docs), or when `order` is [`BondNumber::Unknown`], which is not a
    /// definite multiplicity.
    pub fn add_port(
        &mut self,
        anchor: AtomId,
        handle: AtomId,
        kind: PortKind,
        label: &str,
        order: BondNumber,
    ) -> Result<PortId, MolRsError> {
        self.graph.get_node(anchor)?;
        let handle_atom = self.graph.get_node(handle)?;
        let element = handle_atom.get_str(keys::ELEMENT).unwrap_or_default();
        if element != "H" {
            return Err(MolRsError::validation(format!(
                "port handle {handle:?} is '{element}'; a handle is a capping hydrogen"
            )));
        }
        if !self.is_bonded(anchor, handle) {
            return Err(MolRsError::validation(format!(
                "port handle {handle:?} is not bonded to its anchor {anchor:?}"
            )));
        }
        if order == BondNumber::Unknown {
            return Err(MolRsError::validation(
                "a port order is a definite bond number, never Unknown",
            ));
        }
        let pid = self.graph.add_relation(self.port, &[anchor, handle])?;
        self.graph
            .set_relation_prop(self.port, pid, "port_kind", kind.as_str())?;
        self.graph
            .set_relation_prop(self.port, pid, "port_label", label)?;
        self.graph
            .set_relation_prop(self.port, pid, "port_order", PropValue::from(order))?;
        Ok(pid)
    }

    /// Whether `a` and `b` are joined by a bond (not by a port).
    fn is_bonded(&self, a: AtomId, b: AtomId) -> bool {
        self.graph
            .neighbor_relations(a)
            .any(|(kind, _, other)| kind == self.bond && other == b)
    }

    /// Read a port back, validating its descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::NotFound`] when `id` names no live port, and
    /// [`MolRsError::Validation`] when the relation is missing `port_kind`,
    /// `port_label` or `port_order`, when the stored glyph is outside the
    /// grammar, or when the stored order does not read back as a definite
    /// [`BondNumber`] (an out-of-range or negative code reads as `Unknown`,
    /// which is not a legal port order).
    pub fn port(&self, id: PortId) -> Result<Port, MolRsError> {
        let relation = self.graph.get_relation(self.port, id)?;
        let [anchor, handle] = match relation.nodes.as_slice() {
            [anchor, handle] => [*anchor, *handle],
            other => {
                return Err(MolRsError::validation(format!(
                    "port {id:?} spans {} atoms; a port is (anchor, handle)",
                    other.len()
                )));
            }
        };
        let kind = match relation.props.get("port_kind") {
            Some(PropValue::Str(glyph)) => PortKind::from_str(glyph)?,
            _ => return Err(Self::missing_prop(id, "port_kind")),
        };
        let label = match relation.props.get("port_label") {
            Some(PropValue::Str(label)) => label.clone(),
            _ => return Err(Self::missing_prop(id, "port_label")),
        };
        let order = match relation.props.get("port_order") {
            None => return Err(Self::missing_prop(id, "port_order")),
            stored => match BondNumber::from_prop(stored) {
                BondNumber::Unknown => {
                    return Err(MolRsError::validation(format!(
                        "port {id:?} carries 'port_order' {stored:?}, not a definite bond number"
                    )));
                }
                order => order,
            },
        };
        Ok(Port {
            anchor,
            handle,
            kind,
            label,
            order,
        })
    }

    /// The error a port missing one of its three descriptor props yields.
    fn missing_prop(id: PortId, key: &str) -> MolRsError {
        MolRsError::validation(format!("port {id:?} carries no '{key}' property"))
    }

    /// Iterate over the fragment's port handles.
    pub fn ports(&self) -> impl Iterator<Item = PortId> + '_ {
        self.graph.relation_ids(self.port)
    }

    /// Number of ports.
    pub fn n_ports(&self) -> usize {
        self.graph.n_relations(self.port)
    }

    // ---- per-atom fragment membership ----

    /// Record the fragment instance `atom` came from, under the node prop
    /// `frag_id`.
    ///
    /// Membership is a per-atom fact, so there is no whole-fragment broadcast:
    /// a caller relabelling one atom (a hydrogen a pipeline added, say) must not
    /// have to rewrite the rest.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when `id` exceeds [`i32::MAX`] — node
    /// columns are `i32`, so a wider identifier is not storable — and
    /// [`MolRsError::NotFound`] when `atom` is stale or unknown.
    pub fn set_frag_id(&mut self, atom: AtomId, id: u32) -> Result<(), MolRsError> {
        let stored = i32::try_from(id).map_err(|_| {
            MolRsError::validation(format!(
                "frag_id {id} exceeds {}, the widest identifier a node column stores",
                i32::MAX
            ))
        })?;
        self.graph.set_node(atom, "frag_id", PropValue::Int(stored))
    }

    /// The fragment instance `atom` came from, or `None` when it carries no
    /// `frag_id` (or the handle is stale).
    pub fn frag_id(&self, atom: AtomId) -> Option<u32> {
        let stored = self.graph.get_node(atom).ok()?.get_int("frag_id")?;
        u32::try_from(stored).ok()
    }

    /// Propagate each `frag_id` to the unlabelled degree-1 atoms hanging off a
    /// labelled one, returning how many atoms were labelled.
    ///
    /// One pass over a snapshot of the labels this call started with, never a
    /// fixpoint: an atom labelled *by* this pass does not hand its label on, so
    /// a second call on the result labels nothing. Degree is counted over bonds
    /// alone — a port does not make its handle a degree-2 atom.
    ///
    /// Intended for atoms a pipeline newly attached (hydrogens added by a
    /// conformer, for instance). The precondition is exactly that: a degree-1
    /// atom deliberately left unassigned **will** be relabelled.
    pub fn inherit_frag_ids(&mut self) -> usize {
        let labels: Vec<(AtomId, Option<u32>)> = self
            .graph
            .node_ids()
            .map(|atom| (atom, self.frag_id(atom)))
            .collect();
        let mut pending: Vec<(AtomId, u32)> = Vec::new();
        for &(atom, label) in &labels {
            if label.is_some() {
                continue;
            }
            let mut bonded = self
                .graph
                .neighbor_relations(atom)
                .filter(|&(kind, _, _)| kind == self.bond)
                .map(|(_, _, other)| other);
            let (Some(only), None) = (bonded.next(), bonded.next()) else {
                continue;
            };
            if let Some(&(_, Some(inherited))) = labels.iter().find(|&&(id, _)| id == only) {
                pending.push((atom, inherited));
            }
        }
        let mut labelled = 0;
        for (atom, id) in pending {
            if self.set_frag_id(atom, id).is_ok() {
                labelled += 1;
            }
        }
        labelled
    }

    // ---- Frame round trip ----

    /// Emit the tabular [`Frame`]: an `atoms` block, plus a block per non-empty
    /// relation kind (`bonds`, `ports`), with no relabeling — a fragment's
    /// nodes *are* atoms, so unlike
    /// [`CoarseGrain`](crate::system::coarsegrain::CoarseGrain) there is
    /// nothing to rename. A zero-port fragment's frame is therefore byte-for-
    /// byte an atomistic frame.
    ///
    /// **A partially labelled `frag_id` round-trips exactly.** The column
    /// carries the block's [validity mask](crate::store::block::Block::validity),
    /// so an atom with no `frag_id` is a null cell rather than the default `0`
    /// — which would read back as fragment instance zero — and
    /// [`from_frame`](Self::from_frame) leaves it unassigned again. A caller
    /// that wants every atom labelled assigns the labels (or calls
    /// [`inherit_frag_ids`](Self::inherit_frag_ids)) before emitting; the
    /// frame no longer decides that for it.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when an atom or relation property
    /// contradicts the dtype the Frame schema declares for its key; the
    /// message names the refused column. The inner [`MolGraph`] accepts any
    /// value under a key it has no column for, so a string written under
    /// `"x"` is legal in the graph and only refused here.
    pub fn to_frame(&self) -> Result<Frame, MolRsError> {
        self.graph.to_frame()
    }

    /// Build a fragment from the [`Frame`] emitted by [`Self::to_frame`].
    ///
    /// `bonds` and `ports` are registered before the read, which is what lets
    /// their blocks come back as relations: a frame block whose kind is not
    /// registered on the receiver is skipped.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Parse`] when the frame carries no `atoms` block,
    /// and [`MolRsError::Validation`] when an atom row carries no
    /// [`keys::ELEMENT`].
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        let mut fragment = Self::new();
        fragment.graph.read_frame(frame)?;
        Self::check_elements(&fragment.graph)?;
        Ok(fragment)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use ndarray::Array1;

    use super::{Fragment, Port, PortId, PortKind};
    use crate::error::MolRsError;
    use crate::store::block::Block;
    use crate::store::frame::Frame;
    use crate::system::atomistic::AtomId;
    use crate::system::bond::{BondNumber, BondType};
    use crate::system::molgraph::{Atom, MolGraph, PropValue};

    /// `C0–C1` plus one real capping hydrogen bonded to `C0`: the smallest
    /// graph that can carry a legal port. Returns `(fragment, c0, c1, h)`.
    fn ch_template() -> (Fragment, AtomId, AtomId, AtomId) {
        let mut frag = Fragment::new();
        let c0 = frag.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let c1 = frag.add_atom_xyz("C", 1.54, 0.0, 0.0);
        let h = frag.add_atom_bare("H");
        frag.add_bond(c0, c1).unwrap();
        frag.add_bond(c0, h).unwrap();
        (frag, c0, c1, h)
    }

    /// `ch_template` with one `Symmetric` port `(C0, H)` labelled `"A"`.
    fn ported_template() -> (Fragment, AtomId, AtomId, AtomId, PortId) {
        let (mut frag, c0, c1, h) = ch_template();
        let pid = frag
            .add_port(c0, h, PortKind::Symmetric, "A", BondNumber::Single)
            .expect("a bonded H handle on its anchor is a legal port");
        (frag, c0, c1, h, pid)
    }

    // ---- PortKind: the closed vocabulary -----------------------------------

    #[test]
    fn port_kind_glyphs_round_trip() {
        for (kind, glyph) in [
            (PortKind::Symmetric, "$"),
            (PortKind::Left, "<"),
            (PortKind::Right, ">"),
            (PortKind::Shared, "!"),
        ] {
            assert_eq!(kind.as_str(), glyph);
            assert_eq!(PortKind::from_str(glyph).unwrap(), kind);
        }
    }

    #[test]
    fn port_kind_from_str_rejects_unknown_glyph() {
        for bad in ["$1.5", "X", ""] {
            let err =
                PortKind::from_str(bad).expect_err("only the four grammar glyphs are a PortKind");
            assert!(
                matches!(err, MolRsError::Validation { .. }),
                "a bad glyph is a validation error, got {err:?}"
            );
        }
    }

    // ---- construction and promotion ---------------------------------------

    #[test]
    fn new_registers_bonds_and_ports() {
        let frag = Fragment::new();
        let bonds = frag.kind_id("bonds").expect("'bonds' registered by new()");
        let ports = frag.kind_id("ports").expect("'ports' registered by new()");
        assert_eq!(frag.arity(bonds), 2);
        assert_eq!(frag.arity(ports), 2);
        assert_ne!(bonds, ports, "the two kinds are distinct");
        assert_eq!(frag.n_atoms(), 0);
        assert_eq!(frag.n_bonds(), 0);
    }

    #[test]
    fn try_from_molgraph_rejects_node_without_element() {
        let mut graph = MolGraph::new();
        graph.add_node_with(Atom::new()).expect("fixture node");
        let err = Fragment::try_from_molgraph(graph)
            .expect_err("every fragment node must carry an element");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn try_from_molgraph_keeps_registered_kind_ids() {
        let mut graph = MolGraph::new();
        let bonds = graph.register_kind("bonds", 2);
        let ports = graph.register_kind("ports", 2);
        graph
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");

        let frag = Fragment::try_from_molgraph(graph).expect("both kinds already match");
        assert_eq!(frag.kind_id("bonds"), Some(bonds));
        assert_eq!(frag.kind_id("ports"), Some(ports));
        assert_eq!(
            frag.kind_ids().count(),
            2,
            "re-registration is idempotent: no extra kind appears"
        );
    }

    #[test]
    fn try_from_molgraph_rejects_conflicting_arity() {
        let mut graph = MolGraph::new();
        graph.register_kind("ports", 3);
        let err = Fragment::try_from_molgraph(graph)
            .expect_err("a 3-ary 'ports' kind conflicts with the fragment's own");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    // ---- bonds -------------------------------------------------------------

    #[test]
    fn add_bond_round_trips() {
        let (frag, c0, _c1, _h) = ch_template();
        assert_eq!(frag.n_atoms(), 3);
        assert_eq!(frag.n_bonds(), 2);

        let bonds = frag.kind_id("bonds").expect("'bonds' registered");
        let bid = frag
            .relation_ids(bonds)
            .next()
            .expect("the first bond exists");
        let bond = frag.get_relation(bonds, bid).unwrap();
        assert_eq!(
            BondType::from_prop(bond.props.get("bond_type")),
            BondType::Single,
            "a fragment bond is classed, never the one classless bond"
        );
        assert_eq!(
            BondNumber::from_prop(bond.props.get("bond_number")),
            BondNumber::Single
        );
        assert_eq!(frag.neighbors(c0).count(), 2);
    }

    // ---- ports -------------------------------------------------------------

    #[test]
    fn add_port_rejects_stale_atom() {
        let (mut frag, c0, _c1, h) = ch_template();
        frag.remove_node(h)
            .expect("the handle exists before removal");
        let err = frag
            .add_port(c0, h, PortKind::Symmetric, "A", BondNumber::Single)
            .expect_err("a stale handle cannot be a port endpoint");
        assert!(matches!(err, MolRsError::NotFound { .. }), "{err:?}");
    }

    #[test]
    fn add_port_rejects_non_hydrogen_handle() {
        let (mut frag, c0, c1, _h) = ch_template();
        // C1 is bonded to C0, so only the element rule can reject it.
        let err = frag
            .add_port(c0, c1, PortKind::Symmetric, "A", BondNumber::Single)
            .expect_err("a handle is a real capping hydrogen");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_eq!(frag.n_ports(), 0, "a rejected port leaves no relation");
    }

    #[test]
    fn add_port_rejects_unbonded_handle() {
        let (mut frag, c0, _c1, _h) = ch_template();
        let lone = frag.add_atom_bare("H");
        let err = frag
            .add_port(c0, lone, PortKind::Symmetric, "A", BondNumber::Single)
            .expect_err("the handle must be bonded to its anchor");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_eq!(frag.n_ports(), 0, "a rejected port leaves no relation");
    }

    #[test]
    fn add_port_rejects_unknown_order() {
        let (mut frag, c0, _c1, h) = ch_template();
        let err = frag
            .add_port(c0, h, PortKind::Symmetric, "A", BondNumber::Unknown)
            .expect_err("a port always has a definite multiplicity");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_eq!(frag.n_ports(), 0, "a rejected port leaves no relation");
    }

    #[test]
    fn add_port_records_kind_label_and_order() {
        let (mut frag, c0, _c1, h) = ch_template();
        let pid = frag
            .add_port(c0, h, PortKind::Left, "A", BondNumber::Double)
            .expect("a bonded H handle on its anchor is a legal port");

        let port: Port = frag.port(pid).expect("the port reads back");
        assert_eq!(port.anchor, c0);
        assert_eq!(port.handle, h);
        assert_eq!(port.kind, PortKind::Left);
        assert_eq!(port.label, "A");
        assert_eq!(port.order, BondNumber::Double);

        // Storage form: the glyph as a Str, the order as an Int code.
        let ports = frag.kind_id("ports").expect("'ports' registered");
        let rel = frag.get_relation(ports, pid).unwrap();
        assert_eq!(
            rel.props.get("port_kind"),
            Some(&PropValue::Str("<".into()))
        );
        assert_eq!(
            rel.props.get("port_label"),
            Some(&PropValue::Str("A".into()))
        );
        assert_eq!(rel.props.get("port_order"), Some(&PropValue::Int(2)));
    }

    #[test]
    fn port_rejects_missing_or_unmapped_props() {
        // A missing descriptor prop.
        let (mut frag, _c0, _c1, _h, pid) = ported_template();
        let ports = frag.kind_id("ports").expect("'ports' registered");
        frag.clear_relation_prop(ports, pid, "port_kind").unwrap();
        let err = frag
            .port(pid)
            .expect_err("a port without its kind is unreadable");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");

        // A glyph outside the grammar.
        let (mut frag, _c0, _c1, _h, pid) = ported_template();
        let ports = frag.kind_id("ports").expect("'ports' registered");
        frag.set_relation_prop(ports, pid, "port_kind", "Z")
            .unwrap();
        let err = frag.port(pid).expect_err("'Z' is not a descriptor");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");

        // An order code outside the vocabulary, and a negative one: both read
        // back as `BondNumber::Unknown`, which is not a legal port order.
        for bad in [PropValue::Int(99), PropValue::Int(-1)] {
            let (mut frag, _c0, _c1, _h, pid) = ported_template();
            let ports = frag.kind_id("ports").expect("'ports' registered");
            frag.set_relation_prop(ports, pid, "port_order", bad.clone())
                .unwrap();
            let err = frag
                .port(pid)
                .expect_err("an indefinite bond number is not a port order");
            assert!(
                matches!(err, MolRsError::Validation { .. }),
                "{bad:?}: {err:?}"
            );
        }
    }

    #[test]
    fn n_ports_counts_two_ports_on_one_anchor() {
        let (mut frag, c0, _c1, h) = ch_template();
        let h2 = frag.add_atom_bare("H");
        frag.add_bond(c0, h2).unwrap();

        let p1 = frag
            .add_port(c0, h, PortKind::Left, "a", BondNumber::Single)
            .unwrap();
        let p2 = frag
            .add_port(c0, h2, PortKind::Symmetric, "b", BondNumber::Single)
            .unwrap();
        assert_ne!(p1, p2);
        assert_eq!(frag.n_ports(), 2);

        let ids: Vec<PortId> = frag.ports().collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&p1) && ids.contains(&p2));
        assert_eq!(frag.n_bonds(), 3, "a port is not a bond");
    }

    // ---- per-atom fragment membership --------------------------------------

    #[test]
    fn set_frag_id_per_atom_round_trips() {
        let (mut frag, c0, c1, h) = ch_template();
        frag.set_frag_id(c0, 1).unwrap();
        frag.set_frag_id(c1, 7).unwrap();
        assert_eq!(frag.frag_id(c0), Some(1));
        assert_eq!(frag.frag_id(c1), Some(7));
        assert_eq!(
            frag.frag_id(h),
            None,
            "membership is per atom, not broadcast"
        );
    }

    #[test]
    fn set_frag_id_rejects_above_i32_max() {
        let (mut frag, c0, _c1, _h) = ch_template();
        let too_big = i32::MAX as u32 + 1;
        let err = frag
            .set_frag_id(c0, too_big)
            .expect_err("node columns are i32; a wider id is not storable");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_eq!(frag.frag_id(c0), None, "a rejected id is not written");
    }

    #[test]
    fn inherit_frag_ids_labels_degree_one_neighbours() {
        let (mut frag, c0, c1, h) = ch_template();
        frag.set_frag_id(c0, 3).unwrap();
        frag.set_frag_id(c1, 3).unwrap();
        frag.inherit_frag_ids();
        assert_eq!(frag.frag_id(h), Some(3));
    }

    #[test]
    fn inherit_frag_ids_leaves_orphans_none() {
        let (mut frag, c0, c1, h) = ch_template();
        assert_eq!(frag.inherit_frag_ids(), 0);
        for atom in [c0, c1, h] {
            assert_eq!(frag.frag_id(atom), None, "no label exists to inherit");
        }
    }

    #[test]
    fn inherit_frag_ids_does_not_overwrite() {
        let (mut frag, c0, c1, h) = ch_template();
        frag.set_frag_id(c0, 3).unwrap();
        frag.set_frag_id(c1, 3).unwrap();
        frag.set_frag_id(h, 9).unwrap();
        assert_eq!(frag.inherit_frag_ids(), 0, "nothing is left to label");
        assert_eq!(frag.frag_id(h), Some(9), "an assigned atom is left alone");
    }

    #[test]
    fn inherit_frag_ids_returns_count() {
        let (mut frag, c0, c1, h) = ch_template();
        let h2 = frag.add_atom_bare("H");
        frag.add_bond(c0, h2).unwrap();
        frag.set_frag_id(c0, 4).unwrap();
        frag.set_frag_id(c1, 4).unwrap();
        assert_eq!(
            frag.inherit_frag_ids(),
            2,
            "both hydrogens on C0 are labelled"
        );
        assert_eq!(frag.frag_id(h), Some(4));
        assert_eq!(frag.frag_id(h2), Some(4));
    }

    #[test]
    fn inherit_frag_ids_is_one_pass() {
        // H–C0(labelled)–C1(unlabelled)–H2: one pass labels only the hydrogen
        // on the labelled carbon; C1 is degree 2 and never inherits, so a
        // second pass has nothing left to do.
        let (mut frag, c0, c1, h) = ch_template();
        let h2 = frag.add_atom_bare("H");
        frag.add_bond(c1, h2).unwrap();
        frag.set_frag_id(c0, 5).unwrap();

        assert_eq!(frag.inherit_frag_ids(), 1);
        assert_eq!(frag.frag_id(h), Some(5));
        assert_eq!(frag.frag_id(c1), None, "a degree-2 atom never inherits");
        assert_eq!(frag.frag_id(h2), None);
        assert_eq!(frag.inherit_frag_ids(), 0, "not a fixpoint");
    }

    // ---- Frame round trip ---------------------------------------------------

    #[test]
    fn to_frame_emits_atoms_bonds_and_ports_blocks() {
        let (frag, _c0, _c1, _h, _pid) = ported_template();
        let frame = frag.to_frame().expect("a schema-conforming graph converts");

        assert!(frame.contains_key("atoms"), "a fragment's nodes are atoms");
        assert!(frame.contains_key("bonds"), "no block relabeling");
        assert!(frame.contains_key("ports"));
        assert!(!frame.contains_key("beads"));

        let ports = frame.get("ports").expect("ports block");
        assert_eq!(ports.nrows(), Some(1));
        for col in ["atomi", "atomj", "port_kind", "port_label", "port_order"] {
            assert!(ports.contains_key(col), "ports block carries '{col}'");
        }
    }

    #[test]
    fn to_frame_masks_unlabelled_frag_id() {
        let (mut frag, c0, _c1, _h) = ch_template();
        frag.set_frag_id(c0, 1).unwrap();
        let frame = frag.to_frame().expect("a schema-conforming graph converts");
        let atoms = frame.get("atoms").expect("atoms block");
        assert!(
            atoms.contains_key("frag_id"),
            "the label of the one labelled atom reaches the frame"
        );
        assert_eq!(
            atoms.validity("frag_id"),
            Some(&[true, false, false][..]),
            "an unassigned frag_id is a null cell, not fragment instance 0"
        );
    }

    /// A `Fragment` exposes the graph's own setter through `DerefMut`, and
    /// that setter carries the Frame schema's dtype opinion: a string under
    /// the float key `x` never becomes an atom property of a fragment either.
    #[test]
    fn set_node_through_the_fragment_refuses_a_str_under_a_schema_float_key() {
        use crate::store::block::DType;
        let mut frag = Fragment::new();
        let a = frag.add_atom_bare("C");

        let err = frag
            .set_node(a, "x", "left")
            .expect_err("a str cannot be stored at a schema-float key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("'x'"), "the error names the key, got {msg}");
        assert!(
            msg.contains(DType::Float.name()) && msg.contains(DType::String.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    #[test]
    fn to_frame_omits_ports_block_when_no_ports() {
        let (frag, _c0, _c1, _h) = ch_template();
        let frame = frag.to_frame().expect("a schema-conforming graph converts");
        assert!(frame.contains_key("atoms"));
        assert!(frame.contains_key("bonds"));
        assert!(
            !frame.contains_key("ports"),
            "an empty kind emits no block, so a zero-port frame is an atomistic frame"
        );
    }

    #[test]
    fn from_frame_restores_ports_bonds_and_frag_id() {
        let (mut frag, c0, c1, h, _pid) = ported_template();
        frag.set_frag_id(c0, 2).unwrap();
        frag.set_frag_id(c1, 2).unwrap();
        frag.set_frag_id(h, 2).unwrap();

        let restored =
            Fragment::from_frame(&frag.to_frame().expect("a schema-conforming graph converts"))
                .expect("a fragment frame reads back");
        assert_eq!(restored.n_atoms(), 3);
        assert_eq!(restored.n_bonds(), 2);
        assert_eq!(restored.n_ports(), 1);

        let pid = restored.ports().next().expect("the port survives");
        let port = restored.port(pid).expect("the descriptor reads back");
        assert_eq!(port.kind, PortKind::Symmetric);
        assert_eq!(port.label, "A");
        assert_eq!(port.order, BondNumber::Single);

        let ids: Vec<Option<u32>> = restored.node_ids().map(|n| restored.frag_id(n)).collect();
        assert_eq!(ids, vec![Some(2), Some(2), Some(2)]);
    }

    /// The partially labelled counterpart of
    /// `from_frame_restores_ports_bonds_and_frag_id`: the labelled atom keeps
    /// its id and the unlabelled ones come back unlabelled.
    #[test]
    fn from_frame_restores_a_partially_labelled_frag_id() {
        let (mut frag, c0, _c1, _h) = ch_template();
        frag.set_frag_id(c0, 2).unwrap();

        let restored =
            Fragment::from_frame(&frag.to_frame().expect("a schema-conforming graph converts"))
                .expect("a fragment frame reads back");
        let ids: Vec<Option<u32>> = restored.node_ids().map(|n| restored.frag_id(n)).collect();
        assert_eq!(ids, vec![Some(2), None, None]);
    }

    #[test]
    fn from_frame_rejects_frame_without_atoms() {
        let frame = Frame::new();
        assert!(
            Fragment::from_frame(&frame).is_err(),
            "a frame with no atoms block is not a fragment"
        );
    }

    #[test]
    fn from_frame_rejects_atoms_without_element() {
        let mut atoms = Block::new();
        for key in ["x", "y", "z"] {
            atoms
                .insert(key, Array1::from_vec(vec![0.0_f64, 1.0]).into_dyn())
                .unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let err = Fragment::from_frame(&frame)
            .expect_err("the element invariant holds on the way in too");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn scale_multiplies_offsets_from_the_centre_per_axis() {
        // p' = (p - c) * f + c. Point (1, 2, 3), factor (2, 3, 0.5):
        //   about c = (1, 1, 1) -> (1, 4, 2);  about the origin -> (2, 6, 1.5).
        let factor = [2.0, 3.0, 0.5];
        let cases: [(Option<[f64; 3]>, [f64; 3]); 2] = [
            (Some([1.0, 1.0, 1.0]), [1.0, 4.0, 2.0]),
            (None, [2.0, 6.0, 1.5]),
        ];
        for (about, expected) in cases {
            let mut sys = Fragment::new();
            let id = sys.add_atom_xyz("C", 1.0, 2.0, 3.0);
            let fixed = sys.add_atom_xyz("C", 1.0, 1.0, 1.0);
            sys.scale(factor, about);
            let moved = sys.get_node(id).expect("live handle");
            for (key, want) in ["x", "y", "z"].into_iter().zip(expected) {
                let got = moved.get_f64(key).expect("coordinate kept");
                assert!(
                    (got - want).abs() < 1e-12,
                    "{about:?} {key}: {got} != {want}"
                );
            }
            if about.is_some() {
                // The centre itself is a fixed point of the map.
                let centre = sys.get_node(fixed).expect("live handle");
                for key in ["x", "y", "z"] {
                    let got = centre.get_f64(key).expect("coordinate kept");
                    assert!((got - 1.0).abs() < 1e-12, "centre {key} moved to {got}");
                }
            }
        }
    }
}
