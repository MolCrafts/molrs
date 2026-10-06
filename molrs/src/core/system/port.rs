//! Ports: named attachment points on any [`MolGraph`].
//!
//! A **port** marks one not-yet-used bonding site of a molecular graph: it is
//! an arity-2 relation `(anchor, handle)` of the kind `ports`. The **anchor**
//! is the node that keeps its place in the product molecule; the **handle** is
//! a real node bonded to it, the root of the leaving group a paired
//! descriptor's bond replaces. A **descriptor** is the bracketed site marker
//! of the BigSMILES / CGsmiles line notations (`[$]`, `[<]`, `[>]`, `[!]`)
//! that says where a unit may bond to another. The handle may be any element:
//! an unconsumed descriptor is a capping hydrogen, and a hydroxyl leaving group
//! roots at its O. The leaving group is the handle's branch — what stays
//! connected to the handle once the anchor–handle bond is cut. A **valence**
//! is one specific anchor–handle bond, the bonding slot a port occupies. The
//! descriptor itself is a [`PortKind`] (the closed four-glyph vocabulary), a
//! free-form `label` and a [`BondNumber`] order.
//!
//! Ports are a capability of every graph type — [`Atomistic`], [`CoarseGrain`]
//! or a bare [`MolGraph`] — not a type of their own (operator, 2026-09-28):
//! the `ports` kind is registered on the first [`MolGraph::add_port`] and read
//! back by every graph's `from_frame`. [`MolGraph::link`] joins two
//! compatible ports: it removes both leaving groups, folds their charge onto
//! the anchors and bonds the anchors (see [`crate::system::link`]).
//!
//! References: Lin, T.-S. et al., *BigSMILES: A Structurally-Based Line Notation
//! for Describing Macromolecules*, ACS Cent. Sci. **5**, 1523–1531 (2019),
//! DOI [10.1021/acscentsci.9b00476](https://doi.org/10.1021/acscentsci.9b00476);
//! the CGsmiles documentation, <https://cgsmiles.readthedocs.io>.
//!
//! # Reserved open properties
//!
//! | Name | Where | Meaning |
//! |---|---|---|
//! | `frag_id` | node prop, `Int` | the unit instance a node came from |
//! | `port_kind` | `ports` relation prop, `Str` | the descriptor glyph |
//! | `port_label` | `ports` relation prop, `Str` | the descriptor label |
//! | `port_order` | `ports` relation prop, `Int` | the [`BondNumber`] code |
//! | `ports` | relation kind / frame block | the port table |
//!
//! None is declared by the Frame schema; they round-trip as plain columns. The
//! prefix on the port triple is deliberate: bare `order` is already written as
//! an `F64` relation prop by the UFF typifier and bare `label` by the graph
//! itself, and a frame schema binds a key across *every* block.
//!
//! # Two enums for four roles
//!
//! [`PortKind`] and the notation-side `crate::io::smiles::DescriptorKind`
//! (feature `smiles`) name the same four roles and are deliberately distinct
//! types, mirroring the split between the SMILES AST's `BondKind` and this
//! layer's [`BondType`] / [`BondNumber`]: the AST names what was *written*,
//! `core` names what is *stored*.
//!
//! # Examples
//!
//! ```
//! use molrs::system::atomistic::Atomistic;
//! use molrs::system::bond::BondNumber;
//! use molrs::system::port::PortKind;
//!
//! // C–C with one capping hydrogen on the first carbon, and a symmetric
//! // descriptor `[$A]` sitting on that C–H valence.
//! let mut mol = Atomistic::new();
//! let c0 = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
//! let c1 = mol.add_atom_xyz("C", 1.54, 0.0, 0.0);
//! let h = mol.add_atom_bare("H");
//! mol.add_bond(c0, c1)?;
//! mol.add_bond(c0, h)?;
//! let port = mol.add_port(c0, h, PortKind::Symmetric, "A", BondNumber::Single)?;
//! mol.set_frag_id(c0, 1)?;
//!
//! assert_eq!(mol.n_ports(), 1);
//! let port = mol.port(port)?;
//! assert_eq!(port.kind, PortKind::Symmetric);
//! assert_eq!(port.label, "A");
//! assert_eq!(mol.frag_id(c0), Some(1));
//! # Ok::<(), molrs::MolRsError>(())
//! ```
//!
//! [`Atomistic`]: crate::system::atomistic::Atomistic
//! [`CoarseGrain`]: crate::system::coarsegrain::CoarseGrain

use std::collections::{BTreeSet, HashMap};
use std::str::FromStr;

use slotmap::Key;

use crate::error::MolRsError;
use crate::system::bond::{BondNumber, BondType, write_bond_class};
use crate::system::molgraph::{FRAG_ID, KindId, MolGraph, NodeId, PropValue, RelationId};

/// The relation-kind name ports are stored under.
pub const PORTS: &str = "ports";

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
/// `crate::io::smiles::DescriptorKind` is the *written* one, and the two are
/// distinct types by design (see the module docs).
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

    /// The role a partner port must play: `<` ↔ `>`, while `$` and `!` are
    /// self-complementary.
    ///
    /// An involution: `k.complement().complement() == k`.
    pub fn complement(self) -> PortKind {
        match self {
            PortKind::Left => PortKind::Right,
            PortKind::Right => PortKind::Left,
            PortKind::Symmetric => PortKind::Symmetric,
            PortKind::Shared => PortKind::Shared,
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
/// Read out of a graph by [`MolGraph::port`]; the graph stores the three
/// descriptor facts as relation props, never as a side table.
///
/// `label` is free-form text taken from the input, and `""` means **unnamed**
/// (a bare `$` / `<` / `>` with no label after it). `order` is the multiplicity
/// the bond formed from this port will have; it is recorded here and never
/// interpreted — forming the bond belongs to the caller that pairs two ports.
#[derive(Debug, Clone, PartialEq)]
pub struct Port {
    /// The atom that keeps its place in the product molecule.
    pub anchor: NodeId,
    /// The root of the leaving group this descriptor sits on: bonded to
    /// `anchor`, any element.
    pub handle: NodeId,
    /// The descriptor's role.
    pub kind: PortKind,
    /// The descriptor's label; `""` when unnamed.
    pub label: String,
    /// The multiplicity of the bond this port will form.
    pub order: BondNumber,
}

impl Port {
    /// Whether `other` may pair with this port: `other.kind` is this kind's
    /// [`complement`](PortKind::complement), and the labels and orders are
    /// equal.
    ///
    /// This is **the one** port-compatibility rule on stored ports, following
    /// the CGsmiles pairing rule (Grünewald et al., *J. Chem. Inf. Model.*
    /// 2025, doi:10.1021/acs.jcim.5c00064).
    /// [`MolGraph::link`] pairs ports through it. The CGsmiles
    /// resolver keeps its own private rule on the notation-side
    /// `DescriptorKind` (in `io::smiles`); the two enums are distinct by
    /// design (module docs, "Two enums for four roles").
    ///
    /// Symmetric in its arguments, because [`PortKind::complement`] is an
    /// involution. The anchor and handle atoms are not compared.
    pub fn accepts(&self, other: &Port) -> bool {
        other.kind == self.kind.complement()
            && self.label == other.label
            && self.order == other.order
    }
}

impl MolGraph {
    /// The `bonds` kind, when registered at arity 2.
    pub(crate) fn bonds_kind(&self) -> Option<KindId> {
        self.kind_id("bonds").filter(|&k| self.arity(k) == 2)
    }

    /// The `ports` kind, when registered at arity 2.
    fn ports_kind(&self) -> Option<KindId> {
        self.kind_id(PORTS).filter(|&k| self.arity(k) == 2)
    }

    /// The bond joining `a` and `b`, if any (a port does not count).
    pub(crate) fn bond_between(&self, a: NodeId, b: NodeId) -> Option<RelationId> {
        let bonds = self.bonds_kind()?;
        self.neighbor_relations(a)
            .find(|&(kind, _, other)| kind == bonds && other == b)
            .map(|(_, rid, _)| rid)
    }

    /// Whether `a` and `b` are joined by a `bonds` relation.
    pub fn is_bonded(&self, a: NodeId, b: NodeId) -> bool {
        self.bond_between(a, b).is_some()
    }

    /// Add a `bonds` relation between `a` and `b` classed `bond_type` /
    /// `bond_number`, registering the kind when missing.
    pub(crate) fn add_classed_bond(
        &mut self,
        a: NodeId,
        b: NodeId,
        bond_type: BondType,
        bond_number: BondNumber,
    ) -> Result<RelationId, MolRsError> {
        let bonds = self.try_register_kind("bonds", 2)?;
        let bid = self.add_relation(bonds, &[a, b])?;
        write_bond_class(self, bonds, bid, bond_type, bond_number)?;
        Ok(bid)
    }

    /// Record a descriptor on the `(anchor, handle)` valence, registering the
    /// `ports` kind on first use.
    ///
    /// Endpoint order is load-bearing: `anchor` is the node that stays in the
    /// product molecule, `handle` the node bonded to it that roots the leaving
    /// group a paired descriptor's bond replaces. `label` is free-form; `""`
    /// means unnamed. `order` is the multiplicity the formed bond will carry.
    /// The three descriptor facts are written as the relation props
    /// `port_kind` (the glyph), `port_label` and `port_order`.
    ///
    /// The anchor–handle bond is checked here and only here: removing it
    /// later leaves a stale port that [`leaving_group`](Self::leaving_group)
    /// and [`link`](Self::link) refuse. Removing a node removes every port on
    /// it, so stripping hydrogens drops every port whose handle is one.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::NotFound`] when either node is stale or unknown,
    /// and [`MolRsError::Validation`] when `handle` is not bonded to
    /// `anchor`, when `order` is [`BondNumber::Unknown`], when the valence
    /// already carries a port, or when the graph spells `ports` at an arity
    /// other than 2.
    pub fn add_port(
        &mut self,
        anchor: NodeId,
        handle: NodeId,
        kind: PortKind,
        label: &str,
        order: BondNumber,
    ) -> Result<RelationId, MolRsError> {
        self.get_node(anchor)?;
        self.get_node(handle)?;
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
        if self.port_on(anchor, handle).is_some() {
            return Err(MolRsError::validation(format!(
                "the valence ({anchor:?}, {handle:?}) already carries a port"
            )));
        }
        let ports = self.try_register_kind(PORTS, 2)?;
        let pid = self.add_relation(ports, &[anchor, handle])?;
        self.set_relation_prop(ports, pid, "port_kind", kind.as_str())?;
        self.set_relation_prop(ports, pid, "port_label", label)?;
        self.set_relation_prop(ports, pid, "port_order", PropValue::from(order))?;
        Ok(pid)
    }

    /// Read a port back, validating its descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::NotFound`] when `id` names no live port, and
    /// [`MolRsError::Validation`] when the relation is missing `port_kind`,
    /// `port_label` or `port_order`, when the stored glyph is outside the
    /// grammar, or when the stored order is not a definite [`BondNumber`].
    pub fn port(&self, id: RelationId) -> Result<Port, MolRsError> {
        let ports = self
            .ports_kind()
            .ok_or_else(|| MolRsError::not_found("port", format!("port {}", id.data().as_ffi())))?;
        let relation = self.get_relation(ports, id)?;
        let [anchor, handle] = match relation.nodes.as_slice() {
            [anchor, handle] => [*anchor, *handle],
            other => {
                return Err(MolRsError::validation(format!(
                    "port {} spans {} nodes; a port is (anchor, handle)",
                    id.data().as_ffi(),
                    other.len()
                )));
            }
        };
        let missing = |key: &str| {
            MolRsError::validation(format!(
                "port {} carries no '{key}' property",
                id.data().as_ffi()
            ))
        };
        let kind = match relation.props.get("port_kind") {
            Some(PropValue::Str(glyph)) => PortKind::from_str(glyph)?,
            _ => return Err(missing("port_kind")),
        };
        let label = match relation.props.get("port_label") {
            Some(PropValue::Str(label)) => label.clone(),
            _ => return Err(missing("port_label")),
        };
        let order = match relation.props.get("port_order") {
            None => return Err(missing("port_order")),
            stored => match BondNumber::from_prop(stored) {
                BondNumber::Unknown => {
                    return Err(MolRsError::validation(format!(
                        "port {} carries 'port_order' {stored:?}, not a definite bond number",
                        id.data().as_ffi()
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

    /// The leaving group of `port`: its handle's connected component over
    /// bonds once the anchor–handle bond is cut. It contains the handle, and
    /// contains the anchor only when the handle sits on a ring with it.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when the anchor–handle bond no longer
    /// exists (the port is stale); that is the only error.
    pub fn leaving_group(&self, port: &Port) -> Result<BTreeSet<NodeId>, MolRsError> {
        let cut = self.bond_between(port.handle, port.anchor).ok_or_else(|| {
            MolRsError::validation(format!(
                "port handle {:?} is no longer bonded to its anchor {:?}",
                port.handle, port.anchor
            ))
        })?;
        let bonds = self.bonds_kind().expect("a bond was just found");
        let mut seen = BTreeSet::from([port.handle]);
        let mut stack = vec![port.handle];
        while let Some(node) = stack.pop() {
            for (kind, rid, other) in self.neighbor_relations(node) {
                if kind == bonds && rid != cut && seen.insert(other) {
                    stack.push(other);
                }
            }
        }
        Ok(seen)
    }

    /// Iterate over the graph's port handles; empty without a `ports` kind.
    pub fn ports(&self) -> impl Iterator<Item = RelationId> + '_ {
        self.ports_kind()
            .into_iter()
            .flat_map(move |kind| self.relation_ids(kind))
    }

    /// Number of ports.
    pub fn n_ports(&self) -> usize {
        self.ports_kind().map_or(0, |kind| self.n_relations(kind))
    }

    /// The port on the `(anchor, handle)` valence, if any.
    ///
    /// [`add_port`](Self::add_port) admits one port per valence, so the answer
    /// is unique. After a [`merge`](Self::merge), a port of the merged graph is
    /// found again here by its mapped anchor and handle.
    pub fn port_on(&self, anchor: NodeId, handle: NodeId) -> Option<RelationId> {
        let ports = self.ports_kind()?;
        self.neighbor_relations(anchor)
            .find(|&(kind, rid, other)| {
                kind == ports
                    && other == handle
                    && self
                        .relation_nodes(ports, rid)
                        .is_ok_and(|nodes| nodes[0] == anchor)
            })
            .map(|(_, rid, _)| rid)
    }

    // ---- per-node unit membership ----

    /// Record the unit instance `node` came from, under the node prop
    /// `frag_id`.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when `id` exceeds [`i32::MAX`] — node
    /// columns are `i32` — and [`MolRsError::NotFound`] when `node` is stale or
    /// unknown.
    pub fn set_frag_id(&mut self, node: NodeId, id: u32) -> Result<(), MolRsError> {
        let stored = i32::try_from(id).map_err(|_| {
            MolRsError::validation(format!(
                "frag_id {id} exceeds {}, the widest identifier a node column stores",
                i32::MAX
            ))
        })?;
        self.set_node(node, FRAG_ID, PropValue::Int(stored))
    }

    /// The unit instance `node` came from, or `None` when it carries no
    /// `frag_id` (or the handle is stale).
    pub fn frag_id(&self, node: NodeId) -> Option<u32> {
        let stored = self.get_node(node).ok()?.get_int(FRAG_ID)?;
        u32::try_from(stored).ok()
    }

    /// Propagate each `frag_id` to the unlabelled degree-1 nodes hanging off a
    /// labelled one, returning how many nodes were labelled.
    ///
    /// One pass over a snapshot of the labels this call started with, never a
    /// fixpoint. Degree is counted over bonds alone. Intended for nodes a
    /// pipeline newly attached (hydrogens added by a conformer, for
    /// instance); a degree-1 node deliberately left unassigned **will** be
    /// relabelled.
    pub fn inherit_frag_ids(&mut self) -> usize {
        let Some(bonds) = self.bonds_kind() else {
            return 0;
        };
        let labels: HashMap<NodeId, u32> = self
            .node_ids()
            .filter_map(|node| self.frag_id(node).map(|id| (node, id)))
            .collect();
        let mut pending: Vec<(NodeId, u32)> = Vec::new();
        for node in self.node_ids() {
            if labels.contains_key(&node) {
                continue;
            }
            let mut bonded = self
                .neighbor_relations(node)
                .filter(|&(kind, _, _)| kind == bonds)
                .map(|(_, _, other)| other);
            let (Some(only), None) = (bonded.next(), bonded.next()) else {
                continue;
            };
            if let Some(&inherited) = labels.get(&only) {
                pending.push((node, inherited));
            }
        }
        let mut labelled = 0;
        for (node, id) in pending {
            if self.set_frag_id(node, id).is_ok() {
                labelled += 1;
            }
        }
        labelled
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::{Port, PortKind, RelationId};
    use crate::error::MolRsError;
    use crate::system::atomistic::Atomistic;
    use crate::system::bond::BondNumber;
    use crate::system::coarsegrain::CoarseGrain;
    use crate::system::molgraph::NodeId;
    use crate::system::molgraph::PropValue;

    /// `C0–C1` plus one real capping hydrogen bonded to `C0`: the smallest
    /// graph that can carry a legal port. Returns `(mol, c0, c1, h)`.
    fn ch_template() -> (Atomistic, NodeId, NodeId, NodeId) {
        let mut frag = Atomistic::new();
        let c0 = frag.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let c1 = frag.add_atom_xyz("C", 1.54, 0.0, 0.0);
        let h = frag.add_atom_bare("H");
        frag.add_bond(c0, c1).unwrap();
        frag.add_bond(c0, h).unwrap();
        (frag, c0, c1, h)
    }

    /// `ch_template` with one `Symmetric` port `(C0, H)` labelled `"A"`.
    fn ported_template() -> (Atomistic, NodeId, NodeId, NodeId, RelationId) {
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

    // ---- bonds -------------------------------------------------------------

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
    fn add_port_accepts_a_heavy_leaving_group_handle() {
        // Rework (spec assembly-06 §1, decision 9 / golden C2): a leaving
        // group may root at a heavy atom — here the O of an OH on C0.
        let (mut frag, c0, _c1, _h) = ch_template();
        let o = frag.add_atom_bare("O");
        let oh = frag.add_atom_bare("H");
        frag.add_bond(c0, o).unwrap();
        frag.add_bond(o, oh).unwrap();
        let pid = frag
            .add_port(c0, o, PortKind::Left, "A", BondNumber::Single)
            .expect("a bonded heavy handle is a legal port");
        assert_eq!(frag.n_ports(), 1);
        let port = frag.port(pid).expect("the port reads back");
        assert_eq!(port.anchor, c0);
        assert_eq!(port.handle, o);
    }

    /// Amended 2026-09-26 (spec assembly-06 §1): a world port is
    /// resolved by its (anchor, handle) pair (the port map
    /// `port_on` resolves), so a pair holds one port.
    #[test]
    fn add_port_refuses_a_second_port_on_the_same_valence() {
        let (mut frag, c0, _c1, h, _pid) = ported_template();
        let err = frag
            .add_port(c0, h, PortKind::Left, "B", BondNumber::Single)
            .expect_err("(C0, H) already carries a port");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_eq!(frag.n_ports(), 1, "a rejected port leaves no relation");
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

        let ids: Vec<RelationId> = frag.ports().collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&p1) && ids.contains(&p2));
        assert_eq!(frag.n_bonds(), 3, "a port is not a bond");
    }

    // ---- merge ---------------------------------------------------------------

    // ---- where ports live -------------------------------------------------

    #[test]
    fn add_port_registers_the_ports_kind_on_first_use() {
        let (mut mol, c0, _c1, h) = ch_template();
        assert!(mol.kind_id("ports").is_none(), "no port, no kind");
        assert_eq!(mol.n_ports(), 0);
        mol.add_port(c0, h, PortKind::Symmetric, "", BondNumber::Single)
            .expect("a bonded handle is a legal port");
        assert!(mol.kind_id("ports").is_some());
        assert_eq!(mol.n_ports(), 1);
    }

    #[test]
    fn a_coarse_grain_carries_ports_like_any_graph() {
        let mut cg = CoarseGrain::new();
        let a = cg.add_bead("A", 0.0, 0.0, 0.0);
        let b = cg.add_bead("B", 1.0, 0.0, 0.0);
        cg.add_bond(a, b).expect("bond");
        let pid = cg
            .add_port(a, b, PortKind::Left, "x", BondNumber::Single)
            .expect("a bonded bead is a legal handle");
        assert_eq!(cg.port(pid).expect("reads back").label, "x");
    }

    // ---- merge ---------------------------------------------------------------

    /// A merged graph's ports come across and are found again by their mapped
    /// `(anchor, handle)` valence.
    #[test]
    fn a_merged_port_is_found_by_its_mapped_valence() {
        let (mut world, _w0, _w1, _wh, own) = ported_template();
        let (mut other, c0, _c1, h) = ch_template();
        other
            .add_port(c0, h, PortKind::Left, "B", BondNumber::Single)
            .expect("a bonded H handle on its anchor is a legal port");

        let atom_map = world.merge(other).expect("a graph merges");

        assert_eq!(world.n_ports(), 2);
        let mapped = world
            .port_on(atom_map[&c0], atom_map[&h])
            .expect("the merged port sits on the mapped valence");
        assert_ne!(mapped, own);
        let port = world.port(mapped).expect("the mapped port reads back");
        assert_eq!(port.kind, PortKind::Left);
        assert_eq!(port.label, "B");
        assert_eq!(world.port(own).expect("own port").label, "A");
    }

    // ---- per-node unit membership -------------------------------------------

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

        assert!(
            frame.contains_key("atoms"),
            "an atomistic graph's nodes are atoms"
        );
        assert!(frame.contains_key("bonds"), "no block relabeling");
        assert!(frame.contains_key("ports"));

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
            Atomistic::from_frame(&frag.to_frame().expect("a schema-conforming graph converts"))
                .expect("a ported frame reads back");
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
            Atomistic::from_frame(&frag.to_frame().expect("a schema-conforming graph converts"))
                .expect("a ported frame reads back");
        let ids: Vec<Option<u32>> = restored.node_ids().map(|n| restored.frag_id(n)).collect();
        assert_eq!(ids, vec![Some(2), None, None]);
    }

    // ---- the one compatibility rule ------------------------------------------

    #[test]
    fn port_kind_complement_maps_left_right_and_fixes_the_others() {
        assert_eq!(PortKind::Left.complement(), PortKind::Right);
        assert_eq!(PortKind::Right.complement(), PortKind::Left);
        assert_eq!(PortKind::Symmetric.complement(), PortKind::Symmetric);
        assert_eq!(PortKind::Shared.complement(), PortKind::Shared);
    }

    #[test]
    fn port_kind_complement_is_an_involution() {
        for kind in [
            PortKind::Symmetric,
            PortKind::Left,
            PortKind::Right,
            PortKind::Shared,
        ] {
            assert_eq!(kind.complement().complement(), kind, "{kind:?}");
        }
    }

    /// A materialized [`Port`] on `ch_template`'s `(C0, H)` valence with the
    /// given descriptor; `accepts` reads only kind, label and order.
    fn port_with(kind: PortKind, label: &str, order: BondNumber) -> Port {
        let (_frag, c0, _c1, h) = ch_template();
        Port {
            anchor: c0,
            handle: h,
            kind,
            label: label.to_owned(),
            order,
        }
    }

    #[test]
    fn port_accepts_left_right_with_equal_label_and_order() {
        let left = port_with(PortKind::Left, "a", BondNumber::Single);
        let right = port_with(PortKind::Right, "a", BondNumber::Single);
        assert!(left.accepts(&right));
        assert!(right.accepts(&left));
    }

    #[test]
    fn port_accepts_symmetric_with_symmetric() {
        let a = port_with(PortKind::Symmetric, "", BondNumber::Single);
        let b = port_with(PortKind::Symmetric, "", BondNumber::Single);
        assert!(a.accepts(&b));
    }

    #[test]
    fn port_accepts_refuses_a_label_mismatch() {
        let left = port_with(PortKind::Left, "a", BondNumber::Single);
        let right = port_with(PortKind::Right, "b", BondNumber::Single);
        assert!(!left.accepts(&right));
    }

    #[test]
    fn port_accepts_refuses_an_order_mismatch() {
        let left = port_with(PortKind::Left, "a", BondNumber::Single);
        let right = port_with(PortKind::Right, "a", BondNumber::Double);
        assert!(!left.accepts(&right));
    }

    #[test]
    fn port_accepts_refuses_left_with_left() {
        let a = port_with(PortKind::Left, "a", BondNumber::Single);
        let b = port_with(PortKind::Left, "a", BondNumber::Single);
        assert!(!a.accepts(&b));
    }

    // ---- center ----
}
