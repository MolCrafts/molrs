//! `CGsmiles` coarse-graph intermediate representation, exposed to Python as
//! classes under `molrs.io`.
//!
//! *Coarse-graining* is standing one particle — a **bead** — in for a whole
//! group of atoms, so a polymer can be written and simulated without naming
//! every atom in it. `CGsmiles` is a *line notation* (a molecule written as
//! one string, the way SMILES writes an atomistic one) that states a molecule
//! at one or more such **resolutions**: the coarsest level names beads, each
//! later block says what the names one level up stand for, and the last block
//! is ordinary atomistic SMILES. Reference: Grünewald et al., *J. Chem. Inf.
//! Model.* (2025), DOI 10.1021/acs.jcim.5c00064.
//!
//! [`PyCGSmilesIR`] is the single front door: `molrs.io.CGSmilesIR(text)`
//! parses, and every other class here is a read-only view over one record of
//! the value it returns — a resolution level, a node, an edge, a fragment
//! definition, a resolved descriptor pair, one end of such a pair, or a
//! bonding descriptor. They are produced by `CGSmilesIR` and nowhere else, so
//! none of them has a constructor on the Python side.
//!
//! There is deliberately **no** `CGSmilesReader`: in this binding "Reader"
//! means a lazy, path-backed trajectory cursor (`XYZTrajReader`,
//! `DCDTrajReader`, …), and a text-in / IR-out parser is not that object. The
//! reader-shaped API belongs to molpy, wrapping this class the way its
//! `SmilesReader` wraps `molrs.io.SmilesIR`.
//!
//! # Values at the boundary
//!
//! An enum that *is* a count crosses as the count: [`PyCGEdge::multiplicity`]
//! is `CGBondOrder` read through
//! [`CGBondOrder::multiplicity`](molrs::io::smiles::CGBondOrder::multiplicity),
//! a dimensionless `1..=4`. Every other enum crosses as a *name*, never as
//! the small integer `core` stores such a value as (`BondType::code`: 0
//! unknown, 1 single, 2 double, 3 triple, 4 aromatic). The storage codes are
//! a column encoding, not a boundary encoding, and they are not injective
//! over the enums crossing here: `BondKind::{Up, Down, Any, Ring}` all store
//! as single and `Quadruple` as double, so a caller could not read the
//! written notation back out of a number.
//!
//! Which name depends on whether the notation itself already spells the
//! variant. A descriptor kind does: `$`, `<`, `>`, `!` is what a user types
//! and what a stored port's `port_kind` prop holds
//! ([`PortKind::as_str`](molrs::core::system::PortKind::as_str)), so
//! [`PyBondingDescriptor::kind`] crosses as that same glyph — one spelling
//! for the notation, the column and the boundary, with no third vocabulary to
//! translate between them. The enums the notation does *not* spell out cross
//! as the lowercase spelling of their Rust variant instead:
//! [`PyBondingDescriptor::order`] and [`PyResolvedPair::kind`] are bond-kind
//! names (`"single"`, `"aromatic"`, …) and [`PyPairEnd::end`] is `"sub"` or
//! `"body"`. Lowercase names are what the parent module `io` already uses at
//! this boundary — `build_smiles_emit_options` reads exactly such spellings
//! back into Rust enums.
//!
//! [`bond_kind_name`] lists every variant explicitly, and
//! [`descriptor_kind_name`] delegates to
//! [`DescriptorKind::as_str`](molrs::io::smiles::DescriptorKind::as_str),
//! which does: no input can panic across the seam, and a new variant upstream
//! is a compile error rather than a runtime one.

use molrs::io::smiles::{
    BondKind, BondingDescriptor, CGEdge, CGFragmentDef, CGGraph, CGNode, CGSmilesIR,
    DescriptorKind, EdgeOrigin, FragmentBody, PairEnd, ResolvedPair, parse_cgsmiles,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::system::molgraph::{PyAtomistic, PyCoarseGrain};
use crate::helpers::smiles_error_to_pyerr;
use crate::io::PySmilesIR;

/// The lowercase Python spelling of a [`BondKind`].
///
/// Total by construction — every variant is listed, so this never panics and
/// never invents a spelling. The whole enum crosses by name because the
/// numeric alternative loses information: `Up`, `Down`, `Any` and `Ring` are
/// all stored as `BondType::Single` and `Quadruple` as `BondType::Double`, so
/// no single code round-trips what the notation wrote.
fn bond_kind_name(kind: BondKind) -> &'static str {
    match kind {
        BondKind::Single => "single",
        BondKind::Double => "double",
        BondKind::Triple => "triple",
        BondKind::Quadruple => "quadruple",
        BondKind::Aromatic => "aromatic",
        BondKind::Up => "up",
        BondKind::Down => "down",
        BondKind::Any => "any",
        BondKind::Ring => "ring",
    }
}

/// The Python spelling of a [`DescriptorKind`] — the grammar glyph.
///
/// The glyph is the only spelling a user ever writes (`[$]COC[$]`) and the
/// one a stored port carries in its `port_kind` prop
/// ([`PortKind::as_str`](molrs::core::system::PortKind::as_str)), so the
/// boundary adds no third vocabulary: a kind read off a descriptor here can
/// be handed straight to a graph's ``add_port`` or compared against a port
/// column without a lookup table on the Python side.
///
/// The glyph table itself lives on the enum, as
/// [`DescriptorKind::as_str`](molrs::io::smiles::DescriptorKind::as_str), so
/// this boundary reads the notation's own spelling instead of keeping a second
/// copy of it that could drift from `PortKind::as_str`. It is total there, for
/// the same reason [`bond_kind_name`] is here. `"!"` (the squash operator)
/// cannot reach Python today — the reader refuses it — and is spelled anyway,
/// so the mapping stays a function of the enum rather than of what the reader
/// currently admits.
fn descriptor_kind_name(kind: DescriptorKind) -> &'static str {
    kind.as_str()
}

/// One bonding descriptor: a site at which a fragment may later be joined.
///
/// A fragment written ``[$]COC[$]`` declares two such sites. Joining two of
/// them is *pairing*, and it is what turns a coarse edge into a real bond;
/// which descriptors may pair is decided by all three attributes below at
/// once.
///
/// Attributes
/// ----------
/// kind : {"$", "<", ">", "!"}
///     Which operator was written, as the glyph itself — the same spelling a
///     stored port's ``port_kind`` uses, so it needs no translation to reach
///     :meth:`Atomistic.def_port`. A ``"$"`` pairs only with a ``"$"``, a
///     ``"<"`` only with a ``">"`` (and the other way round). ``"!"`` is the
///     squash operator and never reaches Python: the reader refuses ``[!]``
///     outright.
/// label : str
///     Label distinguishing descriptor classes of the same kind, ``""`` when
///     the descriptor is unnamed (``[$]`` against ``[$a]``). Labels must
///     match character for character to pair, so ``[$a]`` pairs neither
///     ``[$b]`` nor ``[$]``.
/// order : str or None
///     Bond order written next to the bracket (``CC=[$]``), as a lowercase
///     bond-kind name; ``None`` when none was written. For pairing ``None``
///     counts as ``"single"``, so ``[$]`` pairs ``-[$]`` and never ``=[$]``.
///     When neither side of a pair wrote an order the resolved bond is
///     ``"single"`` — except between two atoms both *written* aromatic, where
///     it is ``"aromatic"``; see :attr:`ResolvedPair.kind`.
#[pyclass(
    module = "molrs.io",
    name = "BondingDescriptor",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyBondingDescriptor {
    inner: BondingDescriptor,
}

#[pymethods]
impl PyBondingDescriptor {
    /// Which operator was written, as its glyph (``"$"``, ``"<"``, ``">"``,
    /// ``"!"``) — the spelling a port's `port_kind` also uses.
    #[getter]
    fn kind(&self) -> &'static str {
        descriptor_kind_name(self.inner.kind)
    }

    /// The descriptor's label, ``""`` when it is unnamed.
    #[getter]
    fn label(&self) -> String {
        self.inner.label.clone()
    }

    /// The bond order written beside the bracket, or ``None`` — which counts
    /// as single when this descriptor is matched against another.
    #[getter]
    fn order(&self) -> Option<&'static str> {
        self.inner.order.map(bond_kind_name)
    }

    /// ``BondingDescriptor(kind=…, label=…, order=…)`` — the three attributes
    /// that decide pairing.
    fn __repr__(&self) -> String {
        format!(
            "BondingDescriptor(kind={:?}, label={:?}, order={:?})",
            self.kind(),
            self.inner.label,
            self.order()
        )
    }
}

/// One coarse-grained node: ``[#PEO]``, ``[#A;q=-0.5]``.
///
/// A node is one bead — one particle standing in for a group of atoms — and
/// the text after ``;`` is its *annotations*, the per-bead properties the
/// notation lets a writer state.
///
/// Two occurrences of ``[#A]`` are two distinct nodes that happen to share a
/// name; the name is a *fragment* name to be resolved later, not an identity.
///
/// Attributes
/// ----------
/// name : str
///     The fragment name written after ``#``, without the sigil.
/// charge : float or None
///     Partial charge in elementary-charge units ``e`` (the ``q``
///     annotation), ``None`` when the notation wrote none. It is a partial,
///     never a formal, charge.
/// annotations : list of (str, str)
///     Annotations the dialect does not reserve, as written, in written
///     order. The reserved keys (``q``, ``w``, ``x``) never appear here.
/// descriptors : list of BondingDescriptor
///     Bonding descriptors written beside this node, in written order — the
///     marks saying where this bead may later be joined to another; see
///     :class:`BondingDescriptor`.
/// parent : int or None
///     Index into the *previous* level's ``nodes`` of the node this one was
///     instantiated from; ``None`` in ``levels[0]`` and in every fragment
///     body, which is a template rather than an instance.
#[pyclass(module = "molrs.io", name = "CGNode", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyCGNode {
    inner: CGNode,
}

#[pymethods]
impl PyCGNode {
    /// The fragment name written after ``#``.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Partial charge in ``e``, or ``None``.
    #[getter]
    fn charge(&self) -> Option<f64> {
        self.inner.charge
    }

    /// Unreserved annotations, in written order.
    #[getter]
    fn annotations(&self) -> Vec<(String, String)> {
        self.inner.annotations.clone()
    }

    /// Bonding descriptors written beside this node, in written order.
    #[getter]
    fn descriptors(&self) -> Vec<PyBondingDescriptor> {
        self.inner
            .descriptors
            .iter()
            .map(|descriptor| PyBondingDescriptor {
                inner: descriptor.clone(),
            })
            .collect()
    }

    /// Index of the node one level up that instantiated this one, or ``None``.
    #[getter]
    fn parent(&self) -> Option<usize> {
        self.inner.parent
    }

    /// ``CGNode(name=…, charge=…, descriptors=…, parent=…)``, where
    /// ``descriptors`` is the *number* of descriptors rather than the list.
    fn __repr__(&self) -> String {
        format!(
            "CGNode(name={:?}, charge={:?}, descriptors={}, parent={:?})",
            self.inner.name,
            self.inner.charge,
            self.inner.descriptors.len(),
            self.inner.parent
        )
    }
}

/// One coarse edge, joining ``nodes[i]`` and ``nodes[j]`` of its level.
///
/// Attributes
/// ----------
/// i : int
///     Index of the first endpoint in the level's ``nodes``.
/// j : int
///     Index of the second endpoint in the level's ``nodes``.
/// multiplicity : int
///     How many bonds this edge stands for, ``1`` to ``4``, from the bond
///     symbol that formed it (``-``, ``=``, ``#``, ``$``). A dimensionless
///     count, never a bond kind — the chemistry of a resolved bond is
///     :attr:`ResolvedPair.kind`.
/// derived_from : tuple of (int, int), or None
///     ``(level, pair)`` of the resolved pair one level up that induced this
///     edge — ``ir.pairs[level][pair]`` — or ``None`` when the notation wrote
///     the edge itself. Every derived edge has multiplicity ``1``: one
///     resolved pair is one bond, so a coarse edge of multiplicity *n*
///     induces *n* separate derived edges rather than one multiple edge.
#[pyclass(module = "molrs.io", name = "CGEdge", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyCGEdge {
    inner: CGEdge,
}

#[pymethods]
impl PyCGEdge {
    /// Index of the first endpoint.
    #[getter]
    fn i(&self) -> usize {
        self.inner.i
    }

    /// Index of the second endpoint.
    #[getter]
    fn j(&self) -> usize {
        self.inner.j
    }

    /// How many bonds the edge stands for, ``1..=4``.
    #[getter]
    fn multiplicity(&self) -> u8 {
        self.inner.order.multiplicity()
    }

    /// The ``(level, pair)`` that induced this edge, or ``None`` when the
    /// notation wrote it.
    #[getter]
    fn derived_from(&self) -> Option<(usize, usize)> {
        match self.inner.origin {
            EdgeOrigin::Written => None,
            EdgeOrigin::Derived { level, pair } => Some((level, pair)),
        }
    }

    /// ``CGEdge(i=…, j=…, multiplicity=…, derived_from=…)``.
    fn __repr__(&self) -> String {
        format!(
            "CGEdge(i={}, j={}, multiplicity={}, derived_from={:?})",
            self.inner.i,
            self.inner.j,
            self.multiplicity(),
            self.derived_from()
        )
    }
}

/// One resolution level: coarse-grained nodes and the edges between them.
///
/// Both lists are in parse order — nodes in the order their brackets were
/// read, edges in the order the notation formed them, with every derived edge
/// appended after the written ones.
///
/// Attributes
/// ----------
/// nodes : list of CGNode
///     The beads of this level. A node is addressed by its index here, and
///     that index is what an edge's ``i`` / ``j`` name.
/// edges : list of CGEdge
///     The bonds between them, each naming two indices into ``nodes``.
#[pyclass(module = "molrs.io", name = "CGGraph", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyCGGraph {
    inner: CGGraph,
}

#[pymethods]
impl PyCGGraph {
    /// Nodes in parse order; a node is addressed by its index here.
    #[getter]
    fn nodes(&self) -> Vec<PyCGNode> {
        self.inner
            .nodes
            .iter()
            .map(|node| PyCGNode {
                inner: node.clone(),
            })
            .collect()
    }

    /// Edges in parse order, each naming two indices into ``nodes``.
    #[getter]
    fn edges(&self) -> Vec<PyCGEdge> {
        self.inner
            .edges
            .iter()
            .map(|edge| PyCGEdge {
                inner: edge.clone(),
            })
            .collect()
    }

    /// ``CGGraph(nodes=…, edges=…)`` — the two list *lengths*, not the lists.
    fn __repr__(&self) -> String {
        format!(
            "CGGraph(nodes={}, edges={})",
            self.inner.nodes.len(),
            self.inner.edges.len()
        )
    }
}

/// One entry of a fragment block: ``#PEO=[$]COC[$]``.
///
/// Attributes
/// ----------
/// name : str
///     The fragment name written after ``#``, without the sigil.
/// body : CGGraph or SmilesIR
///     What the name stands for. An intermediate block's bodies are coarse
///     graphs over the next level's nodes; the last block's are atomistic
///     SMILES fragment bodies, with their bonding descriptors intact. There
///     is no ``body_kind``: the Python type *is* the tag, so callers dispatch
///     with ``isinstance``.
///
///     A ``SmilesIR`` body still carrying descriptors is not convertible on
///     its own — :meth:`SmilesIR.to_atomistic` refuses a descriptor-bearing
///     IR with ``ValueError`` rather than drop the descriptors silently. Use
///     :meth:`CGSmilesIR.to_atomistic`, which expands every body *and* bonds
///     the ports the reader paired.
///
///     Such a body's ``repr`` echoes the fragment-table entry as written —
///     ``#PEO=[$]COC[$]``, name and ``=`` included — not a bare SMILES,
///     because the entry's span is the text the IR records.
#[pyclass(
    module = "molrs.io",
    name = "CGFragmentDef",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCGFragmentDef {
    inner: CGFragmentDef,
    /// The entry's own source text, handed to a `SmilesIR` body so that its
    /// `__repr__` echoes what was written. Never re-parsed.
    input: String,
}

#[pymethods]
impl PyCGFragmentDef {
    /// The fragment name written after ``#``.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The body the name stands for: a :class:`CGGraph` or a
    /// :class:`SmilesIR`.
    ///
    /// Raises
    /// ------
    /// Exception
    ///     Only what building the returned Python object can raise (a
    ///     ``MemoryError``, say). The body is already parsed and is copied,
    ///     never re-read, so no notation error can originate here.
    #[getter]
    fn body(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.body {
            FragmentBody::Graph(graph) => Ok(PyCGGraph {
                inner: graph.clone(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind()),
            FragmentBody::Smiles(ir) => Ok(PySmilesIR::from_core(ir.clone(), self.input.clone())
                .into_pyobject(py)?
                .into_any()
                .unbind()),
        }
    }

    /// ``CGFragmentDef(name=…, body=CGGraph)``, or ``body=SmilesIR`` for an
    /// atomistic body: the body's *type*, which is what a caller dispatches
    /// on.
    fn __repr__(&self) -> String {
        let body = match &self.inner.body {
            FragmentBody::Graph(_) => "CGGraph",
            FragmentBody::Smiles(_) => "SmilesIR",
        };
        format!("CGFragmentDef(name={:?}, body={})", self.inner.name, body)
    }
}

/// One end of a [`PyResolvedPair`]: the port that was consumed, and the
/// entity that offered it.
///
/// A *port* is one occurrence of a bonding descriptor, addressed by its index
/// in the list the entity wrote its descriptors in.
///
/// Throughout, ``k`` is the level this end was read at — the ``k`` of the
/// ``ir.pairs[k]`` list the owning :class:`ResolvedPair` came from.
///
/// Attributes
/// ----------
/// end : {"sub", "body"}
///     Which kind of entity offered the port. ``"sub"`` is an intermediate
///     level, where the ports of a bead are the descriptors its child nodes
///     carry one level down; ``"body"`` is the last level, where the ports of
///     a node are the descriptors written on its atomistic body.
/// index : int
///     For ``"sub"``: index into ``levels[k + 1].nodes`` of the child
///     carrying the port, whose own :attr:`CGNode.parent` is the
///     ``levels[k]`` instance the port belongs to. For ``"body"``: index into
///     ``levels[k].nodes`` of the instance whose body holds the port.
/// port : int
///     For ``"sub"``: index into that child's :attr:`CGNode.descriptors`. For
///     ``"body"``: index into the descriptor map of that body, ordered by the
///     atom the descriptor sits on in the order the body's atoms were read,
///     and within one atom in written order. There is no Python object for
///     that map; the index is what :meth:`CGSmilesIR.to_atomistic` resolves
///     against the converted body.
#[pyclass(module = "molrs.io", name = "PairEnd", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPairEnd {
    inner: PairEnd,
}

#[pymethods]
impl PyPairEnd {
    /// Which kind of entity offered the port, lowercased.
    #[getter]
    fn end(&self) -> &'static str {
        match self.inner {
            PairEnd::Sub { .. } => "sub",
            PairEnd::Body { .. } => "body",
        }
    }

    /// Index of the child node (``"sub"``) or the instance (``"body"``).
    #[getter]
    fn index(&self) -> usize {
        match self.inner {
            PairEnd::Sub { node, .. } => node,
            PairEnd::Body { instance, .. } => instance,
        }
    }

    /// Index of the consumed port within that entity's descriptor list.
    #[getter]
    fn port(&self) -> usize {
        match self.inner {
            PairEnd::Sub { port, .. } => port,
            PairEnd::Body { port, .. } => port,
        }
    }

    /// ``PairEnd(end=…, index=…, port=…)``.
    fn __repr__(&self) -> String {
        format!(
            "PairEnd(end={:?}, index={}, port={})",
            self.end(),
            self.index(),
            self.port()
        )
    }
}

/// One bond a written coarse edge stands for, with the two ports it consumed.
///
/// Pairing is what turns "these two beads are bonded" into "this port of this
/// bead is bonded to that port of that one" — the fact an expansion needs and
/// a coarse edge does not carry.
///
/// A pair read from ``ir.pairs[k]`` describes a bond of ``ir.levels[k]``;
/// ``k`` below always means that level.
///
/// Attributes
/// ----------
/// edge : int
///     Index into ``levels[k].edges`` of the edge this pair satisfies, in the
///     final edge list — derived edges are appended to a level before it is
///     resolved, so this index is valid against the list a caller reads.
/// bond : int
///     Which of that edge's :attr:`CGEdge.multiplicity` bonds this is,
///     0-based.
/// src : PairEnd
///     The port the edge's first endpoint offered.
/// dst : PairEnd
///     The port the edge's second endpoint offered.
/// kind : str
///     The chemistry of the resolved bond as a lowercase bond-kind name —
///     the order the notation wrote on either descriptor, ``"aromatic"`` when
///     neither wrote one and both port atoms are written aromatic, and
///     ``"single"`` otherwise.
#[pyclass(
    module = "molrs.io",
    name = "ResolvedPair",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyResolvedPair {
    inner: ResolvedPair,
}

#[pymethods]
impl PyResolvedPair {
    /// Index of the edge this pair satisfies.
    #[getter]
    fn edge(&self) -> usize {
        self.inner.edge
    }

    /// Which of the edge's bonds this is, 0-based.
    #[getter]
    fn bond(&self) -> usize {
        self.inner.bond
    }

    /// The port the edge's first endpoint offered.
    #[getter]
    fn src(&self) -> PyPairEnd {
        PyPairEnd {
            inner: self.inner.src.clone(),
        }
    }

    /// The port the edge's second endpoint offered.
    #[getter]
    fn dst(&self) -> PyPairEnd {
        PyPairEnd {
            inner: self.inner.dst.clone(),
        }
    }

    /// The chemistry of the resolved bond, lowercased.
    #[getter]
    fn kind(&self) -> &'static str {
        bond_kind_name(self.inner.kind)
    }

    /// ``ResolvedPair(edge=…, bond=…, kind=…)`` — the two ends are left out,
    /// each having a repr of its own.
    fn __repr__(&self) -> String {
        format!(
            "ResolvedPair(edge={}, bond={}, kind={:?})",
            self.inner.edge,
            self.inner.bond,
            self.kind()
        )
    }
}

/// Intermediate representation of a parsed `CGsmiles` string.
///
/// An *intermediate representation* is what the string says, in structured
/// form — one step short of a molecule. `CGsmiles` writes a molecule at one
/// or more *resolutions*: the coarse level names beads (``{[#PEO][#PEO]}``),
/// and every block after the first is a table of the fragment bodies the
/// level above named. Constructing this class parses, validates, expands and
/// resolves the whole string; the value it returns is finished, and is read
/// rather than built. Turning it into atoms is a separate call,
/// :meth:`to_atomistic`.
///
/// Attributes
/// ----------
/// levels : list of CGGraph
///     Resolution levels, coarsest first. Level 0 is the base block that was
///     written; every later level is the expansion of the table above it.
/// fragments : list of dict of (str, CGFragmentDef)
///     Fragment tables, coarsest first, each keyed by the name written after
///     ``#`` and in name order. ``fragments[k]`` resolves the names of
///     ``levels[k]``.
/// pairs : list of list of ResolvedPair
///     Resolved descriptor pairs, parallel to :attr:`levels`, in resolution
///     order. A level with nothing to pair against carries an empty list.
///
/// Notes
/// -----
/// ``len(levels) == len(fragments)`` whenever any fragment table was written
/// — one level fewer than blocks, because the last block is atomistic and a
/// level is a graph of beads, not of atoms. A base-only string is the other
/// case: ``fragments == []`` and ``len(levels) == 1``.
///
/// Examples
/// --------
/// >>> ir = molrs.io.CGSmilesIR("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}")
/// >>> len(ir.levels[0].nodes)
/// 5
/// >>> ir.to_atomistic().n_atoms
/// 11
#[pyclass(module = "molrs.io", name = "CGSmilesIR")]
pub struct PyCGSmilesIR {
    inner: CGSmilesIR,
    /// The string that was parsed, kept for `__repr__` and for slicing a
    /// fragment entry's own text out by its span. Never re-parsed.
    input: String,
}

#[pymethods]
impl PyCGSmilesIR {
    /// Parse `text` into its `CGsmiles` intermediate representation.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///     `CGsmiles` string — one ``{...}`` block per resolution, blocks
    ///     separated by ``.``, the last one atomistic.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For every refusal the notation makes, with the offending column
    ///     marked in the message. The four families are: a syntax error
    ///     (an unclosed block or bracket, a bond symbol with nothing after
    ///     it, a malformed annotation); a name no fragment table defines; a
    ///     construct this version refuses outright (the squash operator
    ///     ``[!]``, the virtual bond ``.``, a mapping weight ``w`` other than
    ///     its default, the chirality key ``x``); and a written coarse edge
    ///     whose two fragments offer no free pair of compatible bonding
    ///     descriptors, as in ``{[#A][#B]}.{#A=[$a]C,#B=[$b]C}``.
    ///
    /// Examples
    /// --------
    /// >>> molrs.io.CGSmilesIR("{[#A][#B]}.{#A=[$]C,#B=[$]O}").to_atomistic().n_atoms
    /// 2
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        let inner = parse_cgsmiles(text).map_err(smiles_error_to_pyerr)?;
        Ok(Self {
            inner,
            input: text.to_owned(),
        })
    }

    /// Resolution levels, coarsest first.
    ///
    /// Returns
    /// -------
    /// list of CGGraph
    #[getter]
    fn levels(&self) -> Vec<PyCGGraph> {
        self.inner
            .levels
            .iter()
            .map(|level| PyCGGraph {
                inner: level.clone(),
            })
            .collect()
    }

    /// Fragment tables, coarsest first, each in fragment-name order.
    ///
    /// Name order, not written order: the table is a sorted map in Rust and
    /// the dict is filled from it, so iteration is reproducible across runs.
    ///
    /// Returns
    /// -------
    /// list of dict of (str, CGFragmentDef)
    ///
    /// Raises
    /// ------
    /// Exception
    ///     Only what building the dictionaries themselves can raise (a
    ///     ``MemoryError``, say). Every entry is already parsed, so no
    ///     notation error can originate here.
    #[getter]
    fn fragments<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .fragments
            .iter()
            .map(|table| {
                let dict = PyDict::new(py);
                for (name, def) in table {
                    // The entry's own source text, for the body's `__repr__`
                    // only. A byte range that is not a char boundary yields
                    // the fragment name rather than a panic.
                    let input = self
                        .input
                        .get(def.span.start..def.span.end)
                        .unwrap_or(&def.name)
                        .to_owned();
                    dict.set_item(
                        name,
                        PyCGFragmentDef {
                            inner: def.clone(),
                            input,
                        },
                    )?;
                }
                Ok(dict)
            })
            .collect()
    }

    /// Resolved descriptor pairs, parallel to :attr:`levels`.
    ///
    /// Returns
    /// -------
    /// list of list of ResolvedPair
    #[getter]
    fn pairs(&self) -> Vec<Vec<PyResolvedPair>> {
        self.inner
            .pairs
            .iter()
            .map(|level_pairs| {
                level_pairs
                    .iter()
                    .map(|pair| PyResolvedPair {
                        inner: pair.clone(),
                    })
                    .collect()
            })
            .collect()
    }

    /// Expand the lowest resolution into an all-atom molecular graph.
    ///
    /// Every node of the last level is replaced by a copy of its fragment
    /// body, and every pair the reader resolved becomes one bond, carrying
    /// the class the pairing decided (:attr:`ResolvedPair.kind`). A port left
    /// unpaired creates nothing.
    ///
    /// **Topology only.** A line notation states no geometry, so no atom
    /// carries a position; no hydrogen is added and no chemical perception is
    /// run. The atom count is the heavy-atom count the bodies wrote. Those
    /// are separate steps a caller composes.
    ///
    /// Every atom carries the integer property ``frag_id``: the index, in
    /// ``levels[-1].nodes``, of the node it was expanded from. It is the
    /// fragment-instance key — not ``mol_id`` (a whole molecule; a fragment
    /// instance is smaller than one) and not ``res_id`` (a biopolymer
    /// residue) — so a caller can partition the result by instance without
    /// re-deriving the grouping.
    ///
    /// Atom *indices* are not contracted; the connectivity, the elements and
    /// ``frag_id`` are. A ring-bearing string may pair ports in a different
    /// order than the CGsmiles reference implementation does and so lay its
    /// atoms out differently, while the expansion stays isomorphic to the
    /// reference's as an unlabelled graph.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     Molecular graph with atoms and bonds.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If there is no atomistic body to expand: a base-only string, which
    ///     writes beads and never says what they are made of
    ///     (``"{[#A][#B]}"``). That is the only failure a parsed value can
    ///     reach — a string whose last block is not atomistic, or whose
    ///     bodies do not convert, is already refused by the constructor.
    ///
    /// Examples
    /// --------
    /// >>> ir = molrs.io.CGSmilesIR("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}")
    /// >>> ir.to_atomistic().n_atoms
    /// 11
    fn to_atomistic(&self, py: Python<'_>) -> PyResult<Py<PyAtomistic>> {
        let mol = self.inner.to_atomistic().map_err(smiles_error_to_pyerr)?;
        PyAtomistic::from_core(py, mol)
    }

    /// Read the last fragment table as named, ported :class:`~molrs.Atomistic` templates.
    ///
    /// One entry per fragment the table defines, keyed by the name written
    /// after ``#``. Each body keeps its own atoms and bonds and carries one
    /// ``port`` per bonding descriptor the body wrote — the unsatisfied
    /// valences that joining the fragment to a neighbour consumes. This is
    /// the *template* view of the string, the counterpart of
    /// :meth:`to_atomistic`, which instead expands the whole molecule and
    /// consumes those descriptors as bonds.
    ///
    /// **Topology only.** A line notation states no geometry, so no atom
    /// carries a position; coordinates come from a separate
    /// :class:`molrs.conformer.Conformer` step, in ångström (Å).
    ///
    /// Returns
    /// -------
    /// dict of (str, Atomistic)
    ///     One ported template per fragment definition, in name order.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the string defines no fragment table: a base-only string
    ///     (``"{[#A][#B]}"``) writes beads and never says what they are made
    ///     of, so there is no body to read.
    ///
    /// Examples
    /// --------
    /// >>> ir = molrs.io.CGSmilesIR("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}")
    /// >>> sorted(ir.to_fragment())
    /// ['OH', 'PEO']
    /// >>> ir.to_fragment()["PEO"].n_ports
    /// 2
    fn to_fragment<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let templates = self.inner.to_fragment().map_err(smiles_error_to_pyerr)?;
        let out = PyDict::new(py);
        for (name, template) in templates {
            out.set_item(name, PyAtomistic::from_core(py, template)?)?;
        }
        Ok(out)
    }

    /// Read the coarsest level, ``levels[0]``, as a bead graph.
    ///
    /// One bead per node, in node order, whose only property is
    /// ``bead_type`` (the name written after ``#``); one CG bond per edge.
    /// Only the base block is read, so a base-only string such as
    /// ``"{[#1][#1][#1][#4]}"`` converts. This is how a bead-group pattern for
    /// :class:`molrs.perceive.SubgraphMatcher` is written as notation.
    ///
    /// **No geometry.** No bead carries ``x`` / ``y`` / ``z`` (Å), ``mass``
    /// (g/mol) or ``charge``, and there is no bead membership, so
    /// :meth:`CoarseGrain.center` refuses the result. Edge multiplicities and
    /// node annotations are dropped.
    ///
    /// Returns
    /// -------
    /// CoarseGrain
    ///     The level-0 bead graph.
    ///
    /// Raises
    /// ------
    /// SmilesError
    ///     (a ``ValueError``) if the IR breaks a reader invariant: no levels,
    ///     an edge endpoint out of range, or a CG bond that cannot be added.
    ///     No parsed string reaches this; only a hand-edited IR does.
    ///
    /// Examples
    /// --------
    /// >>> cg = molrs.io.CGSmilesIR("{[#1][#1][#1][#4]}").to_coarsegrain()
    /// >>> cg.n_beads
    /// 4
    fn to_coarsegrain(&self, py: Python<'_>) -> PyResult<Py<PyCoarseGrain>> {
        let cg = self.inner.to_coarsegrain().map_err(smiles_error_to_pyerr)?;
        PyCoarseGrain::from_core(py, cg)
    }

    /// ``CGSmilesIR('…', levels=…)``, quoting the string that was parsed.
    fn __repr__(&self) -> String {
        format!(
            "CGSmilesIR('{}', levels={})",
            self.input,
            self.inner.levels.len()
        )
    }
}
