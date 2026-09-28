//! Python bindings for the ECS molecular graph.
//!
//! The core is an ECS *world*: entities are stable opaque handles, their data
//! lives in aligned component columns, and topology is kind-tagged relations.
//! This module exposes that faithfully:
//!
//! - [`PyGraph`] (`molrs.Graph`) — the domain-agnostic world: stable-handle
//!   entities, by-name component get/set, and the kind-tagged relation API.
//! - [`PyAtomistic`] (`molrs.Atomistic`) / [`PyCoarseGrain`]
//!   (`molrs.CoarseGrain`) — peer leaves that **hold a core [`Atomistic`] /
//!   [`CoarseGrain`] from construction** (never converted from a `MolGraph`,
//!   never converted into each other). They add the
//!   domain builders (`add_atom`/`add_bond`/…) and own `to_frame` /
//!   `from_frame` (`self.inner.to_frame()`, zero conversion). They subclass
//!   `Graph` in Python; the generic graph API is shared via the
//!   [`graph_world_impl!`] macro, which always operates on the receiver's *own*
//!   graph (`self.mol()` / `self.mol_mut()`), so the leaf's graph is the single
//!   data slot.
//!
//! Handles are stable opaque `int`s (generational slotmap keys); removing one
//! entity never invalidates another, and a stale handle raises.
//!
//! Every leaf hands its core value back to Python through the single helper
//! [`from_core_shadowed`], which resolves the *public* class the package
//! installs over the native one. The empty base `MolGraph` such a construction
//! carries is structural, not waste: PyO3 builds a subclass base-then-subclass
//! and `PyGraph`'s only field is a `MolGraph`, so an instance of a class
//! declaring `extends = PyGraph` necessarily has one, and the leaf-first
//! accessors above make sure nothing ever reads it.

use std::collections::HashMap;
use std::str::FromStr;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn};
use pyo3::PyClass;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::False;

use molrs::perceive::rings::max_ring_system_size as core_max_ring_system_size;
use molrs::perceive::smarts::{MatchOptions, Reaction, RingPrimitive, SmartsPattern};
use molrs::spatial::geometry::CenterError;
use molrs::store::keys;
use molrs::system::atomistic::{Atomistic, ExtractedAtomistic};
use molrs::system::bond::{BondNumber, BondType};
use molrs::system::coarsegrain::{CoarseGrain, ExtractedCoarseGrain};
use molrs::system::entity_table::Cell;
use molrs::system::link::LinkError;
use molrs::system::molgraph::{
    KindId, MolGraph, NodeId, PropValue, node_from_u64, node_to_u64, relation_from_u64,
    relation_to_u64,
};
use molrs::system::port::PortKind;

use crate::core::store::frame::PyFrame;
use crate::helpers::molrs_error_to_pyerr;
use crate::op::vector_to_py;

// ---------------------------------------------------------------------------
// Value conversion helpers
// ---------------------------------------------------------------------------

/// Convert a Python scalar to a [`PropValue`].
///
/// `bool` is tried before `int` because a Python `bool` is a subclass of `int`
/// (so `extract::<i64>()` would silently collapse `True`→`1`); `int` is tried
/// before `float` so an integer literal doesn't become a float. Anything that is
/// not `bool` / `int` / `float` / `str` is rejected fail-fast — non-representable
/// values (lists, `None`, arbitrary objects) MUST raise, never be stashed. An
/// integer outside the stored 32-bit range raises `OverflowError` rather than
/// wrapping (or, past 64 bits, silently becoming a float).
pub(crate) fn py_to_prop(value: &Bound<'_, PyAny>) -> PyResult<PropValue> {
    // `extract::<bool>()` matches only a genuine Python `bool`, not an `int`.
    if let Ok(b) = value.extract::<bool>() {
        Ok(PropValue::Bool(b))
    } else if value.hasattr(pyo3::intern!(value.py(), "__index__"))? {
        // An integer (Python `int`, numpy integer). Extracting straight to the
        // stored width raises `OverflowError` out of range instead of wrapping.
        value.extract::<i32>().map(PropValue::Int)
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(PropValue::F64(f))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(PropValue::Str(s))
    } else {
        Err(PyTypeError::new_err(
            "component value must be bool, int, float, or str",
        ))
    }
}

fn cell_to_py(py: Python<'_>, cell: Cell<'_>) -> PyResult<Py<PyAny>> {
    Ok(match cell {
        Cell::F64(v) => v.into_pyobject(py)?.into_any().unbind(),
        Cell::I32(v) => v.into_pyobject(py)?.into_any().unbind(),
        Cell::Str(s) => s.into_pyobject(py)?.into_any().unbind(),
        Cell::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
    })
}

fn prop_to_py(py: Python<'_>, value: &PropValue) -> PyResult<Py<PyAny>> {
    Ok(match value {
        PropValue::F64(v) => v.into_pyobject(py)?.into_any().unbind(),
        PropValue::Int(v) => v.into_pyobject(py)?.into_any().unbind(),
        PropValue::Str(s) => s.into_pyobject(py)?.into_any().unbind(),
        PropValue::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
    })
}

/// Resolve a kind name to a [`KindId`], or raise a Python `ValueError`.
fn kind_id_checked(mol: &MolGraph, kind: &str) -> PyResult<KindId> {
    mol.kind_id(kind)
        .ok_or_else(|| PyValueError::new_err(format!("kind '{kind}' is not registered")))
}

/// Render a [`CenterError`] as a Python `ValueError`, through
/// [`center_error_message`].
fn center_error_to_pyerr(e: CenterError) -> PyErr {
    PyValueError::new_err(center_error_message(e))
}

/// The message of a [`CenterError`] as Python sees it.
///
/// Node ids cross as the `int` handles Python holds (`node_to_u64`), never as
/// the Rust debug form `NodeId(3v1)`. Every variant is matched by name, so a
/// new variant is a compile error here rather than a silent fallback.
pub(crate) fn center_error_message(e: CenterError) -> String {
    match e {
        CenterError::Empty => "center of an empty node set".to_owned(),
        CenterError::NotFound { node } => {
            format!("node {} is not in this graph", node_to_u64(node))
        }
        CenterError::BadPosition { node } => format!(
            "node {} has a missing or non-finite '{}'/'{}'/'{}'",
            node_to_u64(node),
            keys::X,
            keys::Y,
            keys::Z
        ),
        CenterError::BadMass { node } => format!(
            "node {} has a missing, negative or non-finite '{}'",
            node_to_u64(node),
            keys::MASS
        ),
        CenterError::ZeroMass => "total mass is not positive and finite".to_owned(),
    }
}

/// Render a [`LinkError`] as a Python `ValueError`, through
/// [`link_error_message`].
fn link_error_to_pyerr(e: LinkError) -> PyErr {
    PyValueError::new_err(link_error_message(e))
}

/// The message of a [`LinkError`] as Python sees it.
///
/// Port and atom ids cross as the `int` handles Python holds
/// (`relation_to_u64` / `node_to_u64`), never as `PortId(..)` / `NodeId(..)`.
/// Every variant is matched by name, with no catch-all arm.
pub(crate) fn link_error_message(e: LinkError) -> String {
    let port = relation_to_u64;
    let atom = node_to_u64;
    match e {
        LinkError::Port(inner) => format!("port does not read back: {inner}"),
        LinkError::StalePort { port: p } => format!(
            "port {} is stale: its anchor–handle bond no longer exists",
            port(p)
        ),
        LinkError::Incompatible { a, b } => {
            format!("ports {} and {} are not compatible", port(a), port(b))
        }
        LinkError::SameAnchor { a, b } => format!(
            "ports {} and {} share one anchor; a bond cannot join an atom to itself",
            port(a),
            port(b)
        ),
        LinkError::AlreadyBonded { a, b } => {
            format!("anchors {} and {} are already bonded", atom(a), atom(b))
        }
        LinkError::BranchReachesAnchor { port: p } => {
            format!("port {}'s handle branch reaches its own anchor", port(p))
        }
        LinkError::BranchesOverlap => "the two ports' handle branches overlap".to_owned(),
        LinkError::OneSidedCharge { anchor } => format!(
            "charge is present on only part of anchor {} and its handle branch",
            atom(anchor)
        ),
        LinkError::Graph(inner) => format!("graph refused a read: {inner}"),
    }
}

// ---------------------------------------------------------------------------
// Shared generic-world method body
// ---------------------------------------------------------------------------

/// Emits the generic ECS world `#[pymethods]` for a graph type. Always operates
/// on `self.mol()` / `self.mol_mut()` (the receiver's own graph), so each
/// concrete type's graph is the single data slot — a leaf's methods read/write
/// the leaf's own core graph, never an empty base.
macro_rules! graph_world_impl {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            // ---- ports: named attachment points any graph may carry ----

            /// Record a descriptor on the ``(anchor, handle)`` valence; the
            /// ``ports`` relation kind is registered on first use.
            ///
            /// Parameters
            /// ----------
            /// anchor : int
            ///     The node that keeps its place in the product.
            /// handle : int
            ///     A node bonded to ``anchor``: the root of the leaving group.
            ///     Endpoint order is load-bearing.
            /// kind : str
            ///     The notation glyph — one of ``"$"``, ``"<"``, ``">"``, ``"!"``.
            /// label : str, optional
            ///     Free-form descriptor label; ``""`` (the default) means unnamed.
            /// order : int, optional
            ///     Multiplicity of the bond this port will form, ``1`` to ``4``
            ///     (default ``1``).
            ///
            /// Returns
            /// -------
            /// int
            ///     The port's stable relation handle.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If ``kind`` is not one of the four glyphs, ``handle`` is not
            ///     bonded to ``anchor``, ``order`` is not a definite bond
            ///     number, the valence already carries a port, or a handle is
            ///     stale or unknown.
            #[pyo3(signature = (anchor, handle, kind, label="", order=1))]
            fn add_port(
                &mut self,
                anchor: u64,
                handle: u64,
                kind: &str,
                label: &str,
                order: u32,
            ) -> PyResult<u64> {
                let kind = PortKind::from_str(kind).map_err(molrs_error_to_pyerr)?;
                self.mol_mut()
                    .add_port(
                        node_from_u64(anchor),
                        node_from_u64(handle),
                        kind,
                        label,
                        BondNumber::from_code(order),
                    )
                    .map(relation_to_u64)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Number of ports (``0`` when the graph carries none).
            #[getter]
            fn n_ports(&self) -> usize {
                self.mol().n_ports()
            }

            /// Record the unit instance ``node`` came from, under ``frag_id``.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If ``id`` exceeds the widest identifier a node column stores,
            ///     or ``node`` is stale or unknown.
            fn set_frag_id(&mut self, node: u64, id: u32) -> PyResult<()> {
                self.mol_mut()
                    .set_frag_id(node_from_u64(node), id)
                    .map_err(molrs_error_to_pyerr)
            }

            /// The unit instance ``node`` came from, or ``None``.
            fn frag_id(&self, node: u64) -> Option<u32> {
                self.mol().frag_id(node_from_u64(node))
            }

            /// Propagate each ``frag_id`` to the unlabelled degree-1 nodes
            /// hanging off a labelled one (one pass); returns how many were
            /// labelled. The relabel step after a conformer added hydrogens.
            fn inherit_frag_ids(&mut self) -> usize {
                self.mol_mut().inherit_frag_ids()
            }

            /// Join port ``a`` to port ``b`` with a new anchor–anchor bond.
            ///
            /// Both leaving groups are removed, their partial charge (e) folds
            /// onto the anchors, and the anchors are bonded with the port
            /// order. No coordinate moves.
            ///
            /// Returns
            /// -------
            /// int
            ///     The new bond's relation handle.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If a handle names no live port, a port is stale, the two
            ///     ports do not accept each other, share an anchor, their
            ///     anchors are already bonded, or the leaving groups overlap or
            ///     reach an anchor; the graph is unchanged.
            /// OverflowError
            ///     If ``a`` or ``b`` is negative.
            fn link(&mut self, a: u64, b: u64) -> PyResult<u64> {
                self.mol_mut()
                    .link(relation_from_u64(a), relation_from_u64(b))
                    .map(relation_to_u64)
                    .map_err(link_error_to_pyerr)
            }

            // ---- entities ----

            /// Spawn a new entity, returning its stable handle.
            fn spawn(&mut self) -> u64 {
                node_to_u64(self.mol_mut().add_node())
            }

            /// Remove an entity (cascades incident relations). Errors if stale.
            fn despawn(&mut self, h: u64) -> PyResult<()> {
                self.mol_mut()
                    .remove_node(node_from_u64(h))
                    .map(|_| ())
                    .map_err(molrs_error_to_pyerr)
            }

            /// All live entity handles, in row order.
            fn entities(&self) -> Vec<u64> {
                self.mol().node_ids().map(node_to_u64).collect()
            }

            /// Whether `h` is a live entity handle.
            fn has_entity(&self, h: u64) -> bool {
                self.mol().node_table().contains(node_from_u64(h))
            }

            /// Number of entities.
            #[getter]
            fn n_nodes(&self) -> usize {
                self.mol().n_nodes()
            }

            // ---- components ----

            /// Read entity `h`'s component `key` (``None`` if absent).
            ///
            /// `key` is a :class:`molrs.keys.Key` or ``str``.
            fn get(&self, py: Python<'_>, h: u64, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let key = crate::schema::extract_column_key(key)?;
                match self.mol().node_table().value(node_from_u64(h), &key) {
                    Some(cell) => cell_to_py(py, cell),
                    None => Ok(py.None()),
                }
            }

            /// Set entity `h`'s component `key` (``value`` is int|float|str).
            ///
            /// `key` is a :class:`molrs.keys.Key` or ``str``.
            fn set(
                &mut self,
                h: u64,
                key: &Bound<'_, PyAny>,
                value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let key = crate::schema::extract_column_key(key)?;
                let pv = py_to_prop(value)?;
                self.mol_mut()
                    .set_node(node_from_u64(h), &key, pv)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Whether entity `h` has component `key`.
            ///
            /// `key` is a :class:`molrs.keys.Key` or ``str``.
            fn has(&self, h: u64, key: &Bound<'_, PyAny>) -> PyResult<bool> {
                let key = crate::schema::extract_column_key(key)?;
                Ok(self.mol().node_table().has(node_from_u64(h), &key))
            }

            /// Clear entity `h`'s component `key` (no-op if absent).
            ///
            /// `key` is a :class:`molrs.keys.Key` or ``str``.
            fn delete(&mut self, h: u64, key: &Bound<'_, PyAny>) -> PyResult<()> {
                let key = crate::schema::extract_column_key(key)?;
                self.mol_mut()
                    .clear_node(node_from_u64(h), &key)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Component keys currently set on entity `h`, in column order.
            fn node_keys(&self, h: u64) -> Vec<String> {
                self.mol()
                    .node_table()
                    .row_cells(node_from_u64(h))
                    .map(|(k, _)| k.to_owned())
                    .collect()
            }

            // ---- relations ----

            /// Register a relation kind (idempotent for a matching arity).
            fn register_kind(&mut self, kind: &str, arity: usize) -> PyResult<()> {
                let m = self.mol_mut();
                if let Some(kid) = m.kind_id(kind) {
                    let existing = m.arity(kid);
                    if existing != arity {
                        return Err(PyValueError::new_err(format!(
                            "kind '{kind}' already registered with arity {existing}, got {arity}"
                        )));
                    }
                    return Ok(());
                }
                m.register_kind(kind, arity);
                Ok(())
            }

            /// Names of all registered relation kinds.
            fn kinds(&self) -> Vec<String> {
                self.mol()
                    .kind_ids()
                    .map(|kid| self.mol().kind_name(kid).to_owned())
                    .collect()
            }

            /// Fixed endpoint count for a registered relation kind.
            fn kind_arity(&self, kind: &str) -> PyResult<usize> {
                let kid = kind_id_checked(self.mol(), kind)?;
                Ok(self.mol().arity(kid))
            }

            /// Add a relation of `kind` over node handles, returning its handle.
            fn add_relation(&mut self, kind: &str, nodes: Vec<u64>) -> PyResult<u64> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let nids: Vec<NodeId> = nodes.into_iter().map(node_from_u64).collect();
                let rid = self
                    .mol_mut()
                    .add_relation(kid, &nids)
                    .map_err(molrs_error_to_pyerr)?;
                Ok(relation_to_u64(rid))
            }

            /// Endpoint node handles of relation `rh` of `kind`.
            fn relation_nodes(&self, kind: &str, rh: u64) -> PyResult<Vec<u64>> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let nodes = self
                    .mol()
                    .relation_nodes(kid, relation_from_u64(rh))
                    .map_err(molrs_error_to_pyerr)?;
                Ok(nodes.iter().map(|&n| node_to_u64(n)).collect())
            }

            /// Relations of `kind` incident to node `nh`, as
            /// `(relation_handle, other_node_handle)` pairs, via the adjacency
            /// index (O(degree)). Only arity-2 kinds are tracked in adjacency.
            fn incident_relations(&self, nh: u64, kind: &str) -> PyResult<Vec<(u64, u64)>> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let nid = node_from_u64(nh);
                Ok(self
                    .mol()
                    .neighbor_relations(nid)
                    .filter(|(k, _, _)| *k == kid)
                    .map(|(_, rid, other)| (relation_to_u64(rid), node_to_u64(other)))
                    .collect())
            }

            /// Set a property on relation `rh` of `kind`.
            fn set_relation_prop(
                &mut self,
                kind: &str,
                rh: u64,
                key: &str,
                value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let pv = py_to_prop(value)?;
                self.mol_mut()
                    .set_relation_prop(kid, relation_from_u64(rh), key, pv)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Read a property of relation `rh` of `kind` (``None`` if absent).
            fn get_relation_prop(
                &self,
                py: Python<'_>,
                kind: &str,
                rh: u64,
                key: &str,
            ) -> PyResult<Py<PyAny>> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let rel = self
                    .mol()
                    .get_relation(kid, relation_from_u64(rh))
                    .map_err(molrs_error_to_pyerr)?;
                match rel.props.get(key) {
                    Some(v) => prop_to_py(py, v),
                    None => Ok(py.None()),
                }
            }

            /// Property keys currently set on relation `rh` of `kind`.
            fn relation_keys(&self, kind: &str, rh: u64) -> PyResult<Vec<String>> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let rel = self
                    .mol()
                    .get_relation(kid, relation_from_u64(rh))
                    .map_err(molrs_error_to_pyerr)?;
                Ok(rel.props.keys().map(|k| k.to_owned()).collect())
            }

            /// Clear property `key` on relation `rh` of `kind` (no-op if absent).
            fn delete_relation_prop(&mut self, kind: &str, rh: u64, key: &str) -> PyResult<()> {
                let kid = kind_id_checked(self.mol(), kind)?;
                self.mol_mut()
                    .clear_relation_prop(kid, relation_from_u64(rh), key)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Remove relation `rh` of `kind`.
            fn remove_relation(&mut self, kind: &str, rh: u64) -> PyResult<()> {
                let kid = kind_id_checked(self.mol(), kind)?;
                self.mol_mut()
                    .remove_relation(kid, relation_from_u64(rh))
                    .map(|_| ())
                    .map_err(molrs_error_to_pyerr)
            }

            /// Number of relations of `kind`.
            fn n_relations(&self, kind: &str) -> PyResult<usize> {
                let kid = kind_id_checked(self.mol(), kind)?;
                Ok(self.mol().n_relations(kid))
            }

            /// Live relation handles of `kind`, in row order.
            ///
            /// Authoritative enumeration — callers must not probe opaque handle
            /// ranges. Returns an empty list for a registered kind with no
            /// relations; errors only if `kind` is unregistered.
            fn relation_ids(&self, kind: &str) -> PyResult<Vec<u64>> {
                let kid = kind_id_checked(self.mol(), kind)?;
                Ok(self.mol().relation_ids(kid).map(relation_to_u64).collect())
            }

            // ---- zero-copy columns ----

            /// The component column `key`, aligned to row order (length ==
            /// `n_nodes`), as a numpy array of the column's own element type.
            ///
            /// A `f64` column is a **zero-copy view** that writes through to the
            /// world (`col[i] = v` updates the entity at row `i`); `i32`, `bool`
            /// and `str` columns are copied. The view borrows the world's storage;
            /// structural mutation (`spawn`/`despawn`) may reallocate or reorder
            /// the column and invalidate an outstanding view — re-fetch after
            /// such ops.
            ///
            /// Every entity must carry the component: a column with a hole is a
            /// `KeyError` naming how many entities lack it, never a silently
            /// zero-filled array. Use :meth:`validity` to find the holes and
            /// :meth:`get` for entity-wise reads.
            fn column<'py>(
                slf: Bound<'py, $ty>,
                key: &Bound<'_, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                use molrs::system::entity_table::Column;

                let key = crate::schema::extract_column_key(key)?;
                let py = slf.py();
                let this = slf.borrow();
                let table = this.mol().node_table();
                let validity = table.col_validity(&key).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "component '{key}' is absent from every entity"
                    ))
                })?;
                let holes = validity.as_slice().iter().filter(|v| !**v).count();
                if holes > 0 {
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "component '{key}' is absent on {holes} of {} entities; a column \
                         needs it on every one (see validity())",
                        validity.len()
                    )));
                }
                let column = table.column(&key).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "component '{key}' is absent from every entity"
                    ))
                })?;
                match column {
                    Column::F64(data, _) => {
                        let (ptr, len) = (data.as_ptr(), data.len());
                        drop(this);
                        // SAFETY: `slf` owns the backing Vec and is held as the array's base
                        // object, so the memory stays valid for the array's lifetime; the
                        // documented contract forbids structural mutation while held.
                        let view = unsafe { numpy::ndarray::ArrayView1::from_shape_ptr(len, ptr) };
                        Ok(
                            unsafe { numpy::PyArray1::borrow_from_array(&view, slf.into_any()) }
                                .into_any(),
                        )
                    }
                    Column::I32(data, _) => Ok(numpy::PyArray1::from_slice(py, data).into_any()),
                    Column::Bool(data, _) => Ok(numpy::PyArray1::from_slice(py, data).into_any()),
                    Column::Str(data, _) => {
                        let list = pyo3::types::PyList::new(py, data.iter().map(String::as_str))?;
                        py.import("numpy")?.call_method1("array", (list,))
                    }
                }
            }

            /// Names of every component column registered on the node table
            /// (a component set on at least one entity), in no particular order.
            fn columns(&self) -> Vec<String> {
                self.mol()
                    .node_table()
                    .columns()
                    .map(str::to_owned)
                    .collect()
            }

            /// Validity mask (numpy `bool` array, copied) of component column `key`,
            /// aligned to row order. `True` where the entity at that row has the
            /// component set.
            ///
            /// `key` is a :class:`molrs.keys.Key` or ``str``.
            fn validity<'py>(
                &self,
                py: Python<'py>,
                key: &Bound<'_, PyAny>,
            ) -> PyResult<Bound<'py, numpy::PyArray1<bool>>> {
                let key = crate::schema::extract_column_key(key)?;
                let valid =
                    self.mol().node_table().col_validity(&key).ok_or_else(|| {
                        PyValueError::new_err(format!("column '{key}' is absent"))
                    })?;
                Ok(numpy::PyArray1::from_slice(py, valid.as_slice()))
            }

            // ---- zero-copy adopt ----

            /// Zero-copy adopt: **move** `other`'s graph storage into `self`,
            /// leaving `other` empty. Handles in the adopted graph stay valid
            /// (the whole generational slotmap is moved, not reindexed). For
            /// taking ownership of a graph produced elsewhere without a per-node
            /// copy. Defined per leaf so it swaps the leaf's own backing store.
            fn adopt(&mut self, other: &mut $ty) {
                self.inner = std::mem::take(&mut other.inner);
            }
        }
    };
}

// ---------------------------------------------------------------------------
// ExtractedSubgraph — result of radius-ball extraction
// ---------------------------------------------------------------------------

/// Result of :meth:`Atomistic.extract_subgraph` / :meth:`CoarseGrain.extract_subgraph`.
///
/// * ``graph`` — the extracted leaf (``Atomistic`` or ``CoarseGrain``)
/// * ``boundary`` — parent handles with a neighbor outside the ball
/// * ``parent_of`` — ``{new_handle: parent_handle}``
/// * ``hops`` — ``{parent_handle: hops_from_nearest_center}``
/// * ``node_map`` — ``{parent_handle: new_handle}``
#[pyclass(module = "molrs", name = "ExtractedSubgraph", skip_from_py_object)]
pub struct PyExtractedSubgraph {
    graph: Py<PyAny>,
    boundary: Vec<u64>,
    parent_of: HashMap<u64, u64>,
    hops: HashMap<u64, i64>,
    node_map: HashMap<u64, u64>,
}

#[pymethods]
impl PyExtractedSubgraph {
    #[new]
    fn new(
        graph: Py<PyAny>,
        boundary: Vec<u64>,
        parent_of: HashMap<u64, u64>,
        hops: HashMap<u64, i64>,
        node_map: HashMap<u64, u64>,
    ) -> Self {
        Self {
            graph,
            boundary,
            parent_of,
            hops,
            node_map,
        }
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyTuple>)> {
        crate::helpers::reduce_via_type(
            slf.as_any(),
            (
                slf.getattr("graph")?,
                slf.getattr("boundary")?,
                slf.getattr("parent_of")?,
                slf.getattr("hops")?,
                slf.getattr("node_map")?,
            ),
        )
    }

    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyAny> {
        self.graph.clone_ref(py)
    }

    #[getter]
    fn boundary(&self) -> Vec<u64> {
        self.boundary.clone()
    }

    #[getter]
    fn parent_of(&self) -> HashMap<u64, u64> {
        self.parent_of.clone()
    }

    #[getter]
    fn hops(&self) -> HashMap<u64, i64> {
        self.hops.clone()
    }

    #[getter]
    fn node_map(&self) -> HashMap<u64, u64> {
        self.node_map.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtractedSubgraph(n_boundary={}, n_nodes_mapped={})",
            self.boundary.len(),
            self.node_map.len()
        )
    }
}

impl PyExtractedSubgraph {
    fn from_atomistic(py: Python<'_>, ext: ExtractedAtomistic) -> PyResult<Self> {
        let graph = PyAtomistic::from_core(py, ext.graph)?.into_any();
        Ok(Self {
            graph,
            boundary: ext.boundary.into_iter().map(node_to_u64).collect(),
            parent_of: ext
                .parent_of
                .into_iter()
                .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
                .collect(),
            hops: ext
                .hops
                .into_iter()
                .map(|(k, v)| (node_to_u64(k), v))
                .collect(),
            node_map: ext
                .node_map
                .into_iter()
                .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
                .collect(),
        })
    }

    fn from_coarsegrain(py: Python<'_>, ext: ExtractedCoarseGrain) -> PyResult<Self> {
        let graph = PyCoarseGrain::from_core(py, ext.graph)?.into_any();
        Ok(Self {
            graph,
            boundary: ext.boundary.into_iter().map(node_to_u64).collect(),
            parent_of: ext
                .parent_of
                .into_iter()
                .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
                .collect(),
            hops: ext
                .hops
                .into_iter()
                .map(|(k, v)| (node_to_u64(k), v))
                .collect(),
            node_map: ext
                .node_map
                .into_iter()
                .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
                .collect(),
        })
    }
}

// ---------------------------------------------------------------------------
// PyGraph — the generic world
// ---------------------------------------------------------------------------

/// Domain-agnostic ECS world, exposed to Python as `molrs.Graph`.
#[pyclass(module = "molrs", name = "Graph", subclass)]
pub struct PyGraph {
    inner: MolGraph,
}

impl PyGraph {
    fn mol(&self) -> &MolGraph {
        &self.inner
    }
    fn mol_mut(&mut self) -> &mut MolGraph {
        &mut self.inner
    }
}

#[pymethods]
impl PyGraph {
    /// Create an empty world. Extra args are accepted/ignored so a Python
    /// subclass needs no `__new__` shim.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyAny>, _kwargs: Option<&Bound<'_, PyAny>>) -> Self {
        Self {
            inner: MolGraph::new(),
        }
    }
}
graph_world_impl!(PyGraph);

/// Wrap a core graph leaf as the **public** Python class of the same name.
///
/// The package shadows each native leaf with a Python subclass that adds live
/// handle views ([`molrs.views`]), so a graph-out API must construct *that*
/// class — a bare pyclass instance would satisfy `isinstance` yet have no
/// `atoms`, no factories and the wrong `type(...)`. `T::NAME` is the
/// `name = "…"` already written in the `#[pyclass]` attribute, so the public
/// spelling is not declared a second time.
///
/// The `molrs` lookup is deliberately **per call**: `py.import` is a
/// `sys.modules` hit and `getattr` a type-dict read, whereas caching either
/// would make a late re-binding of the public class invisible for the life of
/// the process.
///
/// The whole-value assignment `*object.borrow_mut(py) = leaf` replaces the
/// leaf's single data slot; a leaf that ever grows a second field must assign
/// that slot by name instead.
pub(crate) fn from_core_shadowed<T>(py: Python<'_>, leaf: T) -> PyResult<Py<T>>
where
    T: PyClass<BaseType = PyGraph, Frozen = False>,
{
    // `T::NAME` needs the trait spelled out: the prelude also brings
    // `PyTypeInfo` into scope, whose `NAME` is the same string by a
    // different route. Either way it is the `name = "…"` already written in
    // the `#[pyclass]` attribute, never a second declaration of it.
    let public = py.import("molrs")?.getattr(<T as PyClass>::NAME)?;
    if public.is(py.get_type::<T>()) {
        return Py::new(
            py,
            (
                leaf,
                PyGraph {
                    inner: MolGraph::new(),
                },
            ),
        );
    }
    let object: Py<T> = public.call0()?.extract()?;
    *object.borrow_mut(py) = leaf;
    Ok(object)
}

/// A copy of the graph any Python graph object holds — a `Graph`, an
/// `Atomistic`, a `CoarseGrain` or a subclass of one — as a bare
/// [`MolGraph`]. Graph types are peers: this reads the object's own graph, it
/// converts nothing.
///
/// # Errors
///
/// `TypeError` when `obj` is no graph.
pub(crate) fn molgraph_of(obj: &Bound<'_, PyAny>) -> PyResult<MolGraph> {
    if let Ok(leaf) = obj.cast::<PyAtomistic>() {
        return Ok(leaf.borrow().inner.as_molgraph().clone());
    }
    if let Ok(leaf) = obj.cast::<PyCoarseGrain>() {
        return Ok(leaf.borrow().mol().clone());
    }
    if let Ok(graph) = obj.cast::<PyGraph>() {
        return Ok(graph.borrow().inner.clone());
    }
    Err(PyTypeError::new_err(format!(
        "expected a graph (Graph, Atomistic, CoarseGrain), not {}",
        obj.get_type().name()?
    )))
}

/// Hand a finished `graph` back as an instance of the graph class `cls`
/// (`Graph`, `Atomistic`, `CoarseGrain`, or a subclass of one); `None` means
/// `Graph`. The factory of every graph-producing API.
///
/// # Errors
///
/// `TypeError` when `cls` is no graph class; `ValueError` when `graph` breaks
/// the class's invariant (an `Atomistic` node without `element`).
pub(crate) fn graph_as(
    py: Python<'_>,
    graph: MolGraph,
    cls: Option<&Bound<'_, pyo3::types::PyType>>,
) -> PyResult<Py<PyAny>> {
    let Some(cls) = cls else {
        return Ok(Py::new(py, PyGraph { inner: graph })?.into_any());
    };
    if cls.is_subclass_of::<PyAtomistic>()? {
        let leaf = Atomistic::try_from_molgraph(graph).map_err(molrs_error_to_pyerr)?;
        return Ok(PyAtomistic::from_core(py, leaf)?.into_any());
    }
    if cls.is_subclass_of::<PyCoarseGrain>()? {
        let leaf = CoarseGrain::try_from_molgraph(graph).map_err(molrs_error_to_pyerr)?;
        return Ok(PyCoarseGrain::from_core(py, leaf)?.into_any());
    }
    if cls.is_subclass_of::<PyGraph>()? {
        return Ok(Py::new(py, PyGraph { inner: graph })?.into_any());
    }
    Err(PyTypeError::new_err(format!(
        "cls must be a graph class (Graph, Atomistic, CoarseGrain), not {}",
        cls.name()?
    )))
}

// ---------------------------------------------------------------------------
// PyAtomistic — all-atom leaf (holds a core Atomistic)
// ---------------------------------------------------------------------------

/// All-atom molecular graph, exposed to Python as `molrs.Atomistic`.
///
/// Holds a core [`Atomistic`] from construction; it is never converted from a
/// `MolGraph`. Subclasses `Graph`; the generic API operates on this leaf's own
/// graph.
#[pyclass(module = "molrs._lib", name = "Atomistic", extends = PyGraph, subclass)]
pub struct PyAtomistic {
    inner: Atomistic,
}

impl PyAtomistic {
    fn mol(&self) -> &MolGraph {
        self.inner.as_molgraph()
    }
    fn mol_mut(&mut self) -> &mut MolGraph {
        self.inner.as_molgraph_mut()
    }
}

#[pymethods]
impl PyAtomistic {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyAny>, _kwargs: Option<&Bound<'_, PyAny>>) -> (Self, PyGraph) {
        (
            PyAtomistic {
                inner: Atomistic::new(),
            },
            PyGraph {
                inner: MolGraph::new(),
            },
        )
    }

    // ---- domain builders (operate on the core Atomistic directly) ----

    /// Add an atom with element `symbol` and optional coordinates. Returns its
    /// stable handle.
    #[pyo3(signature = (symbol, x=None, y=None, z=None))]
    fn add_atom(&mut self, symbol: &str, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> u64 {
        let id = match (x, y, z) {
            (Some(x), Some(y), Some(z)) => self.inner.add_atom_xyz(symbol, x, y, z),
            _ => self.inner.add_atom_bare(symbol),
        };
        node_to_u64(id)
    }

    /// Add a bond between two atom handles (default order 1.0). Returns its handle.
    fn add_bond(&mut self, a: u64, b: u64) -> PyResult<u64> {
        self.inner
            .add_bond(node_from_u64(a), node_from_u64(b))
            .map(relation_to_u64)
            .map_err(molrs_error_to_pyerr)
    }

    /// Add an angle over three atom handles (`j` central).
    fn add_angle(&mut self, i: u64, j: u64, k: u64) -> PyResult<u64> {
        self.inner
            .add_angle(node_from_u64(i), node_from_u64(j), node_from_u64(k))
            .map(relation_to_u64)
            .map_err(molrs_error_to_pyerr)
    }

    /// Add a dihedral over four atom handles.
    fn add_dihedral(&mut self, i: u64, j: u64, k: u64, l: u64) -> PyResult<u64> {
        self.inner
            .add_dihedral(
                node_from_u64(i),
                node_from_u64(j),
                node_from_u64(k),
                node_from_u64(l),
            )
            .map(relation_to_u64)
            .map_err(molrs_error_to_pyerr)
    }

    /// Add an improper over four atom handles.
    fn add_improper(&mut self, i: u64, j: u64, k: u64, l: u64) -> PyResult<u64> {
        self.inner
            .add_improper(
                node_from_u64(i),
                node_from_u64(j),
                node_from_u64(k),
                node_from_u64(l),
            )
            .map(relation_to_u64)
            .map_err(molrs_error_to_pyerr)
    }

    /// Perceive angle, dihedral and improper relations from the bond graph.
    ///
    /// Angles are 2-edge paths ``i-j-k`` and proper dihedrals 3-edge paths
    /// ``i-j-k-l`` over the bonds (graph-theory via the native `Topology`-backed
    /// ``Topology``). Impropers are the molecular-mechanics reading: one
    /// ``[centre, i, j, k]`` quartet per atom with **exactly three** neighbours,
    /// centre first, peripherals sorted — not every 3-combination at every
    /// centre of degree >= 3, which would hand an sp3 carbon four quartets.
    /// Whether a trivalent centre is planar enough to carry the term is
    /// force-field data, not a graph property, so every one is emitted.
    ///
    /// Idempotent; ``clear_existing`` wipes existing relations of the requested
    /// kinds first. Returns ``(n_angles_added, n_dihedrals_added,
    /// n_impropers_added)``.
    #[pyo3(signature = (gen_angle=true, gen_dihedral=true, gen_improper=false, clear_existing=false))]
    fn generate_topology(
        &mut self,
        gen_angle: bool,
        gen_dihedral: bool,
        gen_improper: bool,
        clear_existing: bool,
    ) -> PyResult<(usize, usize, usize)> {
        self.inner
            .generate_topology(gen_angle, gen_dihedral, gen_improper, clear_existing)
            .map_err(molrs_error_to_pyerr)
    }

    /// Single-source shortest-path (BFS) distances over the bond graph from
    /// `source` (a node handle), as `(node_handle, hops)` pairs for every atom
    /// reachable from `source` (including `source` itself at distance 0).
    /// Unreachable atoms (a different connected component) are omitted; an
    /// unknown `source` handle yields an empty list.
    #[pyo3(signature = (source, max_hops = None))]
    fn topo_distances(&self, source: u64, max_hops: Option<i64>) -> Vec<(u64, i64)> {
        self.inner
            .topo_distances(node_from_u64(source), max_hops)
            .into_iter()
            .map(|(a, d)| (node_to_u64(a), d))
            .collect()
    }

    /// Number of atoms.
    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Atom count of the largest fused/bridged ring system (naphthalene → 10).
    ///
    /// Acyclic molecules → ``0``. Pure structure fact for molpy region typing.
    fn max_ring_system_size(&self) -> usize {
        core_max_ring_system_size(&self.inner)
    }

    /// Export to a tabular [`Frame`] (atoms / bonds / angles / dihedrals /
    /// impropers blocks). Leaf-owned — `self.inner.to_frame()`, zero conversion.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an atom or relation property contradicts the dtype the Frame
    ///     schema declares for its key (a string under ``"x"``).
    fn to_frame(&self) -> PyResult<PyFrame> {
        PyFrame::from_core_frame(self.inner.to_frame().map_err(molrs_error_to_pyerr)?)
    }

    /// Build an `Atomistic` from a [`Frame`] (registers the chemistry kinds,
    /// then reads the relation blocks). A leaf constructor, not a conversion.
    #[staticmethod]
    fn from_frame(py: Python<'_>, frame: &PyFrame) -> PyResult<Py<PyAtomistic>> {
        let core = frame.clone_core_frame()?;
        let inner = Atomistic::from_frame(&core).map_err(molrs_error_to_pyerr)?;
        PyAtomistic::from_core(py, inner)
    }

    // ---- graph-edit conveniences (forward to core Atomistic fns) ----

    /// Remove an atom by handle, cascading incident bonds / angles / dihedrals /
    /// impropers. Errors if the handle is stale.
    fn remove_atom(&mut self, handle: u64) -> PyResult<()> {
        self.inner
            .remove_atom(node_from_u64(handle))
            .map(|_| ())
            .map_err(molrs_error_to_pyerr)
    }

    /// Remove a bond by handle (a relation handle from :meth:`add_bond`).
    fn remove_bond(&mut self, handle: u64) -> PyResult<()> {
        self.inner
            .remove_bond(relation_from_u64(handle))
            .map(|_| ())
            .map_err(molrs_error_to_pyerr)
    }

    /// Set a bond's chemical class and its localized bond number.
    ///
    /// The two are set together because they are only meaningful together: a
    /// class without a number leaves the bond un-standardized, and a number
    /// without a class leaves a consumer no way to tell aromatic from double.
    ///
    /// Parameters
    /// ----------
    /// handle : int
    ///     A relation handle from :meth:`add_bond`.
    /// bond_type : int
    ///     0 unknown, 1 single, 2 double, 3 triple, 4 aromatic.
    /// bond_number : int
    ///     0 unknown, 1 single, 2 double, 3 triple, 4 quadruple.
    fn set_bond_class(&mut self, handle: u64, bond_type: u32, bond_number: u32) -> PyResult<()> {
        self.inner
            .set_bond_class(
                relation_from_u64(handle),
                BondType::from_code(bond_type),
                BondNumber::from_code(bond_number),
            )
            .map_err(molrs_error_to_pyerr)
    }

    /// Set a plain (non-aromatic) bond, whose class implies its number.
    ///
    /// ``Aromatic`` (4) implies no number — the phase is a Kekulé assignment's
    /// to decide — so it goes through :meth:`set_bond_class`.
    fn set_bond_type(&mut self, handle: u64, bond_type: u32) -> PyResult<()> {
        self.inner
            .set_bond_type(relation_from_u64(handle), BondType::from_code(bond_type))
            .map_err(molrs_error_to_pyerr)
    }

    /// The bond's chemical class code; ``0`` when it has none.
    fn bond_type(&self, handle: u64) -> u32 {
        self.inner.bond_type(relation_from_u64(handle)).code()
    }

    /// The bond's localized (Kekulé) bond number; ``0`` when it has none.
    fn bond_number(&self, handle: u64) -> u32 {
        self.inner.bond_number(relation_from_u64(handle)).code()
    }

    /// Return an independent deep copy of this `Atomistic`.
    ///
    /// **Handles are preserved** (same generational keys as in ``self``).
    fn copy(&self, py: Python<'_>) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.clone())
    }

    /// Structural merge of ``other`` into ``self``.
    ///
    /// Consumes ``other``'s storage (``other`` is left empty). Every node handle
    /// from ``other`` is remapped; returns ``{old_handle: new_handle}``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a property of ``other`` contradicts the type this graph holds
    ///     for that key (a string ``tag`` into an int ``tag`` column).
    fn merge(&mut self, other: &mut Self) -> PyResult<HashMap<u64, u64>> {
        let taken = std::mem::take(&mut other.inner);
        Ok(self
            .inner
            .merge(taken)
            .map_err(molrs_error_to_pyerr)?
            .into_iter()
            .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
            .collect())
    }

    /// Induced subgraph on an explicit list of atom handles.
    ///
    /// Returns ``(subgraph, {parent_handle: new_handle})``. Stale handles raise.
    fn induced_subgraph(
        &self,
        py: Python<'_>,
        nodes: Vec<u64>,
    ) -> PyResult<(Py<PyAtomistic>, HashMap<u64, u64>)> {
        let ids: Vec<_> = nodes.into_iter().map(node_from_u64).collect();
        let (sub, map) = self
            .inner
            .induced_subgraph(&ids)
            .map_err(molrs_error_to_pyerr)?;
        let py_sub = PyAtomistic::from_core(py, sub)?;
        let py_map = map
            .into_iter()
            .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
            .collect();
        Ok((py_sub, py_map))
    }

    /// Radius ball around ``centers`` over the bond graph.
    ///
    /// When ``regenerate_topology`` is true, only bonds are copied and
    /// angles/dihedrals are perceived on the ball. When false, higher-order
    /// terms fully inside the ball are copied from the parent.
    ///
    /// ``max_ring_size`` closes the ball on ring systems: a ring of at most
    /// that many atoms which the radius only partly reaches comes along whole,
    /// together with everything fused or bridged to it. A cut ring is not a
    /// smaller molecule — ring perception, and with it aromaticity and every
    /// ring-aware atom type, reads the remainder as something else. The added
    /// atoms lie beyond ``radius``, so they arrive as context: their ``hops``
    /// are larger than the radius asked for.
    ///
    /// The bound is not a tuning knob, it is the claim's domain. Cutting a
    /// benzene changes the chemistry; cutting a 5000-membered macrocycle
    /// changes nothing any typifier can see, while closing on it would drag the
    /// whole loop into a ball that asked for nine atoms. Pass the largest ring
    /// your typifier can actually distinguish. ``None`` disables closure; the
    /// cost then, and with a bound, follows the ball and its neighbourhood —
    /// never the size of the parent graph.
    #[pyo3(signature = (centers, radius, *, regenerate_topology=false, max_ring_size=None))]
    fn extract_subgraph(
        &self,
        py: Python<'_>,
        centers: Vec<u64>,
        radius: i64,
        regenerate_topology: bool,
        max_ring_size: Option<usize>,
    ) -> PyResult<PyExtractedSubgraph> {
        let ids: Vec<_> = centers.into_iter().map(node_from_u64).collect();
        let groups = match max_ring_size {
            Some(max_ring_size) => {
                molrs::perceive::rings::small_ring_closure(&self.inner, &ids, radius, max_ring_size)
            }
            None => Vec::new(),
        };
        let ext = self
            .inner
            .extract_subgraph(&ids, radius, regenerate_topology, &groups)
            .map_err(molrs_error_to_pyerr)?;
        PyExtractedSubgraph::from_atomistic(py, ext)
    }

    // ---- structural graph hash (WL) ----

    /// Isomorphism-invariant Weisfeiler–Lehman structural hash (``int``).
    ///
    /// A stable, reproducible dedup key over the molecular graph
    /// (element / charge / aromatic node labels, bond-order edge labels):
    /// identical for a node-permuted copy, sensitive to any label or
    /// connectivity change. Reproducible across runs and processes.
    fn structural_hash(&self) -> u64 {
        self.inner.structural_hash()
    }

    /// Deterministic canonical atom ordering (a list of atom handles) from the
    /// WL refinement, so two isomorphic molecules line up node-by-node.
    fn canonical_order(&self) -> Vec<u64> {
        self.inner
            .canonical_order()
            .into_iter()
            .map(node_to_u64)
            .collect()
    }

    /// Whether `self` and `other` are isomorphic as labeled molecular graphs.
    fn is_isomorphic(&self, other: &PyAtomistic) -> bool {
        self.inner.is_isomorphic(other.core())
    }

    /// Mass-weighted centre of every atom.
    ///
    /// ``R = sum_i m_i r_i / sum_i m_i`` over all atoms, reading positions
    /// from ``x`` / ``y`` / ``z`` (Å) and masses from ``mass`` (g/mol). No
    /// periodic imaging: a molecule split across a box face must be unwrapped
    /// first (:meth:`Box.unwrap`).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,), float64
    ///     The centre in Å.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the molecule has no atoms; if an atom lacks a finite ``x`` /
    ///     ``y`` / ``z`` or a finite, non-negative ``mass`` (the message names
    ///     its int handle); or if the total mass is not positive.
    fn center<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let center = self.inner.center().map_err(center_error_to_pyerr)?;
        Ok(vector_to_py(py, &center))
    }
}
graph_world_impl!(PyAtomistic);

impl PyAtomistic {
    /// Wrap an existing core [`Atomistic`] as a Python `Atomistic` object.
    ///
    /// Graph-out APIs (perception, typifiers, copy/from_frame, SMILES) go
    /// through here so they never silently drop the Python view layer.
    pub(crate) fn from_core(py: Python<'_>, inner: Atomistic) -> PyResult<Py<PyAtomistic>> {
        from_core_shadowed(py, PyAtomistic { inner })
    }

    /// Borrow the held core [`Atomistic`] (for domain consumers like the
    /// conformer / force-field typifier that operate on atomistic chemistry).
    pub(crate) fn core(&self) -> &Atomistic {
        &self.inner
    }

    /// Mutably borrow the held core [`Atomistic`] (for in-place chemistry
    /// systems like `perceive_aromaticity` / `compute_gasteiger_charges`).
    pub(crate) fn core_mut(&mut self) -> &mut Atomistic {
        &mut self.inner
    }
}

// ---------------------------------------------------------------------------
// PySmartsMatch / PySmartsPattern — atom-map-aware SMARTS matcher over Atomistic
// ---------------------------------------------------------------------------

/// One SMARTS match, exposed to Python as `molrs.SmartsMatch`.
///
/// ``atoms`` stores molecule atom handles in query-atom order. ``mapping``
/// stores the Daylight atom-map projection (``:1`` → atom handle), and is empty
/// when the query carries no map labels.
#[pyclass(module = "molrs.perceive", name = "SmartsMatch", skip_from_py_object)]
#[derive(Clone)]
pub struct PySmartsMatch {
    atoms: Vec<u64>,
    mapping: HashMap<u32, u64>,
}

#[pymethods]
impl PySmartsMatch {
    /// Molecule atom handles in query-atom order.
    #[getter]
    fn atoms(&self) -> Vec<u64> {
        self.atoms.clone()
    }

    /// Daylight atom-map projection (``:n`` label -> molecule atom handle).
    #[getter]
    fn mapping(&self) -> HashMap<u32, u64> {
        self.mapping.clone()
    }

    fn as_list(&self) -> Vec<u64> {
        self.atoms.clone()
    }

    fn as_dict(&self) -> HashMap<u32, u64> {
        self.mapping.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SmartsMatch(atoms={:?}, mapping={:?})",
            self.atoms, self.mapping
        )
    }
}

/// Compiled SMARTS query, exposed to Python as `molrs.SmartsPattern`.
///
/// A thin wrapper over the core [`SmartsPattern`] (`molrs/src/core/chem/smarts`)
/// — the same backtracking subgraph-isomorphism engine that drives the OPLS-AA
/// typifier. Matching is non-uniquified (RDKit ``uniquify=False``): every
/// distinct query-atom → mol-atom embedding is reported as a
/// :class:`SmartsMatch`.
///
/// Daylight atom maps (``[C:1]``) are parsed and carried through but add **no**
/// match constraint (they are "ignored in molecule SMARTS"); pass
/// ``mapped=True`` to :meth:`find_matches` for the legacy shortcut returning
/// ``{map_number: atom_handle}`` dictionaries.
///
/// Examples
/// --------
/// >>> pat = molrs.SmartsPattern("[C:1][O:2][H:3]")
/// >>> pat.find_matches(methanol)[0].mapping
/// {1: <C>, 2: <O>, 3: <H>}
#[pyclass(module = "molrs.perceive", name = "SmartsPattern")]
pub struct PySmartsPattern {
    inner: SmartsPattern,
}

#[pymethods]
impl PySmartsPattern {
    /// Parse a SMARTS string. Raises ``ValueError`` on a syntax error.
    #[new]
    fn new(smarts: &str) -> PyResult<Self> {
        let inner = SmartsPattern::parse(smarts).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Whether at least one match exists in `mol`.
    #[pyo3(signature = (mol, *, labels=None, root=None))]
    fn has_match(
        &self,
        mol: &PyAtomistic,
        labels: Option<HashMap<u64, String>>,
        root: Option<u64>,
    ) -> bool {
        let core_labels = labels.map(|labels| {
            labels
                .into_iter()
                .map(|(h, l)| (node_from_u64(h), l))
                .collect::<HashMap<NodeId, String>>()
        });
        self.inner.has_match(
            mol.core(),
            MatchOptions {
                labels: core_labels.as_ref(),
                root: root.map(node_from_u64),
                limit: None,
            },
        )
    }

    /// All matches. By default each match is a :class:`SmartsMatch`; with
    /// ``mapped=True`` each match is returned as a ``{atom_map_number:
    /// atom_handle}`` dict. ``labels`` supplies the ``%LABEL`` context, ``root``
    /// pins query atom 0 to one atom handle, and ``limit`` stops after N
    /// embeddings.
    #[pyo3(signature = (mol, *, labels=None, root=None, mapped=false, limit=None))]
    fn find_matches(
        &self,
        py: Python<'_>,
        mol: &PyAtomistic,
        labels: Option<HashMap<u64, String>>,
        root: Option<u64>,
        mapped: bool,
        limit: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let core_labels = labels.map(|labels| {
            labels
                .into_iter()
                .map(|(h, l)| (node_from_u64(h), l))
                .collect::<HashMap<NodeId, String>>()
        });
        let matches = self.inner.find(
            mol.core(),
            MatchOptions {
                labels: core_labels.as_ref(),
                root: root.map(node_from_u64),
                limit,
            },
        );
        if mapped {
            let out: Vec<HashMap<u32, u64>> = matches
                .iter()
                .map(|m| {
                    self.inner
                        .mapped(m)
                        .into_iter()
                        .map(|(label, atom)| (label, node_to_u64(atom)))
                        .collect()
                })
                .collect();
            return Ok(out.into_pyobject(py)?.into_any().unbind());
        }
        let out: Vec<PySmartsMatch> = matches
            .iter()
            .map(|m| PySmartsMatch {
                atoms: m.atoms().iter().map(|&atom| node_to_u64(atom)).collect(),
                mapping: self
                    .inner
                    .mapped(m)
                    .into_iter()
                    .map(|(label, atom)| (label, node_to_u64(atom)))
                    .collect(),
            })
            .collect();
        Ok(out.into_pyobject(py)?.into_any().unbind())
    }

    /// Number of query atoms in the pattern.
    #[getter]
    fn num_query_atoms(&self) -> usize {
        self.inner.num_query_atoms()
    }

    /// Longest shortest-path length (bonds) on the query atom graph.
    ///
    /// Isolated atoms → ``0``. Pure syntax fact for molpy region typing.
    #[getter]
    fn max_bond_depth(&self) -> usize {
        self.inner.max_bond_depth()
    }

    /// Ring primitives used in this pattern (syntax only; no boundedness).
    ///
    /// Each item is ``(kind, n)`` where ``kind`` is one of
    /// ``"sized"`` / ``"membership"`` / ``"ring_count"`` / ``"ring_bond_count"``
    /// and ``n`` is ``None`` for membership.
    #[getter]
    fn ring_primitives(&self) -> Vec<(String, Option<u32>)> {
        self.inner
            .ring_primitives()
            .into_iter()
            .map(|p| match p {
                RingPrimitive::Sized(n) => ("sized".into(), Some(n)),
                RingPrimitive::Membership => ("membership".into(), None),
                RingPrimitive::RingCount(n) => ("ring_count".into(), Some(n)),
                RingPrimitive::RingBondCount(n) => ("ring_bond_count".into(), Some(n)),
            })
            .collect()
    }

    /// The ``:n`` atom-map label of query atom `query_atom` (``None`` if
    /// unlabelled / out of range).
    fn map_label(&self, query_atom: usize) -> Option<u32> {
        self.inner.map_label(query_atom)
    }

    fn __repr__(&self) -> String {
        format!(
            "SmartsPattern(num_query_atoms={})",
            self.inner.num_query_atoms()
        )
    }
}

// ---------------------------------------------------------------------------
// PyReaction — Daylight reaction-SMARTS (SMIRKS) transform over an Atomistic
// ---------------------------------------------------------------------------

/// Compiled reaction SMARTS, exposed to Python as `molrs.Reaction`.
///
/// A thin wrapper over the core [`Reaction`] (`molrs/src/core/chem/smarts`).
/// Parses ``reactants >> products`` (tolerating an ignored ``>agent>`` field),
/// derives the graph edit from the Daylight atom-map diff, and applies it to one
/// matched occurrence in place. Reacting atoms may carry SMARTS queries
/// (RDKit-style reaction SMARTS); only concrete product atoms are addable.
///
/// Examples
/// --------
/// >>> rxn = molrs.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
/// >>> rxn.forming_bonds                 # [(1, 2)]
/// >>> binding = {}                       # match each reactant component ...
/// >>> for pat in rxn.reactant_patterns:  # ... and merge the map->atom dicts
/// ...     binding.update(pat.find_matches(mol, mapped=True)[0])
/// >>> rxn.apply(mol, binding)            # edits `mol` in place
#[pyclass(module = "molrs", name = "Reaction")]
pub struct PyReaction {
    inner: Reaction,
}

#[pymethods]
impl PyReaction {
    /// Parse a reaction SMARTS. Raises ``ValueError`` on a syntax or
    /// map-consistency error (e.g. an atom map that appears on only one side).
    #[new]
    fn new(reaction_smarts: &str) -> PyResult<Self> {
        let inner = Reaction::parse(reaction_smarts).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyTuple>)> {
        crate::helpers::reduce_via_type(slf.as_any(), (slf.borrow().inner.source().to_owned(),))
    }

    /// The reactant components (LHS), one :class:`SmartsPattern` per top-level
    /// ``.`` component, for matching / pairing each independently.
    #[getter]
    fn reactant_patterns(&self) -> Vec<PySmartsPattern> {
        self.inner
            .reactants()
            .iter()
            .map(|p| PySmartsPattern { inner: p.clone() })
            .collect()
    }

    /// The ``(map_a, map_b)`` pairs of newly formed bonds between preserved
    /// atoms — the distance criterion for picking a reacting occurrence. Bonds
    /// that merely change order, and bonds to added atoms, are excluded.
    #[getter]
    fn forming_bonds(&self) -> Vec<(u32, u32)> {
        self.inner.forming_bonds()
    }

    /// Apply the transform to `mol` in place at the occurrence pinned by
    /// `binding` (``{map_number: atom_handle}``). Deletes unmapped-LHS atoms,
    /// adds unmapped-RHS atoms (no coordinates), forms/breaks bonds, then
    /// regenerates angle/dihedral topology and re-perceives aromaticity.
    ///
    /// Returns the deduplicated, deterministically-ordered list of *surviving*
    /// touched atom handles (formed/broken/order-changed bond endpoints, added
    /// atoms, deleted atoms' surviving neighbours, and prop-set atoms). Deleted
    /// atoms' own handles are never included. The caller expands this seed set
    /// into a retype-safe region.
    ///
    /// ``refresh=False`` skips the per-apply whole-graph angle/dihedral
    /// regeneration + aromaticity re-perception: a batch caller (crosslinking a
    /// melt with many edits) passes it and refreshes ONCE at the end, turning an
    /// O(edits × N) cost into O(edits × local). Matching only needs bonds, which
    /// are updated in place regardless.
    #[pyo3(signature = (mol, binding, labels=None, refresh=true))]
    fn apply(
        &self,
        mol: &mut PyAtomistic,
        binding: HashMap<u32, u64>,
        labels: Option<HashMap<u64, String>>,
        refresh: bool,
    ) -> PyResult<Vec<u64>> {
        let resolved: HashMap<u32, NodeId> = binding
            .into_iter()
            .map(|(k, v)| (k, node_from_u64(v)))
            .collect();
        let core_labels: HashMap<NodeId, String> = labels
            .unwrap_or_default()
            .into_iter()
            .map(|(h, l)| (node_from_u64(h), l))
            .collect();
        self.inner
            .apply(mol.core_mut(), &resolved, &core_labels, refresh)
            .map(|touched| touched.into_iter().map(node_to_u64).collect())
            .map_err(molrs_error_to_pyerr)
    }

    /// Compile every binding against the intact graph, then apply the disjoint
    /// transforms as one batch. Leaving groups are deleted with one relation
    /// scan, and one touched-handle list is returned per binding.
    #[pyo3(signature = (mol, bindings, labels=None, refresh=true))]
    fn apply_many(
        &self,
        mol: &mut PyAtomistic,
        bindings: Vec<HashMap<u32, u64>>,
        labels: Option<HashMap<u64, String>>,
        refresh: bool,
    ) -> PyResult<Vec<Vec<u64>>> {
        let resolved: Vec<HashMap<u32, NodeId>> = bindings
            .into_iter()
            .map(|binding| {
                binding
                    .into_iter()
                    .map(|(k, v)| (k, node_from_u64(v)))
                    .collect()
            })
            .collect();
        let core_labels: HashMap<NodeId, String> = labels
            .unwrap_or_default()
            .into_iter()
            .map(|(h, l)| (node_from_u64(h), l))
            .collect();
        self.inner
            .apply_many(mol.core_mut(), &resolved, &core_labels, refresh)
            .map(|sets| {
                sets.into_iter()
                    .map(|touched| touched.into_iter().map(node_to_u64).collect())
                    .collect()
            })
            .map_err(molrs_error_to_pyerr)
    }

    /// ``apply_many`` plus RHS-created handles in product creation order.
    ///
    /// The second list is intentionally not reconstructed by sorting handles:
    /// batch deletion may reuse graph slots in an order unrelated to product
    /// atom order.
    #[pyo3(signature = (mol, bindings, labels=None, refresh=true))]
    #[allow(
        clippy::type_complexity,
        reason = "Python returns products and created handles as a tuple"
    )]
    fn apply_many_detailed(
        &self,
        mol: &mut PyAtomistic,
        bindings: Vec<HashMap<u32, u64>>,
        labels: Option<HashMap<u64, String>>,
        refresh: bool,
    ) -> PyResult<(Vec<Vec<u64>>, Vec<Vec<u64>>)> {
        let resolved: Vec<HashMap<u32, NodeId>> = bindings
            .into_iter()
            .map(|binding| {
                binding
                    .into_iter()
                    .map(|(k, v)| (k, node_from_u64(v)))
                    .collect()
            })
            .collect();
        let core_labels: HashMap<NodeId, String> = labels
            .unwrap_or_default()
            .into_iter()
            .map(|(h, l)| (node_from_u64(h), l))
            .collect();
        self.inner
            .apply_many_detailed(mol.core_mut(), &resolved, &core_labels, refresh)
            .map(|(touched_sets, created_sets)| {
                let handles = |sets: Vec<Vec<NodeId>>| {
                    sets.into_iter()
                        .map(|set| set.into_iter().map(node_to_u64).collect())
                        .collect()
                };
                (handles(touched_sets), handles(created_sets))
            })
            .map_err(molrs_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "Reaction(reactants={}, forming_bonds={:?})",
            self.inner.reactants().len(),
            self.inner.forming_bonds()
        )
    }
}

// ---------------------------------------------------------------------------
// PyCoarseGrain — coarse-grained leaf (holds a core CoarseGrain)
// ---------------------------------------------------------------------------

/// Coarse-grained molecular graph, exposed to Python as `molrs.CoarseGrain`.
#[pyclass(module = "molrs._lib", name = "CoarseGrain", extends = PyGraph, subclass)]
pub struct PyCoarseGrain {
    inner: CoarseGrain,
}

impl PyCoarseGrain {
    fn mol(&self) -> &MolGraph {
        self.inner.as_molgraph()
    }
    fn mol_mut(&mut self) -> &mut MolGraph {
        self.inner.as_molgraph_mut()
    }
}

#[pymethods]
impl PyCoarseGrain {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyAny>, _kwargs: Option<&Bound<'_, PyAny>>) -> (Self, PyGraph) {
        (
            PyCoarseGrain {
                inner: CoarseGrain::new(),
            },
            PyGraph {
                inner: MolGraph::new(),
            },
        )
    }

    /// Add a bead with `bead_type` and optional coordinates. Returns its handle.
    #[pyo3(signature = (bead_type, x=None, y=None, z=None))]
    fn add_bead(&mut self, bead_type: &str, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> u64 {
        let id = match (x, y, z) {
            (Some(x), Some(y), Some(z)) => self.inner.add_bead(bead_type, x, y, z),
            _ => self.inner.add_bead_bare(bead_type),
        };
        node_to_u64(id)
    }

    /// Add a CG bond between two bead handles. Returns its handle.
    fn add_bond(&mut self, a: u64, b: u64) -> PyResult<u64> {
        self.inner
            .add_bond(node_from_u64(a), node_from_u64(b))
            .map(relation_to_u64)
            .map_err(molrs_error_to_pyerr)
    }

    /// Number of beads.
    #[getter]
    fn n_beads(&self) -> usize {
        self.inner.n_beads()
    }

    /// Record the atom handles a bead groups (its membership), replacing any
    /// previous set. An empty list clears the membership. Handles are opaque —
    /// they belong to the caller's source (all-atom) world.
    fn set_bead_members(&mut self, bead: u64, atoms: Vec<u64>) {
        self.inner.set_bead_members(node_from_u64(bead), atoms);
    }

    /// The atom handles a bead groups (empty if none recorded).
    fn bead_members(&self, bead: u64) -> Vec<u64> {
        self.inner.bead_members(node_from_u64(bead)).to_vec()
    }

    /// Bead handles whose membership includes `atom`, in bead-handle order.
    fn beads_of_atom(&self, atom: u64) -> Vec<u64> {
        self.inner
            .beads_of_atom(atom)
            .into_iter()
            .map(node_to_u64)
            .collect()
    }

    /// Export to a tabular :class:`~molrs.Frame` in the shared vocabulary.
    ///
    /// One ``atoms`` row per bead and one ``bonds`` row per CG bond
    /// (``atomi`` / ``atomj`` are the endpoint beads' ``atoms`` rows), plus a
    /// ``members`` block when any bead records membership: one row per
    /// bead–atom pair, ``ibead`` being the bead's row in ``atoms`` and
    /// ``atom`` the opaque atom handle. Coordinates stay in Å.
    ///
    /// Every bead property is written, ``mol_id`` included, so a CoarseGrain
    /// whose beads carry two ``mol_id`` values does not round-trip through
    /// :meth:`from_frame`; select one molecule with ``Frame.subset`` first.
    ///
    /// Returns
    /// -------
    /// Frame
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a bead or bond property contradicts the dtype the Frame schema
    ///     declares for its key (a string under ``"x"``).
    fn to_frame(&self) -> PyResult<PyFrame> {
        PyFrame::from_core_frame(self.inner.to_frame().map_err(molrs_error_to_pyerr)?)
    }

    /// Build a `CoarseGrain` from a frame holding **one molecule**.
    ///
    /// Every ``atoms`` row becomes a bead and every ``bonds`` row a CG bond
    /// (``atomi`` / ``atomj`` are 0-based ``atoms`` rows); a ``members``
    /// block, when present, is read back as bead membership. ``bead_type``
    /// comes from the first column present: ``bead_type``, then ``type``,
    /// then ``type_id`` (rendered in decimal). Other blocks are ignored.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     One molecule in the ``atoms`` / ``bonds`` vocabulary. Select one
    ///     molecule of a many-molecule frame with ``Frame.subset`` first.
    ///
    /// Returns
    /// -------
    /// CoarseGrain
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame has no ``atoms`` rows; if ``mol_id`` holds a null row,
    ///     is not an unsigned-int column, or holds more than one distinct
    ///     value (the message points at ``Frame.subset``); if none of
    ///     ``bead_type`` / ``type`` / ``type_id`` is present, or the chosen one
    ///     has a null row or the wrong dtype (string for ``bead_type`` /
    ///     ``type``, unsigned int for ``type_id``); if a bond row names no bead
    ///     row; or if the ``members`` block lacks an unsigned-int ``ibead`` or
    ///     ``atom`` column, names no bead row, or repeats an
    ///     ``(ibead, atom)`` pair.
    #[staticmethod]
    fn from_frame(py: Python<'_>, frame: &PyFrame) -> PyResult<Py<PyCoarseGrain>> {
        let core = frame.clone_core_frame()?;
        let inner = CoarseGrain::from_frame(&core).map_err(molrs_error_to_pyerr)?;
        PyCoarseGrain::from_core(py, inner)
    }

    // ---- structural graph hash (WL) ----

    /// Isomorphism-invariant Weisfeiler–Lehman structural hash (``int``) of the
    /// bead graph (bead-type node labels, bond-order edge labels). Shares the
    /// same [`MolGraph`] primitive that serves the all-atom case.
    fn structural_hash(&self) -> u64 {
        self.inner.structural_hash()
    }

    /// Deterministic canonical bead ordering (a list of bead handles) from the
    /// WL refinement.
    fn canonical_order(&self) -> Vec<u64> {
        self.inner
            .canonical_order()
            .into_iter()
            .map(node_to_u64)
            .collect()
    }

    /// Whether `self` and `other` are isomorphic as labeled bead graphs.
    fn is_isomorphic(&self, other: &PyCoarseGrain) -> bool {
        self.inner.is_isomorphic(&other.inner)
    }

    /// Independent deep copy. **Handles are preserved**.
    fn copy(&self, py: Python<'_>) -> PyResult<Py<PyCoarseGrain>> {
        PyCoarseGrain::from_core(py, self.inner.clone())
    }

    /// Structural merge of ``other`` into ``self``; ``other`` is emptied.
    /// Returns ``{old_handle: new_handle}``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a property of ``other`` contradicts the type this graph holds
    ///     for that key (a string ``tag`` into an int ``tag`` column).
    fn merge(&mut self, other: &mut Self) -> PyResult<HashMap<u64, u64>> {
        let taken = std::mem::take(&mut other.inner);
        Ok(self
            .inner
            .merge(taken)
            .map_err(molrs_error_to_pyerr)?
            .into_iter()
            .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
            .collect())
    }

    /// Induced subgraph on bead handles. Returns ``(subgraph, node_map)``.
    fn induced_subgraph(
        &self,
        py: Python<'_>,
        nodes: Vec<u64>,
    ) -> PyResult<(Py<PyCoarseGrain>, HashMap<u64, u64>)> {
        let ids: Vec<_> = nodes.into_iter().map(node_from_u64).collect();
        let (sub, map) = self
            .inner
            .induced_subgraph(&ids)
            .map_err(molrs_error_to_pyerr)?;
        let py_sub = PyCoarseGrain::from_core(py, sub)?;
        let py_map = map
            .into_iter()
            .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
            .collect();
        Ok((py_sub, py_map))
    }

    /// Radius ball around bead ``centers`` over CG bonds.
    #[pyo3(signature = (centers, radius))]
    fn extract_subgraph(
        &self,
        py: Python<'_>,
        centers: Vec<u64>,
        radius: i64,
    ) -> PyResult<PyExtractedSubgraph> {
        let ids: Vec<_> = centers.into_iter().map(node_from_u64).collect();
        let ext = self
            .inner
            .extract_subgraph(&ids, radius)
            .map_err(molrs_error_to_pyerr)?;
        PyExtractedSubgraph::from_coarsegrain(py, ext)
    }

    /// Mass-weighted centre of the bead group ``group``.
    ///
    /// ``R = sum_i m_i r_i / sum_i m_i`` over the listed beads, reading
    /// positions from ``x`` / ``y`` / ``z`` (Å) and masses from ``mass``
    /// (g/mol). A bead listed twice counts twice. :meth:`add_bead` writes no
    /// ``mass``, so give each bead one (or read it from a frame's ``mass``
    /// column with :meth:`from_frame`) before asking for a centre.
    ///
    /// No periodic imaging: the group's coordinates are used as stored. A
    /// group that straddles a box face must be unwrapped first
    /// (:meth:`Box.unwrap`), otherwise the centre lands between the images.
    ///
    /// Parameters
    /// ----------
    /// group : Sequence[int]
    ///     Bead handles, e.g. one group from ``SubgraphMatcher.find``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,), float64
    ///     The centre in Å.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``group`` is empty; if a handle is not a live bead of this graph,
    ///     or a bead lacks a finite ``x`` / ``y`` / ``z`` or a finite,
    ///     non-negative ``mass`` (the message names its int handle); or if the
    ///     total mass is not positive.
    /// OverflowError
    ///     If a handle in ``group`` is negative (handles are unsigned ints).
    fn center<'py>(&self, py: Python<'py>, group: Vec<u64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let group: Vec<NodeId> = group.into_iter().map(node_from_u64).collect();
        let center = self.inner.center(&group).map_err(center_error_to_pyerr)?;
        Ok(vector_to_py(py, &center))
    }

    /// The positions of ``beads``, one row per listed bead, in the listed
    /// order (a bead listed twice appears twice).
    ///
    /// Parameters
    /// ----------
    /// beads : Sequence[int]
    ///     Bead handles.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (k, 3), float64
    ///     ``x`` / ``y`` / ``z`` as stored (Å).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a handle is not a live bead of this graph, or a bead lacks a
    ///     finite ``x`` / ``y`` / ``z``; the message names its int handle.
    fn positions<'py>(
        &self,
        py: Python<'py>,
        beads: Vec<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let beads: Vec<NodeId> = beads.into_iter().map(node_from_u64).collect();
        let points = self.inner.positions(&beads).map_err(molrs_error_to_pyerr)?;
        Ok(
            Array2::from_shape_fn((points.len(), 3), |(row, axis)| points[row][axis])
                .into_pyarray(py),
        )
    }

    /// The site axes of ``beads``, one row per listed bead, in the listed
    /// order. A site from ``Coarsener.coarsen`` carries the vector from the
    /// first bead of its group to the site; a one-bead site's axis is zero.
    ///
    /// Parameters
    /// ----------
    /// beads : Sequence[int]
    ///     Bead handles.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (k, 3), float64
    ///     ``axis_x`` / ``axis_y`` / ``axis_z`` as stored (Å).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a handle is not a live bead of this graph, or a bead lacks a
    ///     finite axis; the message names its int handle.
    fn axes<'py>(&self, py: Python<'py>, beads: Vec<u64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let beads: Vec<NodeId> = beads.into_iter().map(node_from_u64).collect();
        let axes = self.inner.axes(&beads).map_err(molrs_error_to_pyerr)?;
        Ok(Array2::from_shape_fn((axes.len(), 3), |(row, c)| axes[row][c]).into_pyarray(py))
    }

    /// The ``bead_type`` of each of ``beads``, in the listed order (a bead
    /// listed twice appears twice).
    ///
    /// Parameters
    /// ----------
    /// beads : Sequence[int]
    ///     Bead handles.
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     One type per listed bead.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a handle is not a live bead of this graph, or a bead carries no
    ///     ``bead_type``; the message names its int handle.
    fn bead_types(&self, beads: Vec<u64>) -> PyResult<Vec<String>> {
        let beads: Vec<NodeId> = beads.into_iter().map(node_from_u64).collect();
        self.inner.bead_types(&beads).map_err(molrs_error_to_pyerr)
    }
}
graph_world_impl!(PyCoarseGrain);

impl PyCoarseGrain {
    /// Wrap an existing core [`CoarseGrain`] as a Python `CoarseGrain` object.
    pub(crate) fn from_core(py: Python<'_>, inner: CoarseGrain) -> PyResult<Py<PyCoarseGrain>> {
        from_core_shadowed(py, PyCoarseGrain { inner })
    }

    /// Borrow the held core [`CoarseGrain`] (for the bead-pattern matcher,
    /// which reads the pattern and the target graph).
    pub(crate) fn core(&self) -> &CoarseGrain {
        &self.inner
    }
}

/// The shared argument seam of the two leaf ``replicate`` methods:
/// ``rotations (N,3,3)`` + ``translations (N,3)`` as rigid motions, and
/// ``frag_ids`` as ``i32`` (an ``int32`` array read directly, any other
/// integer sequence element by element, out-of-range values refused).
fn replicate_args(
    rotations: &PyReadonlyArrayDyn<'_, f64>,
    translations: &PyReadonlyArrayDyn<'_, f64>,
    frag_ids: &Bound<'_, PyAny>,
) -> PyResult<(Vec<molrs::op::rigid::Rigid>, Vec<i32>)> {
    let transforms = crate::op::rigids_from_arrays(rotations, translations)?;
    let frag_ids = match frag_ids.extract::<PyReadonlyArrayDyn<'_, i32>>() {
        Ok(array) => {
            let view = array.as_array();
            if view.ndim() != 1 {
                return Err(PyValueError::new_err(format!(
                    "frag_ids must have shape (N,), got {:?}",
                    view.shape()
                )));
            }
            view.iter().copied().collect()
        }
        Err(_) => frag_ids.extract::<Vec<i32>>()?,
    };
    Ok((transforms, frag_ids))
}

// ---------------------------------------------------------------------------
// Rigid replication
// ---------------------------------------------------------------------------

/// The leaf ``replicate`` method: `$leaf` is the Python class name, `$node`
/// what its nodes are called, `$kept` what else a copy carries.
macro_rules! replicate_impl {
    ($ty:ty, $leaf:literal, $node:literal, $kept:literal) => {
        #[pymethods]
        impl $ty {
            /// Grow this graph by one rigid copy of ``template`` per transform.
            ///
            /// Copy ``c`` is ``template`` moved by ``rotations[c] @ r + translations[c]``
            /// and every node of it is stamped ``frag_id = frag_ids[c]``. Every
            #[doc = concat!("relation kind of ``template`` is copied with it", $kept, ". Column-wise and")]
            /// atomic: on an error this graph is unchanged. ``template`` is never
            /// mutated. The GIL is released while the copies are written.
            ///
            /// Parameters
            /// ----------
            #[doc = concat!("template : ", $leaf)]
            ///     Another graph: this graph cannot be its own template.
            /// rotations : ndarray, shape (N, 3, 3), float64
            /// translations : ndarray, shape (N, 3), float64
            /// frag_ids : ndarray, shape (N,), int32
            ///
            /// Returns
            /// -------
            /// list[int]
            #[doc = concat!("    The new ", $node, " handles, copy-major.")]
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If ``template`` is this graph, on a wrong shape, disagreeing
            ///     counts, or a column of ``template`` whose type contradicts
            ///     the one this graph holds.
            fn replicate(
                slf: &Bound<'_, Self>,
                template: &Bound<'_, Self>,
                rotations: PyReadonlyArrayDyn<'_, f64>,
                translations: PyReadonlyArrayDyn<'_, f64>,
                frag_ids: &Bound<'_, PyAny>,
            ) -> PyResult<Vec<u64>> {
                // Checked before either borrow: borrowing one object both
                // ways would be PyO3's borrow error, not this refusal.
                if slf.is(template) {
                    return Err(PyValueError::new_err(
                        "a graph cannot replicate itself; pass a copy as the template",
                    ));
                }
                let (transforms, frag_ids) = replicate_args(&rotations, &translations, frag_ids)?;
                let template = template.try_borrow()?;
                let mut this = slf.try_borrow_mut()?;
                let (target, source) = (&mut this.inner, &template.inner);
                let added = slf
                    .py()
                    .detach(|| target.replicate(source, &transforms, &frag_ids))
                    .map_err(molrs_error_to_pyerr)?;
                Ok(added.into_iter().map(node_to_u64).collect())
            }
        }
    };
}

replicate_impl!(PyAtomistic, "Atomistic", "atom", "");
replicate_impl!(PyCoarseGrain, "CoarseGrain", "bead", "");

// ---------------------------------------------------------------------------
// Rigid-body moves
// ---------------------------------------------------------------------------
//
// `translate` / `rotate` / `scale` are leaf methods: each resolves to the
// leaf's *own* graph (never the empty base it carries for `issubclass`).

macro_rules! rigid_body_impl {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            /// Translate every node that has coordinates by `delta`. Returns
            /// this graph, so moves chain.
            fn translate(mut slf: PyRefMut<'_, Self>, delta: [f64; 3]) -> PyRefMut<'_, Self> {
                molrs::spatial::geometry::translate(slf.mol_mut(), delta);
                slf
            }

            /// Rotate every node that has coordinates by `angle` radians about
            /// `axis`, pivoting on `about` (default: the origin). Returns this
            /// graph, so moves chain.
            #[pyo3(signature = (axis, angle, about=None))]
            fn rotate(
                mut slf: PyRefMut<'_, Self>,
                axis: [f64; 3],
                angle: f64,
                about: Option<[f64; 3]>,
            ) -> PyResult<PyRefMut<'_, Self>> {
                molrs::spatial::geometry::rotate(slf.mol_mut(), axis, angle, about)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                Ok(slf)
            }

            /// Scale every node that has coordinates by a per-axis `factor`
            /// about `about` (default: the origin). Pass `[s, s, s]` for a
            /// uniform scale. Returns this graph, so moves chain.
            #[pyo3(signature = (factor, about=None))]
            fn scale(
                mut slf: PyRefMut<'_, Self>,
                factor: [f64; 3],
                about: Option<[f64; 3]>,
            ) -> PyRefMut<'_, Self> {
                molrs::spatial::geometry::scale(slf.mol_mut(), factor, about);
                slf
            }
        }
    };
}

rigid_body_impl!(PyAtomistic);
rigid_body_impl!(PyCoarseGrain);

/// The ring facts of a molecule: SSSR rings and the systems they fuse into.
///
/// Perception runs once, in the constructor; every method reads the result.
///
/// Not to be confused with :meth:`Perceive.find_rings`, which answers a
/// different question — it *decorates* a graph with ring flags and hands the
/// graph back. This type *reports*, and never touches the molecule.
///
/// Examples
/// --------
/// >>> rings = molrs.perceive.RingInfo(molrs.io.SmilesIR("c1ccccc1").to_atomistic())
/// >>> rings.num_rings()
/// 1
/// >>> rings.ring_sizes()
/// [6]
#[pyclass(module = "molrs.perceive", name = "RingInfo")]
pub struct PyRingInfo {
    inner: molrs::perceive::rings::RingInfo,
}

#[pymethods]
impl PyRingInfo {
    /// Perceive the rings of `mol` (SSSR / minimum cycle basis).
    #[new]
    fn new(mol: &Bound<'_, PyAtomistic>) -> Self {
        Self {
            inner: molrs::perceive::rings::find_rings(mol.borrow().core()),
        }
    }

    /// Every ring, as a list of atom handles forming a closed path.
    fn rings(&self) -> Vec<Vec<u64>> {
        self.inner
            .rings()
            .iter()
            .map(|ring| ring.iter().map(|&a| node_to_u64(a)).collect())
            .collect()
    }

    /// Number of rings.
    fn num_rings(&self) -> usize {
        self.inner.num_rings()
    }

    /// Atom count of every ring, ascending.
    fn ring_sizes(&self) -> Vec<usize> {
        self.inner.ring_sizes()
    }

    /// Rings that share at least one atom, unioned: benzene → one system of 6,
    /// naphthalene → one of 10, biphenyl → two of 6.
    fn ring_systems(&self) -> Vec<Vec<u64>> {
        self.inner
            .ring_systems()
            .iter()
            .map(|system| system.iter().map(|&a| node_to_u64(a)).collect())
            .collect()
    }

    /// Whether `atom` belongs to any ring.
    fn is_atom_in_ring(&self, atom: u64) -> bool {
        self.inner.is_atom_in_ring(node_from_u64(atom))
    }

    /// Number of rings containing `atom`.
    fn num_atom_rings(&self, atom: u64) -> usize {
        self.inner.num_atom_rings(node_from_u64(atom))
    }

    /// Size of the smallest ring containing `atom`, or ``None``.
    fn smallest_ring_containing_atom(&self, atom: u64) -> Option<usize> {
        self.inner
            .smallest_ring_containing_atom(node_from_u64(atom))
    }

    fn __repr__(&self) -> String {
        format!(
            "RingInfo(num_rings={}, sizes={:?})",
            self.inner.num_rings(),
            self.inner.ring_sizes()
        )
    }
}
