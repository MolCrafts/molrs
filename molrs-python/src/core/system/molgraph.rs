//! Python bindings for the ECS molecular graph.
//!
//! The core is an ECS *world*: entities are stable opaque handles, their data
//! lives in aligned component columns, and topology is kind-tagged relations.
//! This module exposes that faithfully:
//!
//! - [`PyGraph`] (`molrs.Graph`) — the domain-agnostic world: stable-handle
//!   entities, by-name component get/set, and the kind-tagged relation API.
//! - [`PyAtomistic`] (`molrs.Atomistic`) / [`PyCoarseGrain`] (`molrs.CoarseGrain`)
//!   — leaves that **hold a core [`Atomistic`] / [`CoarseGrain`] from
//!   construction** (never converted from a `MolGraph`). They add the
//!   domain builders (`add_atom`/`add_bond`/…) and own `to_frame` /
//!   `from_frame` (`self.inner.to_frame()`, zero conversion). They subclass
//!   `Graph` in Python; the generic graph API is shared via the
//!   `graph_world_body!` macro, which always operates on the receiver's *own*
//!   graph (`self.mol()` / `self.mol_mut()`), so the leaf's graph is the single
//!   data slot.
//!
//! Handles are stable opaque `int`s (generational slotmap keys); removing one
//! entity never invalidates another, and a stale handle raises.

use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use molrs::chem::aromaticity::perceive_aromaticity as core_perceive_aromaticity;
use molrs::chem::smarts::{Reaction, SmartsPattern};
use molrs::system::atomistic::Atomistic;
use molrs::system::coarsegrain::CoarseGrain;
use molrs::system::entity_table::Cell;
use molrs::system::molgraph::{
    KindId, MolGraph, NodeId, PropValue, node_from_u64, node_to_u64, relation_from_u64,
    relation_to_u64,
};

use crate::core::store::frame::PyFrame;
use crate::helpers::molrs_error_to_pyerr;

// ---------------------------------------------------------------------------
// Value conversion helpers
// ---------------------------------------------------------------------------

/// Convert a Python scalar to a [`PropValue`].
///
/// `bool` is tried before `int` because a Python `bool` is a subclass of `int`
/// (so `extract::<i64>()` would silently collapse `True`→`1`); `int` is tried
/// before `float` so an integer literal doesn't become a float. Anything that is
/// not `bool` / `int` / `float` / `str` is rejected fail-fast — non-representable
/// values (lists, `None`, arbitrary objects) MUST raise, never be stashed.
fn py_to_prop(value: &Bound<'_, PyAny>) -> PyResult<PropValue> {
    // `extract::<bool>()` matches only a genuine Python `bool`, not an `int`.
    if let Ok(b) = value.extract::<bool>() {
        Ok(PropValue::Bool(b))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(PropValue::Int(i as i32))
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
            fn get(&self, py: Python<'_>, h: u64, key: &str) -> PyResult<Py<PyAny>> {
                match self.mol().node_table().value(node_from_u64(h), key) {
                    Some(cell) => cell_to_py(py, cell),
                    None => Ok(py.None()),
                }
            }

            /// Set entity `h`'s component `key` (``value`` is int|float|str).
            fn set(&mut self, h: u64, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
                let pv = py_to_prop(value)?;
                self.mol_mut()
                    .set_node(node_from_u64(h), key, pv)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Whether entity `h` has component `key`.
            fn has(&self, h: u64, key: &str) -> bool {
                self.mol().node_table().has(node_from_u64(h), key)
            }

            /// Clear entity `h`'s component `key` (no-op if absent).
            fn delete(&mut self, h: u64, key: &str) -> PyResult<()> {
                self.mol_mut()
                    .clear_node(node_from_u64(h), key)
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

            /// Zero-copy numpy view of the `f64` component column `key`, aligned to
            /// row order (length == `n_nodes`). Writes through to the world:
            /// `col[i] = v` updates the entity at row `i`.
            ///
            /// The view borrows the world's storage; structural mutation
            /// (`spawn`/`despawn`) may reallocate or reorder the column and
            /// invalidate an outstanding view — re-fetch after such ops.
            fn column<'py>(
                slf: Bound<'py, $ty>,
                key: &str,
            ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
                let (ptr, len) = {
                    let this = slf.borrow();
                    let (data, _valid) = this
                        .mol()
                        .node_table()
                        .column_f64(key)
                        .map_err(molrs_error_to_pyerr)?;
                    (data.as_ptr(), data.len())
                };
                // SAFETY: `slf` owns the backing Vec and is held as the array's base
                // object, so the memory stays valid for the array's lifetime; the
                // documented contract forbids structural mutation while held.
                let view = unsafe { numpy::ndarray::ArrayView1::from_shape_ptr(len, ptr) };
                Ok(unsafe { numpy::PyArray1::borrow_from_array(&view, slf.into_any()) })
            }

            /// Validity mask (numpy `bool` array, copied) of component column `key`,
            /// aligned to row order. `True` where the entity at that row has the
            /// component set.
            fn validity<'py>(
                &self,
                py: Python<'py>,
                key: &str,
            ) -> PyResult<Bound<'py, numpy::PyArray1<bool>>> {
                let valid =
                    self.mol().node_table().col_validity(key).ok_or_else(|| {
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
// PyGraph — the generic world
// ---------------------------------------------------------------------------

/// Domain-agnostic ECS world, exposed to Python as `molrs.Graph`.
#[pyclass(name = "Graph", subclass)]
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

// ---------------------------------------------------------------------------
// PyAtomistic — all-atom leaf (holds a core Atomistic)
// ---------------------------------------------------------------------------

/// All-atom molecular graph, exposed to Python as `molrs.Atomistic`.
///
/// Holds a core [`Atomistic`] from construction; it is never converted from a
/// `MolGraph`. Subclasses `Graph`; the generic API operates on this leaf's own
/// graph.
#[pyclass(name = "Atomistic", extends = PyGraph, subclass)]
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

    /// Perceive angle and dihedral relations from the bond graph.
    ///
    /// Angles are 2-edge paths ``i-j-k`` and proper dihedrals 3-edge paths
    /// ``i-j-k-l`` over the bonds (graph-theory via the petgraph-backed
    /// ``Topology``). Idempotent; ``clear_existing`` wipes existing
    /// angle/dihedral relations first. Returns ``(n_angles_added,
    /// n_dihedrals_added)``.
    #[pyo3(signature = (gen_angle=true, gen_dihedral=true, clear_existing=false))]
    fn generate_topology(
        &mut self,
        gen_angle: bool,
        gen_dihedral: bool,
        clear_existing: bool,
    ) -> PyResult<(usize, usize)> {
        self.inner
            .generate_topology(gen_angle, gen_dihedral, clear_existing)
            .map_err(molrs_error_to_pyerr)
    }

    /// Single-source shortest-path (BFS) distances over the bond graph from
    /// `source` (a node handle), as `(node_handle, hops)` pairs for every atom
    /// reachable from `source` (including `source` itself at distance 0).
    /// Unreachable atoms (a different connected component) are omitted; an
    /// unknown `source` handle yields an empty list.
    fn topo_distances(&self, source: u64) -> Vec<(u64, i64)> {
        self.inner
            .topo_distances(node_from_u64(source))
            .into_iter()
            .map(|(a, d)| (node_to_u64(a), d))
            .collect()
    }

    /// Number of atoms.
    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Export to a tabular [`Frame`] (atoms / bonds / angles / dihedrals /
    /// impropers blocks). Leaf-owned — `self.inner.to_frame()`, zero conversion.
    fn to_frame(&self) -> PyResult<PyFrame> {
        PyFrame::from_core_frame(self.inner.to_frame())
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

    /// Set a bond's ``order`` property (a relation handle from :meth:`add_bond`).
    fn set_bond_order(&mut self, handle: u64, order: f64) -> PyResult<()> {
        self.inner
            .set_bond_prop(relation_from_u64(handle), "order", order)
            .map_err(molrs_error_to_pyerr)
    }

    /// Return an independent deep copy of this `Atomistic` (handles preserved).
    fn copy(&self, py: Python<'_>) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.clone())
    }
}
graph_world_impl!(PyAtomistic);

impl PyAtomistic {
    /// Wrap an existing core [`Atomistic`] as a Python `Atomistic` object.
    pub(crate) fn from_core(py: Python<'_>, inner: Atomistic) -> PyResult<Py<PyAtomistic>> {
        Py::new(
            py,
            (
                PyAtomistic { inner },
                PyGraph {
                    inner: MolGraph::new(),
                },
            ),
        )
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
// PySmartsPattern — atom-map-aware SMARTS matcher over an Atomistic
// ---------------------------------------------------------------------------

/// Compiled SMARTS query, exposed to Python as `molrs.SmartsPattern`.
///
/// A thin wrapper over the core [`SmartsPattern`] (`molrs/src/core/chem/smarts`)
/// — the same backtracking subgraph-isomorphism engine that drives the OPLS-AA
/// typifier. Matching is non-uniquified (RDKit ``uniquify=False``): every
/// distinct query-atom → mol-atom embedding is reported.
///
/// Daylight atom maps (``[C:1]``) are parsed and carried through but add **no**
/// match constraint (they are "ignored in molecule SMARTS"); use
/// :meth:`find_matches_mapped` to read them back as ``{map_number: atom_handle}``.
///
/// Examples
/// --------
/// >>> pat = molrs.SmartsPattern("[C:1][O:2][H:3]")
/// >>> pat.find_matches_mapped(methanol)   # [{1: <C>, 2: <O>, 3: <H>}]
#[pyclass(name = "SmartsPattern")]
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
    fn has_match(&self, mol: &PyAtomistic) -> bool {
        self.inner.has_match(mol.core())
    }

    /// All matches, each as a list of atom handles in query-atom order.
    fn find_matches(&self, mol: &PyAtomistic) -> Vec<Vec<u64>> {
        self.inner
            .find_matches(mol.core())
            .into_iter()
            .map(|m| m.into_iter().map(node_to_u64).collect())
            .collect()
    }

    /// All matches, each as a ``{atom_map_number: atom_handle}`` dict. Only
    /// query atoms carrying a ``:n`` map label appear; unlabelled query atoms
    /// still constrain the match but are omitted from the dict.
    fn find_matches_mapped(&self, mol: &PyAtomistic) -> Vec<HashMap<u32, u64>> {
        self.inner
            .find_matches(mol.core())
            .into_iter()
            .map(|m| {
                m.iter()
                    .enumerate()
                    .filter_map(|(i, &atom)| {
                        self.inner.map_label(i).map(|n| (n, node_to_u64(atom)))
                    })
                    .collect()
            })
            .collect()
    }

    /// Number of query atoms in the pattern.
    #[getter]
    fn num_query_atoms(&self) -> usize {
        self.inner.num_query_atoms()
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
/// ...     binding.update(pat.find_matches_mapped(mol)[0])
/// >>> rxn.apply(mol, binding)            # edits `mol` in place
#[pyclass(name = "Reaction")]
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
    fn apply(&self, mol: &mut PyAtomistic, binding: HashMap<u32, u64>) -> PyResult<()> {
        let resolved: HashMap<u32, NodeId> = binding
            .into_iter()
            .map(|(k, v)| (k, node_from_u64(v)))
            .collect();
        self.inner
            .apply(mol.core_mut(), &resolved)
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
#[pyclass(name = "CoarseGrain", extends = PyGraph, subclass)]
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

    /// Export to a tabular [`Frame`] (beads + bonds blocks).
    fn to_frame(&self) -> PyResult<PyFrame> {
        PyFrame::from_core_frame(self.inner.to_frame())
    }

    /// Build a `CoarseGrain` from a [`Frame`] (registers the CG bonds kind).
    #[staticmethod]
    fn from_frame(py: Python<'_>, frame: &PyFrame) -> PyResult<Py<PyCoarseGrain>> {
        let core = frame.clone_core_frame()?;
        let inner = CoarseGrain::from_frame(&core).map_err(molrs_error_to_pyerr)?;
        PyCoarseGrain::from_core(py, inner)
    }
}
graph_world_impl!(PyCoarseGrain);

impl PyCoarseGrain {
    /// Wrap an existing core [`CoarseGrain`] as a Python `CoarseGrain` object.
    pub(crate) fn from_core(py: Python<'_>, inner: CoarseGrain) -> PyResult<Py<PyCoarseGrain>> {
        Py::new(
            py,
            (
                PyCoarseGrain { inner },
                PyGraph {
                    inner: MolGraph::new(),
                },
            ),
        )
    }
}

// ---------------------------------------------------------------------------
// Systems = module-level free functions
// ---------------------------------------------------------------------------
//
// Algorithms are NOT methods on the graph classes; they are module functions
// that take a world. Generic geometry systems accept any of the three types and
// dispatch leaf-first so a leaf resolves to its *own* graph (never the empty
// base it carries for `issubclass`). Chemistry systems require an `Atomistic`,
// so they take `PyAtomistic` directly.

/// Resolve a Python graph object to its own `MolGraph` and run `f` on it.
/// Leaf-first so a `PyAtomistic`/`PyCoarseGrain` uses its core graph, not the
/// empty `PyGraph` base it carries for subclassing.
fn with_world_mut(mol: &Bound<'_, PyAny>, f: impl FnOnce(&mut MolGraph)) -> PyResult<()> {
    if let Ok(leaf) = mol.cast::<PyAtomistic>() {
        f(leaf.borrow_mut().mol_mut());
    } else if let Ok(leaf) = mol.cast::<PyCoarseGrain>() {
        f(leaf.borrow_mut().mol_mut());
    } else if let Ok(g) = mol.cast::<PyGraph>() {
        f(g.borrow_mut().mol_mut());
    } else {
        return Err(PyTypeError::new_err(
            "expected a Graph / Atomistic / CoarseGrain",
        ));
    }
    Ok(())
}

/// Translate every node's coordinates by `delta` (generic geometry system).
#[pyfunction]
pub fn translate(mol: &Bound<'_, PyAny>, delta: [f64; 3]) -> PyResult<()> {
    with_world_mut(mol, |g| molrs::spatial::geometry::translate(g, delta))
}

/// Rotate node coordinates by `angle` radians about `axis` (optionally about a
/// point — defaults to the origin). Generic geometry system.
#[pyfunction]
#[pyo3(signature = (mol, axis, angle, about=None))]
pub fn rotate(
    mol: &Bound<'_, PyAny>,
    axis: [f64; 3],
    angle: f64,
    about: Option<[f64; 3]>,
) -> PyResult<()> {
    with_world_mut(mol, |g| {
        molrs::spatial::geometry::rotate(g, axis, angle, about)
    })
}

/// Scale node coordinates by a per-axis `factor` about an optional center
/// (defaults to the origin). Pass `[s, s, s]` for a uniform scale. Generic
/// geometry system.
#[pyfunction]
#[pyo3(signature = (mol, factor, about=None))]
pub fn scale(mol: &Bound<'_, PyAny>, factor: [f64; 3], about: Option<[f64; 3]>) -> PyResult<()> {
    with_world_mut(mol, |g| molrs::spatial::geometry::scale(g, factor, about))
}

/// Perceive aromaticity in place; returns the number of aromatic atoms found.
/// A chemistry system — operates on an `Atomistic` leaf.
#[pyfunction]
pub fn perceive_aromaticity(mol: &Bound<'_, PyAtomistic>) -> usize {
    core_perceive_aromaticity(mol.borrow_mut().core_mut())
}

/// Add explicit hydrogens, returning a **new** `Atomistic` (chemistry system).
#[pyfunction]
pub fn add_hydrogens(py: Python<'_>, mol: &Bound<'_, PyAtomistic>) -> PyResult<Py<PyAtomistic>> {
    let out = molrs::chem::hydrogens::add_hydrogens(mol.borrow().core());
    PyAtomistic::from_core(py, out)
}

/// Find all SSSR rings; returns each ring as a list of atom handles. A
/// chemistry system — operates on an `Atomistic` leaf.
#[pyfunction]
pub fn find_rings(mol: &Bound<'_, PyAtomistic>) -> Vec<Vec<u64>> {
    let leaf = mol.borrow();
    molrs::chem::rings::find_rings(leaf.core())
        .rings()
        .iter()
        .map(|ring| ring.iter().map(|&a| node_to_u64(a)).collect())
        .collect()
}

/// Compute Gasteiger partial charges; returns `(atom_handle, charge, h_charge)`
/// per heavy atom. A chemistry system — operates on an `Atomistic` leaf.
#[pyfunction]
#[pyo3(signature = (mol, n_iter=6))]
pub fn compute_gasteiger_charges(
    mol: &Bound<'_, PyAtomistic>,
    n_iter: usize,
) -> Vec<(u64, f64, f64)> {
    let leaf = mol.borrow();
    molrs::chem::gasteiger::compute_gasteiger_charges(leaf.core(), n_iter)
        .into_iter()
        .map(|(id, gc)| (node_to_u64(id), gc.charge, gc.h_charge))
        .collect()
}
