//! Live views over the leaf graphs (`molrs.core.Atomistic`, `molrs.core.CoarseGrain`).
//!
//! A view is a `(world, handle)` pair and nothing else: every read and write
//! goes through the owning graph, which stays the only holder of data. The
//! graph interns its views weakly ([`ViewCache`]), so while a view is alive the
//! same handle always answers with the same object (`graph.atoms[0] is atom`).
//!
//! - [`PyNodeRef`] (`molrs.core.NodeRef`) and its classes `Atom`, `VirtualSite`,
//!   `DrudeParticle`, `MasslessSite`, `Bead`: one node. The class follows the
//!   graph type and, for an atom, its stored ``vsite``.
//! - [`PyRelationRef`] (`molrs.core.RelationRef`) and `Bond`, `Angle`, `Dihedral`,
//!   `Improper`, `Port`, `CgBond`: one relation, with its interned endpoints.
//!   The class follows the graph type and the relation kind.
//! - [`PyRefs`] (`molrs.core.Refs`): an ordered handle list of one kind, read as a
//!   sequence of views or, by field name, as a column.
//! - [`PyRelationBuckets`] (`graph.links`): relations selected by view class.

use std::collections::HashMap;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyDict, PyList, PySlice, PyString, PyTuple, PyType, PyWeakrefMethods, PyWeakrefReference,
};
use pyo3::{PyTraverseError, PyVisit, intern};

use molrs::core::EntityCell;
use molrs::core::keys::{BEAD_ATOMS, VSITE};
use molrs::core::{
    KindId, MolGraph, node_from_u64, node_to_u64, relation_from_u64, relation_to_u64,
};

use super::molgraph::{PyAtomistic, PyCoarseGrain, cell_to_py, prop_to_py, py_to_prop};
use crate::core::schema::extract_column_key;
use crate::error::molrs_error_to_pyerr;

// ---------------------------------------------------------------------------
// Interning
// ---------------------------------------------------------------------------

/// A graph's weak index of its live views, keyed by handle.
///
/// Weak, so a view nobody holds is freed; a dead entry is pruned once the
/// table has doubled since the last sweep.
#[derive(Default)]
pub(crate) struct ViewCache {
    nodes: HashMap<u64, Py<PyWeakrefReference>>,
    relations: HashMap<(String, u64), Py<PyWeakrefReference>>,
    watermark: usize,
}

impl ViewCache {
    /// Entries below which no sweep runs.
    const MIN_SWEEP: usize = 1024;

    fn node<'py>(&self, py: Python<'py>, handle: u64) -> Option<Bound<'py, PyAny>> {
        self.nodes.get(&handle)?.bind(py).upgrade()
    }

    fn relation<'py>(&self, py: Python<'py>, kind: &str, handle: u64) -> Option<Bound<'py, PyAny>> {
        self.relations
            .get(&(kind.to_owned(), handle))?
            .bind(py)
            .upgrade()
    }

    fn keep_node(&mut self, py: Python<'_>, handle: u64, view: &Bound<'_, PyAny>) -> PyResult<()> {
        self.sweep(py);
        self.nodes
            .insert(handle, PyWeakrefReference::new(view)?.unbind());
        Ok(())
    }

    fn keep_relation(
        &mut self,
        py: Python<'_>,
        kind: &str,
        handle: u64,
        view: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.sweep(py);
        self.relations.insert(
            (kind.to_owned(), handle),
            PyWeakrefReference::new(view)?.unbind(),
        );
        Ok(())
    }

    fn sweep(&mut self, py: Python<'_>) {
        let entries = self.nodes.len() + self.relations.len();
        if entries < self.watermark.max(Self::MIN_SWEEP) {
            return;
        }
        self.nodes
            .retain(|_, view| view.bind(py).upgrade().is_some());
        self.relations
            .retain(|_, view| view.bind(py).upgrade().is_some());
        self.watermark = 2 * (self.nodes.len() + self.relations.len());
    }

    /// Forget every view: the graph's storage was replaced, so a cached
    /// handle may now name a different node or relation.
    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
        self.relations.clear();
        self.watermark = 0;
    }
}

// ---------------------------------------------------------------------------
// The leaf a view belongs to
// ---------------------------------------------------------------------------

/// A graph that has views: one of the two leaves.
pub(crate) enum Leaf<'py> {
    Atomistic(Bound<'py, PyAtomistic>),
    CoarseGrain(Bound<'py, PyCoarseGrain>),
}

fn kind_id(mol: &MolGraph, kind: &str) -> PyResult<KindId> {
    mol.kind_id(kind)
        .ok_or_else(|| PyValueError::new_err(format!("kind '{kind}' is not registered")))
}

impl<'py> Leaf<'py> {
    /// The leaf `world` is.
    ///
    /// # Errors
    ///
    /// `TypeError` when `world` is neither an `Atomistic` nor a `CoarseGrain`.
    pub(crate) fn of(world: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(graph) = world.cast::<PyAtomistic>() {
            return Ok(Self::Atomistic(graph.clone()));
        }
        if let Ok(graph) = world.cast::<PyCoarseGrain>() {
            return Ok(Self::CoarseGrain(graph.clone()));
        }
        Err(PyTypeError::new_err(format!(
            "views live on an Atomistic or a CoarseGrain, not {}",
            world.get_type().name()?
        )))
    }

    fn py(&self) -> Python<'py> {
        self.object().py()
    }

    fn object(&self) -> &Bound<'py, PyAny> {
        match self {
            Self::Atomistic(graph) => graph.as_any(),
            Self::CoarseGrain(graph) => graph.as_any(),
        }
    }

    pub(crate) fn read<R>(&self, f: impl FnOnce(&MolGraph) -> R) -> PyResult<R> {
        Ok(match self {
            Self::Atomistic(graph) => f(graph.try_borrow()?.mol()),
            Self::CoarseGrain(graph) => f(graph.try_borrow()?.mol()),
        })
    }

    pub(crate) fn write<R>(&self, f: impl FnOnce(&mut MolGraph) -> R) -> PyResult<R> {
        Ok(match self {
            Self::Atomistic(graph) => f(graph.try_borrow_mut()?.mol_mut()),
            Self::CoarseGrain(graph) => f(graph.try_borrow_mut()?.mol_mut()),
        })
    }

    fn cache<R>(&self, f: impl FnOnce(&mut ViewCache) -> R) -> PyResult<R> {
        Ok(match self {
            Self::Atomistic(graph) => f(&mut graph.try_borrow_mut()?.views),
            Self::CoarseGrain(graph) => f(&mut graph.try_borrow_mut()?.views),
        })
    }

    /// The interned view of node `handle`.
    ///
    /// # Errors
    ///
    /// `ValueError` when `handle` is not a live node of this graph.
    pub(crate) fn node(&self, handle: u64) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py();
        if let Some(view) = self.cache(|cache| cache.node(py, handle))? {
            return Ok(view);
        }
        let class = self.read(|mol| NodeClass::of(self, mol, handle))??;
        let base = PyNodeRef {
            world: self.object().clone().unbind(),
            handle,
        };
        let view = class.instantiate(py, base)?.into_bound(py);
        self.cache(|cache| cache.keep_node(py, handle, &view))??;
        Ok(view)
    }

    /// The interned view of relation `handle` of `kind`.
    ///
    /// # Errors
    ///
    /// `ValueError` when `kind` is not registered or `handle` is not a live
    /// relation of it.
    pub(crate) fn relation(&self, kind: &str, handle: u64) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py();
        if let Some(view) = self.cache(|cache| cache.relation(py, kind, handle))? {
            return Ok(view);
        }
        let ends = self.read(|mol| -> PyResult<Vec<u64>> {
            let nodes = mol
                .relation_nodes(kind_id(mol, kind)?, relation_from_u64(handle))
                .map_err(molrs_error_to_pyerr)?;
            Ok(nodes.iter().map(|&n| node_to_u64(n)).collect())
        })??;
        let endpoints = ends
            .into_iter()
            .map(|end| self.node(end))
            .collect::<PyResult<Vec<_>>>()?;
        let base = PyRelationRef {
            world: self.object().clone().unbind(),
            kind: kind.to_owned(),
            handle,
            endpoints: PyTuple::new(py, endpoints)?.unbind(),
        };
        let view = RelationClass::of(self, kind)
            .instantiate(py, base)?
            .into_bound(py);
        self.cache(|cache| cache.keep_relation(py, kind, handle, &view))??;
        Ok(view)
    }

    /// Every node, in row order.
    pub(crate) fn nodes(&self) -> PyResult<PyRefs> {
        let handles = self.read(|mol| mol.node_ids().map(node_to_u64).collect())?;
        Ok(PyRefs {
            world: self.object().clone().unbind(),
            kind: None,
            handles,
        })
    }

    /// Every relation of `kind`, in row order; empty when `kind` is not
    /// registered.
    pub(crate) fn relations(&self, kind: &str) -> PyResult<PyRefs> {
        let handles = self.read(|mol| match mol.kind_id(kind) {
            Some(kid) => mol.relation_ids(kid).map(relation_to_u64).collect(),
            None => Vec::new(),
        })?;
        Ok(PyRefs {
            world: self.object().clone().unbind(),
            kind: Some(kind.to_owned()),
            handles,
        })
    }

    /// Spawn a node carrying `fields` (a ``None`` value is skipped) and
    /// return its view. On a refused field the node is removed again.
    pub(crate) fn create_node(&self, fields: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
        let id = self.write(MolGraph::add_node)?;
        let written = fields.iter().try_for_each(|(key, value)| {
            if value.is_none() {
                return Ok(());
            }
            let key = extract_column_key(&key)?;
            let value = py_to_prop(&value)?;
            self.write(|mol| mol.set_node(id, &key, value))?
                .map_err(molrs_error_to_pyerr)
        });
        if let Err(err) = written {
            self.write(|mol| mol.remove_node(id))?
                .map_err(molrs_error_to_pyerr)?;
            return Err(err);
        }
        self.node(node_to_u64(id))
    }

    /// Spawn a virtual site: a node carrying `fields` plus the ``vsite`` the
    /// class `kind` (``VirtualSite`` when `None`) names, unless `fields`
    /// already sets one.
    pub(crate) fn create_virtual_site(
        &self,
        fields: &Bound<'py, PyDict>,
        kind: Option<&Bound<'py, PyType>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !fields.contains(VSITE)? {
            let vsite = match kind {
                Some(cls) => NodeClass::vsite_of(cls)?,
                None => "virtual",
            };
            fields.set_item(VSITE, vsite)?;
        }
        self.create_node(fields)
    }

    /// Stamp `fields` on relation `handle` of `kind` (just created by a
    /// native writer) and return its view. On a refused field the relation
    /// is removed again.
    pub(crate) fn adopt_relation(
        &self,
        kind: &str,
        handle: u64,
        fields: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(fields) = fields {
            let written = fields.iter().try_for_each(|(key, value)| {
                RelationFields {
                    leaf: self,
                    kind,
                    handle,
                }
                .set(&extract_column_key(&key)?, &value)
            });
            if let Err(err) = written {
                self.write(|mol| -> PyResult<()> {
                    mol.remove_relation(kind_id(mol, kind)?, relation_from_u64(handle))
                        .map(|_| ())
                        .map_err(molrs_error_to_pyerr)
                })??;
                return Err(err);
            }
        }
        self.relation(kind, handle)
    }

    /// The handle of `node`, checked to be a node of this graph.
    ///
    /// A slotmap handle from a foreign graph can alias a live one rather than
    /// fail, so a native writer would happily join the wrong nodes: the
    /// world is compared, not the handle.
    pub(crate) fn own_node(&self, node: &Bound<'py, PyNodeRef>) -> PyResult<u64> {
        let node = node.get();
        if !node.world.bind(self.py()).is(self.object()) {
            return Err(PyValueError::new_err(
                "relation endpoints must belong to this graph",
            ));
        }
        Ok(node.handle)
    }

    /// Remove each of `links` (relation views of this graph).
    pub(crate) fn remove_links(&self, links: &Bound<'py, PyTuple>) -> PyResult<()> {
        for link in links.iter() {
            let link = link.cast::<PyRelationRef>()?.get();
            if !link.world.bind(self.py()).is(self.object()) {
                return Err(PyValueError::new_err("relation belongs to another graph"));
            }
            self.write(|mol| -> PyResult<()> {
                mol.remove_relation(kind_id(mol, &link.kind)?, relation_from_u64(link.handle))
                    .map(|_| ())
                    .map_err(molrs_error_to_pyerr)
            })??;
        }
        Ok(())
    }

    /// Remove each of `nodes` (node views of this graph) and every relation
    /// touching it.
    pub(crate) fn remove_nodes(&self, nodes: &Bound<'py, PyTuple>) -> PyResult<()> {
        for node in nodes.iter() {
            let node = node.cast::<PyNodeRef>()?.get();
            if !node.world.bind(self.py()).is(self.object()) {
                return Err(PyValueError::new_err("node belongs to another graph"));
            }
            self.write(|mol| mol.remove_node(node_from_u64(node.handle)))?
                .map_err(molrs_error_to_pyerr)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Which class a view is
// ---------------------------------------------------------------------------

/// The class of a node view.
enum NodeClass {
    Atom,
    VirtualSite,
    DrudeParticle,
    MasslessSite,
    Bead,
}

impl NodeClass {
    /// An atom's class follows its stored ``vsite``; a bead is a `Bead`.
    fn of(leaf: &Leaf<'_>, mol: &MolGraph, handle: u64) -> PyResult<Self> {
        let id = node_from_u64(handle);
        if !mol.node_table().contains(id) {
            return Err(PyValueError::new_err(format!(
                "cannot bind a node view to stale handle {handle}"
            )));
        }
        Ok(match leaf {
            Leaf::CoarseGrain(_) => Self::Bead,
            Leaf::Atomistic(_) => match mol.node_table().value(id, VSITE) {
                None => Self::Atom,
                Some(EntityCell::Str("drude")) => Self::DrudeParticle,
                Some(EntityCell::Str("massless")) => Self::MasslessSite,
                Some(_) => Self::VirtualSite,
            },
        })
    }

    /// The ``vsite`` a ``def_virtual_site(kind=cls)`` stamps.
    fn vsite_of(cls: &Bound<'_, PyType>) -> PyResult<&'static str> {
        let py = cls.py();
        if cls.is(py.get_type::<PyDrudeParticle>()) {
            Ok("drude")
        } else if cls.is(py.get_type::<PyMasslessSite>()) {
            Ok("massless")
        } else if cls.is(py.get_type::<PyVirtualSite>()) {
            Ok("virtual")
        } else {
            Err(PyTypeError::new_err(format!(
                "kind must be VirtualSite, DrudeParticle or MasslessSite, not {}",
                cls.name()?
            )))
        }
    }

    fn instantiate(self, py: Python<'_>, base: PyNodeRef) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(base);
        Ok(match self {
            Self::Bead => Py::new(py, init.add_subclass(PyBead {}))?.into_any(),
            Self::Atom => Py::new(py, init.add_subclass(PyAtom {}))?.into_any(),
            Self::VirtualSite => {
                let init = init.add_subclass(PyAtom {}).add_subclass(PyVirtualSite {});
                Py::new(py, init)?.into_any()
            }
            Self::DrudeParticle => {
                let init = init
                    .add_subclass(PyAtom {})
                    .add_subclass(PyVirtualSite {})
                    .add_subclass(PyDrudeParticle {});
                Py::new(py, init)?.into_any()
            }
            Self::MasslessSite => {
                let init = init
                    .add_subclass(PyAtom {})
                    .add_subclass(PyVirtualSite {})
                    .add_subclass(PyMasslessSite {});
                Py::new(py, init)?.into_any()
            }
        })
    }
}

/// The class of a relation view.
pub(crate) enum RelationClass {
    /// A kind with no dedicated view class: a plain `RelationRef`.
    RelationRef,
    Bond,
    Angle,
    Dihedral,
    Improper,
    Port,
    CgBond,
}

impl RelationClass {
    fn of(leaf: &Leaf<'_>, kind: &str) -> Self {
        match (leaf, kind) {
            (Leaf::Atomistic(_), "bonds") => Self::Bond,
            (Leaf::Atomistic(_), "angles") => Self::Angle,
            (Leaf::Atomistic(_), "dihedrals") => Self::Dihedral,
            (Leaf::Atomistic(_), "impropers") => Self::Improper,
            (Leaf::Atomistic(_), "ports") => Self::Port,
            (Leaf::CoarseGrain(_), "bonds") => Self::CgBond,
            _ => Self::RelationRef,
        }
    }

    /// The `Atomistic` relation kind whose view class is exactly `cls`
    /// (``Bond`` → ``"bonds"``, …), if any.
    pub(crate) fn atomistic_kind(cls: &Bound<'_, PyAny>) -> Option<&'static str> {
        let py = cls.py();
        [
            (Self::Bond, "bonds"),
            (Self::Angle, "angles"),
            (Self::Dihedral, "dihedrals"),
            (Self::Improper, "impropers"),
            (Self::Port, "ports"),
        ]
        .into_iter()
        .find(|(class, _)| class.type_object(py).is(cls))
        .map(|(_, kind)| kind)
    }

    fn type_object<'py>(&self, py: Python<'py>) -> Bound<'py, PyType> {
        match self {
            Self::RelationRef => py.get_type::<PyRelationRef>(),
            Self::Bond => py.get_type::<PyBond>(),
            Self::Angle => py.get_type::<PyAngle>(),
            Self::Dihedral => py.get_type::<PyDihedral>(),
            Self::Improper => py.get_type::<PyImproper>(),
            Self::Port => py.get_type::<PyPort>(),
            Self::CgBond => py.get_type::<PyCgBond>(),
        }
    }

    fn instantiate(self, py: Python<'_>, base: PyRelationRef) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(base);
        Ok(match self {
            Self::RelationRef => Py::new(py, init)?.into_any(),
            Self::Bond => Py::new(py, init.add_subclass(PyBond {}))?.into_any(),
            Self::Angle => Py::new(py, init.add_subclass(PyAngle {}))?.into_any(),
            Self::Dihedral => Py::new(py, init.add_subclass(PyDihedral {}))?.into_any(),
            Self::Improper => {
                let init = init.add_subclass(PyDihedral {}).add_subclass(PyImproper {});
                Py::new(py, init)?.into_any()
            }
            Self::Port => Py::new(py, init.add_subclass(PyPort {}))?.into_any(),
            Self::CgBond => Py::new(py, init.add_subclass(PyCgBond {}))?.into_any(),
        })
    }
}

// ---------------------------------------------------------------------------
// Field access: the mapping protocol both view kinds share
// ---------------------------------------------------------------------------

/// The fields of one node or one relation.
trait Fields {
    fn keys(&self) -> PyResult<Vec<String>>;
    fn get(&self, key: &str) -> PyResult<Option<Py<PyAny>>>;
    fn set(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()>;
    fn clear(&self, key: &str) -> PyResult<()>;
}

struct NodeFields<'a, 'py> {
    leaf: &'a Leaf<'py>,
    handle: u64,
}

impl Fields for NodeFields<'_, '_> {
    fn keys(&self) -> PyResult<Vec<String>> {
        self.leaf.read(|mol| {
            mol.node_table()
                .row_cells(node_from_u64(self.handle))
                .map(|(key, _)| key.to_owned())
                .collect()
        })
    }

    fn get(&self, key: &str) -> PyResult<Option<Py<PyAny>>> {
        if key == BEAD_ATOMS
            && let Leaf::CoarseGrain(graph) = self.leaf
        {
            return Ok(PyCoarseGrain::bead_atoms(graph, self.handle)?
                .map(|atoms| atoms.into_any().unbind()));
        }
        let py = self.leaf.py();
        self.leaf.read(|mol| {
            mol.node_table()
                .value(node_from_u64(self.handle), key)
                .map(|cell| cell_to_py(py, cell))
                .transpose()
        })?
    }

    fn set(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = py_to_prop(value)?;
        self.leaf
            .write(|mol| mol.set_node(node_from_u64(self.handle), key, value))?
            .map_err(molrs_error_to_pyerr)
    }

    fn clear(&self, key: &str) -> PyResult<()> {
        self.leaf
            .write(|mol| mol.clear_node(node_from_u64(self.handle), key))?
            .map_err(molrs_error_to_pyerr)
    }
}

struct RelationFields<'a, 'py> {
    leaf: &'a Leaf<'py>,
    kind: &'a str,
    handle: u64,
}

impl Fields for RelationFields<'_, '_> {
    fn keys(&self) -> PyResult<Vec<String>> {
        self.leaf.read(|mol| -> PyResult<Vec<String>> {
            let relation = mol
                .get_relation(kind_id(mol, self.kind)?, relation_from_u64(self.handle))
                .map_err(molrs_error_to_pyerr)?;
            Ok(relation.props.keys().cloned().collect())
        })?
    }

    fn get(&self, key: &str) -> PyResult<Option<Py<PyAny>>> {
        let py = self.leaf.py();
        self.leaf.read(|mol| -> PyResult<Option<Py<PyAny>>> {
            let relation = mol
                .get_relation(kind_id(mol, self.kind)?, relation_from_u64(self.handle))
                .map_err(molrs_error_to_pyerr)?;
            relation
                .props
                .get(key)
                .map(|value| prop_to_py(py, value))
                .transpose()
        })?
    }

    fn set(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            return self.clear(key);
        }
        let value = py_to_prop(value)?;
        self.leaf.write(|mol| -> PyResult<()> {
            mol.set_relation_prop(
                kind_id(mol, self.kind)?,
                relation_from_u64(self.handle),
                key,
                value,
            )
            .map_err(molrs_error_to_pyerr)
        })?
    }

    fn clear(&self, key: &str) -> PyResult<()> {
        self.leaf.write(|mol| -> PyResult<()> {
            mol.clear_relation_prop(
                kind_id(mol, self.kind)?,
                relation_from_u64(self.handle),
                key,
            )
            .map_err(molrs_error_to_pyerr)
        })?
    }
}

/// The mapping protocol of a view, over its [`Fields`]: a field name (or a
/// `molrs.core.keys.Key`) reads, writes and deletes one field; a tuple of names
/// reads or writes several at once (``atom["x", "y", "z"]``); assigning
/// ``None`` deletes.
macro_rules! field_mapping_impl {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let leaf = slf.get().leaf(py)?;
                let fields = slf.get().fields(&leaf);
                if let Ok(names) = key.cast::<PyTuple>() {
                    let values = names
                        .iter()
                        .map(|name| field_or_key_error(&fields, &name))
                        .collect::<PyResult<Vec<_>>>()?;
                    return Ok(PyList::new(py, values)?.into_any().unbind());
                }
                field_or_key_error(&fields, key)
            }

            fn __setitem__(
                slf: &Bound<'_, Self>,
                key: &Bound<'_, PyAny>,
                value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let leaf = slf.get().leaf(slf.py())?;
                let fields = slf.get().fields(&leaf);
                if let Ok(names) = key.cast::<PyTuple>() {
                    let values: Vec<Bound<'_, PyAny>> =
                        value.try_iter()?.collect::<PyResult<_>>()?;
                    if values.len() != names.len() {
                        return Err(PyValueError::new_err(format!(
                            "assigning to {} needs {} values, got {}",
                            names.repr()?,
                            names.len(),
                            values.len()
                        )));
                    }
                    for (name, value) in names.iter().zip(&values) {
                        set_or_clear_field(&fields, &extract_column_key(&name)?, value)?;
                    }
                    return Ok(());
                }
                set_or_clear_field(&fields, &extract_column_key(key)?, value)
            }

            fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
                let leaf = slf.get().leaf(slf.py())?;
                let fields = slf.get().fields(&leaf);
                let name = extract_column_key(key)?;
                if fields.get(&name)?.is_none() {
                    return Err(PyKeyError::new_err(name));
                }
                fields.clear(&name)
            }

            fn __contains__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
                let leaf = slf.get().leaf(slf.py())?;
                let fields = slf.get().fields(&leaf);
                if let Ok(names) = key.cast::<PyTuple>() {
                    for name in names.iter() {
                        let Ok(name) = extract_column_key(&name) else {
                            return Ok(false);
                        };
                        if fields.get(&name)?.is_none() {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
                match extract_column_key(key) {
                    Ok(name) => Ok(fields.get(&name)?.is_some()),
                    Err(_) => Ok(false),
                }
            }

            fn __iter__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let leaf = slf.get().leaf(py)?;
                let keys = slf.get().fields(&leaf).keys()?;
                Ok(PyList::new(py, keys)?.try_iter()?.into_any().unbind())
            }

            fn __len__(slf: &Bound<'_, Self>) -> PyResult<usize> {
                let leaf = slf.get().leaf(slf.py())?;
                Ok(slf.get().fields(&leaf).keys()?.len())
            }

            /// The field names, in column order.
            fn keys(slf: &Bound<'_, Self>) -> PyResult<Vec<String>> {
                let leaf = slf.get().leaf(slf.py())?;
                slf.get().fields(&leaf).keys()
            }

            /// The field values, in :meth:`keys` order.
            fn values(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
                Ok(Self::items(slf)?
                    .into_iter()
                    .map(|(_, value)| value)
                    .collect())
            }

            /// ``(name, value)`` pairs, in :meth:`keys` order.
            fn items(slf: &Bound<'_, Self>) -> PyResult<Vec<(String, Py<PyAny>)>> {
                let py = slf.py();
                let leaf = slf.get().leaf(py)?;
                let fields = slf.get().fields(&leaf);
                fields
                    .keys()?
                    .into_iter()
                    .map(|key| {
                        let value = fields.get(&key)?.unwrap_or_else(|| py.None());
                        Ok((key, value))
                    })
                    .collect()
            }

            /// The field `key`, or `default` when it is not set.
            #[pyo3(signature = (key, default = None))]
            fn get(
                slf: &Bound<'_, Self>,
                key: &Bound<'_, PyAny>,
                default: Option<Py<PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let leaf = slf.get().leaf(py)?;
                let value = match extract_column_key(key) {
                    Ok(name) => slf.get().fields(&leaf).get(&name)?,
                    Err(_) => None,
                };
                Ok(value.or(default).unwrap_or_else(|| py.None()))
            }

            /// Write several fields, as ``dict.update`` takes them.
            #[pyo3(signature = (*args, **kwargs))]
            fn update(
                slf: &Bound<'_, Self>,
                args: &Bound<'_, PyTuple>,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<()> {
                let py = slf.py();
                let pairs = py.get_type::<PyDict>().call(args, kwargs)?;
                let pairs = pairs.cast::<PyDict>()?;
                let leaf = slf.get().leaf(py)?;
                let fields = slf.get().fields(&leaf);
                for (key, value) in pairs.iter() {
                    set_or_clear_field(&fields, &extract_column_key(&key)?, &value)?;
                }
                Ok(())
            }
        }
    };
}

fn field_or_key_error(fields: &impl Fields, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let name = extract_column_key(key)?;
    fields.get(&name)?.ok_or_else(|| PyKeyError::new_err(name))
}

/// Write one field; ``None`` deletes it (a no-op when it is not set).
fn set_or_clear_field(fields: &impl Fields, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.is_none() {
        if fields.get(key)?.is_some() {
            fields.clear(key)?;
        }
        return Ok(());
    }
    fields.set(key, value)
}

// ---------------------------------------------------------------------------
// Node views
// ---------------------------------------------------------------------------

/// A live view of one node of a graph.
///
/// A view is made by the graph (``graph.def_atom(...)``, ``graph.atoms[i]``),
/// never constructed directly. It reads and writes the node's fields as a
/// mapping: ``atom["x"]``, ``atom["x", "y", "z"] = (0.0, 0.0, 1.0)``,
/// ``"charge" in atom``. Removing the node leaves the view stale: its reads
/// raise.
#[pyclass(module = "molrs.core", name = "NodeRef", frozen, subclass, weakref)]
pub struct PyNodeRef {
    world: Py<PyAny>,
    handle: u64,
}

impl PyNodeRef {
    fn leaf<'py>(&self, py: Python<'py>) -> PyResult<Leaf<'py>> {
        Leaf::of(self.world.bind(py))
    }

    fn fields<'a, 'py>(&self, leaf: &'a Leaf<'py>) -> NodeFields<'a, 'py> {
        NodeFields {
            leaf,
            handle: self.handle,
        }
    }

    pub(crate) fn world<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.world.bind(py)
    }

    pub(crate) fn handle(&self) -> u64 {
        self.handle
    }
}

#[pymethods]
impl PyNodeRef {
    /// The graph this node belongs to.
    #[getter(world)]
    fn world_getter(&self, py: Python<'_>) -> Py<PyAny> {
        self.world.clone_ref(py)
    }

    /// The node's stable handle in :attr:`world`.
    #[getter(handle)]
    fn handle_getter(&self) -> u64 {
        self.handle
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let class = slf.get_type().name()?;
        let this = slf.get();
        let leaf = this.leaf(py)?;
        let fields = this.fields(&leaf);
        let label = match leaf {
            Leaf::Atomistic(_) => ["element", "type"],
            Leaf::CoarseGrain(_) => ["type", "name"],
        }
        .into_iter()
        .find_map(|key| fields.get(key).ok().flatten());
        let label = match label {
            Some(value) => value.bind(py).str()?.to_string(),
            None => "?".to_owned(),
        };
        Ok(format!("<{class} {}: {label}>", this.handle))
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let row = this
            .leaf(py)?
            .read(|mol| mol.node_table().row(node_from_u64(this.handle)))?
            .ok_or_else(|| PyValueError::new_err("cannot pickle a view of a removed node"))?;
        let restore = py.get_type::<Self>().getattr(intern!(py, "_restore"))?;
        Ok((
            restore,
            PyTuple::new(
                py,
                [
                    this.world.bind(py).clone(),
                    row.into_pyobject(py)?.into_any(),
                ],
            )?,
        ))
    }

    /// Unpickle: the view of the node at `row` of `world`.
    #[classmethod]
    fn _restore<'py>(
        _cls: &Bound<'py, PyType>,
        world: &Bound<'py, PyAny>,
        row: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(world)?;
        let handle = leaf
            .read(|mol| mol.node_ids().nth(row).map(node_to_u64))?
            .ok_or_else(|| PyValueError::new_err(format!("no node at row {row}")))?;
        leaf.node(handle)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.world)
    }
}
field_mapping_impl!(PyNodeRef);

/// A node of an :class:`Atomistic`.
#[pyclass(module = "molrs.core", name = "Atom", extends = PyNodeRef, frozen, subclass)]
pub struct PyAtom {}

/// An atom carrying a ``vsite`` field: a site that is not a nucleus.
#[pyclass(module = "molrs.core", name = "VirtualSite", extends = PyAtom, frozen, subclass)]
pub struct PyVirtualSite {}

/// A Drude shell (``vsite == "drude"``).
#[pyclass(module = "molrs.core", name = "DrudeParticle", extends = PyVirtualSite, frozen, subclass)]
pub struct PyDrudeParticle {}

/// A massless site (``vsite == "massless"``), e.g. the TIP4P M site.
#[pyclass(module = "molrs.core", name = "MasslessSite", extends = PyVirtualSite, frozen, subclass)]
pub struct PyMasslessSite {}

/// A node of a :class:`CoarseGrain`. ``bead["atoms"]`` is the tuple of atom
/// views the bead groups, when it has members.
#[pyclass(module = "molrs.core", name = "Bead", extends = PyNodeRef, frozen, subclass)]
pub struct PyBead {}

// ---------------------------------------------------------------------------
// Relation views
// ---------------------------------------------------------------------------

/// A live view of one relation (a bond, an angle, a port, …) of a graph.
///
/// Made by the graph, never constructed directly. ``endpoints`` are the
/// interned node views, in order; the relation's own fields read and write
/// as a mapping (``bond["order"]``).
#[pyclass(module = "molrs.core", name = "RelationRef", frozen, subclass, weakref)]
pub struct PyRelationRef {
    world: Py<PyAny>,
    kind: String,
    handle: u64,
    endpoints: Py<PyTuple>,
}

impl PyRelationRef {
    fn leaf<'py>(&self, py: Python<'py>) -> PyResult<Leaf<'py>> {
        Leaf::of(self.world.bind(py))
    }

    fn fields<'a, 'py>(&'a self, leaf: &'a Leaf<'py>) -> RelationFields<'a, 'py> {
        RelationFields {
            leaf,
            kind: &self.kind,
            handle: self.handle,
        }
    }

    fn endpoint<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        self.endpoints.bind(py).get_item(index)
    }
}

#[pymethods]
impl PyRelationRef {
    /// The graph this relation belongs to.
    #[getter]
    fn world(&self, py: Python<'_>) -> Py<PyAny> {
        self.world.clone_ref(py)
    }

    /// The relation kind (``"bonds"``, ``"angles"``, …).
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// The relation's stable handle within its kind.
    #[getter]
    fn handle(&self) -> u64 {
        self.handle
    }

    /// The endpoint node views, in order.
    #[getter]
    fn endpoints(&self, py: Python<'_>) -> Py<PyTuple> {
        self.endpoints.clone_ref(py)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let this = slf.get();
        Ok(format!(
            "<{} {}: {}>",
            slf.get_type().name()?,
            this.handle,
            this.endpoints.bind(slf.py()).repr()?
        ))
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let row = this
            .leaf(py)?
            .read(|mol| {
                mol.kind_id(&this.kind)
                    .and_then(|kid| mol.relation_row(kid, relation_from_u64(this.handle)))
            })?
            .ok_or_else(|| PyValueError::new_err("cannot pickle a view of a removed relation"))?;
        let restore = py.get_type::<Self>().getattr(intern!(py, "_restore"))?;
        let args = (this.world.bind(py).clone(), this.kind.clone(), row).into_pyobject(py)?;
        Ok((restore, args))
    }

    /// Unpickle: the view of the relation at `row` of `kind` in `world`.
    #[classmethod]
    fn _restore<'py>(
        _cls: &Bound<'py, PyType>,
        world: &Bound<'py, PyAny>,
        kind: &str,
        row: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(world)?;
        let handle = leaf
            .read(|mol| {
                mol.kind_id(kind)
                    .and_then(|kid| mol.relation_ids(kid).nth(row))
                    .map(relation_to_u64)
            })?
            .ok_or_else(|| PyValueError::new_err(format!("no '{kind}' relation at row {row}")))?;
        leaf.relation(kind, handle)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.world)?;
        visit.call(&self.endpoints)
    }
}
field_mapping_impl!(PyRelationRef);

/// A bond of an :class:`Atomistic`; ``itom`` / ``jtom`` are its atoms.
#[pyclass(module = "molrs.core", name = "Bond", extends = PyRelationRef, frozen, subclass)]
pub struct PyBond {}

#[pymethods]
impl PyBond {
    /// The first atom.
    #[getter]
    fn itom<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.as_super().get().endpoint(slf.py(), 0)
    }

    /// The second atom.
    #[getter]
    fn jtom<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.as_super().get().endpoint(slf.py(), 1)
    }
}

/// An angle ``i–j–k`` of an :class:`Atomistic` (``j`` the vertex).
#[pyclass(module = "molrs.core", name = "Angle", extends = PyRelationRef, frozen, subclass)]
pub struct PyAngle {}

/// A proper dihedral ``i–j–k–l`` of an :class:`Atomistic`.
#[pyclass(module = "molrs.core", name = "Dihedral", extends = PyRelationRef, frozen, subclass)]
pub struct PyDihedral {}

/// An improper of an :class:`Atomistic`, in its style's slot order.
#[pyclass(module = "molrs.core", name = "Improper", extends = PyDihedral, frozen, subclass)]
pub struct PyImproper {}

/// A bond of a :class:`CoarseGrain`.
#[pyclass(module = "molrs.core", name = "CgBond", extends = PyRelationRef, frozen, subclass)]
pub struct PyCgBond {}

/// One unsatisfied valence (a port) of an :class:`Atomistic`.
///
/// A port is the ordered pair ``(anchor, handle_atom)``: the **anchor** keeps
/// its place in the product molecule and the **handle atom** is the capping
/// atom bonded to it that a paired descriptor's bond replaces. The descriptor
/// rides the relation's fields ``port_kind`` (``$``, ``<``, ``>`` or ``!``),
/// ``port_label`` (``""`` when unnamed) and ``port_order``.
///
/// The second endpoint is ``handle_atom``, not ``handle``: ``handle`` is the
/// port's own relation handle. The two answer different questions.
#[pyclass(module = "molrs.core", name = "Port", extends = PyRelationRef, frozen, subclass)]
pub struct PyPort {}

#[pymethods]
impl PyPort {
    /// The atom that keeps its place in the product molecule.
    #[getter]
    fn anchor<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.as_super().get().endpoint(slf.py(), 0)
    }

    /// The atom the descriptor sits on: the root of the leaving group.
    #[getter]
    fn handle_atom<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.as_super().get().endpoint(slf.py(), 1)
    }
}

// ---------------------------------------------------------------------------
// Refs
// ---------------------------------------------------------------------------

/// An ordered collection of views of one kind of one graph.
///
/// ``refs[i]`` is a view, ``refs[a:b]`` a smaller collection; ``refs["x"]``
/// reads a field of every item as a numpy array (``None`` where unset) and
/// ``refs["x", "y", "z"]`` stacks several side by side. Views are made on
/// demand, so ``len(graph.atoms)`` and ``graph.atoms["x"]`` make none.
#[pyclass(module = "molrs.core", name = "Refs", frozen, subclass)]
pub struct PyRefs {
    world: Py<PyAny>,
    /// `None` for nodes, the relation kind otherwise.
    kind: Option<String>,
    handles: Vec<u64>,
}

impl PyRefs {
    fn view<'py>(&self, leaf: &Leaf<'py>, handle: u64) -> PyResult<Bound<'py, PyAny>> {
        match &self.kind {
            None => leaf.node(handle),
            Some(kind) => leaf.relation(kind, handle),
        }
    }

    fn column<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let key = extract_column_key(key)?;
        let world = self.world.bind(py);
        let leaf = Leaf::of(world)?;
        if self.handles.is_empty() {
            // Nothing to read, whatever the kind: an empty `exact_bucket` is
            // under a kind this graph may not even register.
            let numpy = py.import(intern!(py, "numpy"))?;
            return numpy.call_method1(intern!(py, "array"), (PyList::empty(py),));
        }
        if self.kind.is_none() {
            // The whole node table, fully set: the graph's own zero-copy column.
            let dense = leaf.read(|mol| {
                let table = mol.node_table();
                table
                    .handles()
                    .map(node_to_u64)
                    .eq(self.handles.iter().copied())
                    && table
                        .col_validity(&key)
                        .is_some_and(|valid| valid.as_slice().iter().all(|v| *v))
            })?;
            if dense {
                return world.call_method1(intern!(py, "column"), (key,));
            }
        }
        let values = leaf.read(|mol| -> PyResult<Vec<Py<PyAny>>> {
            match &self.kind {
                None => self
                    .handles
                    .iter()
                    .map(|&h| match mol.node_table().value(node_from_u64(h), &key) {
                        Some(cell) => cell_to_py(py, cell),
                        None => Ok(py.None()),
                    })
                    .collect(),
                Some(kind) => {
                    let kid = kind_id(mol, kind)?;
                    self.handles
                        .iter()
                        .map(|&h| {
                            let relation = mol
                                .get_relation(kid, relation_from_u64(h))
                                .map_err(molrs_error_to_pyerr)?;
                            match relation.props.get(&key) {
                                Some(value) => prop_to_py(py, value),
                                None => Ok(py.None()),
                            }
                        })
                        .collect()
                }
            }
        })??;
        let numpy = py.import(intern!(py, "numpy"))?;
        let list = PyList::new(py, values)?;
        match numpy.call_method1(intern!(py, "array"), (&list,)) {
            Ok(array) => Ok(array),
            Err(_) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "dtype"), intern!(py, "object"))?;
                numpy.call_method(intern!(py, "array"), (&list,), Some(&kwargs))
            }
        }
    }

    fn rows(&self, leaf: &Leaf<'_>) -> PyResult<Vec<usize>> {
        leaf.read(|mol| -> PyResult<Vec<usize>> {
            let row = |h: u64| match &self.kind {
                None => mol.node_table().row(node_from_u64(h)),
                Some(kind) => mol
                    .kind_id(kind)
                    .and_then(|kid| mol.relation_row(kid, relation_from_u64(h))),
            };
            self.handles
                .iter()
                .map(|&h| {
                    row(h).ok_or_else(|| {
                        PyValueError::new_err("cannot pickle a view of a removed node or relation")
                    })
                })
                .collect()
        })?
    }
}

#[pymethods]
impl PyRefs {
    fn __len__(&self) -> usize {
        self.handles.len()
    }

    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        if key.is_instance_of::<PyString>()
            || key
                .extract::<PyRef<'_, crate::core::schema::PyKey>>()
                .is_ok()
        {
            return this.column(py, key);
        }
        if let Ok(names) = key.cast::<PyTuple>() {
            let columns = names
                .iter()
                .map(|name| this.column(py, &name))
                .collect::<PyResult<Vec<_>>>()?;
            let numpy = py.import(intern!(py, "numpy"))?;
            return numpy.call_method1(intern!(py, "column_stack"), (PyList::new(py, columns)?,));
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(this.handles.len() as isize)?;
            let mut handles = Vec::with_capacity(indices.slicelength);
            let mut at = indices.start;
            for _ in 0..indices.slicelength {
                handles.push(this.handles[at as usize]);
                at += indices.step;
            }
            let sliced = Self {
                world: this.world.clone_ref(py),
                kind: this.kind.clone(),
                handles,
            };
            return Ok(Bound::new(py, sliced)?.into_any());
        }
        let index: isize = key.extract()?;
        let len = this.handles.len() as isize;
        let at = if index < 0 { index + len } else { index };
        if !(0..len).contains(&at) {
            return Err(PyIndexError::new_err("Refs index out of range"));
        }
        this.view(&Leaf::of(this.world.bind(py))?, this.handles[at as usize])
    }

    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        let leaf = Leaf::of(this.world.bind(py))?;
        let views = this
            .handles
            .iter()
            .map(|&h| this.view(&leaf, h))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, views)?.try_iter()?.into_any())
    }

    fn __contains__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> bool {
        let (world, kind, handle) = if let Ok(node) = item.cast::<PyNodeRef>() {
            let node = node.get();
            (&node.world, None, node.handle)
        } else if let Ok(relation) = item.cast::<PyRelationRef>() {
            let relation = relation.get();
            (
                &relation.world,
                Some(relation.kind.as_str()),
                relation.handle,
            )
        } else {
            return false;
        };
        world.bind(py).is(self.world.bind(py))
            && kind == self.kind.as_deref()
            && self.handles.contains(&handle)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Refs of {}: {}>",
            self.kind.as_deref().unwrap_or("nodes"),
            self.handles.len()
        )
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let rows = this.rows(&Leaf::of(this.world.bind(py))?)?;
        let restore = py.get_type::<Self>().getattr(intern!(py, "_restore"))?;
        let args = (this.world.bind(py).clone(), this.kind.clone(), rows).into_pyobject(py)?;
        Ok((restore, args))
    }

    /// Unpickle: the items at `rows` of `kind` (``None``: nodes) in `world`.
    #[classmethod]
    fn _restore(
        _cls: &Bound<'_, PyType>,
        world: &Bound<'_, PyAny>,
        kind: Option<String>,
        rows: Vec<usize>,
    ) -> PyResult<Self> {
        let leaf = Leaf::of(world)?;
        let handles = leaf.read(|mol| -> PyResult<Vec<u64>> {
            if rows.is_empty() {
                return Ok(Vec::new());
            }
            let all: Vec<u64> = match &kind {
                None => mol.node_ids().map(node_to_u64).collect(),
                Some(kind) => mol
                    .relation_ids(kind_id(mol, kind)?)
                    .map(relation_to_u64)
                    .collect(),
            };
            rows.iter()
                .map(|&row| {
                    all.get(row)
                        .copied()
                        .ok_or_else(|| PyValueError::new_err(format!("no item at row {row}")))
                })
                .collect()
        })??;
        Ok(Self {
            world: world.clone().unbind(),
            kind,
            handles,
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.world)
    }
}

impl PyRefs {
    pub(crate) fn nodes_of(world: &Bound<'_, PyAny>) -> PyResult<Self> {
        Leaf::of(world)?.nodes()
    }

    pub(crate) fn relations_of(world: &Bound<'_, PyAny>, kind: &str) -> PyResult<Self> {
        Leaf::of(world)?.relations(kind)
    }
}

// ---------------------------------------------------------------------------
// links
// ---------------------------------------------------------------------------

/// A graph's relations, selected by view class (``graph.links``).
#[pyclass(module = "molrs.core", name = "RelationBuckets", frozen, subclass)]
pub struct PyRelationBuckets {
    world: Py<PyAny>,
}

impl PyRelationBuckets {
    pub(crate) fn new(world: &Bound<'_, PyAny>) -> Self {
        Self {
            world: world.clone().unbind(),
        }
    }
}

#[pymethods]
impl PyRelationBuckets {
    /// The relations whose view class is exactly `cls`, in row order.
    ///
    /// Every class names one relation kind of the graph (``Bond`` its
    /// ``bonds``, ``Improper`` its ``impropers``); ``RelationRef`` names the
    /// one kind without a class of its own.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``cls`` names more than one kind of this graph (``RelationRef``
    ///     on a graph with two unclassed kinds).
    fn exact_bucket(&self, py: Python<'_>, cls: &Bound<'_, PyType>) -> PyResult<PyRefs> {
        let world = self.world.bind(py);
        let leaf = Leaf::of(world)?;
        let kinds: Vec<String> = leaf.read(|mol| {
            mol.kind_ids()
                .map(|kid| mol.kind_name(kid).to_owned())
                .collect()
        })?;
        let mut matching = kinds
            .into_iter()
            .filter(|kind| RelationClass::of(&leaf, kind).type_object(py).is(cls));
        match (matching.next(), matching.next()) {
            // No kind of this graph has that class: an empty collection under
            // the kind the class stands for, which reads as empty columns.
            (None, _) => {
                let kind = RelationClass::atomistic_kind(cls.as_any())
                    .or_else(|| cls.is(py.get_type::<PyCgBond>()).then_some("bonds"))
                    .unwrap_or("relations");
                Ok(PyRefs {
                    world: world.clone().unbind(),
                    kind: Some(kind.to_owned()),
                    handles: Vec::new(),
                })
            }
            (Some(kind), None) => leaf.relations(&kind),
            (Some(first), Some(second)) => Err(PyTypeError::new_err(format!(
                "{} names more than one relation kind of this graph ('{first}', '{second}')",
                cls.name()?
            ))),
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.world)
    }
}
