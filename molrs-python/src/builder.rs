//! Python bindings for structure builders (`molrs::builder`).

use molrs::{CarbonTubeBuilder, GrapheneBuilder};
use pyo3::PyRefMut;
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::{Borrowed, intern};

use crate::core::spatial::simbox::PyBox;
use crate::core::store::frame::PyFrame;
use crate::core::system::molgraph::{expect_world, try_with_world_mut};
use molrs::system::molgraph::{node_from_u64, node_to_u64};

/// Exact single-wall carbon nanotube builder.
#[pyclass(module = "molrs.builder", name = "CarbonTubeBuilder", subclass)]
pub struct PyCarbonTubeBuilder {
    inner: CarbonTubeBuilder,
}

#[pymethods]
impl PyCarbonTubeBuilder {
    #[new]
    #[pyo3(signature = (n, m, *, length=None, cells=None, bond_length=1.42, periodic=false, vacuum=10.0))]
    fn new(
        n: u32,
        m: u32,
        length: Option<f64>,
        cells: Option<usize>,
        bond_length: f64,
        periodic: bool,
        vacuum: f64,
    ) -> PyResult<Self> {
        if length.is_some() && cells.is_some() {
            return Err(PyTypeError::new_err(
                "length and cells are mutually exclusive",
            ));
        }

        let mut inner = CarbonTubeBuilder::new(n, m)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .with_bond_length(bond_length)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .with_periodic(periodic)
            .with_vacuum(vacuum)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if let Some(cells) = cells {
            inner = inner
                .with_cells(cells)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        if let Some(length) = length {
            inner = inner
                .with_length(length)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        inner
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a fresh frame containing atoms, bonds, and the simulation box.
    #[pyo3(signature = (*, atom_type=None, charge=0.0))]
    fn build(&self, atom_type: Option<String>, charge: f64) -> PyResult<PyFrame> {
        let mut builder = self
            .inner
            .clone()
            .with_charge(charge)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if let Some(atom_type) = atom_type {
            builder = builder
                .with_atom_type(atom_type)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        PyFrame::from_core_frame(
            builder
                .build()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )
    }

    /// Return the matching simulation cell, optionally overriding vacuum.
    #[pyo3(signature = (*, vacuum=None))]
    fn cell(&self, vacuum: Option<f64>) -> PyResult<PyBox> {
        let builder = match vacuum {
            Some(vacuum) => self
                .inner
                .clone()
                .with_vacuum(vacuum)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            None => self.inner.clone(),
        };
        Ok(PyBox {
            inner: builder
                .cell()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn n(&self) -> u32 {
        self.inner.n()
    }

    #[getter]
    fn m(&self) -> u32 {
        self.inner.m()
    }

    #[getter]
    fn cells(&self) -> usize {
        self.inner.cells()
    }

    #[getter]
    fn bond_length(&self) -> f64 {
        self.inner.bond_length()
    }

    #[getter]
    fn periodic(&self) -> bool {
        self.inner.periodic()
    }
}

/// Rectangular graphene (honeycomb) sheet builder.
#[pyclass(module = "molrs.builder", name = "GrapheneBuilder", subclass)]
pub struct PyGrapheneBuilder {
    inner: GrapheneBuilder,
}

#[pymethods]
impl PyGrapheneBuilder {
    #[new]
    #[pyo3(signature = (nx, ny, *, bond_length=1.42, vacuum=10.0, periodic_xy=true))]
    fn new(nx: u32, ny: u32, bond_length: f64, vacuum: f64, periodic_xy: bool) -> PyResult<Self> {
        let inner = GrapheneBuilder::new(nx, ny)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_bond_length(bond_length)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_vacuum(vacuum)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_periodic_xy(periodic_xy);
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, atom_type=None, charge=0.0))]
    fn build(&self, atom_type: Option<String>, charge: f64) -> PyResult<PyFrame> {
        let mut builder = self
            .inner
            .clone()
            .with_charge(charge)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(atom_type) = atom_type {
            builder = builder
                .with_atom_type(atom_type)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        PyFrame::from_core_frame(
            builder
                .build()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )
    }

    #[pyo3(signature = (*, vacuum=None))]
    fn cell(&self, vacuum: Option<f64>) -> PyResult<PyBox> {
        let builder = match vacuum {
            Some(v) => self
                .inner
                .clone()
                .with_vacuum(v)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            None => self.inner.clone(),
        };
        Ok(PyBox {
            inner: builder
                .cell()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    #[getter]
    fn nx(&self) -> u32 {
        self.inner.nx()
    }

    #[getter]
    fn ny(&self) -> u32 {
        self.inner.ny()
    }

    #[getter]
    fn bond_length(&self) -> f64 {
        self.inner.bond_length()
    }

    #[getter]
    fn periodic_xy(&self) -> bool {
        self.inner.periodic_xy()
    }
}

/// A trajectory of points. No chemistry, no facing.
#[pyclass(module = "molrs", name = "Trace")]
pub struct PyTrace {
    inner: molrs::spatial::Trace,
}

#[pymethods]
impl PyTrace {
    #[new]
    fn new(points: Vec<[f64; 3]>) -> Self {
        Self {
            inner: molrs::spatial::Trace::from_arrays(points),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn point(&self, index: usize) -> PyResult<[f64; 3]> {
        self.inner
            .point(index)
            .ok_or_else(|| PyIndexError::new_err(format!("trace index {index} out of range")))
    }

    fn tangent(&self, index: usize) -> PyResult<[f64; 3]> {
        self.inner
            .tangent(index)
            .ok_or_else(|| PyIndexError::new_err(format!("trace index {index} has no tangent")))
    }
}

/// Mark the atoms of one graph that a reaction may bind.
///
/// Every node argument is an int handle or a node view (`NodeRef`, `Atom`, …)
/// — anything with an int `.handle`. Returned nodes are int handles.
#[pyclass(module = "molrs", name = "SiteMap")]
pub struct PySiteMap {
    mol: Py<PyAny>,
}

/// A node argument: an int handle, or any object with an int `.handle` (the
/// `NodeRef` views of `molrs.views`). The one place a binder method turns a
/// Python node into a [`molrs::NodeId`].
struct NodeArg(molrs::NodeId);

impl<'a, 'py> FromPyObject<'a, 'py> for NodeArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(raw) = obj.extract::<u64>() {
            return Ok(Self(node_from_u64(raw)));
        }
        let not_a_node = || {
            PyTypeError::new_err(format!(
                "expected an int node handle or a node view with an int .handle, got {}",
                obj.get_type()
                    .name()
                    .map_or_else(|_| "?".to_string(), |name| name.to_string())
            ))
        };
        let raw = obj
            .getattr(intern!(obj.py(), "handle"))
            .map_err(|_| not_a_node())?
            .extract::<u64>()
            .map_err(|_| not_a_node())?;
        Ok(Self(node_from_u64(raw)))
    }
}

fn as_handles(nodes: Vec<NodeArg>) -> Vec<molrs::NodeId> {
    nodes.into_iter().map(|NodeArg(node)| node).collect()
}

fn names_of(names: &[String]) -> Vec<&str> {
    names.iter().map(String::as_str).collect()
}

#[pymethods]
impl PySiteMap {
    /// Bind to `mol`, which must be a `Graph` or one of its leaves.
    #[new]
    fn new(mol: Bound<'_, PyAny>) -> PyResult<Self> {
        expect_world(&mol)?;
        Ok(Self { mol: mol.unbind() })
    }

    /// The graph these labels are written to.
    #[getter]
    fn mol(&self, py: Python<'_>) -> Py<PyAny> {
        self.mol.clone_ref(py)
    }

    /// Label one atom with a site name.
    fn label(&self, py: Python<'_>, node: NodeArg, name: &str) -> PyResult<()> {
        let bound = self.mol.bind(py);
        try_with_world_mut(bound, |graph| {
            molrs::SiteMap::new(graph).label(node.0, name)
        })
    }

    /// Label `nodes` with `names`, in order.
    #[pyo3(signature = (nodes, *names))]
    fn label_atoms(
        &self,
        py: Python<'_>,
        nodes: Vec<NodeArg>,
        names: Vec<String>,
    ) -> PyResult<Vec<u64>> {
        let bound = self.mol.bind(py);
        try_with_world_mut(bound, |graph| -> Result<Vec<u64>, molrs::SiteError> {
            let marked =
                molrs::SiteMap::new(graph).label_atoms(&as_handles(nodes), &names_of(&names))?;
            Ok(marked.into_iter().map(node_to_u64).collect::<Vec<u64>>())
        })
    }

    /// Label the first atoms of `element`, in node order.
    #[pyo3(signature = (element, *names))]
    fn label_elements(
        &self,
        py: Python<'_>,
        element: &str,
        names: Vec<String>,
    ) -> PyResult<Vec<u64>> {
        let bound = self.mol.bind(py);
        try_with_world_mut(bound, |graph| -> Result<Vec<u64>, String> {
            // The core error counts atoms without naming the element; the
            // caller only asked about one, so say which.
            match molrs::SiteMap::new(graph).label_elements(element, &names_of(&names)) {
                Ok(marked) => Ok(marked.into_iter().map(node_to_u64).collect()),
                Err(molrs::SiteError::TooFewAtoms { needed, found }) => {
                    Err(format!("need {needed} {element} atoms, found {found}"))
                }
                Err(other) => Err(other.to_string()),
            }
        })
    }

    /// Label `nodes[0::step]`, optionally preparing each one's leaving hydrogen.
    #[pyo3(signature = (nodes, step, site, leaving=None, fold_charge=true))]
    fn every_nth(
        &self,
        py: Python<'_>,
        nodes: Vec<NodeArg>,
        step: usize,
        site: &str,
        leaving: Option<String>,
        fold_charge: bool,
    ) -> PyResult<Vec<u64>> {
        let bound = self.mol.bind(py);
        try_with_world_mut(bound, |graph| -> Result<Vec<u64>, molrs::SiteError> {
            let marked = molrs::SiteMap::new(graph).every_nth(
                &as_handles(nodes),
                step,
                site,
                leaving.as_deref(),
                fold_charge,
            )?;
            Ok(marked.into_iter().map(node_to_u64).collect::<Vec<u64>>())
        })
    }

    /// Prepare a leaving hydrogen on every atom already labelled `site`.
    #[pyo3(signature = (site, leaving="h", fold_charge=true))]
    fn prepare_leaving_hydrogens(
        &self,
        py: Python<'_>,
        site: &str,
        leaving: &str,
        fold_charge: bool,
    ) -> PyResult<usize> {
        let bound = self.mol.bind(py);
        try_with_world_mut(bound, |graph| {
            molrs::SiteMap::new(graph).prepare_leaving_hydrogens(site, leaving, fold_charge)
        })
    }

    /// Clear site labels: on `nodes`, or on the whole graph when `None`.
    #[pyo3(signature = (nodes=None))]
    fn clear(&self, py: Python<'_>, nodes: Option<Vec<NodeArg>>) -> PyResult<()> {
        let bound = self.mol.bind(py);
        try_with_world_mut(bound, |graph| {
            let targets = nodes.map(as_handles);
            molrs::SiteMap::new(graph).clear(targets.as_deref())
        })
        .map(|_| ())
    }
}

/// Put the fragments a set of forming bonds joins at a pose. Base class.
///
/// The contract every placer honours: a forming bond is a
/// `(parent-side atom, child-side atom)` pair of handles; after `place`, every
/// tree forming bond — a fragment to its parent in the placement walk — sits
/// at bonding range; a placer that cannot do that raises and never leaves a
/// partial placement behind. A placer does not close rings: a ring-closing
/// bond is formed but not placed, and its length is the caller's concern (an
/// explicit trace, or geometry optimisation afterwards).
#[pyclass(module = "molrs", name = "Placer", subclass)]
pub struct PyPlacer;

#[pymethods]
impl PyPlacer {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyAny>, _kwargs: Option<&Bound<'_, PyAny>>) -> Self {
        PyPlacer
    }

    /// Move whole fragments so every tree forming bond in `bonds` ends at
    /// bonding range. The base class places nothing: a subclass implements it.
    fn place(&self, _mol: &Bound<'_, PyAny>, _bonds: Vec<(u64, u64)>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "Placer.place is abstract; a subclass implements it",
        ))
    }
}

/// The native facing rule a [`PyTracePlacer`] applies. Closed set: the variant
/// is the class, and a new rule is a new class on both sides of the binding.
fn extract_orienter(orienter: &Bound<'_, PyAny>) -> PyResult<Box<dyn molrs::Orienter>> {
    if orienter.is_instance_of::<PyLineOrienter>() {
        Ok(Box::new(molrs::LineOrienter))
    } else if orienter.is_instance_of::<PyTangOrienter>() {
        Ok(Box::new(molrs::TangOrienter))
    } else {
        Err(PyTypeError::new_err(
            "expected a LineOrienter or a TangOrienter",
        ))
    }
}

/// Grow the fragments a set of forming bonds joins out of one another.
///
/// Fragments are node groups read off `res_id` by default. The placer walks
/// the fragment graph breadth-first from the lowest fragment id and moves each
/// other fragment rigidly, once, relative to **its own parent**: the child's
/// anchor lands one bonding range (summed covalent radii plus the buffer) from
/// the parent's reacting atom along a growth direction, and the orienter turns
/// the child to point along it, away from the parent.
///
/// Without a trace the root stays put and a child grows along its parent's
/// outward direction (centroid through reacting atom), so any tree of
/// fragments places, and a ring places along its spanning tree. With a trace
/// (`with_trace`) the fragments must form a single path or a single ring,
/// walked as a path from its lowest id; the trace supplies directions only. A
/// forming bond that closes a ring of fragments is neither placed nor checked:
/// its length is the caller's concern (a closed trace of the ring's size, or
/// geometry optimisation afterwards). Failures raise `ValueError`
/// before any coordinate is written (only a refused coordinate write can fail
/// later); `molrs::Placer::place` lists every case.
#[pyclass(module = "molrs", name = "TracePlacer", extends = PyPlacer)]
pub struct PyTracePlacer {
    inner: molrs::TracePlacer,
}

#[pymethods]
impl PyTracePlacer {
    #[new]
    fn new() -> (Self, PyPlacer) {
        (
            PyTracePlacer {
                inner: molrs::TracePlacer::new(),
            },
            PyPlacer,
        )
    }

    /// Grow a path of fragments along this trace's tangents instead of each
    /// parent's outward direction: the root's outward direction follows the
    /// tangent at sample 0 with its reacting atom on sample 0, and the `k`-th
    /// child grows along the tangent at sample `k`. The chain follows the
    /// curve's shape at bonding range, not the samples themselves. A ring of
    /// fragments is walked as a path from its lowest id; its closing bond is
    /// formed but not placed. `place` then refuses a fragment joined to three
    /// others or more, and a trace with fewer samples than fragments, with
    /// `ValueError`.
    fn with_trace<'py>(mut slf: PyRefMut<'py, Self>, trace: &PyTrace) -> PyRefMut<'py, Self> {
        let current = std::mem::take(&mut slf.inner);
        slf.inner = current.with_trace(trace.inner.clone());
        slf
    }

    /// Face each fragment by this rule instead of `LineOrienter()`.
    fn with_orienter<'py>(
        mut slf: PyRefMut<'py, Self>,
        orienter: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let extracted = extract_orienter(orienter)?;
        let current = std::mem::take(&mut slf.inner);
        slf.inner = current.with_orienter(extracted);
        Ok(slf)
    }

    /// Extra separation (Å) beyond the summed covalent radii.
    fn with_buffer<'py>(mut slf: PyRefMut<'py, Self>, buffer: f64) -> PyRefMut<'py, Self> {
        let current = std::mem::take(&mut slf.inner);
        slf.inner = current.with_buffer(buffer);
        slf
    }

    /// The field that groups nodes into fragments (default `res_id`).
    fn with_group_key<'py>(mut slf: PyRefMut<'py, Self>, key: &str) -> PyRefMut<'py, Self> {
        let current = std::mem::take(&mut slf.inner);
        slf.inner = current.with_group_key(key);
        slf
    }

    /// The field that marks a fragment's site atoms (default `site`).
    fn with_site_key<'py>(mut slf: PyRefMut<'py, Self>, key: &str) -> PyRefMut<'py, Self> {
        let current = std::mem::take(&mut slf.inner);
        slf.inner = current.with_site_key(key);
        slf
    }

    /// Move whole fragments so every tree forming bond's endpoints sit at
    /// bonding range; a ring-closing bond is formed but not placed. `bonds`
    /// are `(parent-side handle, child-side handle)` pairs. Raises
    /// `ValueError` — with nothing moved — when the bonds leave a fragment
    /// unreachable, a trace meets a branched fragment graph or runs out of
    /// samples, a fragment cannot be faced, or an endpoint lacks a fragment
    /// id, element, radius or coordinates.
    fn place(&self, mol: &Bound<'_, PyAny>, bonds: Vec<(u64, u64)>) -> PyResult<()> {
        let pairs: Vec<(molrs::NodeId, molrs::NodeId)> = bonds
            .into_iter()
            .map(|(a, b)| (node_from_u64(a), node_from_u64(b)))
            .collect();
        try_with_world_mut(mol, |graph| {
            molrs::Placer::place(&self.inner, graph, &pairs)
        })
    }
}

/// Which way a fragment faces. A closed set — `LineOrienter` and
/// `TangOrienter` — because a facing rule runs natively inside a placer, so a
/// Python subclass could never be applied; subclassing it is refused.
///
/// `subclass` stays on the pyclass only so the two native rules can extend it;
/// `__init_subclass__` refuses every class Python itself creates.
#[pyclass(module = "molrs", name = "Orienter", subclass)]
pub struct PyOrienter;

#[pymethods]
impl PyOrienter {
    #[new]
    fn new() -> Self {
        PyOrienter
    }

    /// Refuse a Python subclass when it is defined, not when it is used.
    #[classmethod]
    #[pyo3(signature = (**_kwargs))]
    fn __init_subclass__(
        _cls: &Bound<'_, pyo3::types::PyType>,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Orienter is a closed set (LineOrienter, TangOrienter); it cannot be subclassed",
        ))
    }
}

fn orient_with<O: molrs::Orienter>(
    orienter: &O,
    mol: &Bound<'_, PyAny>,
    anchor: [f64; 3],
    body_axis: [f64; 3],
    to_dir: [f64; 3],
    flip: bool,
) -> PyResult<()> {
    try_with_world_mut(mol, |graph| {
        orienter.orient(graph, anchor, body_axis, to_dir, flip)
    })
}

macro_rules! orienter_class {
    ($py:ident, $rust:expr, $name:literal, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(module = "molrs", name = $name, extends = PyOrienter)]
        pub struct $py;

        #[pymethods]
        impl $py {
            #[new]
            fn new() -> (Self, PyOrienter) {
                ($py, PyOrienter)
            }

            /// The direction this rule reads out of `body_axis`, or ``None``.
            fn direction(&self, body_axis: [f64; 3]) -> Option<[f64; 3]> {
                molrs::Orienter::direction(&$rust, body_axis)
            }

            #[pyo3(signature = (mol, anchor, body_axis, to_dir, flip=false))]
            fn orient(
                &self,
                mol: &Bound<'_, PyAny>,
                anchor: [f64; 3],
                body_axis: [f64; 3],
                to_dir: [f64; 3],
                flip: bool,
            ) -> PyResult<()> {
                orient_with(&$rust, mol, anchor, body_axis, to_dir, flip)
            }
        }
    };
}

orienter_class!(
    PyLineOrienter,
    molrs::LineOrienter,
    "LineOrienter",
    "The site axis itself: the fragment's outgoing site points along the trace."
);
orienter_class!(
    PyTangOrienter,
    molrs::TangOrienter,
    "TangOrienter",
    "A perpendicular of the site axis: the fragment meets the trace at an angle."
);
