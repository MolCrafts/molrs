//! Python bindings for structure builders (`molrs::builder`) and the assembly
//! surface under `molrs.builder`: `FragLibrary`, the placers, orienters,
//! reacters, `Finalizer` and `Assembler`. The core `Trace` and `Mapping` they
//! consume live beside the core layout (`core::spatial::trace`,
//! `core::system::mapping`).
//!
//! # Crossing a component (assembly-07 §2)
//!
//! `Placer`, `Orienter` and `Reacter` are subclassable base pyclasses holding
//! an [`Implementation`]: either a native Rust component behind an [`Arc`], or
//! "the Python object itself". A Rust consumer (`Assembler`, `TracePlacer`)
//! takes a `Box<dyn Trait>`, built by the base's `boxed`:
//!
//! - **native** — a `Shared*` newtype over the `Arc` that forwards every trait
//!   method, the batched ones included, so a native batched override (e.g.
//!   `TracePlacer::place_many`) is never replaced by the per-unit default;
//! - **Python** — a `Py*Adapter` over the instance that re-enters Python once
//!   per batch by overriding the batched trait method.
//!
//! These adaptors are new with this surface. `PyTypifier`
//! (`ff/mod.rs`) is **not** their precedent: it wraps no Python subclass in a
//! Rust-trait adaptor, keeping `TypifierState::Python(Option<ForceField>)` and
//! calling `match` on the Python object directly from its `typify` pymethod.
//! It is the pattern here only for the base pyclass's two-state enum (and, on
//! the typifier, its `__init_subclass__` class-creation check, which these
//! bases do not need).

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use molrs::op::rigid::Rigid;
use molrs::op::superpose::Fit;
use molrs::op::types::Vec3;
use molrs::store::keys;
use molrs::system::atomistic::BondId;
use molrs::system::fragment::{Fragment, PortId};
use molrs::system::molgraph::{node_to_u64, relation_from_u64, relation_to_u64};
use molrs::{
    AssembleError, Assembler, BodyAxis, CarbonTubeBuilder, Finalizer, FragLibrary,
    FragLibraryError, GrapheneBuilder, HintOrienter, NullOrienter, OrientError, Orienter,
    PairError, PlaceError, Placer, PortReacter, RandomOrienter, ReactError, Reacter, TracePlacer,
};
use ndarray::{Array1, Array2, Array3};
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayLikeDyn};
use pyo3::PyClassInitializer;
use pyo3::exceptions::{PyException, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::core::spatial::simbox::PyBox;
use crate::core::spatial::trace::PyTrace;
use crate::core::store::frame::PyFrame;
use crate::core::system::frag_graph::PyFragGraph;
use crate::core::system::mapping::PyMapping;
use crate::core::system::molgraph::{PyAtomistic, PyCoarseGrain, PyFragment};
use crate::helpers::{molrs_error_to_pyerr, py_value_err};
use crate::op::{PyFit, points_from_array, rigids_from_arrays};

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

// ---------------------------------------------------------------------------
// Array seams of the assembly surface
// ---------------------------------------------------------------------------

/// A refusal's message with every graph id written as the `int` handle
/// Python holds for it (`node_to_u64` for an atom or bead, `relation_to_u64`
/// for a port), never as the Rust debug form `NodeId(3v1)`. A variant that
/// carries no id keeps its `Display`.
trait HandleMessage {
    fn handle_message(&self) -> String;
}

impl HandleMessage for PlaceError {
    fn handle_message(&self) -> String {
        match self {
            Self::MissingBead { node } => format!(
                "atom {} carries no non-negative '{}'",
                node_to_u64(*node),
                keys::BEAD
            ),
            Self::MissingMass { node } => {
                format!("atom {} carries no '{}'", node_to_u64(*node), keys::MASS)
            }
            Self::MissingCoordinates { node } => {
                format!("atom {} lacks x/y/z coordinates", node_to_u64(*node))
            }
            other => other.to_string(),
        }
    }
}

impl HandleMessage for ReactError {
    fn handle_message(&self) -> String {
        let port = |id| relation_to_u64(id);
        let atom = |id| node_to_u64(id);
        match self {
            Self::StalePort { port: p } => format!(
                "port {} is stale: its anchor–handle bond no longer exists",
                port(*p)
            ),
            Self::Incompatible { a, b } => {
                format!("ports {} and {} are not compatible", port(*a), port(*b))
            }
            Self::SameAnchor { a, b } => format!(
                "ports {} and {} share one anchor; a bond cannot join an atom to itself",
                port(*a),
                port(*b)
            ),
            Self::AlreadyBonded { a, b } => {
                format!("anchors {} and {} are already bonded", atom(*a), atom(*b))
            }
            Self::BranchReachesAnchor { port: p } => {
                format!("port {}'s handle branch reaches its own anchor", port(*p))
            }
            Self::OneSidedCharge { anchor } => format!(
                "charge is present on only part of anchor {} and its handle branch",
                atom(*anchor)
            ),
            other => other.to_string(),
        }
    }
}

impl HandleMessage for PairError {
    fn handle_message(&self) -> String {
        match self.pair {
            Some(i) => format!("pair {i}: {}", self.error.handle_message()),
            None => format!("a pair: {}", self.error.handle_message()),
        }
    }
}

impl HandleMessage for AssembleError {
    fn handle_message(&self) -> String {
        match self {
            Self::Place(e) => format!("placement failed: {}", e.handle_message()),
            Self::React {
                edge: Some(edge),
                error,
            } => format!(
                "edge {edge} could not be linked: {}",
                error.handle_message()
            ),
            Self::React { edge: None, error } => {
                format!("an edge could not be linked: {}", error.handle_message())
            }
            other => other.to_string(),
        }
    }
}

impl HandleMessage for FragLibraryError {
    fn handle_message(&self) -> String {
        let bead = |id| node_to_u64(id);
        match self {
            Self::AmbiguousAssignment { bead: b } => format!(
                "coarse bead {} admits two template-bead assignments in its unit",
                bead(*b)
            ),
            Self::Unmapped { bead: b } => format!(
                "coarse bead {} is covered by no template occurrence",
                bead(*b)
            ),
            Self::Ambiguous { bead: b } => format!(
                "coarse bead {} and every other uncovered bead have several occurrences",
                bead(*b)
            ),
            Self::Unmatchable {
                a,
                b,
                template_a,
                template_b,
            } => format!(
                "coarse bond {}-{} between units '{template_a}' and '{template_b}' has no \
                 free pair of accepting ports",
                bead(*a),
                bead(*b)
            ),
            other => other.to_string(),
        }
    }
}

/// Where the Python adapters of one consumer keep the exception they turned
/// into a Rust error, so the binder surfacing that error can chain it.
///
/// A Rust error carries only the message; the slot keeps the original
/// exception with its type, traceback and `__cause__`. A consumer clears it
/// before a call and takes it when the call fails. A component shared by two
/// calls running at once on two threads shares one slot, so each may chain
/// the other's exception; the message is always its own.
#[derive(Clone, Default)]
struct ErrorSlot(Arc<Mutex<Option<PyErr>>>);

impl ErrorSlot {
    fn lock(&self) -> MutexGuard<'_, Option<PyErr>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Keep `err` as the cause of the Rust error its adapter returns.
    fn keep(&self, err: PyErr) {
        *self.lock() = Some(err);
    }

    /// Drop a kept exception left over from an earlier call.
    fn clear(&self) {
        self.lock().take();
    }

    /// The Python error for a call that failed with `message`: a kept
    /// `BaseException` that is not an `Exception` (`KeyboardInterrupt`,
    /// `SystemExit`) unchanged, else `ValueError(message)` whose
    /// `__cause__` is the kept exception, if any.
    fn raise(&self, py: Python<'_>, message: String) -> PyErr {
        let kept = self.lock().take();
        match kept {
            Some(cause) if !cause.is_instance_of::<PyException>(py) => cause,
            cause => {
                let err = PyValueError::new_err(message);
                err.set_cause(py, cause);
                err
            }
        }
    }
}

/// `units` as the `(N,)` int64 array every Python `*_many` hook receives.
fn units_to_py<'py>(py: Python<'py>, units: &[usize]) -> Bound<'py, PyArray1<i64>> {
    Array1::from_iter(units.iter().map(|&u| u as i64)).into_pyarray(py)
}

/// One motion as `(rotation (3, 3), translation (3,))` numpy arrays.
type RigidArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);

/// Motions as `(rotations (N, 3, 3), translations (N, 3))` numpy arrays.
type RigidsArrays<'py> = (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray2<f64>>);

/// One motion as `(rotation (3, 3), translation (3,))`.
fn rigid_to_py<'py>(py: Python<'py>, rigid: &Rigid) -> RigidArrays<'py> {
    (
        Array2::from_shape_fn((3, 3), |(i, j)| rigid.rotation[i][j]).into_pyarray(py),
        Array1::from(rigid.translation.to_vec()).into_pyarray(py),
    )
}

/// Motions as `(rotations (N, 3, 3), translations (N, 3))`.
fn rigids_to_py<'py>(py: Python<'py>, rigids: &[Rigid]) -> RigidsArrays<'py> {
    let n = rigids.len();
    (
        Array3::from_shape_fn((n, 3, 3), |(c, i, j)| rigids[c].rotation[i][j]).into_pyarray(py),
        Array2::from_shape_fn((n, 3), |(c, i)| rigids[c].translation[i]).into_pyarray(py),
    )
}

/// The two float arrays of a hook's `(rotation(s), translation(s))` return.
type MotionArrays<'py> = (
    PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    PyArrayLikeDyn<'py, f64, AllowTypeChange>,
);

fn motion_arrays<'py>(returned: &Bound<'py, PyAny>, hook: &str) -> PyResult<MotionArrays<'py>> {
    returned.extract::<MotionArrays<'py>>().map_err(|err| {
        PyTypeError::new_err(format!(
            "{hook} must return a (rotations, translations) tuple of float arrays: {err}"
        ))
    })
}

/// Read a per-unit hook's `(rotation (3, 3), translation (3,))`.
fn rigid_from_py(returned: &Bound<'_, PyAny>, hook: &str) -> PyResult<Rigid> {
    let (rotation, translation) = motion_arrays(returned, hook)?;
    let (r, t) = (rotation.as_array(), translation.as_array());
    if r.shape() != [3, 3] || t.shape() != [3] {
        return Err(PyValueError::new_err(format!(
            "{hook} must return rotation (3, 3) and translation (3,), got {:?} and {:?}",
            r.shape(),
            t.shape()
        )));
    }
    Ok(Rigid {
        rotation: std::array::from_fn(|i| std::array::from_fn(|j| r[[i, j]])),
        translation: std::array::from_fn(|i| t[[i]]),
    })
}

/// Read a batched hook's `(rotations (n, 3, 3), translations (n, 3))`.
fn rigids_from_py(returned: &Bound<'_, PyAny>, hook: &str, n: usize) -> PyResult<Vec<Rigid>> {
    let (rotations, translations) = motion_arrays(returned, hook)?;
    let rigids = rigids_from_arrays(&rotations, &translations)?;
    if rigids.len() != n {
        return Err(PyValueError::new_err(format!(
            "{hook} returned {} motions for {n} units",
            rigids.len()
        )));
    }
    Ok(rigids)
}

// ---------------------------------------------------------------------------
// FragLibrary
// ---------------------------------------------------------------------------

/// Named fragment templates, each keeping its own bead labels.
///
/// A template's atoms carry ``bead`` (template-local bead index ``0..k``) and
/// ``bead_type`` (that bead's label). Templates are stored as copies.
#[pyclass(module = "molrs.builder", name = "FragLibrary")]
pub struct PyFragLibrary {
    inner: FragLibrary,
}

#[pymethods]
impl PyFragLibrary {
    #[new]
    fn new() -> Self {
        Self {
            inner: FragLibrary::new(),
        }
    }

    /// Store a copy of ``template`` under ``name``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the name is taken or the template is invalid (no atom, an atom
    ///     without ``bead`` / ``bead_type``, non-contiguous beads, one bead
    ///     with two labels, a disconnected bead pattern, an unreadable port).
    fn insert(&mut self, name: &str, template: PyRef<'_, PyFragment>) -> PyResult<()> {
        self.inner
            .insert(name, template.core().clone())
            .map_err(|e| PyValueError::new_err(e.handle_message()))
    }

    /// A copy of the template stored under ``name``, or ``None``.
    fn get(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyFragment>>> {
        self.inner
            .get(name)
            .map(|template| PyFragment::from_core(py, template.clone()))
            .transpose()
    }

    /// Template names, in lexicographic order.
    fn names(&self) -> Vec<String> {
        self.inner.names().map(str::to_owned).collect()
    }

    /// Cover ``graph`` with template occurrences under ``rules`` and pair
    /// their ports.
    ///
    /// Parameters
    /// ----------
    /// graph : CoarseGrain
    /// rules : list[tuple[str, str]]
    ///     ``(coarse type, template label)`` pairs.
    ///
    /// Returns
    /// -------
    /// Mapping
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a rule is not a ``(str, str)`` pair.
    /// ValueError
    ///     If the rules are empty, license no template, or the cover is
    ///     missing, ambiguous or unmatchable.
    fn map(
        &self,
        py: Python<'_>,
        graph: PyRef<'_, PyCoarseGrain>,
        rules: Vec<(String, String)>,
    ) -> PyResult<PyMapping> {
        let rules: Vec<(&str, &str)> = rules
            .iter()
            .map(|(coarse, label)| (coarse.as_str(), label.as_str()))
            .collect();
        let graph = graph.core();
        py.detach(|| self.inner.map(graph, &rules))
            .map(|inner| PyMapping { inner })
            .map_err(|e| PyValueError::new_err(e.handle_message()))
    }
}

// ---------------------------------------------------------------------------
// Component bases: native (Arc) or Python (the instance's own methods)
// ---------------------------------------------------------------------------

/// What a component base pyclass dispatches to.
enum Implementation<T: ?Sized> {
    /// A native Rust component. A Rust consumer shares it through the
    /// `Shared*` forwarder of its trait.
    Native(Arc<T>),
    /// A Python subclass: its methods are the hooks. A Rust consumer reaches
    /// it through the `Py*Adapter` of its trait.
    Python,
}

fn not_implemented(method: &str) -> PyErr {
    PyNotImplementedError::new_err(format!(
        "{method} must be implemented by a concrete subclass"
    ))
}

// ---- Placer ----------------------------------------------------------------

/// Forwards **every** [`Placer`] method to a shared native placer, so a
/// native batched override stays in force.
struct SharedPlacer(Arc<dyn Placer>);

impl Placer for SharedPlacer {
    fn place(&self, unit: usize, name: &str, fragment: &Fragment) -> Result<Rigid, PlaceError> {
        self.0.place(unit, name, fragment)
    }

    fn place_many(
        &self,
        units: &[usize],
        name: &str,
        fragment: &Fragment,
    ) -> Result<Vec<Rigid>, PlaceError> {
        self.0.place_many(units, name, fragment)
    }
}

/// A Python `Placer` subclass as a Rust [`Placer`]. Re-enters Python once
/// per template group: one template copy, one ``place_many`` call. An
/// exception or a wrong shape becomes [`PlaceError::Other`] with the message,
/// and the exception itself is kept in `errors`.
struct PyPlacerAdapter {
    instance: Py<PyAny>,
    errors: ErrorSlot,
}

impl Placer for PyPlacerAdapter {
    fn place(&self, unit: usize, name: &str, fragment: &Fragment) -> Result<Rigid, PlaceError> {
        self.place_many(&[unit], name, fragment)?
            .into_iter()
            .next()
            .ok_or_else(|| PlaceError::Other("place_many returned no motion".to_owned()))
    }

    fn place_many(
        &self,
        units: &[usize],
        name: &str,
        fragment: &Fragment,
    ) -> Result<Vec<Rigid>, PlaceError> {
        Python::attach(|py| {
            let call = || -> PyResult<Vec<Rigid>> {
                let template = PyFragment::from_core(py, fragment.clone())?;
                let returned = self.instance.bind(py).call_method1(
                    intern!(py, "place_many"),
                    (units_to_py(py, units), name, template),
                )?;
                rigids_from_py(&returned, "place_many", units.len())
            };
            call().map_err(|err| {
                let message = err.to_string();
                self.errors.keep(err);
                PlaceError::Other(message)
            })
        })
    }
}

/// Where each unit of a coarse-grained sequence goes, as a rigid motion of
/// its template. Subclassable.
///
/// A Python subclass implements :meth:`place` or :meth:`place_many` (whose
/// default loops over :meth:`place`). The :class:`Assembler` calls
/// :meth:`place_many` once per template group with a fresh copy of the
/// template; mutating that copy never reaches the library. An exception or a
/// wrong shape surfaces from :meth:`Assembler.assemble` as ``ValueError``
/// carrying the message, raised ``from`` the exception; a
/// ``KeyboardInterrupt`` or ``SystemExit`` propagates unchanged.
///
/// Implementation note: the Rust ``Assembler`` calls a placer through the
/// Rust ``Placer`` trait, so a Python subclass crosses through an explicit
/// trait adaptor (``PyPlacerAdapter``) that enters Python once per
/// ``place_many`` batch. ``molrs.ff.typifier.Typifier`` is the model for this
/// base's native-or-Python two-state dispatch only: its ``typify`` calls the
/// subclass's ``match`` directly and wraps no trait adaptor.
#[pyclass(module = "molrs.builder", name = "Placer", subclass, frozen)]
pub struct PyPlacer {
    state: Implementation<dyn Placer>,
    /// Where a Python adapter this placer reaches keeps its exception: its
    /// own (a Python subclass) or its orienter's (a native `TracePlacer`).
    errors: ErrorSlot,
}

impl PyPlacer {
    fn native(placer: Arc<dyn Placer>, errors: ErrorSlot) -> Self {
        Self {
            state: Implementation::Native(placer),
            errors,
        }
    }

    /// This placer as the `Box<dyn Placer>` a Rust consumer takes.
    fn boxed(slf: &Bound<'_, Self>) -> Box<dyn Placer> {
        let this = slf.get();
        match &this.state {
            Implementation::Native(placer) => Box::new(SharedPlacer(Arc::clone(placer))),
            Implementation::Python => Box::new(PyPlacerAdapter {
                instance: slf.clone().into_any().unbind(),
                errors: this.errors.clone(),
            }),
        }
    }
}

#[pymethods]
impl PyPlacer {
    /// A Python-subclass base. Accepts and ignores any arguments, so a
    /// subclass's own ``__init__`` signature is its own.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            state: Implementation::Python,
            errors: ErrorSlot::default(),
        }
    }

    /// The motion placing unit ``unit``, an instance of template ``name``.
    ///
    /// Returns
    /// -------
    /// tuple[ndarray, ndarray]
    ///     ``rotation (3, 3)`` and ``translation (3,)``.
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     On the base, when a subclass does not implement it.
    /// ValueError
    ///     If a native placer cannot place the unit.
    fn place<'py>(
        &self,
        py: Python<'py>,
        unit: usize,
        name: &str,
        template: PyRef<'py, PyFragment>,
    ) -> PyResult<RigidArrays<'py>> {
        match &self.state {
            Implementation::Native(placer) => {
                self.errors.clear();
                let rigid = placer
                    .place(unit, name, template.core())
                    .map_err(|e| self.errors.raise(py, e.handle_message()))?;
                Ok(rigid_to_py(py, &rigid))
            }
            Implementation::Python => Err(not_implemented("Placer.place")),
        }
    }

    /// :meth:`place` for each of ``units``, all instances of template
    /// ``name``. The Python default loops over :meth:`place`.
    ///
    /// Returns
    /// -------
    /// tuple[ndarray, ndarray]
    ///     ``rotations (N, 3, 3)`` and ``translations (N, 3)``.
    fn place_many<'py>(
        slf: &Bound<'py, Self>,
        units: Vec<usize>,
        name: &str,
        template: &Bound<'py, PyFragment>,
    ) -> PyResult<RigidsArrays<'py>> {
        let py = slf.py();
        let this = slf.get();
        let rigids = match &this.state {
            Implementation::Native(placer) => {
                this.errors.clear();
                placer
                    .place_many(&units, name, template.borrow().core())
                    .map_err(|e| this.errors.raise(py, e.handle_message()))?
            }
            Implementation::Python => units
                .iter()
                .map(|&unit| {
                    let returned =
                        slf.call_method1(intern!(py, "place"), (unit, name, template))?;
                    rigid_from_py(&returned, "place")
                })
                .collect::<PyResult<Vec<_>>>()?,
        };
        Ok(rigids_to_py(py, &rigids))
    }
}

/// Places unit ``i`` by superposing the template's bead reference points
/// (mass-weighted bead centroids) onto the trace's unit ``i`` points,
/// completing an under-determined fit with its orienter (default
/// :class:`NullOrienter`).
///
/// Parameters
/// ----------
/// trace : Trace
/// seq : list[str]
///     Template name of each unit; a unit is only placed with that template.
///
/// Raises
/// ------
/// ValueError
///     If ``seq`` and the trace differ in unit count.
#[pyclass(module = "molrs.builder", name = "TracePlacer", extends = PyPlacer, frozen)]
pub struct PyTracePlacer {
    // The placer the base `PyPlacer` also holds, typed, so `with_orienter`
    // can read its trace and sequence.
    inner: Arc<TracePlacer>,
}

impl PyTracePlacer {
    /// `placer` as a new `TracePlacer` object whose adapters keep their
    /// exceptions in `errors`.
    fn create(placer: TracePlacer, errors: ErrorSlot) -> PyClassInitializer<Self> {
        let inner = Arc::new(placer);
        PyClassInitializer::from(PyPlacer::native(inner.clone(), errors))
            .add_subclass(Self { inner })
    }
}

#[pymethods]
impl PyTracePlacer {
    #[new]
    fn new(trace: &Bound<'_, PyTrace>, seq: Vec<String>) -> PyResult<PyClassInitializer<Self>> {
        let placer = TracePlacer::new(trace.get().inner.clone(), seq)
            .map_err(|e| PyValueError::new_err(e.handle_message()))?;
        Ok(Self::create(placer, ErrorSlot::default()))
    }

    /// A new placer over the same trace and sequence that completes
    /// under-determined fits with ``orienter``. This placer is unchanged.
    ///
    /// A Python orienter receives one :meth:`Orienter.orient_many` call per
    /// :meth:`place_many` batch.
    fn with_orienter(
        &self,
        py: Python<'_>,
        orienter: &Bound<'_, PyOrienter>,
    ) -> PyResult<Py<Self>> {
        let errors = ErrorSlot::default();
        let placer = TracePlacer::new(self.inner.trace().clone(), self.inner.seq().to_vec())
            .map_err(|e| PyValueError::new_err(e.handle_message()))?
            .with_orienter(PyOrienter::boxed(orienter, &errors));
        Py::new(py, Self::create(placer, errors))
    }
}

// ---- Orienter --------------------------------------------------------------

/// Forwards **every** [`Orienter`] method to a shared native orienter, so a
/// native batched override stays in force.
struct SharedOrienter(Arc<dyn Orienter>);

impl Orienter for SharedOrienter {
    fn orient(
        &self,
        unit: usize,
        fragment: &Fragment,
        fit: &Fit,
        hint: Option<Vec3>,
    ) -> Result<Rigid, OrientError> {
        self.0.orient(unit, fragment, fit, hint)
    }

    fn orient_many(
        &self,
        units: &[usize],
        fragment: &Fragment,
        fits: &[Fit],
        hints: &[Option<Vec3>],
    ) -> Result<Vec<Rigid>, OrientError> {
        self.0.orient_many(units, fragment, fits, hints)
    }
}

/// A Python `Orienter` subclass as a Rust [`Orienter`]. Re-enters Python once
/// per batch: one template copy, one ``orient_many`` call. An exception or a
/// wrong shape becomes [`OrientError::Other`] with the message, and the
/// exception itself is kept in `errors`.
///
/// **Hints are all-or-none.** They cross as one `(N, 3)` array, or `None`
/// when no unit of the batch has one; a batch where only some units have a
/// hint is refused with [`OrientError::Other`], since one array cannot say
/// which rows are missing. A [`Trace`](molrs::spatial::Trace) gives every
/// unit a hint or none, so a `TracePlacer` batch never mixes them.
struct PyOrienterAdapter {
    instance: Py<PyAny>,
    errors: ErrorSlot,
}

impl Orienter for PyOrienterAdapter {
    fn orient(
        &self,
        unit: usize,
        fragment: &Fragment,
        fit: &Fit,
        hint: Option<Vec3>,
    ) -> Result<Rigid, OrientError> {
        self.orient_many(&[unit], fragment, std::slice::from_ref(fit), &[hint])?
            .into_iter()
            .next()
            .ok_or_else(|| OrientError::Other("orient_many returned no motion".to_owned()))
    }

    fn orient_many(
        &self,
        units: &[usize],
        fragment: &Fragment,
        fits: &[Fit],
        hints: &[Option<Vec3>],
    ) -> Result<Vec<Rigid>, OrientError> {
        if fits.len() != units.len() || hints.len() != units.len() {
            return Err(OrientError::Other(format!(
                "batch of {} units has {} fits and {} hints",
                units.len(),
                fits.len(),
                hints.len()
            )));
        }
        let hints: Option<Vec<Vec3>> = if hints.iter().all(Option::is_none) {
            None
        } else {
            Some(
                units
                    .iter()
                    .zip(hints)
                    .map(|(&unit, hint)| {
                        hint.ok_or_else(|| {
                            OrientError::Other(format!(
                                "unit {unit} has no hint but others of its batch do; a \
                                 Python orienter takes hints for every unit or none"
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()?,
            )
        };
        Python::attach(|py| {
            let call = || -> PyResult<Vec<Rigid>> {
                let template = PyFragment::from_core(py, fragment.clone())?;
                let fits = PyList::new(py, fits.iter().map(|&fit| PyFit::from_core(fit)))?;
                let hints = hints.as_ref().map(|hints| {
                    Array2::from_shape_fn((hints.len(), 3), |(n, a)| hints[n][a]).into_pyarray(py)
                });
                let returned = self.instance.bind(py).call_method1(
                    intern!(py, "orient_many"),
                    (units_to_py(py, units), template, fits, hints),
                )?;
                rigids_from_py(&returned, "orient_many", units.len())
            };
            call().map_err(|err| {
                let message = err.to_string();
                self.errors.keep(err);
                OrientError::Other(message)
            })
        })
    }
}

/// Completes a superposition fit the trace leaves under-determined (a spin
/// about its axis, or any rotation) with one member of its optimal family.
/// Subclassable.
///
/// A Python subclass implements :meth:`orient` or :meth:`orient_many` (whose
/// default loops over :meth:`orient`). A :class:`TracePlacer` calls
/// :meth:`orient_many` once per ``place_many`` batch with a fresh copy of the
/// template. An exception or a wrong shape surfaces from
/// :meth:`Assembler.assemble` as ``ValueError`` carrying the message, raised
/// ``from`` the exception; a ``KeyboardInterrupt`` or ``SystemExit``
/// propagates unchanged.
///
/// **Hints are all-or-none.** :meth:`orient_many` receives ``hints`` as one
/// ``(N, 3)`` array, or ``None`` when no unit of the batch has one; a batch
/// where only some units have a hint is refused. A :class:`~molrs.Trace`
/// carries hints for every unit or for none.
///
/// Implementation note: the Rust ``TracePlacer`` calls an orienter through
/// the Rust ``Orienter`` trait, so a Python subclass crosses through an
/// explicit trait adaptor (``PyOrienterAdapter``) that enters Python once per
/// ``orient_many`` batch. ``molrs.ff.typifier.Typifier`` is the model for
/// this base's native-or-Python two-state dispatch only: its ``typify`` calls
/// the subclass's ``match`` directly and wraps no trait adaptor.
#[pyclass(module = "molrs.builder", name = "Orienter", subclass, frozen)]
pub struct PyOrienter {
    state: Implementation<dyn Orienter>,
}

impl PyOrienter {
    fn native(orienter: impl Orienter + 'static) -> Self {
        Self {
            state: Implementation::Native(Arc::new(orienter)),
        }
    }

    /// This orienter as the `Box<dyn Orienter>` a Rust consumer takes; a
    /// Python adapter keeps its exception in the consumer's `errors`.
    fn boxed(slf: &Bound<'_, Self>, errors: &ErrorSlot) -> Box<dyn Orienter> {
        match &slf.get().state {
            Implementation::Native(orienter) => Box::new(SharedOrienter(Arc::clone(orienter))),
            Implementation::Python => Box::new(PyOrienterAdapter {
                instance: slf.clone().into_any().unbind(),
                errors: errors.clone(),
            }),
        }
    }
}

#[pymethods]
impl PyOrienter {
    /// A Python-subclass base. Accepts and ignores any arguments, so a
    /// subclass's own ``__init__`` signature is its own.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            state: Implementation::Python,
        }
    }

    /// Orient unit ``unit``, an instance of ``template``, given its ``fit``
    /// and its optional direction ``hint`` (shape ``(3,)``).
    ///
    /// Returns
    /// -------
    /// tuple[ndarray, ndarray]
    ///     ``rotation (3, 3)`` and ``translation (3,)``.
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     On the base, when a subclass does not implement it.
    /// ValueError
    ///     If a native orienter cannot complete the fit.
    fn orient<'py>(
        &self,
        py: Python<'py>,
        unit: usize,
        template: PyRef<'py, PyFragment>,
        fit: &Bound<'py, PyFit>,
        hint: Option<PyArrayLikeDyn<'py, f64, AllowTypeChange>>,
    ) -> PyResult<RigidArrays<'py>> {
        match &self.state {
            Implementation::Native(orienter) => {
                let hint = match hint {
                    None => None,
                    Some(hint) => {
                        let h = hint.as_array();
                        if h.shape() != [3] {
                            return Err(PyValueError::new_err(format!(
                                "hint must have shape (3,), got {:?}",
                                h.shape()
                            )));
                        }
                        Some([h[[0]], h[[1]], h[[2]]])
                    }
                };
                let rigid = orienter
                    .orient(unit, template.core(), &fit.get().inner, hint)
                    .map_err(py_value_err)?;
                Ok(rigid_to_py(py, &rigid))
            }
            Implementation::Python => Err(not_implemented("Orienter.orient")),
        }
    }

    /// :meth:`orient` for each ``units[n]`` with ``fits[n]`` and
    /// ``hints[n]``. The Python default loops over :meth:`orient`.
    ///
    /// Parameters
    /// ----------
    /// units : array_like of int
    /// template : Fragment
    /// fits : list[Fit]
    /// hints : ndarray, shape (N, 3), or None
    ///
    /// Returns
    /// -------
    /// tuple[ndarray, ndarray]
    ///     ``rotations (N, 3, 3)`` and ``translations (N, 3)``.
    fn orient_many<'py>(
        slf: &Bound<'py, Self>,
        units: Vec<usize>,
        template: &Bound<'py, PyFragment>,
        fits: Vec<Bound<'py, PyFit>>,
        hints: Option<PyArrayLikeDyn<'py, f64, AllowTypeChange>>,
    ) -> PyResult<RigidsArrays<'py>> {
        let py = slf.py();
        let hints: Vec<Option<Vec3>> = match hints {
            None => vec![None; units.len()],
            Some(hints) => points_from_array(&hints, "hints")?
                .into_iter()
                .map(Some)
                .collect(),
        };
        let rigids = match &slf.get().state {
            Implementation::Native(orienter) => {
                let fits: Vec<Fit> = fits.iter().map(|fit| fit.get().inner).collect();
                orienter
                    .orient_many(&units, template.borrow().core(), &fits, &hints)
                    .map_err(py_value_err)?
            }
            Implementation::Python => {
                if fits.len() != units.len() || hints.len() != units.len() {
                    return Err(PyValueError::new_err(format!(
                        "batch of {} units has {} fits and {} hints",
                        units.len(),
                        fits.len(),
                        hints.len()
                    )));
                }
                units
                    .iter()
                    .zip(&fits)
                    .zip(&hints)
                    .map(|((&unit, fit), hint)| {
                        let hint = hint.map(|h| Array1::from(h.to_vec()).into_pyarray(py));
                        let returned =
                            slf.call_method1(intern!(py, "orient"), (unit, template, fit, hint))?;
                        rigid_from_py(&returned, "orient")
                    })
                    .collect::<PyResult<Vec<_>>>()?
            }
        };
        Ok(rigids_to_py(py, &rigids))
    }
}

/// Leaves every fit as the superposition returned it.
#[pyclass(module = "molrs.builder", name = "NullOrienter", extends = PyOrienter, frozen)]
pub struct PyNullOrienter;

#[pymethods]
impl PyNullOrienter {
    #[new]
    fn new() -> (Self, PyOrienter) {
        (Self, PyOrienter::native(NullOrienter))
    }
}

/// A uniform member of each fit's family, a pure function of
/// ``(seed, unit)``: a uniform spin angle, or a Haar-random rotation for a
/// free fit (Shoemake 1992).
#[pyclass(module = "molrs.builder", name = "RandomOrienter", extends = PyOrienter, frozen)]
pub struct PyRandomOrienter;

#[pymethods]
impl PyRandomOrienter {
    #[new]
    fn new(seed: u64) -> (Self, PyOrienter) {
        (Self, PyOrienter::native(RandomOrienter::new(seed)))
    }
}

/// Turns a body axis of the template onto each unit's direction hint.
///
/// Parameters
/// ----------
/// axis : {"principal", "dipole"}
///     The top gyration eigenvector, or the charge dipole.
///
/// Raises
/// ------
/// ValueError
///     If ``axis`` is neither.
#[pyclass(module = "molrs.builder", name = "HintOrienter", extends = PyOrienter, frozen)]
pub struct PyHintOrienter;

#[pymethods]
impl PyHintOrienter {
    #[new]
    #[pyo3(signature = (axis="principal"))]
    fn new(axis: &str) -> PyResult<(Self, PyOrienter)> {
        let axis = match axis {
            "principal" => BodyAxis::Principal,
            "dipole" => BodyAxis::Dipole,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown body axis {other:?}; expected 'principal' or 'dipole'"
                )));
            }
        };
        Ok((Self, PyOrienter::native(HintOrienter::new(axis))))
    }
}

// ---- Reacter ---------------------------------------------------------------

/// Forwards **every** [`Reacter`] method to a shared native reacter, so a
/// native batched override stays in force.
struct SharedReacter(Arc<dyn Reacter>);

impl Reacter for SharedReacter {
    fn link(&self, world: &mut Fragment, a: PortId, b: PortId) -> Result<BondId, ReactError> {
        self.0.link(world, a, b)
    }

    fn link_many(
        &self,
        world: &mut Fragment,
        pairs: &[(PortId, PortId)],
    ) -> Result<Vec<BondId>, PairError> {
        self.0.link_many(world, pairs)
    }
}

/// A Python `Reacter` subclass as a Rust [`Reacter`]. Re-enters Python once
/// per batch: the world is moved (`std::mem::take`, no atom copied) into a
/// fresh `molrs.Fragment`, handed to ``link_many``, and moved back out
/// whatever the call did, leaving that Python object an empty fragment.
///
/// A failure (an exception, or a return that is not one bond handle per
/// pair) becomes [`PairError`] with [`ReactError::Other`] carrying the
/// message, and the exception itself is kept in `errors`. The pair is the
/// exception's [`PAIR_TAG`] attribute when it names an index into this batch
/// (`Some(i)`), else `None`: an untagged exception does not say which pair
/// failed, and no index is made up for it.
struct PyReacterAdapter {
    instance: Py<PyAny>,
    errors: ErrorSlot,
}

/// The attribute the base `Reacter.link_many` sets on an exception raised by
/// `link` for pair `i` (the index into the `pairs` it was given).
const PAIR_TAG: &str = "molrs_pair";

impl Reacter for PyReacterAdapter {
    fn link(&self, world: &mut Fragment, a: PortId, b: PortId) -> Result<BondId, ReactError> {
        self.link_many(world, &[(a, b)])
            .map_err(|failed| failed.error)?
            .into_iter()
            .next()
            .ok_or_else(|| ReactError::Other("link_many returned no bond".to_owned()))
    }

    fn link_many(
        &self,
        world: &mut Fragment,
        pairs: &[(PortId, PortId)],
    ) -> Result<Vec<BondId>, PairError> {
        Python::attach(|py| {
            let mut call = || -> PyResult<Vec<BondId>> {
                // The empty Python object is made first, so a failure to make
                // it cannot lose the world.
                let lent = PyFragment::from_core(py, Fragment::new())?;
                // Borrowed before the world is taken, so a refused borrow
                // leaves the world where it was.
                std::mem::swap(lent.try_borrow_mut(py)?.core_mut(), world);
                let handles: Vec<(u64, u64)> = pairs
                    .iter()
                    .map(|&(a, b)| (relation_to_u64(a), relation_to_u64(b)))
                    .collect();
                let returned = self
                    .instance
                    .bind(py)
                    .call_method1(intern!(py, "link_many"), (&lent, handles));
                // The call has returned, so a borrow still held on `lent` was
                // taken by another thread that has released the GIL. It ends;
                // wait for it rather than lose the world.
                loop {
                    if let Ok(mut back) = lent.try_borrow_mut(py) {
                        *world = std::mem::take(back.core_mut());
                        break;
                    }
                    py.detach(std::thread::yield_now);
                }
                let bonds: Vec<u64> = returned?.extract()?;
                if bonds.len() != pairs.len() {
                    return Err(PyValueError::new_err(format!(
                        "link_many returned {} bonds for {} pairs",
                        bonds.len(),
                        pairs.len()
                    )));
                }
                Ok(bonds.into_iter().map(relation_from_u64).collect())
            };
            call().map_err(|err| {
                let pair = err
                    .value(py)
                    .getattr(intern!(py, PAIR_TAG))
                    .and_then(|tag| tag.extract::<usize>())
                    .ok()
                    .filter(|&i| i < pairs.len());
                let message = err.to_string();
                self.errors.keep(err);
                PairError {
                    pair,
                    error: ReactError::Other(message),
                }
            })
        })
    }
}

/// Joins two ports of one world fragment into a bond. Subclassable.
///
/// A Python subclass implements :meth:`link` or :meth:`link_many` (whose
/// default loops over :meth:`link`). Port and bond handles cross as ``int``.
/// The :class:`Assembler` calls :meth:`link_many` once per assembly.
///
/// **Which edge failed.** The default :meth:`link_many` sets
/// ``exc.molrs_pair = i`` on an exception raised by :meth:`link` for pair
/// ``i`` and re-raises it, and :meth:`Assembler.assemble` names edge ``i`` in
/// its ``ValueError``. An override that raises without that attribute is
/// reported as failing on "an edge"; an override that knows the pair may set
/// ``molrs_pair`` itself.
///
/// **The ``world`` handed to a Python** :meth:`link_many` **is valid only
/// during the call.** The world is moved into it, not copied, and moved back
/// out when the call returns: afterwards that object is an empty fragment
/// (``n_atoms == 0``), never a stale view of the assembled world. An
/// exception surfaces from :meth:`Assembler.assemble` as ``ValueError``
/// carrying the message, raised ``from`` the exception; a
/// ``KeyboardInterrupt`` or ``SystemExit`` propagates unchanged.
///
/// Implementation note: the Rust ``Assembler`` calls a reacter through the
/// Rust ``Reacter`` trait, so a Python subclass crosses through an explicit
/// trait adaptor (``PyReacterAdapter``) that enters Python once per
/// ``link_many`` batch. ``molrs.ff.typifier.Typifier`` is the model for this
/// base's native-or-Python two-state dispatch only: its ``typify`` calls the
/// subclass's ``match`` directly and wraps no trait adaptor.
#[pyclass(module = "molrs.builder", name = "Reacter", subclass, frozen)]
pub struct PyReacter {
    state: Implementation<dyn Reacter>,
    /// Where this reacter's Python adapter keeps its exception.
    errors: ErrorSlot,
}

impl PyReacter {
    fn native(reacter: impl Reacter + 'static) -> Self {
        Self {
            state: Implementation::Native(Arc::new(reacter)),
            errors: ErrorSlot::default(),
        }
    }

    /// This reacter as the `Box<dyn Reacter>` a Rust consumer takes.
    fn boxed(slf: &Bound<'_, Self>) -> Box<dyn Reacter> {
        let this = slf.get();
        match &this.state {
            Implementation::Native(reacter) => Box::new(SharedReacter(Arc::clone(reacter))),
            Implementation::Python => Box::new(PyReacterAdapter {
                instance: slf.clone().into_any().unbind(),
                errors: this.errors.clone(),
            }),
        }
    }
}

#[pymethods]
impl PyReacter {
    /// A Python-subclass base. Accepts and ignores any arguments, so a
    /// subclass's own ``__init__`` signature is its own.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            state: Implementation::Python,
            errors: ErrorSlot::default(),
        }
    }

    /// Join port ``a`` to port ``b`` of ``world`` in place.
    ///
    /// Returns
    /// -------
    /// int
    ///     The new anchor–anchor bond handle.
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     On the base, when a subclass does not implement it.
    /// ValueError
    ///     If a native reacter refuses the pair.
    fn link(&self, mut world: PyRefMut<'_, PyFragment>, a: u64, b: u64) -> PyResult<u64> {
        match &self.state {
            Implementation::Native(reacter) => reacter
                .link(world.core_mut(), relation_from_u64(a), relation_from_u64(b))
                .map(relation_to_u64)
                .map_err(|e| PyValueError::new_err(e.handle_message())),
            Implementation::Python => Err(not_implemented("Reacter.link")),
        }
    }

    /// Join every ``(a, b)`` port pair of ``world`` in order. The Python
    /// default loops over :meth:`link`.
    ///
    /// Returns
    /// -------
    /// list[int]
    ///     The new bond handles, in pair order.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a native reacter refuses a pair (the message names its index;
    ///     the pairs before it stay linked).
    ///
    /// On a Python subclass, an exception from :meth:`link` for pair ``i``
    /// is re-raised with ``molrs_pair = i`` set on it.
    fn link_many(
        slf: &Bound<'_, Self>,
        world: &Bound<'_, PyFragment>,
        pairs: Vec<(u64, u64)>,
    ) -> PyResult<Vec<u64>> {
        let py = slf.py();
        match &slf.get().state {
            Implementation::Native(reacter) => {
                let pairs: Vec<(PortId, PortId)> = pairs
                    .iter()
                    .map(|&(a, b)| (relation_from_u64(a), relation_from_u64(b)))
                    .collect();
                reacter
                    .link_many(world.borrow_mut().core_mut(), &pairs)
                    .map(|bonds| bonds.into_iter().map(relation_to_u64).collect())
                    .map_err(|e| PyValueError::new_err(e.handle_message()))
            }
            Implementation::Python => pairs
                .into_iter()
                .enumerate()
                .map(|(i, (a, b))| {
                    slf.call_method1(intern!(py, "link"), (world, a, b))
                        .and_then(|bond| bond.extract::<u64>())
                        .inspect_err(|err| {
                            // An exception that refuses attributes stays
                            // untagged and is reported as "an edge".
                            let _ = err.value(py).setattr(intern!(py, PAIR_TAG), i);
                        })
                })
                .collect(),
        }
    }
}

/// The port-driven reacter: deletes each port's handle branch, folds its
/// charge onto the anchor, and bonds the two anchors with the port order.
#[pyclass(module = "molrs.builder", name = "PortReacter", extends = PyReacter, frozen)]
pub struct PyPortReacter;

#[pymethods]
impl PyPortReacter {
    #[new]
    fn new() -> (Self, PyReacter) {
        (Self, PyReacter::native(PortReacter))
    }
}

// ---------------------------------------------------------------------------
// Finalizer / Assembler
// ---------------------------------------------------------------------------

/// Completes an assembled molecule's angles, dihedrals and, when
/// ``impropers`` is true, impropers from its bond graph. Idempotent.
#[pyclass(module = "molrs.builder", name = "Finalizer", frozen)]
pub struct PyFinalizer {
    inner: Finalizer,
}

#[pymethods]
impl PyFinalizer {
    #[new]
    #[pyo3(signature = (impropers=false))]
    fn new(impropers: bool) -> Self {
        Self {
            inner: Finalizer::new(impropers),
        }
    }

    /// Add every missing angle, dihedral (and improper, when configured) of
    /// ``mol`` in place.
    ///
    /// Returns
    /// -------
    /// tuple[int, int, int]
    ///     ``(angles, dihedrals, impropers)`` added.
    fn finalize(&self, mol: &Bound<'_, PyAtomistic>) -> PyResult<(usize, usize, usize)> {
        self.inner
            .finalize(mol.borrow_mut().core_mut())
            .map_err(molrs_error_to_pyerr)
    }
}

/// Builds one placed, linked world :class:`~molrs.Fragment` from a
/// :class:`~molrs.FragGraph`.
///
/// Parameters
/// ----------
/// library : FragLibrary
///     Copied at construction.
/// placer : Placer
/// reacter : Reacter
///
/// Native components run without the GIL; a Python subclass is called once
/// per batch (see :class:`Placer`, :class:`Orienter`, :class:`Reacter`).
#[pyclass(module = "molrs.builder", name = "Assembler", frozen)]
pub struct PyAssembler {
    inner: Assembler,
    /// The placer's and the reacter's exception slots.
    placer_errors: ErrorSlot,
    reacter_errors: ErrorSlot,
}

#[pymethods]
impl PyAssembler {
    #[new]
    fn new(
        library: PyRef<'_, PyFragLibrary>,
        placer: &Bound<'_, PyPlacer>,
        reacter: &Bound<'_, PyReacter>,
    ) -> Self {
        Self {
            inner: Assembler::new(
                library.inner.clone(),
                PyPlacer::boxed(placer),
                PyReacter::boxed(reacter),
            ),
            placer_errors: placer.get().errors.clone(),
            reacter_errors: reacter.get().errors.clone(),
        }
    }

    /// Place every node of ``graph`` as a copy of its template and link every
    /// edge. The GIL is released for the whole assembly; the returned world
    /// is moved into the new fragment, not copied.
    ///
    /// Atoms are grouped by template name, not node order; ``frag_id`` (the
    /// node index) is the unit key. Ports no edge names stay on the world.
    ///
    /// Returns
    /// -------
    /// Fragment
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For every refusal, with the message: an unknown template, a port
    ///     ordinal past a template's ports, a placement, orientation or link
    ///     failure. A Python subclass's exception is the ``__cause__``.
    /// KeyboardInterrupt, SystemExit
    ///     Raised by a Python subclass, unchanged.
    fn assemble(&self, py: Python<'_>, graph: &Bound<'_, PyFragGraph>) -> PyResult<Py<PyFragment>> {
        let graph = &graph.get().inner;
        self.placer_errors.clear();
        self.reacter_errors.clear();
        let world = py.detach(|| self.inner.assemble(graph)).map_err(|e| {
            let errors = match &e {
                AssembleError::React { .. } => &self.reacter_errors,
                _ => &self.placer_errors,
            };
            errors.raise(py, e.handle_message())
        })?;
        PyFragment::from_core(py, world)
    }
}
