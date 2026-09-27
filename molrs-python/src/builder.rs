//! Python bindings for `molrs::builder`: the structure generators
//! `CarbonTubeBuilder` and `GrapheneBuilder`, each building a fresh `Frame`,
//! and trace assembly — `Assembler(library, TracePlacer()).assemble(traces,
//! names)`, which places and links one world `Fragment`.

use std::collections::HashMap;

use molrs::builder::{AssembleError, Assembler, PlaceError, TracePlacer};
use molrs::spatial::Trace;
use molrs::system::fragment::Fragment;
use molrs::system::link::LinkManyError;
use molrs::{CarbonTubeBuilder, GrapheneBuilder};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyMapping;

use crate::core::spatial::simbox::PyBox;
use crate::core::spatial::trace::PyTrace;
use crate::core::store::frame::PyFrame;
use crate::core::system::molgraph::{
    PyAtomistic, PyFragment, center_error_message, link_error_message,
};
use crate::helpers::molrs_error_to_pyerr;

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

/// The translation-only placer — ``molrs.builder.TracePlacer``.
///
/// Each copy keeps its template's orientation and its centre of mass (Å,
/// weights ``mass``) lands on its trace point. Orientation and overlap are
/// left to a later relaxation. Takes no arguments.
///
/// Examples
/// --------
/// >>> assembler = molrs.builder.Assembler(library, molrs.builder.TracePlacer())
#[pyclass(module = "molrs.builder", name = "TracePlacer", frozen)]
pub struct PyTracePlacer;

#[pymethods]
impl PyTracePlacer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "TracePlacer()".to_owned()
    }
}

/// One placed, linked world from traces and their unit names —
/// ``molrs.builder.Assembler``.
///
/// Unit ``k`` of trace ``t`` is a copy of ``library[names[t][k]]`` placed on
/// the trace's point ``k``. In a trace of two or more units, unit ``i``'s one
/// ``>`` port joins unit ``i + 1``'s one ``<`` port; the leaving groups are
/// removed. Every atom gets ``frag_id`` (the unit's trace-major ordinal,
/// 0-based) and ``mol_id`` (the trace's ordinal + 1).
///
/// Parameters
/// ----------
/// library : Mapping[str, Fragment | Atomistic]
///     Name → template, copied at construction. An ``Atomistic`` is a
///     template with no ports; it can only fill a one-unit trace.
/// placer : TracePlacer
///     Turns a template and its points into one rigid motion per copy.
///
/// Raises
/// ------
/// TypeError
///     If ``library`` is not a mapping of ``str`` to ``Fragment`` or
///     ``Atomistic``, or ``placer`` is not a :class:`TracePlacer`.
/// ValueError
///     If an ``Atomistic`` value does not convert to a ``Fragment``.
///
/// Examples
/// --------
/// >>> world = molrs.builder.Assembler(
/// ...     {"PMA": pma, "Li": li}, molrs.builder.TracePlacer()
/// ... ).assemble(traces, names)
#[pyclass(module = "molrs.builder", name = "Assembler", frozen)]
pub struct PyAssembler {
    inner: Assembler,
}

#[pymethods]
impl PyAssembler {
    #[new]
    fn new(library: &Bound<'_, PyMapping>, placer: &Bound<'_, PyTracePlacer>) -> PyResult<Self> {
        // `TracePlacer` carries no state: the argument's type is the whole
        // choice, checked by the extraction above.
        let _ = placer;
        let mut templates = HashMap::new();
        for item in library.items()?.iter() {
            let (name, value): (String, Bound<'_, PyAny>) = item.extract()?;
            let template = if let Ok(fragment) = value.cast::<PyFragment>() {
                fragment.borrow().core().clone()
            } else if let Ok(mol) = value.cast::<PyAtomistic>() {
                Fragment::try_from_molgraph(mol.borrow().core().clone().into_inner())
                    .map_err(molrs_error_to_pyerr)?
            } else {
                return Err(PyTypeError::new_err(format!(
                    "library['{name}'] must be a Fragment or an Atomistic, not {}",
                    value.get_type().name()?
                )));
            };
            templates.insert(name, template);
        }
        Ok(Self {
            inner: Assembler::new(templates, Box::new(TracePlacer::new())),
        })
    }

    /// Place and link every trace; return the world.
    ///
    /// The GIL is released while assembling.
    ///
    /// Parameters
    /// ----------
    /// traces : Sequence[Trace]
    ///     One trace per molecule; its points are the unit positions (Å).
    /// names : Sequence[Sequence[str]]
    ///     One library name per point of each trace.
    ///
    /// Returns
    /// -------
    /// Fragment
    ///     The world; chain-end ports and the ports of one-unit traces stay
    ///     on it. Empty when ``traces`` is empty.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Naming the trace, unit and name at fault: the counts differ, a
    ///     name is not in the library, a unit lacks the one ``>`` / ``<``
    ///     port a join needs, a template has no centre of mass, or a join is
    ///     refused.
    fn assemble(
        &self,
        py: Python<'_>,
        traces: Vec<PyRef<'_, PyTrace>>,
        names: Vec<Vec<String>>,
    ) -> PyResult<Py<PyFragment>> {
        let traces: Vec<Trace> = traces.iter().map(|trace| trace.inner.clone()).collect();
        let assembler = &self.inner;
        let world = py
            .detach(|| assembler.assemble(&traces, &names))
            .map_err(|e| PyValueError::new_err(assemble_error_message(e)))?;
        PyFragment::from_core(py, world)
    }

    fn __repr__(&self) -> String {
        "Assembler(TracePlacer())".to_owned()
    }
}

/// The message of an [`AssembleError`] as Python sees it: node and port ids
/// as int handles, never `NodeId(..)` / `PortId(..)`. Every variant is
/// matched by name; the wording of the outer sentence is the core's.
fn assemble_error_message(e: AssembleError) -> String {
    match e {
        AssembleError::Place {
            name,
            trace,
            unit,
            source,
        } => {
            let reason = match source {
                PlaceError::Template(center) => format!(
                    "the template has no centre of mass: {}",
                    center_error_message(center)
                ),
                point @ PlaceError::NonFinitePoint { .. } => point.to_string(),
            };
            format!("unit {unit} of trace {trace} ('{name}') cannot be placed: {reason}")
        }
        AssembleError::Link {
            trace,
            unit,
            source,
        } => {
            let reason = match source {
                LinkManyError::Pair { pair, source } => {
                    format!("pair {pair} is refused: {}", link_error_message(source))
                }
                batch @ (LinkManyError::PortReused { .. }
                | LinkManyError::DuplicateBond { .. }
                | LinkManyError::BranchesOverlap { .. }) => batch.to_string(),
            };
            format!(
                "unit {unit} of trace {trace} cannot join unit {}: {reason}",
                unit + 1
            )
        }
        e @ (AssembleError::LengthMismatch { .. }
        | AssembleError::SequenceLength { .. }
        | AssembleError::TooManyUnits { .. }
        | AssembleError::UnknownName { .. }
        | AssembleError::MissingPort { .. }
        | AssembleError::AmbiguousPort { .. }
        | AssembleError::Template { .. }
        | AssembleError::Replicate { .. }) => e.to_string(),
    }
}
