//! Python bindings for structure builders (`molrs::builder`).

use molrs::{CarbonTubeBuilder, GrapheneBuilder};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::core::spatial::simbox::PyBox;
use crate::core::store::frame::PyFrame;

/// Exact single-wall carbon nanotube builder.
#[pyclass(module = "molrs.builder", name = "CarbonTubeBuilder")]
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
#[pyclass(module = "molrs.builder", name = "GrapheneBuilder")]
pub struct PyGrapheneBuilder {
    inner: GrapheneBuilder,
}

#[pymethods]
impl PyGrapheneBuilder {
    #[new]
    #[pyo3(signature = (nx, ny, *, bond_length=1.42, vacuum=10.0, periodic_xy=true))]
    fn new(
        nx: u32,
        ny: u32,
        bond_length: f64,
        vacuum: f64,
        periodic_xy: bool,
    ) -> PyResult<Self> {
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
