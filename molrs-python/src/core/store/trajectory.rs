// PyO3 bindings for `Trajectory` (frame sequence) and observable records.
// Hosts `molrs.Trajectory`, `molrs.ScalarObservable`, `molrs.VectorObservable`.
#![allow(clippy::too_many_arguments)]

use molrs::store::block::Column;
use molrs::store::trajectory::{ObservableData, ObservableRecord, Trajectory as CoreTrajectory};
use molrs::types::{F, I, U};
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use crate::core::store::frame::PyFrame;
use crate::helpers::{NpF, molrs_error_to_pyerr};

#[pyclass(module = "molrs", name = "Trajectory", from_py_object, subclass)]
#[derive(Clone)]
pub struct PyTrajectory {
    pub(crate) inner: CoreTrajectory,
}

#[pyclass(module = "molrs", name = "ScalarObservable", from_py_object)]
#[derive(Clone)]
pub struct PyScalarObservable {
    pub(crate) inner: ObservableRecord,
}

#[pyclass(module = "molrs", name = "VectorObservable", from_py_object)]
#[derive(Clone)]
pub struct PyVectorObservable {
    pub(crate) inner: ObservableRecord,
}

#[pymethods]
impl PyTrajectory {
    #[new]
    #[pyo3(signature = (frames, step=None, time=None))]
    fn new(
        frames: Vec<PyRef<'_, PyFrame>>,
        step: Option<PyReadonlyArray1<'_, i64>>,
        time: Option<PyReadonlyArray1<'_, NpF>>,
    ) -> PyResult<Self> {
        let core_frames: Vec<_> = frames
            .iter()
            .map(|frame| frame.clone_core_frame())
            .collect::<PyResult<_>>()?;
        let mut inner = CoreTrajectory::from_frames(core_frames);
        if let Some(step) = step {
            inner.step = Some(step.as_slice()?.to_vec());
        }
        if let Some(time) = time {
            inner.time = Some(time.as_slice()?.iter().copied().map(|v| v as F).collect());
        }
        inner.validate().map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (frames, step=None, time=None))]
    fn from_frames(
        frames: Vec<PyRef<'_, PyFrame>>,
        step: Option<PyReadonlyArray1<'_, i64>>,
        time: Option<PyReadonlyArray1<'_, NpF>>,
    ) -> PyResult<Self> {
        Self::new(frames, step, time)
    }

    fn __len__(&self) -> usize {
        self.inner.frames.len()
    }

    fn __getitem__(&self, index: usize) -> PyResult<PyFrame> {
        let frame = self
            .inner
            .frames
            .get(index)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err(index.to_string()))?;
        PyFrame::from_core_frame(frame)
    }

    #[getter]
    fn frames(&self) -> PyResult<Vec<PyFrame>> {
        self.inner
            .frames
            .iter()
            .map(|f| PyFrame::from_core_frame(f.clone()))
            .collect()
    }

    #[getter]
    fn step<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<i64>>> {
        self.inner.step.as_ref().map(|step| {
            ArrayD::from_shape_vec(IxDyn(&[step.len()]), step.clone())
                .unwrap()
                .into_pyarray(py)
        })
    }

    #[getter]
    fn time<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<NpF>>> {
        self.inner.time.as_ref().map(|time| {
            let values: Vec<NpF> = time.iter().copied().map(|v| v as NpF).collect();
            ArrayD::from_shape_vec(IxDyn(&[values.len()]), values)
                .unwrap()
                .into_pyarray(py)
        })
    }

    /// Read a frame-sequence Zarr archive into a `Trajectory`.
    ///
    /// Requires the ``fs`` feature (default on desktop; omitted for Pyodide).
    #[staticmethod]
    #[cfg(feature = "fs")]
    fn read_zarr(path: &str) -> PyResult<Self> {
        let inner =
            molrs::io::store::zarr::read_trajectory_file(path).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Write this trajectory to a frame-sequence Zarr archive.
    ///
    /// Requires the ``fs`` feature (default on desktop; omitted for Pyodide).
    #[cfg(feature = "fs")]
    fn write_zarr(&self, path: &str) -> PyResult<()> {
        molrs::io::store::zarr::write_trajectory_file(path, &self.inner)
            .map_err(molrs_error_to_pyerr)
    }

    /// Number of frames in the trajectory.
    fn count_frames(&self) -> usize {
        self.inner.frames.len()
    }
}

#[pymethods]
impl PyScalarObservable {
    #[new]
    #[pyo3(signature = (name, data, description="", unit=None, axes=None, time_dependent=false, sampling=None, domain=None, target=None))]
    fn new(
        name: &str,
        data: &Bound<'_, PyAny>,
        description: &str,
        unit: Option<String>,
        axes: Option<Vec<String>>,
        time_dependent: bool,
        sampling: Option<String>,
        domain: Option<String>,
        target: Option<String>,
    ) -> PyResult<Self> {
        Self::new_impl(
            name,
            data,
            description,
            unit,
            axes,
            time_dependent,
            sampling,
            domain,
            target,
        )
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        observable_data_to_pyobject(py, &self.inner.data)
    }

    #[getter]
    fn kind(&self) -> &'static str {
        "scalar"
    }
}

#[pymethods]
impl PyVectorObservable {
    #[new]
    #[pyo3(signature = (name, data, description="", unit=None, axes=None, time_dependent=false, sampling=None, domain=None, target=None))]
    fn new(
        name: &str,
        data: &Bound<'_, PyAny>,
        description: &str,
        unit: Option<String>,
        axes: Option<Vec<String>>,
        time_dependent: bool,
        sampling: Option<String>,
        domain: Option<String>,
        target: Option<String>,
    ) -> PyResult<Self> {
        Self::new_impl(
            name,
            data,
            description,
            unit,
            axes,
            time_dependent,
            sampling,
            domain,
            target,
        )
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        observable_data_to_pyobject(py, &self.inner.data)
    }

    #[getter]
    fn kind(&self) -> &'static str {
        "vector"
    }
}

impl PyScalarObservable {
    fn new_impl(
        name: &str,
        data: &Bound<'_, PyAny>,
        description: &str,
        unit: Option<String>,
        axes: Option<Vec<String>>,
        time_dependent: bool,
        sampling: Option<String>,
        domain: Option<String>,
        target: Option<String>,
    ) -> PyResult<Self> {
        let mut inner = ObservableRecord::scalar(name, py_any_to_column(data)?);
        apply_common_metadata(
            &mut inner,
            description,
            unit,
            axes,
            time_dependent,
            sampling,
            domain,
            target,
        );
        Ok(Self { inner })
    }
}

impl PyVectorObservable {
    fn new_impl(
        name: &str,
        data: &Bound<'_, PyAny>,
        description: &str,
        unit: Option<String>,
        axes: Option<Vec<String>>,
        time_dependent: bool,
        sampling: Option<String>,
        domain: Option<String>,
        target: Option<String>,
    ) -> PyResult<Self> {
        let mut inner = ObservableRecord::vector(name, py_any_to_column(data)?);
        apply_common_metadata(
            &mut inner,
            description,
            unit,
            axes,
            time_dependent,
            sampling,
            domain,
            target,
        );
        Ok(Self { inner })
    }
}

fn apply_common_metadata(
    observable: &mut ObservableRecord,
    description: &str,
    unit: Option<String>,
    axes: Option<Vec<String>>,
    time_dependent: bool,
    sampling: Option<String>,
    domain: Option<String>,
    target: Option<String>,
) {
    observable.description = description.to_string();
    observable.unit = unit;
    observable.axes = axes.unwrap_or_default();
    observable.time_dependent = time_dependent;
    observable.sampling = sampling;
    observable.domain = domain;
    observable.target = target;
}

fn py_any_to_column(value: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        return Ok(Column::from_float(
            arr.as_array().mapv(|v| v as F).into_dyn(),
        ));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, f64>>() {
        return Ok(Column::from_float(
            arr.as_array().mapv(|v| v as F).into_dyn(),
        ));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, i32>>() {
        return Ok(Column::from_int(arr.as_array().mapv(|v| v as I).into_dyn()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, i64>>() {
        return Ok(Column::from_int(arr.as_array().mapv(|v| v as I).into_dyn()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, u32>>() {
        return Ok(Column::from_uint(
            arr.as_array().mapv(|v| v as U).into_dyn(),
        ));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, u64>>() {
        return Ok(Column::from_uint(
            arr.as_array().mapv(|v| v as U).into_dyn(),
        ));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, bool>>() {
        return Ok(Column::from_bool(arr.as_array().to_owned().into_dyn()));
    }
    if let Ok(strings) = value.extract::<Vec<String>>() {
        return Ok(Column::from_string(
            ArrayD::from_shape_vec(IxDyn(&[strings.len()]), strings).unwrap(),
        ));
    }
    if let Ok(v) = value.extract::<f64>() {
        return Ok(Column::from_float(ArrayD::from_elem(IxDyn(&[]), v as F)));
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(Column::from_int(ArrayD::from_elem(IxDyn(&[]), v as I)));
    }
    if let Ok(v) = value.extract::<u64>() {
        return Ok(Column::from_uint(ArrayD::from_elem(IxDyn(&[]), v as U)));
    }
    if let Ok(v) = value.extract::<bool>() {
        return Ok(Column::from_bool(ArrayD::from_elem(IxDyn(&[]), v)));
    }
    if let Ok(v) = value.extract::<String>() {
        return Ok(Column::from_string(ArrayD::from_elem(IxDyn(&[]), v)));
    }
    Err(PyTypeError::new_err(
        "observable data must be a supported numpy array, scalar, or list[str]",
    ))
}

fn observable_data_to_pyobject(py: Python<'_>, data: &ObservableData) -> PyResult<Py<PyAny>> {
    match data {
        ObservableData::Column(column) => column_to_pyobject(py, column),
    }
}

fn column_to_pyobject(py: Python<'_>, column: &Column) -> PyResult<Py<PyAny>> {
    match column {
        // .mapv through ColumnHolder's Deref produces an owned ArrayD<NpF>.
        Column::Float(array) => Ok(array
            .array()
            .mapv(|v| v as NpF)
            .into_pyarray(py)
            .into_any()
            .unbind()),
        // For non-Float columns the caller expects an owned numpy array, so we
        // deep-clone the inner ArrayD out of the holder. `.array().clone()`
        // takes &ArrayD<T> and calls ArrayD::clone (deep-copy), detaching from
        // any foreign-backed holder as a side effect.
        Column::Int(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::UInt(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::Bool(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::U8(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::String(array) => {
            if array.ndim() == 0 {
                let value = array.iter().next().cloned().unwrap_or_default();
                Ok(value.into_pyobject(py)?.unbind().into_any())
            } else {
                let values: Vec<String> = array.iter().cloned().collect();
                Ok(PyList::new(py, values)?.into_any().unbind())
            }
        }
    }
}
