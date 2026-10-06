// PyO3 bindings for `Trajectory` (frame sequence) and observable records.
// Hosts `molrs.store.Trajectory`, `molrs.store.ScalarObservable`, `molrs.store.VectorObservable`.
#![allow(clippy::too_many_arguments)]

use molrs::op::types::{F, I, Idx};
use molrs::store::Column;
use molrs::store::{ObservableData, ObservableRecord, Trajectory as CoreTrajectory};
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PySlice};

use crate::core::store::frame::PyFrame;
use crate::error::molrs_error_to_pyerr;

/// An in-memory frame sequence with optional per-frame ``step`` / ``time``
/// labels.
///
/// ``len(traj)``, ``traj[i]`` (negative indices count from the end) and
/// iteration give frames; ``traj[a:b:c]`` is the sub-trajectory of those
/// frames with their labels, and :meth:`map` builds a new trajectory frame by
/// frame. Lazy, seekable reading from disk is a file reader's
/// (:mod:`molrs.io`), not this container's.
///
/// Parameters
/// ----------
/// frames
///     The :class:`Frame` sequence; each frame is copied in.
/// step
///     Optional ``int64`` step label per frame.
/// time
///     Optional ``float64`` time per frame.
///
/// Raises
/// ------
/// ValueError
///     If ``step`` or ``time`` does not have one entry per frame.
#[pyclass(module = "molrs.store", name = "Trajectory", from_py_object, subclass)]
#[derive(Clone)]
pub struct PyTrajectory {
    pub(crate) inner: CoreTrajectory,
}

#[pyclass(
    module = "molrs.store",
    name = "ScalarObservable",
    from_py_object,
    subclass
)]
#[derive(Clone)]
pub struct PyScalarObservable {
    pub(crate) inner: ObservableRecord,
}

#[pyclass(
    module = "molrs.store",
    name = "VectorObservable",
    from_py_object,
    subclass
)]
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
        time: Option<PyReadonlyArray1<'_, f64>>,
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

    fn __len__(&self) -> usize {
        self.inner.frames.len()
    }

    /// The frame at an integer index (negative counts from the end), or the
    /// sub-trajectory a slice selects, its ``step`` / ``time`` sliced alike.
    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        let n = this.inner.frames.len();
        if let Ok(slice) = key.cast::<PySlice>() {
            let ix = slice.indices(n as isize)?;
            let indices: Vec<usize> = (0..ix.slicelength)
                .map(|k| (ix.start + k as isize * ix.step) as usize)
                .collect();
            let sub = Self {
                inner: this.inner.select(&indices),
            };
            return Ok(Bound::new(py, sub)?.into_any().unbind());
        }
        let index: isize = key.extract()?;
        let resolved = if index < 0 { index + n as isize } else { index };
        if resolved < 0 || resolved as usize >= n {
            return Err(PyIndexError::new_err("trajectory index out of range"));
        }
        let frame = this.inner.frames[resolved as usize].clone();
        Ok(Bound::new(py, PyFrame::from_core_frame(frame)?)?
            .into_any()
            .unbind())
    }

    /// A new trajectory of ``func(frame)`` for every frame, keeping this
    /// one's ``step`` / ``time`` labels. This trajectory is not modified.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``func`` returns something other than a :class:`Frame`.
    fn map(&self, func: &Bound<'_, PyAny>) -> PyResult<Self> {
        let frames = self
            .inner
            .frames
            .iter()
            .map(|frame| {
                let mapped = func.call1((PyFrame::from_core_frame(frame.clone())?,))?;
                let mapped = mapped.cast::<PyFrame>().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "Trajectory.map: func must return a Frame, got {}",
                        mapped
                            .get_type()
                            .name()
                            .map_or_else(|_| "?".into(), |n| n.to_string())
                    ))
                })?;
                mapped.borrow().clone_core_frame()
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: CoreTrajectory {
                frames,
                step: self.inner.step.clone(),
                time: self.inner.time.clone(),
            },
        })
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
    fn time<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner.time.as_ref().map(|time| {
            let values: Vec<f64> = time.to_vec();
            ArrayD::from_shape_vec(IxDyn(&[values.len()]), values)
                .unwrap()
                .into_pyarray(py)
        })
    }

    fn __repr__(&self) -> String {
        format!("Trajectory(n_frames={})", self.inner.frames.len())
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        crate::pickle::reduce_via_type(slf.as_any(), (this.frames()?, this.step(py), this.time(py)))
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
        Self::build(
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

    /// The contract spelling of the observable's kind.
    #[getter]
    fn kind(&self) -> &str {
        self.inner.kind.as_str()
    }

    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    #[getter]
    fn unit(&self) -> Option<String> {
        self.inner.unit.clone()
    }

    #[getter]
    fn axes(&self) -> Vec<String> {
        self.inner.axes.clone()
    }

    #[getter]
    fn time_dependent(&self) -> bool {
        self.inner.time_dependent
    }

    #[getter]
    fn sampling(&self) -> Option<String> {
        self.inner.sampling.clone()
    }

    #[getter]
    fn domain(&self) -> Option<String> {
        self.inner.domain.clone()
    }

    #[getter]
    fn target(&self) -> Option<String> {
        self.inner.target.clone()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                this.name(),
                this.data(slf.py())?,
                this.description(),
                this.unit(),
                this.axes(),
                this.time_dependent(),
                this.sampling(),
                this.domain(),
                this.target(),
            ),
        )
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
        Self::build(
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

    /// The contract spelling of the observable's kind.
    #[getter]
    fn kind(&self) -> &str {
        self.inner.kind.as_str()
    }

    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    #[getter]
    fn unit(&self) -> Option<String> {
        self.inner.unit.clone()
    }

    #[getter]
    fn axes(&self) -> Vec<String> {
        self.inner.axes.clone()
    }

    #[getter]
    fn time_dependent(&self) -> bool {
        self.inner.time_dependent
    }

    #[getter]
    fn sampling(&self) -> Option<String> {
        self.inner.sampling.clone()
    }

    #[getter]
    fn domain(&self) -> Option<String> {
        self.inner.domain.clone()
    }

    #[getter]
    fn target(&self) -> Option<String> {
        self.inner.target.clone()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                this.name(),
                this.data(slf.py())?,
                this.description(),
                this.unit(),
                this.axes(),
                this.time_dependent(),
                this.sampling(),
                this.domain(),
                this.target(),
            ),
        )
    }
}

impl PyScalarObservable {
    pub(crate) fn build(
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
    pub(crate) fn build(
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
            arr.as_array().mapv(|v| v as Idx).into_dyn(),
        ));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, u64>>() {
        return Ok(Column::from_uint(
            arr.as_array().mapv(|v| v as Idx).into_dyn(),
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
        return Ok(Column::from_uint(ArrayD::from_elem(IxDyn(&[]), v as Idx)));
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
        // .mapv through ColumnHolder's Deref produces an owned ArrayD<f64>.
        Column::Float(array) => Ok(array
            .array()
            .mapv(|v| v)
            .into_pyarray(py)
            .into_any()
            .unbind()),
        Column::Int(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::Int8(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::Int16(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::Int64(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::UInt(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::Bool(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::U8(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::UInt16(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::UInt32(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::Complex64(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
        Column::Complex128(array) => Ok(array.array().clone().into_pyarray(py).into_any().unbind()),
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
