use std::cell::RefCell;
use std::rc::Rc;

use molrs::block::Column;
use molrs::molrec::{
    CellBox, MolRec as CoreMolRec, ObservableData, ObservableKind, ObservableRecord, RecordBox,
    SchemaValue, Trajectory as CoreTrajectory,
};
use molrs::types::{F, I, U};
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

use crate::forcefield::PyForceField;
use crate::frame::PyFrame;
use crate::helpers::{NpF, molrs_error_to_pyerr};
use crate::simbox::PyBox;

#[pyclass(name = "Trajectory", unsendable)]
#[derive(Clone)]
pub struct PyTrajectory {
    pub(crate) inner: CoreTrajectory,
}

#[pyclass(name = "MolRec", unsendable)]
#[derive(Clone)]
pub struct PyMolRec {
    inner: Rc<RefCell<CoreMolRec>>,
}

#[pyclass(name = "Observables", unsendable)]
#[derive(Clone)]
pub struct PyObservables {
    inner: Rc<RefCell<CoreMolRec>>,
}

#[pyclass(name = "ScalarObservable", unsendable)]
#[derive(Clone)]
pub struct PyScalarObservable {
    pub(crate) inner: ObservableRecord,
}

#[pyclass(name = "VectorObservable", unsendable)]
#[derive(Clone)]
pub struct PyVectorObservable {
    pub(crate) inner: ObservableRecord,
}

#[pymethods]
impl PyTrajectory {
    #[new]
    #[pyo3(signature = (frames, step=None, time=None, r#box=None))]
    fn new(
        frames: Vec<PyRef<'_, PyFrame>>,
        step: Option<PyReadonlyArray1<'_, i64>>,
        time: Option<PyReadonlyArray1<'_, NpF>>,
        r#box: Option<&Bound<'_, PyAny>>,
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
        inner.box_data = parse_record_box(r#box)?;
        inner.validate().map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (frames, step=None, time=None, r#box=None))]
    fn from_frames(
        frames: Vec<PyRef<'_, PyFrame>>,
        step: Option<PyReadonlyArray1<'_, i64>>,
        time: Option<PyReadonlyArray1<'_, NpF>>,
        r#box: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Self::new(frames, step, time, r#box)
    }

    fn __len__(&self) -> usize {
        self.inner.frames.len()
    }

    fn __getitem__(&self, index: usize) -> PyResult<PyFrame> {
        let mut frame = self
            .inner
            .frames
            .get(index)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err(index.to_string()))?;
        if let Some(cell) = self.inner.box_data.as_ref().and_then(|b| b.cell_at(index)) {
            if let Ok(simbox) = cell.to_simbox() {
                frame.simbox = Some(simbox);
            }
        }
        PyFrame::from_core_frame(frame)
    }

    #[getter]
    fn frames(&self) -> PyResult<Vec<PyFrame>> {
        self.inner
            .frames
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut frame = f.clone();
                if let Some(cell) = self.inner.box_data.as_ref().and_then(|b| b.cell_at(i)) {
                    if let Ok(simbox) = cell.to_simbox() {
                        frame.simbox = Some(simbox);
                    }
                }
                PyFrame::from_core_frame(frame)
            })
            .collect()
    }

    #[getter]
    fn step<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<i64>>> {
        self.inner
            .step
            .as_ref()
            .map(|step| ArrayD::from_shape_vec(IxDyn(&[step.len()]), step.clone()).unwrap().into_pyarray(py))
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

    #[getter]
    fn box_<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        record_box_to_pyobject(py, &self.inner.box_data)
    }
}

#[pymethods]
impl PyMolRec {
    #[new]
    fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(CoreMolRec::default())),
        }
    }

    /// Set the canonical frame.
    fn set_frame(&self, frame: &PyFrame) -> PyResult<()> {
        let core = frame.clone_core_frame()?;
        self.inner.borrow_mut().set_frame(core);
        Ok(())
    }

    /// Append one frame to the trajectory (creates it if needed).
    fn add_frame(&self, frame: &PyFrame) -> PyResult<()> {
        let core = frame.clone_core_frame()?;
        self.inner.borrow_mut().add_frame(core);
        Ok(())
    }

    /// Set the trajectory from a Trajectory object.
    fn set_trajectory(&self, trajectory: &PyTrajectory) {
        self.inner
            .borrow_mut()
            .set_trajectory(Some(trajectory.inner.clone()));
    }

    /// Set the forcefield; auto-populates method metadata.
    fn set_forcefield(&self, forcefield: &PyForceField) {
        self.inner.borrow_mut().set_forcefield(&forcefield.inner);
    }

    #[staticmethod]
    fn read_zarr(path: &str) -> PyResult<Self> {
        let inner = CoreMolRec::read_zarr(path).map_err(molrs_error_to_pyerr)?;
        Ok(Self {
            inner: Rc::new(RefCell::new(inner)),
        })
    }

    fn write_zarr(&self, path: &str) -> PyResult<()> {
        self.inner
            .borrow()
            .write_zarr(path)
            .map_err(molrs_error_to_pyerr)
    }

    fn count_frames(&self) -> usize {
        self.inner.borrow().count_frames()
    }

    #[getter]
    fn frame(&self) -> PyResult<PyFrame> {
        PyFrame::from_core_frame(self.inner.borrow().frame.clone())
    }

    #[getter]
    fn trajectory(&self) -> PyResult<Option<PyTrajectory>> {
        Ok(self.inner.borrow().trajectory.as_ref().map(|traj| PyTrajectory {
            inner: CoreTrajectory {
                frames: traj.frames.clone(),
                step: traj.step.clone(),
                time: traj.time.clone(),
                box_data: self.inner.borrow().box_data.clone(),
            },
        }))
    }

    #[getter]
    fn observables(&self) -> PyObservables {
        PyObservables {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        json_to_pyobject(py, &self.inner.borrow().meta)
    }

    #[setter]
    fn set_meta(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.borrow_mut().meta = py_any_to_json(value)?;
        Ok(())
    }

    #[getter]
    fn method<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        json_to_pyobject(py, &self.inner.borrow().method)
    }

    #[setter]
    fn set_method(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.borrow_mut().method = py_any_to_json(value)?;
        Ok(())
    }

    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        json_to_pyobject(py, &self.inner.borrow().parameters)
    }

    #[setter]
    fn set_parameters(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.borrow_mut().parameters = py_any_to_json(value)?;
        Ok(())
    }

    #[getter]
    fn box_<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        record_box_to_pyobject(py, &self.inner.borrow().box_data)
    }

    #[setter]
    fn set_box_(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.inner.borrow_mut().box_data = parse_record_box(value)?;
        Ok(())
    }
}

#[pymethods]
impl PyObservables {
    fn __len__(&self) -> usize {
        self.inner.borrow().observables.len()
    }

    fn keys(&self) -> Vec<String> {
        self.inner.borrow().observables.keys().cloned().collect()
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.borrow().observables.contains_key(name)
    }

    fn get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Py<PyAny>> {
        let Some(observable) = self.inner.borrow().observables.get(name).cloned() else {
            return Ok(py.None());
        };
        observable_to_pyobject(py, observable)
    }

    fn add(&self, observable: &Bound<'_, PyAny>) -> PyResult<()> {
        let record = extract_observable(observable)?;
        self.inner.borrow_mut().add_observable(record);
        Ok(())
    }

    #[pyo3(signature = (name, data, description="", unit=None, axes=None, time_dependent=false, sampling=None, domain=None, target=None))]
    fn add_scalar(
        &self,
        name: &str,
        data: &Bound<'_, PyAny>,
        description: &str,
        unit: Option<String>,
        axes: Option<Vec<String>>,
        time_dependent: bool,
        sampling: Option<String>,
        domain: Option<String>,
        target: Option<String>,
    ) -> PyResult<PyScalarObservable> {
        let observable = PyScalarObservable::new_impl(
            name,
            data,
            description,
            unit,
            axes,
            time_dependent,
            sampling,
            domain,
            target,
        )?;
        self.inner
            .borrow_mut()
            .add_observable(observable.inner.clone());
        Ok(observable)
    }

    #[pyo3(signature = (name, data, description="", unit=None, axes=None, time_dependent=false, sampling=None, domain=None, target=None))]
    fn add_vector(
        &self,
        name: &str,
        data: &Bound<'_, PyAny>,
        description: &str,
        unit: Option<String>,
        axes: Option<Vec<String>>,
        time_dependent: bool,
        sampling: Option<String>,
        domain: Option<String>,
        target: Option<String>,
    ) -> PyResult<PyVectorObservable> {
        let observable = PyVectorObservable::new_impl(
            name,
            data,
            description,
            unit,
            axes,
            time_dependent,
            sampling,
            domain,
            target,
        )?;
        self.inner
            .borrow_mut()
            .add_observable(observable.inner.clone());
        Ok(observable)
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

fn extract_observable(observable: &Bound<'_, PyAny>) -> PyResult<ObservableRecord> {
    if let Ok(observable) = observable.extract::<PyRef<'_, PyScalarObservable>>() {
        return Ok(observable.inner.clone());
    }
    if let Ok(observable) = observable.extract::<PyRef<'_, PyVectorObservable>>() {
        return Ok(observable.inner.clone());
    }
    Err(PyTypeError::new_err(
        "expected ScalarObservable or VectorObservable",
    ))
}

fn parse_record_box(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<RecordBox>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(pybox) = value.extract::<PyRef<'_, PyBox>>() {
        return Ok(Some(RecordBox::Static {
            cell: CellBox::from_simbox(&pybox.inner),
        }));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let mut cells = Vec::with_capacity(list.len());
        for item in list.iter() {
            let pybox = item.extract::<PyRef<'_, PyBox>>()?;
            cells.push(CellBox::from_simbox(&pybox.inner));
        }
        return Ok(Some(RecordBox::Dynamic { cells }));
    }
    Err(PyTypeError::new_err("box must be a Box, list[Box], or None"))
}

fn record_box_to_pyobject(py: Python<'_>, box_data: &Option<RecordBox>) -> PyResult<Py<PyAny>> {
    match box_data {
        None => Ok(py.None()),
        Some(RecordBox::Static { cell }) => {
            let pybox = Py::new(
                py,
                PyBox {
                    inner: cell.to_simbox().map_err(molrs_error_to_pyerr)?,
                },
            )?;
            Ok(pybox.into_bound(py).into_any().unbind())
        }
        Some(RecordBox::Dynamic { cells }) => {
            let mut boxes = Vec::with_capacity(cells.len());
            for cell in cells {
                let pybox = Py::new(
                    py,
                    PyBox {
                        inner: cell.to_simbox().map_err(molrs_error_to_pyerr)?,
                    },
                )?;
                boxes.push(pybox);
            }
            Ok(PyList::new(py, boxes)?.into_any().unbind())
        }
    }
}

fn py_any_to_json(value: &Bound<'_, PyAny>) -> PyResult<SchemaValue> {
    if value.is_none() {
        return Ok(JsonValue::Null);
    }
    if let Ok(v) = value.extract::<bool>() {
        return Ok(JsonValue::Bool(v));
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(JsonValue::Number(JsonNumber::from(v)));
    }
    if let Ok(v) = value.extract::<f64>() {
        let number = JsonNumber::from_f64(v)
            .ok_or_else(|| PyValueError::new_err("cannot encode NaN or Infinity to JSON"))?;
        return Ok(JsonValue::Number(number));
    }
    if let Ok(v) = value.extract::<String>() {
        return Ok(JsonValue::String(v));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut out = JsonMap::new();
        for (key, val) in dict.iter() {
            out.insert(key.extract::<String>()?, py_any_to_json(&val)?);
        }
        return Ok(JsonValue::Object(out));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_any_to_json(&item)?);
        }
        return Ok(JsonValue::Array(out));
    }
    Err(PyTypeError::new_err(
        "schema values must be composed of dict, list, str, bool, int, float, or None",
    ))
}

fn json_to_pyobject(py: Python<'_>, value: &SchemaValue) -> PyResult<Py<PyAny>> {
    match value {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(v) => Ok(pyo3::types::PyBool::new(py, *v).to_owned().into_any().unbind()),
        JsonValue::Number(v) => {
            if let Some(i) = v.as_i64() {
                Ok(i64::into_pyobject(i, py)?.unbind().into_any())
            } else if let Some(f) = v.as_f64() {
                Ok(f64::into_pyobject(f, py)?.unbind().into_any())
            } else {
                Err(PyValueError::new_err("unsupported JSON number"))
            }
        }
        JsonValue::String(v) => Ok(v.into_pyobject(py)?.unbind().into_any()),
        JsonValue::Array(values) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                out.push(json_to_pyobject(py, value)?);
            }
            Ok(PyList::new(py, out)?.into_any().unbind())
        }
        JsonValue::Object(map) => {
            let out = PyDict::new(py);
            for (key, value) in map {
                out.set_item(key, json_to_pyobject(py, value)?)?;
            }
            Ok(out.into_any().unbind())
        }
    }
}

fn py_any_to_column(value: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        return Ok(Column::Float(arr.as_array().mapv(|v| v as F).into_dyn()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, f64>>() {
        return Ok(Column::Float(arr.as_array().mapv(|v| v as F).into_dyn()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, i32>>() {
        return Ok(Column::Int(arr.as_array().mapv(|v| v as I).into_dyn()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, i64>>() {
        return Ok(Column::Int(arr.as_array().mapv(|v| v as I).into_dyn()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, u32>>() {
        return Ok(Column::UInt(arr.as_array().mapv(|v| v as U).into_dyn()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, u64>>() {
        return Ok(Column::UInt(arr.as_array().mapv(|v| v as U).into_dyn()));
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, bool>>() {
        return Ok(Column::Bool(arr.as_array().to_owned().into_dyn()));
    }
    if let Ok(strings) = value.extract::<Vec<String>>() {
        return Ok(Column::String(ArrayD::from_shape_vec(IxDyn(&[strings.len()]), strings).unwrap()));
    }
    if let Ok(v) = value.extract::<f64>() {
        return Ok(Column::Float(ArrayD::from_elem(IxDyn(&[]), v as F)));
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(Column::Int(ArrayD::from_elem(IxDyn(&[]), v as I)));
    }
    if let Ok(v) = value.extract::<u64>() {
        return Ok(Column::UInt(ArrayD::from_elem(IxDyn(&[]), v as U)));
    }
    if let Ok(v) = value.extract::<bool>() {
        return Ok(Column::Bool(ArrayD::from_elem(IxDyn(&[]), v)));
    }
    if let Ok(v) = value.extract::<String>() {
        return Ok(Column::String(ArrayD::from_elem(IxDyn(&[]), v)));
    }
    Err(PyTypeError::new_err(
        "observable data must be a supported numpy array, scalar, or list[str]",
    ))
}

fn observable_to_pyobject(py: Python<'_>, observable: ObservableRecord) -> PyResult<Py<PyAny>> {
    match observable.kind {
        ObservableKind::Scalar => Ok(Py::new(py, PyScalarObservable { inner: observable })?
            .into_bound(py)
            .into_any()
            .unbind()),
        ObservableKind::Vector => Ok(Py::new(py, PyVectorObservable { inner: observable })?
            .into_bound(py)
            .into_any()
            .unbind()),
    }
}

fn observable_data_to_pyobject(py: Python<'_>, data: &ObservableData) -> PyResult<Py<PyAny>> {
    match data {
        ObservableData::Column(column) => column_to_pyobject(py, column),
    }
}

fn column_to_pyobject(py: Python<'_>, column: &Column) -> PyResult<Py<PyAny>> {
    match column {
        Column::Float(array) => Ok(array
            .clone()
            .mapv(|v| v as NpF)
            .into_pyarray(py)
            .into_any()
            .unbind()),
        Column::Int(array) => Ok(array.clone().into_pyarray(py).into_any().unbind()),
        Column::UInt(array) => Ok(array.clone().into_pyarray(py).into_any().unbind()),
        Column::Bool(array) => Ok(array.clone().into_pyarray(py).into_any().unbind()),
        Column::U8(array) => Ok(array.clone().into_pyarray(py).into_any().unbind()),
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
