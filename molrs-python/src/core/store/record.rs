// PyO3 bindings for the record aggregate.
// Hosts `molrs.Record` and the `molrs.Observables` view.

use molrs::store::record::{MolRec as CoreMolRec, Observables as CoreObservables};
use molrs::store::trajectory::{ObservableKind, ObservableRecord};
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::core::store::frame::PyFrame;
use crate::core::store::trajectory::{PyScalarObservable, PyTrajectory, PyVectorObservable};
use crate::helpers::molrs_error_to_pyerr;

#[pyclass(module = "molrs", name = "Record", subclass)]
pub struct PyMolRec {
    pub(crate) inner: CoreMolRec,
}

/// Live view onto a record's `observables` section.
///
/// Holds the owning record rather than a copy, so `record.observables.add(...)`
/// mutates the record it came from.
#[pyclass(module = "molrs", name = "Observables")]
pub struct PyObservables {
    owner: Py<PyMolRec>,
}

#[pymethods]
impl PyMolRec {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreMolRec::new(),
        }
    }

    /// Set the snapshot section.
    fn set_frame(&mut self, frame: &Bound<'_, PyFrame>) -> PyResult<()> {
        self.inner.frame = Some(frame.borrow().clone_core_frame()?);
        Ok(())
    }

    /// Set the system-definition section.
    fn set_system(&mut self, frame: &Bound<'_, PyFrame>) -> PyResult<()> {
        self.inner.system = Some(frame.borrow().clone_core_frame()?);
        Ok(())
    }

    /// Append a frame to the trajectory section, creating it when absent.
    fn add_frame(&mut self, frame: &Bound<'_, PyFrame>) -> PyResult<()> {
        self.inner.add_frame(frame.borrow().clone_core_frame()?);
        Ok(())
    }

    /// Replace the trajectory section.
    fn set_trajectory(&mut self, trajectory: &Bound<'_, PyTrajectory>) {
        self.inner.trajectory = Some(trajectory.borrow().inner.clone());
    }

    /// Record the force field into `method` as a `classical` method.
    ///
    /// Force-field parameters are *not* a record-root section; the contract puts
    /// scientific parameters under `system/parameters` and the method identity
    /// under `method/`.
    fn set_forcefield(&mut self, forcefield: &Bound<'_, crate::ff::PyForceField>) {
        let name = forcefield.borrow().inner.name.clone();
        self.inner
            .method
            .insert("type".into(), JsonValue::String("classical".into()));
        self.inner.method.insert(
            "classical".into(),
            serde_json::json!({ "force_field": { "name": name } }),
        );
    }

    /// Number of frames the record carries.
    fn count_frames(&self) -> usize {
        self.inner.count_frames()
    }

    #[getter]
    fn frame(&self) -> PyResult<Option<PyFrame>> {
        self.inner
            .frame
            .clone()
            .map(PyFrame::from_core_frame)
            .transpose()
    }

    #[getter]
    fn system(&self) -> PyResult<Option<PyFrame>> {
        self.inner
            .system
            .clone()
            .map(PyFrame::from_core_frame)
            .transpose()
    }

    #[getter]
    fn trajectory(&self) -> Option<PyTrajectory> {
        self.inner
            .trajectory
            .clone()
            .map(|inner| PyTrajectory { inner })
    }

    #[getter]
    fn observables(slf: Py<Self>) -> PyObservables {
        PyObservables { owner: slf }
    }

    #[getter]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        json_map_to_dict(py, &self.inner.meta)
    }

    #[setter]
    fn set_meta(&mut self, value: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner.meta = dict_to_json_map(value)?;
        Ok(())
    }

    #[getter]
    fn method<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        json_map_to_dict(py, &self.inner.method)
    }

    #[setter]
    fn set_method(&mut self, value: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner.method = dict_to_json_map(value)?;
        Ok(())
    }

    #[getter]
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        json_map_to_dict(py, &self.inner.status)
    }

    #[setter]
    fn set_status(&mut self, value: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner.status = dict_to_json_map(value)?;
        Ok(())
    }

    #[getter]
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        json_map_to_dict(py, &self.inner.metrics)
    }

    #[setter]
    fn set_metrics(&mut self, value: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner.metrics = dict_to_json_map(value)?;
        Ok(())
    }

    /// Read a record from a store root.
    ///
    /// Requires the ``fs`` feature (default on desktop; omitted for Pyodide).
    #[staticmethod]
    #[cfg(feature = "fs")]
    fn read(path: &str) -> PyResult<Self> {
        let inner = molrs::io::store::zarr::read_record_file(path).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Write this record to a store root.
    ///
    /// Requires the ``fs`` feature (default on desktop; omitted for Pyodide).
    #[cfg(feature = "fs")]
    fn write(&self, path: &str) -> PyResult<()> {
        molrs::io::store::zarr::write_record_file(path, &self.inner).map_err(molrs_error_to_pyerr)
    }
}

#[pymethods]
impl PyObservables {
    fn __len__(&self, py: Python<'_>) -> usize {
        self.with(py, |obs| obs.len())
    }

    fn __contains__(&self, py: Python<'_>, name: &str) -> bool {
        self.with(py, |obs| obs.contains(name))
    }

    fn keys(&self, py: Python<'_>) -> Vec<String> {
        self.with(py, |obs| obs.names().cloned().collect())
    }

    /// Fetch one observable, or `None` when absent.
    fn get(&self, py: Python<'_>, name: &str) -> Option<Py<PyAny>> {
        let record = self.with(py, |obs| obs.get(name).cloned())?;
        observable_to_pyobject(py, record).ok()
    }

    /// Insert an already-built observable.
    fn add(&self, py: Python<'_>, observable: &Bound<'_, PyAny>) -> PyResult<()> {
        let record = pyobject_to_observable(observable)?;
        self.with_mut(py, |obs| obs.insert(record))
            .map_err(molrs_error_to_pyerr)
    }

    /// Build and insert a scalar observable, returning it.
    #[pyo3(signature = (name, data, description="", unit=None, axes=None, time_dependent=false, sampling=None, domain=None, target=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_scalar(
        &self,
        py: Python<'_>,
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
        let observable = PyScalarObservable::build(
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
        self.with_mut(py, |obs| obs.insert(observable.inner.clone()))
            .map_err(molrs_error_to_pyerr)?;
        Ok(observable)
    }

    /// Build and insert a vector observable, returning it.
    #[pyo3(signature = (name, data, description="", unit=None, axes=None, time_dependent=false, sampling=None, domain=None, target=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_vector(
        &self,
        py: Python<'_>,
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
        let observable = PyVectorObservable::build(
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
        self.with_mut(py, |obs| obs.insert(observable.inner.clone()))
            .map_err(molrs_error_to_pyerr)?;
        Ok(observable)
    }
}

impl PyObservables {
    fn with<R>(&self, py: Python<'_>, f: impl FnOnce(&CoreObservables) -> R) -> R {
        f(&self.owner.borrow(py).inner.observables)
    }

    fn with_mut<R>(&self, py: Python<'_>, f: impl FnOnce(&mut CoreObservables) -> R) -> R {
        f(&mut self.owner.borrow_mut(py).inner.observables)
    }
}

fn observable_to_pyobject(py: Python<'_>, record: ObservableRecord) -> PyResult<Py<PyAny>> {
    Ok(match record.kind {
        ObservableKind::Scalar => Py::new(py, PyScalarObservable { inner: record })?.into_any(),
        ObservableKind::Vector => Py::new(py, PyVectorObservable { inner: record })?.into_any(),
    })
}

fn pyobject_to_observable(value: &Bound<'_, PyAny>) -> PyResult<ObservableRecord> {
    if let Ok(scalar) = value.cast::<PyScalarObservable>() {
        return Ok(scalar.borrow().inner.clone());
    }
    if let Ok(vector) = value.cast::<PyVectorObservable>() {
        return Ok(vector.borrow().inner.clone());
    }
    Err(PyTypeError::new_err(
        "expected a ScalarObservable or VectorObservable",
    ))
}

// ---------------------------------------------------------------------------
// JSON <-> Python conversion for the metadata sections
// ---------------------------------------------------------------------------

fn json_map_to_dict<'py>(
    py: Python<'py>,
    map: &JsonMap<String, JsonValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in map {
        dict.set_item(key, json_to_py(py, value)?)?;
    }
    Ok(dict)
}

fn json_to_py(py: Python<'_>, value: &JsonValue) -> PyResult<Py<PyAny>> {
    Ok(match value {
        JsonValue::Null => py.None(),
        JsonValue::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any().unbind()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any().unbind()
            } else {
                n.as_f64()
                    .unwrap_or(f64::NAN)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        }
        JsonValue::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        JsonValue::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        JsonValue::Object(map) => json_map_to_dict(py, map)?.into_any().unbind(),
    })
}

fn dict_to_json_map(dict: &Bound<'_, PyDict>) -> PyResult<JsonMap<String, JsonValue>> {
    let mut map = JsonMap::new();
    for (key, value) in dict.iter() {
        let key: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("record metadata keys must be strings"))?;
        map.insert(key, py_to_json(&value)?);
    }
    Ok(map)
}

fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if value.is_none() {
        return Ok(JsonValue::Null);
    }
    // bool before int: Python bools are ints.
    if let Ok(b) = value.cast::<PyBool>() {
        return Ok(JsonValue::Bool(b.is_true()));
    }
    if let Ok(i) = value.cast::<PyInt>() {
        return Ok(JsonValue::from(i.extract::<i64>()?));
    }
    if let Ok(f) = value.cast::<PyFloat>() {
        return Ok(serde_json::Number::from_f64(f.extract::<f64>()?)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null));
    }
    if let Ok(s) = value.cast::<PyString>() {
        return Ok(JsonValue::String(s.extract::<String>()?));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        return Ok(JsonValue::Object(dict_to_json_map(dict)?));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_json(&item)?);
        }
        return Ok(JsonValue::Array(items));
    }
    Err(PyKeyError::new_err(format!(
        "record metadata values must be JSON-compatible, got {}",
        value.get_type().name()?
    )))
}
