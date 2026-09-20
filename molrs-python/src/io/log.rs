//! Structured LAMMPS log, exposed to Python as classes under `molrs.io`.
//!
//! Each class is a read-only view over the corresponding
//! `molrs::io::log::lammps` struct; nested values are handed out as the
//! matching Python class, thermo tables as NumPy arrays.

use molrs::io::log::lammps::{
    LammpsCpuUse, LammpsLoadBalance, LammpsLog, LammpsLogHeader, LammpsLoopTime, LammpsMemoryUsage,
    LammpsNeighborStatistics, LammpsPerformance, LammpsRun, LammpsThermo, LammpsTimingBreakdown,
    LammpsTimingRow, LammpsWarning,
};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Header lines that precede the first run.
#[pyclass(
    module = "molrs.io",
    name = "LammpsLogHeader",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsLogHeader {
    inner: LammpsLogHeader,
}

#[pymethods]
impl PyLammpsLogHeader {
    /// Header lines, verbatim.
    #[getter]
    fn lines(&self) -> Vec<String> {
        self.inner.lines.clone()
    }

    /// The header joined back into one string.
    fn raw_text(&self) -> String {
        self.inner.raw_text()
    }

    fn __repr__(&self) -> String {
        format!("LammpsLogHeader(lines={})", self.inner.lines.len())
    }
}

/// The ``Per MPI rank memory allocation`` line of a run.
#[pyclass(
    module = "molrs.io",
    name = "LammpsMemoryUsage",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsMemoryUsage {
    inner: LammpsMemoryUsage,
}

#[pymethods]
impl PyLammpsMemoryUsage {
    #[getter]
    fn minimum(&self) -> f64 {
        self.inner.minimum
    }
    #[getter]
    fn average(&self) -> f64 {
        self.inner.average
    }
    #[getter]
    fn maximum(&self) -> f64 {
        self.inner.maximum
    }
    #[getter]
    fn units(&self) -> String {
        self.inner.units.clone()
    }
    #[getter]
    fn raw_line(&self) -> String {
        self.inner.raw_line.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "LammpsMemoryUsage(min={}, avg={}, max={} {})",
            self.inner.minimum, self.inner.average, self.inner.maximum, self.inner.units
        )
    }
}

/// One run's thermo table: ``columns`` names the fields, ``rows`` is the
/// ``(n_rows, n_columns)`` float64 array, and ``thermo["Step"]`` is a column.
#[pyclass(
    module = "molrs.io",
    name = "LammpsThermo",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsThermo {
    inner: LammpsThermo,
}

#[pymethods]
impl PyLammpsThermo {
    /// Column names in file order.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.columns.clone()
    }

    /// The whole table as an ``(n_rows, n_columns)`` float64 array.
    #[getter]
    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n_cols = self.inner.columns.len();
        if self.inner.rows.is_empty() {
            return Ok(PyArray2::zeros(py, [0, n_cols], false));
        }
        PyArray2::from_vec2(py, &self.inner.rows)
            .map_err(|e| PyValueError::new_err(format!("ragged thermo table: {e}")))
    }

    /// Number of thermo rows.
    #[getter]
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    /// The thermo lines as they appeared in the file.
    #[getter]
    fn raw_lines(&self) -> Vec<String> {
        self.inner.raw_lines.clone()
    }

    /// One column by name as a float64 array; ``KeyError`` for an unknown name.
    fn __getitem__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let Some(index) = self.inner.columns.iter().position(|c| c == name) else {
            return Err(PyKeyError::new_err(format!(
                "no thermo column {name:?}; columns are {:?}",
                self.inner.columns
            )));
        };
        let column: Vec<f64> = self.inner.rows.iter().map(|row| row[index]).collect();
        Ok(PyArray1::from_vec(py, column))
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.columns.iter().any(|c| c == name)
    }

    fn __len__(&self) -> usize {
        self.inner.n_rows()
    }

    fn __repr__(&self) -> String {
        format!(
            "LammpsThermo(columns={:?}, n_rows={})",
            self.inner.columns,
            self.inner.n_rows()
        )
    }
}

/// The ``Loop time of ...`` line of a run.
#[pyclass(
    module = "molrs.io",
    name = "LammpsLoopTime",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsLoopTime {
    inner: LammpsLoopTime,
}

#[pymethods]
impl PyLammpsLoopTime {
    #[getter]
    fn seconds(&self) -> f64 {
        self.inner.seconds
    }
    #[getter]
    fn procs(&self) -> i64 {
        self.inner.procs
    }
    #[getter]
    fn steps(&self) -> Option<i64> {
        self.inner.steps
    }
    #[getter]
    fn atoms(&self) -> Option<i64> {
        self.inner.atoms
    }
    #[getter]
    fn raw_line(&self) -> String {
        self.inner.raw_line.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "LammpsLoopTime(seconds={}, procs={}, steps={:?}, atoms={:?})",
            self.inner.seconds, self.inner.procs, self.inner.steps, self.inner.atoms
        )
    }
}

/// The ``Performance: ...`` line of a run.
#[pyclass(
    module = "molrs.io",
    name = "LammpsPerformance",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsPerformance {
    inner: LammpsPerformance,
}

#[pymethods]
impl PyLammpsPerformance {
    #[getter]
    fn ns_per_day(&self) -> f64 {
        self.inner.ns_per_day
    }
    #[getter]
    fn hours_per_ns(&self) -> f64 {
        self.inner.hours_per_ns
    }
    #[getter]
    fn timesteps_per_second(&self) -> f64 {
        self.inner.timesteps_per_second
    }
    #[getter]
    fn atom_steps_per_second(&self) -> Option<f64> {
        self.inner.atom_steps_per_second
    }
    #[getter]
    fn atom_steps_units(&self) -> Option<String> {
        self.inner.atom_steps_units.clone()
    }
    #[getter]
    fn raw_line(&self) -> String {
        self.inner.raw_line.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "LammpsPerformance(ns_per_day={}, timesteps_per_second={})",
            self.inner.ns_per_day, self.inner.timesteps_per_second
        )
    }
}

/// The ``... % CPU use with N MPI tasks x M OpenMP threads`` line of a run.
#[pyclass(
    module = "molrs.io",
    name = "LammpsCpuUse",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsCpuUse {
    inner: LammpsCpuUse,
}

#[pymethods]
impl PyLammpsCpuUse {
    #[getter]
    fn percent(&self) -> f64 {
        self.inner.percent
    }
    #[getter]
    fn mpi_tasks(&self) -> i64 {
        self.inner.mpi_tasks
    }
    #[getter]
    fn omp_threads(&self) -> Option<i64> {
        self.inner.omp_threads
    }
    #[getter]
    fn raw_line(&self) -> String {
        self.inner.raw_line.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "LammpsCpuUse(percent={}, mpi_tasks={}, omp_threads={:?})",
            self.inner.percent, self.inner.mpi_tasks, self.inner.omp_threads
        )
    }
}

/// One row of an MPI-task or thread timing breakdown.
#[pyclass(
    module = "molrs.io",
    name = "LammpsTimingRow",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsTimingRow {
    inner: LammpsTimingRow,
}

#[pymethods]
impl PyLammpsTimingRow {
    #[getter]
    fn section(&self) -> String {
        self.inner.section.clone()
    }
    #[getter]
    fn min_time(&self) -> f64 {
        self.inner.min_time
    }
    #[getter]
    fn avg_time(&self) -> f64 {
        self.inner.avg_time
    }
    #[getter]
    fn max_time(&self) -> f64 {
        self.inner.max_time
    }
    #[getter]
    fn percent_varavg(&self) -> f64 {
        self.inner.percent_varavg
    }
    #[getter]
    fn percent_total(&self) -> f64 {
        self.inner.percent_total
    }
    #[getter]
    fn raw_line(&self) -> String {
        self.inner.raw_line.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "LammpsTimingRow(section={:?}, avg_time={}, percent_total={})",
            self.inner.section, self.inner.avg_time, self.inner.percent_total
        )
    }
}

/// A timing breakdown table (``MPI task timing breakdown`` or thread timing).
#[pyclass(
    module = "molrs.io",
    name = "LammpsTimingBreakdown",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsTimingBreakdown {
    inner: LammpsTimingBreakdown,
}

#[pymethods]
impl PyLammpsTimingBreakdown {
    #[getter]
    fn title(&self) -> String {
        self.inner.title.clone()
    }
    #[getter]
    fn rows(&self) -> Vec<PyLammpsTimingRow> {
        self.inner
            .rows
            .iter()
            .map(|row| PyLammpsTimingRow { inner: row.clone() })
            .collect()
    }
    #[getter]
    fn raw_lines(&self) -> Vec<String> {
        self.inner.raw_lines.clone()
    }
    fn __len__(&self) -> usize {
        self.inner.rows.len()
    }
    fn __repr__(&self) -> String {
        format!(
            "LammpsTimingBreakdown(title={:?}, rows={})",
            self.inner.title,
            self.inner.rows.len()
        )
    }
}

/// One ``Nlocal`` / ``Nghost`` / ``Neighs`` load-balance block.
#[pyclass(
    module = "molrs.io",
    name = "LammpsLoadBalance",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsLoadBalance {
    inner: LammpsLoadBalance,
}

#[pymethods]
impl PyLammpsLoadBalance {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn average(&self) -> f64 {
        self.inner.average
    }
    #[getter]
    fn maximum(&self) -> f64 {
        self.inner.maximum
    }
    #[getter]
    fn minimum(&self) -> f64 {
        self.inner.minimum
    }
    #[getter]
    fn histogram(&self) -> Vec<i64> {
        self.inner.histogram.clone()
    }
    #[getter]
    fn raw_lines(&self) -> Vec<String> {
        self.inner.raw_lines.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "LammpsLoadBalance(name={:?}, average={}, min={}, max={})",
            self.inner.name, self.inner.average, self.inner.minimum, self.inner.maximum
        )
    }
}

/// The neighbor-list statistics block of a run.
#[pyclass(
    module = "molrs.io",
    name = "LammpsNeighborStatistics",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsNeighborStatistics {
    inner: LammpsNeighborStatistics,
}

#[pymethods]
impl PyLammpsNeighborStatistics {
    #[getter]
    fn total_neighbors(&self) -> Option<i64> {
        self.inner.total_neighbors
    }
    #[getter]
    fn ave_neighs_per_atom(&self) -> Option<f64> {
        self.inner.ave_neighs_per_atom
    }
    #[getter]
    fn ave_special_neighs_per_atom(&self) -> Option<f64> {
        self.inner.ave_special_neighs_per_atom
    }
    #[getter]
    fn neighbor_list_builds(&self) -> Option<i64> {
        self.inner.neighbor_list_builds
    }
    #[getter]
    fn dangerous_builds(&self) -> Option<i64> {
        self.inner.dangerous_builds
    }
    #[getter]
    fn raw_lines(&self) -> Vec<String> {
        self.inner.raw_lines.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "LammpsNeighborStatistics(total_neighbors={:?}, neighbor_list_builds={:?}, dangerous_builds={:?})",
            self.inner.total_neighbors,
            self.inner.neighbor_list_builds,
            self.inner.dangerous_builds
        )
    }
}

/// A ``WARNING:`` line, with where it appeared.
#[pyclass(
    module = "molrs.io",
    name = "LammpsWarning",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLammpsWarning {
    inner: LammpsWarning,
}

#[pymethods]
impl PyLammpsWarning {
    #[getter]
    fn message(&self) -> String {
        self.inner.message.clone()
    }
    #[getter]
    fn raw_line(&self) -> String {
        self.inner.raw_line.clone()
    }
    #[getter]
    fn line_number(&self) -> Option<usize> {
        self.inner.line_number
    }
    #[getter]
    fn run_index(&self) -> Option<usize> {
        self.inner.run_index
    }
    fn __repr__(&self) -> String {
        format!(
            "LammpsWarning(message={:?}, line_number={:?}, run_index={:?})",
            self.inner.message, self.inner.line_number, self.inner.run_index
        )
    }
}

/// One ``run`` command: its setup lines, thermo table, timing and warnings.
#[pyclass(module = "molrs.io", name = "LammpsRun", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyLammpsRun {
    inner: LammpsRun,
}

#[pymethods]
impl PyLammpsRun {
    /// Position of the run in the log, from 0.
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }
    /// Lines between the previous run and this run's thermo header.
    #[getter]
    fn setup_log(&self) -> Vec<String> {
        self.inner.setup_log.clone()
    }
    #[getter]
    fn memory(&self) -> Option<PyLammpsMemoryUsage> {
        self.inner
            .memory
            .clone()
            .map(|inner| PyLammpsMemoryUsage { inner })
    }
    #[getter]
    fn thermo(&self) -> Option<PyLammpsThermo> {
        self.inner
            .thermo
            .clone()
            .map(|inner| PyLammpsThermo { inner })
    }
    #[getter]
    fn loop_time(&self) -> Option<PyLammpsLoopTime> {
        self.inner
            .loop_time
            .clone()
            .map(|inner| PyLammpsLoopTime { inner })
    }
    #[getter]
    fn performance(&self) -> Option<PyLammpsPerformance> {
        self.inner
            .performance
            .clone()
            .map(|inner| PyLammpsPerformance { inner })
    }
    #[getter]
    fn cpu_use(&self) -> Option<PyLammpsCpuUse> {
        self.inner
            .cpu_use
            .clone()
            .map(|inner| PyLammpsCpuUse { inner })
    }
    #[getter]
    fn mpi_task_timing(&self) -> Option<PyLammpsTimingBreakdown> {
        self.inner
            .mpi_task_timing
            .clone()
            .map(|inner| PyLammpsTimingBreakdown { inner })
    }
    #[getter]
    fn thread_timing(&self) -> Option<PyLammpsTimingBreakdown> {
        self.inner
            .thread_timing
            .clone()
            .map(|inner| PyLammpsTimingBreakdown { inner })
    }
    #[getter]
    fn load_balance(&self) -> Vec<PyLammpsLoadBalance> {
        self.inner
            .load_balance
            .iter()
            .map(|lb| PyLammpsLoadBalance { inner: lb.clone() })
            .collect()
    }
    #[getter]
    fn neighbor_statistics(&self) -> Option<PyLammpsNeighborStatistics> {
        self.inner
            .neighbor_statistics
            .clone()
            .map(|inner| PyLammpsNeighborStatistics { inner })
    }
    #[getter]
    fn warnings(&self) -> Vec<PyLammpsWarning> {
        self.inner
            .warnings
            .iter()
            .map(|w| PyLammpsWarning { inner: w.clone() })
            .collect()
    }
    /// Lines of this run the parser did not classify.
    #[getter]
    fn unparsed_log(&self) -> Vec<String> {
        self.inner.unparsed_log.clone()
    }
    #[getter]
    fn raw_text(&self) -> String {
        self.inner.raw_text.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "LammpsRun(index={}, thermo={})",
            self.inner.index,
            match &self.inner.thermo {
                Some(t) => format!("{} rows", t.n_rows()),
                None => "None".to_string(),
            }
        )
    }
}

/// A parsed LAMMPS log: header, one `LammpsRun` per ``run``, and warnings.
#[pyclass(module = "molrs.io", name = "LammpsLog", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyLammpsLog {
    inner: LammpsLog,
}

impl PyLammpsLog {
    pub(crate) fn new(inner: LammpsLog) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyLammpsLog {
    /// The path the log was read from (``"<string>"`` when parsed from text).
    #[getter]
    fn path(&self) -> String {
        self.inner.path.clone()
    }
    /// The ``LAMMPS (...)`` banner, when present.
    #[getter]
    fn version(&self) -> Option<String> {
        self.inner.version.clone()
    }
    #[getter]
    fn header(&self) -> PyLammpsLogHeader {
        PyLammpsLogHeader {
            inner: self.inner.header.clone(),
        }
    }
    /// One entry per ``run``, in file order.
    #[getter]
    fn runs(&self) -> Vec<PyLammpsRun> {
        self.inner
            .runs
            .iter()
            .map(|run| PyLammpsRun { inner: run.clone() })
            .collect()
    }
    /// The ``Total wall time:`` value, when present.
    #[getter]
    fn total_wall_time(&self) -> Option<String> {
        self.inner.total_wall_time.clone()
    }
    /// Warnings outside any run.
    #[getter]
    fn warnings(&self) -> Vec<PyLammpsWarning> {
        self.inner
            .warnings
            .iter()
            .map(|w| PyLammpsWarning { inner: w.clone() })
            .collect()
    }
    #[getter]
    fn raw_text(&self) -> String {
        self.inner.raw_text.clone()
    }
    /// Thermo style the log was parsed with.
    #[getter]
    fn style(&self) -> String {
        self.inner.style.clone()
    }

    /// The whole log as nested plain Python values (JSON-friendly).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        super::lammps_log_to_pydict(py, &self.inner)
    }

    fn __len__(&self) -> usize {
        self.inner.runs.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "LammpsLog(path={:?}, runs={}, version={:?})",
            self.inner.path,
            self.inner.runs.len(),
            self.inner.version
        )
    }
}
