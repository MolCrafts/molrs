//! Python binding of the unit-level point sequence
//! (`molrs::core::spatial::trace`).

use molrs::spatial::Trace;
use ndarray::{Array1, Array2};
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLikeDyn};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

use crate::helpers::molrs_error_to_pyerr;
use crate::op::points_from_array;

/// Ordered points split into consecutive non-empty units, with an optional
/// unit-length direction hint per unit. No chemistry.
///
/// Parameters
/// ----------
/// points : array_like, shape (n, 3), float64
///     Every point, in order (Å).
/// offsets : list[int], optional
///     Unit ``i`` is ``points[offsets[i]:offsets[i + 1]]``. Default: one point
///     per unit.
/// hints : array_like, shape (n_units, 3), optional
///     One direction per unit, stored normalised.
///
/// Raises
/// ------
/// ValueError
///     On a wrong shape, offsets that do not start at 0, end at the point
///     count and strictly increase, or a hint that is not a direction.
#[pyclass(module = "molrs", name = "Trace", frozen, skip_from_py_object)]
pub struct PyTrace {
    pub(crate) inner: Trace,
}

impl PyTrace {
    fn check_unit(&self, i: usize) -> PyResult<()> {
        if i < self.inner.n_units() {
            return Ok(());
        }
        Err(PyIndexError::new_err(format!(
            "unit {i} is out of range for a trace of {} units",
            self.inner.n_units()
        )))
    }
}

#[pymethods]
impl PyTrace {
    #[new]
    #[pyo3(signature = (points, offsets=None, hints=None))]
    fn new(
        points: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
        offsets: Option<Vec<usize>>,
        hints: Option<PyArrayLikeDyn<'_, f64, AllowTypeChange>>,
    ) -> PyResult<Self> {
        let points = points_from_array(&points, "points")?;
        let trace = match offsets {
            None => Trace::from_points(points),
            Some(offsets) => Trace::ragged(points, offsets).map_err(molrs_error_to_pyerr)?,
        };
        let inner = match hints {
            None => trace,
            Some(hints) => trace
                .with_hints(points_from_array(&hints, "hints")?)
                .map_err(molrs_error_to_pyerr)?,
        };
        Ok(Self { inner })
    }

    /// Number of units.
    #[getter]
    fn n_units(&self) -> usize {
        self.inner.n_units()
    }

    /// Unit ``i``'s points, shape ``(k, 3)``.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If ``i`` is not a unit index.
    fn unit<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_unit(i)?;
        let points = self.inner.unit(i).unwrap_or_default();
        Ok(Array2::from_shape_fn((points.len(), 3), |(p, a)| points[p][a]).into_pyarray(py))
    }

    /// Unit ``i``'s unit-length hint, shape ``(3,)``, or ``None`` when the
    /// trace has no hints.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If ``i`` is not a unit index.
    fn hint<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        self.check_unit(i)?;
        Ok(self
            .inner
            .hint(i)
            .map(|h| Array1::from(h.to_vec()).into_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        format!(
            "<Trace n_units={} n_points={}>",
            self.inner.n_units(),
            self.inner.points().len()
        )
    }
}
