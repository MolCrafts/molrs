//! Potentials of mean force and torque (`molrs::compute::pmft`): `PMFTXY`.

#![allow(clippy::type_complexity)]

use super::order::orientation_pairs;
use super::{collect_frames, collect_neighbors};
use crate::error::py_value_err;
use molrs::compute::{Compute, PMFTXY, PMFTXYArgs};
use molrs::op::types::F;
use molrs::store::{Frame as CoreFrame, FrameAccess};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// PMFTXY
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "PMFTXY")]
pub struct PyPMFTXY {
    inner: PMFTXY,
}

#[pymethods]
impl PyPMFTXY {
    #[new]
    fn new(x_max: f64, y_max: f64, n_x: usize, n_y: usize) -> PyResult<Self> {
        Ok(Self {
            inner: PMFTXY::new(x_max, y_max, n_x, n_y).map_err(py_value_err)?,
        })
    }

    /// Returns per-frame `(raw_counts, density, pmf)`. If the first frame carries
    /// an `"orientations"` topology block (one `(head, tail)` atom pair per query
    /// particle), every bond is rotated into that particle's local frame — the
    /// per-particle 2-D angle is `atan2` of its `head − tail` axis, recomputed per
    /// frame from that frame's positions. Without the block the analyzer works in
    /// the lab frame.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<
        Vec<(
            Bound<'py, PyArray2<u64>>,
            Bound<'py, PyArray2<f64>>,
            Bound<'py, PyArray2<f64>>,
        )>,
    > {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let neighbors = collect_neighbors(nlists)?;
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        // Per-frame per-particle orientation angles derived from the frame's
        // `orientations` block, or `None` (lab frame) when the block is absent.
        let orient_angles: Option<Vec<Vec<F>>> = if first.contains_block("orientations") {
            let pairs = orientation_pairs(first)?;
            let mut per_frame = Vec::with_capacity(refs.len());
            for f in &refs {
                let xyz = f.coords().map_err(py_value_err)?;
                let n = xyz.nrows();
                let mut angles = Vec::with_capacity(pairs.len());
                for &(head, tail) in &pairs {
                    if head >= n || tail >= n {
                        return Err(PyValueError::new_err(
                            "orientations atom index out of range",
                        ));
                    }
                    angles.push(
                        (xyz[[head, 1]] - xyz[[tail, 1]]).atan2(xyz[[head, 0]] - xyz[[tail, 0]]),
                    );
                }
                per_frame.push(angles);
            }
            Some(per_frame)
        } else {
            None
        };
        let args = PMFTXYArgs {
            nlists: &neighbors,
            query_orientations: orient_angles.as_deref(),
        };
        let results = self.inner.compute(&refs, args).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                (
                    r.raw_counts.into_pyarray(py),
                    r.density.into_pyarray(py),
                    r.pmf.into_pyarray(py),
                )
            })
            .collect())
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPMFTXY>()?;
    Ok(())
}
