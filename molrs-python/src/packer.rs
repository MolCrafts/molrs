//! Python wrapper for the Molpack molecular packer.
//!
//! This module provides [`PyPacker`] (a Packmol-grade molecular packer) and
//! [`PyPackResult`] (the output container). The packing algorithm uses a
//! GENCAN optimizer to minimize inter-molecular overlap subject to geometric
//! constraints.
//!
//! # Algorithm Phases
//!
//! 1. **Phase 0** -- Per-type sequential packing.
//! 2. **Phase 1** -- Geometric constraint fitting.
//! 3. **Phase 2** -- Main loop with inflated tolerance and movebad heuristic.
//!
//! # References
//!
//! - Martinez, L. et al. (2009). J. Comput. Chem. 30, 2157-2164.
//!   doi:10.1002/jcc.21224 (Packmol)

use crate::frame::PyFrame;
use crate::helpers::{pack_error_to_pyerr, NpF};
use crate::target::PyTarget;
use molrs_pack::handler::ProgressHandler;
use molrs_pack::packer::{Molpack, PackResult};
use molrs_pack::F;
use numpy::IntoPyArray;
use numpy::PyArray2;
use pyo3::prelude::*;

/// Result container for a packing run.
///
/// Exposed to Python as `molrs.PackResult`.
///
/// Contains the final positions, element symbols, convergence status, and
/// constraint violation metrics.
///
/// Attributes
/// ----------
/// positions : numpy.ndarray, shape (N, 3), dtype float
///     Final atom positions after packing.
/// elements : list[str]
///     Element symbol for each atom, matching the order of ``positions``.
/// frame : Frame
///     Frame with ``"atoms"`` block containing ``x``, ``y``, ``z``,
///     ``element``, and ``mol_id`` columns.
/// converged : bool
///     Whether the packing converged below the tolerance.
/// fdist : float
///     Maximum inter-molecular distance violation.
/// frest : float
///     Maximum constraint violation.
///
/// Examples
/// --------
/// >>> result = packer.pack(targets)
/// >>> result.converged
/// True
/// >>> result.positions.shape
/// (1500, 3)
#[pyclass(name = "PackResult", from_py_object)]
#[derive(Clone)]
pub struct PyPackResult {
    inner: PackResult,
}

#[pymethods]
impl PyPackResult {
    /// Final atom positions as an ``(N, 3)`` array.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N, 3), dtype float
    ///     Cartesian coordinates of all atoms.
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        let pos = self.inner.positions();
        let n = pos.len();
        let flat: Vec<F> = pos.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        let arr = ndarray::Array2::from_shape_vec((n, 3), flat).expect("positions shape");
        arr.into_pyarray(py)
    }

    /// Element symbols for each atom, in the same order as ``positions``.
    ///
    /// Returns
    /// -------
    /// list[str]
    #[getter]
    fn elements(&self) -> Vec<String> {
        let atoms = self.inner.frame.get("atoms").expect("no atoms block");
        atoms
            .get_string("element")
            .expect("no element column")
            .iter()
            .cloned()
            .collect()
    }

    /// Frame with the packed system.
    ///
    /// The ``"atoms"`` block contains columns ``x``, ``y``, ``z`` (float),
    /// ``element`` (str), and ``mol_id`` (int, zero-based molecule index).
    ///
    /// Returns
    /// -------
    /// Frame
    #[getter]
    fn frame(&self) -> PyResult<PyFrame> {
        PyFrame::from_core_frame(self.inner.frame.clone())
    }

    /// Whether the packing converged within the requested tolerance.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Maximum inter-molecular distance violation.
    ///
    /// A value near zero indicates all molecules are well-separated.
    ///
    /// Returns
    /// -------
    /// float
    #[getter]
    fn fdist(&self) -> F {
        self.inner.fdist
    }

    /// Maximum geometric constraint violation.
    ///
    /// A value near zero indicates all constraints are satisfied.
    ///
    /// Returns
    /// -------
    /// float
    #[getter]
    fn frest(&self) -> F {
        self.inner.frest
    }

    /// Number of atoms in the packed system.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn natoms(&self) -> usize {
        self.inner.natoms()
    }

    fn __repr__(&self) -> String {
        format!(
            "PackResult(converged={}, fdist={:.4}, frest={:.4}, natoms={})",
            self.inner.converged,
            self.inner.fdist,
            self.inner.frest,
            self.inner.natoms()
        )
    }
}

/// Molecular packer implementing the Packmol algorithm.
///
/// Exposed to Python as `molrs.Packer`.
///
/// Builder-style configuration via :meth:`with_tolerance`,
/// :meth:`with_precision`, :meth:`with_maxit`, and :meth:`with_progress`.
/// All builder methods return **new** `Packer` instances.
///
/// Parameters
/// ----------
/// tolerance : float, optional
///     Minimum inter-atomic distance between different molecules.
///     Default ``2.0`` (angstroms).
/// precision : float, optional
///     Convergence precision for the GENCAN optimizer.
///     Default ``0.01``.
///
/// Examples
/// --------
/// >>> packer = Packer(tolerance=2.0, precision=0.01)
/// >>> packer = packer.with_maxit(30)
/// >>> result = packer.pack(targets, max_loops=200, seed=42)
/// >>> result.converged
/// True
#[pyclass(name = "Packer")]
pub struct PyPacker {
    tolerance: F,
    precision: F,
    maxit: usize,
    nloop0: usize,
    sidemax: F,
    movefrac: F,
    movebadrandom: bool,
    disable_movebad: bool,
    pbc: Option<([F; 3], [F; 3])>,
    progress: bool,
}

#[pymethods]
impl PyPacker {
    /// Create a new packer with default parameters.
    ///
    /// Parameters
    /// ----------
    /// tolerance : float, optional
    ///     Minimum inter-molecular distance. Default ``2.0``.
    /// precision : float, optional
    ///     GENCAN convergence precision. Default ``0.01``.
    ///
    /// Returns
    /// -------
    /// Packer
    #[new]
    #[pyo3(signature = (tolerance=2.0, precision=0.01))]
    fn new(tolerance: NpF, precision: NpF) -> Self {
        PyPacker {
            tolerance,
            precision,
            maxit: 20,
            nloop0: 0,
            sidemax: 1000.0,
            movefrac: 0.05,
            movebadrandom: false,
            disable_movebad: false,
            pbc: None,
            progress: true,
        }
    }

    /// Return a new packer with updated tolerance. The original is unchanged.
    ///
    /// Parameters
    /// ----------
    /// tolerance : float
    ///     Minimum inter-molecular distance.
    ///
    /// Returns
    /// -------
    /// Packer
    fn with_tolerance(&self, tolerance: NpF) -> Self {
        PyPacker {
            tolerance,
            precision: self.precision,
            maxit: self.maxit,
            nloop0: self.nloop0,
            sidemax: self.sidemax,
            movefrac: self.movefrac,
            movebadrandom: self.movebadrandom,
            disable_movebad: self.disable_movebad,
            pbc: self.pbc,
            progress: self.progress,
        }
    }

    /// Return a new packer with updated precision. The original is unchanged.
    ///
    /// Parameters
    /// ----------
    /// precision : float
    ///     GENCAN convergence precision.
    ///
    /// Returns
    /// -------
    /// Packer
    fn with_precision(&self, precision: NpF) -> Self {
        PyPacker {
            tolerance: self.tolerance,
            precision,
            maxit: self.maxit,
            nloop0: self.nloop0,
            sidemax: self.sidemax,
            movefrac: self.movefrac,
            movebadrandom: self.movebadrandom,
            disable_movebad: self.disable_movebad,
            pbc: self.pbc,
            progress: self.progress,
        }
    }

    /// Return a new packer with updated GENCAN inner iteration count.
    /// The original is unchanged.
    ///
    /// Parameters
    /// ----------
    /// maxit : int
    ///     Maximum GENCAN inner iterations per outer loop.
    ///
    /// Returns
    /// -------
    /// Packer
    fn with_maxit(&self, maxit: usize) -> Self {
        PyPacker {
            tolerance: self.tolerance,
            precision: self.precision,
            maxit,
            nloop0: self.nloop0,
            sidemax: self.sidemax,
            movefrac: self.movefrac,
            movebadrandom: self.movebadrandom,
            disable_movebad: self.disable_movebad,
            pbc: self.pbc,
            progress: self.progress,
        }
    }

    /// Return a new packer with updated initialization outer loop count.
    ///
    /// Parameters
    /// ----------
    /// nloop0 : int
    ///     Packmol ``nloop0`` value. Use ``0`` for Packmol default.
    ///
    /// Returns
    /// -------
    /// Packer
    fn with_nloop0(&self, nloop0: usize) -> Self {
        PyPacker {
            tolerance: self.tolerance,
            precision: self.precision,
            maxit: self.maxit,
            nloop0,
            sidemax: self.sidemax,
            movefrac: self.movefrac,
            movebadrandom: self.movebadrandom,
            disable_movebad: self.disable_movebad,
            pbc: self.pbc,
            progress: self.progress,
        }
    }

    /// Return a new packer with updated initial global half-size.
    fn with_sidemax(&self, sidemax: NpF) -> Self {
        PyPacker {
            tolerance: self.tolerance,
            precision: self.precision,
            maxit: self.maxit,
            nloop0: self.nloop0,
            sidemax,
            movefrac: self.movefrac,
            movebadrandom: self.movebadrandom,
            disable_movebad: self.disable_movebad,
            pbc: self.pbc,
            progress: self.progress,
        }
    }

    /// Return a new packer with updated movebad move fraction.
    fn with_movefrac(&self, movefrac: NpF) -> Self {
        PyPacker {
            tolerance: self.tolerance,
            precision: self.precision,
            maxit: self.maxit,
            nloop0: self.nloop0,
            sidemax: self.sidemax,
            movefrac,
            movebadrandom: self.movebadrandom,
            disable_movebad: self.disable_movebad,
            pbc: self.pbc,
            progress: self.progress,
        }
    }

    /// Return a new packer with Packmol ``movebadrandom`` enabled or disabled.
    fn with_movebadrandom(&self, enabled: bool) -> Self {
        PyPacker {
            tolerance: self.tolerance,
            precision: self.precision,
            maxit: self.maxit,
            nloop0: self.nloop0,
            sidemax: self.sidemax,
            movefrac: self.movefrac,
            movebadrandom: enabled,
            disable_movebad: self.disable_movebad,
            pbc: self.pbc,
            progress: self.progress,
        }
    }

    /// Return a new packer with Packmol ``disable_movebad`` enabled or disabled.
    fn with_disable_movebad(&self, disabled: bool) -> Self {
        PyPacker {
            tolerance: self.tolerance,
            precision: self.precision,
            maxit: self.maxit,
            nloop0: self.nloop0,
            sidemax: self.sidemax,
            movefrac: self.movefrac,
            movebadrandom: self.movebadrandom,
            disable_movebad: disabled,
            pbc: self.pbc,
            progress: self.progress,
        }
    }

    /// Return a new packer with periodic box bounds.
    fn with_pbc(&self, min: [NpF; 3], max: [NpF; 3]) -> Self {
        PyPacker {
            tolerance: self.tolerance,
            precision: self.precision,
            maxit: self.maxit,
            nloop0: self.nloop0,
            sidemax: self.sidemax,
            movefrac: self.movefrac,
            movebadrandom: self.movebadrandom,
            disable_movebad: self.disable_movebad,
            pbc: Some((min, max)),
            progress: self.progress,
        }
    }

    /// Return a new packer with a periodic box starting at the origin.
    fn with_pbc_box(&self, lengths: [NpF; 3]) -> Self {
        self.with_pbc([0.0, 0.0, 0.0], lengths)
    }

    /// Return a new packer with progress output enabled or disabled.
    /// The original is unchanged.
    ///
    /// Parameters
    /// ----------
    /// enabled : bool
    ///     ``True`` to print progress to stdout.
    ///
    /// Returns
    /// -------
    /// Packer
    fn with_progress(&self, enabled: bool) -> Self {
        PyPacker {
            tolerance: self.tolerance,
            precision: self.precision,
            maxit: self.maxit,
            nloop0: self.nloop0,
            sidemax: self.sidemax,
            movefrac: self.movefrac,
            movebadrandom: self.movebadrandom,
            disable_movebad: self.disable_movebad,
            pbc: self.pbc,
            progress: enabled,
        }
    }

    /// Run the packing algorithm.
    ///
    /// Parameters
    /// ----------
    /// targets : list[Target]
    ///     Molecules to pack, each specifying template geometry, copy count,
    ///     and optional constraints.
    /// max_loops : int, optional
    ///     Maximum outer optimization loops. Default ``200``.
    /// seed : int | None, optional
    ///     Random seed for initial placement. ``None`` uses the deterministic
    ///     default seed ``0``.
    ///
    /// Returns
    /// -------
    /// PackResult
    ///     Container with packed coordinates and convergence info.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If packing encounters an unrecoverable error (e.g. zero atoms).
    ///
    /// Examples
    /// --------
    /// >>> result = packer.pack([water_target, ethanol_target], max_loops=200, seed=42)
    /// >>> result.converged
    /// True
    #[pyo3(signature = (targets, max_loops=200, seed=None))]
    fn pack(
        &self,
        targets: Vec<PyTarget>,
        max_loops: usize,
        seed: Option<u64>,
    ) -> PyResult<PyPackResult> {
        let rust_targets: Vec<_> = targets.into_iter().map(|t| t.inner).collect();

        let mut packer = Molpack::new()
            .tolerance(self.tolerance)
            .precision(self.precision)
            .maxit(self.maxit)
            .nloop0(self.nloop0)
            .sidemax(self.sidemax)
            .movefrac(self.movefrac)
            .movebadrandom(self.movebadrandom)
            .disable_movebad(self.disable_movebad);

        if let Some((min, max)) = self.pbc {
            packer = packer.pbc(min, max);
        }

        if self.progress {
            packer = packer.add_handler(ProgressHandler::new());
        }

        let result = packer
            .pack(&rust_targets, max_loops, seed)
            .map_err(pack_error_to_pyerr)?;

        Ok(PyPackResult { inner: result })
    }

    fn __repr__(&self) -> String {
        format!(
            "Packer(tolerance={:.2}, precision={:.4})",
            self.tolerance, self.precision
        )
    }
}
