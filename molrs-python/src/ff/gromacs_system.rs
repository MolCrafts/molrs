//! `molrs.ff.write_gromacs_system`: the writer half of
//! `read_gromacs_system`, over `GromacsTopFfWriter::write_system_str`.

use std::path::PathBuf;

use molrs::ff::forcefield::writers::gromacs::GromacsTopFfWriter;
use pyo3::prelude::*;

use crate::core::store::frame::PyFrame;
use crate::ff::PyForceField;
use crate::helpers::{io_error_to_pyerr, path_str};

/// Write a force field and a typed frame as one GROMACS topology — the
/// inverse of :func:`read_gromacs_system`.
///
/// The force-field directives as :func:`write_gromacs_top_ff` writes them,
/// then one ``[ moleculetype ]`` per molecule of ``frame`` (its ``atoms``
/// ``type`` / ``charge`` / ``mass`` and the relation blocks, typed by the
/// force field's labels), ``[ system ]`` and ``[ molecules ]``. Coordinates
/// go to a ``.gro`` (:func:`molrs.io.write_gro`).
///
/// Parameters
/// ----------
/// path : str | os.PathLike
///     The ``.top`` file to write.
/// forcefield : ForceField
///     Holds a type for every label ``frame`` uses.
/// frame : Frame
///     Typed system (string ``type`` columns).
/// precision : int
///     Decimal places for floating coefficients.
///
/// Raises
/// ------
/// ValueError
///     A style, parameter or row GROMACS cannot express, or a label the force
///     field lacks.
/// OSError
///     The file cannot be written.
#[pyfunction]
#[pyo3(signature = (path, forcefield, frame, *, precision = 6))]
pub fn write_gromacs_system(
    path: PathBuf,
    forcefield: PyRef<'_, PyForceField>,
    frame: &PyFrame,
    precision: usize,
) -> PyResult<()> {
    let text = frame
        .with_frame(|f| {
            GromacsTopFfWriter::new()
                .with_precision(precision)
                .write_system_str(&forcefield.inner, f)
        })?
        .map_err(crate::ff::ir::write_err)?;
    std::fs::write(path_str(&path)?, text).map_err(io_error_to_pyerr)
}
