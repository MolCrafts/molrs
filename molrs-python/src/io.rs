//! I/O functions for reading molecular data files and parsing SMILES notation.
//!
//! Provides free functions to load PDB and XYZ files into [`PyFrame`]s, and
//! a SMILES parser that produces an intermediate representation convertible to
//! an [`PyAtomistic`] molecular graph.

use crate::frame::PyFrame;
use crate::helpers::{io_error_to_pyerr, smiles_error_to_pyerr};
use crate::molgraph::PyAtomistic;
use molrs::io::lammps_data::read_lammps_data;
use molrs::io::lammps_dump::read_lammps_dump;
use molrs::io::pdb::read_pdb_frame;
use molrs::io::xyz::read_xyz_frame;
use pyo3::prelude::*;

/// Read a PDB file and return a Frame.
///
/// The resulting frame contains an ``"atoms"`` block with columns ``symbol``
/// (str), ``x``/``y``/``z`` (float), ``name`` (str), ``resname`` (str), and
/// ``resid`` (int). If CRYST1 records are present a ``Box`` is also attached.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.pdb`` file on disk.
///
/// Returns
/// -------
/// Frame
///     Parsed molecular data.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
///
/// Examples
/// --------
/// >>> frame = molrs.read_pdb("molecule.pdb")
/// >>> atoms = frame["atoms"]
/// >>> symbols = atoms.view("symbol")
#[pyfunction]
pub fn read_pdb(path: &str) -> PyResult<PyFrame> {
    let frame = read_pdb_frame(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an XYZ file and return a Frame.
///
/// The resulting frame contains an ``"atoms"`` block with columns ``symbol``
/// (str) and ``x``/``y``/``z`` (float).
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.xyz`` file on disk.
///
/// Returns
/// -------
/// Frame
///     Parsed molecular data.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
///
/// Examples
/// --------
/// >>> frame = molrs.read_xyz("molecule.xyz")
/// >>> n_atoms = frame["atoms"].nrows
#[pyfunction]
pub fn read_xyz(path: &str) -> PyResult<PyFrame> {
    let frame = read_xyz_frame(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read a LAMMPS data file and return a Frame.
///
/// Parameters
/// ----------
/// path : str
///     Path to a LAMMPS data file on disk.
///
/// Returns
/// -------
/// Frame
///     Parsed molecular data with atoms, bonds, and box metadata.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
///
/// Examples
/// --------
/// >>> frame = molrs.read_lammps_data("system.data")
/// >>> atoms = frame["atoms"]
#[pyfunction]
pub fn read_lammps(path: &str) -> PyResult<PyFrame> {
    let frame = read_lammps_data(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read a LAMMPS dump trajectory file and return a list of Frames.
///
/// Parameters
/// ----------
/// path : str
///     Path to a LAMMPS dump file (e.g. ``.lammpstrj``) on disk.
///
/// Returns
/// -------
/// list[Frame]
///     All frames in the trajectory.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
///
/// Examples
/// --------
/// >>> frames = molrs.read_lammps_dump("trajectory.lammpstrj")
/// >>> len(frames)
/// 100
#[pyfunction]
pub fn read_lammps_traj(path: &str) -> PyResult<Vec<PyFrame>> {
    let frames = read_lammps_dump(path).map_err(io_error_to_pyerr)?;
    frames
        .into_iter()
        .map(PyFrame::from_core_frame)
        .collect()
}

/// Intermediate representation of a parsed SMILES or SMARTS string.
///
/// This is the raw syntax tree produced by the parser. Convert it to a
/// molecular graph via :meth:`to_atomistic`.
///
/// Attributes
/// ----------
/// n_components : int
///     Number of disconnected components (fragments separated by ``'.'``
///     in the SMILES string).
///
/// Examples
/// --------
/// >>> ir = molrs.parse_smiles("CCO")
/// >>> ir.n_components
/// 1
/// >>> mol = ir.to_atomistic()
/// >>> mol.n_atoms
/// 3
#[pyclass(name = "SmilesIR", unsendable)]
pub struct PySmilesIR {
    inner: molrs::smiles::SmilesIR,
    input: String,
}

#[pymethods]
impl PySmilesIR {
    /// Number of disconnected molecular components.
    ///
    /// Fragments separated by ``'.'`` in the SMILES string are counted as
    /// separate components.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn n_components(&self) -> usize {
        self.inner.components.len()
    }

    /// Convert the SMILES intermediate representation to an all-atom
    /// molecular graph.
    ///
    /// Hydrogen atoms that are implicit in the SMILES string are **not**
    /// added here; use :func:`generate_3d` with ``add_hydrogens=True`` for
    /// that.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     Molecular graph with atoms and bonds.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the IR contains invalid ring-closure or stereochemistry data.
    ///
    /// Examples
    /// --------
    /// >>> mol = parse_smiles("c1ccccc1").to_atomistic()
    /// >>> mol.n_atoms
    /// 6
    fn to_atomistic(&self) -> PyResult<PyAtomistic> {
        let mol = molrs::smiles::to_atomistic(&self.inner).map_err(smiles_error_to_pyerr)?;
        Ok(PyAtomistic { inner: mol })
    }

    fn __repr__(&self) -> String {
        format!(
            "SmilesIR('{}', components={})",
            self.input,
            self.inner.components.len()
        )
    }
}

/// Parse a SMILES string into an intermediate representation.
///
/// The returned :class:`SmilesIR` can be converted to an :class:`Atomistic`
/// molecular graph via :meth:`SmilesIR.to_atomistic`.
///
/// Parameters
/// ----------
/// smiles : str
///     SMILES string (e.g. ``"CCO"`` for ethanol, ``"c1ccccc1"`` for benzene).
///
/// Returns
/// -------
/// SmilesIR
///     Parsed intermediate representation.
///
/// Raises
/// ------
/// ValueError
///     If the SMILES string is syntactically invalid.
///
/// Examples
/// --------
/// >>> ir = molrs.parse_smiles("CCO")
/// >>> mol = ir.to_atomistic()
/// >>> mol.n_atoms
/// 3
#[pyfunction]
pub fn parse_smiles(smiles: &str) -> PyResult<PySmilesIR> {
    let ir = molrs::smiles::parse_smiles(smiles).map_err(smiles_error_to_pyerr)?;
    Ok(PySmilesIR {
        inner: ir,
        input: smiles.to_owned(),
    })
}
