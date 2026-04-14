//! I/O functions for reading and writing molecular data files, and parsing
//! SMILES notation.
//!
//! ## Supported formats
//!
//! | Format | Read | Write |
//! |--------|------|-------|
//! | PDB | [`read_pdb`] | [`write_pdb`] |
//! | XYZ | [`read_xyz`], [`read_xyz_traj`] | [`write_xyz`] |
//! | LAMMPS data | [`read_lammps`] | [`write_lammps`] |
//! | LAMMPS dump | [`read_lammps_traj`] | [`write_lammps_traj`] |

use crate::frame::PyFrame;
use crate::helpers::{io_error_to_pyerr, molrs_error_to_pyerr, smiles_error_to_pyerr};
use crate::molgraph::PyAtomistic;
use molrs_io::chgcar::read_chgcar;
use molrs_io::cube::{read_cube, write_cube};
use molrs_io::lammps_data::{read_lammps_data, write_lammps_data};
use molrs_io::lammps_dump::{read_lammps_dump, write_lammps_dump};
use molrs_io::pdb::{read_pdb_frame, write_pdb_frame};
use molrs_io::xyz::{read_xyz_frame, read_xyz_traj, write_xyz_frame};
use pyo3::prelude::*;
use std::fs::File;
use std::io::BufWriter;

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

/// Read an XYZ file and return a single Frame.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.xyz`` file on disk.
///
/// Returns
/// -------
/// Frame
#[pyfunction]
pub fn read_xyz(path: &str) -> PyResult<PyFrame> {
    let frame = read_xyz_frame(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read all frames from an XYZ trajectory file.
///
/// Parameters
/// ----------
/// path : str
///     Path to a multi-frame ``.xyz`` file.
///
/// Returns
/// -------
/// list[Frame]
#[pyfunction]
pub fn read_xyz_trajectory(path: &str) -> PyResult<Vec<PyFrame>> {
    let frames = read_xyz_traj(path).map_err(io_error_to_pyerr)?;
    frames.into_iter().map(PyFrame::from_core_frame).collect()
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
    frames.into_iter().map(PyFrame::from_core_frame).collect()
}

/// Read a VASP CHGCAR or CHGDIF file.
///
/// Returns a Frame containing:
///
/// - ``"atoms"`` block with ``symbol``, ``x``, ``y``, ``z`` (Cartesian Å)
/// - ``simbox``: triclinic periodic box
/// - grid ``"chgcar"``: a :class:`Grid` with at least ``"total"`` (and
///   ``"diff"`` for spin-polarised ISPIN=2 calculations)
///
/// The volumetric values are stored **raw** (ρ × V_cell, units e).
/// Divide by ``simbox.volume()`` to get charge density in e/Å³.
///
/// Parameters
/// ----------
/// path : str
///     Path to a CHGCAR or CHGDIF file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// ValueError
///     On parse errors.
/// IOError
///     If the file cannot be opened.
///
/// Examples
/// --------
/// >>> frame = molrs.read_chgcar("CHGCAR")
/// >>> grid = frame["chgcar"]
/// >>> total = grid["total"]          # shape (nx, ny, nz)
/// >>> density = total / frame.simbox.volume()
#[pyfunction]
pub fn read_chgcar_file(path: &str) -> PyResult<PyFrame> {
    let frame = read_chgcar(path).map_err(molrs_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read a Gaussian Cube file.
///
/// Returns a Frame containing:
///
/// - ``"atoms"`` block with ``element``, ``x``, ``y``, ``z``,
///   ``atomic_number``, ``charge``
/// - grid ``"cube"``: a :class:`Grid` with ``"density"`` (scalar field)
///   or ``"mo_<idx>"`` arrays (MO variant)
///
/// Values are stored as-is from the file (no unit conversion).
/// The unit system is recorded in ``frame.meta["cube_units"]``
/// (``"bohr"`` or ``"angstrom"``).
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.cube`` file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// ValueError
///     On parse errors.
/// IOError
///     If the file cannot be opened.
///
/// Examples
/// --------
/// >>> frame = molrs.read_cube_file("density.cube")
/// >>> grid = frame["cube"]
/// >>> density = grid["density"]       # shape (nx, ny, nz)
#[pyfunction]
pub fn read_cube_file(path: &str) -> PyResult<PyFrame> {
    let frame = read_cube(path).map_err(molrs_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a Frame to a Gaussian Cube file.
///
/// The Frame must contain a ``"cube"`` grid and an ``"atoms"`` block.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_cube_file(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    write_cube(path, &core_frame).map_err(molrs_error_to_pyerr)
}

// ============================================================================
// Writers
// ============================================================================

/// Write a Frame to a PDB file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_pdb(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    let file = File::create(path).map_err(io_error_to_pyerr)?;
    let mut buf = BufWriter::new(file);
    write_pdb_frame(&mut buf, &core_frame).map_err(io_error_to_pyerr)
}

/// Write a Frame to an XYZ file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_xyz(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    let file = File::create(path).map_err(io_error_to_pyerr)?;
    let mut buf = BufWriter::new(file);
    write_xyz_frame(&mut buf, &core_frame).map_err(io_error_to_pyerr)
}

/// Write a Frame to a LAMMPS data file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_lammps(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    write_lammps_data(path, &core_frame).map_err(io_error_to_pyerr)
}

/// Write Frames to a LAMMPS dump trajectory file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
///     Frames to write.
#[pyfunction]
pub fn write_lammps_traj(path: &str, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    write_lammps_dump(path, &core_frames).map_err(io_error_to_pyerr)
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
    inner: molrs_smiles::SmilesIR,
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
        let mol = molrs_smiles::to_atomistic(&self.inner).map_err(smiles_error_to_pyerr)?;
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
    let ir = molrs_smiles::parse_smiles(smiles).map_err(smiles_error_to_pyerr)?;
    Ok(PySmilesIR {
        inner: ir,
        input: smiles.to_owned(),
    })
}
