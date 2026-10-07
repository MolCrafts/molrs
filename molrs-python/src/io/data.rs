//! Structure file formats (`molrs::io::data`): one or a few frames in, one
//! or a few frames out — PDB, XYZ, GRO, LAMMPS data and molecule templates,
//! MOL2, XSF, CHGCAR, cube, AMBER inpcrd / prmtop structure, antechamber
//! ``.ac`` and ``prep``. Every reader emits the canonical column names
//! (`molrs.core.keys`); the format's own spelling never reaches Python.

use crate::core::frame::PyFrame;
use crate::error::{io_error_to_pyerr, molrs_error_to_pyerr};
use crate::path::path_str;
use molrs::io::data::ac::read_ac as read_ac_rs;
use molrs::io::data::chgcar::read_chgcar as read_chgcar_rs;
use molrs::io::data::cube::{read_cube as read_cube_rs, write_cube as write_cube_rs};
use molrs::io::data::gro::{
    read_gro as read_gro_rs, write_gro as write_gro_rs, write_gro_traj as write_gro_traj_rs,
};
use molrs::io::data::inpcrd::read_amber_inpcrd as read_amber_inpcrd_rs;
use molrs::io::data::lammps_data::{
    read_lammps_data as read_lammps_data_rs, write_lammps_data as write_lammps_data_rs,
};
use molrs::io::data::lammps_molecule::{
    read_lammps_molecule as read_lammps_molecule_rs,
    write_lammps_molecule as write_lammps_molecule_rs,
};
use molrs::io::data::mol2::{read_mol2 as read_mol2_rs, write_mol2 as write_mol2_rs};
use molrs::io::data::pdb::{read_pdb_frame, read_pdb_traj, write_pdb_frame, write_pdb_traj};
use molrs::io::data::prep::{
    PrepAtom, PrepResidue, read_prep as read_prep_rs, write_prep as write_prep_rs,
};
use molrs::io::data::prmtop::read_amber_prmtop as read_amber_prmtop_rs;
use molrs::io::data::xsf::{read_xsf as read_xsf_rs, write_xsf as write_xsf_rs};
use molrs::io::data::xyz::{read_xyz_frame, write_xyz_frame, write_xyz_traj};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::json::json_object_to_pydict;
use pyo3::types::PyDict;
use serde_json::Value as JsonValue;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

/// Read a PDB file and return a Frame.
///
/// The resulting frame contains an ``"atoms"`` block with columns ``element``
/// (str), ``x``/``y``/``z`` (float), ``id`` and ``res_id`` (uint), ``name``,
/// ``res_name``, ``chain``, ``icode`` and ``altloc`` (str; ``""`` for none),
/// ``occupancy`` and ``b_factor`` (float). If CRYST1 records are present a
/// ``Box`` is also attached.
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
/// >>> frame = molrs.io.read_pdb("molecule.pdb")
/// >>> atoms = frame["atoms"]
/// >>> symbols = atoms["symbol"]
#[pyfunction]
pub fn read_pdb(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = read_pdb_frame(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))?;
    PyFrame::from_core_frame(frame)
}

/// Read every MODEL of a PDB file as a trajectory (one Frame per MODEL).
///
/// A single-model (or MODEL-less) PDB returns a one-element list.
///
/// Parameters
/// ----------
/// path : str
///     Path to a (possibly multi-MODEL) ``.pdb`` file.
///
/// Returns
/// -------
/// list[Frame]
#[pyfunction]
pub fn read_pdb_trajectory(path: PathBuf) -> PyResult<Vec<PyFrame>> {
    let path = path_str(&path)?;
    let frames = read_pdb_traj(path).map_err(io_error_to_pyerr)?;
    frames.into_iter().map(PyFrame::from_core_frame).collect()
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
pub fn read_xyz(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = read_xyz_frame(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read a LAMMPS data file and return a Frame.
///
/// Every typed block (``atoms``, ``bonds``, ``angles``, ``dihedrals``,
/// ``impropers``) carries the numeric ``type_id`` and the string ``type``: the
/// file's type label, or the id spelled as a label when the file has no
/// ``* Type Labels`` section. The ``* Coeffs`` sections are kept verbatim in
/// ``frame.meta["lammps_coeffs_text"]`` (``molrs.io.read_lammps_data_coeffs(frame)``
/// reads them into a force field); the type-label inventories,
/// header counts, unit style and the box axes the header named are in
/// ``frame.meta`` too.
///
/// Parameters
/// ----------
/// path : str
///     Path to a LAMMPS data file on disk.
/// atom_style : str, optional
///     The ``Atoms`` column layout (``"full"``, ``"atomic"``, ``"charge"``,
///     …), as LAMMPS's ``atom_style`` governs ``read_data``. Without it the
///     section's ``# style`` comment decides, and without that the column
///     count.
///
/// Returns
/// -------
/// Frame
///     Parsed molecular data with atoms, bonds, and box metadata.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed, or ``atom_style`` is not an
///     atom style.
///
/// Examples
/// --------
/// >>> frame = molrs.io.read_lammps_data("system.data")
/// >>> atoms = frame["atoms"]
#[pyfunction]
#[pyo3(signature = (path, atom_style = None))]
pub fn read_lammps_data(path: PathBuf, atom_style: Option<&str>) -> PyResult<PyFrame> {
    use molrs::io::data::lammps_data::LAMMPSDataReader;
    use molrs::io::reader::FrameReader;
    let path = path_str(&path)?;
    let frame = match atom_style {
        None => read_lammps_data_rs(path).map_err(io_error_to_pyerr)?,
        Some(style) => {
            let file = File::open(path).map_err(io_error_to_pyerr)?;
            LAMMPSDataReader::new(std::io::BufReader::new(file))
                .with_atom_style(style)
                .map_err(io_error_to_pyerr)?
                .read()
                .map_err(io_error_to_pyerr)?
                .ok_or_else(|| {
                    pyo3::exceptions::PyIOError::new_err("no frame in LAMMPS data file")
                })?
        }
    };
    PyFrame::from_core_frame(frame)
}

/// Read the first frame of a GROMACS GRO file.
///
/// The ``"atoms"`` block carries ``res_id``, ``res_name``, ``name``,
/// ``element`` (inferred from the atom name), ``id`` and ``x``/``y``/``z`` in
/// Å (converted from the file's nm), plus ``vx``/``vy``/``vz`` in Å/ps when
/// the file has velocities. The box is ``frame.box``. Every frame of a
/// multi-frame file: :func:`read_gro_trajectory`.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed, or holds no frame.
#[pyfunction]
pub fn read_gro(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = read_gro_rs(path)
        .map_err(io_error_to_pyerr)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            pyo3::exceptions::PyIOError::new_err(format!("{path}: GRO file holds no frame"))
        })?;
    PyFrame::from_core_frame(frame)
}

/// Read every frame of a GROMACS GRO file, in file order.
///
/// Each frame as :func:`read_gro` describes it; a single-frame file returns a
/// one-element list. Inverse of :func:`write_gro_trajectory`.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
#[pyfunction]
pub fn read_gro_trajectory(path: PathBuf) -> PyResult<Vec<PyFrame>> {
    let path = path_str(&path)?;
    let frames = read_gro_rs(path).map_err(io_error_to_pyerr)?;
    frames.into_iter().map(PyFrame::from_core_frame).collect()
}

/// Write a Frame to a GROMACS GRO file.
///
/// Reads ``x``/``y``/``z`` (Å, written as nm) from the ``"atoms"`` block and,
/// when present, ``res_id``, ``res_name``, ``name`` (else ``element``),
/// ``id`` and ``vx``/``vy``/``vz``. The box is taken from ``frame.box``.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be written.
/// ValueError
///     If the frame is missing the ``"atoms"`` block or coordinate columns.
#[pyfunction]
pub fn write_gro(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| write_gro_rs(path, f).map_err(io_error_to_pyerr))?
}

/// Write Frames as one multi-frame GROMACS GRO trajectory, each as
/// :func:`write_gro` writes it. Inverse of :func:`read_gro_trajectory`.
#[pyfunction]
pub fn write_gro_trajectory(path: PathBuf, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    write_gro_traj_rs(path, &core_frames).map_err(io_error_to_pyerr)
}

/// Read a VASP CHGCAR or CHGDIF file.
///
/// Returns a Frame containing:
///
/// - ``"atoms"`` block with ``symbol``, ``x``, ``y``, ``z`` (Cartesian Å)
/// - ``box``: triclinic periodic box
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
/// >>> frame = molrs.io.read_chgcar("CHGCAR")
/// >>> grid = frame["chgcar"]
/// >>> total = grid["total"]          # shape (nx, ny, nz)
/// >>> density = total / frame.box.volume()
#[pyfunction]
pub fn read_chgcar(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = read_chgcar_rs(path).map_err(molrs_error_to_pyerr)?;
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
/// >>> frame = molrs.io.read_cube("density.cube")
/// >>> grid = frame["cube"]
/// >>> density = grid["density"]       # shape (nx, ny, nz)
#[pyfunction]
pub fn read_cube(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = read_cube_rs(path).map_err(molrs_error_to_pyerr)?;
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
pub fn write_cube(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| write_cube_rs(path, f).map_err(molrs_error_to_pyerr))?
}

/// Read a Tripos MOL2 file and return the first molecule as a Frame.
///
/// Canonical columns on atoms: ``id``, ``name``, ``x``/``y``/``z``, ``type``
/// (the SYBYL atom type), optional ``res_id``/``res_name`` (the MOL2
/// substructure) and ``charge``. Bonds carry ``atomi``/``atomj`` (0-based),
/// ``type`` (the SYBYL bond type token), and the chemical
/// ``bond_type``/``bond_number`` codes.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.mol2`` file.
///
/// Returns
/// -------
/// Frame
#[pyfunction]
pub fn read_mol2(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = read_mol2_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an AMBER ASCII inpcrd / restrt coordinate file.
///
/// Fixed-width Fortran ``6F12.7`` layout. Without ``frame``, returns a new
/// Frame with ``id``, ``name``, ``x``/``y``/``z``, optional ``vel`` (shape
/// ``[n, 3]``), optional box, and meta ``title`` / ``timestep``.
///
/// With ``frame`` (say, the structure :func:`read_amber_prmtop` read), the
/// file's coordinates go into that frame in place and it is returned: ``x`` /
/// ``y`` / ``z`` (and ``vel``) replace those atom columns, every other column
/// stays; the file's box and meta keys are set. A frame without an ``atoms``
/// block receives the file's whole block.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.inpcrd`` or restart file.
/// frame : Frame, optional
///     Frame to receive the coordinates.
///
/// Returns
/// -------
/// Frame
///     ``frame`` itself when given, else a new frame.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed, or ``frame``'s atom count
///     differs from the file's (``frame`` is then unchanged).
#[pyfunction]
#[pyo3(signature = (path, frame = None))]
pub fn read_amber_inpcrd<'py>(
    py: Python<'py>,
    path: PathBuf,
    frame: Option<Bound<'py, PyFrame>>,
) -> PyResult<Bound<'py, PyAny>> {
    let path = path_str(&path)?;
    match frame {
        None => {
            let frame = read_amber_inpcrd_rs(path).map_err(io_error_to_pyerr)?;
            Ok(Bound::new(py, PyFrame::from_core_frame(frame)?)?.into_any())
        }
        Some(target) => {
            target
                .borrow()
                .with_frame_mut(|f| molrs::io::data::inpcrd::read_amber_inpcrd_into(path, f))?
                .map_err(io_error_to_pyerr)?;
            Ok(target.into_any())
        }
    }
}

/// Read an AMBER prmtop **structure** file into a Frame.
///
/// Structure / connectivity only: atoms (name, type, charge in electron units,
/// mass, optional atomic_number/element, res_id), bonds/angles/dihedrals with
/// 0-based indices and type labels, plus POINTERS meta. Force-field parameter
/// tables are not assembled.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.prmtop`` / ``.parm7`` file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
#[pyfunction]
pub fn read_amber_prmtop(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = read_amber_prmtop_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an Antechamber ``.ac`` file into a Frame.
#[pyfunction]
pub fn read_ac(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = read_ac_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an Amber prep file into a nested dict (serde JSON shape).
#[pyfunction]
pub fn read_prep<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let path = path_str(&path)?;
    let res = read_prep_rs(path).map_err(io_error_to_pyerr)?;
    prep_residue_to_pydict(py, &res)
}

/// Write an Amber prep residue from a nested dict.
#[pyfunction]
pub fn write_prep(path: PathBuf, residue: &Bound<'_, PyAny>) -> PyResult<()> {
    let path = path_str(&path)?;
    let res = py_to_prep_residue(residue)?;
    write_prep_rs(path, &res).map_err(io_error_to_pyerr)
}

fn py_to_prep_residue(residue: &Bound<'_, PyAny>) -> PyResult<PrepResidue> {
    let name: String = residue.get_item("name")?.extract()?;
    let atoms_list = residue.get_item("atoms")?;
    let mut atoms = Vec::new();
    for item in atoms_list.try_iter()? {
        let d = item?;
        atoms.push(PrepAtom {
            index: d.get_item("index")?.extract()?,
            name: d.get_item("name")?.extract()?,
            atom_type: d.get_item("atom_type")?.extract()?,
            tree_type: d
                .get_item("tree_type")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(|| "M".into()),
            na: d
                .get_item("na")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            nb: d
                .get_item("nb")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            nc: d
                .get_item("nc")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            r: d.get_item("r")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            theta: d
                .get_item("theta")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            phi: d
                .get_item("phi")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            charge: d
                .get_item("charge")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            element: d
                .get_item("element")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or_default(),
        });
    }
    let mut impropers = Vec::new();
    if let Ok(imps) = residue.get_item("impropers") {
        for item in imps.try_iter()? {
            let row: Vec<String> = item?.extract()?;
            impropers.push(row);
        }
    }
    Ok(PrepResidue {
        name,
        atoms,
        head_atom: None,
        tail_atom: None,
        impropers,
    })
}

fn prep_residue_to_pydict<'py>(py: Python<'py>, res: &PrepResidue) -> PyResult<Bound<'py, PyDict>> {
    let value = serde_json::to_value(res)
        .map_err(|e| PyValueError::new_err(format!("failed to serialize prep residue: {e}")))?;
    match value {
        JsonValue::Object(map) => json_object_to_pydict(py, &map),
        _ => Err(PyValueError::new_err(
            "internal error: prep residue did not serialize to an object",
        )),
    }
}

/// Write a Frame to a Tripos MOL2 file.
///
/// Expects format-native atom columns (``atom_type``, optional
/// ``subst_id``/``subst_name``). Canonical renames are applied by the
/// :mod:`molrs.io` façade before calling this binding.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_mol2(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| write_mol2_rs(path, f).map_err(io_error_to_pyerr))?
}

/// Read a LAMMPS molecule template (native ``.mol`` or JSON).
#[pyfunction]
pub fn read_lammps_molecule(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = read_lammps_molecule_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an XSF (XCrySDen Structure File) and return a Frame.
///
/// Crystal structures (`CRYSTAL` + `PRIMVEC`/`CONVVEC`) yield a periodic box;
/// molecular structures (`MOLECULE`) yield a free box. Atoms carry
/// ``atomic_number``, ``element``, and ``x``/``y``/``z`` (Å).
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.xsf`` file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
#[pyfunction]
pub fn read_xsf(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = read_xsf_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a Frame to an XSF (XCrySDen Structure File).
///
/// A defined periodic box produces `CRYSTAL` + `PRIMVEC`/`CONVVEC`; otherwise
/// the structure is written as `MOLECULE`. Atoms need ``atomic_number`` and
/// ``x``/``y``/``z``.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be written.
#[pyfunction]
pub fn write_xsf(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| write_xsf_rs(path, f).map_err(io_error_to_pyerr))?
}

/// Write a Frame as a LAMMPS molecule template.
///
/// Parameters
/// ----------
/// path : str
///     Output path.
/// frame : Frame
///     Molecule frame.
/// format : str
///     ``"native"`` or ``"json"`` (default ``"native"``).
#[pyfunction]
#[pyo3(signature = (path, frame, format = "native"))]
pub fn write_lammps_molecule(path: PathBuf, frame: &PyFrame, format: &str) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| write_lammps_molecule_rs(path, f, format).map_err(io_error_to_pyerr))?
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
pub fn write_pdb(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| {
        let file = File::create(path).map_err(io_error_to_pyerr)?;
        let mut buf = BufWriter::new(file);
        write_pdb_frame(&mut buf, f).map_err(io_error_to_pyerr)
    })?
}

/// Write a list of Frames to a multi-MODEL PDB trajectory.
///
/// Each frame becomes one ``MODEL``/``ENDMDL`` block; a shared ``CRYST1`` is
/// written once from the first frame. Inverse of :func:`read_pdb_trajectory`.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
///     Frames to write, in order.
#[pyfunction]
pub fn write_pdb_trajectory(path: PathBuf, frames: Vec<PyFrame>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    let file = File::create(path).map_err(io_error_to_pyerr)?;
    let mut buf = BufWriter::new(file);
    write_pdb_traj(&mut buf, &core_frames).map_err(io_error_to_pyerr)
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
pub fn write_xyz(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| {
        let file = File::create(path).map_err(io_error_to_pyerr)?;
        let mut buf = BufWriter::new(file);
        write_xyz_frame(&mut buf, f).map_err(io_error_to_pyerr)
    })?
}

/// Write Frames as one multi-frame Extended XYZ trajectory, each as
/// :func:`write_xyz` writes it. Inverse of :func:`read_xyz_trajectory`.
#[pyfunction]
pub fn write_xyz_trajectory(path: PathBuf, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    let file = File::create(path).map_err(io_error_to_pyerr)?;
    let mut buf = BufWriter::new(file);
    write_xyz_traj(&mut buf, &core_frames).map_err(io_error_to_pyerr)?;
    std::io::Write::flush(&mut buf).map_err(io_error_to_pyerr)
}

/// Write a Frame to a LAMMPS data file.
///
/// The box is ``frame.box``. A frame without one is a non-periodic system and
/// is written inside the bounds of its coordinates widened by 1 length unit on
/// every side (``BOXLESS_MARGIN``): read it with ``boundary s s s``, which
/// shrink-wraps that box to the atoms. A frame meant to be periodic carries
/// its box.
///
/// Atoms, topology, ``Masses`` and the ``* Type Labels`` sections are
/// written; ``* Coeffs`` are not (:func:`molrs.io.write_lammps_data_coeffs`
/// returns them for the same labels). A system with Drude particles (a
/// ``drudes`` block, or atoms whose ``vsite`` is ``"drude"``) gets a header
/// comment with the ``fix drude`` C/D/N flags in atom-type order.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
/// type_labels : dict[str, list[str]], optional
///     Extra type labels per block (``{"atoms": [...], "bonds": [...]}``),
///     declared in the file even when no row uses them; ids stay dense and
///     follow the sorted labels. ``frame`` itself is not modified.
///
/// Raises
/// ------
/// ValueError
///     If a declared label is empty or contains ``,``, or a block is not
///     ``atoms`` / ``bonds`` / ``angles`` / ``dihedrals`` / ``impropers`` /
///     ``cmaps``.
/// IOError
///     If the frame cannot be written.
#[pyfunction]
#[pyo3(signature = (path, frame, *, type_labels = None))]
pub fn write_lammps_data(
    path: PathBuf,
    frame: &PyFrame,
    type_labels: Option<std::collections::BTreeMap<String, Vec<String>>>,
) -> PyResult<()> {
    let path = path_str(&path)?;
    match type_labels {
        None => frame.with_frame(|f| write_lammps_data_rs(path, f).map_err(io_error_to_pyerr))?,
        Some(extra) => {
            let mut work = frame.clone_core_frame()?;
            for (block, labels) in &extra {
                molrs::core::TypeLabels::declare(&mut work, block, labels)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
            }
            write_lammps_data_rs(path, &work).map_err(io_error_to_pyerr)
        }
    }
}

/// Register this module's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_pdb, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_pdb_trajectory, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_xyz, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_lammps_data, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_gro, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_gro_trajectory, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_gro, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_gro_trajectory, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_chgcar, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_cube, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_cube, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_mol2, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_amber_inpcrd, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_amber_prmtop, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_ac, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_prep, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_prep, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_mol2, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_lammps_molecule, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_xsf, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_xsf, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_lammps_molecule, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_pdb, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_pdb_trajectory, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_xyz, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_xyz_trajectory, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_lammps_data, m)?)?;
    Ok(())
}
