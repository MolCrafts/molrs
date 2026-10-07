//! Force-field file writers (`molrs::io::writers`): a
//! [`PyForceField`] (and, for a whole-system format, a typed frame) out to
//! each engine's force-field text.

use std::path::PathBuf;

use pyo3::prelude::*;

use crate::core::frame::PyFrame;
use crate::error::io_error_to_pyerr;
use crate::ff::forcefield::PyForceField;
use crate::path::path_str;
use molrs::io::gromacs::GromacsTopForcefieldWriter;

/// Write a ForceField as GROMACS force-field directives.
///
/// Writes ``[ defaults ]``, ``[ atomtypes ]``, ``[ nonbond_params ]``,
/// ``[ pairtypes ]``, ``[ bondtypes ]``, ``[ angletypes ]``,
/// ``[ dihedraltypes ]`` and ``[ cmaptypes ]`` in GROMACS units (nm, kJ/mol,
/// degrees) — the inverse of :func:`read_gromacs_top_forcefield`. No molecule section
/// is written: a force field holds no molecule. A style or parameter GROMACS
/// directives cannot express raises ``ValueError`` naming it. ``precision`` is
/// the number of decimal places for floating coefficients.
#[pyfunction]
#[pyo3(name = "write_gromacs_top_forcefield", signature = (path, forcefield, precision = 6))]
pub fn write_gromacs_top_forcefield_py(
    path: PathBuf,
    forcefield: &PyForceField,
    precision: usize,
) -> PyResult<()> {
    use molrs::io::writer::ForceFieldWriter;
    molrs::io::gromacs::GromacsTopForcefieldWriter::new()
        .with_precision(precision)
        .write(&forcefield.inner, path_str(&path)?)
        .map_err(crate::ff::ir::write_err)
}

/// Write a ForceField as an AMBER frcmod file.
///
/// Writes ``MASS``, ``BOND``, ``ANGLE``, ``DIHE``, ``IMPROPER`` and ``NONBON``
/// in AMBER's conventions (``RK = k``, ``TK = k``, degrees — molrs's own —
/// and ``R*/2`` from sigma), so tleap can ``loadamberparams`` it. A style or parameter a frcmod
/// cannot express raises ``ValueError`` naming it.
#[pyfunction]
#[pyo3(name = "write_amber_frcmod", signature = (path, forcefield))]
pub fn write_amber_frcmod_py(path: PathBuf, forcefield: &PyForceField) -> PyResult<()> {
    molrs::io::write_amber_frcmod(path_str(&path)?, &forcefield.inner)
        .map_err(crate::ff::ir::write_err)
}

/// Write a ForceField to OpenMM force-field XML.
///
/// Every style is written in OpenMM's own schema and units (nm, kJ/mol,
/// radians, ½k harmonic terms), or refused naming the style when OpenMM's
/// tags cannot hold it (see the force-field IR guide, "OpenMM").
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Output file.
/// forcefield : ForceField
/// precision : int, optional
///     Decimals per number; ``None`` (default) writes each number in the
///     shortest form that reads back to the same float.
///
/// Raises
/// ------
/// ValueError
///     A style, parameter or 1-4 setting with no OpenMM form, named.
#[pyfunction]
#[pyo3(name = "write_openmm_xml_forcefield", signature = (path, forcefield, precision = None))]
pub fn write_openmm_xml_forcefield_py(
    path: PathBuf,
    forcefield: &PyForceField,
    precision: Option<usize>,
) -> PyResult<()> {
    molrs::io::write_openmm_xml_forcefield(path_str(&path)?, &forcefield.inner, precision)
        .map_err(crate::ff::ir::write_err)
}

/// Write a :class:`ForceField` to a LAMMPS force-field include (``*.ff``).
///
/// Coefficient writing, keyed by the system's type labels: ``frame``'s
/// ``atoms`` / ``bonds`` / ``angles`` / ``dihedrals`` / ``impropers`` type
/// labels are walked in id order and each is looked up in ``forcefield``
/// (every label matched to a type name exactly).
/// Force-field types no label uses are not written.
///
/// Inverse of :func:`read_lammps_forcefield`, and the identity on coefficients:
/// the force-field IR follows the LAMMPS standard. A force field declared in another LAMMPS
/// unit style than ``units`` has its energies and lengths converted through
/// the lj reduced hub — never hard-coded eV/kcal factors. A split ``lj/cut`` +
/// ``coul/cut`` pair is recombined as ``lj/cut/coul/cut`` so geometric mixing
/// is not defeated by a hybrid wildcard. Every style is written through its
/// LAMMPS form in the IR registry (``molrs.ff.ir``, ``StyleSpec.lammps``):
/// the built-ins LAMMPS has (``dihedral periodic`` as ``fourier``,
/// ``improper periodic`` as ``cvff``, the ``class2`` styles with their
/// cross-term lines at zero, …) and a style registered with
/// ``lammps="positional"``, each parameter converted by its dimension.
///
/// Parameters
/// ----------
/// path : str
///     Destination path for the include.
/// forcefield : ForceField
///     Force field, in the force-field IR (adopts the LAMMPS standard).
/// frame : Frame
///     The system whose type labels select the coefficients.
/// precision : int, optional
///     Decimal places for floating coefficients (default 6).
/// skip_pair_style : bool, optional
///     When true, omit the ``pair_style`` line: the input script sets its own
///     before the include. Only that line is skipped: ``special_bonds`` and
///     ``pair_modify mix`` / ``shift`` are the force field's and stay
///     (LAMMPS's defaults, ``0 0 0`` and ``geometric`` for ``lj/cut``, are not
///     molrs's); ``pair_modify`` needs a pair style, so read such an include
///     after the input's ``pair_style``.
/// skip_special_bonds : bool, optional
///     When true, omit ``special_bonds``: the input script states its own 1-4
///     weights, which an include read after them would override.
/// skip_units : bool, optional
///     When true, omit the ``units`` line so the include can follow ``units``
///     already set in the input script.
/// units : str, optional
///     LAMMPS ``units`` style for the written file: ``"real"`` (default),
///     ``"metal"``, or ``"lj"``.
/// cmap_file : str, optional
///     The ``fix cmap`` file the include names on its ``fix cmap all cmap
///     <file>`` line (where :func:`write_lammps_cmap` saved it). Required
///     exactly when ``frame`` has a ``cmaps`` block; that fix must reach
///     LAMMPS before ``read_data <data> fix cmap crossterm CMAP``.
///
/// Raises
/// ------
/// ValueError
///     On a frame type label the force field does not define (the message
///     names the block and the label), a style without a LAMMPS form holding
///     a used type ("LAMMPS has no form for …"), a bad units keyword, or
///     missing required parameters.
#[pyfunction]
#[pyo3(
    name = "write_lammps_forcefield",
    signature = (
        path,
        forcefield,
        frame,
        *,
        precision = 6,
        skip_pair_style = false,
        skip_special_bonds = false,
        skip_units = false,
        units = "real",
        cmap_file = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn write_lammps_forcefield_py(
    path: PathBuf,
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    skip_pair_style: bool,
    skip_special_bonds: bool,
    skip_units: bool,
    units: &str,
    cmap_file: Option<String>,
) -> PyResult<()> {
    use molrs::core::TypeLabels;
    use molrs::io::lammps::parse_lammps_units_style;
    use molrs::io::{
        lammps::LammpsForcefieldWriteOptions, lammps::LammpsForcefieldWriter,
        writer::ForceFieldWriter,
    };
    let units = parse_lammps_units_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    frame
        .with_frame(molrs::io::lammps::refuse_pair_overrides)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let labels = frame
        .with_frame(TypeLabels::from_frame)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let writer = LammpsForcefieldWriter::with_options(
        &labels,
        LammpsForcefieldWriteOptions {
            precision,
            skip_pair_style,
            skip_special_bonds,
            skip_units,
            units,
            cmap_file,
        },
    );
    writer
        .write(&forcefield.inner, path_str(&path)?)
        .map_err(crate::ff::ir::write_err)
}

/// Serialize a :class:`ForceField` to a LAMMPS force-field include string
/// (same labels, format and unit conversion as :func:`write_lammps_forcefield`).
#[pyfunction]
#[pyo3(
    name = "write_lammps_forcefield_str",
    signature = (
        forcefield,
        frame,
        *,
        precision = 6,
        skip_pair_style = false,
        skip_special_bonds = false,
        skip_units = false,
        units = "real",
        cmap_file = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn write_lammps_forcefield_str_py(
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    skip_pair_style: bool,
    skip_special_bonds: bool,
    skip_units: bool,
    units: &str,
    cmap_file: Option<String>,
) -> PyResult<String> {
    use molrs::core::TypeLabels;
    use molrs::io::lammps::parse_lammps_units_style;
    use molrs::io::{
        lammps::LammpsForcefieldWriteOptions, lammps::LammpsForcefieldWriter,
        writer::ForceFieldWriter,
    };
    let units = parse_lammps_units_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    frame
        .with_frame(molrs::io::lammps::refuse_pair_overrides)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let labels = frame
        .with_frame(TypeLabels::from_frame)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let writer = LammpsForcefieldWriter::with_options(
        &labels,
        LammpsForcefieldWriteOptions {
            precision,
            skip_pair_style,
            skip_special_bonds,
            skip_units,
            units,
            cmap_file,
        },
    );
    writer
        .write_str(&forcefield.inner)
        .map_err(crate::ff::ir::write_err)
}

/// Serialize a :class:`ForceField` to LAMMPS data-file ``* Coeffs`` sections.
///
/// Same labels, form map and ``units`` conversion as
/// :func:`write_lammps_forcefield`, but emits ``Pair Coeffs`` / ``Bond Coeffs``
/// / … blocks whose integer ids are ``frame``'s type-label ids, not
/// input-script ``*_coeff`` lines. ``Pair Coeffs`` holds self pairs only; a
/// used explicit cross pair raises ``ValueError``.
#[pyfunction]
#[pyo3(
    name = "write_lammps_data_coeffs",
    signature = (
        forcefield,
        frame,
        *,
        precision = 6,
        units = "real",
    )
)]

pub fn write_lammps_data_coeffs_py(
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    units: &str,
) -> PyResult<String> {
    use molrs::core::TypeLabels;
    use molrs::io::lammps::parse_lammps_units_style;
    use molrs::io::{lammps::LammpsForcefieldWriteOptions, lammps::LammpsForcefieldWriter};
    let units = parse_lammps_units_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    frame
        .with_frame(molrs::io::lammps::refuse_pair_overrides)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let labels = frame
        .with_frame(TypeLabels::from_frame)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let writer = LammpsForcefieldWriter::with_options(
        &labels,
        LammpsForcefieldWriteOptions {
            precision,
            units,
            ..LammpsForcefieldWriteOptions::default()
        },
    );
    writer
        .write_data_coeffs_str(&forcefield.inner)
        .map_err(crate::ff::ir::write_err)
}

/// Write the LAMMPS ``fix cmap`` file of ``frame``'s CMAP crossterms.
///
/// The ``cmaps`` block's type labels are walked in id order — the crossterm
/// types :func:`molrs.io.write_lammps_data` gives the ``CMAP`` section — and
/// each label's ``cmap charmm`` grid is written in CHARMM's layout (a
/// ``# UNITS:`` line, ``# <φ>`` rows of five ``precision``-decimal values),
/// energies converted to ``units``. A CHARMM file read with
/// :func:`read_lammps_cmap` is written back line for line.
///
/// Raises
/// ------
/// ValueError
///     No ``cmaps`` label, a label without a row, a grid that is not 24×24,
///     more than six maps, or a style other than ``charmm``.
#[pyfunction]
#[pyo3(
    name = "write_lammps_cmap",
    signature = (path, forcefield, frame, *, precision = 6, units = "real")
)]

pub fn write_lammps_cmap_py(
    path: PathBuf,
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    units: &str,
) -> PyResult<()> {
    use molrs::core::TypeLabels;
    use molrs::io::lammps::parse_lammps_units_style;
    use molrs::io::{lammps::LammpsForcefieldWriteOptions, lammps::LammpsForcefieldWriter};
    let units = parse_lammps_units_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let labels = frame
        .with_frame(TypeLabels::from_frame)?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let options = LammpsForcefieldWriteOptions {
        precision,
        units,
        ..LammpsForcefieldWriteOptions::default()
    };
    let text = LammpsForcefieldWriter::with_options(&labels, options)
        .write_cmap_str(&forcefield.inner)
        .map_err(crate::ff::ir::write_err)?;
    std::fs::write(&path, text)
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {e}", path.display())))
}

/// Write a force field and a typed frame as one GROMACS topology — the
/// inverse of :func:`read_gromacs_system`.
///
/// The force-field directives as :func:`write_gromacs_top_forcefield` writes them,
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
            GromacsTopForcefieldWriter::new()
                .with_precision(precision)
                .write_system_str(&forcefield.inner, f)
        })?
        .map_err(crate::ff::ir::write_err)?;
    std::fs::write(path_str(&path)?, text).map_err(io_error_to_pyerr)
}

/// Register the force-field writers.
/// Write a force field as a molrs force-field XML file — the inverse of
/// :func:`read_molrs_xml_forcefield`: one element per style, every number in
/// a form that reads back to the same float.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     File to write.
/// forcefield : ForceField
///     The force field.
///
/// Raises
/// ------
/// ValueError
///     What the layout cannot hold, by name: a style of a category other
///     than bond, angle, dihedral, improper and pair; a bonded style's own
///     parameters; a string or array parameter; a declared unit system or
///     special-bonds weights.
#[pyfunction]
#[pyo3(name = "write_molrs_xml_forcefield", signature = (path, forcefield))]
pub fn write_molrs_xml_forcefield_py(path: PathBuf, forcefield: &PyForceField) -> PyResult<()> {
    molrs::io::write_molrs_xml_forcefield(path_str(&path)?, &forcefield.inner)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_gromacs_top_forcefield_py, m)?,
    )?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_gromacs_system, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_amber_frcmod_py, m)?)?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_openmm_xml_forcefield_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_molrs_xml_forcefield_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_lammps_forcefield_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_lammps_forcefield_str_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_lammps_data_coeffs_py, m)?,
    )?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_lammps_cmap_py, m)?)?;
    Ok(())
}
