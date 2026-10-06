//! Force-field file readers (`molrs::ff::forcefield::readers`): each file
//! format's force-field directives into a [`PyForceField`]. Structure
//! formats are `molrs.io`'s; a reader here maps parameters, not geometry.

use std::path::PathBuf;

use pyo3::prelude::*;

use super::PyForceField;
use crate::core::store::frame::PyFrame;
use crate::path::path_str;

/// Read a force-field definition from an XML file.
#[pyfunction]
#[pyo3(name = "read_forcefield_xml")]
pub fn read_forcefield_xml_py(path: PathBuf) -> PyResult<PyForceField> {
    let forcefield = molrs::ff::forcefield::xml::read_forcefield_xml(path_str(&path)?)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyForceField { inner: forcefield })
}

/// Read an OpenMM force-field XML file into a :class:`ForceField`.
///
/// Reads OpenMM's own ``<ForceField>`` schema — CHARMM36, AMBER and OPLS-AA
/// ports alike (nm, kJ/mol, radians, ``½k`` harmonic terms) — into the
/// force-field IR, whose definitions follow LAMMPS (Å, kcal/mol, degrees,
/// un-halved ``K``): harmonic bonds and angles, Urey-Bradley
/// (``angle charmm``), periodic and Ryckaert-Bellemans torsions
/// (``dihedral periodic`` / ``multi/harmonic``), periodic impropers (stored in
/// the order OpenMM prices them) and CHARMM's harmonic ``CustomTorsionForce``
/// impropers, CMAP (``cmap charmm``), ``NonbondedForce`` (``lj/cut`` +
/// ``coul/cut``) and ``LennardJonesForce`` with NBFIX and 1-4 parameters
/// (``lj/charmm`` + ``coul/charmm``). Coulomb styles state OpenMM's own
/// constant. A field whose ``lj/charmm`` declares ``one_four="epsilon14"``
/// needs :meth:`ForceField.materialize_one_four` on its frames before it
/// compiles. Distinct from :func:`read_forcefield_xml`, which also reads
/// molrs's own schema.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Path to an OpenMM force-field XML (``charmm36.xml``, ``oplsaa.xml``, …).
///
/// Returns
/// -------
/// ForceField
///
/// Raises
/// ------
/// ValueError
///     On a malformed document, a section or row with no IR form (named:
///     ``Custom*Force`` other than the harmonic improper, ``<Script>``,
///     ``ordering="smirnoff"``, …), or a missing/non-numeric required
///     attribute (reading is total — never a silent skip).
#[pyfunction]
#[pyo3(name = "read_opls_xml")]
pub fn read_opls_xml_py(path: PathBuf) -> PyResult<PyForceField> {
    use molrs::ff::forcefield::readers::ForceFieldReader;
    let forcefield = molrs::ff::forcefield::readers::opls::OplsXmlReader::new()
        .read(path_str(&path)?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a LAMMPS force-field include (``*.ff``) into a :class:`ForceField`.
///
/// Parses the ``pair_style``/``pair_coeff`` + ``bond_style``/``angle_style``/
/// ``dihedral_style``/``improper_style`` include that
/// :func:`write_lammps_forcefield` emits. the force-field IR follows the LAMMPS standard, so
/// every coefficient is stored as written (``K``, degrees) and the force field
/// declares the file's ``units``; the ``fourier`` dihedral is molrs's
/// ``periodic``. The ``special_bonds`` line is recorded on the force field.
/// Distinct from :func:`read_forcefield_xml` (molrs's own schema) and
/// :func:`read_opls_xml` (OPLS-AA / GROMACS XML).
///
/// Per-atom charge and mass live in the LAMMPS *data* file, not this include, so
/// they are not read here: Coulomb charges are drawn from the frame at evaluation
/// time and masses are irrelevant to geometry relaxation.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Path to a LAMMPS force-field include (``*.ff``).
///
/// Returns
/// -------
/// ForceField
///
/// Raises
/// ------
/// ValueError
///     On an unsupported style, a coefficient before its style declaration, a
///     wrong-arity type label, or a non-numeric parameter (reading is total —
///     never a silent skip).
#[pyfunction]
#[pyo3(name = "read_lammps_forcefield")]
pub fn read_lammps_forcefield_py(path: PathBuf) -> PyResult<PyForceField> {
    use molrs::ff::forcefield::readers::ForceFieldReader;
    let forcefield = molrs::ff::forcefield::readers::lammps::LammpsFfReader::new()
        .read(path_str(&path)?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read AMBER prmtop force-field parameter tables into a :class:`ForceField`.
///
/// Structure/connectivity is :func:`molrs.io.read_amber_prmtop`; this parses
/// harmonic bond/angle tables (``k = K``, AMBER's and LAMMPS's form; θ₀ to
/// degrees), periodic dihedrals (phases to degrees), impropers in AMBER's atom
/// order, and LJ A/B → σ/ε, in LAMMPS ``real`` units.
#[pyfunction]
#[pyo3(name = "read_amber_prmtop_ff")]
pub fn read_amber_prmtop_ff_py(path: PathBuf) -> PyResult<PyForceField> {
    let forcefield = molrs::ff::forcefield::readers::prmtop::read_amber_prmtop_ff(path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read the force-field directives of a GROMACS topology into a
/// :class:`ForceField`.
///
/// Reads ``[ defaults ]`` (nbfunc 1; comb-rule 2 or 3 → ``mixing``
/// ``arithmetic`` / ``geometric``; gen-pairs, fudgeLJ, fudgeQQ →
/// ``special_bonds``), ``[ atomtypes ]``, ``[ nonbond_params ]`` (explicit
/// cross rows), ``[ pairtypes ]`` (``lj/charmm`` ``epsilon14`` / ``sigma14``,
/// declared ``one_four = "epsilon14"``), ``[ bondtypes ]`` (funct 1, 3),
/// ``[ angletypes ]`` (funct 1, 5 → ``angle charmm``), ``[ dihedraltypes ]``
/// (funct 1, 9 → ``dihedral periodic``; 3 → ``multi/harmonic`` /
/// ``nharmonic``; 5 → ``opls``; 2, 4 → impropers) and ``[ cmaptypes ]``
/// (``cmap charmm``), converting GROMACS units (nm, kJ/mol, ``½k`` harmonic
/// terms) to the force-field IR (LAMMPS standard, ``real``: Å, kcal/mol,
/// ``K = k/2``; degrees stay degrees). See the Force-field IR guide,
/// "GROMACS topologies".
///
/// What the IR cannot hold raises ``ValueError`` naming it: an unsupported
/// function code or comb-rule, ``[ constrainttypes ]``,
/// ``[ implicit_genborn_params ]``, any unknown section, and every molecule
/// section — read a whole topology with :func:`read_gromacs_system`, or skip
/// them here.
///
/// ``include`` follows ``#include`` relative to the including file and then
/// each of ``include_dirs`` (default false: ignored). Each name in
/// ``skip_directives`` (bracket-less, case-insensitive, e.g.
/// ``"constrainttypes"``) is read past, rows and all, instead of refused.
#[pyfunction]
#[pyo3(
    name = "read_gromacs_top_ff",
    signature = (path, include = false, *, include_dirs = Vec::new(), skip_directives = Vec::new())
)]

pub fn read_gromacs_top_ff_py(
    path: PathBuf,
    include: bool,
    include_dirs: Vec<PathBuf>,
    skip_directives: Vec<String>,
) -> PyResult<PyForceField> {
    use molrs::ff::forcefield::readers::ForceFieldReader;
    let forcefield = gromacs_top_ff_reader(include, &include_dirs, &skip_directives)
        .read(path_str(&path)?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a whole GROMACS topology into a :class:`ForceField` and a typed
/// :class:`Frame`.
///
/// The directives as :func:`read_gromacs_top_ff` reads them, and the molecule
/// sections: ``atoms`` (``type``, ``charge``, ``mass``, ``name``, ``res_id``,
/// ``res_name``, ``mol_id``), ``bonds``, ``angles``, ``dihedrals``,
/// ``impropers``, ``cmaps`` (each row's ``type`` the force-field type
/// GROMACS's own lookup picks; a row with parameters of its own gets a type
/// of its own), ``constraints``, ``exclusions`` and ``pairs`` (every
/// intramolecular pair GROMACS prices; the ``[ pairs ]`` rows flagged
/// ``is_14``, with per-pair override columns where they carry parameters).
/// Atom indices are 0-based. ``#include`` is followed, relative to the
/// including file and then each of ``include_dirs`` (GROMACS's
/// ``share/gromacs/top`` for ``#include "charmm27.ff/forcefield.itp"``).
/// Coordinates come from the ``.gro`` (:func:`molrs.io.read_gro`).
///
/// Returns ``(forcefield, frame)``. Anything the IR cannot hold (virtual
/// sites, restraints, free-energy B states, …) raises ``ValueError`` naming
/// it and where.
#[pyfunction]
#[pyo3(
    name = "read_gromacs_system",
    signature = (path, *, include_dirs = Vec::new(), skip_directives = Vec::new())
)]

pub fn read_gromacs_system_py(
    path: PathBuf,
    include_dirs: Vec<PathBuf>,
    skip_directives: Vec<String>,
) -> PyResult<(PyForceField, PyFrame)> {
    let (forcefield, frame) = gromacs_top_ff_reader(true, &include_dirs, &skip_directives)
        .read_system(path_str(&path)?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((
        PyForceField { inner: forcefield },
        PyFrame::from_core_frame(frame)?,
    ))
}

/// The GROMACS reader the Python entry points configure.
fn gromacs_top_ff_reader(
    include: bool,
    include_dirs: &[PathBuf],
    skip_directives: &[String],
) -> molrs::ff::forcefield::readers::gromacs::GromacsTopFfReader {
    let reader = include_dirs.iter().fold(
        molrs::ff::forcefield::readers::gromacs::GromacsTopFfReader::new().with_include(include),
        |reader, dir| reader.with_include_dir(dir),
    );
    skip_directives
        .iter()
        .fold(reader, |reader, name| reader.with_skipped_directive(name))
}

/// Parse LAMMPS data-file ``* Coeffs`` sections into a :class:`ForceField`.
///
/// ``coeffs_text`` may contain ``Pair Coeffs`` / ``Bond Coeffs`` / … blocks
/// (and an optional ``units`` line). Default styles are harmonic / ``lj/cut``
/// when the data file has no style directives. Optional ``*_labels`` maps are
/// 1-based type id → label string (from Type Labels sections).
#[pyfunction]
#[pyo3(
    name = "read_lammps_data_coeffs",
    signature = (
        coeffs_text,
        units = "real",
        atom_labels = None,
        bond_labels = None,
        angle_labels = None,
        dihedral_labels = None,
        improper_labels = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn read_lammps_data_coeffs_py(
    coeffs_text: &str,
    units: &str,
    atom_labels: Option<std::collections::HashMap<u32, String>>,
    bond_labels: Option<std::collections::HashMap<u32, String>>,
    angle_labels: Option<std::collections::HashMap<u32, String>>,
    dihedral_labels: Option<std::collections::HashMap<u32, String>>,
    improper_labels: Option<std::collections::HashMap<u32, String>>,
) -> PyResult<PyForceField> {
    use molrs::ff::forcefield::lammps_units::parse_style;
    use molrs::ff::forcefield::readers::lammps::LammpsFfReader;
    use molrs::ff::forcefield::readers::lammps::LammpsTypeLabelMaps;
    use std::collections::BTreeMap;

    let units = parse_style(units).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let to_btree = |m: Option<std::collections::HashMap<u32, String>>| -> BTreeMap<u32, String> {
        m.unwrap_or_default().into_iter().collect()
    };
    let labels = LammpsTypeLabelMaps {
        atom: to_btree(atom_labels),
        bond: to_btree(bond_labels),
        angle: to_btree(angle_labels),
        dihedral: to_btree(dihedral_labels),
        improper: to_btree(improper_labels),
    };
    let forcefield = LammpsFfReader::new()
        .read_data_coeffs(coeffs_text, &labels, units)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a LAMMPS ``fix cmap`` file (CHARMM format) into a :class:`ForceField`
/// of one ``cmap charmm`` style.
///
/// Map ``t`` of the file (1-based: the crossterm type a data file's ``CMAP``
/// section gives) is the row named ``"t"``, with the synthetic endpoints
/// ``t-t-t-t-t``, its ``grid`` the 24×24 map, φ-major, as written. The force
/// field is in the file's ``UNITS:`` tag, else ``real``. An incomplete map, a
/// seventh map, or a line running past a map's end raises ``ValueError``
/// (LAMMPS would drop the values).
#[pyfunction]
#[pyo3(name = "read_lammps_cmap")]
pub fn read_lammps_cmap_py(path: PathBuf) -> PyResult<PyForceField> {
    let text = std::fs::read_to_string(&path)
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {e}", path.display())))?;
    let forcefield = molrs::ff::forcefield::readers::lammps::LammpsFfReader::new()
        .read_cmap_str(&text)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Register the force-field readers.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(
        m,
        "molrs.ff.forcefield",
        wrap_pyfunction!(read_forcefield_xml_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.ff.forcefield",
        wrap_pyfunction!(read_opls_xml_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.ff.forcefield",
        wrap_pyfunction!(read_lammps_forcefield_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.ff.forcefield",
        wrap_pyfunction!(read_amber_prmtop_ff_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.ff.forcefield",
        wrap_pyfunction!(read_gromacs_top_ff_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.ff.forcefield",
        wrap_pyfunction!(read_gromacs_system_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.ff.forcefield",
        wrap_pyfunction!(read_lammps_data_coeffs_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.ff.forcefield",
        wrap_pyfunction!(read_lammps_cmap_py, m)?,
    )?;
    Ok(())
}
