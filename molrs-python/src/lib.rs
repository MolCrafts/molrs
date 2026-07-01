//! Python bindings for the molrs molecular simulation library.
//!
//! This crate provides PyO3-based Python bindings (`import molrs`) exposing
//! the core data model, I/O, neighbor search, force-field evaluation,
//! 3D coordinate generation, molecular packing, and analysis routines.
//!
//! # Module Layout
//!
//! | Python class         | Rust wrapper      | Purpose                                    |
//! |----------------------|-------------------|--------------------------------------------|
//! | `Block`              | [`PyBlock`]       | Heterogeneous column store (numpy arrays)  |
//! | `Frame`              | [`PyFrame`]       | Collection of named `Block`s + `SimBox`    |
//! | `Box`                | [`PyBox`]         | Simulation box / periodic boundaries       |
//! | `LinkedCell`         | [`PyLinkedCell`]  | Link-cell neighbor list (legacy API)       |
//! | `NeighborQuery`      | [`PyNeighborQuery`]| Spatial neighbor query (freud-style API)   |
//! | `NeighborList`       | [`PyNeighborList`]| Query result with pair indices + distances |
//! | `Atomistic`          | [`PyAtomistic`]   | All-atom molecular graph                   |
//! | `MMFFTypifier`       | [`PyMMFFTypifier`]| MMFF94 atom-type assignment                |
//! | `OplsTypifier`       | [`PyOplsTypifier`]| OPLS-AA atom-type + bonded assignment      |
//! | `Potentials`         | [`PyPotentials`]  | Compiled energy/force evaluator            |
//! | `RDF` / `MSD` / `Cluster` |              | Structural analysis                        |
//!
//! # Float Precision
//!
//! By default all floating-point arrays use `f32` (numpy `float32`).
//! Enable the `f64` feature for double precision (`float64`).

use pyo3::prelude::*;

mod error;
mod helpers;
mod store;

mod simbox;
use simbox::PyBox;

mod linkedcell;
use linkedcell::{PyLinkedCell, PyNeighborList, PyNeighborQuery};

mod block;
use block::PyBlock;

mod frame;
use frame::PyFrame;

mod io;

mod molrec;
use molrec::{PyMolRec, PyObservables, PyScalarObservable, PyTrajectory, PyVectorObservable};

mod region;
use region::{PyCuboid, PyHollowSphere, PyRegion, PySphere};

pub(crate) mod molgraph;
use molgraph::{PyAtomistic, PyCoarseGrain, PyGraph};
use molgraph::{
    add_hydrogens, compute_gasteiger_charges, find_rings, perceive_aromaticity, rotate, scale,
    translate,
};

mod conformer;
use conformer::{PyConformer, PyConformerReport, PyConformerStageReport};

mod forcefield;
use forcefield::{
    PyForceField, PyLBFGS, PyMMFFTypifier, PyOplsTypifier, PyOptReport, PyPotentials,
};

mod compute;
use compute::{
    PyCenterOfMass, PyCenterOfMassResult, PyCluster, PyClusterCenters, PyClusterCentersResult,
    PyClusterResult, PyDescriptorRow, PyGyrationTensor, PyInertiaTensor, PyKMeans, PyKMeansResult,
    PyMSD, PyMSDResult, PyMSDTimeSeries, PyPca2, PyPcaResult, PyRDF, PyRDFResult,
    PyRadiusOfGyration,
};

mod compute_extra;
mod compute_fit;
mod compute_travis;
mod dielectric;
mod signal;
mod transport;
mod validate;

/// Register the `keys` submodule mirroring `molrs_core::store::keys` so Python code
/// references the field-name convention by name (`molrs.keys.X`) instead of
/// scattering string literals.
fn register_keys(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    use ::molrs::store::keys;
    let m = PyModule::new(parent.py(), "keys")?;
    m.add("X", keys::X)?;
    m.add("Y", keys::Y)?;
    m.add("Z", keys::Z)?;
    m.add("COORDS", keys::COORDS.to_vec())?;
    m.add("ELEMENT", keys::ELEMENT)?;
    m.add("BEAD_TYPE", keys::BEAD_TYPE)?;
    m.add("CHARGE", keys::CHARGE)?;
    m.add("ORDER", keys::ORDER)?;
    m.add("MASS", keys::MASS)?;
    m.add("TYPE", keys::TYPE)?;
    m.add("ID", keys::ID)?;
    m.add("MOL_ID", keys::MOL_ID)?;
    m.add("SYMBOL", keys::SYMBOL)?;
    m.add("NAME", keys::NAME)?;
    m.add("VX", keys::VX)?;
    m.add("VY", keys::VY)?;
    m.add("VZ", keys::VZ)?;
    m.add("VELOCITIES", keys::VELOCITIES.to_vec())?;
    m.add("XYZ", keys::XYZ)?;
    m.add("RES_ID", keys::RES_ID)?;
    m.add("RES_NAME", keys::RES_NAME)?;
    m.add("ATOMI", keys::ATOMI)?;
    m.add("ATOMJ", keys::ATOMJ)?;
    m.add("ATOMK", keys::ATOMK)?;
    m.add("ATOML", keys::ATOML)?;
    m.add("ENDPOINTS", keys::ENDPOINTS.to_vec())?;
    parent.add_submodule(&m)?;
    Ok(())
}

/// Root Python module for the molrs library.
///
/// Registered classes and free functions are listed in the module-level
/// documentation above.
#[pymodule]
fn molrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // SimBox + neighbors
    m.add_class::<PyBox>()?;
    m.add_class::<PyLinkedCell>()?;
    m.add_class::<PyNeighborQuery>()?;
    m.add_class::<PyNeighborList>()?;

    // Public exceptions
    m.add(
        "BlockDtypeError",
        m.py().get_type::<error::BlockDtypeError>(),
    )?;

    // Block + Frame
    m.add_class::<PyBlock>()?;
    m.add_class::<PyFrame>()?;

    // I/O + SMILES
    // Readers
    m.add_function(wrap_pyfunction!(io::read_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_pdb_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_xyz, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_xyz_trajectory, m)?)?;
    m.add_class::<io::PyXYZTrajReader>()?;
    m.add_function(wrap_pyfunction!(io::read_lammps, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_lammps_traj, m)?)?;
    m.add_class::<io::PyLAMMPSTrajReader>()?;
    m.add_function(wrap_pyfunction!(io::read_dcd, m)?)?;
    m.add_class::<io::PyDcdTrajReader>()?;
    m.add_function(wrap_pyfunction!(io::read_trr, m)?)?;
    m.add_class::<io::PyTrrTrajReader>()?;
    m.add_function(wrap_pyfunction!(io::read_xtc, m)?)?;
    m.add_class::<io::PyXtcTrajReader>()?;
    m.add_function(wrap_pyfunction!(io::read_gro, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_chgcar_file, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_cube_file, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_cube_file, m)?)?;
    // Writers
    m.add_function(wrap_pyfunction!(io::write_gro, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_pdb_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_xyz, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_lammps, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_lammps_traj, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_dcd, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_trr, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_xtc, m)?)?;
    // SMILES
    m.add_function(wrap_pyfunction!(io::parse_smiles, m)?)?;
    m.add_class::<io::PySmilesIR>()?;

    // MolRec
    m.add_class::<PyTrajectory>()?;
    m.add_class::<PyMolRec>()?;
    m.add_class::<PyObservables>()?;
    m.add_class::<PyScalarObservable>()?;
    m.add_class::<PyVectorObservable>()?;

    // Regions
    m.add_class::<PySphere>()?;
    m.add_class::<PyHollowSphere>()?;
    m.add_class::<PyCuboid>()?;
    m.add_class::<PyRegion>()?;

    // Molecular graph hierarchy (base before subclasses)
    m.add_class::<PyGraph>()?;
    m.add_class::<PyAtomistic>()?;
    m.add_class::<PyCoarseGrain>()?;

    // Systems = module-level free functions (no algorithm methods on the classes)
    m.add_function(wrap_pyfunction!(translate, m)?)?;
    m.add_function(wrap_pyfunction!(rotate, m)?)?;
    m.add_function(wrap_pyfunction!(scale, m)?)?;
    m.add_function(wrap_pyfunction!(perceive_aromaticity, m)?)?;
    m.add_function(wrap_pyfunction!(add_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(find_rings, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gasteiger_charges, m)?)?;

    // Field-name convention (`molrs.keys.X`, `molrs.keys.ELEMENT`, …)
    register_keys(m)?;

    // Conformer generation
    m.add_class::<PyConformer>()?;
    m.add_class::<PyConformerReport>()?;
    m.add_class::<PyConformerStageReport>()?;

    // Force field
    m.add_class::<PyForceField>()?;
    m.add_class::<PyMMFFTypifier>()?;
    m.add_class::<PyOplsTypifier>()?;
    m.add_class::<PyPotentials>()?;
    m.add_class::<PyOptReport>()?;
    m.add_class::<PyLBFGS>()?;
    m.add_function(wrap_pyfunction!(forcefield::read_forcefield_xml_py, m)?)?;
    m.add_function(wrap_pyfunction!(forcefield::read_forcefield_xml_str_py, m)?)?;
    m.add_function(wrap_pyfunction!(forcefield::read_opls_xml_py, m)?)?;
    m.add_function(wrap_pyfunction!(forcefield::read_opls_xml_str_py, m)?)?;
    m.add_function(wrap_pyfunction!(forcefield::read_lammps_forcefield_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        forcefield::read_lammps_forcefield_str_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(forcefield::intramolecular_pairs_py, m)?)?;
    m.add_function(wrap_pyfunction!(forcefield::extract_coords_py, m)?)?;
    m.add_function(wrap_pyfunction!(forcefield::build_mmff_potentials_py, m)?)?;

    // Compute analyses
    m.add_class::<PyRDF>()?;
    m.add_class::<PyRDFResult>()?;
    m.add_class::<PyMSD>()?;
    m.add_class::<PyMSDResult>()?;
    m.add_class::<PyMSDTimeSeries>()?;
    m.add_class::<PyCluster>()?;
    m.add_class::<PyClusterResult>()?;
    m.add_class::<PyClusterCenters>()?;
    m.add_class::<PyClusterCentersResult>()?;
    m.add_class::<PyCenterOfMass>()?;
    m.add_class::<PyCenterOfMassResult>()?;
    m.add_class::<PyGyrationTensor>()?;
    m.add_class::<PyInertiaTensor>()?;
    m.add_class::<PyRadiusOfGyration>()?;
    m.add_class::<PyDescriptorRow>()?;
    m.add_class::<PyPca2>()?;
    m.add_class::<PyPcaResult>()?;
    m.add_class::<PyKMeans>()?;
    m.add_class::<PyKMeansResult>()?;

    // Additional analyzers ported from freud (Steinhardt, Nematic, …).
    compute_extra::register(m)?;

    // Raw-compute + explicit-fit classes (phase-02 compute/fit repoint):
    // VACF / EinsteinConductivity / GreenKuboConductivity / … + LinearFit /
    // RunningIntegral / Plateau / DebyeFit / PowerSpectrum / IRSpectrum /
    // RamanSpectrum at the top level (`molrs.VACF`, `molrs.LinearFit`, …).
    compute_fit::register(m)?;

    // TRAVIS-parity computes: geometric distributions (ADF/DDF/distance),
    // Van Hove, Legendre reorientation, hydrogen bonds, spatial distribution,
    // radical-Voronoi tessellation + domain/void analysis.
    compute_travis::register(m)?;

    // Signal processing
    m.add_function(wrap_pyfunction!(signal::signal_acf_fft, m)?)?;
    m.add_function(wrap_pyfunction!(signal::signal_apply_window, m)?)?;
    m.add_function(wrap_pyfunction!(signal::signal_frequency_grid, m)?)?;

    // Dielectric
    dielectric::register_dielectric(m)?;
    transport::register_transport(m)?;

    // Validation
    validate::register_validate(m)?;

    Ok(())
}
