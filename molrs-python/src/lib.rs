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
//! | `Perceive`           | [`PyPerceive`]    | Chemical perception (graph in / graph out) |
//! | `MMFF94Typifier`     | [`PyMMFF94Typifier`]| MMFF94 atom-type assignment              |
//! | `MMFF94STypifier`    | [`PyMMFF94STypifier`]| MMFF94s (static) atom-type assignment   |
//! | `OPLSAATypifier`     | [`PyOPLSAATypifier`]| OPLS-AA atom-type + bonded assignment    |
//! | `AtdTypifier`        | [`PyAtdTypifier`] | antechamber atom types (7 `-at` tables)    |
//! | `BccModel`           | [`PyBccModel`]    | AM1-BCC / ABCG2 bond-charge corrections    |
//! | `MullikenModel`      | [`PyMullikenModel`]| QM Mulliken charges, unchanged            |
//! | `GasteigerModel`     | [`PyGasteigerModel`]| Gasteiger / PEOE charges (no QM input)   |
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
mod schema;
mod spectrum_checks;
mod store;

// Mirrors the molrs core module layout: core/ (store · spatial · system), io/,
// compute/, ff/, conformer/, signal/.
mod core;
mod builder;
use crate::core::spatial::linkedcell::{PyLinkedCell, PyNeighborList, PyNeighborQuery};
use crate::core::spatial::region::{
    PyCuboid, PyHollowSphere, PyParallelepiped, PyRegion, PySphere,
};
use crate::core::spatial::simbox::PyBox;
use crate::core::store::block::PyBlock;
use crate::core::store::frame::{PyFrame, PyMetaValue};
use crate::core::store::record::{PyMolRec, PyObservables};
use crate::core::store::trajectory::{PyScalarObservable, PyTrajectory, PyVectorObservable};
use crate::core::system::element::PyElement;
use crate::core::system::molgraph::{
    PyAtomistic, PyCoarseGrain, PyExtractedSubgraph, PyGraph, PyReaction, PySmartsMatch,
    PySmartsPattern,
};
use crate::core::system::molgraph::{PyRingInfo, align_direction, rotate, scale, translate};
use crate::core::units::{PyQuantity, PyUnit, PyUnitRegistry};
use crate::builder::{PyCarbonTubeBuilder, PyGrapheneBuilder};

mod io;

// Chemical perception: one layer above `core`, mirroring `molrs::perceive`.
mod perceive;
use crate::perceive::PyPerceive;

mod conformer;
use conformer::{PyConformer, PyConformerReport, PyConformerStageReport};

mod ff;
use ff::atd::PyAtdTypifier;
use ff::charge::{PyBccModel, PyGasteigerModel, PyMullikenModel};
use ff::{
    PyForceField, PyLBFGS, PyMMFF94STypifier, PyMMFF94Typifier, PyOPLSAATypifier, PyOptReport,
    PyPotentials, PyTypifier,
};

mod compute;
use compute::{
    PyCenterOfMass, PyCenterOfMassResult, PyCluster, PyClusterCenters, PyClusterCentersResult,
    PyClusterResult, PyDescriptorRow, PyGyrationTensor, PyInertiaTensor, PyKMeans, PyKMeansResult,
    PyMSD, PyMSDResult, PyMSDTimeSeries, PyPca2, PyPcaResult, PyRDF, PyRDFResult,
    PyRadiusOfGyration,
};

mod signal;

/// Register the `keys` submodule mirroring `molrs_core::store::keys` so Python code
/// references the field-name convention by name (`molrs.keys.X`) instead of
/// scattering string literals.

/// Root Python module for the molrs library.
///
/// Registered classes and free functions are listed in the module-level
/// documentation above.
#[pymodule]
#[pyo3(name = "_lib")]
fn molrs_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add("UnitsError", m.py().get_type::<error::UnitsError>())?;

    // Block + Frame
    m.add_class::<PyBlock>()?;
    m.add_class::<PyMetaValue>()?;
    m.add_class::<PyFrame>()?;
    m.add(
        "FRAME_SCHEMA_VERSION",
        ::molrs::store::frame::FRAME_SCHEMA_VERSION,
    )?;

    // Native units
    m.add_class::<PyUnit>()?;
    m.add_class::<PyQuantity>()?;
    m.add_class::<PyUnitRegistry>()?;

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
    m.add_class::<io::PySmilesIR>()?;
    m.add_function(wrap_pyfunction!(io::write_smiles_from_atomistic, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_smarts, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_local_smarts, m)?)?;

    // Trajectory (frame sequence) + observable records
    m.add_class::<PyTrajectory>()?;
    m.add_class::<PyScalarObservable>()?;
    m.add_class::<PyVectorObservable>()?;

    // MolRec record aggregate
    m.add_class::<PyMolRec>()?;
    m.add_class::<PyObservables>()?;

    // Regions
    m.add_class::<PySphere>()?;
    m.add_class::<PyHollowSphere>()?;
    m.add_class::<PyCuboid>()?;
    m.add_class::<PyParallelepiped>()?;
    m.add_class::<PyRegion>()?;

    // Molecular graph hierarchy (base before subclasses)
    m.add_class::<PyElement>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyAtomistic>()?;
    m.add_class::<PyCoarseGrain>()?;
    m.add_class::<PyExtractedSubgraph>()?;
    m.add_class::<PySmartsMatch>()?;
    m.add_class::<PySmartsPattern>()?;
    m.add_class::<PyReaction>()?;

    // Structure builders (graphene, nanotubes, …)
    m.add_class::<PyCarbonTubeBuilder>()?;
    m.add_class::<PyGrapheneBuilder>()?;

    // Systems = module-level free functions (no algorithm methods on the classes)
    m.add_function(wrap_pyfunction!(translate, m)?)?;
    m.add_function(wrap_pyfunction!(rotate, m)?)?;
    m.add_function(wrap_pyfunction!(scale, m)?)?;
    m.add_function(wrap_pyfunction!(align_direction, m)?)?;

    // Chemical perception, as a builder: graph in / graph out, non-mutating.
    m.add_class::<PyPerceive>()?;
    m.add_class::<PyRingInfo>()?;

    // Field-name convention (`molrs.keys.X`, `molrs.keys.ELEMENT`, …)
    schema::register_keys(m)?;
    schema::register_schema(m)?;

    // Conformer generation
    m.add_class::<PyConformer>()?;
    m.add_class::<PyConformerReport>()?;
    m.add_class::<PyConformerStageReport>()?;

    // Force field
    m.add_class::<PyForceField>()?;
    m.add_class::<ff::PyFragmentScaling>()?;
    m.add_class::<PyTypifier>()?;
    m.add_class::<PyMMFF94Typifier>()?;
    m.add_class::<PyMMFF94STypifier>()?;
    m.add_class::<PyOPLSAATypifier>()?;
    m.add_class::<PyAtdTypifier>()?;
    m.add_class::<PyPotentials>()?;

    // Charge models — Python's first native AM1-BCC. One calling convention:
    // `needs_equivalencing()` + `assign(mol, qm=None)`; `BccModel` adds `correct`.
    m.add_class::<PyBccModel>()?;
    m.add_class::<PyMullikenModel>()?;
    m.add_class::<PyGasteigerModel>()?;
    m.add_class::<PyOptReport>()?;
    m.add_class::<PyLBFGS>()?;
    m.add_function(wrap_pyfunction!(ff::read_forcefield_xml_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::read_forcefield_xml_str_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::read_opls_xml_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::read_opls_xml_str_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::read_lammps_forcefield_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::read_lammps_forcefield_str_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::write_lammps_forcefield_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::write_lammps_forcefield_str_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::intramolecular_pairs_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::extract_coords_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::compute_k_ij_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::fragment_scaling_data_py, m)?)?;
    m.add_function(wrap_pyfunction!(ff::scale_lj_py, m)?)?;

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
    compute::extra::register(m)?;

    // Raw-compute + explicit-fit classes (phase-02 compute/fit repoint):
    // VACF / EinsteinConductivity / GreenKuboConductivity / … + LinearFit /
    // CumulativeTrapezoid / Plateau / DebyeFit / PowerSpectrum / IRSpectrum /
    // RamanSpectrum at the top level (`molrs.VACF`, `molrs.LinearFit`, …).
    compute::fitting::register(m)?;

    // analysis-parity computes: geometric distributions (ADF/DDF/distance),
    // Van Hove, Legendre reorientation, hydrogen bonds, spatial distribution,
    // radical-Voronoi tessellation + domain/void analysis.
    compute::analysis::register(m)?;

    // Signal processing
    m.add_function(wrap_pyfunction!(signal::signal_acf_fft, m)?)?;
    m.add_function(wrap_pyfunction!(signal::signal_apply_window, m)?)?;
    m.add_function(wrap_pyfunction!(signal::signal_frequency_grid, m)?)?;

    // Dielectric
    compute::dielectric::register_dielectric(m)?;
    compute::transport::register_transport(m)?;

    // Validation
    spectrum_checks::register_check(m)?;

    Ok(())
}
