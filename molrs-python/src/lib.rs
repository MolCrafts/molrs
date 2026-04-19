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
//! | `Potentials`         | [`PyPotentials`]  | Compiled energy/force evaluator            |
//! | `RDF` / `MSD` / `Cluster` |              | Structural analysis                        |
//!
//! # Float Precision
//!
//! By default all floating-point arrays use `f32` (numpy `float32`).
//! Enable the `f64` feature for double precision (`float64`).

use pyo3::prelude::*;

mod helpers;
mod store;

mod simbox;
use simbox::PyBox;

mod linkedcell;
use linkedcell::{PyLinkedCell, PyNeighborList, PyNeighborQuery};

mod block;
use block::PyBlock;

mod frame;
use frame::{PyFrame, PyGrid};

mod io;

mod molrec;
use molrec::{
    PyGridObservable, PyMolRec, PyObservables, PyScalarObservable, PyTrajectory,
    PyVectorObservable,
};

mod region;
use region::{PyHollowSphere, PyRegion, PySphere};

pub(crate) mod molgraph;
use molgraph::PyAtomistic;

mod embed;
use embed::{PyEmbedOptions, PyEmbedReport, PyEmbedResult, PyStageReport};

mod forcefield;
use forcefield::{PyForceField, PyMMFFTypifier, PyPotentials};

mod compute;
use compute::{
    PyCenterOfMass, PyCenterOfMassResult, PyCluster, PyClusterCenters, PyClusterCentersResult,
    PyClusterResult, PyDescriptorRow, PyGyrationTensor, PyInertiaTensor, PyKMeans, PyKMeansResult,
    PyMSD, PyMSDResult, PyMSDTimeSeries, PyPca2, PyPcaResult, PyRDF, PyRDFResult,
    PyRadiusOfGyration,
};

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

    // Block + Frame + Grid
    m.add_class::<PyBlock>()?;
    m.add_class::<PyFrame>()?;
    m.add_class::<PyGrid>()?;

    // I/O + SMILES
    // Readers
    m.add_function(wrap_pyfunction!(io::read_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_xyz, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_xyz_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_lammps, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_lammps_traj, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_chgcar_file, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_cube_file, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_cube_file, m)?)?;
    // Writers
    m.add_function(wrap_pyfunction!(io::write_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_xyz, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_lammps, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_lammps_traj, m)?)?;
    // SMILES
    m.add_function(wrap_pyfunction!(io::parse_smiles, m)?)?;
    m.add_class::<io::PySmilesIR>()?;

    // MolRec
    m.add_class::<PyTrajectory>()?;
    m.add_class::<PyMolRec>()?;
    m.add_class::<PyObservables>()?;
    m.add_class::<PyScalarObservable>()?;
    m.add_class::<PyVectorObservable>()?;
    m.add_class::<PyGridObservable>()?;

    // Regions
    m.add_class::<PySphere>()?;
    m.add_class::<PyHollowSphere>()?;
    m.add_class::<PyRegion>()?;

    // Molecular graph
    m.add_class::<PyAtomistic>()?;

    // Embed
    m.add_class::<PyEmbedOptions>()?;
    m.add_class::<PyEmbedReport>()?;
    m.add_class::<PyEmbedResult>()?;
    m.add_class::<PyStageReport>()?;
    m.add_function(wrap_pyfunction!(embed::generate_3d_py, m)?)?;

    // Force field
    m.add_class::<PyForceField>()?;
    m.add_class::<PyMMFFTypifier>()?;
    m.add_class::<PyPotentials>()?;
    m.add_function(wrap_pyfunction!(forcefield::read_forcefield_xml_py, m)?)?;
    m.add_function(wrap_pyfunction!(forcefield::extract_coords_py, m)?)?;

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

    Ok(())
}
