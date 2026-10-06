//! Python bindings for `molrs::spatial` (`molrs.spatial`): the simulation
//! cell (`Box`), neighbour search (`NeighborList`, `Neighbors`,
//! `NeighborQuery`, `VerletSkin`), geometric regions (`Region` and its
//! solids), triangle meshes (`TriMesh`) and point paths (`Trace`).

pub mod mesh;
pub mod neighborlist;
pub mod region;
pub mod simbox;
pub mod trace;

use pyo3::prelude::*;

/// Register `molrs.spatial`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<simbox::PyBox>()?;
    m.add_class::<neighborlist::PyNeighborList>()?;
    m.add_class::<neighborlist::PyNeighbors>()?;
    m.add_class::<neighborlist::PyNeighborQuery>()?;
    m.add_class::<neighborlist::PyVerletSkin>()?;
    m.add_class::<mesh::PyTriMesh>()?;
    m.add_class::<region::PySphere>()?;
    m.add_class::<region::PyCuboid>()?;
    m.add_class::<region::PyParallelepiped>()?;
    m.add_class::<region::PyHalfSpace>()?;
    m.add_class::<region::PyCylinder>()?;
    m.add_class::<region::PyEllipsoid>()?;
    m.add_class::<region::PyPolyhedron>()?;
    m.add_class::<region::PySphereUnion>()?;
    m.add_class::<region::PyRegion>()?;
    m.add_class::<trace::PyTrace>()?;
    Ok(())
}
