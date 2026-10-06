//! Python bindings for `molrs::system` (`molrs.system`): the molecular-graph
//! hierarchy (`Graph`, `Atomistic`, `CoarseGrain`), its live node and
//! relation views (`Atom`, `Bond`, …, `Refs`, `RelationBuckets`), the
//! extracted-ball result, elements and the index-only `Topology`.

pub mod element;
pub mod molgraph;
pub mod topology;
pub mod views;

use pyo3::prelude::*;

/// Register `molrs.system` (each base class before its subclasses).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<element::PyElement>()?;
    m.add_class::<topology::PyTopology>()?;
    m.add_class::<molgraph::PyGraph>()?;
    m.add_class::<molgraph::PyAtomistic>()?;
    m.add_class::<molgraph::PyCoarseGrain>()?;
    m.add_class::<molgraph::PyExtractedSubgraph>()?;
    m.add_class::<views::PyNodeRef>()?;
    m.add_class::<views::PyAtom>()?;
    m.add_class::<views::PyVirtualSite>()?;
    m.add_class::<views::PyDrudeParticle>()?;
    m.add_class::<views::PyMasslessSite>()?;
    m.add_class::<views::PyBead>()?;
    m.add_class::<views::PyRelationRef>()?;
    m.add_class::<views::PyBond>()?;
    m.add_class::<views::PyAngle>()?;
    m.add_class::<views::PyDihedral>()?;
    m.add_class::<views::PyImproper>()?;
    m.add_class::<views::PyPort>()?;
    m.add_class::<views::PyCGBond>()?;
    m.add_class::<views::PyRefs>()?;
    m.add_class::<views::PyRelationBuckets>()?;
    Ok(())
}
