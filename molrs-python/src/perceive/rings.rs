//! Ring facts (`molrs::perceive::rings`): [`PyRingInfo`] reports the SSSR
//! rings of a molecule and the systems they fuse into, without touching it.

use molrs::core::{node_from_u64, node_to_u64};
use pyo3::prelude::*;

use crate::core::molgraph::PyAtomistic;

/// The ring facts of a molecule: SSSR rings and the systems they fuse into.
///
/// Perception runs once, in the constructor; every method reads the result.
///
/// Not to be confused with :meth:`Perceive.find_rings`, which answers a
/// different question — it *decorates* a graph with ring flags and hands the
/// graph back. This type *reports*, and never touches the molecule.
///
/// Examples
/// --------
/// >>> rings = molrs.perceive.RingInfo(molrs.io.smiles.SmilesIr("c1ccccc1").to_atomistic())
/// >>> rings.num_rings()
/// 1
/// >>> rings.ring_sizes()
/// [6]
#[pyclass(module = "molrs.perceive", name = "RingInfo", subclass)]
pub struct PyRingInfo {
    inner: molrs::perceive::rings::RingInfo,
}

#[pymethods]
impl PyRingInfo {
    /// Perceive the rings of `mol` (SSSR / minimum cycle basis).
    #[new]
    fn new(mol: &Bound<'_, PyAtomistic>) -> Self {
        Self {
            inner: molrs::perceive::rings::find_rings(mol.borrow().core()),
        }
    }

    /// Every ring, as a list of atom handles forming a closed path.
    fn rings(&self) -> Vec<Vec<u64>> {
        self.inner
            .rings()
            .iter()
            .map(|ring| ring.iter().map(|&a| node_to_u64(a)).collect())
            .collect()
    }

    /// Number of rings.
    fn num_rings(&self) -> usize {
        self.inner.num_rings()
    }

    /// Atom count of every ring, ascending.
    fn ring_sizes(&self) -> Vec<usize> {
        self.inner.ring_sizes()
    }

    /// Rings that share at least one atom, unioned: benzene → one system of 6,
    /// naphthalene → one of 10, biphenyl → two of 6.
    fn ring_systems(&self) -> Vec<Vec<u64>> {
        self.inner
            .ring_systems()
            .iter()
            .map(|system| system.iter().map(|&a| node_to_u64(a)).collect())
            .collect()
    }

    /// Atom count of the largest fused / bridged ring system (naphthalene →
    /// 10); ``0`` for an acyclic molecule.
    fn max_ring_system_size(&self) -> usize {
        self.inner.max_ring_system_size()
    }

    /// Whether `atom` belongs to any ring.
    fn is_atom_in_ring(&self, atom: u64) -> bool {
        self.inner.is_atom_in_ring(node_from_u64(atom))
    }

    /// Number of rings containing `atom`.
    fn num_atom_rings(&self, atom: u64) -> usize {
        self.inner.num_atom_rings(node_from_u64(atom))
    }

    /// Size of the smallest ring containing `atom`, or ``None``.
    fn smallest_ring_containing_atom(&self, atom: u64) -> Option<usize> {
        self.inner
            .smallest_ring_containing_atom(node_from_u64(atom))
    }

    fn __repr__(&self) -> String {
        format!(
            "RingInfo(num_rings={}, sizes={:?})",
            self.inner.num_rings(),
            self.inner.ring_sizes()
        )
    }
}
