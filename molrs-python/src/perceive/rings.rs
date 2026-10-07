//! Ring facts (`molrs::perceive::perceive_rings`): [`PyRingSet`] reports the SSSR
//! rings of a molecule and the systems they fuse into, without touching it.

use molrs::core::{node_from_u64, node_to_u64};
use pyo3::prelude::*;

use crate::core::molgraph::PyAtomistic;

/// The ring facts of a molecule: SSSR rings and the systems they fuse into.
///
/// Returned by :func:`perceive_rings`, which runs the perception once; every
/// method reads the result.
///
/// Not to be confused with :func:`assign_rings`, which answers a
/// different question — it *decorates* a graph with ring flags and hands the
/// graph back. This type *reports*, and never touches the molecule.
///
/// Examples
/// --------
/// >>> rings = molrs.perceive.perceive_rings(molrs.io.smiles.SmilesIr("c1ccccc1").to_atomistic())
/// >>> rings.n_rings()
/// 1
/// >>> rings.ring_sizes()
/// [6]
#[pyclass(module = "molrs.perceive", name = "RingSet", subclass)]
pub struct PyRingSet {
    inner: molrs::perceive::RingSet,
}

#[pymethods]
impl PyRingSet {
    /// Every ring, as a list of atom handles forming a closed path.
    fn rings(&self) -> Vec<Vec<u64>> {
        self.inner
            .rings()
            .iter()
            .map(|ring| ring.iter().map(|&a| node_to_u64(a)).collect())
            .collect()
    }

    /// Number of rings.
    fn n_rings(&self) -> usize {
        self.inner.n_rings()
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
    fn n_atom_rings(&self, atom: u64) -> usize {
        self.inner.n_atom_rings(node_from_u64(atom))
    }

    /// Size of the smallest ring containing `atom`, or ``None``.
    fn smallest_ring_containing_atom(&self, atom: u64) -> Option<usize> {
        self.inner
            .smallest_ring_containing_atom(node_from_u64(atom))
    }

    fn __repr__(&self) -> String {
        format!(
            "RingSet(n_rings={}, sizes={:?})",
            self.inner.n_rings(),
            self.inner.ring_sizes()
        )
    }
}

/// Perceive the rings of ``mol`` (SSSR / minimum cycle basis) as a
/// :class:`RingSet` side table; ``mol`` is left untouched.
#[pyfunction(name = "perceive_rings")]
pub(super) fn perceive_rings_py(mol: &Bound<'_, PyAtomistic>) -> PyRingSet {
    PyRingSet {
        inner: molrs::perceive::perceive_rings(mol.borrow().core()),
    }
}
