//! Python bindings for the chemical-perception layer (`molrs::perceive`,
//! Python `molrs.perceive`).
//!
//! Every perception is a free function at the Rust name, in one of two shapes:
//! `perceive_<fact>(mol)` reports a side table (here `perceive_rings` →
//! `RingInfo`) and leaves the graph alone; `assign_<fact>(mol)` writes the fact
//! onto a **clone** of the molecule as atom / bond props and returns it, so the
//! steps compose. Every `assign_*` takes `&PyAtomistic` (a shared borrow) and
//! returns a **new** `Atomistic`: the non-mutating contract is enforced by the
//! borrow checker rather than by convention. `add_hydrogens` is an edit of the
//! graph and keeps its verb.
//!
//! Chemical perception is all-atom, so every function is typed against
//! `Atomistic`; a `CoarseGrain` leaf is a `TypeError` from PyO3's own
//! extraction, not a wrong answer.
//!
//! [`PySubgraphMatcher`] (`molrs.perceive.SubgraphMatcher`) is the
//! coarse-grained counterpart: it snapshots a bead pattern and lists every
//! occurrence of it in a target `CoarseGrain` as bead-handle groups, in
//! pattern order and without partitioning overlaps; it is typed against
//! `CoarseGrain` the same way, so an `Atomistic` target is a `TypeError`.

mod rings;
mod smarts;

use pyo3::prelude::*;

use molrs::perceive::{
    EquivalenceOptions, SubgraphMatcher, UnknownBondPolicy, add_hydrogens, assign_aromaticity,
    assign_bcc_bond_types, assign_bcc_bond_types_from_connectivity, assign_bond_orders,
    assign_equivalence_classes, assign_kekule_bond_orders, assign_rings, assign_rotatable_bonds,
    assign_stereo,
};

use molrs::core::node_to_u64;

use crate::core::molgraph::{PyAtomistic, PyCoarseGrain};

use crate::error::molrs_error_to_pyerr;

/// Perceive rings (SSSR) and write them onto a clone of ``mol``.
///
/// Every atom and every bond receives ``is_in_ring`` (0/1) and ``n_rings`` —
/// including the acyclic ones, which are explicitly flagged ``0`` rather than
/// left unset. For the ring list itself use :func:`perceive_rings`.
///
/// Examples
/// --------
/// >>> perceived = molrs.perceive.assign_rings(mol)
/// >>> perceived.get(atom, "is_in_ring")
/// 1
/// >>> mol.has(atom, "is_in_ring")   # the input is untouched
/// False
#[pyfunction(name = "assign_rings")]
fn assign_rings_py(py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
    mol.derive(py, assign_rings(mol.core()))
}

/// Bring a molecule to the standard aromatic representation.
///
/// On return every aromatic atom carries ``is_aromatic``, every aromatic bond
/// carries ``bond_type = 4``, and every bond carries an integer
/// ``bond_number`` — the localized Lewis structure. Nothing carries a
/// fractional order: aromaticity is a bond *type*, not the number 1.5.
///
/// An input that already declares its aromatic bonds (a lowercase SMILES) is
/// *kekulized*; one that does not is *perceived* from its integer bond
/// numbers, rings, valences and electron counts, and any assignment it already
/// stated is kept. Hydrogens are neither added nor required.
#[pyfunction(name = "assign_aromaticity")]
fn assign_aromaticity_py(py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
    mol.derive(py, assign_aromaticity(mol.core()))
}

/// Add the hydrogens implied by each heavy atom's open valence.
///
/// Returns a new graph: the heavy-atom skeleton of ``mol`` plus the perceived
/// hydrogens and their bonds; ``mol`` is left untouched.
///
/// Raises
/// ------
/// ValueError
///     If repletion reports a stale atom handle on the graph it built — an
///     invariant no molecule built through this package can break.
#[pyfunction(name = "add_hydrogens")]
fn add_hydrogens_py(py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
    let out = add_hydrogens(mol.core()).map_err(molrs_error_to_pyerr)?;
    mol.derive(py, out)
}

/// Perceive stereochemistry from 3-D coordinates and write it onto a clone of
/// ``mol``: a ``stereo`` prop appears only where a real descriptor was
/// perceived — ``"CW"`` / ``"CCW"`` on atoms, ``"E"`` / ``"Z"`` /
/// ``"either"`` on bonds.
#[pyfunction(name = "assign_stereo")]
fn assign_stereo_py(py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
    mol.derive(py, assign_stereo(mol.core()))
}

/// Perceive rotatable bonds and write them onto a clone of ``mol``.
///
/// A bond is rotatable when it is a single, acyclic bond with two non-terminal
/// endpoints. Every bond receives ``is_rotatable`` (0/1).
///
/// Parameters
/// ----------
/// mol : Atomistic
///     The molecule to perceive; left untouched.
/// unknown_bond : {"not_rotatable", "single"}, default "not_rotatable"
///     What a bond with no ``bond_type`` written counts as (a graph read
///     from connectivity alone). ``"not_rotatable"`` never guesses;
///     ``"single"`` lets it rotate under the degree and ring rules.
///
/// Raises
/// ------
/// ValueError
///     If ``unknown_bond`` is not one of the two policies.
#[pyfunction(name = "assign_rotatable_bonds")]
#[pyo3(signature = (mol, *, unknown_bond = "not_rotatable"))]
fn assign_rotatable_bonds_py(
    py: Python<'_>,
    mol: &PyAtomistic,
    unknown_bond: &str,
) -> PyResult<Py<PyAtomistic>> {
    let unknown = match unknown_bond {
        "not_rotatable" => UnknownBondPolicy::NotRotatable,
        "single" => UnknownBondPolicy::AsSingle,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown_bond must be 'not_rotatable' or 'single', got {other:?}"
            )));
        }
    };
    mol.derive(py, assign_rotatable_bonds(mol.core(), unknown))
}

/// Perceive antechamber's BCC bond types from the bond orders ``mol`` states
/// and write them onto a clone: every bond receives a ``bcc_bond_type`` in
/// ``{1, 2, 3, 6, 7, 8, 9}`` — the alphabet AM1-BCC's atom-type rules and
/// correction table are keyed on, which distinguishes aromatic bonds (7/8) and
/// *delocalized* ones (9, e.g. a carboxylate's two equivalent C–O bonds) from
/// plain orders. The bond's ``type`` — the caller's force-field label — is
/// neither read nor written.
#[pyfunction(name = "assign_bcc_bond_types")]
fn assign_bcc_bond_types_py(py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
    mol.derive(py, assign_bcc_bond_types(mol.core()))
}

/// :func:`assign_bcc_bond_types` as antechamber runs it: the bond orders are
/// judged from the connectivity alone (``bondtype -j full``), whatever orders
/// ``mol`` states. Every hydrogen must be drawn.
#[pyfunction(name = "assign_bcc_bond_types_from_connectivity")]
fn assign_bcc_bond_types_from_connectivity_py(
    py: Python<'_>,
    mol: &PyAtomistic,
) -> PyResult<Py<PyAtomistic>> {
    mol.derive(py, assign_bcc_bond_types_from_connectivity(mol.core()))
}

/// Judge every bond's order from the connectivity alone, as antechamber's
/// ``bondtype -j full`` does, and write it onto a clone of ``mol``: every
/// judged bond gets a localized ``bond_number`` (1/2/3) and the ``bond_type``
/// it implies. The answer follows the atom and bond order, as antechamber's
/// does; the bonds of a residue no valence state closes are left untouched.
#[pyfunction(name = "assign_bond_orders")]
fn assign_bond_orders_py(py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
    mol.derive(py, assign_bond_orders(mol.core()))
}

/// Assign a localized (Kekulé) ``bond_number`` to every aromatic bond of a
/// clone of ``mol``.
///
/// Kekulization and nothing else: a molecule whose aromatic bonds are not
/// marked yet comes back unchanged, because deciding *which* bonds are
/// aromatic belongs to :func:`assign_aromaticity`. An aromatic bond that
/// already carries a legal number keeps it; a system with no legal assignment
/// is left entirely unchanged rather than half-assigned.
#[pyfunction(name = "assign_kekule_bond_orders")]
fn assign_kekule_bond_orders_py(py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
    mol.derive(py, assign_kekule_bond_orders(mol.core()))
}

/// Perceive charge-equivalence classes and write them onto a clone of ``mol``:
/// every atom receives an ``equiv_class`` id, antechamber's default ``-eq 1``
/// partition (the one AM1-BCC averages its AM1 charges over). Whether to
/// average is a property of the charge model (:meth:`BccModel.needs_equivalencing`),
/// not of the graph.
#[pyfunction(name = "assign_equivalence_classes")]
fn assign_equivalence_classes_py(py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
    mol.derive(
        py,
        assign_equivalence_classes(mol.core(), EquivalenceOptions::default()),
    )
}

/// Bead-group pattern matching over coarse-grained graphs —
/// `molrs.perceive.SubgraphMatcher`.
///
/// Snapshots a bead pattern once; :meth:`find` lists every induced occurrence
/// of it in a target :class:`~molrs.core.CoarseGrain`. Beads match on equal
/// ``bead_type``; bonds match on adjacency.
///
/// ``find`` does **not** partition: overlapping groups are all returned, and a
/// caller that needs disjoint groups selects among them.
///
/// Parameters
/// ----------
/// pattern : CoarseGrain
///     The bead pattern, e.g. ``CgSmilesIr("{[#1][#4]}").to_coarsegrain()``.
///     It is copied, so later edits to it do not affect the matcher.
///
/// Raises
/// ------
/// TypeError
///     If ``pattern`` is not a :class:`~molrs.core.CoarseGrain`.
///
/// Examples
/// --------
/// The two groups below share the middle bead:
///
/// >>> pattern = molrs.io.smiles.CgSmilesIr("{[#1][#4]}").to_coarsegrain()
/// >>> target = molrs.io.smiles.CgSmilesIr("{[#1][#4][#1]}").to_coarsegrain()
/// >>> len(molrs.perceive.SubgraphMatcher(pattern).find(target))
/// 2
#[pyclass(module = "molrs.perceive", name = "SubgraphMatcher", frozen)]
pub struct PySubgraphMatcher {
    inner: SubgraphMatcher,
}

#[pymethods]
impl PySubgraphMatcher {
    #[new]
    fn new(pattern: PyRef<'_, PyCoarseGrain>) -> Self {
        Self {
            inner: SubgraphMatcher::new(pattern.core()),
        }
    }

    /// Every induced occurrence of the pattern in ``target``.
    ///
    /// One group per distinct bead set: ``group[i]`` is the target bead
    /// matched to pattern bead ``i``, in the pattern's bead order. Groups are
    /// not partitioned, so two groups may share beads. The GIL is released
    /// while matching.
    ///
    /// Parameters
    /// ----------
    /// target : CoarseGrain
    ///     The bead graph to search.
    ///
    /// Returns
    /// -------
    /// list[list[int]]
    ///     Target bead handles, one list per group; ``[]`` when there is no
    ///     occurrence (or the pattern or target is empty).
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``target`` is not a :class:`~molrs.core.CoarseGrain`.
    fn find(&self, py: Python<'_>, target: PyRef<'_, PyCoarseGrain>) -> Vec<Vec<u64>> {
        let (matcher, target) = (&self.inner, target.core());
        py.detach(|| matcher.find(target))
            .into_iter()
            .map(|group| group.into_iter().map(node_to_u64).collect())
            .collect()
    }

    fn __repr__(&self) -> String {
        "SubgraphMatcher()".to_string()
    }
}

/// Register `molrs.perceive`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(assign_rings_py, m)?,
        wrap_pyfunction!(assign_aromaticity_py, m)?,
        wrap_pyfunction!(add_hydrogens_py, m)?,
        wrap_pyfunction!(assign_stereo_py, m)?,
        wrap_pyfunction!(assign_rotatable_bonds_py, m)?,
        wrap_pyfunction!(assign_bcc_bond_types_py, m)?,
        wrap_pyfunction!(assign_bcc_bond_types_from_connectivity_py, m)?,
        wrap_pyfunction!(assign_bond_orders_py, m)?,
        wrap_pyfunction!(assign_kekule_bond_orders_py, m)?,
        wrap_pyfunction!(assign_equivalence_classes_py, m)?,
        wrap_pyfunction!(rings::perceive_rings_py, m)?,
    ] {
        crate::add_function(m, "molrs.perceive", f)?;
    }
    m.add_class::<PySubgraphMatcher>()?;
    m.add_class::<rings::PyRingInfo>()?;
    m.add_class::<smarts::PySmartsPattern>()?;
    m.add_class::<smarts::PySmartsMatch>()?;
    m.add_class::<smarts::PyReaction>()?;
    Ok(())
}
