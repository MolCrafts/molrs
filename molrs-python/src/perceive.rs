//! Python bindings for the chemical-perception layer (`molrs::perceive`).
//!
//! One class, [`PyPerceive`] (`molrs.Perceive`), mirroring the Rust builder: the
//! layer's free functions have four different shapes (a side table, an in-place
//! mutation returning a count, a graph-out transform, and maps), and the builder
//! normalises all four to a single contract —
//!
//! > **graph in / graph out, non-mutating** — each `find_*` clones the molecule,
//! > writes the perceived facts onto the clone as atom / bond props, and returns
//! > it. The input is never touched.
//!
//! That contract is the whole reason the builder exists, and it is the property a
//! binding is most likely to lose: handing PyO3 a `&mut` and returning `None` would
//! still "work" for a caller who only looks at the output. Every method here takes
//! `&PyAtomistic` (a shared borrow) and returns a **new** `Atomistic`, so the shape
//! is enforced by the borrow checker rather than by convention.
//!
//! The free functions this layer also publishes (`molrs.perceive_aromaticity`,
//! `molrs.add_hydrogens`, `molrs.find_rings`) are unchanged and stay where they are
//! registered, in [`crate::core::system::molgraph`] — the builder does not replace
//! them: `find_rings` the free function returns the ring **list**, `Perceive`'s
//! returns the annotated **graph**, and a graph composes with the next finder.
//!
//! Perception is all-atom, so every method is typed against `Atomistic`; a
//! `CoarseGrain` leaf is a `TypeError` from PyO3's own extraction, not a wrong
//! answer.

use pyo3::prelude::*;

use molrs::perceive::Perceive;

use crate::core::system::molgraph::PyAtomistic;

/// Chemical perception, as a builder — `molrs.Perceive`.
///
/// Exposed to Python as `molrs.Perceive`. Every ``find_*`` method is graph-in /
/// graph-out and **non-mutating**.
///
/// Props written (atom / bond components on the returned clone):
///
/// ===============================  ==========================  ==========================
/// Method                           Atom props                  Bond props
/// ===============================  ==========================  ==========================
/// ``find_rings``                   ``is_in_ring``, ``n_rings`` ``is_in_ring``, ``n_rings``
/// ``find_aromaticity``             ``is_aromatic``             ``bond_type``, ``bond_number``
/// ``find_hydrogens``               — (adds H atoms)            — (adds H bonds)
/// ``find_stereo``                  ``stereo``                  ``stereo``
/// ``find_rotatable``               —                           ``is_rotatable``
/// ``find_bond_types``              —                           ``bcc_bond_type``
/// ``find_kekule_orders``           —                           ``bond_number``
/// ``find_equivalence_classes``     ``equiv_class``             —
/// ===============================  ==========================  ==========================
///
/// Examples
/// --------
/// >>> perceived = molrs.Perceive().find_rings(mol)
/// >>> perceived.get(atom, "is_in_ring")
/// 1
/// >>> mol.has(atom, "is_in_ring")   # the input is untouched
/// False
// `subclass`: molpy layers a thin `Perceive` over this one so its finders
// return molpy graphs. Without it the base type is final and molpy cannot
// import at all.
#[pyclass(module = "molrs", name = "Perceive", subclass)]
#[derive(Debug)]
pub struct PyPerceive {
    inner: Perceive,
}

#[pymethods]
impl PyPerceive {
    /// Create a perception builder with default settings.
    #[new]
    fn new() -> Self {
        Self {
            inner: Perceive::new(),
        }
    }

    /// Perceive rings (SSSR) and project them onto the graph.
    ///
    /// Every atom and every bond receives ``is_in_ring`` (0/1) and ``n_rings`` —
    /// including the acyclic ones, which are explicitly flagged ``0`` rather than
    /// left unset.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule to perceive; left untouched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     A clone of ``mol`` carrying the ring props.
    fn find_rings(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.find_rings(mol.core()))
    }

    /// Bring a molecule to the standard aromatic representation.
    ///
    /// On return every aromatic atom carries ``is_aromatic``, every aromatic
    /// bond carries ``bond_type = 4``, and every bond carries an integer
    /// ``bond_number`` — the localized Lewis structure. Nothing carries a
    /// fractional order: aromaticity is a bond *type*, not the number 1.5.
    ///
    /// Two inputs get two treatments. An input that already declares its
    /// aromatic bonds (a lowercase SMILES) is *kekulized* — the notation
    /// answered which bonds are aromatic, and only the phase is missing. An
    /// input that does not is *perceived* from its integer bond numbers, rings,
    /// valences and electron counts, and any assignment it already stated is
    /// kept.
    ///
    /// Hydrogens are neither added nor required: implicit hydrogens are read
    /// off each atom's valence, so :meth:`find_hydrogens` stays an independent
    /// operation and running it first changes no answer here.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule to standardize; left untouched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     A clone of ``mol`` carrying both facts about every bond.
    fn find_aromaticity(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.find_aromaticity(mol.core()))
    }

    /// Add the hydrogens implied by each heavy atom's open valence.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule to fill; left untouched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     A new graph: the heavy-atom skeleton of ``mol`` plus the perceived
    ///     hydrogens and their bonds.
    fn find_hydrogens(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.find_hydrogens(mol.core()))
    }

    /// Perceive stereochemistry from 3-D coordinates and project it onto the graph.
    ///
    /// A ``stereo`` prop appears only where a real descriptor was perceived:
    /// ``"CW"`` / ``"CCW"`` on atoms, ``"E"`` / ``"Z"`` / ``"either"`` on bonds.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule to perceive; left untouched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     A clone of ``mol`` carrying a ``stereo`` prop on each perceived
    ///     stereocentre and stereo bond, and none elsewhere.
    fn find_stereo(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.find_stereo(mol.core()))
    }

    /// Perceive rotatable bonds and project them onto the graph.
    ///
    /// A bond is rotatable when it is a single, acyclic bond with two non-terminal
    /// endpoints. Every bond is flagged — non-rotatable ones explicitly with ``0``.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule to perceive; left untouched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     A clone of ``mol`` with ``is_rotatable`` (0/1) on every bond.
    fn find_rotatable(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.find_rotatable(mol.core()))
    }

    /// Perceive antechamber bond types and project them onto the graph.
    ///
    /// Every bond receives a ``bcc_bond_type`` prop in ``{1, 2, 3, 6, 7, 8, 9}`` —
    /// the alphabet AM1-BCC's atom-type rules and correction table are keyed on,
    /// which distinguishes aromatic bonds (7/8) and *delocalized* ones (9, e.g. a
    /// carboxylate's two equivalent C–O bonds) from plain orders. The bond's
    /// ``type`` — the caller's force-field label — is neither read nor written.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule to perceive; left untouched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     A clone of ``mol`` with ``bcc_bond_type`` on every bond.
    fn find_bond_types(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.find_bond_types(mol.core()))
    }

    /// Assign a localized (Kekulé) ``bond_number`` to every aromatic bond.
    ///
    /// Kekulization and nothing else: a molecule whose aromatic bonds are not
    /// marked yet comes back unchanged, because deciding *which* bonds are
    /// aromatic belongs to :meth:`find_aromaticity`. Reach for this directly
    /// when the input already declares its aromatic subgraph and only the phase
    /// is missing.
    ///
    /// An aromatic bond that already carries a legal number keeps it — a file
    /// that round-trips does not come back renumbered. A system with no legal
    /// assignment is left entirely unchanged rather than half-assigned.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule to kekulize; left untouched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     A clone of ``mol`` whose aromatic bonds carry a legal localized
    ///     number.
    fn find_kekule_orders(
        &self,
        py: Python<'_>,
        mol: &PyAtomistic,
    ) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.find_kekule_orders(mol.core()))
    }

    /// Perceive charge-equivalence classes and project them onto the graph.
    ///
    /// antechamber's default ``-eq 1`` — the path-score partition AM1-BCC averages
    /// its AM1 charges over. Perception stops at the classes: whether to average is
    /// a property of the charge model (`BccModel` declares it via
    /// :meth:`BccModel.needs_equivalencing`), not of the graph.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule to perceive; left untouched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     A clone of ``mol`` with an ``equiv_class`` id on every atom.
    fn find_equivalence_classes(
        &self,
        py: Python<'_>,
        mol: &PyAtomistic,
    ) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.find_equivalence_classes(mol.core()))
    }

    fn __repr__(&self) -> String {
        "Perceive()".to_string()
    }
}
