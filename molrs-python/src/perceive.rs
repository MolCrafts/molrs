//! Python bindings for the chemical-perception layer (`molrs::perceive`).
//!
//! The main class, [`PyPerceive`] (`molrs.perceive.Perceive`), mirrors the Rust
//! builder: the layer's free functions have four different shapes (a side
//! table, an in-place mutation returning a count, a graph-out transform, and
//! maps), and the builder normalises all four to a single contract —
//!
//! > **graph in / graph out, non-mutating** — each `find_*` clones the molecule,
//! > writes the perceived facts onto the clone as atom / bond props, and returns
//! > it. The input is never touched.
//!
//! That contract is the whole reason the builder exists, and it is the property a
//! binding is most likely to lose: handing PyO3 a `&mut` and returning `None` would
//! still "work" for a caller who only looks at the output. Every `Perceive` method takes
//! `&PyAtomistic` (a shared borrow) and returns a **new** `Atomistic`, so the shape
//! is enforced by the borrow checker rather than by convention.
//!
//! No perception step is bound as a free function: Python reaches every one of
//! them through a method — `molrs.perceive.Perceive.find_hydrogens`,
//! `.find_aromaticity`, `.find_rings` and the rest of the `find_*` family. The
//! layer's other two classes are registered in
//! [`crate::core::system::molgraph`] and are not replaced by the builder:
//! `molrs.perceive.RingInfo` answers a different question — it *reports* the
//! ring list, where `Perceive.find_rings` hands back the annotated graph
//! that composes with the next finder — and `molrs.perceive.SmartsPattern`
//! matches a query against a perceived graph.
//!
//! Chemical perception is all-atom, so every `Perceive` method is typed
//! against `Atomistic`; a `CoarseGrain` leaf is a `TypeError` from PyO3's own
//! extraction, not a wrong answer.
//!
//! The coarse-grained classes are [`PySubgraphMatcher`]
//! (`molrs.perceive.SubgraphMatcher`) and [`PyCoarsener`]
//! (`molrs.perceive.Coarsener`). `SubgraphMatcher` snapshots a bead pattern and
//! lists every occurrence of it in a target `CoarseGrain` as bead-handle
//! groups, in pattern order and without partitioning overlaps; it is typed
//! against `CoarseGrain` the same way, so an `Atomistic` target is a
//! `TypeError`. `Coarsener(source)` holds a `CoarseGrain` or `Atomistic` source
//! and `coarsen(groups, names)` maps disjoint node groups of it onto the sites
//! of a new `CoarseGrain` (centre of mass, summed mass, one type per group).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use molrs::perceive::{CoarsenError, Coarsener, Perceive, SubgraphMatcher};
use molrs::system::molgraph::{MolGraph, NodeId, node_from_u64, node_to_u64};

use crate::core::system::molgraph::{PyAtomistic, PyCoarseGrain, center_error_message};
use crate::helpers::molrs_error_to_pyerr;

/// Chemical perception, as a builder — `molrs.perceive.Perceive`.
///
/// Exposed to Python as `molrs.perceive.Perceive`. Every ``find_*`` method is graph-in /
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
/// >>> perceived = molrs.perceive.Perceive().find_rings(mol)
/// >>> perceived.get(atom, "is_in_ring")
/// 1
/// >>> mol.has(atom, "is_in_ring")   # the input is untouched
/// False
// `subclass`: molpy layers a thin `Perceive` over this one so its finders
// return molpy graphs. Without it the base type is final and molpy cannot
// import at all.
#[pyclass(module = "molrs.perceive", name = "Perceive", subclass)]
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
        mol.derive(py, self.inner.find_rings(mol.core()))
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
        mol.derive(py, self.inner.find_aromaticity(mol.core()))
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
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If repletion reports a stale atom handle on the graph it built — an
    ///     invariant no molecule built through this package can break.
    fn find_hydrogens(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
        let out = self
            .inner
            .find_hydrogens(mol.core())
            .map_err(molrs_error_to_pyerr)?;
        mol.derive(py, out)
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
        mol.derive(py, self.inner.find_stereo(mol.core()))
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
    /// unknown_bond : {"not_rotatable", "single"}, default "not_rotatable"
    ///     What a bond with no ``bond_type`` written counts as (a graph read
    ///     from connectivity alone). ``"not_rotatable"`` never guesses;
    ///     ``"single"`` lets it rotate under the degree and ring rules.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     A clone of ``mol`` with ``is_rotatable`` (0/1) on every bond.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``unknown_bond`` is not one of the two policies.
    #[pyo3(signature = (mol, *, unknown_bond = "not_rotatable"))]
    fn find_rotatable(
        &self,
        py: Python<'_>,
        mol: &PyAtomistic,
        unknown_bond: &str,
    ) -> PyResult<Py<PyAtomistic>> {
        use molrs::perceive::rotatable::UnknownBondPolicy;
        let unknown = match unknown_bond {
            "not_rotatable" => UnknownBondPolicy::NotRotatable,
            "single" => UnknownBondPolicy::AsSingle,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown_bond must be 'not_rotatable' or 'single', got {other:?}"
                )));
            }
        };
        mol.derive(py, self.inner.find_rotatable(mol.core(), unknown))
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
        mol.derive(py, self.inner.find_bond_types(mol.core()))
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
    fn find_kekule_orders(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
        mol.derive(py, self.inner.find_kekule_orders(mol.core()))
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
        mol.derive(py, self.inner.find_equivalence_classes(mol.core()))
    }

    fn __repr__(&self) -> String {
        "Perceive()".to_string()
    }
}

/// Bead-group pattern matching over coarse-grained graphs —
/// `molrs.perceive.SubgraphMatcher`.
///
/// Snapshots a bead pattern once; :meth:`find` lists every induced occurrence
/// of it in a target :class:`~molrs.CoarseGrain`. Beads match on equal
/// ``bead_type``; bonds match on adjacency.
///
/// ``find`` does **not** partition: overlapping groups are all returned, and a
/// caller that needs disjoint groups selects among them.
///
/// Parameters
/// ----------
/// pattern : CoarseGrain
///     The bead pattern, e.g. ``CGSmilesIR("{[#1][#4]}").to_coarsegrain()``.
///     It is copied, so later edits to it do not affect the matcher.
///
/// Raises
/// ------
/// TypeError
///     If ``pattern`` is not a :class:`~molrs.CoarseGrain`.
///
/// Examples
/// --------
/// The two groups below share the middle bead:
///
/// >>> pattern = molrs.io.CGSmilesIR("{[#1][#4]}").to_coarsegrain()
/// >>> target = molrs.io.CGSmilesIR("{[#1][#4][#1]}").to_coarsegrain()
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
    ///     If ``target`` is not a :class:`~molrs.CoarseGrain`.
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

/// The graph a [`PyCoarsener`] holds: either public leaf that carries nodes
/// with positions and masses.
enum CoarsenSource {
    CoarseGrain(Py<PyCoarseGrain>),
    Atomistic(Py<PyAtomistic>),
}

/// Coarse-graining of a held source graph — `molrs.perceive.Coarsener`.
///
/// Site ``I`` of the result stands for ``groups[I]``: it sits at the group's
/// mass-weighted centre (Å; no periodic imaging, so unwrap first), carries the
/// group's summed ``mass`` and ``bead_type = names[I]``, and records the
/// group's handles as its members. Two sites are bonded once when a source
/// bond joins their groups. The groups must be disjoint; ``SubgraphMatcher``
/// returns overlapping ones, so the caller selects first.
///
/// Parameters
/// ----------
/// source : CoarseGrain or Atomistic
///     The graph whose nodes are grouped. The object is held, not copied, and
///     read at each :meth:`coarsen` call.
///
/// Raises
/// ------
/// TypeError
///     If ``source`` is neither a :class:`~molrs.CoarseGrain` nor an
///     :class:`~molrs.Atomistic`.
///
/// Examples
/// --------
/// >>> groups = molrs.perceive.SubgraphMatcher(pattern).find(cg)
/// >>> sites = molrs.perceive.Coarsener(cg).coarsen(groups, ["PMA"] * len(groups))
#[pyclass(module = "molrs.perceive", name = "Coarsener", frozen)]
pub struct PyCoarsener {
    source: CoarsenSource,
}

#[pymethods]
impl PyCoarsener {
    #[new]
    fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        let source = if let Ok(cg) = source.cast::<PyCoarseGrain>() {
            CoarsenSource::CoarseGrain(cg.clone().unbind())
        } else if let Ok(mol) = source.cast::<PyAtomistic>() {
            CoarsenSource::Atomistic(mol.clone().unbind())
        } else {
            return Err(PyTypeError::new_err(format!(
                "Coarsener source must be a CoarseGrain or an Atomistic, not {}",
                source.get_type().name()?
            )));
        };
        Ok(Self { source })
    }

    /// A new :class:`~molrs.CoarseGrain` with one site per group.
    ///
    /// The GIL is released while mapping.
    ///
    /// Parameters
    /// ----------
    /// groups : Sequence[Sequence[int]]
    ///     Disjoint, non-empty node-handle groups of the source.
    /// names : Sequence[str]
    ///     One site ``bead_type`` per group.
    ///
    /// Returns
    /// -------
    /// CoarseGrain
    ///     The sites, in group order; empty when ``groups`` is empty.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``groups`` and ``names`` differ in length, a group is empty, a
    ///     handle is listed twice, or a group has no centre (a stale handle, a
    ///     missing coordinate or mass, a non-positive total mass); handles are
    ///     named by their int value.
    fn coarsen(
        &self,
        py: Python<'_>,
        groups: Vec<Vec<u64>>,
        names: Vec<String>,
    ) -> PyResult<Py<PyCoarseGrain>> {
        let groups: Vec<Vec<NodeId>> = groups
            .into_iter()
            .map(|group| group.into_iter().map(node_from_u64).collect())
            .collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let run = |graph: &MolGraph| py.detach(|| Coarsener::new(graph).coarsen(&groups, &names));
        let sites = match &self.source {
            CoarsenSource::CoarseGrain(cg) => run(cg.bind(py).borrow().core().as_molgraph()),
            CoarsenSource::Atomistic(mol) => run(mol.bind(py).borrow().core().as_molgraph()),
        }
        .map_err(|e| PyValueError::new_err(coarsen_error_message(e)))?;
        PyCoarseGrain::from_core(py, sites)
    }

    fn __repr__(&self) -> String {
        let source = match self.source {
            CoarsenSource::CoarseGrain(_) => "CoarseGrain",
            CoarsenSource::Atomistic(_) => "Atomistic",
        };
        format!("Coarsener(<{source}>)")
    }
}

/// The message of a [`CoarsenError`] as Python sees it: node ids as int
/// handles, never `NodeId(..)`. Every variant is matched by name.
fn coarsen_error_message(e: CoarsenError) -> String {
    match e {
        CoarsenError::Center { group, source } => {
            format!("group {group}: {}", center_error_message(source))
        }
        e @ (CoarsenError::LengthMismatch { .. }
        | CoarsenError::EmptyGroup { .. }
        | CoarsenError::Overlap { .. }) => e.to_string(),
    }
}
