//! SMARTS matching and reaction transforms (`molrs::perceive::smarts`):
//! [`PySmartsPattern`] and the [`PySmartsMatch`] it yields, and the
//! reaction-SMARTS [`PyReaction`]. A pattern is a query over a *perceived*
//! graph — matching needs ring membership and aromaticity — so it is
//! perception's, not a text format's; the SMILES / SMARTS front-end that
//! writes the text is `molrs.io`'s.

use std::collections::HashMap;

use molrs::perceive::smarts::{MatchOptions, Reaction, RingPrimitive, SmartsPattern};
use molrs::system::{NodeId, node_from_u64, node_to_u64};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::core::system::molgraph::PyAtomistic;
use crate::error::molrs_error_to_pyerr;

// ---------------------------------------------------------------------------
// PySmartsMatch / PySmartsPattern — atom-map-aware SMARTS matcher over Atomistic
// ---------------------------------------------------------------------------

/// One SMARTS match, exposed to Python as `molrs.perceive.SmartsMatch`.
///
/// ``atoms`` stores molecule atom handles in query-atom order. ``mapping``
/// stores the Daylight atom-map projection (``:1`` → atom handle), and is empty
/// when the query carries no map labels.
#[pyclass(
    module = "molrs.perceive",
    name = "SmartsMatch",
    skip_from_py_object,
    subclass
)]
#[derive(Clone)]
pub struct PySmartsMatch {
    atoms: Vec<u64>,
    mapping: HashMap<u32, u64>,
}

#[pymethods]
impl PySmartsMatch {
    /// Molecule atom handles in query-atom order.
    #[getter]
    fn atoms(&self) -> Vec<u64> {
        self.atoms.clone()
    }

    /// Daylight atom-map projection (``:n`` label -> molecule atom handle).
    #[getter]
    fn mapping(&self) -> HashMap<u32, u64> {
        self.mapping.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SmartsMatch(atoms={:?}, mapping={:?})",
            self.atoms, self.mapping
        )
    }
}

/// Compiled SMARTS query, exposed to Python as `molrs.perceive.SmartsPattern`.
///
/// A thin wrapper over the core [`SmartsPattern`] (`molrs/src/core/chem/smarts`)
/// — the same backtracking subgraph-isomorphism engine that drives the OPLS-AA
/// typifier. Matching is non-uniquified (RDKit ``uniquify=False``): every
/// distinct query-atom → mol-atom embedding is reported as a
/// :class:`SmartsMatch`.
///
/// Daylight atom maps (``[C:1]``) are parsed and carried through but add **no**
/// match constraint (they are "ignored in molecule SMARTS"); each match's
/// :attr:`SmartsMatch.mapping` is its ``{map_number: atom_handle}`` dict.
///
/// Examples
/// --------
/// >>> pat = molrs.perceive.SmartsPattern("[C:1][O:2][H:3]")
/// >>> pat.find_matches(methanol)[0].mapping
/// {1: <C>, 2: <O>, 3: <H>}
#[pyclass(module = "molrs.perceive", name = "SmartsPattern", subclass)]
pub struct PySmartsPattern {
    inner: SmartsPattern,
}

#[pymethods]
impl PySmartsPattern {
    /// Parse a SMARTS string. Raises ``ValueError`` on a syntax error.
    #[new]
    fn new(smarts: &str) -> PyResult<Self> {
        let inner = SmartsPattern::parse(smarts).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Whether at least one match exists in `mol`.
    #[pyo3(signature = (mol, *, labels=None, root=None))]
    fn has_match(
        &self,
        mol: &PyAtomistic,
        labels: Option<HashMap<u64, String>>,
        root: Option<u64>,
    ) -> bool {
        let core_labels = labels.map(|labels| {
            labels
                .into_iter()
                .map(|(h, l)| (node_from_u64(h), l))
                .collect::<HashMap<NodeId, String>>()
        });
        self.inner.has_match(
            mol.core(),
            MatchOptions {
                labels: core_labels.as_ref(),
                root: root.map(node_from_u64),
                limit: None,
            },
        )
    }

    /// All matches, each a :class:`SmartsMatch`. ``labels`` supplies the
    /// ``%LABEL`` context, ``root`` pins query atom 0 to one atom handle, and
    /// ``limit`` stops after N embeddings.
    #[pyo3(signature = (mol, *, labels=None, root=None, limit=None))]
    fn find_matches(
        &self,
        mol: &PyAtomistic,
        labels: Option<HashMap<u64, String>>,
        root: Option<u64>,
        limit: Option<usize>,
    ) -> Vec<PySmartsMatch> {
        let core_labels = labels.map(|labels| {
            labels
                .into_iter()
                .map(|(h, l)| (node_from_u64(h), l))
                .collect::<HashMap<NodeId, String>>()
        });
        let matches = self.inner.find(
            mol.core(),
            MatchOptions {
                labels: core_labels.as_ref(),
                root: root.map(node_from_u64),
                limit,
            },
        );
        matches
            .iter()
            .map(|m| PySmartsMatch {
                atoms: m.atoms().iter().map(|&atom| node_to_u64(atom)).collect(),
                mapping: self
                    .inner
                    .mapped(m)
                    .into_iter()
                    .map(|(label, atom)| (label, node_to_u64(atom)))
                    .collect(),
            })
            .collect()
    }

    /// Number of query atoms in the pattern.
    #[getter]
    fn num_query_atoms(&self) -> usize {
        self.inner.num_query_atoms()
    }

    /// Longest shortest-path length (bonds) on the query atom graph.
    ///
    /// Isolated atoms → ``0``. Pure syntax fact for molpy region typing.
    #[getter]
    fn max_bond_depth(&self) -> usize {
        self.inner.max_bond_depth()
    }

    /// Ring primitives used in this pattern (syntax only; no boundedness).
    ///
    /// Each item is ``(kind, n)`` where ``kind`` is one of
    /// ``"sized"`` / ``"membership"`` / ``"ring_count"`` / ``"ring_bond_count"``
    /// and ``n`` is ``None`` for membership.
    #[getter]
    fn ring_primitives(&self) -> Vec<(String, Option<u32>)> {
        self.inner
            .ring_primitives()
            .into_iter()
            .map(|p| match p {
                RingPrimitive::Sized(n) => ("sized".into(), Some(n)),
                RingPrimitive::Membership => ("membership".into(), None),
                RingPrimitive::RingCount(n) => ("ring_count".into(), Some(n)),
                RingPrimitive::RingBondCount(n) => ("ring_bond_count".into(), Some(n)),
            })
            .collect()
    }

    /// The ``:n`` atom-map label of query atom `query_atom` (``None`` if
    /// unlabelled / out of range).
    fn map_label(&self, query_atom: usize) -> Option<u32> {
        self.inner.map_label(query_atom)
    }

    fn __repr__(&self) -> String {
        format!(
            "SmartsPattern(num_query_atoms={})",
            self.inner.num_query_atoms()
        )
    }
}

// ---------------------------------------------------------------------------
// PyReaction — Daylight reaction-SMARTS (SMIRKS) transform over an Atomistic
// ---------------------------------------------------------------------------

/// Compiled reaction SMARTS, exposed to Python as `molrs.perceive.Reaction`.
///
/// A thin wrapper over the core [`Reaction`] (`molrs/src/core/chem/smarts`).
/// Parses ``reactants >> products`` (tolerating an ignored ``>agent>`` field),
/// derives the graph edit from the Daylight atom-map diff, and applies it to one
/// matched occurrence in place. Reacting atoms may carry SMARTS queries
/// (RDKit-style reaction SMARTS); only concrete product atoms are addable.
///
/// Examples
/// --------
/// >>> rxn = molrs.perceive.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
/// >>> rxn.forming_bonds                 # [(1, 2)]
/// >>> binding = {}                       # match each reactant component ...
/// >>> for pat in rxn.reactant_patterns:  # ... and merge the map->atom dicts
/// ...     binding.update(pat.find_matches(mol)[0].mapping)
/// >>> rxn.apply(mol, binding)            # edits `mol` in place
#[pyclass(module = "molrs.perceive", name = "Reaction", subclass)]
pub struct PyReaction {
    inner: Reaction,
}

#[pymethods]
impl PyReaction {
    /// Parse a reaction SMARTS. Raises ``ValueError`` on a syntax or
    /// map-consistency error (e.g. an atom map that appears on only one side).
    #[new]
    fn new(reaction_smarts: &str) -> PyResult<Self> {
        let inner = Reaction::parse(reaction_smarts).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        crate::pickle::reduce_via_type(slf.as_any(), (slf.borrow().inner.source().to_owned(),))
    }

    /// The reactant components (LHS), one :class:`SmartsPattern` per top-level
    /// ``.`` component, for matching / pairing each independently.
    #[getter]
    fn reactant_patterns(&self) -> Vec<PySmartsPattern> {
        self.inner
            .reactants()
            .iter()
            .map(|p| PySmartsPattern { inner: p.clone() })
            .collect()
    }

    /// The ``(map_a, map_b)`` pairs of newly formed bonds between preserved
    /// atoms — the distance criterion for picking a reacting occurrence. Bonds
    /// that merely change order, and bonds to added atoms, are excluded.
    #[getter]
    fn forming_bonds(&self) -> Vec<(u32, u32)> {
        self.inner.forming_bonds()
    }

    /// Apply the transform to `mol` in place at the occurrence pinned by
    /// `binding` (``{map_number: atom_handle}``). Deletes unmapped-LHS atoms,
    /// adds unmapped-RHS atoms (no coordinates), forms/breaks bonds, then
    /// regenerates angle/dihedral topology and re-perceives aromaticity.
    ///
    /// Returns the deduplicated, deterministically-ordered list of *surviving*
    /// touched atom handles (formed/broken/order-changed bond endpoints, added
    /// atoms, deleted atoms' surviving neighbours, and prop-set atoms). Deleted
    /// atoms' own handles are never included. The caller expands this seed set
    /// into a retype-safe region.
    ///
    /// ``refresh=False`` skips the per-apply whole-graph angle/dihedral
    /// regeneration + aromaticity re-perception: a batch caller (crosslinking a
    /// melt with many edits) passes it and refreshes ONCE at the end, turning an
    /// O(edits × N) cost into O(edits × local). Matching only needs bonds, which
    /// are updated in place regardless.
    #[pyo3(signature = (mol, binding, labels=None, refresh=true))]
    fn apply(
        &self,
        mol: &mut PyAtomistic,
        binding: HashMap<u32, u64>,
        labels: Option<HashMap<u64, String>>,
        refresh: bool,
    ) -> PyResult<Vec<u64>> {
        let resolved: HashMap<u32, NodeId> = binding
            .into_iter()
            .map(|(k, v)| (k, node_from_u64(v)))
            .collect();
        let core_labels: HashMap<NodeId, String> = labels
            .unwrap_or_default()
            .into_iter()
            .map(|(h, l)| (node_from_u64(h), l))
            .collect();
        self.inner
            .apply(mol.core_mut(), &resolved, &core_labels, refresh)
            .map(|touched| touched.into_iter().map(node_to_u64).collect())
            .map_err(molrs_error_to_pyerr)
    }

    /// Compile every binding against the intact graph, then apply the disjoint
    /// transforms as one batch. Leaving groups are deleted with one relation
    /// scan, and one touched-handle list is returned per binding.
    #[pyo3(signature = (mol, bindings, labels=None, refresh=true))]
    fn apply_many(
        &self,
        mol: &mut PyAtomistic,
        bindings: Vec<HashMap<u32, u64>>,
        labels: Option<HashMap<u64, String>>,
        refresh: bool,
    ) -> PyResult<Vec<Vec<u64>>> {
        let resolved: Vec<HashMap<u32, NodeId>> = bindings
            .into_iter()
            .map(|binding| {
                binding
                    .into_iter()
                    .map(|(k, v)| (k, node_from_u64(v)))
                    .collect()
            })
            .collect();
        let core_labels: HashMap<NodeId, String> = labels
            .unwrap_or_default()
            .into_iter()
            .map(|(h, l)| (node_from_u64(h), l))
            .collect();
        self.inner
            .apply_many(mol.core_mut(), &resolved, &core_labels, refresh)
            .map(|sets| {
                sets.into_iter()
                    .map(|touched| touched.into_iter().map(node_to_u64).collect())
                    .collect()
            })
            .map_err(molrs_error_to_pyerr)
    }

    /// ``apply_many`` plus RHS-created handles in product creation order.
    ///
    /// The second list is intentionally not reconstructed by sorting handles:
    /// batch deletion may reuse graph slots in an order unrelated to product
    /// atom order.
    #[pyo3(signature = (mol, bindings, labels=None, refresh=true))]
    #[allow(
        clippy::type_complexity,
        reason = "Python returns products and created handles as a tuple"
    )]
    fn apply_many_detailed(
        &self,
        mol: &mut PyAtomistic,
        bindings: Vec<HashMap<u32, u64>>,
        labels: Option<HashMap<u64, String>>,
        refresh: bool,
    ) -> PyResult<(Vec<Vec<u64>>, Vec<Vec<u64>>)> {
        let resolved: Vec<HashMap<u32, NodeId>> = bindings
            .into_iter()
            .map(|binding| {
                binding
                    .into_iter()
                    .map(|(k, v)| (k, node_from_u64(v)))
                    .collect()
            })
            .collect();
        let core_labels: HashMap<NodeId, String> = labels
            .unwrap_or_default()
            .into_iter()
            .map(|(h, l)| (node_from_u64(h), l))
            .collect();
        self.inner
            .apply_many_detailed(mol.core_mut(), &resolved, &core_labels, refresh)
            .map(|(touched_sets, created_sets)| {
                let handles = |sets: Vec<Vec<NodeId>>| {
                    sets.into_iter()
                        .map(|set| set.into_iter().map(node_to_u64).collect())
                        .collect()
                };
                (handles(touched_sets), handles(created_sets))
            })
            .map_err(molrs_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "Reaction(reactants={}, forming_bonds={:?})",
            self.inner.reactants().len(),
            self.inner.forming_bonds()
        )
    }
}
