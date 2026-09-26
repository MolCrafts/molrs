//! Crate-internal labelled, induced subgraph matching under a caller-supplied node-label compatibility relation (VF2 lineage, sharing `graph_hash`'s feasibility check).
//!
//! A **match** of a pattern graph `P` in a target graph `T` is an injective
//! map `m` from pattern nodes to target nodes such that
//!
//! - `compatible(label(p), label(m(p)))` holds for every pattern node `p`,
//!   where `compatible` is the caller's relation — it need not be equality,
//!   nor symmetric;
//! - the map is **induced**: two matched target nodes share an arity-2 edge
//!   iff their pattern nodes do, and then with equal edge labels.
//!
//! What a label is depends on the [`Labelling`] a snapshot was taken with:
//! [`Labelling::Structural`] reads `graph_hash`'s vocabulary
//! ([`node_label_str`] — element, else `bead_type` — and the `GraphView` bond
//! bits), [`Labelling::BeadType`] reads `bead_type` alone and leaves every
//! edge unlabelled. Pattern and target are always labelled alike.
//!
//! Degree is a pruning bound (`deg T ≥ deg P`), never part of a label, so a
//! pattern bead of degree 1 matches a chain-interior target bead.
//!
//! # Algorithm
//!
//! State-space backtracking in the VF2 lineage (L. P. Cordella, P. Foggia,
//! C. Sansone and M. Vento, "A (sub)graph isomorphism algorithm for matching
//! large graphs", *IEEE TPAMI* **26**, 1367 (2004),
//! doi:10.1109/TPAMI.2004.75), the same lineage as
//! [`is_isomorphic`](super::graph_hash::is_isomorphic), whose structural
//! feasibility check [`feasible`] this search shares.
//!
//! - **Candidates.** Per pattern node, the target nodes whose label the
//!   relation accepts and whose degree is large enough.
//! - **Order.** Fewest candidates first; each later node is the
//!   fewest-candidate pattern node adjacent to one already placed
//!   (neighbourhood extension, as in the SMARTS matcher), so its images come
//!   from its anchor's image neighbourhood rather than the whole target. A
//!   disconnected pattern restarts from the fewest-candidate unplaced node.
//! - **Result.** Every match, `match[i]` = the target node of pattern node
//!   row `i`, sorted lexicographically by target node row. Automorphic images
//!   are all reported; grouping them is the caller's business.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::store::keys;
use crate::system::graph_hash::{GraphView, adjacency_map, feasible, node_label_str};
use crate::system::molgraph::{MolGraph, NodeId};

/// Empty slot in a partial map.
const UNMAPPED: usize = usize::MAX;

/// The vocabulary a [`MatchGraph`] labels its nodes and edges with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Labelling {
    /// `graph_hash`'s vocabulary: a node is [`node_label_str`] (element,
    /// else `bead_type`), an edge its `GraphView` bond bits.
    Structural,
    /// A coarse bead is its `bead_type` (`""` when absent) and every edge is
    /// unlabelled, whatever `element` / `bond_type` / `bond_number` columns a
    /// frame-derived coarse graph also carries.
    BeadType,
}

impl Labelling {
    /// The label of node `id` of `g` in this vocabulary.
    fn node(self, g: &MolGraph, id: NodeId) -> String {
        match self {
            Self::Structural => node_label_str(g, id),
            Self::BeadType => g
                .get_node(id)
                .ok()
                .and_then(|atom| atom.get_str(keys::BEAD_TYPE).map(str::to_owned))
                .unwrap_or_default(),
        }
    }
}

/// A labelled snapshot of one graph: taken once, matched against any number
/// of times, as a pattern or as a target.
#[derive(Debug, Clone)]
pub(crate) struct MatchGraph {
    /// The vocabulary `labels` and the edge labels of `adj` are in.
    labelling: Labelling,
    /// Node handles in row order.
    nodes: Vec<NodeId>,
    /// Label per node row.
    labels: Vec<String>,
    /// Neighbour row → edge label, per node row.
    adj: Vec<HashMap<usize, u64>>,
}

impl MatchGraph {
    /// Snapshot `g`'s nodes, labels and labelled adjacency under
    /// `labelling`.
    pub(crate) fn new(g: &MolGraph, labelling: Labelling) -> Self {
        let view = GraphView::build(g);
        let mut adj = adjacency_map(&view);
        if labelling == Labelling::BeadType {
            for label in adj.iter_mut().flat_map(HashMap::values_mut) {
                *label = 0;
            }
        }
        let labels = view.nodes.iter().map(|&id| labelling.node(g, id)).collect();
        Self {
            labelling,
            nodes: view.nodes,
            labels,
            adj,
        }
    }
}

/// A live graph is snapshotted in the graph-hash vocabulary.
impl From<&MolGraph> for MatchGraph {
    fn from(g: &MolGraph) -> Self {
        Self::new(g, Labelling::Structural)
    }
}

/// What [`SubgraphMatcher::find_all`] runs against: a live graph, snapshotted
/// per call in the pattern's vocabulary, or a [`MatchGraph`] reused across
/// calls.
pub(crate) trait MatchTarget {
    /// This target as a snapshot in `labelling`.
    fn snapshot(&self, labelling: Labelling) -> Cow<'_, MatchGraph>;
}

impl MatchTarget for MolGraph {
    fn snapshot(&self, labelling: Labelling) -> Cow<'_, MatchGraph> {
        Cow::Owned(MatchGraph::new(self, labelling))
    }
}

impl MatchTarget for MatchGraph {
    /// # Panics
    ///
    /// When the snapshot was taken in another vocabulary than the pattern's:
    /// labels of two vocabularies are not comparable, so matching them would
    /// answer a question nobody asked.
    fn snapshot(&self, labelling: Labelling) -> Cow<'_, MatchGraph> {
        assert_eq!(
            self.labelling, labelling,
            "a pattern and its target snapshot must share one labelling"
        );
        Cow::Borrowed(self)
    }
}

/// A labelled, induced subgraph matcher over one pattern graph.
///
/// Built once per pattern and run against any number of targets; the
/// pattern's snapshot fixes the [`Labelling`] every target is read in.
#[derive(Debug, Clone)]
pub(crate) struct SubgraphMatcher {
    pattern: MatchGraph,
}

impl SubgraphMatcher {
    /// A matcher over `pattern`: a live graph (snapshotted in the graph-hash
    /// vocabulary) or a snapshot taken in the vocabulary of the caller's
    /// choice.
    pub(crate) fn new(pattern: impl Into<MatchGraph>) -> Self {
        Self {
            pattern: pattern.into(),
        }
    }

    /// The pattern snapshot, reusable as a [`MatchTarget`].
    pub(crate) fn pattern(&self) -> &MatchGraph {
        &self.pattern
    }

    /// Every induced match of the pattern in `target` under
    /// `compatible(pattern_label, target_label)`.
    ///
    /// `match[i]` is the target node assigned to pattern node row `i`; the
    /// matches are sorted lexicographically by target node row. An empty
    /// pattern has no matches.
    pub(crate) fn find_all<T: MatchTarget + ?Sized>(
        &self,
        target: &T,
        compatible: impl Fn(&str, &str) -> bool,
    ) -> Vec<Vec<NodeId>> {
        let n = self.pattern.labels.len();
        if n == 0 {
            return Vec::new();
        }
        let target = target.snapshot(self.pattern.labelling);
        let p_adj = &self.pattern.adj;

        let is_cand: Vec<Vec<bool>> = (0..n)
            .map(|u| {
                target
                    .labels
                    .iter()
                    .zip(&target.adj)
                    .map(|(t_label, nbrs)| {
                        nbrs.len() >= p_adj[u].len() && compatible(&self.pattern.labels[u], t_label)
                    })
                    .collect()
            })
            .collect();
        let counts: Vec<usize> = is_cand
            .iter()
            .map(|row| row.iter().filter(|&&c| c).count())
            .collect();
        if counts.contains(&0) {
            return Vec::new();
        }

        let order = self.search_order(&counts);
        let mut search = Search {
            order: &order,
            p_adj,
            t_adj: &target.adj,
            is_cand: &is_cand,
            map_pt: vec![UNMAPPED; n],
            map_tp: vec![UNMAPPED; target.nodes.len()],
            found: Vec::new(),
        };
        search.extend(0);

        let mut found = search.found;
        found.sort_unstable();
        found
            .into_iter()
            .map(|m| m.into_iter().map(|row| target.nodes[row]).collect())
            .collect()
    }

    /// Placement order as `(pattern row, anchor row)`: fewest candidates
    /// first (ties by row), then always the fewest-candidate unplaced node
    /// adjacent to a placed one, anchored on its earliest-placed neighbour.
    /// `None` anchors a component root.
    fn search_order(&self, counts: &[usize]) -> Vec<(usize, Option<usize>)> {
        let adj = &self.pattern.adj;
        let n = counts.len();
        let mut placed_at = vec![UNMAPPED; n];
        let mut order = Vec::with_capacity(n);
        for step in 0..n {
            let unplaced = || (0..n).filter(|&u| placed_at[u] == UNMAPPED);
            let frontier = unplaced()
                .filter(|&u| adj[u].keys().any(|&w| placed_at[w] != UNMAPPED))
                .min_by_key(|&u| (counts[u], u));
            let next = frontier.or_else(|| unplaced().min_by_key(|&u| (counts[u], u)));
            let Some(u) = next else {
                break;
            };
            let anchor = adj[u]
                .keys()
                .copied()
                .filter(|&w| placed_at[w] != UNMAPPED)
                .min_by_key(|&w| placed_at[w]);
            placed_at[u] = step;
            order.push((u, anchor));
        }
        order
    }
}

/// Mutable state of one `find_all` backtracking run.
struct Search<'a> {
    /// `(pattern row, anchor row)` per depth.
    order: &'a [(usize, Option<usize>)],
    p_adj: &'a [HashMap<usize, u64>],
    t_adj: &'a [HashMap<usize, u64>],
    /// `is_cand[p][t]`: label accepted and degree sufficient.
    is_cand: &'a [Vec<bool>],
    /// Pattern row → target row.
    map_pt: Vec<usize>,
    /// Target row → pattern row.
    map_tp: Vec<usize>,
    /// Complete matches as target rows, indexed by pattern row.
    found: Vec<Vec<usize>>,
}

impl Search<'_> {
    /// Place the pattern node at `depth`, recursing to the leaves.
    fn extend(&mut self, depth: usize) {
        if depth == self.order.len() {
            self.found.push(self.map_pt.clone());
            return;
        }
        let (u, anchor) = self.order[depth];
        match anchor {
            Some(a) => {
                let t_adj = self.t_adj;
                for &v in t_adj[self.map_pt[a]].keys() {
                    self.try_place(depth, u, v);
                }
            }
            None => {
                for v in 0..self.map_tp.len() {
                    self.try_place(depth, u, v);
                }
            }
        }
    }

    /// Map pattern `u` to target `v` if free, a candidate and structurally
    /// feasible, then recurse.
    fn try_place(&mut self, depth: usize, u: usize, v: usize) {
        if self.map_tp[v] != UNMAPPED || !self.is_cand[u][v] {
            return;
        }
        if !feasible(u, v, self.p_adj, self.t_adj, &self.map_pt, &self.map_tp) {
            return;
        }
        self.map_pt[u] = v;
        self.map_tp[v] = u;
        self.extend(depth + 1);
        self.map_pt[u] = UNMAPPED;
        self.map_tp[v] = UNMAPPED;
    }
}

#[cfg(test)]
mod tests {
    use super::SubgraphMatcher;
    use crate::system::coarsegrain::CoarseGrain;
    use crate::system::molgraph::{MolGraph, NodeId};

    /// A coarse-grained graph with one bead per entry of `types` (row = index)
    /// and one CG bond (edge label 0) per `(row, row)` pair.
    fn cg(types: &[&str], bonds: &[(usize, usize)]) -> CoarseGrain {
        let mut g = CoarseGrain::new();
        let ids: Vec<NodeId> = types.iter().map(|t| g.add_bead_bare(t)).collect();
        for &(a, b) in bonds {
            g.add_bond(ids[a], ids[b]).expect("fixture bond");
        }
        g
    }

    /// Run the matcher and translate every match into target node rows.
    ///
    /// Contract under test: `match[i]` is the target node assigned to pattern
    /// node row `i`, and `find_all` returns its matches sorted
    /// lexicographically by those target rows, so the expected vectors below
    /// are exact.
    fn rows(
        pattern: &MolGraph,
        target: &MolGraph,
        compatible: impl Fn(&str, &str) -> bool,
    ) -> Vec<Vec<usize>> {
        let order: Vec<NodeId> = target.node_ids().collect();
        let row = |id: NodeId| {
            order
                .iter()
                .position(|&n| n == id)
                .expect("a match names live target nodes")
        };
        SubgraphMatcher::new(pattern)
            .find_all(target, compatible)
            .into_iter()
            .map(|m| m.into_iter().map(row).collect())
            .collect()
    }

    fn equal(p: &str, t: &str) -> bool {
        p == t
    }

    #[test]
    fn equality_finds_both_embeddings_of_an_edge_in_a_path() {
        let pattern = cg(&["A", "B"], &[(0, 1)]);
        let target = cg(&["A", "B", "A"], &[(0, 1), (1, 2)]);
        assert_eq!(
            rows(pattern.as_molgraph(), target.as_molgraph(), equal),
            vec![vec![0, 1], vec![2, 1]]
        );
    }

    #[test]
    fn caller_relation_licenses_coarse_types_to_pattern_labels() {
        // (coarse type, template label) pairs; compatible(pattern, target).
        let rules = [("1", "B"), ("4", "A")];
        let compatible = |p: &str, t: &str| rules.iter().any(|&(ct, label)| ct == t && label == p);

        let pattern = cg(&["B", "A"], &[(0, 1)]);
        let one_four = cg(&["1", "4"], &[(0, 1)]);
        let one_one = cg(&["1", "1"], &[(0, 1)]);

        assert_eq!(
            rows(pattern.as_molgraph(), one_four.as_molgraph(), compatible),
            vec![vec![0, 1]]
        );
        assert_eq!(
            rows(pattern.as_molgraph(), one_one.as_molgraph(), compatible),
            Vec::<Vec<usize>>::new(),
            "no rule licenses coarse type '1' as label 'A'"
        );
    }

    #[test]
    fn edge_matches_every_arm_of_a_star() {
        let pattern = cg(&["A", "B"], &[(0, 1)]);
        let star = cg(&["B", "A", "A", "A"], &[(0, 1), (0, 2), (0, 3)]);
        assert_eq!(
            rows(pattern.as_molgraph(), star.as_molgraph(), equal),
            vec![vec![1, 0], vec![2, 0], vec![3, 0]]
        );
    }

    #[test]
    fn match_is_induced_so_a_path_misses_a_triangle() {
        let path = cg(&["A", "B", "C"], &[(0, 1), (1, 2)]);
        let triangle = cg(&["A", "B", "C"], &[(0, 1), (1, 2), (2, 0)]);
        assert_eq!(
            rows(path.as_molgraph(), triangle.as_molgraph(), equal),
            Vec::<Vec<usize>>::new(),
            "target A–C is bonded but pattern A–C is not"
        );
    }

    #[test]
    fn triangle_matches_triangle() {
        let triangle = cg(&["A", "B", "C"], &[(0, 1), (1, 2), (2, 0)]);
        let matches = rows(triangle.as_molgraph(), triangle.as_molgraph(), equal);
        assert!(!matches.is_empty());
        assert_eq!(matches, vec![vec![0, 1, 2]], "distinct labels fix the map");
    }
}
