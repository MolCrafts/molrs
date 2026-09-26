//! [`Mapping`]: a [`FragGraph`] plus, per unit, the coarse source bead ids, their template bead labels and the (coarse type, template label) rules that licensed them.
//!
//! A `Mapping` is what `builder::FragLibrary::map` produces and what a
//! hand-built assembly states directly: which coarse beads each unit stands
//! for, and which template bead each of them is. A coarse bead is one
//! particle of a coarse-grained model, standing for a group of atoms; a unit
//! is one copy of a template (an all-atom fragment whose atoms are grouped
//! into template beads).

use std::collections::HashSet;

use crate::error::MolRsError;
use crate::spatial::Trace;
use crate::store::keys;
use crate::system::coarsegrain::CoarseGrain;
use crate::system::frag_graph::FragGraph;
use crate::system::molgraph::NodeId;

/// A [`FragGraph`] whose every unit is tied to coarse source beads.
///
/// For unit `u`, `sources[u]` holds its coarse bead ids in template-bead
/// order and `labels[u]` the template bead label each of those beads was
/// assigned. `rules` is the (coarse type, template label) relation that
/// licensed the assignment: the only bridge between a coarse `bead_type` and
/// a template label, which are never compared with each other.
///
/// `sources[u]` is defined up to a port-preserving label automorphism of the
/// template: two correspondences that give every coarse bead the same label
/// and the same ports are equally valid, and `FragLibrary::map` reports the
/// lexicographically smallest.
///
/// Invariants, established by [`new`](Self::new): one non-empty source list
/// per node, `labels[u]` as long as `sources[u]`, no source id in two places,
/// and every label licensed by some rule. Whether a source bead's *current* type
/// satisfies its rule is checked when the mapping is traced
/// ([`trace`](Self::trace)), against the coarse graph it is traced over.
#[derive(Debug, Clone)]
pub struct Mapping {
    graph: FragGraph,
    sources: Vec<Vec<NodeId>>,
    labels: Vec<Vec<String>>,
    rules: Vec<(String, String)>,
}

impl Mapping {
    /// Tie `graph`'s units to their source beads and template labels.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when the number of source lists
    /// differs from the node count, when a unit has no source bead (it could
    /// be neither traced nor placed), when `labels` does not have the shape of
    /// `sources` (list count or any `labels[u].len() != sources[u].len()`),
    /// when one source id appears twice (within a unit or across units), or
    /// when a label is the template label of no rule.
    pub fn new(
        graph: FragGraph,
        sources: Vec<Vec<NodeId>>,
        labels: Vec<Vec<String>>,
        rules: Vec<(String, String)>,
    ) -> Result<Self, MolRsError> {
        let n = graph.nodes().len();
        if sources.len() != n {
            return Err(MolRsError::validation(format!(
                "mapping has {} source lists for a {n}-node frag graph",
                sources.len()
            )));
        }
        if labels.len() != n {
            return Err(MolRsError::validation(format!(
                "mapping has {} label lists for a {n}-node frag graph",
                labels.len()
            )));
        }
        let licensed: HashSet<&str> = rules.iter().map(|(_, label)| label.as_str()).collect();
        let mut seen: HashSet<NodeId> = HashSet::new();
        for (u, (src, lab)) in sources.iter().zip(&labels).enumerate() {
            if src.is_empty() {
                return Err(MolRsError::validation(format!(
                    "unit {u} has no source bead"
                )));
            }
            if src.len() != lab.len() {
                return Err(MolRsError::validation(format!(
                    "unit {u} has {} source beads but {} labels",
                    src.len(),
                    lab.len()
                )));
            }
            if let Some(&dup) = src.iter().find(|&&id| !seen.insert(id)) {
                return Err(MolRsError::validation(format!(
                    "source bead {dup:?} of unit {u} already belongs to a unit"
                )));
            }
            if let Some(label) = lab.iter().find(|l| !licensed.contains(l.as_str())) {
                return Err(MolRsError::validation(format!(
                    "unit {u} assigns template label '{label}', which no rule licenses"
                )));
            }
        }
        Ok(Self {
            graph,
            sources,
            labels,
            rules,
        })
    }

    /// The unit-level topology.
    pub fn graph(&self) -> &FragGraph {
        &self.graph
    }

    /// Unit `unit`'s coarse source bead ids in template-bead order, or `None`
    /// when `unit` is not a unit index.
    pub fn sources(&self, unit: usize) -> Option<&[NodeId]> {
        self.sources.get(unit).map(Vec::as_slice)
    }

    /// Unit `unit`'s template bead labels, aligned with
    /// [`sources`](Self::sources), or `None` when `unit` is not a unit index.
    pub fn labels(&self, unit: usize) -> Option<&[String]> {
        self.labels.get(unit).map(Vec::as_slice)
    }

    /// The (coarse type, template label) relation that licensed the labels.
    pub fn rules(&self) -> &[(String, String)] {
        &self.rules
    }

    /// Number of units (frag-graph nodes).
    pub fn n_units(&self) -> usize {
        self.sources.len()
    }

    /// The source beads' positions as a ragged [`Trace`]: unit `u` holds the
    /// x/y/z (Å, copied as stored) of `sources(u)`, in template-bead order.
    ///
    /// Every source bead is checked against this mapping's own record first:
    /// its current `bead_type` `t` and its template label `L` must satisfy
    /// `(t, L) ∈ rules()`. A coarse graph that is not the one mapped, or that
    /// was edited after mapping, is refused rather than traced.
    ///
    /// The unit sequence a placer pairs with this trace is
    /// [`graph().nodes()`](FragGraph::nodes); point order within a unit is
    /// template-bead order by construction, so no separate bead sequence
    /// exists.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] naming the unit, bead, type and label when a
    /// bead's `(bead_type, label)` pair is licensed by no rule, or naming the
    /// unit and bead when the bead has no `bead_type` or lacks any of x/y/z;
    /// the error of [`CoarseGrain::get_bead`] when a source bead is not in
    /// `source`.
    pub fn trace(&self, source: &CoarseGrain) -> Result<Trace, MolRsError> {
        let licensed: HashSet<(&str, &str)> = self
            .rules
            .iter()
            .map(|(t, l)| (t.as_str(), l.as_str()))
            .collect();
        let mut points = Vec::with_capacity(self.sources.iter().map(Vec::len).sum());
        let mut offsets = Vec::with_capacity(self.sources.len() + 1);
        offsets.push(0);
        for (u, (src, lab)) in self.sources.iter().zip(&self.labels).enumerate() {
            for (&bead, label) in src.iter().zip(lab) {
                let atom = source.get_bead(bead)?;
                // `CoarseGrain` guarantees every bead a `bead_type`; this
                // guards that invariant rather than a reachable input.
                let Some(bead_type) = atom.get_str(keys::BEAD_TYPE) else {
                    return Err(MolRsError::validation(format!(
                        "source bead {bead:?} of unit {u} has no bead_type"
                    )));
                };
                if !licensed.contains(&(bead_type, label.as_str())) {
                    return Err(MolRsError::validation(format!(
                        "source bead {bead:?} of unit {u} has type '{bead_type}', which no rule \
                         licenses for its template label '{label}'"
                    )));
                }
                let Some(point) = atom.position() else {
                    return Err(MolRsError::validation(format!(
                        "source bead {bead:?} of unit {u} lacks x/y/z coordinates"
                    )));
                };
                points.push(point);
            }
            offsets.push(points.len());
        }
        Trace::ragged(points, offsets)
    }
}

#[cfg(test)]
mod tests {
    use super::Mapping;
    use crate::error::MolRsError;
    use crate::system::coarsegrain::CoarseGrain;
    use crate::system::frag_graph::FragGraph;
    use crate::system::molgraph::{MolGraph, NodeId};

    /// Four distinct live node ids; `Mapping` stores them opaquely.
    fn ids() -> [NodeId; 4] {
        let mut g = MolGraph::new();
        [g.add_node(), g.add_node(), g.add_node(), g.add_node()]
    }

    fn two_unit_graph() -> FragGraph {
        FragGraph::path(vec!["PMA".to_owned(), "PMA".to_owned()], (0, 1))
            .expect("a 2-node path is valid")
    }

    fn ba() -> Vec<String> {
        vec!["B".to_owned(), "A".to_owned()]
    }

    fn rules() -> Vec<(String, String)> {
        vec![
            ("1".to_owned(), "B".to_owned()),
            ("4".to_owned(), "A".to_owned()),
        ]
    }

    #[test]
    fn new_keeps_graph_sources_labels_and_rules() {
        let [n0, n1, n2, n3] = ids();
        let m = Mapping::new(
            two_unit_graph(),
            vec![vec![n0, n1], vec![n2, n3]],
            vec![ba(), ba()],
            rules(),
        )
        .expect("a consistent mapping is accepted");

        assert_eq!(m.n_units(), 2);
        assert_eq!(m.graph().nodes(), two_unit_graph().nodes());
        assert_eq!(m.graph().edges(), two_unit_graph().edges());
        assert_eq!(m.sources(0), Some(&[n0, n1][..]));
        assert_eq!(m.sources(1), Some(&[n2, n3][..]));
        assert_eq!(m.sources(2), None);
        assert_eq!(m.labels(0), Some(ba().as_slice()));
        assert_eq!(m.labels(1), Some(ba().as_slice()));
        assert_eq!(m.labels(2), None);
        assert_eq!(m.rules(), rules().as_slice());
    }

    #[test]
    fn new_refuses_fewer_source_lists_than_nodes() {
        let [n0, n1, _, _] = ids();
        let err = Mapping::new(two_unit_graph(), vec![vec![n0, n1]], vec![ba()], rules())
            .expect_err("two nodes need two source lists");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn new_refuses_labels_of_a_different_length_than_sources() {
        let [n0, n1, n2, n3] = ids();
        let err = Mapping::new(
            two_unit_graph(),
            vec![vec![n0, n1], vec![n2, n3]],
            vec![ba(), vec!["B".to_owned()]],
            rules(),
        )
        .expect_err("labels[1] has one entry for two source beads");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn new_refuses_one_source_id_in_two_units() {
        let [n0, n1, n2, _] = ids();
        let err = Mapping::new(
            two_unit_graph(),
            vec![vec![n0, n1], vec![n2, n1]],
            vec![ba(), ba()],
            rules(),
        )
        .expect_err("bead n1 cannot belong to two units");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn new_refuses_a_label_in_no_rule() {
        let [n0, n1, n2, n3] = ids();
        let err = Mapping::new(
            two_unit_graph(),
            vec![vec![n0, n1], vec![n2, n3]],
            vec![ba(), vec!["B".to_owned(), "Z".to_owned()]],
            rules(),
        )
        .expect_err("label 'Z' is licensed by no rule");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn new_refuses_a_unit_with_no_sources() {
        // Unit 1's source and label lists are both empty, so the shapes
        // agree; such a unit can be neither traced nor placed.
        let [n0, n1, _, _] = ids();
        let err = Mapping::new(
            two_unit_graph(),
            vec![vec![n0, n1], Vec::new()],
            vec![ba(), Vec::new()],
            rules(),
        )
        .expect_err("unit 1 has no source bead");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    /// Rows A "4" (0,0,0), B "1" (1,0,0), A "4" (2,0,0), B "1" (3,0,0);
    /// bonds 0-1, 2-3.
    fn four_bead_cg() -> (CoarseGrain, [NodeId; 4]) {
        let mut cg = CoarseGrain::new();
        let n0 = cg.add_bead("4", 0.0, 0.0, 0.0);
        let n1 = cg.add_bead("1", 1.0, 0.0, 0.0);
        let n2 = cg.add_bead("4", 2.0, 0.0, 0.0);
        let n3 = cg.add_bead("1", 3.0, 0.0, 0.0);
        cg.add_bond(n0, n1).expect("n0-n1 bond");
        cg.add_bond(n2, n3).expect("n2-n3 bond");
        (cg, [n0, n1, n2, n3])
    }

    /// Two "U" units; each lists its beads in template order B then A.
    fn b_then_a_mapping(ids: [NodeId; 4]) -> Mapping {
        let [n0, n1, n2, n3] = ids;
        let graph = FragGraph::path(vec!["U".to_owned(), "U".to_owned()], (0, 1))
            .expect("a 2-node path is valid");
        Mapping::new(
            graph,
            vec![vec![n1, n0], vec![n3, n2]],
            vec![ba(), ba()],
            rules(),
        )
        .expect("a consistent mapping is accepted")
    }

    #[test]
    fn trace_gives_source_positions_in_template_bead_order() {
        let (cg, ids) = four_bead_cg();
        let t = b_then_a_mapping(ids)
            .trace(&cg)
            .expect("every source bead is licensed and has coordinates");
        assert_eq!(t.n_units(), 2);
        assert_eq!(t.unit(0), Some(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]][..]));
        assert_eq!(t.unit(1), Some(&[[3.0, 0.0, 0.0], [2.0, 0.0, 0.0]][..]));
        assert_eq!(t.unit(2), None);
    }

    #[test]
    fn trace_refuses_a_bead_type_the_rules_do_not_license_for_its_label() {
        let (mut cg, ids) = four_bead_cg();
        let m = b_then_a_mapping(ids);
        let n0 = ids[0];
        // n0 is unit 0's template bead "A"; ("7", "A") is in no rule.
        cg.set_node(n0, "bead_type", "7").expect("retype n0");
        let err = m.trace(&cg).expect_err("('7', 'A') is not licensed");
        let MolRsError::Validation { message } = &err else {
            panic!("expected Validation, got {err:?}");
        };
        assert!(message.contains("unit 0"), "names the unit: {message}");
        assert!(
            message.contains(&format!("{n0:?}")),
            "names the bead: {message}"
        );
        assert!(message.contains("'7'"), "names the type: {message}");
        assert!(message.contains("'A'"), "names the label: {message}");
    }

    #[test]
    fn trace_refuses_a_bead_without_coordinates() {
        let mut cg = CoarseGrain::new();
        let n0 = cg.add_bead("4", 0.0, 0.0, 0.0);
        let n1 = cg.add_bead_bare("1");
        let n2 = cg.add_bead("4", 2.0, 0.0, 0.0);
        let n3 = cg.add_bead("1", 3.0, 0.0, 0.0);
        let m = b_then_a_mapping([n0, n1, n2, n3]);
        assert!(m.trace(&cg).is_err(), "n1 has no x/y/z");
    }
}
