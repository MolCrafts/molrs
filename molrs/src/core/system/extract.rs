//! Induced subgraph and radius-ball extraction on [`MolGraph`].
//!
//! These are pure structure-fact kernels: which nodes lie within *R* hops of a
//! set of centers, and what is the induced edge set. Region / force-field policy
//! lives in consumers (e.g. molpy), not here.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::MolRsError;
use crate::system::molgraph::{KindId, MolGraph, NodeId};

/// Result of [`MolGraph::induced_subgraph`].
#[derive(Debug, Clone)]
pub struct InducedSubgraph {
    pub graph: MolGraph,
    /// Parent [`NodeId`] → node id in [`Self::graph`].
    pub node_map: HashMap<NodeId, NodeId>,
}

/// Result of [`MolGraph::extract_ball`].
///
/// Boundary / hops keys are **parent** handles (before remap). `node_map` maps
/// parent → new; `parent_of` is the inverse over new handles.
#[derive(Debug, Clone)]
pub struct ExtractedBall {
    pub graph: MolGraph,
    /// Selected parent nodes that have a `kind`-neighbor outside the ball.
    pub boundary: Vec<NodeId>,
    /// New node id → parent node id.
    pub parent_of: HashMap<NodeId, NodeId>,
    /// Parent node id → hops from the nearest center.
    pub hops: HashMap<NodeId, i64>,
    /// Parent node id → new node id.
    pub node_map: HashMap<NodeId, NodeId>,
}

impl MolGraph {
    /// Induced subgraph on an explicit node set.
    ///
    /// Every handle in `nodes` must be live (fail-fast). All registered relation
    /// kinds are copied when every endpoint lies in the set. Empty `nodes`
    /// yields an empty graph (kinds still mirrored from `self` so leaf
    /// re-wrapping stays consistent).
    pub fn induced_subgraph(&self, nodes: &[NodeId]) -> Result<InducedSubgraph, MolRsError> {
        for &n in nodes {
            if !self.node_table().contains(n) {
                return Err(MolRsError::not_found("node", format!("NodeId {n:?}")));
            }
        }
        // Dedup while preserving first-seen order for deterministic maps.
        let mut seen = HashSet::with_capacity(nodes.len());
        let mut ordered: Vec<NodeId> = Vec::with_capacity(nodes.len());
        for &n in nodes {
            if seen.insert(n) {
                ordered.push(n);
            }
        }
        Ok(self.materialize_induced(&ordered, /* only_kind */ None))
    }

    /// Multi-source BFS ball over a 2-ary relation `kind`, then materialise.
    ///
    /// - `radius < 0` → validation error.
    /// - `kind` must be registered and arity 2.
    /// - Stale centers fail-fast.
    /// - `copy_higher_order = false` copies only `kind` relations (O(ball×degree)
    ///   via the adjacency index). `true` also copies every other kind by
    ///   scanning its relation table (small-graph / verbatim path).
    pub fn extract_ball(
        &self,
        centers: &[NodeId],
        radius: i64,
        kind: KindId,
        copy_higher_order: bool,
    ) -> Result<ExtractedBall, MolRsError> {
        if radius < 0 {
            return Err(MolRsError::validation(format!(
                "extract_ball radius must be >= 0, got {radius}"
            )));
        }
        let kidx = kind.0 as usize;
        if kidx >= self.kind_ids().count() {
            return Err(MolRsError::not_found("kind", format!("KindId {kind:?}")));
        }
        if self.arity(kind) != 2 {
            return Err(MolRsError::validation(format!(
                "extract_ball requires arity-2 kind, '{}' has arity {}",
                self.kind_name(kind),
                self.arity(kind)
            )));
        }
        for &c in centers {
            if !self.node_table().contains(c) {
                return Err(MolRsError::not_found("node", format!("NodeId {c:?}")));
            }
        }

        if centers.is_empty() {
            let induced = self.materialize_induced(&[], only_kind(copy_higher_order, kind));
            return Ok(ExtractedBall {
                graph: induced.graph,
                boundary: Vec::new(),
                parent_of: HashMap::new(),
                hops: HashMap::new(),
                node_map: induced.node_map,
            });
        }

        // Multi-source BFS: hops[v] = min distance to any center.
        let mut hops: HashMap<NodeId, i64> = HashMap::new();
        let mut queue = VecDeque::new();
        for &c in centers {
            if hops.insert(c, 0).is_none() {
                queue.push_back(c);
            }
        }
        while let Some(cur) = queue.pop_front() {
            let d = hops[&cur];
            if d >= radius {
                continue;
            }
            for (k, _rid, other) in self.neighbor_relations(cur) {
                if k != kind {
                    continue;
                }
                let nd = d + 1;
                if nd > radius {
                    continue;
                }
                match hops.get(&other) {
                    Some(&old) if old <= nd => {}
                    _ => {
                        hops.insert(other, nd);
                        queue.push_back(other);
                    }
                }
            }
        }

        let selected: HashSet<NodeId> = hops.keys().copied().collect();
        let mut boundary: Vec<NodeId> = selected
            .iter()
            .copied()
            .filter(|&n| {
                self.neighbor_relations(n)
                    .any(|(k, _, other)| k == kind && !selected.contains(&other))
            })
            .collect();
        boundary.sort_by_key(|n| n_as_sort_key(*n));

        let mut ordered: Vec<NodeId> = selected.iter().copied().collect();
        ordered.sort_by_key(|n| n_as_sort_key(*n));

        let induced = self.materialize_induced(&ordered, only_kind(copy_higher_order, kind));
        let parent_of: HashMap<NodeId, NodeId> = induced
            .node_map
            .iter()
            .map(|(&parent, &new)| (new, parent))
            .collect();

        Ok(ExtractedBall {
            graph: induced.graph,
            boundary,
            parent_of,
            hops,
            node_map: induced.node_map,
        })
    }

    /// Build a new graph with mirrored kind registry, remapped nodes in
    /// `ordered`, and relations either of every kind or only `only_kind`.
    fn materialize_induced(
        &self,
        ordered: &[NodeId],
        only_kind: Option<KindId>,
    ) -> InducedSubgraph {
        let mut graph = MolGraph::new();
        // Mirror every kind so Atomistic/CoarseGrain re-wrap sees stable names.
        for kid in self.kind_ids() {
            graph.register_kind(self.kind_name(kid), self.arity(kid));
        }

        let mut node_map: HashMap<NodeId, NodeId> = HashMap::with_capacity(ordered.len());
        for &old in ordered {
            let payload = self.read_atom(old);
            let new_id = graph.add_node_with(payload);
            node_map.insert(old, new_id);
        }

        let selected: HashSet<NodeId> = ordered.iter().copied().collect();

        for kid in self.kind_ids() {
            if let Some(only) = only_kind
                && kid != only
            {
                continue;
            }
            let self_kind = graph
                .kind_id(self.kind_name(kid))
                .expect("kind mirrored at start of materialize_induced");
            for rid in self.relation_ids(kid) {
                let rel = self.read_relation(kid, rid);
                if !rel.nodes.iter().all(|n| selected.contains(n)) {
                    continue;
                }
                let mapped: smallvec::SmallVec<[NodeId; 4]> =
                    rel.nodes.iter().map(|n| node_map[n]).collect();
                if let Ok(new_rid) = graph.add_relation(self_kind, &mapped) {
                    graph.write_relation_props(self_kind, new_rid, &rel.props);
                }
            }
        }

        InducedSubgraph { graph, node_map }
    }
}

fn only_kind(copy_higher_order: bool, kind: KindId) -> Option<KindId> {
    if copy_higher_order { None } else { Some(kind) }
}

/// Stable sort key for NodeId without depending on slotmap Key internals in call sites.
fn n_as_sort_key(n: NodeId) -> u64 {
    use slotmap::Key;
    n.data().as_ffi()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::atomistic::Atomistic;
    use crate::system::coarsegrain::CoarseGrain;
    use crate::system::molgraph::Atom;

    fn linear_chain(n: usize) -> (Atomistic, Vec<crate::system::atomistic::AtomId>) {
        let mut mol = Atomistic::new();
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            ids.push(mol.add_atom_xyz("C", i as f64, 0.0, 0.0));
        }
        for w in ids.windows(2) {
            mol.add_bond(w[0], w[1]).unwrap();
        }
        (mol, ids)
    }

    #[test]
    fn induced_subgraph_path4_middle_two() {
        let (mut mol, ids) = linear_chain(4);
        mol.generate_topology(true, false, false).unwrap();
        assert!(mol.n_angles() >= 2);

        let sub = mol
            .as_molgraph()
            .induced_subgraph(&[ids[1], ids[2]])
            .unwrap();
        assert_eq!(sub.graph.n_nodes(), 2);
        assert_eq!(sub.node_map.len(), 2);
        let bond = sub.graph.kind_id("bonds").unwrap();
        assert_eq!(sub.graph.n_relations(bond), 1);
        let angle = sub.graph.kind_id("angles").unwrap();
        assert_eq!(
            sub.graph.n_relations(angle),
            0,
            "angles need 3 endpoints; incomplete ones must not copy"
        );
    }

    #[test]
    fn induced_subgraph_stale_fails() {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_bare("C");
        mol.remove_atom(a).unwrap();
        let err = mol.as_molgraph().induced_subgraph(&[a]);
        assert!(err.is_err());
    }

    #[test]
    fn extract_ball_linear_multi_center() {
        let (mol, ids) = linear_chain(10);
        // centers={2,7}, radius=1 → {1,2,3,6,7,8}
        let ball = mol
            .as_molgraph()
            .extract_ball(&[ids[2], ids[7]], 1, mol.bond_kind(), false)
            .unwrap();
        assert_eq!(ball.hops.len(), 6);
        assert_eq!(ball.hops[&ids[2]], 0);
        assert_eq!(ball.hops[&ids[1]], 1);
        assert_eq!(ball.hops[&ids[3]], 1);
        assert_eq!(ball.hops[&ids[7]], 0);
        let bset: HashSet<_> = ball.boundary.iter().copied().collect();
        for edge in [ids[1], ids[3], ids[6], ids[8]] {
            assert!(bset.contains(&edge), "missing boundary {edge:?}");
        }
        assert_eq!(ball.graph.n_nodes(), 6);
    }

    #[test]
    fn extract_ball_radius_zero_singleton() {
        let (mol, ids) = linear_chain(5);
        let ball = mol
            .as_molgraph()
            .extract_ball(&[ids[2]], 0, mol.bond_kind(), false)
            .unwrap();
        assert_eq!(ball.graph.n_nodes(), 1);
        assert_eq!(ball.hops[&ids[2]], 0);
        // center has neighbors outside → boundary
        assert_eq!(ball.boundary, vec![ids[2]]);
    }

    #[test]
    fn extract_ball_negative_radius_errors() {
        let (mol, ids) = linear_chain(2);
        let err = mol
            .as_molgraph()
            .extract_ball(&[ids[0]], -1, mol.bond_kind(), false);
        assert!(err.is_err());
    }

    #[test]
    fn extract_ball_stale_center_errors() {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_bare("C");
        mol.remove_atom(a).unwrap();
        let err = mol
            .as_molgraph()
            .extract_ball(&[a], 1, mol.bond_kind(), false);
        assert!(err.is_err());
    }

    #[test]
    fn atomistic_extract_regenerate_topology() {
        let (mol, ids) = linear_chain(5);
        // parent has no angles
        assert_eq!(mol.n_angles(), 0);
        let ext = mol
            .extract_subgraph(&[ids[2]], 2, /* regenerate_topology */ true)
            .unwrap();
        // ball = atoms 0..4 all five; has 2-edge paths → angles
        assert_eq!(ext.graph.n_atoms(), 5);
        assert!(
            ext.graph.n_angles() > 0,
            "regenerate must perceive angles from ball bonds alone"
        );
    }

    #[test]
    fn atomistic_extract_copy_higher_order_angles() {
        let (mut mol, ids) = linear_chain(4);
        mol.generate_topology(true, false, false).unwrap();
        let parent_angles = mol.n_angles();
        assert!(parent_angles > 0);
        let ext = mol
            .extract_subgraph(&[ids[1]], 2, /* regenerate_topology */ false)
            .unwrap();
        // full chain is in the ball of radius 2 from index 1
        assert_eq!(ext.graph.n_atoms(), 4);
        assert_eq!(ext.graph.n_angles(), parent_angles);
    }

    #[test]
    fn coarsegrain_extract_preserves_membership() {
        let mut cg = CoarseGrain::new();
        let b0 = cg.add_bead("A", 0.0, 0.0, 0.0);
        let b1 = cg.add_bead("B", 1.0, 0.0, 0.0);
        cg.add_bond(b0, b1).unwrap();
        cg.set_bead_members(b0, vec![10, 20]);
        let ext = cg.extract_subgraph(&[b0], 0).unwrap();
        assert_eq!(ext.graph.n_beads(), 1);
        let new_b = *ext.node_map.get(&b0).unwrap();
        assert_eq!(ext.graph.bead_members(new_b), &[10, 20]);
    }

    #[test]
    fn empty_centers_empty_ball() {
        let (mol, _) = linear_chain(3);
        let ball = mol
            .as_molgraph()
            .extract_ball(&[], 1, mol.bond_kind(), false)
            .unwrap();
        assert_eq!(ball.graph.n_nodes(), 0);
        assert!(ball.hops.is_empty());
    }

    #[test]
    fn empty_induced() {
        let g = MolGraph::new();
        let sub = g.induced_subgraph(&[]).unwrap();
        assert_eq!(sub.graph.n_nodes(), 0);
        assert!(sub.node_map.is_empty());
    }

    #[test]
    #[allow(unused_variables)]
    fn materialize_does_not_need_atom_payload_import() {
        let mut g = MolGraph::new();
        let n = g.add_node_with(Atom::new());
        let _ = g.induced_subgraph(&[n]).unwrap();
    }
}
