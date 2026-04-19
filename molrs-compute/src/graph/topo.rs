//! Kahn topological sort + cycle detection for the Graph DAG.

use std::collections::{HashMap, VecDeque};

use super::node::NodeId;
use crate::error::ComputeError;

/// Return `nodes` in a topological order consistent with `deps`.
///
/// `deps[n]` lists the nodes that `n` depends on (its predecessors). A valid
/// order visits every predecessor of `n` before visiting `n` itself.
///
/// Ties are broken by ascending `NodeId` so the output is deterministic.
///
/// # Errors
///
/// Returns [`ComputeError::CyclicDependency`] if the graph has a cycle; the
/// reported node list contains every node that could not be sorted.
///
/// Currently unused — `Graph::run` relies on the structural invariant that
/// insertion order is a topological order (a `Slot` can only be captured
/// after its owning node has been registered). Kept as a defense-in-depth
/// utility for future parallel execution schemes.
#[allow(dead_code)]
pub(crate) fn topological_order(
    nodes: &[NodeId],
    deps: &HashMap<NodeId, Vec<NodeId>>,
) -> Result<Vec<NodeId>, ComputeError> {
    let mut successors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut in_deg: HashMap<NodeId, usize> = HashMap::with_capacity(nodes.len());
    for &n in nodes {
        in_deg.insert(n, 0);
        successors.entry(n).or_default();
    }
    for (&node, preds) in deps.iter() {
        for &pred in preds {
            successors.entry(pred).or_default().push(node);
            *in_deg
                .get_mut(&node)
                .expect("dep node missing from node set") += 1;
        }
    }

    let mut ready: Vec<NodeId> = in_deg
        .iter()
        .filter_map(|(n, d)| if *d == 0 { Some(*n) } else { None })
        .collect();
    ready.sort();
    let mut queue: VecDeque<NodeId> = ready.into();

    let mut order = Vec::with_capacity(nodes.len());
    while let Some(n) = queue.pop_front() {
        order.push(n);
        let mut succ = successors.remove(&n).unwrap_or_default();
        succ.sort();
        for s in succ {
            let d = in_deg.get_mut(&s).expect("successor missing from in_deg");
            *d -= 1;
            if *d == 0 {
                queue.push_back(s);
            }
        }
    }

    if order.len() != nodes.len() {
        let mut unresolved: Vec<NodeId> = in_deg
            .into_iter()
            .filter_map(|(n, d)| if d > 0 { Some(n) } else { None })
            .collect();
        unresolved.sort();
        return Err(ComputeError::CyclicDependency { nodes: unresolved });
    }

    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(n: u32) -> NodeId {
        NodeId::new(n)
    }

    #[test]
    fn empty_graph() {
        let order = topological_order(&[], &HashMap::new()).unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn single_node_no_deps() {
        let order = topological_order(&[node(0)], &HashMap::new()).unwrap();
        assert_eq!(order, vec![node(0)]);
    }

    #[test]
    fn linear_dependency() {
        // 0 -> 1 -> 2 (i.e. 1 depends on 0, 2 depends on 1)
        let mut deps: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        deps.insert(node(1), vec![node(0)]);
        deps.insert(node(2), vec![node(1)]);
        let order = topological_order(&[node(0), node(1), node(2)], &deps).unwrap();
        assert_eq!(order, vec![node(0), node(1), node(2)]);
    }

    #[test]
    fn diamond_dependency() {
        //     0
        //    / \
        //   1   2
        //    \ /
        //     3
        let mut deps: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        deps.insert(node(1), vec![node(0)]);
        deps.insert(node(2), vec![node(0)]);
        deps.insert(node(3), vec![node(1), node(2)]);
        let order = topological_order(&[node(0), node(1), node(2), node(3)], &deps).unwrap();
        assert_eq!(order[0], node(0));
        assert_eq!(order[3], node(3));
        // 1 and 2 can be in either order between 0 and 3; deterministic by id:
        assert_eq!(order, vec![node(0), node(1), node(2), node(3)]);
    }

    #[test]
    fn direct_cycle_detected() {
        // 0 -> 1 -> 0
        let mut deps: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        deps.insert(node(0), vec![node(1)]);
        deps.insert(node(1), vec![node(0)]);
        let err = topological_order(&[node(0), node(1)], &deps).unwrap_err();
        match err {
            ComputeError::CyclicDependency { nodes } => {
                assert_eq!(nodes, vec![node(0), node(1)]);
            }
            other => panic!("expected CyclicDependency, got {other:?}"),
        }
    }

    #[test]
    fn self_loop_detected() {
        // 0 -> 0
        let mut deps: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        deps.insert(node(0), vec![node(0)]);
        let err = topological_order(&[node(0)], &deps).unwrap_err();
        assert!(matches!(err, ComputeError::CyclicDependency { .. }));
    }

    #[test]
    fn cycle_among_subset_reports_only_unresolved() {
        // 0 (independent), 1 <-> 2
        let mut deps: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        deps.insert(node(1), vec![node(2)]);
        deps.insert(node(2), vec![node(1)]);
        let err = topological_order(&[node(0), node(1), node(2)], &deps).unwrap_err();
        match err {
            ComputeError::CyclicDependency { nodes } => {
                assert_eq!(nodes, vec![node(1), node(2)]);
            }
            other => panic!("expected CyclicDependency, got {other:?}"),
        }
    }
}
