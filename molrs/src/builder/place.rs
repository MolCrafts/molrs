//! Where a fragment sits, and which way it faces, are different motions.
//!
//! [`Placer`] decides where fragments go; [`Orienter`] decides which way each
//! one faces. Neither does the other's job, and the variant is the class: a
//! facing rule is a [`LineOrienter`] or a [`TangOrienter`], a placer is a
//! [`TracePlacer`], never a flag on one type.
//!
//! The unit of placement is the **fragment** — a group of nodes a
//! [`TracePlacer`] reads off a field. Nothing here is named after the repeat
//! unit: what a topology calls its pieces is the topology's business.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::spatial::Trace;
use crate::spatial::geometry::{self, normalize, perpendicular};
use crate::store::keys;
use crate::system::element::Element;
use crate::system::molgraph::{MolGraph, NodeId};
use crate::types::F;

/// Put the fragments a set of forming bonds joins at a pose.
///
/// Implementations mutate `world`'s coordinates in place. `bonds` are the
/// node pairs the reaction is about to join, each `(parent-side atom,
/// child-side atom)`. A placer that cannot place every fragment the bonds
/// join returns an error and never leaves a partial placement behind.
pub trait Placer {
    /// Move whole fragments so each forming bond's endpoints sit at bonding
    /// range.
    ///
    /// # Errors
    ///
    /// For [`TracePlacer`], exactly:
    ///
    /// - [`PlaceError::MissingGroupField`] — no node carries the group field;
    /// - [`PlaceError::MissingNode`] — a bond endpoint is not in the graph;
    /// - [`PlaceError::Ungrouped`] — a bond endpoint carries no fragment id;
    /// - [`PlaceError::MissingElement`] / [`PlaceError::UnknownElement`] — a
    ///   cross-fragment bond endpoint has no element, or one with no
    ///   tabulated covalent radius;
    /// - [`PlaceError::MissingCoordinates`] — a node the placement reads (a
    ///   child's anchor or far site; for the straight default trace, also the
    ///   first bond's endpoints and every node of its parent) lacks x, y or z;
    /// - [`PlaceError::Unreachable`] — the bonds join fragments into more than
    ///   one connected piece, so some fragment has no path from the root;
    /// - [`PlaceError::Index`] — the trace has fewer samples than fragments to
    ///   place;
    /// - [`PlaceError::Unorientable`] — a child's site axis, or the trace
    ///   tangent where it lands, has no direction;
    /// - [`PlaceError::Graph`] — the graph refused a coordinate write.
    ///
    /// Every error except [`PlaceError::Graph`] is raised before `world` is
    /// written, so `world` is then untouched.
    fn place(&self, world: &mut MolGraph, bonds: &[(NodeId, NodeId)]) -> Result<(), PlaceError>;
}

/// One endpoint of a forming bond: where it is and how big it is.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Endpoint {
    node: NodeId,
    radius: F,
}

/// A forming bond between two different fragments.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Junction {
    from: i64,
    to: i64,
    from_end: Endpoint,
    to_end: Endpoint,
}

/// Lay the fragments of a topology along a trajectory.
///
/// Walks the fragment graph the forming bonds imply and, for each child
/// fragment on its own, lands its anchor — the atom the bond reaches — on the
/// next [`Trace`] sample, then turns the fragment's site axis to the trace
/// tangent through its [`Orienter`]. Only whole fragments move.
///
/// `trace` defaults to a straight line, so a path of fragments becomes a
/// straight chain: samples sit one fragment *advance* apart (the fragment's own
/// span plus one closing bond), starting one span behind the first parent's
/// reacting atom, and the line leaves that parent away from its own body.
pub struct TracePlacer {
    trace: Option<Trace>,
    orienter: Box<dyn Orienter>,
    buffer: F,
    group_key: String,
    site_key: String,
}

impl Default for TracePlacer {
    fn default() -> Self {
        Self::new()
    }
}

impl TracePlacer {
    /// A placer with a straight default trace and [`LineOrienter`] facing.
    pub fn new() -> Self {
        Self {
            trace: None,
            orienter: Box::new(LineOrienter),
            buffer: 0.0,
            group_key: keys::RES_ID.to_string(),
            site_key: keys::SITE.to_string(),
        }
    }

    /// Lay the fragments out on this trace instead of the straight default.
    pub fn with_trace(mut self, trace: Trace) -> Self {
        self.trace = Some(trace);
        self
    }

    /// Face each fragment by this rule instead of [`LineOrienter`].
    pub fn with_orienter(mut self, orienter: Box<dyn Orienter>) -> Self {
        self.orienter = orienter;
        self
    }

    /// Extra separation (Å) beyond the summed covalent radii.
    pub fn with_buffer(mut self, buffer: F) -> Self {
        self.buffer = buffer;
        self
    }

    /// The field that groups nodes into fragments (default [`keys::RES_ID`]).
    pub fn with_group_key(mut self, key: &str) -> Self {
        self.group_key = key.to_string();
        self
    }

    /// The field that marks a fragment's site atoms (default [`keys::SITE`]).
    pub fn with_site_key(mut self, key: &str) -> Self {
        self.site_key = key.to_string();
        self
    }

    fn position(world: &MolGraph, node: NodeId) -> Option<[F; 3]> {
        let atom = world.get_node(node).ok()?;
        Some([
            atom.get_f64(keys::X)?,
            atom.get_f64(keys::Y)?,
            atom.get_f64(keys::Z)?,
        ])
    }

    /// [`Self::position`], or [`PlaceError::MissingCoordinates`].
    fn located(world: &MolGraph, node: NodeId) -> Result<[F; 3], PlaceError> {
        Self::position(world, node).ok_or(PlaceError::MissingCoordinates { node })
    }

    fn radius(world: &MolGraph, node: NodeId) -> Result<F, PlaceError> {
        let symbol = world
            .get_node(node)
            .map_err(|_| PlaceError::MissingNode { node })?
            .get_str(keys::ELEMENT)
            .ok_or(PlaceError::MissingElement { node })?
            .to_string();
        let element = Element::by_symbol(&symbol).ok_or_else(|| PlaceError::UnknownElement {
            symbol: symbol.clone(),
        })?;
        Ok(element.covalent_radius() as F)
    }

    /// Node ids grouped by the fragment field, ordered by fragment id.
    fn fragments(&self, world: &MolGraph) -> Result<BTreeMap<i64, Vec<NodeId>>, PlaceError> {
        let mut groups: BTreeMap<i64, Vec<NodeId>> = BTreeMap::new();
        let mut seen_any = false;
        for node in world.node_ids() {
            let Some(group) = world
                .get_node(node)
                .ok()
                .and_then(|a| a.get_int(&self.group_key))
            else {
                continue;
            };
            seen_any = true;
            groups.entry(group as i64).or_default().push(node);
        }
        if !seen_any {
            return Err(PlaceError::MissingGroupField {
                key: self.group_key.clone(),
            });
        }
        Ok(groups)
    }

    /// ``(from, to, endpoints)`` per bond that joins two different fragments.
    fn junctions(
        &self,
        world: &MolGraph,
        bonds: &[(NodeId, NodeId)],
        groups: &BTreeMap<i64, Vec<NodeId>>,
    ) -> Result<Vec<Junction>, PlaceError> {
        let owner: HashMap<NodeId, i64> = groups
            .iter()
            .flat_map(|(&group, nodes)| nodes.iter().map(move |&node| (node, group)))
            .collect();
        let mut junctions = Vec::new();
        let fragment_of = |node: NodeId| -> Result<i64, PlaceError> {
            if let Some(&group) = owner.get(&node) {
                return Ok(group);
            }
            match world.get_node(node) {
                Ok(_) => Err(PlaceError::Ungrouped { node }),
                Err(_) => Err(PlaceError::MissingNode { node }),
            }
        };
        for &(a, b) in bonds {
            let (from, to) = (fragment_of(a)?, fragment_of(b)?);
            if from == to {
                continue;
            }
            junctions.push(Junction {
                from,
                to,
                from_end: Endpoint {
                    node: a,
                    radius: Self::radius(world, a)?,
                },
                to_end: Endpoint {
                    node: b,
                    radius: Self::radius(world, b)?,
                },
            });
        }
        Ok(junctions)
    }

    /// The fragment's other site: the first site-labelled node of `nodes`
    /// that is not `anchor`. `None` for a fragment with no second site.
    fn far_site(&self, world: &MolGraph, nodes: &[NodeId], anchor: NodeId) -> Option<NodeId> {
        nodes.iter().copied().find(|&node| {
            node != anchor
                && world
                    .get_node(node)
                    .ok()
                    .is_some_and(|a| a.get_str(&self.site_key).is_some_and(|l| !l.is_empty()))
        })
    }

    /// A straight line leaving `first`'s parent away from its own body, with
    /// `samples` points one fragment advance apart.
    fn straight_trace(
        &self,
        world: &MolGraph,
        first: &Junction,
        parent: &[NodeId],
        child: &[NodeId],
        samples: usize,
    ) -> Result<Trace, PlaceError> {
        let closing = first.from_end.radius + first.to_end.radius + self.buffer;
        let reacting = Self::located(world, first.from_end.node)?;
        let target = Self::located(world, first.to_end.node)?;
        let span_length = match self.far_site(world, child, first.to_end.node) {
            Some(far) => {
                let far = Self::located(world, far)?;
                let span = [far[0] - target[0], far[1] - target[1], far[2] - target[2]];
                (span[0] * span[0] + span[1] * span[1] + span[2] * span[2]).sqrt()
            }
            None => 0.0,
        };
        let advance = closing + span_length;

        // Grow away from the parent's body, not along an arbitrary axis: the
        // parent keeps every atom the reaction did not consume, so a direction
        // read off overlapping templates would bury the child in them.
        let mut centroid = [0.0; 3];
        for &node in parent {
            let position = Self::located(world, node)?;
            for axis in 0..3 {
                centroid[axis] += position[axis];
            }
        }
        let count = parent.len() as F;
        let direction = normalize([
            reacting[0] - centroid[0] / count,
            reacting[1] - centroid[1] / count,
            reacting[2] - centroid[2] / count,
        ])
        .unwrap_or([1.0, 0.0, 0.0]);

        let origin = [
            reacting[0] - direction[0] * span_length,
            reacting[1] - direction[1] * span_length,
            reacting[2] - direction[2] * span_length,
        ];
        Ok(Trace::from_arrays(
            (0..samples)
                .map(|k| {
                    let distance = advance * k as F;
                    [
                        origin[0] + direction[0] * distance,
                        origin[1] + direction[1] * distance,
                        origin[2] + direction[2] * distance,
                    ]
                })
                .collect(),
        ))
    }

    /// Breadth-first placement steps over the fragment graph the junctions
    /// span, from its lowest fragment id. A fragment no forming bond reaches
    /// is not part of the placement and is left alone.
    ///
    /// # Errors
    ///
    /// [`PlaceError::Unreachable`] naming the lowest fragment the junctions
    /// touch that has no path from the root.
    fn steps(junctions: &[Junction]) -> Result<Vec<(Junction, bool)>, PlaceError> {
        let mut adjacency: HashMap<i64, Vec<(Junction, bool)>> = HashMap::new();
        let mut joined = BTreeSet::new();
        for junction in junctions {
            adjacency
                .entry(junction.from)
                .or_default()
                .push((*junction, false));
            adjacency
                .entry(junction.to)
                .or_default()
                .push((*junction, true));
            joined.extend([junction.from, junction.to]);
        }
        let Some(&root) = joined.first() else {
            return Ok(Vec::new());
        };
        let mut seen = HashSet::from([root]);
        let mut queue = VecDeque::from([root]);
        let mut steps = Vec::new();
        while let Some(parent) = queue.pop_front() {
            for &(junction, reversed) in adjacency.get(&parent).into_iter().flatten() {
                let child = if reversed { junction.from } else { junction.to };
                if !seen.insert(child) {
                    continue;
                }
                queue.push_back(child);
                steps.push((junction, reversed));
            }
        }
        match joined.into_iter().find(|fragment| !seen.contains(fragment)) {
            Some(fragment) => Err(PlaceError::Unreachable { fragment }),
            None => Ok(steps),
        }
    }
}

impl Placer for TracePlacer {
    fn place(&self, world: &mut MolGraph, bonds: &[(NodeId, NodeId)]) -> Result<(), PlaceError> {
        let groups = self.fragments(world)?;
        let junctions = self.junctions(world, bonds, &groups)?;
        let Some(first) = junctions.first() else {
            return Ok(());
        };
        let steps = Self::steps(&junctions)?;
        // Every junction endpoint's fragment is a key of `groups`: `junctions`
        // read each fragment id off `groups` itself.
        let trace = match &self.trace {
            Some(trace) => trace.clone(),
            None => self.straight_trace(
                world,
                first,
                &groups[&first.from],
                &groups[&first.to],
                junctions.len() + 2,
            )?,
        };

        // `steps` visits every child fragment exactly once and the root is
        // never a child, so each fragment's new coordinates are computed from
        // its own untouched nodes. They are all written only once every
        // fragment is placed, so an error leaves `world` untouched.
        let mut placed: Vec<(NodeId, [F; 3])> = Vec::new();
        for (index, (junction, reversed)) in steps.iter().enumerate() {
            let index = index + 1;
            let (child, child_end) = if *reversed {
                (junction.from, junction.from_end)
            } else {
                (junction.to, junction.to_end)
            };
            let sample = trace.point(index).ok_or(PlaceError::Index { index })?;
            let nodes = &groups[&child];
            let anchor = Self::located(world, child_end.node)?;

            // Rigid motion over just this fragment's nodes. A node without a
            // full x, y, z is left where it is.
            let delta = [
                sample[0] - anchor[0],
                sample[1] - anchor[1],
                sample[2] - anchor[2],
            ];
            let mut moved: Vec<(NodeId, [F; 3])> = nodes
                .iter()
                .filter_map(|&node| {
                    let p = Self::position(world, node)?;
                    Some((node, [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]]))
                })
                .collect();

            // Facing is measured in the frame the fragment is now in.
            let anchor = [
                anchor[0] + delta[0],
                anchor[1] + delta[1],
                anchor[2] + delta[2],
            ];
            if let Some(far) = self.far_site(world, nodes, child_end.node) {
                let far = Self::located(world, far)?;
                let far = [far[0] + delta[0], far[1] + delta[1], far[2] + delta[2]];
                let body_axis = [far[0] - anchor[0], far[1] - anchor[1], far[2] - anchor[2]];
                let unorientable = PlaceError::Unorientable {
                    fragment: Some(child),
                };
                let from_dir = self
                    .orienter
                    .direction(body_axis)
                    .ok_or_else(|| unorientable.clone())?;
                let to_dir = trace.tangent(index).ok_or(unorientable)?;
                if let Some((axis, angle)) = geometry::alignment(from_dir, to_dir) {
                    let (sin_a, cos_a) = angle.sin_cos();
                    for (_, position) in &mut moved {
                        *position = geometry::rotate_point(*position, axis, cos_a, sin_a, anchor);
                    }
                }
            }

            placed.append(&mut moved);
        }
        for (node, position) in placed {
            for (value, key) in position.into_iter().zip([keys::X, keys::Y, keys::Z]) {
                world
                    .set_node(node, key, value)
                    .map_err(|error| PlaceError::Graph(error.to_string()))?;
            }
        }
        Ok(())
    }
}

/// Turn a fragment about an anchor. No translation.
///
/// A subclass answers one question — which direction its rule reads out of the
/// fragment's own site axis — and inherits the rigid motion that applies it.
pub trait Orienter: Send + Sync {
    /// The direction this rule takes from `body_axis`, or `None` when the axis
    /// is degenerate.
    fn direction(&self, body_axis: [F; 3]) -> Option<[F; 3]>;

    /// Rotate `mol` about `anchor` so the rule's direction points along
    /// `to_dir`. `flip` sends it along `-to_dir` instead.
    ///
    /// # Errors
    ///
    /// [`PlaceError::Unorientable`] when `body_axis` has no direction under
    /// this rule; `mol` is then left untouched.
    fn orient(
        &self,
        mol: &mut MolGraph,
        anchor: [F; 3],
        body_axis: [F; 3],
        to_dir: [F; 3],
        flip: bool,
    ) -> Result<(), PlaceError> {
        let from_dir = self
            .direction(body_axis)
            .ok_or(PlaceError::Unorientable { fragment: None })?;
        geometry::orient(mol, anchor, from_dir, to_dir, flip);
        Ok(())
    }
}

/// The site axis itself: the fragment's outgoing site points along the trace,
/// so a chain of them runs straight.
pub struct LineOrienter;

impl Orienter for LineOrienter {
    fn direction(&self, body_axis: [F; 3]) -> Option<[F; 3]> {
        normalize(body_axis)
    }
}

/// A perpendicular of the site axis: the fragment meets the trace at an angle,
/// so the chain kinks by that angle at every unit.
pub struct TangOrienter;

impl Orienter for TangOrienter {
    fn direction(&self, body_axis: [F; 3]) -> Option<[F; 3]> {
        perpendicular(body_axis)
    }
}

/// A [`Placer`] could not place the fragments it was given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaceError {
    /// `index` is past the end of the trace.
    Index { index: usize },
    /// No node carries the field that groups nodes into fragments.
    MissingGroupField { key: String },
    /// A forming bond reached a node the graph does not hold.
    MissingNode { node: NodeId },
    /// A forming bond reached a node that carries no fragment id, so no
    /// fragment is known to move.
    Ungrouped { node: NodeId },
    /// A forming bond reached a node with no element, so no radius is known.
    MissingElement { node: NodeId },
    /// A forming bond reached an element with no tabulated radius.
    UnknownElement { symbol: String },
    /// A node the placement reads lacks a full x, y, z.
    MissingCoordinates { node: NodeId },
    /// The forming bonds join fragments into more than one connected piece;
    /// `fragment` (the lowest such id) has no path from the root fragment.
    Unreachable { fragment: i64 },
    /// A fragment cannot be faced: its site axis has no direction under the
    /// orienter (zero, shorter than 1e-6 Å, non-finite, or so long its squared
    /// length overflows), or the trace has no tangent where it lands.
    /// `fragment` is the fragment id when a placer was placing one.
    Unorientable { fragment: Option<i64> },
    /// The graph refused an edit (a stale handle, a type conflict).
    Graph(String),
}

impl std::fmt::Display for PlaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Index { index } => write!(f, "trace sample {index} does not exist"),
            Self::MissingGroupField { key } => {
                write!(
                    f,
                    "no node carries '{key}'; nothing groups nodes into fragments"
                )
            }
            Self::MissingNode { node } => write!(f, "graph holds no node {node:?}"),
            Self::Ungrouped { node } => {
                write!(f, "bond endpoint {node:?} carries no fragment id")
            }
            Self::MissingElement { node } => {
                write!(f, "node {node:?} carries no element, so no radius is known")
            }
            Self::UnknownElement { symbol } => {
                write!(f, "no covalent radius tabulated for element '{symbol}'")
            }
            Self::MissingCoordinates { node } => {
                write!(f, "node {node:?} lacks a full x, y, z to place by")
            }
            Self::Unreachable { fragment } => write!(
                f,
                "fragment {fragment} has no path of forming bonds from the root fragment"
            ),
            Self::Unorientable {
                fragment: Some(fragment),
            } => write!(
                f,
                "fragment {fragment} cannot be faced: its site axis or the trace tangent has no direction"
            ),
            Self::Unorientable { fragment: None } => {
                write!(f, "the site axis has no direction to face")
            }
            Self::Graph(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for PlaceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PropValue;
    use crate::system::atomistic::Atomistic;
    use crate::types::I;

    /// Two fragments, `O(a)-C-C` and `C-C-O(b)`, ids 1 and 2. The child's site
    /// axis is deliberately **not** along the joining direction, so a placer
    /// that never oriented anything would fail the axis tests below. Returns
    /// the world and its nodes as `[o1, c1, c2, c3, c4, o2]`.
    fn world() -> (Atomistic, Vec<NodeId>) {
        let mut mol = Atomistic::new();
        let o1 = mol.add_atom_xyz("O", 0.0, 0.0, 0.0);
        let c1 = mol.add_atom_xyz("C", 1.4, 0.0, 0.0);
        let c2 = mol.add_atom_xyz("C", 2.9, 0.0, 0.0);
        let c3 = mol.add_atom_xyz("C", 4.4, 0.0, 0.0);
        let c4 = mol.add_atom_xyz("C", 5.2, 1.2, 0.0);
        let o2 = mol.add_atom_xyz("O", 6.0, 2.4, 0.0);
        for (a, b) in [(o1, c1), (c1, c2), (c3, c4), (c4, o2)] {
            mol.add_bond(a, b)
                .expect("a fresh bond between distinct atoms");
        }
        for node in [o1, c1, c2] {
            mol.set_atom(node, keys::RES_ID, PropValue::Int(1 as I))
                .unwrap();
        }
        for node in [c3, c4, o2] {
            mol.set_atom(node, keys::RES_ID, PropValue::Int(2 as I))
                .unwrap();
        }
        mol.set_atom(o1, keys::SITE, PropValue::Str("a".to_string()))
            .unwrap();
        mol.set_atom(o2, keys::SITE, PropValue::Str("b".to_string()))
            .unwrap();
        (mol, vec![o1, c1, c2, c3, c4, o2])
    }

    fn position(mol: &Atomistic, node: NodeId) -> [F; 3] {
        TracePlacer::position(mol.as_molgraph(), node).expect("atom carries x, y, z")
    }

    fn distance(mol: &Atomistic, a: NodeId, b: NodeId) -> F {
        let (p, q) = (position(mol, a), position(mol, b));
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    }

    #[test]
    fn the_two_orienters_differ_by_the_axis_they_read() {
        let axis = [1.0, 2.0, 3.0];
        let line = LineOrienter.direction(axis).unwrap();
        let tangent = TangOrienter.direction(axis).unwrap();
        let unit = normalize(axis).unwrap();
        assert!((line[0] - unit[0]).abs() < 1e-12);
        let dot = line[0] * tangent[0] + line[1] * tangent[1] + line[2] * tangent[2];
        assert!(dot.abs() < 1e-12, "perpendicular dot {dot}");
    }

    #[test]
    fn a_degenerate_axis_has_no_direction() {
        assert!(LineOrienter.direction([0.0, 0.0, 0.0]).is_none());
        assert!(TangOrienter.direction([0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn a_placer_puts_the_joining_atoms_at_bonding_range() {
        let (mut mol, nodes) = world();
        let (o1, c3, c4, o2) = (nodes[0], nodes[3], nodes[4], nodes[5]);
        let before = distance(&mol, c3, o2);
        TracePlacer::new()
            .place(mol.as_molgraph_mut(), &[(o1, c3)])
            .unwrap();

        let expected = Element::by_symbol("O").unwrap().covalent_radius() as F
            + Element::by_symbol("C").unwrap().covalent_radius() as F;
        assert!(
            (distance(&mol, o1, c3) - expected).abs() < 1e-6,
            "joining atoms sit {} apart, expected {expected}",
            distance(&mol, o1, c3)
        );
        // The child moved as a rigid body, so its own span is unchanged.
        assert!((distance(&mol, c3, o2) - before).abs() < 1e-9);
        assert!(distance(&mol, c3, c4) > 0.0);
    }

    /// Unit cross product magnitude: 0 when the two vectors are parallel.
    fn parallelism(a: [F; 3], b: [F; 3]) -> F {
        let cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let magnitude = (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        let scale = (a[0].powi(2) + a[1].powi(2) + a[2].powi(2)).sqrt()
            * (b[0].powi(2) + b[1].powi(2) + b[2].powi(2)).sqrt();
        magnitude / scale
    }

    #[test]
    fn the_line_orienter_turns_the_child_axis_along_the_trace() {
        let (mut mol, nodes) = world();
        let (o1, c3, o2) = (nodes[0], nodes[3], nodes[5]);
        TracePlacer::new()
            .place(mol.as_molgraph_mut(), &[(o1, c3)])
            .unwrap();
        // The trace grows away from the parent's own body, so the child's site
        // axis ends up on that same line — and it did not start there.
        let (p, q, r) = (position(&mol, o1), position(&mol, c3), position(&mol, o2));
        let trace_direction = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let child_axis = [r[0] - q[0], r[1] - q[1], r[2] - q[2]];
        assert!(
            parallelism(trace_direction, child_axis) < 1e-6,
            "line facing left the child axis off the trace"
        );
    }

    #[test]
    fn the_tang_orienter_turns_it_across_the_trace() {
        let (mut mol, nodes) = world();
        let (o1, c3, o2) = (nodes[0], nodes[3], nodes[5]);
        TracePlacer::new()
            .with_orienter(Box::new(TangOrienter))
            .place(mol.as_molgraph_mut(), &[(o1, c3)])
            .unwrap();
        let (p, q, r) = (position(&mol, o1), position(&mol, c3), position(&mol, o2));
        let trace_direction = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let child_axis = [r[0] - q[0], r[1] - q[1], r[2] - q[2]];
        // Perpendicular: the cross product magnitude is the whole product.
        assert!(
            (parallelism(trace_direction, child_axis) - 1.0).abs() < 1e-6,
            "tangent facing did not turn the child across the trace"
        );
    }

    #[test]
    fn nothing_moves_without_a_cross_fragment_bond() {
        let (mut mol, nodes) = world();
        let before: Vec<[F; 3]> = mol
            .as_molgraph()
            .node_ids()
            .map(|node| position(&mol, node))
            .collect();
        let within = vec![(nodes[0], nodes[1]), (nodes[3], nodes[4])];
        TracePlacer::new()
            .place(mol.as_molgraph_mut(), &within)
            .unwrap();
        let after: Vec<[F; 3]> = mol
            .as_molgraph()
            .node_ids()
            .map(|node| position(&mol, node))
            .collect();
        assert_eq!(before, after);
    }

    /// [`world`] with the child's far site `o2` pushed to `x = 1e200`, so the
    /// child's site axis is finite but its squared length overflows.
    fn overflowing_world() -> (Atomistic, Vec<NodeId>) {
        let (mut mol, nodes) = world();
        mol.set_atom(nodes[5], keys::X, PropValue::F64(1e200))
            .unwrap();
        (mol, nodes)
    }

    /// A short explicit trace, so the straight default (which reads the same
    /// overflowing span) is not what a test exercises.
    fn short_trace() -> Trace {
        Trace::from_arrays(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    }

    #[test]
    fn the_line_orienter_has_no_direction_for_an_overflowing_axis() {
        // |axis|^2 overflows to inf; dividing by it gives [0, 0, 0], which is
        // not a direction.
        assert_eq!(LineOrienter.direction([1e200, 0.0, 0.0]), None);
    }

    #[test]
    fn the_tang_orienter_has_no_direction_for_an_overflowing_axis() {
        // Must answer `None`, not panic in `perpendicular`.
        assert_eq!(TangOrienter.direction([1e200, 0.0, 0.0]), None);
    }

    #[test]
    fn a_non_finite_axis_has_no_direction() {
        for axis in [
            [F::INFINITY, 0.0, 0.0],
            [0.0, F::NEG_INFINITY, 0.0],
            [F::NAN, 0.0, 0.0],
        ] {
            assert_eq!(LineOrienter.direction(axis), None, "line, axis {axis:?}");
            assert_eq!(TangOrienter.direction(axis), None, "tang, axis {axis:?}");
        }
    }

    #[test]
    fn line_placing_a_fragment_with_an_overflowing_site_axis_is_an_error() {
        let (mut mol, nodes) = overflowing_world();
        let result = TracePlacer::new()
            .with_trace(short_trace())
            .place(mol.as_molgraph_mut(), &[(nodes[0], nodes[3])]);
        assert!(
            result.is_err(),
            "an unorientable fragment was placed silently: {result:?}"
        );
    }

    #[test]
    fn tang_placing_a_fragment_with_an_overflowing_site_axis_is_an_error() {
        let (mut mol, nodes) = overflowing_world();
        let result = TracePlacer::new()
            .with_trace(short_trace())
            .with_orienter(Box::new(TangOrienter))
            .place(mol.as_molgraph_mut(), &[(nodes[0], nodes[3])]);
        assert!(
            result.is_err(),
            "an unorientable fragment was placed silently: {result:?}"
        );
    }

    #[test]
    fn extra_relation_kinds_do_not_change_where_fragments_land() {
        let (mut bare, nodes) = world();
        let (mut typed, typed_nodes) = world();
        assert_eq!(nodes, typed_nodes, "both worlds are built identically");
        let (angles, _, _) = typed.generate_topology(true, true, true, false).unwrap();
        assert!(angles > 0, "the guard needs relations beyond bonds");

        for mol in [&mut bare, &mut typed] {
            TracePlacer::new()
                .place(mol.as_molgraph_mut(), &[(nodes[0], nodes[3])])
                .unwrap();
        }
        for &node in &nodes {
            assert_eq!(
                position(&bare, node),
                position(&typed, node),
                "node {node:?}"
            );
        }
    }

    #[test]
    fn a_world_without_fragment_ids_is_refused() {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_xyz("O", 0.0, 0.0, 0.0);
        let b = mol.add_atom_xyz("C", 1.4, 0.0, 0.0);
        let error = TracePlacer::new()
            .place(mol.as_molgraph_mut(), &[(a, b)])
            .expect_err("no res_id anywhere");
        assert!(matches!(error, PlaceError::MissingGroupField { .. }));
    }

    /// `n` fragments with ids `1..=n`, each `C(a)-C-O(b)`: the head carbon is
    /// the anchor a bond from the previous fragment reaches, the tail oxygen
    /// the far site. Fragment `k` is laid out at `x = 4k`, with a bent middle
    /// so no site axis is degenerate. Returns `[head, middle, tail]` per
    /// fragment, in fragment-id order.
    fn chain(n: usize) -> (Atomistic, Vec<[NodeId; 3]>) {
        let mut mol = Atomistic::new();
        let mut fragments = Vec::with_capacity(n);
        for k in 1..=n {
            let x0 = 4.0 * k as F;
            let head = mol.add_atom_xyz("C", x0, 0.0, 0.0);
            let middle = mol.add_atom_xyz("C", x0 + 1.2, 0.9, 0.0);
            let tail = mol.add_atom_xyz("O", x0 + 2.4, 0.0, 0.0);
            for (a, b) in [(head, middle), (middle, tail)] {
                mol.add_bond(a, b)
                    .expect("a fresh bond between distinct atoms");
            }
            for node in [head, middle, tail] {
                mol.set_atom(node, keys::RES_ID, PropValue::Int(k as I))
                    .unwrap();
            }
            mol.set_atom(head, keys::SITE, PropValue::Str("a".to_string()))
                .unwrap();
            mol.set_atom(tail, keys::SITE, PropValue::Str("b".to_string()))
                .unwrap();
            fragments.push([head, middle, tail]);
        }
        (mol, fragments)
    }

    #[test]
    fn a_child_anchor_without_coordinates_is_refused() {
        let (mut mol, nodes) = world();
        let (o1, c3) = (nodes[0], nodes[3]);
        mol.clear_atom(c3, keys::X).unwrap();
        // An explicit trace, so the straight default does not read the anchor
        // first: this is the placement loop's own read.
        let error = TracePlacer::new()
            .with_trace(short_trace())
            .place(mol.as_molgraph_mut(), &[(o1, c3)])
            .expect_err("the child anchor has no x");
        assert_eq!(error, PlaceError::MissingCoordinates { node: c3 });
    }

    #[test]
    fn a_child_far_site_without_coordinates_is_refused() {
        let (mut mol, nodes) = world();
        let (o1, c3, o2) = (nodes[0], nodes[3], nodes[5]);
        mol.clear_atom(o2, keys::Y).unwrap();
        let error = TracePlacer::new()
            .with_trace(short_trace())
            .place(mol.as_molgraph_mut(), &[(o1, c3)])
            .expect_err("the child far site has no y");
        assert_eq!(error, PlaceError::MissingCoordinates { node: o2 });
    }

    #[test]
    fn the_straight_trace_refuses_a_first_bond_endpoint_without_coordinates() {
        let (mut mol, nodes) = world();
        let (o1, c3) = (nodes[0], nodes[3]);
        mol.clear_atom(o1, keys::Z).unwrap();
        let error = TracePlacer::new()
            .place(mol.as_molgraph_mut(), &[(o1, c3)])
            .expect_err("the parent's reacting atom has no z");
        assert_eq!(error, PlaceError::MissingCoordinates { node: o1 });
    }

    #[test]
    fn the_straight_trace_refuses_a_parent_node_without_coordinates() {
        let (mut mol, nodes) = world();
        let (o1, c1, c3) = (nodes[0], nodes[1], nodes[3]);
        // c1 is neither bond endpoint; only the parent-centroid read touches it.
        mol.clear_atom(c1, keys::X).unwrap();
        let error = TracePlacer::new()
            .place(mol.as_molgraph_mut(), &[(o1, c3)])
            .expect_err("a parent node has no x");
        assert_eq!(error, PlaceError::MissingCoordinates { node: c1 });
    }

    #[test]
    fn bonds_joining_two_separate_pieces_are_unreachable() {
        let (mut mol, fragments) = chain(4);
        let bonds = [
            (fragments[0][2], fragments[1][0]),
            (fragments[2][2], fragments[3][0]),
        ];
        let error = TracePlacer::new()
            .place(mol.as_molgraph_mut(), &bonds)
            .expect_err("fragments 3-4 have no path from root 1");
        assert_eq!(error, PlaceError::Unreachable { fragment: 3 });
    }

    #[test]
    fn a_bond_endpoint_without_a_fragment_id_is_ungrouped() {
        let (mut mol, nodes) = world();
        let stray = mol.add_atom_xyz("C", 9.0, 0.0, 0.0);
        let error = TracePlacer::new()
            .place(mol.as_molgraph_mut(), &[(nodes[0], stray)])
            .expect_err("the stray atom carries no res_id");
        assert_eq!(error, PlaceError::Ungrouped { node: stray });
    }

    #[test]
    fn a_bond_endpoint_the_graph_does_not_hold_is_missing() {
        let (mut mol, nodes) = world();
        let stale = mol.add_atom_xyz("C", 9.0, 0.0, 0.0);
        mol.set_atom(stale, keys::RES_ID, PropValue::Int(2 as I))
            .unwrap();
        mol.remove_atom(stale).unwrap();
        let error = TracePlacer::new()
            .place(mol.as_molgraph_mut(), &[(nodes[0], stale)])
            .expect_err("the endpoint was removed");
        assert_eq!(error, PlaceError::MissingNode { node: stale });
    }

    #[test]
    fn an_explicit_trace_shorter_than_the_fragments_is_an_index_error() {
        let (mut mol, fragments) = chain(3);
        let bonds = [
            (fragments[0][2], fragments[1][0]),
            (fragments[1][2], fragments[2][0]),
        ];
        // Samples 0 and 1 only: the second child needs sample 2.
        let trace = Trace::from_arrays(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let error = TracePlacer::new()
            .with_trace(trace)
            .place(mol.as_molgraph_mut(), &bonds)
            .expect_err("three fragments on a two-sample trace");
        assert_eq!(error, PlaceError::Index { index: 2 });
    }

    #[test]
    fn a_failing_later_child_leaves_an_earlier_child_unwritten() {
        let (mut mol, fragments) = chain(3);
        // The second child's site axis overflows, so it cannot be faced.
        mol.set_atom(fragments[2][2], keys::X, PropValue::F64(1e200))
            .unwrap();
        let bonds = [
            (fragments[0][2], fragments[1][0]),
            (fragments[1][2], fragments[2][0]),
        ];
        let trace = Trace::from_arrays(vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ]);
        let first_child = fragments[1];
        let before: Vec<[F; 3]> = first_child.iter().map(|&n| position(&mol, n)).collect();
        // Sample 1 is not where the first child's anchor already sits, so a
        // partial write would move it.
        assert_ne!(before[0], [2.0, 0.0, 0.0]);

        let error = TracePlacer::new()
            .with_trace(trace)
            .place(mol.as_molgraph_mut(), &bonds)
            .expect_err("the second child cannot be oriented");
        assert_eq!(error, PlaceError::Unorientable { fragment: Some(3) });
        let after: Vec<[F; 3]> = first_child.iter().map(|&n| position(&mol, n)).collect();
        assert_eq!(
            before, after,
            "the first child was written before the error"
        );
    }

    #[test]
    fn a_lower_fragment_no_bond_touches_does_not_block_placement() {
        let (mut mol, fragments) = chain(3);
        // Fragment 1 is untouched; the only forming bond joins 2 and 3.
        let (reacting, anchor) = (fragments[1][2], fragments[2][0]);
        TracePlacer::new()
            .place(mol.as_molgraph_mut(), &[(reacting, anchor)])
            .expect("fragments 2 and 3 form one connected piece");

        let expected = Element::by_symbol("O").unwrap().covalent_radius() as F
            + Element::by_symbol("C").unwrap().covalent_radius() as F;
        let got = distance(&mol, reacting, anchor);
        assert!(
            (got - expected).abs() < 1e-6,
            "joining atoms sit {got} apart, expected {expected}"
        );
    }

    #[test]
    fn a_fragment_no_bond_touches_is_left_alone() {
        let (mut mol, fragments) = chain(3);
        let before: Vec<[F; 3]> = fragments[0].iter().map(|&n| position(&mol, n)).collect();
        TracePlacer::new()
            .place(mol.as_molgraph_mut(), &[(fragments[1][2], fragments[2][0])])
            .unwrap();
        let after: Vec<[F; 3]> = fragments[0].iter().map(|&n| position(&mol, n)).collect();
        assert_eq!(before, after);
    }
}
