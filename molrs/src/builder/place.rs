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
use crate::system::molgraph::{MolGraph, NodeId, node_to_u64};
use crate::types::F;

/// Put the fragments a set of forming bonds joins at a pose.
///
/// Implementations mutate `world`'s coordinates in place. `bonds` are the
/// node pairs the reaction is about to join, each `(parent-side atom,
/// child-side atom)`. A placer that cannot place every fragment the bonds
/// join returns an error and never leaves a partial placement behind.
///
/// A placer knows only local information: it places each fragment relative to
/// its parent in the placement walk. It does not close rings. A ring's global
/// shape is the caller's to give — an explicit trace of the right size — or
/// geometry optimisation's to find afterwards.
pub trait Placer {
    /// Move whole fragments so every tree forming bond — one joining a
    /// fragment to its parent in the placement walk — ends at bonding range.
    ///
    /// A forming bond the walk does not cross closes a ring of fragments. It is
    /// neither placed nor checked: its length is whatever the tree leaves it,
    /// and closing it is the caller's concern (an explicit trace, or geometry
    /// optimisation afterwards). It is still a forming bond, so the reaction
    /// still forms it.
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
    /// - [`PlaceError::Unreachable`] — the bonds join fragments into more than
    ///   one connected piece, so some fragment has no path from the root;
    /// - [`PlaceError::Branched`] — a trace was given but some fragment is
    ///   joined to three others or more, so the fragments are neither a path
    ///   nor a ring;
    /// - [`PlaceError::Index`] — the trace has fewer samples than fragments to
    ///   place;
    /// - [`PlaceError::MissingCoordinates`] — a node the placement reads lacks
    ///   x, y or z: a forming bond's endpoint; a child's far site, or every
    ///   node of a child with no far site; without a trace, every node of a
    ///   parent; with one, every node of the root;
    /// - [`PlaceError::Unorientable`] — a child's site axis, or the trace
    ///   tangent it grows along, has no direction;
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

impl Junction {
    /// This junction as a step out of the fragment already placed: out of
    /// `to` when `reversed`, else out of `from`.
    fn step(&self, reversed: bool) -> Step {
        if reversed {
            Step {
                parent: self.to,
                child: self.from,
                parent_end: self.to_end,
                child_end: self.from_end,
            }
        } else {
            Step {
                parent: self.from,
                child: self.to,
                parent_end: self.from_end,
                child_end: self.to_end,
            }
        }
    }
}

/// A junction the walk crosses, oriented from the fragment already placed to
/// the one it places.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Step {
    parent: i64,
    child: i64,
    parent_end: Endpoint,
    child_end: Endpoint,
}

/// A walk over the fragment graph the junctions span.
///
/// `steps` is a spanning tree: every fragment but `root` is some step's child
/// exactly once, and its parent is placed before it. A junction the tree does
/// not use closes a ring of fragments; the walk leaves it out, so nothing
/// places it.
struct Walk {
    root: i64,
    steps: Vec<Step>,
}

impl Walk {
    /// Walk breadth-first from the lowest fragment id or, `as_path`, as one
    /// path: from the lowest id joined to at most one other fragment (the
    /// lowest id when every fragment is joined to two, a ring), each step
    /// leaving from the fragment the previous one placed. `None` when there is
    /// no junction.
    ///
    /// # Errors
    ///
    /// - [`PlaceError::Unreachable`] naming the lowest fragment the junctions
    ///   touch that has no path from the root;
    /// - [`PlaceError::Branched`], `as_path` only, naming the lowest fragment
    ///   joined to three others or more.
    fn new(junctions: &[Junction], as_path: bool) -> Result<Option<Self>, PlaceError> {
        let mut adjacency: HashMap<i64, Vec<(usize, bool)>> = HashMap::new();
        let mut partners: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
        for (index, junction) in junctions.iter().enumerate() {
            adjacency
                .entry(junction.from)
                .or_default()
                .push((index, false));
            adjacency
                .entry(junction.to)
                .or_default()
                .push((index, true));
            partners
                .entry(junction.from)
                .or_default()
                .insert(junction.to);
            partners
                .entry(junction.to)
                .or_default()
                .insert(junction.from);
        }
        let Some(&lowest) = partners.keys().next() else {
            return Ok(None);
        };
        let root = if as_path {
            partners
                .iter()
                .find(|(_, joined)| joined.len() <= 1)
                .map_or(lowest, |(&fragment, _)| fragment)
        } else {
            lowest
        };
        let leaving = |parent: i64| {
            adjacency
                .get(&parent)
                .into_iter()
                .flatten()
                .map(move |&(index, reversed)| junctions[index].step(reversed))
        };

        let mut seen = HashSet::from([root]);
        let mut tree = Vec::new();
        let mut queue = VecDeque::from([root]);
        while let Some(parent) = queue.pop_front() {
            for step in leaving(parent) {
                if seen.insert(step.child) {
                    queue.push_back(step.child);
                    tree.push(step);
                }
            }
        }
        if let Some(&fragment) = partners.keys().find(|fragment| !seen.contains(fragment)) {
            return Err(PlaceError::Unreachable { fragment });
        }
        if !as_path {
            return Ok(Some(Self { root, steps: tree }));
        }

        // Connected, and no fragment joined to more than two others: a path
        // walked from an end, or a ring walked round one way from `root`.
        if let Some((&fragment, _)) = partners.iter().find(|(_, joined)| joined.len() >= 3) {
            return Err(PlaceError::Branched { fragment });
        }
        let mut placed = HashSet::from([root]);
        let mut steps = Vec::new();
        let mut tip = root;
        while let Some(step) = leaving(tip).find(|step| !placed.contains(&step.child)) {
            placed.insert(step.child);
            tip = step.child;
            steps.push(step);
        }
        Ok(Some(Self { root, steps }))
    }
}

/// Grow the fragments a set of forming bonds joins out of one another.
///
/// Walks the fragment graph the forming bonds imply from a root fragment and
/// places every other fragment relative to **its own parent**: the child
/// moves rigidly so its anchor — the atom the bond reaches — sits one
/// bonding range (summed covalent radii plus the buffer) from the parent's
/// reacting atom, along a growth direction, and its [`Orienter`] direction
/// points along that same growth direction, away from the parent. Only whole
/// fragments move, each exactly once, so every forming bond of the spanning
/// tree ends at bonding range.
///
/// - **No trace (default).** The walk is breadth-first from the lowest
///   fragment id, which stays where it is. A child grows along its parent's
///   *outward* direction: from the parent's centroid through its reacting
///   atom, as the parent now stands (`+x` when those coincide). Any connected
///   fragment graph places — paths, stars, combs, rings — along the walk's
///   spanning tree.
/// - **With a trace ([`Self::with_trace`]).** The fragments must form a single
///   path, walked from its lowest-id end, or a single ring, walked round one
///   way from its lowest id as if it were a path. That root turns so its
///   outward direction follows the tangent at sample 0 and slides its
///   reacting atom onto sample 0; the `k`-th child grows along the tangent at
///   sample `k`.
///   The trace supplies directions only, so the chain follows the curve's
///   shape at bonding range rather than landing on the samples.
///
/// A fragment's site axis runs from its anchor to its far site (the first
/// other site-labelled node). A fragment with no far site has no site axis;
/// its body axis, anchor to centroid, is turned along the growth direction
/// instead, so it points away from its parent.
///
/// A forming bond the walk does not cross closes a ring of fragments. It is
/// not placed and not checked: it ends wherever the spanning tree leaves it.
/// A ring's size and shape are the caller's — a closed trace of the right
/// size lays the ring out, and geometry optimisation relaxes the closing bond.
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
    /// A placer with no trace and [`LineOrienter`] facing.
    pub fn new() -> Self {
        Self {
            trace: None,
            orienter: Box::new(LineOrienter),
            buffer: 0.0,
            group_key: keys::RES_ID.to_string(),
            site_key: keys::SITE.to_string(),
        }
    }

    /// Grow a path of fragments along this trace's tangents instead of each
    /// parent's outward direction. A ring of fragments is walked as a path
    /// from its lowest id; the bond that closes it is formed but not placed,
    /// so a closed trace of the ring's size is what brings its ends together.
    /// A fragment joined to three others or more is refused with
    /// [`PlaceError::Branched`].
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

    /// Where `node` stands now: its pending placement in `moved`, else the
    /// graph's own coordinates.
    fn current(
        world: &MolGraph,
        moved: &HashMap<NodeId, [F; 3]>,
        node: NodeId,
    ) -> Result<[F; 3], PlaceError> {
        match moved.get(&node) {
            Some(&position) => Ok(position),
            None => Self::located(world, node),
        }
    }

    /// Mean position of `nodes` as they stand now.
    fn centroid(
        world: &MolGraph,
        moved: &HashMap<NodeId, [F; 3]>,
        nodes: &[NodeId],
    ) -> Result<[F; 3], PlaceError> {
        let mut sum = [0.0; 3];
        for &node in nodes {
            let position = Self::current(world, moved, node)?;
            for axis in 0..3 {
                sum[axis] += position[axis];
            }
        }
        let count = nodes.len() as F;
        Ok([sum[0] / count, sum[1] / count, sum[2] / count])
    }

    /// Unit vector from a fragment's centroid through its `site` atom. A site
    /// atom sits on the fragment's surface, so this points out of the fragment
    /// — the way a partner should approach from. `+x` for a one-atom fragment
    /// or a site at the centroid, where any direction will do.
    fn outward(site: [F; 3], centroid: [F; 3]) -> [F; 3] {
        normalize([
            site[0] - centroid[0],
            site[1] - centroid[1],
            site[2] - centroid[2],
        ])
        .unwrap_or([1.0, 0.0, 0.0])
    }

    /// Where `nodes` land when their fragment turns about `anchor` by `turn`
    /// (a unit axis and an angle) and then slides `anchor` onto `target`. Reads
    /// the graph's own coordinates; a node without a full x, y, z is left out,
    /// so it stays where it is.
    fn posed(
        world: &MolGraph,
        nodes: &[NodeId],
        anchor: [F; 3],
        turn: Option<([F; 3], F)>,
        target: [F; 3],
    ) -> Vec<(NodeId, [F; 3])> {
        let turn = turn.map(|(axis, angle)| {
            let (sin_a, cos_a) = angle.sin_cos();
            (axis, cos_a, sin_a)
        });
        nodes
            .iter()
            .filter_map(|&node| {
                let mut p = Self::position(world, node)?;
                if let Some((axis, cos_a, sin_a)) = turn {
                    p = geometry::rotate_point(p, axis, cos_a, sin_a, anchor);
                }
                Some((
                    node,
                    [
                        p[0] - anchor[0] + target[0],
                        p[1] - anchor[1] + target[1],
                        p[2] - anchor[2] + target[2],
                    ],
                ))
            })
            .collect()
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

    /// The direction in `step.child`'s own frame that is turned along the
    /// growth direction: the orienter's reading of its site axis, or its body
    /// axis when it has no far site (`None` for a one-atom child, which has
    /// nothing to turn).
    fn heading(
        &self,
        world: &MolGraph,
        nodes: &[NodeId],
        step: &Step,
        anchor: [F; 3],
    ) -> Result<Option<[F; 3]>, PlaceError> {
        let Some(far) = self.far_site(world, nodes, step.child_end.node) else {
            let centroid = Self::centroid(world, &HashMap::new(), nodes)?;
            return Ok(normalize([
                centroid[0] - anchor[0],
                centroid[1] - anchor[1],
                centroid[2] - anchor[2],
            ]));
        };
        let far = Self::located(world, far)?;
        let site_axis = [far[0] - anchor[0], far[1] - anchor[1], far[2] - anchor[2]];
        self.orienter
            .direction(site_axis)
            .map(Some)
            .ok_or(PlaceError::Unorientable {
                fragment: Some(step.child),
            })
    }
}

impl Placer for TracePlacer {
    fn place(&self, world: &mut MolGraph, bonds: &[(NodeId, NodeId)]) -> Result<(), PlaceError> {
        let groups = self.fragments(world)?;
        let junctions = self.junctions(world, bonds, &groups)?;
        let Some(walk) = Walk::new(&junctions, self.trace.is_some())? else {
            return Ok(());
        };
        // Every fragment id the walk names was read off `groups` by
        // `junctions`, so indexing `groups` by it cannot miss.

        // Each fragment moves at most once and is read from the graph's own
        // coordinates when it does; a parent is read as it now stands through
        // `moved`. Nothing is written until every fragment is placed, so an
        // error leaves `world` untouched.
        let mut moved: HashMap<NodeId, [F; 3]> = HashMap::new();
        if let Some(trace) = &self.trace {
            if trace.len() <= walk.steps.len() {
                return Err(PlaceError::Index { index: trace.len() });
            }
            // A junction joins two fragments, so the root has a first step.
            let reacting_end = walk.steps[0].parent_end.node;
            let nodes = &groups[&walk.root];
            let reacting = Self::located(world, reacting_end)?;
            let outward = Self::outward(reacting, Self::centroid(world, &moved, nodes)?);
            let unorientable = PlaceError::Unorientable {
                fragment: Some(walk.root),
            };
            let tangent = trace.tangent(0).ok_or(unorientable)?;
            let sample = trace.point(0).ok_or(PlaceError::Index { index: 0 })?;
            let turn = geometry::alignment(outward, tangent);
            moved.extend(Self::posed(world, nodes, reacting, turn, sample));
        }

        for (index, step) in walk.steps.iter().enumerate() {
            let site = Self::current(world, &moved, step.parent_end.node)?;
            let direction = match &self.trace {
                Some(trace) => trace.tangent(index + 1).ok_or(PlaceError::Unorientable {
                    fragment: Some(step.child),
                })?,
                None => Self::outward(site, Self::centroid(world, &moved, &groups[&step.parent])?),
            };
            let reach = step.parent_end.radius + step.child_end.radius + self.buffer;
            let target = [
                site[0] + direction[0] * reach,
                site[1] + direction[1] * reach,
                site[2] + direction[2] * reach,
            ];
            let nodes = &groups[&step.child];
            let anchor = Self::located(world, step.child_end.node)?;
            let turn = self
                .heading(world, nodes, step, anchor)?
                .and_then(|heading| geometry::alignment(heading, direction));
            moved.extend(Self::posed(world, nodes, anchor, turn, target));
        }

        for (node, position) in moved {
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
        // `alignment` only ever yields a unit axis and a finite angle, which
        // `rotate` accepts; its error is still carried rather than dropped.
        geometry::orient(mol, anchor, from_dir, to_dir, flip)
            .map_err(|error| PlaceError::Graph(error.to_string()))
    }
}

/// The site axis itself: the fragment's outgoing site points along the growth
/// direction, so a chain of them runs away from each parent.
pub struct LineOrienter;

impl Orienter for LineOrienter {
    fn direction(&self, body_axis: [F; 3]) -> Option<[F; 3]> {
        normalize(body_axis)
    }
}

/// A perpendicular of the site axis: the fragment meets the growth direction
/// at an angle, so the chain kinks by that angle at every unit.
pub struct TangOrienter;

impl Orienter for TangOrienter {
    fn direction(&self, body_axis: [F; 3]) -> Option<[F; 3]> {
        perpendicular(body_axis)
    }
}

/// A [`Placer`] could not place the fragments it was given.
///
/// Nodes print as the opaque `u64` handle the bindings expose
/// ([`node_to_u64`]).
#[derive(Debug, Clone, PartialEq)]
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
    /// A trace lays out the fragments as one path, but `fragment` is joined
    /// to three others or more, so the forming bonds branch there.
    Branched { fragment: i64 },
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
            Self::MissingNode { node } => {
                write!(f, "graph holds no node {}", node_to_u64(*node))
            }
            Self::Ungrouped { node } => write!(
                f,
                "bond endpoint {} carries no fragment id",
                node_to_u64(*node)
            ),
            Self::MissingElement { node } => write!(
                f,
                "node {} carries no element, so no radius is known",
                node_to_u64(*node)
            ),
            Self::UnknownElement { symbol } => {
                write!(f, "no covalent radius tabulated for element '{symbol}'")
            }
            Self::MissingCoordinates { node } => write!(
                f,
                "node {} lacks a full x, y, z to place by",
                node_to_u64(*node)
            ),
            Self::Unreachable { fragment } => write!(
                f,
                "fragment {fragment} has no path of forming bonds from the root fragment"
            ),
            Self::Branched { fragment } => write!(
                f,
                "fragment {fragment} is joined to three fragments or more; a trace lays the fragments out as one path"
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

    // ----- Parent-relative placement: branched and cyclic topologies -----

    /// Summed covalent radii of the two atoms a forming bond joins: the
    /// bonding range at the default zero buffer.
    fn bonding_range(mol: &Atomistic, a: NodeId, b: NodeId) -> F {
        let radius = |node: NodeId| {
            let symbol = mol
                .as_molgraph()
                .get_node(node)
                .unwrap()
                .get_str(keys::ELEMENT)
                .unwrap()
                .to_string();
            Element::by_symbol(&symbol).unwrap().covalent_radius() as F
        };
        radius(a) + radius(b)
    }

    /// Every forming bond ends at bonding range, within 1e-6 Å.
    fn assert_all_at_bonding_range(mol: &Atomistic, bonds: &[(NodeId, NodeId)]) {
        for &(a, b) in bonds {
            let (got, expected) = (distance(mol, a, b), bonding_range(mol, a, b));
            assert!(
                (got - expected).abs() < 1e-6,
                "forming bond {a:?}-{b:?} is {got} Å long, expected {expected}"
            );
        }
    }

    /// Append one `C(a)-C-O(b)` unit (the [`chain`] fragment) with fragment id
    /// `id`, its head at `origin`. Returns `[head, middle, tail]`.
    fn add_unit(mol: &mut Atomistic, id: i64, origin: [F; 3]) -> [NodeId; 3] {
        let [x0, y0, z0] = origin;
        let head = mol.add_atom_xyz("C", x0, y0, z0);
        let middle = mol.add_atom_xyz("C", x0 + 1.2, y0 + 0.9, z0);
        let tail = mol.add_atom_xyz("O", x0 + 2.4, y0, z0);
        for (a, b) in [(head, middle), (middle, tail)] {
            mol.add_bond(a, b)
                .expect("a fresh bond between distinct atoms");
        }
        for node in [head, middle, tail] {
            mol.set_atom(node, keys::RES_ID, PropValue::Int(id as I))
                .unwrap();
        }
        mol.set_atom(head, keys::SITE, PropValue::Str("a".to_string()))
            .unwrap();
        mol.set_atom(tail, keys::SITE, PropValue::Str("b".to_string()))
            .unwrap();
        [head, middle, tail]
    }

    fn all_positions(mol: &Atomistic) -> Vec<[F; 3]> {
        mol.as_molgraph()
            .node_ids()
            .map(|node| position(mol, node))
            .collect()
    }

    #[test]
    fn a_star_puts_every_arm_bond_at_bonding_range() {
        // Core fragment 1: a carbon with three oxygen sites at 120 degrees.
        // Three arms hang off it, each two units long (ids 2..=4 next to the
        // core, 5..=7 at the arm ends). The arms start stacked far away, so
        // nothing is at bonding range before placement.
        let mut mol = Atomistic::new();
        let centre = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        mol.set_atom(centre, keys::RES_ID, PropValue::Int(1 as I))
            .unwrap();
        let mut core_sites = Vec::new();
        for k in 0..3 {
            let angle = 2.0 * std::f64::consts::PI * k as F / 3.0;
            let site = mol.add_atom_xyz("O", 1.43 * angle.cos(), 1.43 * angle.sin(), 0.0);
            mol.add_bond(centre, site)
                .expect("a fresh bond between distinct atoms");
            mol.set_atom(site, keys::RES_ID, PropValue::Int(1 as I))
                .unwrap();
            mol.set_atom(site, keys::SITE, PropValue::Str(format!("s{k}")))
                .unwrap();
            core_sites.push(site);
        }
        let mut bonds = Vec::new();
        for (k, &site) in core_sites.iter().enumerate() {
            let near = add_unit(&mut mol, 2 + k as i64, [20.0 + 4.0 * k as F, 7.0, 0.0]);
            let far = add_unit(&mut mol, 5 + k as i64, [20.0 + 4.0 * k as F, -7.0, 3.0]);
            bonds.push((site, near[0]));
            bonds.push((near[2], far[0]));
        }

        TracePlacer::new()
            .place(mol.as_molgraph_mut(), &bonds)
            .expect("a star is one connected fragment tree");
        assert_all_at_bonding_range(&mol, &bonds);
    }

    #[test]
    fn a_comb_puts_the_branch_bond_at_bonding_range() {
        // Backbone 1-2-3; branch fragment 4 hangs off fragment 2's middle
        // carbon, which is a site of its own.
        let mut mol = Atomistic::new();
        let backbone: Vec<[NodeId; 3]> = (1..=3)
            .map(|id| add_unit(&mut mol, id, [4.0 * id as F, 0.0, 0.0]))
            .collect();
        let branch = add_unit(&mut mol, 4, [30.0, -9.0, 2.0]);
        mol.set_atom(backbone[1][1], keys::SITE, PropValue::Str("c".to_string()))
            .unwrap();
        let bonds = [
            (backbone[0][2], backbone[1][0]),
            (backbone[1][2], backbone[2][0]),
            (backbone[1][1], branch[0]),
        ];

        TracePlacer::new()
            .place(mol.as_molgraph_mut(), &bonds)
            .expect("a comb is one connected fragment tree");
        assert_all_at_bonding_range(&mol, &bonds);
    }

    /// [`chain`] of `n` fragments closed into a ring: fragment `k`'s tail
    /// bonds to fragment `k + 1`'s head, and the last tail to the first head.
    /// Returns the world, its `[head, middle, tail]` per fragment, and the
    /// `n` forming bonds in that order, closing bond last.
    fn ring(n: usize) -> (Atomistic, Vec<[NodeId; 3]>, Vec<(NodeId, NodeId)>) {
        let (mol, fragments) = chain(n);
        let bonds = (0..n)
            .map(|k| (fragments[k][2], fragments[(k + 1) % n][0]))
            .collect();
        (mol, fragments, bonds)
    }

    #[test]
    fn a_ring_places_its_tree_bonds_and_leaves_the_closing_bond_to_the_caller() {
        // Five fragments bonded head-to-tail in a cycle. A placer knows only
        // each fragment's parent: it places the spanning tree of its walk, and
        // the one bond the tree leaves out closes the ring at whatever length
        // the tree leaves it. Closing it is geometry optimisation's job.
        let (mut mol, _, bonds) = ring(5);
        let before = all_positions(&mol);

        TracePlacer::new()
            .place(mol.as_molgraph_mut(), &bonds)
            .expect("a ring places without a trace");

        let at_range = bonds
            .iter()
            .filter(|&&(a, b)| (distance(&mol, a, b) - bonding_range(&mol, a, b)).abs() < 1e-6)
            .count();
        assert_eq!(
            at_range,
            bonds.len() - 1,
            "every spanning-tree bond, and only those, sits at bonding range"
        );
        assert_ne!(
            before,
            all_positions(&mol),
            "the ring placed without writing coordinates"
        );
    }

    #[test]
    fn an_explicit_trace_lays_a_ring_out_as_a_path_from_the_root() {
        // Five fragments in a cycle and a closed circular trace of six
        // samples (the last one back on the first), so its tangents run
        // round the ring. The ring is walked as a path from its root
        // (fragment 1, the lowest id); the bond back to the root is left
        // unchecked.
        let n = 5;
        let (mut mol, fragments, bonds) = ring(n);
        let radius = 6.0;
        let samples: Vec<[F; 3]> = (0..=n)
            .map(|k| {
                let angle = 2.0 * std::f64::consts::PI * k as F / n as F;
                [radius * angle.cos(), radius * angle.sin(), 0.0]
            })
            .collect();
        let trace = Trace::from_arrays(samples);

        TracePlacer::new()
            .with_trace(trace.clone())
            .place(mol.as_molgraph_mut(), &bonds)
            .expect("an explicit trace places a ring as a path, not as Branched");

        // The root slides its reacting atom onto sample 0. Which of its two
        // bonds the path leaves by is the placer's choice; the path is the
        // other n - 1 bonds, each oriented (parent side, child side).
        let sample = trace.point(0).unwrap();
        let on_sample = |node: NodeId| {
            let p = position(&mol, node);
            (0..3).all(|axis| (p[axis] - sample[axis]).abs() < 1e-8)
        };
        let path: Vec<(NodeId, NodeId)> = if on_sample(fragments[0][2]) {
            // 1 -> 2 -> ... -> n
            bonds[..n - 1].to_vec()
        } else if on_sample(fragments[0][0]) {
            // 1 -> n -> ... -> 2
            bonds[1..].iter().rev().map(|&(a, b)| (b, a)).collect()
        } else {
            panic!("neither of the root's reacting atoms sits on trace sample 0");
        };

        // The k-th child grows along the tangent at sample k: its anchor sits
        // one bonding range from its parent's reacting atom along it.
        for (index, &(site, anchor)) in path.iter().enumerate() {
            let tangent = trace.tangent(index + 1).unwrap();
            let reach = bonding_range(&mol, site, anchor);
            let (p, q) = (position(&mol, site), position(&mol, anchor));
            for axis in 0..3 {
                let expected = p[axis] + tangent[axis] * reach;
                assert!(
                    (q[axis] - expected).abs() < 1e-8,
                    "child {} anchor is off the tangent at sample {} on axis {axis}: {} vs {expected}",
                    index + 1,
                    index + 1,
                    q[axis]
                );
            }
        }
        assert_all_at_bonding_range(&mol, &path);
    }

    #[test]
    fn an_explicit_trace_refuses_a_star_as_branched_and_moves_nothing() {
        // Core fragment 1 has three fragment neighbours (2, 3, 4): a tree that
        // is neither a path nor a cycle, so a trace cannot lay it out.
        let mut mol = Atomistic::new();
        let centre = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        mol.set_atom(centre, keys::RES_ID, PropValue::Int(1 as I))
            .unwrap();
        let mut bonds = Vec::new();
        for k in 0..3 {
            let angle = 2.0 * std::f64::consts::PI * k as F / 3.0;
            let site = mol.add_atom_xyz("O", 1.43 * angle.cos(), 1.43 * angle.sin(), 0.0);
            mol.add_bond(centre, site)
                .expect("a fresh bond between distinct atoms");
            mol.set_atom(site, keys::RES_ID, PropValue::Int(1 as I))
                .unwrap();
            mol.set_atom(site, keys::SITE, PropValue::Str(format!("s{k}")))
                .unwrap();
            let arm = add_unit(&mut mol, 2 + k as i64, [20.0 + 4.0 * k as F, 7.0, 0.0]);
            bonds.push((site, arm[0]));
        }
        // Samples to spare, so a short trace is not what is refused.
        let trace = Trace::from_arrays((0..6).map(|k| [2.0 * k as F, 0.0, 0.0]).collect());
        let before = all_positions(&mol);

        let error = TracePlacer::new()
            .with_trace(trace)
            .place(mol.as_molgraph_mut(), &bonds)
            .expect_err("a star is not a path");
        assert_eq!(error, PlaceError::Branched { fragment: 1 });
        assert_eq!(
            before,
            all_positions(&mol),
            "a refused star wrote coordinates"
        );
    }

    #[test]
    fn neighbouring_fragments_of_a_chain_do_not_clash() {
        // Units `C(a)(H)-C-O(b)` with sp3 geometry at the head: the head's
        // hydrogen and middle carbon sit at tetrahedral positions, leaving the
        // fourth tetrahedral direction (-x) free for the incoming bond. A
        // placement that turned the child so its hydrogen pointed at the
        // parent's oxygen would put them ~0.3 Å apart.
        let mut mol = Atomistic::new();
        let mut units: Vec<Vec<NodeId>> = Vec::new();
        for id in 1..=3 {
            let x0 = 6.0 * id as F;
            let head = mol.add_atom_xyz("C", x0, 0.0, 0.0);
            let hydrogen = mol.add_atom_xyz("H", x0 + 0.3633, -0.5139, 0.8900);
            let middle = mol.add_atom_xyz("C", x0 + 0.5133, 1.4519, 0.0);
            let tail = mol.add_atom_xyz("O", x0 + 1.9433, 1.4519, 0.0);
            for (a, b) in [(head, hydrogen), (head, middle), (middle, tail)] {
                mol.add_bond(a, b)
                    .expect("a fresh bond between distinct atoms");
            }
            for node in [head, hydrogen, middle, tail] {
                mol.set_atom(node, keys::RES_ID, PropValue::Int(id as I))
                    .unwrap();
            }
            mol.set_atom(head, keys::SITE, PropValue::Str("a".to_string()))
                .unwrap();
            mol.set_atom(tail, keys::SITE, PropValue::Str("b".to_string()))
                .unwrap();
            units.push(vec![head, hydrogen, middle, tail]);
        }
        let bonds = [(units[0][3], units[1][0]), (units[1][3], units[2][0])];

        TracePlacer::new()
            .place(mol.as_molgraph_mut(), &bonds)
            .expect("a linear chain places");
        assert_all_at_bonding_range(&mol, &bonds);
        for (i, first) in units.iter().enumerate() {
            for second in &units[i + 1..] {
                for &a in first {
                    for &b in second {
                        if bonds.contains(&(a, b)) || bonds.contains(&(b, a)) {
                            continue;
                        }
                        let d = distance(&mol, a, b);
                        assert!(d >= 1.0, "atoms {a:?} and {b:?} clash at {d} Å");
                    }
                }
            }
        }
    }
}
