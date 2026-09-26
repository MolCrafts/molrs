//! [`Assembler`]: the composition point that turns a
//! [`FragGraph`](crate::system::frag_graph::FragGraph) into one placed and
//! linked world [`Fragment`](crate::system::fragment::Fragment).
//!
//! The Assembler composes three pieces it is handed and never reimplements:
//! a [`FragLibrary`] of templates, a [`Placer`] that answers where each unit
//! goes, and a [`Reacter`] that joins two ports. It validates the whole graph
//! before placing anything, places and replicates one template group at a
//! time, then links every edge in **one** [`Reacter::link_many`] batch.
//!
//! Connection is port-only: an edge addresses a port by its ordinal in the
//! template's [`ordered_ports`](Fragment::ordered_ports), and ports no edge
//! names stay on the returned world as ports. Completing the bonded topology
//! and labelling residues are separate pipeline steps the caller composes; the
//! Assembler's output is the linked world and nothing more.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::builder::library::FragLibrary;
use crate::builder::place::{PlaceError, Placer};
use crate::builder::react::{ReactError, Reacter};
use crate::error::MolRsError;
use crate::system::atomistic::AtomId;
use crate::system::frag_graph::FragGraph;
use crate::system::fragment::{Fragment, PortId};
use crate::types::I;

/// Why an [`Assembler`] refused a [`FragGraph`].
#[derive(Debug)]
pub enum AssembleError {
    /// Node `node` names `name`, which is not a template of the library.
    UnknownTemplate {
        /// The graph node.
        node: usize,
        /// The template name it carries.
        name: String,
    },
    /// Edge `edge` addresses port ordinal `ordinal` of node `node`, but that
    /// node's template has fewer ordered ports.
    NoSuchPort {
        /// The graph edge.
        edge: usize,
        /// The endpoint node whose template lacks the ordinal.
        node: usize,
        /// The requested port ordinal.
        ordinal: usize,
    },
    /// Node index `node` does not fit the `i32` `frag_id` column.
    FragIdOverflow {
        /// The graph node.
        node: usize,
    },
    /// The [`Placer`] could not place a unit.
    Place(PlaceError),
    /// The [`Reacter`] refused edge `edge`.
    React {
        /// The graph edge (its index in [`FragGraph::edges`]); `None` when the
        /// reacter could not name the failing pair.
        edge: Option<usize>,
        /// Why the pair was refused.
        error: ReactError,
    },
    /// A graph operation failed: a template's ports do not read back, or the
    /// world refused a replicated copy.
    Graph(MolRsError),
}

impl fmt::Display for AssembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTemplate { node, name } => {
                write!(
                    f,
                    "node {node} names template '{name}', which the library lacks"
                )
            }
            Self::NoSuchPort {
                edge,
                node,
                ordinal,
            } => write!(
                f,
                "edge {edge} addresses port ordinal {ordinal} of node {node}, \
                 which its template does not have"
            ),
            Self::FragIdOverflow { node } => {
                write!(f, "node index {node} does not fit the i32 frag_id column")
            }
            Self::Place(e) => write!(f, "placement failed: {e}"),
            Self::React {
                edge: Some(edge),
                error,
            } => write!(f, "edge {edge} could not be linked: {error}"),
            Self::React { edge: None, error } => {
                write!(f, "an edge could not be linked: {error}")
            }
            Self::Graph(e) => write!(f, "graph operation failed: {e}"),
        }
    }
}

impl std::error::Error for AssembleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Place(e) => Some(e),
            Self::React { error, .. } => Some(error),
            Self::Graph(e) => Some(e),
            _ => None,
        }
    }
}

/// Builds one placed, linked world [`Fragment`] from a [`FragGraph`].
///
/// The **world** is the single output fragment that holds every placed unit:
/// one copy of its template per graph node, moved into position by the
/// placer, with the copies joined port to port by the reacter.
pub struct Assembler {
    library: FragLibrary,
    placer: Box<dyn Placer>,
    reacter: Box<dyn Reacter>,
}

impl Assembler {
    /// An assembler over `library`, placing units with `placer` and joining
    /// ports with `reacter`.
    pub fn new(library: FragLibrary, placer: Box<dyn Placer>, reacter: Box<dyn Reacter>) -> Self {
        Self {
            library,
            placer,
            reacter,
        }
    }

    /// Place every node of `graph` as a copy of its template and link every
    /// edge.
    ///
    /// 1. **Validate** — before anything is placed: every node names a library
    ///    template, every edge ordinal exists in its template's ordered ports,
    ///    and every node index fits `frag_id`.
    /// 2. **Place and replicate** — nodes grouped by template name (in name
    ///    order); per group one [`Placer::place_many`] and one
    ///    [`Fragment::replicate`], each copy stamped `frag_id` = node index.
    /// 3. **Link** — every edge resolved to its two world ports, joined in one
    ///    [`Reacter::link_many`] call.
    ///
    /// Ports no edge names remain on the returned world. World atoms are laid
    /// out grouped by template name, not in node order; `frag_id` (the node
    /// index) is the unit key, so read a unit's atoms by `frag_id`, never by
    /// row.
    ///
    /// The world is a [`Fragment`]. Topology perception takes an
    /// [`Atomistic`](crate::system::atomistic::Atomistic); convert with
    /// `Atomistic::try_from_molgraph(world.into_inner())`
    /// ([`Atomistic::try_from_molgraph`](crate::system::atomistic::Atomistic::try_from_molgraph)).
    /// The `ports` relation kind, with any ports left unlinked, stays on the
    /// converted graph.
    ///
    /// # Errors
    ///
    /// [`AssembleError::UnknownTemplate`], [`AssembleError::NoSuchPort`] and
    /// [`AssembleError::FragIdOverflow`] from validation (nothing placed);
    /// [`AssembleError::Place`] from the placer, or when its `place_many`
    /// returns a rigid count other than the group's unit count;
    /// [`AssembleError::React`]
    /// naming the failing edge; [`AssembleError::Graph`] when a template's
    /// ports do not read back or the world refuses a copy.
    pub fn assemble(&self, graph: &FragGraph) -> Result<Fragment, AssembleError> {
        let nodes = graph.nodes();

        // ---- 1. validate: nothing is placed until every check passes ----
        // Per template: (anchor row, handle row) of each port, by ordinal.
        let mut port_rows: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
        for (node, name) in nodes.iter().enumerate() {
            let template =
                self.library
                    .get(name)
                    .ok_or_else(|| AssembleError::UnknownTemplate {
                        node,
                        name: name.clone(),
                    })?;
            if !port_rows.contains_key(name.as_str()) {
                port_rows.insert(name, Self::port_rows(template)?);
            }
            I::try_from(node).map_err(|_| AssembleError::FragIdOverflow { node })?;
        }
        for (edge, e) in graph.edges().iter().enumerate() {
            for (node, ordinal) in [(e.a, e.port_a), (e.b, e.port_b)] {
                if ordinal >= port_rows[nodes[node].as_str()].len() {
                    return Err(AssembleError::NoSuchPort {
                        edge,
                        node,
                        ordinal,
                    });
                }
            }
        }

        // ---- 2. place and replicate, one template group at a time ----
        let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (node, name) in nodes.iter().enumerate() {
            groups.entry(name).or_default().push(node);
        }
        let mut world = Fragment::new();
        // The world atoms of each node, in template node-row order.
        let mut node_atoms: Vec<Vec<AtomId>> = vec![Vec::new(); nodes.len()];
        for (name, units) in &groups {
            let template = self
                .library
                .get(name)
                .expect("every node's template was checked above");
            let rigids = self
                .placer
                .place_many(units, name, template)
                .map_err(AssembleError::Place)?;
            if rigids.len() != units.len() {
                return Err(AssembleError::Place(PlaceError::Other(format!(
                    "place_many for template '{name}' returned {} rigids for {} units",
                    rigids.len(),
                    units.len()
                ))));
            }
            let frag_ids: Vec<I> = units
                .iter()
                .map(|&u| I::try_from(u).expect("every node index was checked to fit I"))
                .collect();
            let atoms = world
                .replicate(template, &rigids, &frag_ids)
                .map_err(AssembleError::Graph)?;
            let n = template.n_atoms();
            for (c, &unit) in units.iter().enumerate() {
                node_atoms[unit] = atoms[c * n..(c + 1) * n].to_vec();
            }
        }

        // ---- 3. resolve every edge to its two world ports; link once ----
        let mut world_ports: HashMap<(AtomId, AtomId), PortId> =
            HashMap::with_capacity(world.n_ports());
        for id in world.ports() {
            let port = world.port(id).map_err(AssembleError::Graph)?;
            world_ports.insert((port.anchor, port.handle), id);
        }
        let resolve = |node: usize, ordinal: usize| -> Result<PortId, AssembleError> {
            let (anchor_row, handle_row) = port_rows[nodes[node].as_str()][ordinal];
            let atoms = &node_atoms[node];
            let key = (atoms[anchor_row], atoms[handle_row]);
            world_ports.get(&key).copied().ok_or_else(|| {
                AssembleError::Graph(MolRsError::validation(format!(
                    "port ordinal {ordinal} of node {node} has no world port after replication"
                )))
            })
        };
        let pairs: Vec<(PortId, PortId)> = graph
            .edges()
            .iter()
            .map(|e| Ok((resolve(e.a, e.port_a)?, resolve(e.b, e.port_b)?)))
            .collect::<Result<_, AssembleError>>()?;
        self.reacter
            .link_many(&mut world, &pairs)
            .map_err(|e| AssembleError::React {
                edge: e.pair,
                error: e.error,
            })?;

        // ---- 4. the world ----
        Ok(world)
    }

    /// `(anchor row, handle row)` of each of `template`'s ports, by ordinal.
    fn port_rows(template: &Fragment) -> Result<Vec<(usize, usize)>, AssembleError> {
        let row = |atom: AtomId| {
            template.node_table().row(atom).ok_or_else(|| {
                AssembleError::Graph(MolRsError::validation(format!(
                    "template port names {atom:?}, which is no live atom"
                )))
            })
        };
        template
            .ordered_ports()
            .map_err(AssembleError::Graph)?
            .into_iter()
            .map(|id| {
                let port = template.port(id).map_err(AssembleError::Graph)?;
                Ok((row(port.anchor)?, row(port.handle)?))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{AssembleError, Assembler};
    use crate::builder::{FragLibrary, PairError, PlaceError, Placer, PortReacter, ReactError};
    use crate::builder::{Reacter, TracePlacer};
    use crate::op::rigid::Rigid;
    use crate::spatial::Trace;
    use crate::store::keys;
    use crate::system::atomistic::{AtomId, BondId};
    use crate::system::bond::BondNumber;
    use crate::system::frag_graph::{FragEdge, FragGraph};
    use crate::system::fragment::{Fragment, PortId, PortKind};

    // ---- fixtures ----------------------------------------------------------
    //
    // Template U is one bead (bead 0, bead_type "U"): a carbon anchor at the
    // origin with two H handles at x = −1 (port `<`, ordinal 0) and x = +1
    // (port `>`, ordinal 1), both unlabelled, order Single. The H masses are
    // equal, so the bead's mass-weighted centroid is the carbon's position
    // (0,0,0) exactly. With k = 1 bead per unit the fit is `Free`; the default
    // NullOrienter keeps R = I, so each unit is a pure translation onto its
    // trace point and the carbons land at x = 0, 1.54, 3.08.
    //
    // A three-unit path joins ordinal 1 (`>`) of unit i to ordinal 0 (`<`) of
    // unit i + 1: two links, each removing two handles (−4 atoms, −4 handle
    // bonds) and adding one C–C bond. So 9 − 4 = 5 atoms, 6 − 4 + 2 = 4
    // bonds, and the two end handles keep their ports (2). All values are
    // hand-derived; no external program produced any of them.

    const C_MASS: f64 = 12.011;
    const H_MASS: f64 = 1.008;
    const TOL: f64 = 1e-12;

    /// Template U (see the fixture note above).
    fn template_u() -> Fragment {
        one_bead_template("U")
    }

    /// A one-bead template shaped like U whose atoms carry `bead_type`.
    fn one_bead_template(bead_type: &str) -> Fragment {
        let mut u = Fragment::new();
        let c = u.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let left = u.add_atom_xyz("H", -1.0, 0.0, 0.0);
        let right = u.add_atom_xyz("H", 1.0, 0.0, 0.0);
        for (id, mass) in [(c, C_MASS), (left, H_MASS), (right, H_MASS)] {
            u.set_node(id, keys::MASS, mass).expect("stamp mass");
            u.set_node(id, keys::BEAD, 0_i32).expect("stamp bead");
            u.set_node(id, keys::BEAD_TYPE, bead_type)
                .expect("stamp bead_type");
        }
        u.add_bond(c, left).expect("C–H left");
        u.add_bond(c, right).expect("C–H right");
        u.add_port(c, left, PortKind::Left, "", BondNumber::Single)
            .expect("port <");
        u.add_port(c, right, PortKind::Right, "", BondNumber::Single)
            .expect("port >");
        u
    }

    fn library_u() -> FragLibrary {
        let mut lib = FragLibrary::new();
        lib.insert("U", template_u())
            .expect("U is a valid template");
        lib
    }

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_owned()).collect()
    }

    /// `FragGraph::path([U, U, U], (1, 0))`.
    fn three_unit_path() -> FragGraph {
        FragGraph::path(names(&["U", "U", "U"]), (1, 0)).expect("three-unit path")
    }

    /// A TracePlacer over one point per unit at x = 0, 1.54, 3.08.
    fn three_point_placer() -> TracePlacer {
        let trace = Trace::from_points(vec![[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [3.08, 0.0, 0.0]]);
        TracePlacer::new(trace, names(&["U", "U", "U"])).expect("seq matches the trace")
    }

    /// Places every unit where its template already sits.
    struct IdentityPlacer;

    impl Placer for IdentityPlacer {
        fn place(
            &self,
            _unit: usize,
            _name: &str,
            _fragment: &Fragment,
        ) -> Result<Rigid, PlaceError> {
            Ok(Rigid::IDENTITY)
        }
    }

    /// Counts every `place` and `place_many` call, then delegates.
    struct CountingPlacer {
        inner: Box<dyn Placer>,
        calls: Arc<AtomicUsize>,
    }

    impl Placer for CountingPlacer {
        fn place(&self, unit: usize, name: &str, fragment: &Fragment) -> Result<Rigid, PlaceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.place(unit, name, fragment)
        }

        fn place_many(
            &self,
            units: &[usize],
            name: &str,
            fragment: &Fragment,
        ) -> Result<Vec<Rigid>, PlaceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.place_many(units, name, fragment)
        }
    }

    /// `PortReacter` for `link`; counts `link_many` calls and loops over
    /// `link` like the default.
    struct CountingReacter {
        link_many_calls: Arc<AtomicUsize>,
    }

    impl Reacter for CountingReacter {
        fn link(&self, world: &mut Fragment, a: PortId, b: PortId) -> Result<BondId, ReactError> {
            PortReacter.link(world, a, b)
        }

        fn link_many(
            &self,
            world: &mut Fragment,
            pairs: &[(PortId, PortId)],
        ) -> Result<Vec<BondId>, PairError> {
            self.link_many_calls.fetch_add(1, Ordering::SeqCst);
            pairs
                .iter()
                .enumerate()
                .map(|(i, &(a, b))| {
                    self.link(world, a, b).map_err(|error| PairError {
                        pair: Some(i),
                        error,
                    })
                })
                .collect()
        }
    }

    fn element(world: &Fragment, atom: AtomId) -> String {
        world
            .get_node(atom)
            .expect("live atom")
            .get_str(keys::ELEMENT)
            .expect("atom carries an element")
            .to_owned()
    }

    /// The carbons of `world`, sorted by `frag_id`, as `(frag_id, position)`.
    fn carbons(world: &Fragment) -> Vec<(u32, [f64; 3])> {
        let mut out: Vec<(u32, [f64; 3])> = world
            .node_ids()
            .filter(|&id| element(world, id) == "C")
            .map(|id| {
                let frag = world
                    .frag_id(id)
                    .expect("every placed atom carries frag_id");
                let pos = world
                    .get_node(id)
                    .expect("live atom")
                    .position()
                    .expect("a placed U carbon has coordinates");
                (frag, pos)
            })
            .collect();
        out.sort_by_key(|&(frag, _)| frag);
        out
    }

    // ---- ac-006: the three-unit chain -------------------------------------

    #[test]
    fn assembles_a_three_unit_chain_placed_on_its_trace() {
        let assembler = Assembler::new(
            library_u(),
            Box::new(three_point_placer()),
            Box::new(PortReacter),
        );

        let world = assembler
            .assemble(&three_unit_path())
            .expect("the U chain assembles");

        assert_eq!(world.n_atoms(), 5, "3 C + the 2 end handles");
        assert_eq!(world.n_bonds(), 4, "2 C–C + 2 end C–H");
        assert_eq!(world.n_ports(), 2, "the two end ports stay unpaired");

        let carbons = carbons(&world);
        let frag_ids: Vec<u32> = carbons.iter().map(|&(f, _)| f).collect();
        assert_eq!(frag_ids, vec![0, 1, 2], "carbon frag_id = node index");
        for (&(frag, pos), want_x) in carbons.iter().zip([0.0, 1.54, 3.08]) {
            let want = [want_x, 0.0, 0.0];
            for k in 0..3 {
                assert!(
                    (pos[k] - want[k]).abs() <= TOL,
                    "carbon of unit {frag}: got {pos:?}, want {want:?} (tol {TOL})"
                );
            }
        }
    }

    #[test]
    fn links_every_edge_in_one_link_many_call() {
        let link_many_calls = Arc::new(AtomicUsize::new(0));
        let assembler = Assembler::new(
            library_u(),
            Box::new(three_point_placer()),
            Box::new(CountingReacter {
                link_many_calls: Arc::clone(&link_many_calls),
            }),
        );

        let world = assembler
            .assemble(&three_unit_path())
            .expect("the U chain assembles");

        assert_eq!(
            link_many_calls.load(Ordering::SeqCst),
            1,
            "both edges go through one batch"
        );
        assert_eq!(world.n_bonds(), 4, "both edges were linked");
    }

    // ---- ac-007: validation precedes placement ----------------------------

    #[test]
    fn refuses_an_unknown_template_before_placing() {
        let calls = Arc::new(AtomicUsize::new(0));
        let assembler = Assembler::new(
            library_u(),
            Box::new(CountingPlacer {
                inner: Box::new(IdentityPlacer),
                calls: Arc::clone(&calls),
            }),
            Box::new(PortReacter),
        );
        // Node 0 names a known template, so a placer running before the name
        // check would already have been called for it.
        let graph = FragGraph::path(names(&["U", "X"]), (1, 0)).expect("two-node path");

        let err = assembler
            .assemble(&graph)
            .expect_err("X is not in the library");

        assert!(
            matches!(
                err,
                AssembleError::UnknownTemplate { node: 1, ref name } if name == "X"
            ),
            "got {err:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no unit was placed");
    }

    #[test]
    fn refuses_a_port_ordinal_past_the_template_before_placing() {
        let calls = Arc::new(AtomicUsize::new(0));
        let assembler = Assembler::new(
            library_u(),
            Box::new(CountingPlacer {
                inner: Box::new(IdentityPlacer),
                calls: Arc::clone(&calls),
            }),
            Box::new(PortReacter),
        );
        // U has ordinals 0 and 1 only.
        let graph = FragGraph::new(
            names(&["U", "U"]),
            vec![FragEdge {
                a: 0,
                b: 1,
                port_a: 5,
                port_b: 0,
            }],
        )
        .expect("the graph itself is well-formed");

        let err = assembler
            .assemble(&graph)
            .expect_err("ordinal 5 does not exist on U");

        assert!(
            matches!(
                err,
                AssembleError::NoSuchPort {
                    edge: 0,
                    node: 0,
                    ordinal: 5
                }
            ),
            "got {err:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no unit was placed");
    }

    // ---- Assembler-own behaviour, with stub placers ------------------------

    /// Translates unit `u` to x = 10·u, R = I.
    struct TenPerUnitPlacer;

    impl Placer for TenPerUnitPlacer {
        fn place(
            &self,
            unit: usize,
            _name: &str,
            _fragment: &Fragment,
        ) -> Result<Rigid, PlaceError> {
            Ok(Rigid {
                rotation: Rigid::IDENTITY.rotation,
                translation: [10.0 * unit as f64, 0.0, 0.0],
            })
        }
    }

    /// Refuses every unit with `PlaceError::Other("boom")`.
    struct FailingPlacer;

    impl Placer for FailingPlacer {
        fn place(
            &self,
            _unit: usize,
            _name: &str,
            _fragment: &Fragment,
        ) -> Result<Rigid, PlaceError> {
            Err(PlaceError::Other("boom".to_owned()))
        }
    }

    /// A broken batch override: one rigid fewer than `units`.
    struct ShortBatchPlacer;

    impl Placer for ShortBatchPlacer {
        fn place(
            &self,
            _unit: usize,
            _name: &str,
            _fragment: &Fragment,
        ) -> Result<Rigid, PlaceError> {
            Ok(Rigid::IDENTITY)
        }

        fn place_many(
            &self,
            units: &[usize],
            _name: &str,
            _fragment: &Fragment,
        ) -> Result<Vec<Rigid>, PlaceError> {
            Ok(vec![Rigid::IDENTITY; units.len().saturating_sub(1)])
        }
    }

    /// Nodes [U, V, U]: the two U nodes form one placement group, yet every
    /// copy is stamped with its own node index and placed by its own rigid.
    #[test]
    fn groups_units_by_template_but_stamps_node_order_frag_ids() {
        let mut library = library_u();
        library
            .insert("V", one_bead_template("V"))
            .expect("V is a valid template");
        let assembler = Assembler::new(library, Box::new(TenPerUnitPlacer), Box::new(PortReacter));
        let graph = FragGraph::path(names(&["U", "V", "U"]), (1, 0)).expect("three-node path");

        let world = assembler.assemble(&graph).expect("U–V–U assembles");

        let carbons = carbons(&world);
        let frag_ids: Vec<u32> = carbons.iter().map(|&(f, _)| f).collect();
        assert_eq!(frag_ids, vec![0, 1, 2], "carbon frag_id = node index");
        for &(frag, pos) in &carbons {
            let want = [10.0 * f64::from(frag), 0.0, 0.0];
            for k in 0..3 {
                assert!(
                    (pos[k] - want[k]).abs() <= TOL,
                    "carbon of unit {frag}: got {pos:?}, want {want:?} (tol {TOL})"
                );
            }
        }
        let v_carbon = world
            .node_ids()
            .find(|&id| world.frag_id(id) == Some(1) && element(&world, id) == "C")
            .expect("node 1 holds a carbon");
        assert_eq!(
            world
                .get_node(v_carbon)
                .expect("live atom")
                .get_str(keys::BEAD_TYPE),
            Some("V"),
            "node 1 is a copy of V"
        );
    }

    #[test]
    fn maps_a_failing_link_to_its_edge_index() {
        let assembler =
            Assembler::new(library_u(), Box::new(IdentityPlacer), Box::new(PortReacter));
        // Edge 0 joins `>` to `<`; edge 1 joins `>` to `>`.
        let graph = FragGraph::new(
            names(&["U", "U", "U"]),
            vec![
                FragEdge {
                    a: 0,
                    b: 1,
                    port_a: 1,
                    port_b: 0,
                },
                FragEdge {
                    a: 1,
                    b: 2,
                    port_a: 1,
                    port_b: 1,
                },
            ],
        )
        .expect("the graph itself is well-formed");

        let err = assembler
            .assemble(&graph)
            .expect_err("edge 1 pairs two `>` ports");

        assert!(
            matches!(
                err,
                AssembleError::React {
                    edge: Some(1),
                    error: ReactError::Incompatible { .. }
                }
            ),
            "got {err:?}"
        );
    }

    /// A batch reacter that refuses without naming the pair.
    struct AnonymousFailureReacter;

    impl Reacter for AnonymousFailureReacter {
        fn link(&self, world: &mut Fragment, a: PortId, b: PortId) -> Result<BondId, ReactError> {
            PortReacter.link(world, a, b)
        }

        fn link_many(
            &self,
            _world: &mut Fragment,
            _pairs: &[(PortId, PortId)],
        ) -> Result<Vec<BondId>, PairError> {
            Err(PairError {
                pair: None,
                error: ReactError::Other("refused".to_owned()),
            })
        }
    }

    /// Amended 2026-09-26: a batch failure that cannot name its pair maps to
    /// `React { edge: None }`, never a made-up index.
    #[test]
    fn an_unnamed_pair_failure_maps_to_an_unnamed_edge() {
        let assembler = Assembler::new(
            library_u(),
            Box::new(three_point_placer()),
            Box::new(AnonymousFailureReacter),
        );

        let err = assembler
            .assemble(&three_unit_path())
            .expect_err("the reacter refuses the batch");

        assert!(
            matches!(
                err,
                AssembleError::React {
                    edge: None,
                    error: ReactError::Other(_)
                }
            ),
            "got {err:?}"
        );
        assert!(err.to_string().contains("an edge"), "got {err}");
    }

    #[test]
    fn maps_a_placer_failure_to_place() {
        let assembler = Assembler::new(library_u(), Box::new(FailingPlacer), Box::new(PortReacter));

        let err = assembler
            .assemble(&three_unit_path())
            .expect_err("the placer refuses every unit");

        assert!(
            matches!(err, AssembleError::Place(PlaceError::Other(ref msg)) if msg == "boom"),
            "got {err:?}"
        );
    }

    /// The `Place` display carries the "placement failed: " prefix exactly
    /// once, followed by the placer's own message.
    #[test]
    fn place_error_display_prefixes_placement_failed_once() {
        let err = AssembleError::Place(PlaceError::Other("boom".into()));

        let text = err.to_string();

        assert_eq!(text, "placement failed: boom");
        assert_eq!(text.matches("placement failed").count(), 1);
    }

    /// Amended 2026-09-26: a placer breaking the `place_many` count contract
    /// is a placement failure, not a graph failure.
    #[test]
    fn a_placer_returning_the_wrong_count_is_a_place_error() {
        let assembler = Assembler::new(
            library_u(),
            Box::new(ShortBatchPlacer),
            Box::new(PortReacter),
        );

        let err = assembler
            .assemble(&three_unit_path())
            .expect_err("three units, two rigids");

        assert!(
            matches!(err, AssembleError::Place(PlaceError::Other(_))),
            "got {err:?}"
        );
    }

    // FragIdOverflow needs a node index above i32::MAX, i.e. a FragGraph of
    // more than 2^31 nodes; no small fixture reaches it, so it has no unit
    // test here.

    // ---- ac-009: the Assembler door on the shared heavy-atom golden -------

    /// `{[#A][#B]}.{#A=CC=[>],#B=[<]=CO}` — the heavy-atom skeleton of
    /// prop-1-en-1-ol, C–C=C–O. The `=` belongs to each descriptor, so both
    /// ports carry order 2 (spec assembly-06 § Domain basis). Hand-derived
    /// golden: heavy atoms C, C, C, O in chain order, bond numbers
    /// (1, 2, 1) along the chain, frag_id [0, 0, 1, 1].
    #[cfg(feature = "smiles")]
    const SHARED_GOLDEN: &str = "{[#A][#B]}.{#A=CC=[>],#B=[<]=CO}";

    /// Walk the unbranched chain of `world` from the terminal atom of
    /// frag 0, returning its elements, bond numbers and frag_ids in order.
    #[cfg(feature = "smiles")]
    fn chain(world: &Fragment) -> (Vec<String>, Vec<BondNumber>, Vec<u32>) {
        use std::collections::BTreeMap;

        let mut adj: BTreeMap<AtomId, Vec<(AtomId, BondNumber)>> = BTreeMap::new();
        for (_, bond) in world.bonds() {
            let (a, b) = (bond.nodes[0], bond.nodes[1]);
            let n = BondNumber::from_prop(bond.props.get(keys::BOND_NUMBER));
            adj.entry(a).or_default().push((b, n));
            adj.entry(b).or_default().push((a, n));
        }
        let start = world
            .node_ids()
            .find(|&id| world.frag_id(id) == Some(0) && adj.get(&id).map_or(0, Vec::len) == 1)
            .expect("frag 0 holds a chain end");
        let mut elements = vec![element(world, start)];
        let mut numbers = Vec::new();
        let mut frags = vec![world.frag_id(start).expect("frag_id")];
        let (mut prev, mut here) = (None, start);
        while let Some(&(next, n)) = adj[&here].iter().find(|&&(o, _)| Some(o) != prev) {
            elements.push(element(world, next));
            numbers.push(n);
            frags.push(world.frag_id(next).expect("frag_id"));
            prev = Some(here);
            here = next;
        }
        (elements, numbers, frags)
    }

    #[cfg(feature = "smiles")]
    #[test]
    fn assembler_door_reproduces_the_shared_heavy_atom_golden() {
        use crate::io::smiles::parse_cgsmiles;

        // Test-fixture exception (architecture rules): the templates are
        // built through io::smiles, one per definition.
        let mut templates = parse_cgsmiles(SHARED_GOLDEN)
            .expect("the golden string parses")
            .to_fragment()
            .expect("per-definition templates build");
        let mut library = FragLibrary::new();
        for name in ["A", "B"] {
            let template = templates
                .remove(name)
                .unwrap_or_else(|| panic!("the table defines #{name}"));
            library
                .insert(name, template)
                .unwrap_or_else(|e| panic!("template {name} is valid: {e}"));
        }
        let graph = FragGraph::new(
            names(&["A", "B"]),
            vec![FragEdge {
                a: 0,
                b: 1,
                port_a: 0,
                port_b: 0,
            }],
        )
        .expect("one edge between the single ports");
        let assembler = Assembler::new(library, Box::new(IdentityPlacer), Box::new(PortReacter));

        let world = assembler.assemble(&graph).expect("A–B assembles");

        assert_eq!(world.n_ports(), 0, "both ports were consumed");
        assert_eq!(world.n_atoms(), 4, "only the four heavy atoms remain");
        assert!(
            world.node_ids().all(|id| element(&world, id) != "H"),
            "no handle atom is left"
        );
        assert_eq!(world.n_bonds(), 3, "an unbranched chain of four atoms");
        let (elements, numbers, frags) = chain(&world);
        assert_eq!(elements, vec!["C", "C", "C", "O"]);
        assert_eq!(
            numbers,
            vec![BondNumber::Single, BondNumber::Double, BondNumber::Single]
        );
        assert_eq!(frags, vec![0, 0, 1, 1]);
    }
}
