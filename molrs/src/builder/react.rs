//! Port chemistry: [`Reacter`] joins two ports of one world fragment, and
//! [`PortReacter`] is the port-driven implementation.
//!
//! A **port** ([`Port`](crate::system::fragment::Port)) is an open attachment
//! point on a fragment: an **anchor** atom that will gain the new bond, and a
//! **handle** atom bonded to it (usually a capping hydrogen) that stands in
//! for the partner until the join. Joining two ports removes each handle
//! together with everything hanging off it (its *branch*, the leaving group)
//! and bonds the two anchors directly, as a condensation reaction would.
//!
//! `PortReacter::link` validates before it writes. It pairs ports only through
//! [`Port::accepts`](crate::system::fragment::Port::accepts), deletes each
//! port's handle branch, bonds the two anchors with the port order, and folds
//! the deleted atoms' partial charge (in e) onto their anchor:
//! `q_a' = q_a + Σ_{i∈D} q_i`, where `D` is the handle and its branch (the
//! practice of pysimm's `random_walk` polymer builder). The sum of all charges,
//! and the sum within each `frag_id` unit, is therefore unchanged by a link.

use std::collections::BTreeSet;
use std::fmt;

use crate::error::MolRsError;
use crate::store::keys;
use crate::system::atomistic::{AtomId, BondId};
use crate::system::fragment::{Fragment, Port, PortId};

/// Joins two ports of one world [`Fragment`] into a bond.
///
/// An implementor consumes both ports: after a successful [`link`](Self::link)
/// the two anchors are bonded and neither port remains.
pub trait Reacter: Send + Sync {
    /// Join port `a` to port `b` of `world` and return the new anchor–anchor
    /// bond.
    ///
    /// # Errors
    ///
    /// A [`ReactError`] naming why the pair cannot be joined.
    fn link(&self, world: &mut Fragment, a: PortId, b: PortId) -> Result<BondId, ReactError>;

    /// Join every pair in order and return the new bonds in pair order.
    ///
    /// The default is a loop over [`link`](Self::link), so it is correct for
    /// every implementor. It is the one batch call the `Assembler` makes and
    /// the one call the Python adaptor crosses.
    ///
    /// # Errors
    ///
    /// The first failing pair, as a [`PairError`] carrying its index in
    /// `pairs` (`Some(i)`). The pairs before it stay linked. An override that
    /// cannot tell which pair failed reports `None` rather than an index.
    fn link_many(
        &self,
        world: &mut Fragment,
        pairs: &[(PortId, PortId)],
    ) -> Result<Vec<BondId>, PairError> {
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

/// The pair a [`Reacter::link_many`] batch failed on, and why.
#[derive(Debug)]
pub struct PairError {
    /// Index of the failing pair in the batch; `None` when the implementor
    /// could not name it (e.g. a batch reacter outside this crate).
    pub pair: Option<usize>,
    /// Why that pair was refused.
    pub error: ReactError,
}

impl fmt::Display for PairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.pair {
            Some(i) => write!(f, "pair {i}: {}", self.error),
            None => write!(f, "a pair: {}", self.error),
        }
    }
}

impl std::error::Error for PairError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Why a [`Reacter`] refused to join two ports.
#[derive(Debug)]
pub enum ReactError {
    /// A port did not read back (stale id or malformed descriptor).
    Port(MolRsError),
    /// The port reads back, but its anchor–handle bond no longer exists: the
    /// bond was removed through the inner graph after
    /// [`Fragment::add_port`] checked it, so the port has no leaving group.
    StalePort {
        /// The stale port.
        port: PortId,
    },
    /// [`Port::accepts`] refuses the pair: the kinds are not complements, or
    /// the labels or orders differ.
    Incompatible {
        /// The first port.
        a: PortId,
        /// The second port.
        b: PortId,
    },
    /// Both ports sit on one anchor, and a bond cannot join an atom to itself.
    SameAnchor {
        /// The first port.
        a: PortId,
        /// The second port.
        b: PortId,
    },
    /// The two anchors are already bonded; linking them would add a duplicate
    /// bond (reachable through a `FragGraph` with two edges between one node
    /// pair).
    AlreadyBonded {
        /// The first port's anchor.
        a: AtomId,
        /// The second port's anchor.
        b: AtomId,
    },
    /// The port's branch — its handle's component once the anchor–handle bond
    /// is cut — contains the port's own anchor: the handle sits on a ring, so
    /// deleting the branch would delete the anchor.
    BranchReachesAnchor {
        /// The offending port.
        port: PortId,
    },
    /// The two branches share an atom, or one port's anchor lies in the other
    /// port's branch.
    BranchesOverlap,
    /// Charge is present on only part of `{anchor} ∪ branch`, so folding it
    /// would invent or lose charge.
    OneSidedCharge {
        /// The anchor of the offending port.
        anchor: AtomId,
    },
    /// The graph refused a read during validation.
    Graph(MolRsError),
    /// A failure reported by an implementor outside this crate. A Python
    /// exception raised by a Python-side reacter keeps only its message and
    /// surfaces back in Python as `ValueError`.
    Other(String),
}

impl fmt::Display for ReactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Port(e) => write!(f, "port does not read back: {e}"),
            Self::StalePort { port } => write!(
                f,
                "port {port:?} is stale: its anchor–handle bond no longer exists"
            ),
            Self::Incompatible { a, b } => {
                write!(f, "ports {a:?} and {b:?} are not compatible")
            }
            Self::SameAnchor { a, b } => write!(
                f,
                "ports {a:?} and {b:?} share one anchor; a bond cannot join an atom to itself"
            ),
            Self::AlreadyBonded { a, b } => {
                write!(f, "anchors {a:?} and {b:?} are already bonded")
            }
            Self::BranchReachesAnchor { port } => {
                write!(f, "port {port:?}'s handle branch reaches its own anchor")
            }
            Self::BranchesOverlap => write!(f, "the two ports' handle branches overlap"),
            Self::OneSidedCharge { anchor } => write!(
                f,
                "charge is present on only part of anchor {anchor:?} and its handle branch"
            ),
            Self::Graph(e) => write!(f, "graph refused a read: {e}"),
            Self::Other(msg) => write!(f, "link failed: {msg}"),
        }
    }
}

impl std::error::Error for ReactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Port(e) | Self::Graph(e) => Some(e),
            _ => None,
        }
    }
}

/// The port-driven [`Reacter`].
///
/// [`link`](Reacter::link) validates both ports before it writes anything, so
/// a refusal leaves the world unchanged. It refuses, in this order:
///
/// 1. a port that does not read back → [`ReactError::Port`];
/// 2. a port whose anchor–handle bond is gone → [`ReactError::StalePort`];
/// 3. `!a.accepts(&b)` → [`ReactError::Incompatible`];
/// 4. two ports on one anchor → [`ReactError::SameAnchor`];
/// 5. anchors already bonded → [`ReactError::AlreadyBonded`];
/// 6. a branch containing its own anchor → [`ReactError::BranchReachesAnchor`];
/// 7. overlapping branches, or an anchor inside the other branch →
///    [`ReactError::BranchesOverlap`];
/// 8. charge on only part of `{anchor} ∪ branch` →
///    [`ReactError::OneSidedCharge`] (an atom without `charge` counts as
///    absent; a side with no charge at all is fine).
///
/// A port's **branch** is its [`Fragment::leaving_group`]. It writes by
/// folding each branch's charge (e) onto its anchor, `q_a' = q_a + Σ_{i∈D} q_i`
/// with `D` the branch, then removing both branches (their ports go with them)
/// and bonding the two anchors, classed from the port order through
/// [`BondNumber::implied_type`](crate::system::bond::BondNumber::implied_type)
/// and written with [`Fragment::set_bond_class`]. The folding puts the whole
/// leaving-group charge on the one anchor, as pysimm's `random_walk` does,
/// rather than spreading it as AMBER's `prepgen` does; it conserves total and
/// per-`frag_id` charge.
///
/// # Panics
///
/// A write that fails after validation passed is a broken invariant of this
/// type (every atom written is live, the removed set is duplicate-free and
/// leaves both anchors, and the new bond is fresh), so it panics rather than
/// returning a half-applied edit as an error.
#[derive(Debug, Clone, Copy, Default)]
pub struct PortReacter;

impl Reacter for PortReacter {
    fn link(&self, world: &mut Fragment, a: PortId, b: PortId) -> Result<BondId, ReactError> {
        // ---- validate: nothing is written until every check passes ----
        let pa = world.port(a).map_err(ReactError::Port)?;
        let pb = world.port(b).map_err(ReactError::Port)?;
        // `leaving_group`'s only error is a missing anchor–handle bond.
        let branch_a = world
            .leaving_group(&pa)
            .map_err(|_| ReactError::StalePort { port: a })?;
        let branch_b = world
            .leaving_group(&pb)
            .map_err(|_| ReactError::StalePort { port: b })?;
        if !pa.accepts(&pb) {
            return Err(ReactError::Incompatible { a, b });
        }
        if pa.anchor == pb.anchor {
            return Err(ReactError::SameAnchor { a, b });
        }
        if world.is_bonded(pa.anchor, pb.anchor) {
            return Err(ReactError::AlreadyBonded {
                a: pa.anchor,
                b: pb.anchor,
            });
        }
        if branch_a.contains(&pa.anchor) {
            return Err(ReactError::BranchReachesAnchor { port: a });
        }
        if branch_b.contains(&pb.anchor) {
            return Err(ReactError::BranchReachesAnchor { port: b });
        }
        if !branch_a.is_disjoint(&branch_b)
            || branch_a.contains(&pb.anchor)
            || branch_b.contains(&pa.anchor)
        {
            return Err(ReactError::BranchesOverlap);
        }
        let folded_a = Self::folded_charge(world, &pa, &branch_a)?;
        let folded_b = Self::folded_charge(world, &pb, &branch_b)?;

        // ---- write ----
        const VALIDATED: &str = "validated before the first write";
        for (anchor, folded) in [(pa.anchor, folded_a), (pb.anchor, folded_b)] {
            if let Some(q) = folded {
                world.set_node(anchor, keys::CHARGE, q).expect(VALIDATED);
            }
        }
        let doomed: Vec<AtomId> = branch_a.iter().chain(&branch_b).copied().collect();
        world.remove_nodes(&doomed).expect(VALIDATED);
        let bond = world.add_bond(pa.anchor, pb.anchor).expect(VALIDATED);
        world
            .set_bond_class(bond, pa.order.implied_type(), pa.order)
            .expect(VALIDATED);
        Ok(bond)
    }
}

impl PortReacter {
    /// The anchor's charge after folding `branch` onto it: `None` when no atom
    /// of `{anchor} ∪ branch` carries a charge, `Some(q_a + Σ q_i)` when every
    /// atom does.
    fn folded_charge(
        world: &Fragment,
        port: &Port,
        branch: &BTreeSet<AtomId>,
    ) -> Result<Option<f64>, ReactError> {
        let charge = |atom: AtomId| -> Result<Option<f64>, ReactError> {
            world
                .get_node(atom)
                .map(|props| props.get_f64(keys::CHARGE))
                .map_err(ReactError::Graph)
        };
        let anchor_q = charge(port.anchor)?;
        let mut present = usize::from(anchor_q.is_some());
        let mut branch_q = 0.0;
        for &atom in branch {
            if let Some(q) = charge(atom)? {
                present += 1;
                branch_q += q;
            }
        }
        match anchor_q {
            _ if present == 0 => Ok(None),
            Some(q) if present == branch.len() + 1 => Ok(Some(q + branch_q)),
            _ => Err(ReactError::OneSidedCharge {
                anchor: port.anchor,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{PairError, PortReacter, ReactError, Reacter};
    use crate::store::keys;
    use crate::system::atomistic::AtomId;
    use crate::system::bond::BondNumber;
    use crate::system::fragment::{Fragment, Port, PortId, PortKind};

    // ---- fixtures ----------------------------------------------------------
    //
    // Every fixture is one hand-built world `Fragment` of 2–5 atoms per unit.
    // Charges are dyadic rationals (k / 2^n), so every sum below is exact in
    // binary floating point and the goldens are compared with `==`.

    /// Add an atom of `element` carrying `charge` and (when given) `frag_id`.
    fn atom(world: &mut Fragment, element: &str, charge: Option<f64>, frag: u32) -> AtomId {
        let a = world.add_atom_bare(element);
        if let Some(q) = charge {
            world
                .set_node(a, keys::CHARGE, q)
                .expect("fixture atom takes a charge");
        }
        world.set_frag_id(a, frag).expect("fixture frag_id fits");
        a
    }

    /// One unit: an `anchor` atom bonded to an H handle, with a port
    /// `(anchor, H)` of `kind`, label `"p"`, order `Single`.
    /// Returns `(anchor, handle, port)`.
    fn h_unit(
        world: &mut Fragment,
        anchor_element: &str,
        anchor_q: Option<f64>,
        handle_q: Option<f64>,
        kind: PortKind,
        frag: u32,
    ) -> (AtomId, AtomId, PortId) {
        let anchor = atom(world, anchor_element, anchor_q, frag);
        let handle = atom(world, "H", handle_q, frag);
        world.add_bond(anchor, handle).expect("fixture bond");
        let port = world
            .add_port(anchor, handle, kind, "p", BondNumber::Single)
            .expect("a bonded H handle is a legal port");
        (anchor, handle, port)
    }

    fn charge(world: &Fragment, a: AtomId) -> f64 {
        world
            .get_node(a)
            .expect("live atom")
            .get_f64(keys::CHARGE)
            .expect("atom carries a charge")
    }

    fn total_charge(world: &Fragment) -> f64 {
        world
            .nodes()
            .map(|(_, atom)| atom.get_f64(keys::CHARGE).unwrap_or(0.0))
            .sum()
    }

    /// Per-`frag_id` charge sums (atoms without a charge count as 0).
    fn frag_sums(world: &Fragment) -> BTreeMap<u32, f64> {
        let mut sums = BTreeMap::new();
        for id in world.node_ids().collect::<Vec<_>>() {
            let frag = world.frag_id(id).expect("fixture atoms carry frag_id");
            let q = world
                .get_node(id)
                .expect("live atom")
                .get_f64(keys::CHARGE)
                .unwrap_or(0.0);
            *sums.entry(frag).or_insert(0.0) += q;
        }
        sums
    }

    fn frag_count(world: &Fragment, frag: u32) -> usize {
        world
            .node_ids()
            .filter(|&id| world.frag_id(id) == Some(frag))
            .count()
    }

    fn is_bonded(world: &Fragment, a: AtomId, b: AtomId) -> bool {
        world.bonds().any(|(_, bond)| {
            let n = bond.nodes.as_slice();
            (n[0] == a && n[1] == b) || (n[0] == b && n[1] == a)
        })
    }

    /// Everything a refused link must leave untouched.
    #[derive(Debug, PartialEq)]
    struct Snapshot {
        atoms: Vec<(AtomId, Option<f64>)>,
        bonds: Vec<Vec<AtomId>>,
        ports: Vec<Port>,
    }

    fn snapshot(world: &Fragment) -> Snapshot {
        Snapshot {
            atoms: world
                .nodes()
                .map(|(id, atom)| (id, atom.get_f64(keys::CHARGE)))
                .collect(),
            bonds: world.bonds().map(|(_, bond)| bond.nodes.to_vec()).collect(),
            ports: world
                .ports()
                .collect::<Vec<_>>()
                .into_iter()
                .map(|id| world.port(id).expect("fixture port reads back"))
                .collect(),
        }
    }

    /// Run a link that must be refused, assert the world is unchanged, and
    /// return the refusal.
    fn refuse(world: &mut Fragment, a: PortId, b: PortId) -> ReactError {
        let before = snapshot(world);
        let err = PortReacter
            .link(world, a, b)
            .expect_err("this pair must be refused");
        assert_eq!(
            snapshot(world),
            before,
            "a refusal leaves atoms, bonds, charges and ports unchanged ({err:?})"
        );
        err
    }

    // ---- charge-folding goldens C1–C5 ---------------------------------------

    #[test]
    fn c1_single_h_handle_folds_onto_anchor() {
        let mut world = Fragment::new();
        let (a0, h0, p0) = h_unit(&mut world, "C", Some(-0.25), Some(0.125), PortKind::Left, 0);
        let (a1, h1, p1) = h_unit(
            &mut world,
            "C",
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            1,
        );

        PortReacter
            .link(&mut world, p0, p1)
            .expect("complementary ports link");

        assert_eq!(charge(&world, a0), -0.125, "C1: -0.25 + 0.125");
        assert_eq!(charge(&world, a1), 0.0, "partner: -0.125 + 0.125");
        assert!(world.get_node(h0).is_err(), "the handle H is removed");
        assert!(
            world.get_node(h1).is_err(),
            "the partner handle H is removed"
        );
        assert_eq!(world.n_atoms(), 2);
        assert_eq!(world.n_ports(), 0, "both ports go with their handles");
        assert_eq!(world.n_bonds(), 1);
        assert!(is_bonded(&world, a0, a1), "the anchors are bonded");
    }

    /// Regression example (spec assembly-06, golden C2; hard-coded golden:
    /// C +0.5, O −0.625, H +0.375 → C +0.25 with the 2-atom OH branch
    /// removed). The handle is the heavy O, the branch is {O, H}.
    #[test]
    fn c2_oh_branch_folds_onto_carbon() {
        let mut world = Fragment::new();
        let c = atom(&mut world, "C", Some(0.5), 0);
        let o = atom(&mut world, "O", Some(-0.625), 0);
        let h = atom(&mut world, "H", Some(0.375), 0);
        world.add_bond(c, o).unwrap();
        world.add_bond(o, h).unwrap();
        let p0 = world
            .add_port(c, o, PortKind::Left, "p", BondNumber::Single)
            .expect("a bonded O leaving group is a legal handle");
        let (n, nh, p1) = h_unit(
            &mut world,
            "N",
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            1,
        );

        PortReacter
            .link(&mut world, p0, p1)
            .expect("complementary ports link");

        assert_eq!(charge(&world, c), 0.25, "C2: 0.5 - 0.625 + 0.375");
        assert_eq!(
            frag_count(&world, 0),
            1,
            "the OH branch's 2 atoms are removed"
        );
        assert!(world.get_node(o).is_err(), "O is removed");
        assert!(world.get_node(h).is_err(), "the hydroxyl H is removed");
        assert!(world.get_node(nh).is_err(), "the partner handle is removed");
        assert!(is_bonded(&world, c, n), "C is bonded to the partner anchor");
    }

    #[test]
    fn c3_two_h_handles_both_linked() {
        let mut world = Fragment::new();
        let a = atom(&mut world, "C", Some(-0.5), 0);
        let h1 = atom(&mut world, "H", Some(0.125), 0);
        let h2 = atom(&mut world, "H", Some(0.125), 0);
        world.add_bond(a, h1).unwrap();
        world.add_bond(a, h2).unwrap();
        let pa1 = world
            .add_port(a, h1, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let pa2 = world
            .add_port(a, h2, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let (b1, _, pb1) = h_unit(
            &mut world,
            "C",
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            1,
        );
        let (b2, _, pb2) = h_unit(
            &mut world,
            "C",
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            2,
        );

        let bonds = PortReacter
            .link_many(&mut world, &[(pa1, pb1), (pa2, pb2)])
            .expect("both pairs link");

        assert_eq!(bonds.len(), 2, "one new bond per pair");
        assert_eq!(charge(&world, a), -0.25, "C3: -0.5 + 0.125 + 0.125");
        assert!(is_bonded(&world, a, b1));
        assert!(is_bonded(&world, a, b2));
        assert_eq!(world.n_ports(), 0);
        assert_eq!(world.n_atoms(), 3);
    }

    #[test]
    fn c4_neutral_units_stay_neutral_per_frag_id() {
        let mut world = Fragment::new();
        // unit 0: C(-0.25) – C anchor(+0.125) – H handle(+0.125); sum 0
        let c0 = atom(&mut world, "C", Some(-0.25), 0);
        let (a0, _, p0) = h_unit(&mut world, "C", Some(0.125), Some(0.125), PortKind::Left, 0);
        world.add_bond(c0, a0).unwrap();
        // unit 1: O anchor(-0.5) – H handle(+0.25), plus H(+0.25) on O; sum 0
        let (a1, _, p1) = h_unit(&mut world, "O", Some(-0.5), Some(0.25), PortKind::Right, 1);
        let h = atom(&mut world, "H", Some(0.25), 1);
        world.add_bond(a1, h).unwrap();

        PortReacter.link(&mut world, p0, p1).expect("link");

        assert_eq!(total_charge(&world), 0.0, "C4: total stays 0");
        let sums = frag_sums(&world);
        assert_eq!(sums.get(&0), Some(&0.0), "frag 0 sums to 0");
        assert_eq!(sums.get(&1), Some(&0.0), "frag 1 sums to 0");
    }

    #[test]
    fn c5_integer_charged_units_keep_minus_one_per_frag_id() {
        let mut world = Fragment::new();
        // unit 0: O anchor(-1.25) + H handle(+0.25); sum -1
        let (_, _, p0) = h_unit(&mut world, "O", Some(-1.25), Some(0.25), PortKind::Left, 0);
        // unit 1: C anchor(-0.5) + O(-0.625) + H handle(+0.125); sum -1
        let (a1, _, p1) = h_unit(&mut world, "C", Some(-0.5), Some(0.125), PortKind::Right, 1);
        let o = atom(&mut world, "O", Some(-0.625), 1);
        world.add_bond(a1, o).unwrap();

        PortReacter.link(&mut world, p0, p1).expect("link");

        let sums = frag_sums(&world);
        assert_eq!(sums.get(&0), Some(&-1.0), "C5: frag 0 sums to -1");
        assert_eq!(sums.get(&1), Some(&-1.0), "C5: frag 1 sums to -1");
        assert_eq!(total_charge(&world), -2.0);
    }

    // ---- the new bond --------------------------------------------------------

    #[test]
    fn link_returns_the_anchor_bond_classed_from_the_port_order() {
        let mut world = Fragment::new();
        let a = atom(&mut world, "C", None, 0);
        let ha = atom(&mut world, "H", None, 0);
        let b = atom(&mut world, "C", None, 1);
        let hb = atom(&mut world, "H", None, 1);
        world.add_bond(a, ha).unwrap();
        world.add_bond(b, hb).unwrap();
        let pa = world
            .add_port(a, ha, PortKind::Left, "p", BondNumber::Double)
            .unwrap();
        let pb = world
            .add_port(b, hb, PortKind::Right, "p", BondNumber::Double)
            .unwrap();

        let bid = PortReacter.link(&mut world, pa, pb).expect("link");

        let bonds = world.kind_id("bonds").expect("'bonds' registered");
        let bond = world
            .get_relation(bonds, bid)
            .expect("the returned bond is live");
        let mut ends = bond.nodes.to_vec();
        ends.sort();
        let mut expected = vec![a, b];
        expected.sort();
        assert_eq!(ends, expected, "the returned bond joins the two anchors");
        assert_eq!(
            BondNumber::from_prop(bond.props.get(keys::BOND_NUMBER)),
            BondNumber::Double,
            "the bond order comes from the port order"
        );
    }

    // ---- refusals (each leaves the world unchanged) ---------------------------

    #[test]
    fn same_kind_ports_are_incompatible() {
        let mut world = Fragment::new();
        let (_, _, p0) = h_unit(&mut world, "C", Some(-0.25), Some(0.25), PortKind::Left, 0);
        let (_, _, p1) = h_unit(&mut world, "C", Some(-0.25), Some(0.25), PortKind::Left, 1);
        assert!(
            !world.port(p0).unwrap().accepts(&world.port(p1).unwrap()),
            "fixture: Port::accepts refuses `<`/`<`"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, ReactError::Incompatible { a, b } if a == p0 && b == p1),
            "{err:?}"
        );
    }

    #[test]
    fn label_mismatch_is_incompatible_through_port_accepts() {
        let mut world = Fragment::new();
        let (_, _, p0) = h_unit(&mut world, "C", None, None, PortKind::Left, 0);
        let b = atom(&mut world, "C", None, 1);
        let hb = atom(&mut world, "H", None, 1);
        world.add_bond(b, hb).unwrap();
        let p1 = world
            .add_port(b, hb, PortKind::Right, "q", BondNumber::Single)
            .unwrap();
        assert!(
            !world.port(p0).unwrap().accepts(&world.port(p1).unwrap()),
            "fixture: Port::accepts refuses unequal labels"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(matches!(err, ReactError::Incompatible { .. }), "{err:?}");
    }

    #[test]
    fn ring_bonded_handle_reaches_its_anchor() {
        let mut world = Fragment::new();
        // Triangle C0–C1–C2–C0; the port handle C1 sits on the ring, so its
        // branch after cutting C0–C1 still reaches C0.
        let c0 = atom(&mut world, "C", None, 0);
        let c1 = atom(&mut world, "C", None, 0);
        let c2 = atom(&mut world, "C", None, 0);
        world.add_bond(c0, c1).unwrap();
        world.add_bond(c1, c2).unwrap();
        world.add_bond(c2, c0).unwrap();
        let p0 = world
            .add_port(c0, c1, PortKind::Left, "p", BondNumber::Single)
            .expect("a bonded heavy handle is a legal port");
        let (_, _, p1) = h_unit(&mut world, "C", None, None, PortKind::Right, 1);

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, ReactError::BranchReachesAnchor { port } if port == p0),
            "{err:?}"
        );
    }

    #[test]
    fn shared_handle_branches_overlap() {
        let mut world = Fragment::new();
        // A – X – B, with X the handle of both ports.
        let a = atom(&mut world, "C", None, 0);
        let x = atom(&mut world, "O", None, 0);
        let b = atom(&mut world, "C", None, 1);
        world.add_bond(a, x).unwrap();
        world.add_bond(x, b).unwrap();
        let pa = world
            .add_port(a, x, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let pb = world
            .add_port(b, x, PortKind::Right, "p", BondNumber::Single)
            .unwrap();

        let err = refuse(&mut world, pa, pb);

        assert!(matches!(err, ReactError::BranchesOverlap), "{err:?}");
    }

    #[test]
    fn anchor_inside_the_other_branch_is_overlap() {
        let mut world = Fragment::new();
        // H1 – A – Y – B: port a = (A, H1), port b = (B, Y). b's branch
        // {Y, A, H1} holds anchor A.
        let h1 = atom(&mut world, "H", None, 0);
        let a = atom(&mut world, "C", None, 0);
        let y = atom(&mut world, "C", None, 1);
        let b = atom(&mut world, "C", None, 1);
        world.add_bond(h1, a).unwrap();
        world.add_bond(a, y).unwrap();
        world.add_bond(y, b).unwrap();
        let pa = world
            .add_port(a, h1, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let pb = world
            .add_port(b, y, PortKind::Right, "p", BondNumber::Single)
            .unwrap();

        let err = refuse(&mut world, pa, pb);

        assert!(matches!(err, ReactError::BranchesOverlap), "{err:?}");
    }

    #[test]
    fn charged_anchor_with_uncharged_handle_is_one_sided() {
        let mut world = Fragment::new();
        let (a0, _, p0) = h_unit(&mut world, "C", Some(-0.25), None, PortKind::Left, 0);
        let (_, _, p1) = h_unit(
            &mut world,
            "C",
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            1,
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, ReactError::OneSidedCharge { anchor } if anchor == a0),
            "{err:?}"
        );
    }

    #[test]
    fn charged_handle_with_uncharged_anchor_is_one_sided() {
        let mut world = Fragment::new();
        let (_, _, p0) = h_unit(
            &mut world,
            "C",
            Some(-0.125),
            Some(0.125),
            PortKind::Left,
            0,
        );
        let (a1, _, p1) = h_unit(&mut world, "C", None, Some(0.125), PortKind::Right, 1);

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, ReactError::OneSidedCharge { anchor } if anchor == a1),
            "{err:?}"
        );
    }

    /// Amended 2026-09-26: two compatible ports whose anchors are already
    /// bonded would otherwise gain a duplicate anchor–anchor bond.
    #[test]
    fn link_refuses_anchors_that_are_already_bonded() {
        let mut world = Fragment::new();
        let (a0, _, p0) = h_unit(&mut world, "C", None, None, PortKind::Left, 0);
        let (a1, _, p1) = h_unit(&mut world, "C", None, None, PortKind::Right, 1);
        world
            .add_bond(a0, a1)
            .expect("fixture: the anchors are bonded");
        assert!(
            world.port(p0).unwrap().accepts(&world.port(p1).unwrap()),
            "fixture: the ports themselves are compatible"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, ReactError::AlreadyBonded { a, b } if a == a0 && b == a1),
            "{err:?}"
        );
    }

    /// Amended 2026-09-26: two compatible ports on one anchor are their own
    /// refusal, not a `Graph(Validation)`.
    #[test]
    fn link_refuses_two_ports_on_one_anchor() {
        let mut world = Fragment::new();
        let c = atom(&mut world, "C", None, 0);
        let h1 = atom(&mut world, "H", None, 0);
        let h2 = atom(&mut world, "H", None, 0);
        world.add_bond(c, h1).unwrap();
        world.add_bond(c, h2).unwrap();
        let p0 = world
            .add_port(c, h1, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let p1 = world
            .add_port(c, h2, PortKind::Right, "p", BondNumber::Single)
            .unwrap();
        assert!(
            world.port(p0).unwrap().accepts(&world.port(p1).unwrap()),
            "fixture: the ports themselves are compatible"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, ReactError::SameAnchor { a, b } if a == p0 && b == p1),
            "{err:?}"
        );
    }

    /// Amended 2026-09-26: the anchor–handle bond is checked only at
    /// `add_port`; removing it through the inner graph leaves a port that
    /// still reads back but is stale.
    #[test]
    fn link_refuses_a_port_whose_handle_bond_is_gone() {
        let mut world = Fragment::new();
        let (a0, h0, p0) = h_unit(&mut world, "C", None, None, PortKind::Left, 0);
        let (_, _, p1) = h_unit(&mut world, "C", None, None, PortKind::Right, 1);
        let bonds = world.kind_id("bonds").expect("'bonds' registered");
        let cut = world
            .neighbor_relations(a0)
            .find(|&(kind, _, other)| kind == bonds && other == h0)
            .map(|(_, rid, _)| rid)
            .expect("fixture: the anchor–handle bond exists");
        world
            .as_molgraph_mut()
            .remove_relation(bonds, cut)
            .expect("the inner graph removes the bond");
        assert!(
            world.port(p0).is_ok(),
            "fixture: the stale port still reads back"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, ReactError::StalePort { port } if port == p0),
            "{err:?}"
        );
    }

    // ---- batch ---------------------------------------------------------------

    #[test]
    fn link_many_reports_the_failing_pair_index() {
        let mut world = Fragment::new();
        let (_, _, p0) = h_unit(&mut world, "C", None, None, PortKind::Left, 0);
        let (_, _, p1) = h_unit(&mut world, "C", None, None, PortKind::Right, 1);
        let (_, _, p2) = h_unit(&mut world, "C", None, None, PortKind::Left, 2);
        let (_, _, p3) = h_unit(&mut world, "C", None, None, PortKind::Left, 3);
        let (_, _, p4) = h_unit(&mut world, "C", None, None, PortKind::Left, 4);
        let (_, _, p5) = h_unit(&mut world, "C", None, None, PortKind::Right, 5);

        let err = PortReacter
            .link_many(&mut world, &[(p0, p1), (p2, p3), (p4, p5)])
            .expect_err("the second pair is `<`/`<`");

        let PairError { pair, error } = err;
        assert_eq!(pair, Some(1), "the index of the failing pair");
        assert!(
            matches!(error, ReactError::Incompatible { a, b } if a == p2 && b == p3),
            "{error:?}"
        );
    }
}
