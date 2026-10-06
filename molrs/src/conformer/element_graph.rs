//! The [`ElementGraph`] bound behind the generic
//! [`Conformer::generate`](super::Conformer::generate).
//!
//! An embedding is only defined for a graph whose nodes are chemical elements:
//! ETKDG reads `element` for bond-length estimation, ring geometry and
//! force-field selection. This module names that requirement once, as a trait,
//! so the single entry point can hand the caller back the same typed world it
//! was given — an [`Atomistic`] in, an `Atomistic` out, with its ports and
//! `frag_id` properties intact.

use molrs::error::MolRsError;
use molrs::system::Atomistic;
use molrs::system::MolGraph;

/// A typed wrapper over a [`MolGraph`] whose every node carries an `element`.
///
/// # Contract
///
/// - [`try_from_molgraph`](ElementGraph::try_from_molgraph) promotes a
///   [`MolGraph`] only when **every** node carries the `element` property, and
///   otherwise fails naming the first node that does not. It is the inherent
///   promotion of the implementing type, so it also resolves that type's
///   relation kinds by name.
/// - [`as_molgraph`](ElementGraph::as_molgraph) borrows the very graph the
///   value was built from — a pure forward to the inherent accessor, never a
///   rebuild, a copy or a filtered view.
///
/// Together the two are exactly what a generic embedding needs: read the graph
/// out of one typed world, and put the embedded graph back into the same one.
/// The trait carries no post-embed hook; relabelling atoms a pipeline added is
/// the caller's step, composed at the call site (see the module example on
/// [`conformer`](crate::conformer)).
///
/// # Implementors
///
/// [`Atomistic`], and no one else.
/// [`CoarseGrain`](crate::system::CoarseGrain) owns the same pair
/// of inherent methods and deliberately does **not** implement this trait: its
/// nodes carry bead types, not elements, so every `element` lookup the
/// embedding performs would be a lookup for a property that a coarse-grained
/// graph has no reason to hold. Excluded by contract, not by oversight.
///
/// # Placement
///
/// The trait lives beside its only consumer rather than in `core`, which has
/// no use for it. The in-repo precedent is `io::reader::FromFrame`
/// (`molrs/src/io/reader.rs:69`): likewise a `Sized` trait with a
/// `Self`-returning constructor, defined in the module of the one generic
/// function that consumes it (`FrameReader::read_as<T: FromFrame>`). Both are
/// static-dispatch bounds, never used as trait objects.
pub trait ElementGraph: Sized {
    /// Borrow the underlying [`MolGraph`].
    fn as_molgraph(&self) -> &MolGraph;

    /// Promote a [`MolGraph`] back into this type.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when a node carries no `element`, and
    /// when the graph already spells one of the type's standard relation kinds
    /// at a conflicting arity.
    fn try_from_molgraph(mol: MolGraph) -> Result<Self, MolRsError>;
}

impl ElementGraph for Atomistic {
    fn as_molgraph(&self) -> &MolGraph {
        Atomistic::as_molgraph(self)
    }

    fn try_from_molgraph(mol: MolGraph) -> Result<Self, MolRsError> {
        Atomistic::try_from_molgraph(mol)
    }
}

#[cfg(test)]
mod tests {
    use super::ElementGraph;
    use molrs::error::MolRsError;
    use molrs::system::Atomistic;
    use molrs::system::BondNumber;
    use molrs::system::NodeId;
    use molrs::system::PortKind;
    use molrs::system::{Atom, MolGraph};

    /// `H–C–C–H` with a `$` port on each C–H valence and a `frag_id` on every
    /// atom: four atoms, three bonds, two ports, two distinct fragment labels
    /// (so a per-atom check cannot pass by broadcasting a single label).
    fn ported_pair() -> (Atomistic, [NodeId; 4]) {
        let mut frag = Atomistic::new();
        let c0 = frag.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let c1 = frag.add_atom_xyz("C", 1.54, 0.0, 0.0);
        let h0 = frag.add_atom_bare("H");
        let h1 = frag.add_atom_bare("H");
        frag.add_bond(c0, c1).expect("both carbons are live");
        frag.add_bond(c0, h0).expect("both endpoints are live");
        frag.add_bond(c1, h1).expect("both endpoints are live");
        frag.add_port(c0, h0, PortKind::Symmetric, "A", BondNumber::Single)
            .expect("a bonded H handle on its anchor is a legal port");
        frag.add_port(c1, h1, PortKind::Symmetric, "B", BondNumber::Single)
            .expect("a bonded H handle on its anchor is a legal port");
        for (atom, id) in [(c0, 7), (h0, 7), (c1, 9), (h1, 9)] {
            frag.set_frag_id(atom, id).expect("the atom is live");
        }
        (frag, [c0, c1, h0, h1])
    }

    /// The conversions `Conformer::generate` performs around the embed, with
    /// no embed in between: `Atomistic -> MolGraph -> Atomistic`.
    fn round_trip(frag: &Atomistic) -> Atomistic {
        <Atomistic as ElementGraph>::try_from_molgraph(
            <Atomistic as ElementGraph>::as_molgraph(frag).clone(),
        )
        .expect("every node carries an element")
    }

    // ---- round trip, no ETKDG ---------------------------------------------

    #[test]
    fn round_trip_preserves_port_count() {
        let (frag, _atoms) = ported_pair();
        assert_eq!(frag.n_ports(), 2);
        assert_eq!(
            round_trip(&frag).n_ports(),
            2,
            "the ports kind survives the promotion at each end"
        );
    }

    #[test]
    fn round_trip_preserves_frag_ids() {
        let (frag, atoms) = ported_pair();
        let back = round_trip(&frag);
        assert_eq!(back.n_atoms(), 4);
        for atom in atoms {
            assert_eq!(
                back.frag_id(atom),
                frag.frag_id(atom),
                "frag_id of {atom:?} changed across the round trip"
            );
        }
        assert_eq!(back.frag_id(atoms[0]), Some(7), "C0 keeps its own label");
        assert_eq!(back.frag_id(atoms[1]), Some(9), "C1 keeps its own label");
    }

    #[test]
    fn round_trip_preserves_bond_count() {
        let (frag, _atoms) = ported_pair();
        assert_eq!(frag.n_bonds(), 3);
        assert_eq!(
            round_trip(&frag).n_bonds(),
            3,
            "a port is never counted as a bond"
        );
    }

    #[test]
    fn round_trip_keeps_bonds_and_ports_distinct() {
        let (frag, _atoms) = ported_pair();
        let back = round_trip(&frag);
        let bonds = back.kind_id("bonds").expect("'bonds' is registered");
        let ports = back
            .kind_id("ports")
            .expect("'ports' survives the round trip");
        assert_ne!(bonds, ports, "the two kinds must not alias");
        assert_eq!(back.n_relations(bonds), 3);
        assert_eq!(back.n_relations(ports), 2);
    }

    // ---- the bound itself --------------------------------------------------

    fn assert_bound<M: ElementGraph>() {}

    #[test]
    fn atomistic_satisfies_the_bound() {
        assert_bound::<Atomistic>();
    }

    #[test]
    fn as_molgraph_through_the_trait_borrows_the_same_graph() {
        let (frag, _atoms) = ported_pair();
        assert!(
            std::ptr::eq(
                <Atomistic as ElementGraph>::as_molgraph(&frag),
                frag.as_molgraph()
            ),
            "the trait method is a pure forward to the inherent borrow"
        );
    }

    #[test]
    fn try_from_molgraph_through_the_trait_rejects_node_without_element() {
        let mut graph = MolGraph::new();
        graph.add_node_with(Atom::new()).expect("fixture node");
        let err = <Atomistic as ElementGraph>::try_from_molgraph(graph)
            .expect_err("a node without 'element' is not an element-bearing graph");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }
}
