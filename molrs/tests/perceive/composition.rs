//! Composition — perceived facts accumulate on the graph.
//!
//! Category: integration. Because every `find_*` is graph-in / graph-out, the
//! result of one is a legal input to the next, and the facts written by earlier
//! stages must survive the later clones.

use molrs::perceive::Perceive;

use crate::fixtures::{atom_int, benzene, bond_between, bond_int};

#[test]
fn test_rings_then_aromaticity_keeps_both_facts() {
    let (mol, carbons) = benzene();
    let p = Perceive::new();

    let g = p.find_rings(&mol);
    let g = p.find_aromaticity(&g);

    for &c in &carbons {
        assert_eq!(
            atom_int(&g, c, "is_in_ring"),
            Some(1),
            "ring fact from stage 1 must survive stage 2"
        );
        assert_eq!(
            atom_int(&g, c, "is_aromatic"),
            Some(1),
            "aromaticity fact from stage 2 must be present"
        );
    }
}

#[test]
fn test_aromaticity_then_rings_keeps_both_facts() {
    let (mol, carbons) = benzene();
    let p = Perceive::new();

    let g = p.find_aromaticity(&mol);
    let g = p.find_rings(&g);

    for &c in &carbons {
        assert_eq!(
            atom_int(&g, c, "is_aromatic"),
            Some(1),
            "aromaticity fact from stage 1 must survive stage 2"
        );
        assert_eq!(
            atom_int(&g, c, "is_in_ring"),
            Some(1),
            "ring perception is bond-order independent: still 1 after 1.5 orders"
        );
    }
}

#[test]
fn test_three_stage_chain_keeps_all_bond_facts() {
    let (mol, carbons) = benzene();
    let p = Perceive::new();

    let g = p.find_rings(&mol);
    let g = p.find_aromaticity(&g);
    let g = p.find_rotatable(&g);

    for i in 0..6 {
        let bid = bond_between(&g, carbons[i], carbons[(i + 1) % 6]);
        assert_eq!(bond_int(&g, bid, "is_in_ring"), Some(1), "bond {i}: ring");
        assert_eq!(
            bond_int(&g, bid, "is_aromatic"),
            Some(1),
            "bond {i}: aromatic"
        );
        assert_eq!(
            bond_int(&g, bid, "is_rotatable"),
            Some(0),
            "bond {i}: not rotatable"
        );
    }
}
