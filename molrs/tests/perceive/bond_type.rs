//! BCC bond-type perception — `Perceive::find_bond_types`.
//!
//! The bulk of this perception is pinned by the antechamber oracle in the `ff`
//! test target (`typifier::am1bcc_antechamber`, 323 bonds over 37 molecules). What
//! lives here is what that oracle *cannot* say:
//!
//! * the **aromatic promotion boundary** — the oracle has no biphenyl and no
//!   7-membered aromatic ring, so nothing there would notice if the boundary
//!   silently widened;
//! * the **delocalization** rule stated as itself, rather than as a charge;
//! * the two types the oracle never contains — 6 (reachable) and 11 (not);
//! * **determinism** under a permutation of the input, which is a property of the
//!   algorithm and cannot be a property of any single fixture.
//!
//! Fixtures are built in code. Coordinates are idealised and no assertion depends
//! on them — this perception reads connectivity, bond order and aromatic flags
//! only.

use molrs::perceive::Perceive;
use molrs::perceive::bond_type::BCC_BOND_TYPE;
use molrs::store::keys;
use molrs::{AtomId, Atomistic, BondId};

// ── fixtures ────────────────────────────────────────────────────────────────

/// A bond spec: `(i, j, order, aromatic)` over atom indices.
type BondSpec = (usize, usize, f64, bool);

/// Build a molecule from elements and bond specs. Coordinates are a line — this
/// perception never reads them.
fn mol(elements: &[&str], bonds: &[BondSpec]) -> (Atomistic, Vec<AtomId>) {
    let mut g = Atomistic::new();
    let ids: Vec<AtomId> = elements
        .iter()
        .enumerate()
        .map(|(i, el)| g.add_atom_xyz(el, i as f64, 0.0, 0.0))
        .collect();
    for (i, j, order, aromatic) in bonds {
        let bid = g.add_bond(ids[*i], ids[*j]).expect("add bond");
        g.set_bond_prop(bid, keys::ORDER, *order)
            .expect("set order");
        if *aromatic {
            g.set_bond_prop(bid, "is_aromatic", true)
                .expect("set aromatic");
        }
    }
    (g, ids)
}

/// The perceived BCC type of the bond joining `a` and `b`.
fn bond_type(g: &Atomistic, a: AtomId, b: AtomId) -> i32 {
    let bid = crate::fixtures::bond_between(g, a, b);
    g.get_bond(bid)
        .expect("bond is live")
        .props
        .get(BCC_BOND_TYPE)
        .and_then(|v| v.as_f64())
        .expect("perceived BCC bond type") as i32
}

/// Every perceived type, keyed by the unordered atom-index pair — the shape a
/// permutation test can compare.
fn types_by_pair(g: &Atomistic, ids: &[AtomId]) -> Vec<((usize, usize), i32)> {
    let index: std::collections::HashMap<AtomId, usize> = ids
        .iter()
        .copied()
        .enumerate()
        .map(|(i, a)| (a, i))
        .collect();
    let mut out: Vec<((usize, usize), i32)> = g
        .bonds()
        .map(|(bid, bond)| {
            let (a, b) = (index[&bond.nodes[0]], index[&bond.nodes[1]]);
            let ty = bond
                .props
                .get(BCC_BOND_TYPE)
                .and_then(|v| v.as_f64())
                .unwrap_or_else(|| panic!("bond {bid:?} has no perceived type"))
                as i32;
            ((a.min(b), a.max(b)), ty)
        })
        .collect();
    out.sort_unstable();
    out
}

/// Acetate, `CC(=O)[O-]`: the carboxylate whose two oxygens must not be
/// distinguishable.
fn acetate() -> (Atomistic, Vec<AtomId>) {
    mol(
        &["C", "C", "O", "O", "H", "H", "H"],
        &[
            (0, 1, 1.0, false),
            (1, 2, 2.0, false), // C=O
            (1, 3, 1.0, false), // C-O(-)
            (0, 4, 1.0, false),
            (0, 5, 1.0, false),
            (0, 6, 1.0, false),
        ],
    )
}

/// Nitromethane, `C[N+](=O)[O-]`.
fn nitromethane() -> (Atomistic, Vec<AtomId>) {
    mol(
        &["C", "N", "O", "O", "H", "H", "H"],
        &[
            (0, 1, 1.0, false),
            (1, 2, 2.0, false), // N=O
            (1, 3, 1.0, false), // N-O(-)
            (0, 4, 1.0, false),
            (0, 5, 1.0, false),
            (0, 6, 1.0, false),
        ],
    )
}

/// Nitrite, `[O-]N=O`: a nitrogen of degree 2 carrying two terminal oxygens.
fn nitrite() -> (Atomistic, Vec<AtomId>) {
    mol(&["O", "N", "O"], &[(0, 1, 1.0, false), (1, 2, 2.0, false)])
}

/// Nitrate, `[O-][N+](=O)[O-]`: three terminal oxygens on one nitrogen, two of
/// them (0 and 3) topologically identical.
fn nitrate() -> (Atomistic, Vec<AtomId>) {
    mol(
        &["O", "N", "O", "O"],
        &[
            (0, 1, 1.0, false), // N-O(-)
            (1, 2, 2.0, false), // N=O
            (1, 3, 1.0, false), // N-O(-)  — identical to bond 0 in every respect
        ],
    )
}

/// Push an aromatic carbocycle of `n` CH units, returning its carbons in ring
/// order. Ring bonds carry the aromatic flag and order 1.5 — no Kekulé structure,
/// exactly what a mol2 `ar` bond or an RDKit aromatic perception hands over.
fn push_aromatic_ring(g: &mut Atomistic, n: usize) -> Vec<AtomId> {
    let carbons: Vec<AtomId> = (0..n)
        .map(|i| g.add_atom_xyz("C", i as f64, 0.0, 0.0))
        .collect();
    for i in 0..n {
        let bid = g
            .add_bond(carbons[i], carbons[(i + 1) % n])
            .expect("ring bond");
        g.set_bond_prop(bid, keys::ORDER, 1.5).expect("ring order");
        g.set_bond_prop(bid, "is_aromatic", true)
            .expect("ring aromatic");
    }
    for &c in &carbons {
        let h = g.add_atom_xyz("H", 0.0, 1.0, 0.0);
        let bid = g.add_bond(c, h).expect("C-H bond");
        g.set_bond_prop(bid, keys::ORDER, 1.0).expect("C-H order");
    }
    carbons
}

// ── ac-004: the aromatic promotion boundary ─────────────────────────────────

#[test]
fn benzene_ring_bonds_are_aromatic_types_and_alternate() {
    let mut g = Atomistic::new();
    let ring = push_aromatic_ring(&mut g, 6);
    let typed = Perceive::new().find_bond_types(&g);

    let ring_types: Vec<i32> = (0..6)
        .map(|i| bond_type(&typed, ring[i], ring[(i + 1) % 6]))
        .collect();
    assert!(
        ring_types.iter().all(|t| matches!(t, 7 | 8)),
        "benzene ring bonds must be aromatic (7/8), got {ring_types:?}"
    );
    // A Kekulé structure, not a smear: each carbon carries exactly one double.
    assert_eq!(
        ring_types.iter().filter(|t| **t == 8).count(),
        3,
        "benzene needs three aromatic doubles, got {ring_types:?}"
    );
    // The C-H bonds are not in the ring and stay plain singles.
    for (_, bond) in typed.bonds() {
        let is_ch = [bond.nodes[0], bond.nodes[1]]
            .iter()
            .any(|a| !ring.contains(a));
        if is_ch {
            assert_eq!(
                bond.props.get(BCC_BOND_TYPE).and_then(|v| v.as_f64()),
                Some(1.0)
            );
        }
    }
}

#[test]
fn heteroaromatic_ring_bonds_are_promoted() {
    // Pyridine (6-ring, one N) and thiophene (5-ring, one S): promotion must not
    // be a carbon-only affair, and the 5-ring must be in range.
    let (pyridine, p) = mol(
        &["C", "C", "C", "N", "C", "C", "H", "H", "H", "H", "H"],
        &[
            (0, 1, 1.5, true),
            (1, 2, 1.5, true),
            (2, 3, 1.5, true),
            (3, 4, 1.5, true),
            (4, 5, 1.5, true),
            (5, 0, 1.5, true),
            (0, 6, 1.0, false),
            (1, 7, 1.0, false),
            (2, 8, 1.0, false),
            (4, 9, 1.0, false),
            (5, 10, 1.0, false),
        ],
    );
    let typed = Perceive::new().find_bond_types(&pyridine);
    for i in 0..6 {
        let t = bond_type(&typed, p[i], p[(i + 1) % 6]);
        assert!(matches!(t, 7 | 8), "pyridine ring bond {i} typed {t}");
    }

    let (thiophene, s) = mol(
        &["C", "C", "C", "S", "C", "H", "H", "H", "H"],
        &[
            (0, 1, 1.5, true),
            (1, 2, 1.5, true),
            (2, 3, 1.5, true),
            (3, 4, 1.5, true),
            (4, 0, 1.5, true),
            (0, 5, 1.0, false),
            (1, 6, 1.0, false),
            (2, 7, 1.0, false),
            (4, 8, 1.0, false),
        ],
    );
    let typed = Perceive::new().find_bond_types(&thiophene);
    for i in 0..5 {
        let t = bond_type(&typed, s[i], s[(i + 1) % 5]);
        assert!(matches!(t, 7 | 8), "thiophene ring bond {i} typed {t}");
    }
    // Sulfur stays two-valent: both of its ring bonds are aromatic *singles*.
    assert_eq!(bond_type(&typed, s[2], s[3]), 7);
    assert_eq!(bond_type(&typed, s[3], s[4]), 7);
}

#[test]
fn biphenyl_inter_ring_bond_is_not_promoted() {
    // Both endpoints are aromatic, but they share no ring — so the bond joining the
    // two rings stays a plain single. This is the boundary antechamber draws, and
    // the reason it draws it at "a ring both atoms are in" rather than "both atoms
    // are aromatic".
    let mut g = Atomistic::new();
    let ring_a = push_aromatic_ring(&mut g, 6);
    let ring_b = push_aromatic_ring(&mut g, 6);
    let bid = g.add_bond(ring_a[0], ring_b[0]).expect("inter-ring bond");
    g.set_bond_prop(bid, keys::ORDER, 1.0)
        .expect("inter-ring order");

    let typed = Perceive::new().find_bond_types(&g);
    let t = bond_type(&typed, ring_a[0], ring_b[0]);
    assert!(
        !matches!(t, 7 | 8),
        "biphenyl's inter-ring bond must not be promoted, got {t}"
    );
    assert_eq!(t, 1);
    // The rings themselves are still promoted.
    assert!(matches!(bond_type(&typed, ring_a[1], ring_a[2]), 7 | 8));
    assert!(matches!(bond_type(&typed, ring_b[1], ring_b[2]), 7 | 8));
}

#[test]
fn seven_membered_aromatic_ring_bonds_are_not_promoted() {
    // Promotion is restricted to rings of size 5 or 6. A 7-membered aromatic ring
    // (tropylium-like) is aromatic by flag and by ring membership, and is still not
    // promoted — its bonds keep the plain Kekulé types.
    let mut g = Atomistic::new();
    let ring = push_aromatic_ring(&mut g, 7);
    let typed = Perceive::new().find_bond_types(&g);

    for i in 0..7 {
        let t = bond_type(&typed, ring[i], ring[(i + 1) % 7]);
        assert!(
            !matches!(t, 7 | 8),
            "7-ring bond {i} must not be promoted, got {t}"
        );
        assert!(matches!(t, 1 | 2), "7-ring bond {i} typed {t}");
    }
}

// ── ac-003: delocalized (type 9) ────────────────────────────────────────────

#[test]
fn acetate_both_carboxylate_bonds_are_delocalized() {
    // The input distinguishes them — one is drawn C=O, the other C-O(-). The
    // perception must not: both are type 9, so the two oxygens correct identically
    // downstream. (Charge equality is asserted end-to-end in the `ff` target.)
    let (g, ids) = acetate();
    let typed = Perceive::new().find_bond_types(&g);

    assert_eq!(bond_type(&typed, ids[1], ids[2]), 9, "the drawn C=O");
    assert_eq!(bond_type(&typed, ids[1], ids[3]), 9, "the drawn C-O(-)");
    assert_eq!(bond_type(&typed, ids[0], ids[1]), 1, "the C-C is untouched");
}

#[test]
fn nitromethane_both_nitro_bonds_are_delocalized() {
    let (g, ids) = nitromethane();
    let typed = Perceive::new().find_bond_types(&g);

    assert_eq!(bond_type(&typed, ids[1], ids[2]), 9, "the drawn N=O");
    assert_eq!(bond_type(&typed, ids[1], ids[3]), 9, "the drawn N-O(-)");
    assert_eq!(bond_type(&typed, ids[0], ids[1]), 1, "the C-N is untouched");
}

#[test]
fn a_lone_terminal_oxygen_does_not_delocalize_a_neutral_carbonyl() {
    // Acetone's C=O: a terminal oxygen, but on a carbon with only ONE of them, so
    // the carbon is not a conjugated centre and the bond stays a plain double. This
    // is the guard that keeps rule 9 from swallowing every carbonyl in chemistry.
    let (g, ids) = mol(
        &["C", "C", "C", "O"],
        &[
            (0, 1, 1.0, false),
            (1, 2, 1.0, false),
            (1, 3, 2.0, false), // C=O
        ],
    );
    let typed = Perceive::new().find_bond_types(&g);
    assert_eq!(bond_type(&typed, ids[1], ids[3]), 2);
}

#[test]
fn sulfone_and_phosphate_bonds_are_delocalized() {
    // Part 5: hypervalent centres. Real dimethyl sulfone has S at degree 4 (two
    // methyls + two O); this minimal fixture uses degree 3 (one C + two terminal O),
    // which is still hypervalent under the (Z=16, degree 3, two terminal O/S) rule.
    // Sulfoxide-like S with degree 3 and exactly two terminal O:
    let (sulfone, s) = mol(
        &["C", "S", "O", "O"],
        &[(0, 1, 1.0, false), (1, 2, 2.0, false), (1, 3, 2.0, false)],
    );
    let typed = Perceive::new().find_bond_types(&sulfone);
    assert_eq!(bond_type(&typed, s[1], s[2]), 9);
    assert_eq!(bond_type(&typed, s[1], s[3]), 9);

    // Phosphate: P of degree 4 with two terminal oxygens.
    let (phosphate, p) = mol(
        &["C", "O", "P", "O", "O", "O", "C"],
        &[
            (0, 1, 1.0, false),
            (1, 2, 1.0, false),
            (2, 3, 2.0, false), // P=O   (terminal)
            (2, 4, 1.0, false), // P-O(-) (terminal)
            (2, 5, 1.0, false), // P-O-C  (bridging)
            (5, 6, 1.0, false),
        ],
    );
    let typed = Perceive::new().find_bond_types(&phosphate);
    assert_eq!(bond_type(&typed, p[2], p[3]), 9, "P=O");
    assert_eq!(bond_type(&typed, p[2], p[4]), 9, "P-O(-)");
    assert_eq!(
        bond_type(&typed, p[2], p[5]),
        1,
        "the bridging P-O-C is NOT delocalized"
    );
}

// ── ac-005: types 6 and 11 ──────────────────────────────────────────────────

#[test]
fn type_6_is_reachable_and_symmetric_on_nitrite() {
    // Type 6 exists and must be produced: BCCPARM has type-6 rows for 21|31, 22|31,
    // 23|31, 24|31, 25|31 and 25|51, and FIVE of those six pairs have no type-9 row
    // at all — typing them 9 instead would be a missing-parameter error, not a
    // small numeric drift. Nitrite (N of degree 2, two terminal O) is the canonical
    // reachable case, and AmberTools25 types both of its N-O bonds 6.
    let (g, ids) = nitrite();
    let typed = Perceive::new().find_bond_types(&g);

    assert_eq!(bond_type(&typed, ids[0], ids[1]), 6, "the drawn N-O(-)");
    assert_eq!(bond_type(&typed, ids[1], ids[2]), 6, "the drawn N=O");
}

#[test]
fn type_11_is_unreachable() {
    // Type 11 occupies 26 same-type diagonal rows of BCCPARM.DAT, every one of them
    // worth exactly 0.0000, and no rule in bondtype.c emits it. Confirmed against
    // AmberTools25 over a 33-molecule probe sweep: the emitted alphabet is
    // {1,2,3,6,7,8,9}. It is asserted unreachable here rather than implemented as a
    // dead branch. Type 10 is likewise never emitted — it is the *unresolved*
    // aromatic precursor (the SYBYL `ar` token), an input that this perception
    // resolves into 7 or 8.
    let mut molecules = vec![acetate().0, nitromethane().0, nitrite().0, nitrate().0];
    let mut benzene = Atomistic::new();
    push_aromatic_ring(&mut benzene, 6);
    molecules.push(benzene);

    for g in &molecules {
        let typed = Perceive::new().find_bond_types(g);
        for (bid, bond) in typed.bonds() {
            let t = bond
                .props
                .get(BCC_BOND_TYPE)
                .and_then(|v| v.as_f64())
                .unwrap() as i32;
            assert!(
                matches!(t, 1 | 2 | 3 | 6 | 7 | 8 | 9),
                "bond {bid:?} got type {t}, which is outside the emitted alphabet"
            );
            assert_ne!(t, 10, "type 10 is a precursor and must be resolved to 7/8");
            assert_ne!(t, 11, "type 11 is unreachable");
        }
    }
}

#[test]
fn aromatic_type_10_input_is_resolved_to_7_or_8() {
    // A caller handing us a mol2 `ar` bond (BCC type 10) gets it RESOLVED, not
    // echoed: 10 is the unresolved precursor of 7/8, not a peer of them.
    let mut g = Atomistic::new();
    let ring = push_aromatic_ring(&mut g, 6);
    let bond_ids: Vec<BondId> = g.bonds().map(|(bid, _)| bid).collect();
    for bid in bond_ids {
        let bond = g.get_bond(bid).expect("bond");
        let ring_bond = ring.contains(&bond.nodes[0]) && ring.contains(&bond.nodes[1]);
        if ring_bond {
            g.set_bond_prop(bid, BCC_BOND_TYPE, 10)
                .expect("set type 10");
        }
    }

    let typed = Perceive::new().find_bond_types(&g);
    for i in 0..6 {
        let t = bond_type(&typed, ring[i], ring[(i + 1) % 6]);
        assert!(matches!(t, 7 | 8), "type 10 survived as {t}");
    }
}

// ── determinism: the input's bond order must not leak into the answer ────────

#[test]
fn perception_is_invariant_under_bond_permutation() {
    // THE property that the type-6 repair exists to deliver. antechamber's
    // /*part3*/ scans only the FIRST non-partner neighbour (its `break` sits
    // outside the `if`) and its mirrored branch assigns 6 unconditionally, so its
    // answer moves when the input file lists the same bonds in a different order.
    // Ours must not — for ANY molecule, including the aromatic ones, whose Kekulé
    // tie-break is therefore keyed on atom order alone.
    let cases: Vec<(&str, Vec<&str>, Vec<BondSpec>)> = vec![
        (
            "nitrate",
            vec!["O", "N", "O", "O"],
            vec![(0, 1, 1.0, false), (1, 2, 2.0, false), (1, 3, 1.0, false)],
        ),
        (
            "nitrite",
            vec!["O", "N", "O"],
            vec![(0, 1, 1.0, false), (1, 2, 2.0, false)],
        ),
        (
            "nitromethane",
            vec!["C", "N", "O", "O", "H", "H", "H"],
            vec![
                (0, 1, 1.0, false),
                (1, 2, 2.0, false),
                (1, 3, 1.0, false),
                (0, 4, 1.0, false),
                (0, 5, 1.0, false),
                (0, 6, 1.0, false),
            ],
        ),
        (
            "acetate",
            vec!["C", "C", "O", "O", "H", "H", "H"],
            vec![
                (0, 1, 1.0, false),
                (1, 2, 2.0, false),
                (1, 3, 1.0, false),
                (0, 4, 1.0, false),
                (0, 5, 1.0, false),
                (0, 6, 1.0, false),
            ],
        ),
        (
            "imidazole",
            vec!["C", "C", "N", "C", "N", "H", "H", "H", "H"],
            vec![
                (0, 1, 1.5, true),
                (1, 2, 1.5, true),
                (2, 3, 1.5, true),
                (3, 4, 1.5, true),
                (4, 0, 1.5, true),
                (0, 5, 1.0, false),
                (1, 6, 1.0, false),
                (3, 7, 1.0, false),
                (4, 8, 1.0, false),
            ],
        ),
    ];

    for (name, elements, bonds) in cases {
        let (g, ids) = mol(&elements, &bonds);
        let reference = types_by_pair(&Perceive::new().find_bond_types(&g), &ids);

        // Every rotation of the bond list, and the same again with every bond's
        // endpoints swapped. A rotation is enough to move ANY bond into first
        // position, which is what the first-neighbour scan was sensitive to.
        for shift in 0..bonds.len() {
            let mut rotated: Vec<BondSpec> = bonds.clone();
            rotated.rotate_left(shift);
            let swapped: Vec<BondSpec> = rotated
                .iter()
                .map(|(i, j, o, ar)| (*j, *i, *o, *ar))
                .collect();

            for variant in [rotated, swapped] {
                let (g2, ids2) = mol(&elements, &variant);
                let got = types_by_pair(&Perceive::new().find_bond_types(&g2), &ids2);
                assert_eq!(
                    got, reference,
                    "{name}: bond order leaked into the perception (shift {shift})"
                );
            }
        }
    }
}

#[test]
fn nitrate_topologically_identical_oxygens_get_the_same_type() {
    // DELIBERATE DIVERGENCE FROM ANTECHAMBER — the reason the type-6 rule is
    // repaired rather than ported.
    //
    // Nitrate's O0 and O3 are the same atom by every measure: both terminal, both
    // singly bonded to the same nitrogen, indistinguishable under the graph's
    // automorphism. AmberTools25 nevertheless types the two bonds 6 and 9 (it stops
    // its neighbour scan at the first non-partner neighbour, so the answer depends
    // on the order the bonds happened to be listed), which splits the two oxygens'
    // final AM1-BCC charges by 0.28 e — a symmetry break with no chemistry behind
    // it. Here they agree, and the whole nitro group comes out delocalized.
    let (g, ids) = nitrate();
    let typed = Perceive::new().find_bond_types(&g);

    let t0 = bond_type(&typed, ids[0], ids[1]);
    let t3 = bond_type(&typed, ids[1], ids[3]);
    assert_eq!(
        t0, t3,
        "the two identical O(-) must type identically (antechamber: 6 vs 9)"
    );
    assert_eq!(t0, 9, "a conjugated nitro nitrogen delocalizes its bonds");
    assert_eq!(bond_type(&typed, ids[1], ids[2]), 9, "and so does its N=O");
}

// ── the builder contract ────────────────────────────────────────────────────

#[test]
fn find_bond_types_does_not_touch_the_input() {
    let (g, ids) = acetate();
    let before: Vec<Option<f64>> = g
        .bonds()
        .map(|(_, b)| b.props.get(BCC_BOND_TYPE).and_then(|v| v.as_f64()))
        .collect();

    let typed = Perceive::new().find_bond_types(&g);

    let after: Vec<Option<f64>> = g
        .bonds()
        .map(|(_, b)| b.props.get(BCC_BOND_TYPE).and_then(|v| v.as_f64()))
        .collect();
    assert_eq!(before, after, "the input graph was mutated");
    assert!(before.iter().all(Option::is_none));
    assert_eq!(bond_type(&typed, ids[1], ids[2]), 9);

    // Bond orders are not rewritten either: the perceived Kekulé structure stays
    // internal.
    for (bid, bond) in typed.bonds() {
        let order = bond.props.get(keys::ORDER).and_then(|v| v.as_f64());
        let original = g
            .get_bond(bid)
            .ok()
            .and_then(|b| b.props.get(keys::ORDER).and_then(|v| v.as_f64()));
        assert_eq!(order, original, "bond order was rewritten");
    }
}

#[test]
fn perception_is_idempotent() {
    // Running it twice must not move the answer — the second pass reads its own
    // 7/8 as aromatic markings and must re-derive the same Kekulé structure.
    let mut g = Atomistic::new();
    push_aromatic_ring(&mut g, 6);
    let all: Vec<AtomId> = g.atoms().map(|(aid, _)| aid).collect();

    let once = Perceive::new().find_bond_types(&g);
    let twice = Perceive::new().find_bond_types(&once);

    assert_eq!(types_by_pair(&once, &all), types_by_pair(&twice, &all));
}

#[test]
fn an_empty_graph_perceives_nothing() {
    let typed = Perceive::new().find_bond_types(&Atomistic::new());
    assert_eq!(typed.n_bonds(), 0);
}
