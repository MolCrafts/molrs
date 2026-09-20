//! Tests for MMFF94 typifier.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::ff::typifier::mmff::MMFF94Typifier;
    use molrs::system::molgraph::Atom;
    use molrs::{AtomId, Atomistic};

    fn atom(sym: &str) -> Atom {
        let mut a = Atom::new();
        a.set("element", sym);
        a
    }

    fn bond_order(mol: &mut Atomistic, a: AtomId, b: AtomId, order: f64) {
        if let Ok(bid) = mol.add_bond(a, b) {
            // The old float encoding, split into the two facts it conflated.
            let _ = if (order - 1.5).abs() < 1e-6 {
                mol.set_bond_class(
                    bid,
                    crate::system::bond::BondType::Aromatic,
                    crate::system::bond::BondNumber::Unknown,
                )
            } else {
                mol.set_bond_type(bid, crate::system::bond::BondType::from_code(order as u32))
            };
        }
    }

    fn test_typifier() -> MMFF94Typifier {
        MMFF94Typifier::new()
    }

    // -----------------------------------------------------------------------
    // XML loading tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_mmff_params() {
        let typifier = test_typifier();
        let params = typifier.params();
        // Should have loaded ~90+ atom types
        assert!(
            params.props.len() > 80,
            "expected >80 atom props, got {}",
            params.props.len()
        );
        // Type 1 = CR (sp3 carbon)
        let p1 = params.get_prop(1).expect("type 1 should exist");
        assert_eq!(p1.atno, 6);
        assert_eq!(p1.crd, 4);
        assert_eq!(p1.val, 4);
    }

    // -----------------------------------------------------------------------
    // Atom typing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ethane_atom_types() {
        // CH3-CH3 through the live typify path (RDKit-validated front-end): both
        // C are MMFF type 1 (CR), all H are type 5 (HC). Atom rows follow
        // insertion order — c1, c2, then the six H — so rows 0..2 are carbons.
        let typifier = test_typifier();
        let mut mol = Atomistic::new();
        let c1 = mol.add_atom(atom("C"));
        let c2 = mol.add_atom(atom("C"));
        bond_order(&mut mol, c1, c2, 1.0);
        for _ in 0..3 {
            let h = mol.add_atom(atom("H"));
            bond_order(&mut mol, c1, h, 1.0);
        }
        for _ in 0..3 {
            let h = mol.add_atom(atom("H"));
            bond_order(&mut mol, c2, h, 1.0);
        }

        let frame = typifier.typify(&mol).expect("typify ethane").to_frame();
        let types = frame
            .get("atoms")
            .unwrap()
            .get_string("type")
            .expect("atoms.type column");
        assert_eq!(types[0], "1", "C1 should be MMFF type 1 (CR)");
        assert_eq!(types[1], "1", "C2 should be MMFF type 1 (CR)");
        for (i, t) in types.iter().enumerate().skip(2) {
            assert_eq!(t, "5", "H at row {i} should be type 5 (HC)");
        }
    }

    // -----------------------------------------------------------------------
    // Bond / angle / torsion type classification
    // -----------------------------------------------------------------------
    //
    // The seven unit tests that lived here drove `MMFF94Typifier::typify_bond` /
    // `typify_angle` / `typify_dihedral` — three front-door methods over
    // `typifier/mmff/classify.rs`, a second implementation of MMFF's context
    // rules that `ff/mmff/params.rs` already implements correctly. All three are
    // deleted (`mmff-orthogonal-02`), and so are the tests, because the values
    // they pinned were WRONG:
    //
    //   * `typify_bond(37, 37, 1.5) == 1` — an aromatic bond is bond type **0**.
    //     `getMMFFBondType` returns 1 only for a bond that is SINGLE and joins two
    //     sbmb/arom types; after MMFF aromaticity perception a ring bond is
    //     AROMATIC, never SINGLE. Backwards.
    //   * `typify_angle(bt_ij, bt_jk)` — the signature cannot express the rule.
    //     A C-C-C angle in cyclopropane is angle type **3**, and no function of
    //     two bond types can say so: ring membership is not among its arguments.
    //
    // The replacements assert against RDKit's answers, on molecules rather than on
    // bare integers: `tests/ff/typifier/mmff_labels.rs`.

    // -----------------------------------------------------------------------
    // Full frame builder tests
    // -----------------------------------------------------------------------
}
