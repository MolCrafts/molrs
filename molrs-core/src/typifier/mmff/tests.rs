//! Tests for MMFF94 typifier.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::molgraph::{Atom, AtomId, MolGraph, PropValue};
    use crate::rings::find_rings;
    use crate::typifier::Typifier;
    use crate::typifier::mmff::MMFFTypifier;

    fn atom(sym: &str) -> Atom {
        let mut a = Atom::new();
        a.set("element", sym);
        a
    }

    fn atom_xyz(sym: &str, x: f64, y: f64, z: f64) -> Atom {
        Atom::xyz(sym, x, y, z)
    }

    fn bond_order(mol: &mut MolGraph, a: AtomId, b: AtomId, order: f64) {
        if let Ok(bid) = mol.add_bond(a, b)
            && let Ok(bond) = mol.get_bond_mut(bid)
        {
            bond.props
                .insert("order".to_string(), PropValue::F64(order));
        }
    }

    fn test_typifier() -> MMFFTypifier {
        MMFFTypifier::mmff94().expect("load MMFF94")
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
        // Equivalence table should be loaded
        assert!(!params.equiv.is_empty());
        let eq1 = params.get_equiv(1).expect("equiv for type 1");
        assert_eq!(eq1.eq1, 1);
        assert_eq!(eq1.eq2, 1);
    }

    // -----------------------------------------------------------------------
    // Atom typing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ethane_atom_types() {
        // CH3-CH3: both C should be type 1 (CR), all H should be type 5 (HC)
        let typifier = test_typifier();
        let mut mol = MolGraph::new();
        let c1 = mol.add_atom(atom("C"));
        let c2 = mol.add_atom(atom("C"));
        bond_order(&mut mol, c1, c2, 1.0);
        // Add explicit H
        for _ in 0..3 {
            let h = mol.add_atom(atom("H"));
            bond_order(&mut mol, c1, h, 1.0);
        }
        for _ in 0..3 {
            let h = mol.add_atom(atom("H"));
            bond_order(&mut mol, c2, h, 1.0);
        }

        let ring_info = find_rings(&mol);
        let types = typifier.assign_atom_types(&mol, &ring_info);

        assert_eq!(types[&c1], 1, "C1 should be type 1 (CR)");
        assert_eq!(types[&c2], 1, "C2 should be type 1 (CR)");
        // All H should be type 5
        for (aid, atom) in mol.atoms() {
            if atom.get_str("element") == Some("H") {
                assert_eq!(types[&aid], 5, "H should be type 5 (HC)");
            }
        }
    }

    #[test]
    fn test_benzene_atom_types() {
        // Benzene: 6 aromatic C should be type 37 (CB), H should be type 5
        let typifier = test_typifier();
        let mut mol = MolGraph::new();
        let cs: Vec<AtomId> = (0..6).map(|_| mol.add_atom(atom("C"))).collect();
        for i in 0..6 {
            bond_order(&mut mol, cs[i], cs[(i + 1) % 6], 1.5);
        }
        // Add H to each C
        for &c in &cs {
            let h = mol.add_atom(atom("H"));
            bond_order(&mut mol, c, h, 1.0);
        }

        let ring_info = find_rings(&mol);
        let types = typifier.assign_atom_types(&mol, &ring_info);

        for &c in &cs {
            assert_eq!(
                types[&c], 37,
                "benzene C should be type 37 (CB), got {}",
                types[&c]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Bond type classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bond_type_normal() {
        let typifier = test_typifier();
        // Normal single bond between sp3 atoms
        assert_eq!(typifier.classify_bond_type(1, 1, 1.0), 0);
    }

    #[test]
    fn test_bond_type_aromatic() {
        let typifier = test_typifier();
        // Aromatic bond
        assert_eq!(typifier.classify_bond_type(37, 37, 1.5), 1);
    }

    // -----------------------------------------------------------------------
    // Angle type classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_angle_type_normal() {
        let typifier = test_typifier();
        assert_eq!(typifier.classify_angle_type(0, 0), 0);
    }

    #[test]
    fn test_angle_type_one_delocalized() {
        let typifier = test_typifier();
        assert_eq!(typifier.classify_angle_type(1, 0), 1);
        assert_eq!(typifier.classify_angle_type(0, 1), 1);
    }

    #[test]
    fn test_angle_type_both_delocalized() {
        let typifier = test_typifier();
        assert_eq!(typifier.classify_angle_type(1, 1), 2);
    }

    // -----------------------------------------------------------------------
    // Torsion type classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_torsion_type_normal() {
        let typifier = test_typifier();
        assert_eq!(typifier.classify_torsion_type(0, 0, 0), 0);
    }

    #[test]
    fn test_torsion_type_central_delocalized() {
        let typifier = test_typifier();
        assert_eq!(typifier.classify_torsion_type(0, 1, 0), 1);
    }

    // -----------------------------------------------------------------------
    // Full frame builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_ethane_frame() {
        let typifier = test_typifier();

        let mut mol = MolGraph::new();
        let c1 = mol.add_atom(atom_xyz("C", 0.0, 0.0, 0.0));
        let c2 = mol.add_atom(atom_xyz("C", 1.54, 0.0, 0.0));
        bond_order(&mut mol, c1, c2, 1.0);
        // Add explicit H
        let h_positions = [
            (c1, [0.0, 1.0, 0.0]),
            (c1, [0.0, -0.5, 0.87]),
            (c1, [0.0, -0.5, -0.87]),
            (c2, [1.54, 1.0, 0.0]),
            (c2, [1.54, -0.5, 0.87]),
            (c2, [1.54, -0.5, -0.87]),
        ];
        for (c, [hx, hy, hz]) in h_positions {
            let h = mol.add_atom(atom_xyz("H", hx, hy, hz));
            bond_order(&mut mol, c, h, 1.0);
        }

        let frame = typifier.typify(&mol).expect("build frame");

        // Check atoms block
        let atoms = frame.get("atoms").expect("atoms block");
        assert_eq!(atoms.nrows(), Some(8));
        assert!(atoms.contains_key("type"));
        assert!(atoms.contains_key("charge"));

        // Check bonds block
        let bonds = frame.get("bonds").expect("bonds block");
        assert_eq!(bonds.nrows(), Some(7));
        assert!(bonds.contains_key("type"));

        // Check angles block
        let angles = frame.get("angles").expect("angles block");
        assert_eq!(angles.nrows(), Some(12));

        // Check dihedrals block
        let dihedrals = frame.get("dihedrals").expect("dihedrals block");
        assert_eq!(dihedrals.nrows(), Some(9));

        // Check pairs block
        assert!(frame.contains_key("pairs"));
    }
}
