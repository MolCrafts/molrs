//! End-to-end MMFF94 typification: MolGraph (built in code) -> typed Atomistic
//! -> Frame, atom/bond/angle/dihedral labels, and full build.

use molrs::ff::typifier::mmff::MMFFTypifier;
use molrs::system::molgraph::{Atom, PropValue};
use molrs::{AtomId, Atomistic};

fn typifier() -> MMFFTypifier {
    MMFFTypifier::mmff94().expect("load embedded MMFF94")
}

fn bond(mol: &mut Atomistic, a: AtomId, b: AtomId, order: f64) {
    if let Ok(bid) = mol.add_bond(a, b) {
        let _ = mol.set_bond_prop(bid, "order", PropValue::F64(order));
    }
}

// ---------------------------------------------------------------------------
// Parameter loading
// ---------------------------------------------------------------------------

#[test]
fn embedded_mmff94_loads_atom_prop_table() {
    let t = typifier();
    let params = t.params();
    // Type 1 = CR (sp3 carbon): atomic number 6, coordination 4, valence 4.
    let p1 = params.get_prop(1).expect("type 1");
    assert_eq!(p1.atno, 6);
    assert_eq!(p1.crd, 4);
    assert_eq!(p1.val, 4);
    // Type 5 = HC (hydrogen on carbon): atomic number 1.
    assert_eq!(params.get_prop(5).expect("type 5").atno, 1);
}

// ---------------------------------------------------------------------------
// Bond / angle / torsion typification
// ---------------------------------------------------------------------------

#[test]
fn bond_typification_normal_vs_aromatic() {
    let t = typifier();
    assert_eq!(t.typify_bond(1, 1, 1.0), 0, "sp3 single -> normal");
    assert_eq!(t.typify_bond(37, 37, 1.5), 1, "aromatic -> delocalized");
}

#[test]
fn angle_typification_counts_delocalized_bonds() {
    let t = typifier();
    assert_eq!(t.typify_angle(0, 0), 0);
    assert_eq!(t.typify_angle(1, 0), 1);
    assert_eq!(t.typify_angle(0, 1), 1);
    assert_eq!(t.typify_angle(1, 1), 2);
}

#[test]
fn torsion_typification_central_bond_delocalized() {
    let t = typifier();
    assert_eq!(t.typify_dihedral(0, 0, 0), 0);
    assert_eq!(t.typify_dihedral(0, 1, 0), 1, "central delocalized");
}

// ---------------------------------------------------------------------------
// Frame builder (Typifier trait) — topology block shapes
// ---------------------------------------------------------------------------

#[test]
fn typify_ethane_produces_expected_topology_blocks() {
    let t = typifier();
    let mut mol = Atomistic::new();
    let c1 = mol.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
    let c2 = mol.add_atom(Atom::xyz("C", 1.54, 0.0, 0.0));
    bond(&mut mol, c1, c2, 1.0);
    let hpos = [
        (c1, [0.0, 1.0, 0.0]),
        (c1, [0.0, -0.5, 0.87]),
        (c1, [0.0, -0.5, -0.87]),
        (c2, [1.54, 1.0, 0.0]),
        (c2, [1.54, -0.5, 0.87]),
        (c2, [1.54, -0.5, -0.87]),
    ];
    for (c, [x, y, z]) in hpos {
        let h = mol.add_atom(Atom::xyz("H", x, y, z));
        bond(&mut mol, c, h, 1.0);
    }

    // typify returns a labeled Atomistic; materialize it to inspect blocks.
    let frame = t.typify(&mol).expect("typify").to_frame();

    let atoms = frame.get("atoms").expect("atoms block");
    assert_eq!(atoms.nrows(), Some(8));
    assert!(atoms.contains_key("type"));
    assert!(atoms.contains_key("charge"));

    // 1 C-C + 6 C-H = 7 bonds.
    assert_eq!(frame.get("bonds").expect("bonds").nrows(), Some(7));
    // H-C-H (3 per C) + H-C-C (3 per C) = 12 angles.
    assert_eq!(frame.get("angles").expect("angles").nrows(), Some(12));
    // H-C-C-H = 3x3 = 9 dihedrals.
    assert_eq!(frame.get("dihedrals").expect("dihedrals").nrows(), Some(9));
    // typify is pairs-free now — the neighbour list is built by `build()`.
    assert!(!frame.contains_key("pairs"));
}

// ---------------------------------------------------------------------------
// Full build path (typify + compile).
//
// The typify half works; the compile half currently fails on stretch-bend
// params (see suite report and tests/ff/potential/mmff.rs). We pin the typify
// output here and the documented build failure.
// ---------------------------------------------------------------------------

#[test]
fn methane_typifies_then_build_fails_on_stretch_bend() {
    let t = typifier();
    let mut mol = Atomistic::new();
    let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
    let geo = [
        [0.63, 0.63, 0.63],
        [-0.63, -0.63, 0.63],
        [-0.63, 0.63, -0.63],
        [0.63, -0.63, -0.63],
    ];
    for g in geo {
        let h = mol.add_atom_xyz("H", g[0], g[1], g[2]);
        bond(&mut mol, c, h, 1.0);
    }

    // Typify succeeds: methane has 5 atoms, 4 bonds, 6 H-C-H angles.
    let frame = t.typify(&mol).expect("typify methane").to_frame();
    assert_eq!(frame.get("atoms").unwrap().nrows(), Some(5));
    assert_eq!(frame.get("bonds").unwrap().nrows(), Some(4));
    assert_eq!(frame.get("angles").unwrap().nrows(), Some(6));

    // Methane's carbon is four-coordinate (no out-of-plane term) and it has no
    // dihedrals and no non-bonded pairs (every atom pair is 1-2 or 1-3 excluded),
    // so build() resolves the bond/angle/stretch-bend kernels and yields finite
    // energy + forces.
    let pots = t.build(&mol).expect("build potentials");
    let coords = molrs::ff::potential::extract_coords(&frame).expect("coords");
    let (e, forces) = pots.calc_energy_forces(&coords);
    assert!(e.is_finite(), "energy not finite: {e}");
    assert!(forces.iter().all(|f| f.is_finite()), "non-finite force");
}

/// Planar benzene: 6 aromatic carbons (type 37) + 6 hydrogens (type 5).
fn benzene() -> Atomistic {
    let mut mol = Atomistic::new();
    let r_c = 1.39;
    let r_h = 2.47;
    let (mut cs, mut hs) = (Vec::new(), Vec::new());
    for i in 0..6 {
        let ang = std::f64::consts::PI / 3.0 * i as f64;
        cs.push(mol.add_atom_xyz("C", r_c * ang.cos(), r_c * ang.sin(), 0.0));
        hs.push(mol.add_atom_xyz("H", r_h * ang.cos(), r_h * ang.sin(), 0.0));
    }
    for i in 0..6 {
        bond(&mut mol, cs[i], cs[(i + 1) % 6], 1.5); // aromatic ring bond
        bond(&mut mol, cs[i], hs[i], 1.0);
    }
    mol
}

/// RED before per-instance migration: benzene's stretch-bend type `1_37_37_37`
/// has no explicit STBN row, so the shared-table kernel path errored with
/// `mmff_stbn: unknown '1_37_37_37'`. After the typifier bakes per-instance
/// stretch-bend params via the RDKit-validated `energy::params` resolver (which
/// has the dfsb default-row fallback), build() must resolve every MMFF term.
#[test]
fn benzene_build_resolves_stretch_bend() {
    let t = typifier();
    let mol = benzene();
    let pots = t
        .build(&mol)
        .expect("benzene build should resolve all MMFF terms (incl. stretch-bend)");
    let frame = t.typify(&mol).expect("typify").to_frame();
    let coords = molrs::ff::potential::extract_coords(&frame).expect("coords");
    let (e, forces) = pots.calc_energy_forces(&coords);
    assert!(e.is_finite(), "benzene energy not finite: {e}");
    assert!(
        forces.iter().all(|f| f.is_finite()),
        "non-finite benzene force"
    );
}
