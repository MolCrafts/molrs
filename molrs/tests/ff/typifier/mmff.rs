//! End-to-end MMFF94 typification: MolGraph (built in code) -> typed Atomistic
//! -> Frame, atom/bond/angle/dihedral labels, and the compile path.

use molrs::ff::potential::{Potentials, intramolecular_pairs};
use molrs::ff::typifier::mmff::MMFF94Typifier;
use molrs::store::frame::Frame;
use molrs::system::molgraph::{Atom, PropValue};
use molrs::{AtomId, Atomistic};

fn typifier() -> MMFF94Typifier {
    MMFF94Typifier::new()
}

fn bond(mol: &mut Atomistic, a: AtomId, b: AtomId, order: f64) {
    if let Ok(bid) = mol.add_bond(a, b) {
        let _ = mol.set_bond_prop(bid, "order", PropValue::F64(order));
    }
}

/// The standard route: typify → `Frame` (+ neighbour list) → `to_potentials`.
///
/// Returns both, because a caller who wants an energy needs the Frame's coords
/// anyway — which is exactly the shape the deleted `build(mol)` shortcut hid.
fn compile(t: &MMFF94Typifier, mol: &Atomistic) -> Result<(Potentials, Frame), String> {
    let mut frame = t.typify(mol)?.to_frame();
    frame.insert("pairs", intramolecular_pairs(&frame));
    let pots = t.ff().to_potentials(&frame)?;
    Ok((pots, frame))
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
//
// The three tests here drove `typify_bond` / `typify_angle` / `typify_dihedral`,
// the front-door forwards over `typifier/mmff/classify.rs`. That module was a
// second implementation of MMFF's context rules — one `ff/mmff/params.rs` already
// implements correctly — and it was wrong: it pinned `typify_bond(37, 37, 1.5)
// == 1` (an aromatic bond is bond type **0**: `getMMFFBondType` needs SINGLE, and
// after aromaticity perception a ring bond is AROMATIC) and its angle classifier
// took only two bond types, so it could never return the **3** that a cyclopropane
// C-C-C angle actually has.
//
// mmff-orthogonal-02 deletes the module, the three methods, and these tests. The
// replacement asserts the type codes off a typed Frame, against RDKit's rules, on
// real molecules: `tests/ff/typifier/mmff_labels.rs`.

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
    // typify is pairs-free: the neighbour list is the consumer's to build.
    assert!(!frame.contains_key("pairs"));
}

// ---------------------------------------------------------------------------
// Full compile path (typify -> Frame -> to_potentials).
// ---------------------------------------------------------------------------

#[test]
fn methane_typifies_then_compiles_to_finite_energy() {
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
    // so the compile resolves the bond/angle/stretch-bend kernels and yields
    // finite energy + forces.
    let (pots, frame) = compile(&t, &mol).expect("compile potentials");
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
/// `mmff_stbn: unknown '1_37_37_37'`. Now the typifier bakes per-instance
/// stretch-bend params via the RDKit-faithful `ff::mmff::params` resolver (which
/// has the dfsb default-row fallback), so every MMFF term resolves.
#[test]
fn benzene_compile_resolves_stretch_bend() {
    let t = typifier();
    let mol = benzene();
    let (pots, frame) =
        compile(&t, &mol).expect("benzene should resolve all MMFF terms (incl. stretch-bend)");
    let coords = molrs::ff::potential::extract_coords(&frame).expect("coords");
    let (e, forces) = pots.calc_energy_forces(&coords);
    assert!(e.is_finite(), "benzene energy not finite: {e}");
    assert!(
        forces.iter().all(|f| f.is_finite()),
        "non-finite benzene force"
    );
}
