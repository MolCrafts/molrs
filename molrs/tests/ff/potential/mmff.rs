//! MMFF94 potential kernels (bond/angle/torsion/oop/vdw/ele).
//!
//! The MMFF kernel structs (`MMFFBondStretch`, `MMFFAngleBend`, ...) expose no
//! public constructor — they are built only through `mmff_*_ctor` during
//! `ForceField::to_potentials`. The genuine public path is therefore
//! `MMFFTypifier::build(mol) -> Potentials`, which we drive here with molecules
//! built in code. Direct unit construction of each kernel is covered by inline
//! `#[cfg(test)]` modules in src.
//!
//! All generic-path MMFF kernels are wired: stretch-bend merges its per-angle
//! `r0_ij` / `r0_kj` / `theta0` (via the typifier's `merge_stbn_r0`), and
//! out-of-plane (`mmff_oop`) resolves each trigonal centre's `koop` through the
//! canonical equivalence-degraded key the typifier emits, so `build()` yields a
//! complete `Potentials` with finite energy and forces.

use molrs::ff::typifier::mmff::MMFFTypifier;
use molrs::system::molgraph::PropValue;
use molrs::types::F;
use molrs::{AtomId, Atomistic};

fn typifier() -> MMFFTypifier {
    MMFFTypifier::mmff94().expect("load embedded MMFF94")
}

fn bond(mol: &mut Atomistic, a: AtomId, b: AtomId, order: F) {
    if let Ok(bid) = mol.add_bond(a, b) {
        let _ = mol.set_bond_prop(bid, "order", PropValue::F64(order));
    }
}

/// Ethane (C2H6) with explicit hydrogens at a plausible geometry.
fn ethane() -> Atomistic {
    let mut mol = Atomistic::new();
    let positions = [
        ("C", [0.0, 0.0, 0.0]),
        ("C", [1.54, 0.0, 0.0]),
        ("H", [-0.36, 1.03, 0.0]),
        ("H", [-0.36, -0.51, 0.89]),
        ("H", [-0.36, -0.51, -0.89]),
        ("H", [1.90, 1.03, 0.0]),
        ("H", [1.90, -0.51, 0.89]),
        ("H", [1.90, -0.51, -0.89]),
    ];
    let ids: Vec<AtomId> = positions
        .iter()
        .map(|(s, [x, y, z])| mol.add_atom_xyz(s, *x, *y, *z))
        .collect();
    bond(&mut mol, ids[0], ids[1], 1.0);
    for h in 2..5 {
        bond(&mut mol, ids[0], ids[h], 1.0);
    }
    for h in 5..8 {
        bond(&mut mol, ids[1], ids[h], 1.0);
    }
    mol
}

#[test]
fn ethane_typifies_to_a_complete_frame() {
    // The typification half: `typify` returns a labeled Atomistic that
    // materializes (`to_frame`) into atoms/bonds/angles/dihedrals blocks ready
    // for compile. The neighbour list (`pairs`) is `build()`'s job, not typify's.
    let mol = ethane();
    let frame = typifier().typify(&mol).expect("typify ethane").to_frame();
    assert_eq!(frame.get("atoms").unwrap().nrows(), Some(8));
    assert_eq!(frame.get("bonds").unwrap().nrows(), Some(7));
    assert_eq!(frame.get("angles").unwrap().nrows(), Some(12));
    assert_eq!(frame.get("dihedrals").unwrap().nrows(), Some(9));
    // typify is pairs-free now — `build()` owns the neighbour list.
    assert!(!frame.contains_key("pairs"));
    // The angles block carries the stretch-bend type column the stbn kernel
    // reads — confirming the topology side is wired up.
    assert!(frame.get("angles").unwrap().contains_key("stbn_type"));
}

#[test]
fn ethane_build_succeeds_with_all_kernels() {
    // Ethane's carbons are four-coordinate, so MMFF defines no out-of-plane
    // term; every kernel resolves (stretch-bend params merged, oop correctly
    // skipped) and build() yields finite energy + forces.
    let mol = ethane();
    let pots = typifier().build(&mol).expect("build potentials");
    let frame = typifier().typify(&mol).expect("typify").to_frame();
    let coords = molrs::ff::potential::extract_coords(&frame).expect("coords");
    let (e, forces) = pots.calc_energy_forces(&coords);
    assert!(e.is_finite(), "energy not finite: {e}");
    assert!(
        forces.iter().all(|f| f.is_finite()),
        "non-finite force component"
    );
}
