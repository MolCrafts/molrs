//! MMFF94 potential kernels (bond/angle/torsion/oop/vdw/ele).
//!
//! The MMFF kernel structs (`MMFFBondStretch`, `MMFFAngleBend`, ...) expose no
//! public constructor — they are built only through `mmff_*_ctor` during
//! `ForceField::to_potentials`. The public path is therefore
//! typify → `Frame` → `to_potentials`, which we drive here with molecules built in
//! code. Direct unit construction of each kernel is covered by inline
//! `#[cfg(test)]` modules in src.
//!
//! All MMFF kernels are wired: stretch-bend merges its per-angle `r0_ij` /
//! `r0_kj` / `theta0`, and out-of-plane (`mmff_oop`) reads the `koop` the typifier
//! resolved for each trigonal centre — so the compile yields a complete
//! `Potentials` with finite energy and forces.
//!
//! Six of the seven styles are `ParamSource::PerInstance`: their kernels read
//! Frame columns and carry **zero** type rows. `pair/mmff_vdw` is the exception
//! and keeps its real 95-row table. That boundary is asserted in
//! `tests/ff/potential/param_source.rs`.

use molrs::ff::potential::{Potentials, intramolecular_pairs};
use molrs::ff::typifier::mmff::MMFF94Typifier;
use molrs::store::frame::Frame;
use molrs::system::molgraph::PropValue;
use molrs::types::F;
use molrs::{AtomId, Atomistic};

fn typifier() -> MMFF94Typifier {
    MMFF94Typifier::new()
}

/// The standard route: typify → `Frame` (+ neighbour list) → `to_potentials`.
fn compile(t: &MMFF94Typifier, mol: &Atomistic) -> Result<(Potentials, Frame), String> {
    let mut frame = t.typify(mol)?.to_frame();
    frame.insert("pairs", intramolecular_pairs(&frame));
    let pots = t.ff().to_potentials(&frame)?;
    Ok((pots, frame))
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
    // for compile. The neighbour list (`pairs`) is the consumer's, not typify's.
    let mol = ethane();
    let frame = typifier().typify(&mol).expect("typify ethane").to_frame();
    assert_eq!(frame.get("atoms").unwrap().nrows(), Some(8));
    assert_eq!(frame.get("bonds").unwrap().nrows(), Some(7));
    assert_eq!(frame.get("angles").unwrap().nrows(), Some(12));
    assert_eq!(frame.get("dihedrals").unwrap().nrows(), Some(9));
    // typify is pairs-free — the consumer owns the neighbour list.
    assert!(!frame.contains_key("pairs"));
    // The angles block carries the stretch-bend type column the stbn kernel
    // reads — confirming the topology side is wired up.
    assert!(frame.get("angles").unwrap().contains_key("stbn_type"));
}

#[test]
fn ethane_compiles_with_all_kernels() {
    // Ethane's carbons are four-coordinate, so MMFF defines no out-of-plane
    // term; every kernel resolves (stretch-bend params merged, oop correctly
    // skipped) and the compile yields finite energy + forces.
    let mol = ethane();
    let (pots, frame) = compile(&typifier(), &mol).expect("compile potentials");
    let coords = molrs::ff::potential::extract_coords(&frame).expect("coords");
    let (e, forces) = pots.calc_energy_forces(&coords);
    assert!(e.is_finite(), "energy not finite: {e}");
    assert!(
        forces.iter().all(|f| f.is_finite()),
        "non-finite force component"
    );
}
