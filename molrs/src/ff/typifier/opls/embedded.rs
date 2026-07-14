//! The shipped OPLS-AA parameter set, assembled from the compiled table.
//!
//! [`OPLSAATypifier::oplsaa`](super::OPLSAATypifier::oplsaa) used to `include_str!`
//! 346 KB of XML and run two parsers over it — one for the potential
//! [`ForceField`], one for the typing metadata — on every construction. Both
//! halves now come from [`crate::ff::params::oplsaa`], which holds the same
//! numbers in molrs units, already converted.
//!
//! The XML readers are not gone: [`OPLSAATypifier::from_xml_str`](super::OPLSAATypifier::from_xml_str)
//! still parses a caller's own OPLS / CL&P / CL&Pol file, layers and all. What is
//! gone is molrs re-parsing *its own* parameter set at runtime.

use crate::ff::constants::VACUUM_DIELECTRIC;
use crate::ff::forcefield::{ForceField, SpecialBonds};
use crate::ff::params::oplsaa::{
    OPLSAA_ANGLES, OPLSAA_ATOMS, OPLSAA_BONDS, OPLSAA_COULOMB_14, OPLSAA_DIHEDRALS, OPLSAA_LJ_14,
    OPLSAA_NAME,
};
use molrs::units::constants::COULOMB_REAL;

use super::meta::{OplsTypeRow, OplsTypingMeta};

/// Build the shipped [`ForceField`].
///
/// Style order is the source file's section order, and it is load-bearing:
/// `ForceField` lookups scan and take the first match.
pub(super) fn force_field() -> ForceField {
    let mut ff = ForceField::new(OPLSAA_NAME);

    let bonds = ff.def_bondstyle("harmonic");
    for row in OPLSAA_BONDS {
        bonds.def_bondtype(row.i, row.j, &[("k0", row.k0), ("r0", row.r0)]);
    }

    let angles = ff.def_anglestyle("harmonic");
    for row in OPLSAA_ANGLES {
        angles.def_angletype(
            row.i,
            row.j,
            row.k,
            &[("k0", row.k0), ("theta0", row.theta0)],
        );
    }

    let dihedrals = ff.def_dihedralstyle("opls");
    for row in OPLSAA_DIHEDRALS {
        dihedrals.def_dihedraltype(
            row.i,
            row.j,
            row.k,
            row.l,
            &[
                ("f1", row.f1),
                ("f2", row.f2),
                ("f3", row.f3),
                ("f4", row.f4),
            ],
        );
    }

    // Atoms carry mass + charge; the LJ pair style carries ε / σ. Charges are
    // per-atom at evaluation time, so `coul/cut` has no rows at all — and the
    // combining rule is the kernel's job, not the table's.
    //
    // But its CONSTANTS are not the kernel's job. `coul/cut` is the buffered Coulomb
    // `E = k·qᵢqⱼ/(D·(r + δ))`; OPLS is the unbuffered case (δ = 0, the semantic
    // default) in vacuum (D = 1.0) with CODATA's k. This style used to be defined
    // with EMPTY params and merely happened to agree with the constant the kernel
    // held privately — the right numbers for the wrong reason. OPLS now says them.
    let atoms = ff.def_atomstyle("full");
    for row in OPLSAA_ATOMS {
        atoms.def_atomtype(row.name, &[("mass", row.mass), ("charge", row.charge)]);
    }

    let lj = ff.def_pairstyle("lj/cut", &[]);
    for row in OPLSAA_ATOMS {
        lj.def_pairtype(
            row.name,
            None,
            &[("epsilon", row.epsilon), ("sigma", row.sigma)],
        );
    }
    ff.def_pairstyle(
        "coul/cut",
        &[("coulomb", COULOMB_REAL), ("dielectric", VACUUM_DIELECTRIC)],
    );

    // OPLS excludes 1-2 / 1-3 (molrs omits them from the neighbour list) and
    // scales 1-4 by the source's own weights.
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, OPLSAA_LJ_14],
        coul: [0.0, 0.0, OPLSAA_COULOMB_14],
    });
    ff
}

/// Build the shipped typing metadata (SMARTS `def`, `overrides`, `priority`,
/// `layer`) — the half of the parameter set no energy test can see, and the half
/// that decides what gets typed at all.
pub(super) fn typing_meta() -> OplsTypingMeta {
    let mut meta = OplsTypingMeta::new();
    for row in OPLSAA_ATOMS {
        meta.insert(
            row.name,
            OplsTypeRow {
                class: row.class.to_owned(),
                def: row.def.map(str::to_owned),
                overrides: row.overrides.iter().map(|s| (*s).to_owned()).collect(),
                priority: row.priority,
                layer: row.layer,
            },
        );
    }
    meta
}
