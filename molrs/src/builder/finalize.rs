//! [`Finalizer`]: the composable step that completes an assembled molecule's
//! bonded topology (angles, dihedrals and, when asked, impropers).
//!
//! A force field scores more than bonds: an **angle** is three atoms bonded in
//! sequence (i–j–k), a **dihedral** (torsion) is four atoms bonded in sequence
//! (i–j–k–l), and an **improper** is a central atom with three bonded
//! neighbours, used to keep planar groups flat. All three follow from the bond
//! graph, and an assembly only writes bonds, so they must be generated
//! afterwards.
//!
//! `Finalizer` is a deliberate second entry point to
//! [`Atomistic::generate_topology`]: a thin, Python-visible pipeline step that
//! replaces molpy's `StructureFinalizer`. It is not a second implementation —
//! every angle, dihedral and improper it adds is perceived by
//! `generate_topology` itself.

use crate::error::MolRsError;
use crate::system::atomistic::Atomistic;

/// Completes the angles, dihedrals and (optionally) impropers of a molecule
/// from its bond graph.
///
/// Angles and dihedrals are always generated: every assembled molecule needs
/// both, and no caller turns them off. Impropers are the one knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Finalizer {
    impropers: bool,
}

impl Finalizer {
    /// A finalizer that also generates impropers when `impropers` is `true`.
    pub fn new(impropers: bool) -> Self {
        Self { impropers }
    }

    /// Add every angle and dihedral (and, when configured, improper) that the
    /// bond graph of `mol` implies and `mol` does not yet hold.
    ///
    /// Delegates to [`Atomistic::generate_topology`] with
    /// `clear_existing = false`, so it is idempotent: a second call adds
    /// nothing. Returns `(n_angles_added, n_dihedrals_added,
    /// n_impropers_added)`.
    ///
    /// An [`Assembler`](crate::builder::Assembler) returns a
    /// [`Fragment`](crate::system::fragment::Fragment); it becomes the
    /// `Atomistic` this takes through
    /// `Atomistic::try_from_molgraph(fragment.into_inner())`
    /// ([`Atomistic::try_from_molgraph`]). The `ports` relation kind stays on
    /// it, with any unlinked ports, and is not part of the bonded topology.
    ///
    /// # Errors
    ///
    /// The error of [`Atomistic::generate_topology`].
    pub fn finalize(&self, mol: &mut Atomistic) -> Result<(usize, usize, usize), MolRsError> {
        mol.generate_topology(true, true, self.impropers, false)
    }
}

#[cfg(test)]
mod tests {
    use super::Finalizer;
    use crate::system::atomistic::Atomistic;

    // Hand-derived from the bond graph C0–C1–C2–C3 (n-butane's heavy-atom
    // skeleton): angles are the 2-edge paths C0–C1–C2 and C1–C2–C3 (2),
    // proper dihedrals the 3-edge paths, here only C0–C1–C2–C3 (1). With
    // impropers off none is generated. `generate_topology` is idempotent by
    // canonical endpoints, so a second pass adds nothing. No external
    // program produced any value here.

    /// A bare C–C–C–C chain with bonds only.
    fn butane_skeleton() -> Atomistic {
        let mut mol = Atomistic::new();
        let c: Vec<_> = (0..4).map(|_| mol.add_atom_bare("C")).collect();
        for w in c.windows(2) {
            mol.add_bond(w[0], w[1]).expect("fixture bond");
        }
        mol
    }

    #[test]
    fn finalize_adds_missing_angles_and_dihedrals_once() {
        let mut mol = butane_skeleton();
        let finalizer = Finalizer::new(false);

        let first = finalizer.finalize(&mut mol).expect("first finalize");
        assert_eq!(
            first,
            (2, 1, 0),
            "C–C–C–C: 2 angles, 1 dihedral, no impropers"
        );
        assert_eq!(mol.n_angles(), 2);
        assert_eq!(mol.n_dihedrals(), 1);
        assert_eq!(mol.n_impropers(), 0);

        let second = finalizer.finalize(&mut mol).expect("second finalize");
        assert_eq!(second, (0, 0, 0), "a second pass adds nothing");
        assert_eq!(mol.n_angles(), 2);
        assert_eq!(mol.n_dihedrals(), 1);
    }
}
