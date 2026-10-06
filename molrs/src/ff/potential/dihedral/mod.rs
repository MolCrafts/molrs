//! Dihedral potential kernels.

pub(crate) mod charmm;
pub(crate) mod class2;
pub(crate) mod harmonic;
pub(crate) mod mmff;
pub(crate) mod multi_harmonic;
pub(crate) mod opls;
pub(crate) mod periodic;
pub(crate) mod uff;

pub use charmm::{DihedralCharmm, dihedral_charmm_ctor};
pub use class2::{DihedralClass2, dihedral_class2_ctor};
pub use harmonic::dihedral_harmonic_ctor;
pub use mmff::{MMFFTorsion, mmff_torsion_ctor};
pub use multi_harmonic::{
    DihedralMultiHarmonic, dihedral_multi_harmonic_ctor, dihedral_nharmonic_ctor,
};
pub use opls::{DihedralOPLS, dihedral_opls_ctor};
pub use periodic::{DihedralPeriodic, dihedral_periodic_ctor};
pub use uff::{UffTorsion, uff_torsion_ctor};
