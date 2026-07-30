//! Dihedral potential kernels.

pub mod charmm;
pub mod class2;
pub mod mmff;
pub mod multi_harmonic;
pub mod opls;
pub mod periodic;
pub mod uff;

pub use charmm::{DihedralCharmm, dihedral_charmm_ctor};
pub use class2::{DihedralClass2, dihedral_class2_ctor};
pub use mmff::{MMFFTorsion, mmff_torsion_ctor};
pub use multi_harmonic::{DihedralMultiHarmonic, dihedral_multi_harmonic_ctor};
pub use opls::{DihedralOPLS, dihedral_opls_ctor};
pub use periodic::{DihedralPeriodic, dihedral_periodic_ctor};
pub use uff::{UffTorsion, uff_torsion_ctor};
