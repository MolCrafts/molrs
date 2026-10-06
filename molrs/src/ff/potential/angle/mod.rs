//! Angle potential kernels.

pub(crate) mod charmm;
pub(crate) mod class2;
pub(crate) mod harmonic;
pub(crate) mod mmff;
pub(crate) mod uff;

pub use charmm::{AngleCharmm, CharmmAngleParams, angle_charmm_ctor};
pub use class2::{AngleClass2, angle_class2_ctor};
pub use harmonic::{AngleHarmonic, angle_harmonic_ctor};
pub use mmff::{MMFFAngleBend, MMFFStretchBend, mmff_angle_ctor, mmff_stbn_ctor};
pub use uff::{UffAngle, uff_angle_ctor};

// The angle-geometry chain rule (force = `dE/dθ / sin θ` times the gradient
// of `cos θ`) is independent of the bending potential, so every angle kernel
// routes its `dE/dθ` through this one helper.
pub(crate) use crate::ff::potential::geometry::accumulate_angle_forces;
