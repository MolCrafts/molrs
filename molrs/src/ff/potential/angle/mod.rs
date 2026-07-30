//! Angle potential kernels.

pub mod class2;
pub mod harmonic;
pub mod mmff;
pub mod uff;

pub use class2::{AngleClass2, angle_class2_ctor};
pub use harmonic::{AngleHarmonic, angle_harmonic_ctor};
pub use mmff::{MMFFAngleBend, MMFFStretchBend, mmff_angle_ctor, mmff_stbn_ctor};
pub use uff::{UffAngle, uff_angle_ctor};

use molrs::types::F;

/// Distribute an angle bending force onto the three atoms `i-j-k` (vertex `j`)
/// given `de_dtheta = dE/dtheta`.
///
/// The angle-geometry chain rule (force = `dE/dtheta / sin(theta)` times the
/// gradient of `cos(theta)`) is independent of the specific bending potential,
/// so every angle kernel routes its `dE/dtheta` through this one helper.
pub(crate) fn accumulate_angle_forces(
    coords: &[F],
    i: usize,
    j: usize,
    k: usize,
    de_dtheta: F,
    forces: &mut [F],
) {
    let rji = [
        coords[i * 3] - coords[j * 3],
        coords[i * 3 + 1] - coords[j * 3 + 1],
        coords[i * 3 + 2] - coords[j * 3 + 2],
    ];
    let rjk = [
        coords[k * 3] - coords[j * 3],
        coords[k * 3 + 1] - coords[j * 3 + 1],
        coords[k * 3 + 2] - coords[j * 3 + 2],
    ];

    let d_ji_sq = rji[0] * rji[0] + rji[1] * rji[1] + rji[2] * rji[2];
    let d_jk_sq = rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2];
    let d_ji = d_ji_sq.sqrt();
    let d_jk = d_jk_sq.sqrt();
    if d_ji < 1e-12 || d_jk < 1e-12 {
        return;
    }

    let dot = rji[0] * rjk[0] + rji[1] * rjk[1] + rji[2] * rjk[2];
    let cos_theta = (dot / (d_ji * d_jk)).clamp(-1.0, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(1e-12);
    let prefactor = de_dtheta / sin_theta;

    for dim in 0..3 {
        let dcos_dri = rjk[dim] / (d_ji * d_jk) - cos_theta * rji[dim] / d_ji_sq;
        let dcos_drk = rji[dim] / (d_ji * d_jk) - cos_theta * rjk[dim] / d_jk_sq;
        let dcos_drj = -dcos_dri - dcos_drk;

        forces[i * 3 + dim] += prefactor * dcos_dri;
        forces[k * 3 + dim] += prefactor * dcos_drk;
        forces[j * 3 + dim] += prefactor * dcos_drj;
    }
}
