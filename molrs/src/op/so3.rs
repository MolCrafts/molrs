//! Uniform sampling of rotations (Haar measure on SO(3)) and of directions
//! (uniform on S²), from uniforms or from a seeded counter-based stream.
//!
//! SO(3) is the set of 3D rotation matrices; its **Haar measure** is the one
//! probability distribution over rotations that no fixed rotation can change,
//! the meaning of "every orientation equally likely". S² is the unit sphere of
//! directions in 3D and S³ the unit sphere in 4D, where unit quaternions live.
//!
//! # Rotations
//!
//! Shoemake, "Uniform random rotations", *Graphics Gems III* (1992),
//! doi:10.1016/B978-0-08-050755-2.50036-1. For `u₁, u₂, u₃ ~ U[0, 1)`, with
//! `r₁ = √(1−u₁)`, `r₂ = √u₁`, `θ₁ = 2πu₂`, `θ₂ = 2πu₃`, the quaternion
//! `q = (r₂ cos θ₂, r₁ sin θ₁, r₁ cos θ₁, r₂ sin θ₂)` is uniform on S³, hence
//! its rotation is Haar-distributed. (Uniform axis plus uniform angle is
//! **not** Haar: it gives `E[R] = I/3` and `E[tr R] = 1`, where Haar gives 0.)
//!
//! # Random stream
//!
//! Counter-based SplitMix64 (Steele, Lea & Flood, "Fast splittable
//! pseudorandom number generators", OOPSLA 2014).
//! Lane `lane` of draw `index` is
//! `u = (splitmix64(seed + c·γ) >> 11)·2⁻⁵³` with the counter
//! `c = 4·index + lane` and `γ = 0x9E3779B97F4A7C15`, where `splitmix64(s)` is
//! the output of one SplitMix64 step from state `s` (so this is the `(c+1)`-th
//! output of the sequential stream seeded with `seed`). Every draw is a pure
//! function of `(seed, index)`: a batch can be split, reordered or computed in
//! parallel without changing any value. Rotations use lanes 0–2, angles lane 3.

use crate::op::rigid::quat_to_matrix;
use crate::op::types::{F, Mat3, Quat, Vec3};

use std::f64::consts::TAU;

/// The SplitMix64 increment γ (the odd integer nearest 2⁶⁴/φ).
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// Lanes (independent uniforms) per draw index.
const LANES: u64 = 4;

/// Lane of [`random_angles`]; rotations use lanes 0–2.
const ANGLE_LANE: u64 = 3;

/// One SplitMix64 step from `state`: advance by γ, then mix.
#[inline]
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(GOLDEN_GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in `[0, 1)` for lane `lane` of draw `index`.
#[inline]
fn uniform(seed: u64, index: u64, lane: u64) -> F {
    let c = index.wrapping_mul(LANES).wrapping_add(lane);
    let bits = splitmix64(seed.wrapping_add(c.wrapping_mul(GOLDEN_GAMMA)));
    (bits >> 11) as F * (1.0 / (1u64 << 53) as F)
}

/// Shoemake's unit quaternion `(w, x, y, z)` for `u ∈ [0, 1]³` (module docs);
/// uniform on S³ when `u` is uniform on `[0, 1)³`.
fn quat_from_uniform(u: [F; 3]) -> Quat {
    let r1 = (1.0 - u[0]).sqrt();
    let r2 = u[0].sqrt();
    let (s1, c1) = (TAU * u[1]).sin_cos();
    let (s2, c2) = (TAU * u[2]).sin_cos();
    [r2 * c2, r1 * s1, r1 * c1, r2 * s2]
}

/// The rotation matrix of Shoemake's unit quaternion for `u ∈ [0, 1]³`
/// (module docs); Haar-distributed on SO(3) when `u` is uniform on `[0, 1)³`.
pub fn rotation_from_uniform(u: [F; 3]) -> Mat3 {
    quat_to_matrix(quat_from_uniform(u))
}

/// A unit vector from `u ∈ [0, 1]²`: `z = 2u₀ − 1`, `φ = 2πu₁`,
/// `(√(1−z²) cos φ, √(1−z²) sin φ, z)`. Uniform on the sphere when `u` is
/// uniform on `[0, 1)²` (Archimedes: `z` is uniform on a sphere).
pub fn unit_vector_from_uniform(u: [F; 2]) -> Vec3 {
    let z = 2.0 * u[0] - 1.0;
    let phi = TAU * u[1];
    let r = (1.0 - z * z).max(0.0).sqrt();
    [r * phi.cos(), r * phi.sin(), z]
}

/// Haar-random rotations, one per entry of `indices`: draw `index` of the
/// stream `seed` (lanes 0–2 through [`rotation_from_uniform`]).
///
/// Each rotation depends only on `(seed, index)`, so
/// `random_rotations(s, &[5])[0] == random_rotations(s, &[0, 5])[1]`.
pub fn random_rotations(seed: u64, indices: &[u64]) -> Vec<Mat3> {
    indices
        .iter()
        .map(|&i| {
            rotation_from_uniform([
                uniform(seed, i, 0),
                uniform(seed, i, 1),
                uniform(seed, i, 2),
            ])
        })
        .collect()
}

/// Uniform angles in `[0, 2π)` radians, one per entry of `indices`: `2π·u`
/// with `u` lane 3 of draw `index` of the stream `seed`.
///
/// Each angle depends only on `(seed, index)`.
pub fn random_angles(seed: u64, indices: &[u64]) -> Vec<F> {
    indices
        .iter()
        .map(|&i| TAU * uniform(seed, i, ANGLE_LANE))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::linalg::det3;
    use crate::op::types::{F, Mat3};

    const MAP_TOL: F = 1e-15;
    const SEED: u64 = 20260926;

    fn assert_mat_close(got: &Mat3, want: &Mat3, tol: F, what: &str) {
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (got[r][c] - want[r][c]).abs() < tol,
                    "{what}[{r}][{c}]: expected {want:?}, got {got:?}"
                );
            }
        }
    }

    /// max |(RᵀR − I)_ij|
    fn orthogonality_defect(r: &Mat3) -> F {
        let mut worst: F = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let rtr: F = (0..3).map(|k| r[k][i] * r[k][j]).sum();
                let id = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((rtr - id).abs());
            }
        }
        worst
    }

    // ---------- Shoemake map goldens ----------

    #[test]
    fn rotation_from_uniform_u1_one_is_identity() {
        // r1 = 0, r2 = 1, θ2 = 0 → q = (1, 0, 0, 0).
        let r = rotation_from_uniform([1.0, 0.0, 0.0]);
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_mat_close(&r, &eye, MAP_TOL, "R(1,0,0)");
    }

    #[test]
    fn rotation_from_uniform_origin_is_half_turn_about_y() {
        // r1 = 1, r2 = 0, θ1 = 0 → q = (0, 0, 1, 0).
        let r = rotation_from_uniform([0.0, 0.0, 0.0]);
        let want = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
        assert_mat_close(&r, &want, MAP_TOL, "R(0,0,0)");
    }

    #[test]
    fn rotation_from_uniform_midpoint_is_half_turn_about_x_plus_z() {
        // r1 = r2 = √½, θ1 = θ2 = π/2 → q = (0, √½, 0, √½): half turn about
        // (x̂ + ẑ)/√2, R = 2nnᵀ − I.
        let r = rotation_from_uniform([0.5, 0.25, 0.25]);
        let want = [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]];
        assert_mat_close(&r, &want, MAP_TOL, "R(0.5,0.25,0.25)");
    }

    #[test]
    fn rotation_from_uniform_eighth_theta2_is_quarter_turn_about_z() {
        // r1 = 0, r2 = 1, θ2 = π/4 → q = (cos π/4, 0, 0, sin π/4) = Rz(90°).
        let r = rotation_from_uniform([1.0, 0.0, 0.125]);
        let want = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert_mat_close(&r, &want, MAP_TOL, "R(1,0,0.125)");
    }

    #[test]
    fn quat_from_uniform_u1_one_is_the_identity_quaternion() {
        let q = quat_from_uniform([1.0, 0.0, 0.0]);
        for (d, want) in [1.0, 0.0, 0.0, 0.0].iter().enumerate() {
            assert!((q[d] - want).abs() < MAP_TOL, "q = {q:?}");
        }
    }

    #[test]
    fn quat_from_uniform_is_unit_over_a_grid() {
        // 10 × 10 × 10 cell-centred grid on [0, 1)³: 1000 draws.
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let u = [
                        (i as F + 0.5) / 10.0,
                        (j as F + 0.5) / 10.0,
                        (k as F + 0.5) / 10.0,
                    ];
                    let q = quat_from_uniform(u);
                    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                    assert!((n - 1.0).abs() < 1e-15, "u = {u:?}: |q| = {n}");
                }
            }
        }
    }

    #[test]
    fn unit_vector_from_uniform_equator_at_zero_azimuth_is_x() {
        // z = 2·0.5 − 1 = 0, φ = 0 → (1, 0, 0).
        let v = unit_vector_from_uniform([0.5, 0.0]);
        for (d, want) in [1.0, 0.0, 0.0].iter().enumerate() {
            assert!((v[d] - want).abs() < MAP_TOL, "v = {v:?}");
        }
    }

    // ---------- seeded draws ----------

    #[test]
    fn random_rotations_are_proper_orthogonal() {
        let indices: Vec<u64> = (0..1000).collect();
        let rs = random_rotations(SEED, &indices);
        assert_eq!(rs.len(), 1000);
        for (n, r) in rs.iter().enumerate() {
            let defect = orthogonality_defect(r);
            assert!(defect < 1e-14, "draw {n}: ‖RᵀR − I‖max = {defect}");
            let d = det3(r);
            assert!((d - 1.0).abs() < 1e-14, "draw {n}: det R = {d}");
        }
    }

    #[test]
    fn random_rotations_are_addressed_by_index() {
        let alone = random_rotations(SEED, &[5]);
        let batched = random_rotations(SEED, &[0, 5]);
        assert_eq!(alone[0], batched[1]);
    }

    #[test]
    fn random_rotations_depend_on_the_seed() {
        assert_ne!(
            random_rotations(SEED, &[0]),
            random_rotations(SEED + 1, &[0])
        );
    }

    #[test]
    fn random_angles_are_addressed_by_index() {
        let alone = random_angles(SEED, &[5]);
        let batched = random_angles(SEED, &[0, 5]);
        assert_eq!(alone.len(), 1);
        assert_eq!(batched.len(), 2);
        assert_eq!(alone[0].to_bits(), batched[1].to_bits());
    }

    #[test]
    fn random_rotations_have_haar_first_moments() {
        // Haar measure on SO(3): E[R] = 0, E[tr R] = 0; Var[tr R] = 1 and
        // Var[R_ij] = 1/3, so at N = 10⁴ the bounds are ≥ 5σ.
        // (Axis-plus-uniform-angle sampling gives E[tr R] = 1 and fails.)
        let n = 10_000;
        let indices: Vec<u64> = (0..n).collect();
        let rs = random_rotations(SEED, &indices);
        let mut mean: Mat3 = [[0.0; 3]; 3];
        let mut mean_trace: F = 0.0;
        for r in &rs {
            for (i, (mean_row, r_row)) in mean.iter_mut().zip(r).enumerate() {
                for (m, x) in mean_row.iter_mut().zip(r_row) {
                    *m += x / n as F;
                }
                mean_trace += r_row[i] / n as F;
            }
        }
        assert!(mean_trace.abs() < 0.05, "mean tr R = {mean_trace}");
        for (i, row) in mean.iter().enumerate() {
            for (j, m) in row.iter().enumerate() {
                assert!(m.abs() < 0.03, "mean R[{i}][{j}] = {m}");
            }
        }
    }
}
