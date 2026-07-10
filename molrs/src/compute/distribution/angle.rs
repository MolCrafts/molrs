//! Angle observable: the angle θ ∈ [0, π] at the vertex of each atom triple.
//!
//! Ported from the reference implementation's angular distribution (`src/df.cpp` ADF mode; the angle
//! itself is `Angle`/`Angle_Deg` in `src/xdvector3.cpp`, here kept in radians):
//! for a triple i–j–k with `j` the vertex, `θ = arccos((r_ij · r_kj) /
//! (|r_ij| |r_kj|))`. The dot-product argument is clamped to `[-1, 1]` so a
//! collinear triple yields exactly 0 or π instead of a rounding NaN.

use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use crate::compute::error::ComputeError;

use crate::compute::util::MicHelper;

use super::observable::{AtomGroups, Observable, displacement, dot, norm, positions};

/// Angle θ ∈ [0, π] (radians) at atom `j` of each triple i–j–k (arity 3).
#[derive(Debug, Clone, Default)]
pub struct AngleObservable;

impl Observable for AngleObservable {
    fn arity(&self) -> usize {
        3
    }

    fn is_angular(&self) -> bool {
        true
    }

    fn natural_range(&self) -> Option<(F, F)> {
        Some((0.0, std::f64::consts::PI))
    }

    fn sample<FA: FrameAccess>(
        &self,
        frame: &FA,
        groups: &AtomGroups,
    ) -> Result<Vec<F>, ComputeError> {
        let mut out = Vec::with_capacity(groups.len());
        self.sample_into(frame, groups, &mut out)?;
        Ok(out)
    }

    fn sample_into<FA: FrameAccess>(
        &self,
        frame: &FA,
        groups: &AtomGroups,
        out: &mut Vec<F>,
    ) -> Result<(), ComputeError> {
        if groups.arity() != 3 {
            return Err(ComputeError::BadShape {
                expected: "arity 3".to_string(),
                got: format!("arity {}", groups.arity()),
            });
        }
        let (xp, yp, zp) = positions(frame)?;
        let (xs, ys, zs) = (xp.slice(), yp.slice(), zp.slice());
        let mic = MicHelper::from_simbox(frame.simbox_ref());
        out.clear();
        out.reserve(groups.len());
        for g in 0..groups.len() {
            let t = groups.tuple(g);
            let (i, j, k) = (t[0] as usize, t[1] as usize, t[2] as usize);
            // r_ij = i - j, r_kj = k - j (j is the vertex).
            let r_ij = displacement(&mic, xs, ys, zs, j, i);
            let r_kj = displacement(&mic, xs, ys, zs, j, k);
            let n_ij = norm(r_ij);
            let n_kj = norm(r_kj);
            if n_ij == 0.0 || n_kj == 0.0 {
                return Err(ComputeError::NonFinite {
                    where_: "AngleObservable: zero-length bond vector",
                    index: g,
                });
            }
            let cos = (dot(r_ij, r_kj) / (n_ij * n_kj)).clamp(-1.0, 1.0);
            out.push(cos.acos());
        }
        Ok(())
    }
}
