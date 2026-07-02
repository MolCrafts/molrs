//! Distance observable: minimum-image separation of selected atom pairs.
//!
//! The pairwise distance distribution generalizes the RDF to arbitrary
//! user-selected pairs (reference implementation distance DF, `src/tddf.cpp` /
//! `CTimeDiff`/`CDF` distance mode). The minimum-image convention is delegated
//! to [`SimBox::delta`](molrs::spatial::region::simbox::SimBox::delta) so a
//! distance DF and [`compute::rdf`](crate::compute::rdf) return the same value
//! for the same pair under PBC.

use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use crate::compute::error::ComputeError;

use crate::compute::util::MicHelper;

use super::observable::{AtomGroups, Observable, displacement, norm, positions};

/// Distance between the two atoms of each pair (arity 2), minimum-image under PBC.
#[derive(Debug, Clone, Default)]
pub struct DistanceObservable;

impl Observable for DistanceObservable {
    fn arity(&self) -> usize {
        2
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
        if groups.arity() != 2 {
            return Err(ComputeError::BadShape {
                expected: "arity 2".to_string(),
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
            let d = displacement(&mic, xs, ys, zs, t[0] as usize, t[1] as usize);
            out.push(norm(d));
        }
        Ok(())
    }
}
