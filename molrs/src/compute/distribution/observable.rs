//! The `Observable` abstraction: frame + atom-group selection → scalar samples.
//!
//! Mirrors the reference implementation's separation between *what* atoms an analysis runs over
//! (`CAtomGroup` / the observation list assembled in `src/tddf.cpp` and
//! `src/geodens.cpp`) and the geometric quantity extracted per tuple. Here the
//! selection is the frozen [`AtomGroups`] index container and the extractor is
//! any [`Observable`] (distance / angle / dihedral).

use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use crate::compute::error::ComputeError;
use crate::compute::util::{MicHelper, Positions, get_positions_ref};

/// A frozen container of atom-index tuples, all of one arity.
///
/// `arity` is 2 for a distance, 3 for an angle (vertex in the middle), 4 for a
/// dihedral. Indices are stored row-major in `flat` (`flat.len() == arity *
/// n_groups`). This is the minimal selection surface; richer selection (SMARTS,
/// a DSL) is explicitly out of scope for this link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomGroups {
    arity: usize,
    flat: Vec<u32>,
}

impl AtomGroups {
    /// Build from row-major flat indices. Returns
    /// [`ComputeError::BadShape`] if `flat.len()` is not a multiple of `arity`,
    /// or [`ComputeError::OutOfRange`] if `arity` is 0.
    pub fn new(arity: usize, flat: Vec<u32>) -> Result<Self, ComputeError> {
        if arity == 0 {
            return Err(ComputeError::OutOfRange {
                field: "AtomGroups::arity",
                value: "0".to_string(),
            });
        }
        if !flat.len().is_multiple_of(arity) {
            return Err(ComputeError::BadShape {
                expected: format!("multiple of arity {arity}"),
                got: format!("{} indices", flat.len()),
            });
        }
        Ok(Self { arity, flat })
    }

    /// Convenience: pairs (arity 2) from `[(i, j), ...]`.
    pub fn pairs(tuples: &[(u32, u32)]) -> Self {
        let mut flat = Vec::with_capacity(tuples.len() * 2);
        for &(i, j) in tuples {
            flat.push(i);
            flat.push(j);
        }
        Self { arity: 2, flat }
    }

    /// Convenience: triples (arity 3) from `[(i, j, k), ...]` with `j` the vertex.
    pub fn triples(tuples: &[(u32, u32, u32)]) -> Self {
        let mut flat = Vec::with_capacity(tuples.len() * 3);
        for &(i, j, k) in tuples {
            flat.push(i);
            flat.push(j);
            flat.push(k);
        }
        Self { arity: 3, flat }
    }

    /// Convenience: quadruples (arity 4) from `[(i, j, k, l), ...]`.
    pub fn quads(tuples: &[(u32, u32, u32, u32)]) -> Self {
        let mut flat = Vec::with_capacity(tuples.len() * 4);
        for &(i, j, k, l) in tuples {
            flat.push(i);
            flat.push(j);
            flat.push(k);
            flat.push(l);
        }
        Self { arity: 4, flat }
    }

    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Number of tuples.
    pub fn len(&self) -> usize {
        self.flat.len() / self.arity
    }

    pub fn is_empty(&self) -> bool {
        self.flat.is_empty()
    }

    /// The `i`-th tuple as a slice of length `arity`.
    pub fn tuple(&self, i: usize) -> &[u32] {
        &self.flat[i * self.arity..(i + 1) * self.arity]
    }
}

/// A stateless per-frame extractor: each selected tuple → one scalar sample.
///
/// The contract mirrors the stateless [`Compute`](crate::compute::traits::Compute)
/// trait: `&self` is an immutable parameter bag and identical inputs yield
/// identical samples.
pub trait Observable {
    /// Atom indices consumed per sample (2 / 3 / 4).
    fn arity(&self) -> usize;

    /// Whether samples are angular (radians), enabling the sin θ ADF correction.
    fn is_angular(&self) -> bool {
        false
    }

    /// The closed `[min, max]` range of the sample, used to size the histogram
    /// when the caller does not override it (distance has no natural max, so it
    /// returns `None`).
    fn natural_range(&self) -> Option<(F, F)> {
        None
    }

    /// Extract one scalar per group from `frame`. Returns
    /// [`ComputeError::BadShape`] if `groups.arity()` ≠ [`arity`](Self::arity),
    /// and propagates degenerate-geometry errors (e.g. zero-length vectors).
    fn sample<FA: FrameAccess>(
        &self,
        frame: &FA,
        groups: &AtomGroups,
    ) -> Result<Vec<F>, ComputeError>;

    /// Sample into a caller-provided buffer (cleared first), reusing its
    /// allocation across frames.
    ///
    /// The default delegates to [`sample`](Self::sample); the built-in
    /// observables override it to fill `out` in place, so the per-frame
    /// histogram loop avoids allocating a fresh `Vec` for every frame. Values
    /// are identical to [`sample`](Self::sample) — this is a pure allocation
    /// optimization.
    fn sample_into<FA: FrameAccess>(
        &self,
        frame: &FA,
        groups: &AtomGroups,
        out: &mut Vec<F>,
    ) -> Result<(), ComputeError> {
        out.clear();
        out.extend(self.sample(frame, groups)?);
        Ok(())
    }
}

/// The x/y/z columns of a frame's `atoms` block as three parallel slices.
///
/// Each entry is a [`Positions`] — a zero-copy borrow for the contiguous
/// `Frame`/`FrameView` case, an owned `Vec` only for non-contiguous views.
/// Reach the raw `&[F]` via [`Positions::slice`].
pub(crate) type PosCols<'a> = (Positions<'a>, Positions<'a>, Positions<'a>);

/// Borrow the `atoms` x/y/z columns as three parallel slices.
///
/// Thin wrapper over [`compute::util::get_positions_ref`](crate::compute::util)
/// (the shared column extractor). Unlike a materialized `N×3` array this keeps
/// the columns in their native SoA layout — observables index `xs[i]`/`ys[i]`/
/// `zs[i]` directly, with no per-frame copy.
pub(crate) fn positions<FA: FrameAccess>(frame: &FA) -> Result<PosCols<'_>, ComputeError> {
    let (xp, yp, zp) = get_positions_ref(frame)?;
    let n = xp.slice().len();
    if yp.slice().len() != n || zp.slice().len() != n {
        return Err(ComputeError::DimensionMismatch {
            expected: n,
            got: yp.slice().len().min(zp.slice().len()),
            what: "atoms x/y/z length",
        });
    }
    Ok((xp, yp, zp))
}

/// Minimum-image displacement `b - a` using a per-frame [`MicHelper`] hoisted by
/// the caller (built once with [`MicHelper::from_simbox`] rather than resolved
/// per pair). Free boundaries fall back to the raw separation. This is the one
/// minimum-image implementation across `compute`, so distance DFs agree with
/// [`compute::rdf`](crate::compute::rdf) on the same pair (ac-003).
#[inline]
pub(crate) fn displacement(
    mic: &MicHelper,
    xs: &[F],
    ys: &[F],
    zs: &[F],
    a: usize,
    b: usize,
) -> [F; 3] {
    let from = [xs[a], ys[a], zs[a]];
    let to = [xs[b], ys[b], zs[b]];
    mic.disp(from, to)
}

pub(crate) fn norm(v: [F; 3]) -> F {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

pub(crate) fn dot(a: [F; 3], b: [F; 3]) -> F {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [F; 3], b: [F; 3]) -> [F; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomgroups_validates_arity_multiple() {
        assert!(AtomGroups::new(3, vec![0, 1, 2, 3]).is_err());
        assert!(AtomGroups::new(0, vec![]).is_err());
        let g = AtomGroups::new(2, vec![0, 1, 2, 3]).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g.tuple(1), &[2, 3]);
    }

    #[test]
    fn empty_groups_report_empty() {
        let g = AtomGroups::pairs(&[]);
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }
}
