//! The `Observable` abstraction: frame + atom-group selection → scalar samples.
//!
//! Mirrors TRAVIS's separation between *what* atoms an analysis runs over
//! (`CAtomGroup` / the observation list assembled in `src/tddf.cpp` and
//! `src/geodens.cpp`) and the geometric quantity extracted per tuple. Here the
//! selection is the frozen [`AtomGroups`] index container and the extractor is
//! any [`Observable`] (distance / angle / dihedral).

use molrs::spatial::region::simbox::SimBox;
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;
use ndarray::Array2;

use crate::compute::error::ComputeError;

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
}

/// Read the `atoms` x/y/z columns of a frame into an `N×3` array.
///
/// Thin wrapper over [`compute::util::get_positions_ref`](crate::compute::util)
/// (the shared column extractor) materialized into an owned `Array2` so the
/// observables can index per tuple.
pub(crate) fn positions<FA: FrameAccess>(frame: &FA) -> Result<Array2<F>, ComputeError> {
    let (xp, yp, zp) = crate::compute::util::get_positions_ref(frame)?;
    let (xs, ys, zs) = (xp.slice(), yp.slice(), zp.slice());
    let n = xs.len();
    if ys.len() != n || zs.len() != n {
        return Err(ComputeError::DimensionMismatch {
            expected: n,
            got: ys.len().min(zs.len()),
            what: "atoms x/y/z length",
        });
    }
    let mut pos = Array2::<F>::zeros((n, 3));
    for i in 0..n {
        pos[[i, 0]] = xs[i];
        pos[[i, 1]] = ys[i];
        pos[[i, 2]] = zs[i];
    }
    Ok(pos)
}

/// Minimum-image displacement `b - a` when a `SimBox` is present, else the raw
/// separation. Routes through the shared
/// [`compute::util::mic_disp`](crate::compute::util) so there is one
/// minimum-image implementation across `compute`, and distance DFs agree with
/// [`compute::rdf`](crate::compute::rdf) on the same pair (ac-003).
pub(crate) fn displacement(simbox: Option<&SimBox>, pos: &Array2<F>, a: usize, b: usize) -> [F; 3] {
    let from = [pos[[a, 0]], pos[[a, 1]], pos[[a, 2]]];
    let to = [pos[[b, 0]], pos[[b, 1]], pos[[b, 2]]];
    crate::compute::util::mic_disp(simbox, from, to)
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
