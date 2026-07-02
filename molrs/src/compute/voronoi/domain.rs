//! Domain (microheterogeneity) analysis over a radical-Voronoi tessellation.
//!
//! Merges face-adjacent cells that share the same user label into connected
//! domains via union-find over the cell-adjacency graph — the aggregation reference implementation
//! performs in `src/domain.cpp` / `src/posdomain.cpp` (e.g. polar vs. apolar
//! domains in ionic liquids). Returns the domain size distribution, count, and
//! largest-domain fraction.

use molrs::types::F;

use super::UnionFind;
use super::cell::VoronoiCells;
use crate::compute::error::ComputeError;

/// Outcome of a [`DomainAnalysis`].
#[derive(Debug, Clone)]
pub struct DomainResult {
    /// Domain sizes (atoms per domain), descending.
    pub sizes: Vec<usize>,
    /// Number of domains.
    pub count: usize,
    /// Fraction of labelled atoms in the largest domain.
    pub largest_fraction: F,
    /// Domain id (a representative cell index) per cell.
    pub domain_of: Vec<usize>,
}

/// Partition cells into same-label face-adjacent domains.
#[derive(Debug, Clone, Copy, Default)]
pub struct DomainAnalysis;

impl DomainAnalysis {
    /// Merge face-adjacent cells sharing the same `labels[i]` into domains.
    /// `labels` length must equal the cell count.
    pub fn analyze(
        &self,
        cells: &VoronoiCells,
        labels: &[i64],
    ) -> Result<DomainResult, ComputeError> {
        let n = cells.len();
        if labels.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: labels.len(),
                what: "domain labels length",
            });
        }
        let mut uf = UnionFind::new(n);
        for i in 0..n {
            for j in cells.neighbors(i) {
                let j = j as usize;
                if j < n && j > i && labels[i] == labels[j] {
                    uf.union(i, j);
                }
            }
        }

        // Domain ids are cell indices in `0..n`, so a flat `Vec` keyed by the
        // union-find root replaces the `HashMap` (no hashing): each root's tally is
        // an integer count, and roots with a non-zero count are exactly the
        // `HashMap`'s keys — an identical size multiset.
        let mut domain_of = vec![0usize; n];
        let mut size_of = vec![0usize; n];
        for (i, d) in domain_of.iter_mut().enumerate() {
            let r = uf.find(i);
            *d = r;
            size_of[r] += 1;
        }

        let mut sizes: Vec<usize> = size_of.into_iter().filter(|&c| c > 0).collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let count = sizes.len();
        let largest_fraction = if n == 0 {
            0.0
        } else {
            sizes.first().copied().unwrap_or(0) as F / n as F
        };

        Ok(DomainResult {
            sizes,
            count,
            largest_fraction,
            domain_of,
        })
    }
}
