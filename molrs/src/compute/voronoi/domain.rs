//! Domain (microheterogeneity) analysis over a radical-Voronoi tessellation.
//!
//! Merges face-adjacent cells that share the same user label into connected
//! domains via union-find over the cell-adjacency graph — the aggregation TRAVIS
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

        let mut domain_of = vec![0usize; n];
        let mut size_of: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for (i, d) in domain_of.iter_mut().enumerate() {
            let r = uf.find(i);
            *d = r;
            *size_of.entry(r).or_insert(0) += 1;
        }

        let mut sizes: Vec<usize> = size_of.values().copied().collect();
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
