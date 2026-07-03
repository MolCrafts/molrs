//! Domain (microheterogeneity) analysis over a radical-Voronoi tessellation.
//!
//! Merges face-adjacent cells that share the same user label into connected
//! domains via the native connected-components over the cell-adjacency graph —
//! the aggregation reference implementation performs in `src/domain.cpp` /
//! `src/posdomain.cpp` (e.g. polar vs. apolar domains in ionic liquids).
//! Returns the domain size distribution, count, and largest-domain fraction.

use molrs::types::F;

use super::cell::VoronoiCells;
use crate::compute::error::ComputeError;
use crate::core::system::topology::Topology;

/// Outcome of a [`DomainAnalysis`].
#[derive(Debug, Clone)]
pub struct DomainResult {
    /// Domain sizes (atoms per domain), descending.
    pub sizes: Vec<usize>,
    /// Number of domains.
    pub count: usize,
    /// Fraction of labelled atoms in the largest domain.
    pub largest_fraction: F,
    /// Domain id (0-based component label) per cell.
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
        // Build the same-label cell-adjacency graph and cluster it with the
        // native connected-components (BFS) facility instead of a local
        // union-find: each cell is a node, and a face-adjacency between two
        // same-label cells is an edge.
        let mut topo = Topology::with_atoms(n);
        for i in 0..n {
            for j in cells.neighbors(i) {
                let j = j as usize;
                if j < n && j > i && labels[i] == labels[j] {
                    topo.add_bond(i, j);
                }
            }
        }
        let component_of = topo.connected_components();

        // `connected_components` labels every cell in `0..n` with a contiguous
        // 0-based component id (isolated cells get their own), so a flat `Vec`
        // keyed by that id tallies domain sizes without hashing. Connected
        // components are a graph invariant, so the partition — and thus the size
        // multiset — is identical to the old union-find roots; only the label
        // integers differ.
        let mut domain_of = vec![0usize; n];
        let mut size_of = vec![0usize; n];
        for (i, d) in domain_of.iter_mut().enumerate() {
            let c = component_of[i] as usize;
            *d = c;
            size_of[c] += 1;
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
