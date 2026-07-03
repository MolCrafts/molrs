//! Void (cavity / free-volume) analysis over a radical-Voronoi tessellation.
//!
//! Following the reference implementation's domain-style void aggregation (`src/void.cpp`): a set of
//! **probe generators** is tessellated *together with* the atoms, and the cells
//! belonging to probes are the unoccupied regions. Face-adjacent probe cells are
//! merged (connected-components) into cavities; each cavity's volume is the sum
//! of its probe-cell volumes, and the total void fraction is the probe volume
//! over the box volume.
//!
//! The caller builds one [`VoronoiCells`] over `atoms ++ probes` and passes a
//! boolean mask marking which generators are probes — keeping this a pure
//! consumer of the tessellation (no second geometry path).

use molrs::types::F;

use super::cell::VoronoiCells;
use crate::compute::error::ComputeError;
use crate::core::system::topology::Topology;

/// Outcome of a [`VoidAnalysis`].
#[derive(Debug, Clone)]
pub struct VoidResult {
    /// Cavity volumes (Å³), descending.
    pub cavity_volumes: Vec<F>,
    /// Total unoccupied (probe) volume (Å³).
    pub total_void_volume: F,
    /// Void fraction = total void volume / box volume.
    pub void_fraction: F,
}

/// Aggregate probe cells of a combined atom+probe tessellation into cavities.
#[derive(Debug, Clone, Copy, Default)]
pub struct VoidAnalysis;

impl VoidAnalysis {
    /// `is_void[i]` marks cell `i` as a void probe. Adjacent probe cells merge
    /// into one cavity. `box_volume` normalizes the void fraction.
    pub fn analyze(
        &self,
        cells: &VoronoiCells,
        is_void: &[bool],
        box_volume: F,
    ) -> Result<VoidResult, ComputeError> {
        let n = cells.len();
        if is_void.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: is_void.len(),
                what: "void mask length",
            });
        }
        if !box_volume.is_finite() || box_volume <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "VoidAnalysis::box_volume",
                value: box_volume.to_string(),
            });
        }

        // Build the void-cell adjacency graph and cluster it with the native
        // connected-components (BFS) facility instead of a local union-find:
        // only void cells are linked; non-void cells stay isolated and never
        // contribute.
        let mut topo = Topology::with_atoms(n);
        for (i, &vi) in is_void.iter().enumerate() {
            if !vi {
                continue;
            }
            for j in cells.neighbors(i) {
                let j = j as usize;
                if j < n && j > i && is_void[j] {
                    topo.add_bond(i, j);
                }
            }
        }
        let component_of = topo.connected_components();

        // Cavity ids are contiguous 0-based component labels, so flat `Vec`s
        // keyed by that label replace the `HashMap` (no hashing). Connected
        // components are a graph invariant, so each cavity holds the same cells
        // as the old union-find roots, and volumes are still summed in ascending
        // `i` order → bit-identical floats; `seen` marks the labels that owned
        // ≥ 1 probe cell.
        let mut vol_of = vec![0.0 as F; n];
        let mut seen = vec![false; n];
        let mut total = 0.0;
        for (i, &vi) in is_void.iter().enumerate() {
            if !vi {
                continue;
            }
            let c = component_of[i] as usize;
            vol_of[c] += cells.volumes[i];
            seen[c] = true;
            total += cells.volumes[i];
        }

        let mut cavity_volumes: Vec<F> = (0..n).filter(|&c| seen[c]).map(|c| vol_of[c]).collect();
        cavity_volumes.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());

        Ok(VoidResult {
            cavity_volumes,
            total_void_volume: total,
            void_fraction: total / box_volume,
        })
    }
}
