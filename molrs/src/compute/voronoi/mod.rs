//! Radical (Laguerre) Voronoi tessellation + its first two consumers.
//!
//! Gated behind the `voronoi` feature (which implies `compute`). The default
//! backend is a **native pure-Rust** cell-by-cell radical tessellation
//! ([`RadicalVoronoi`]) — no C/C++ FFI, WASM-clean — ported from voro++
//! (`src/v_cell.cpp`, `src/v_rad_option.h`, `src/v_container_prd.cpp`) as used
//! by TRAVIS (`vorowrapper.cpp`). Two real consumers ship with it:
//! [`DomainAnalysis`] (microheterogeneity / ionic-liquid domains, `domain.cpp`)
//! and [`VoidAnalysis`] (cavity / free-volume, `void.cpp`).
//!
//! Layer: `compute` → `core` (`SimBox`); no new dependency.

mod cell;
mod domain;
mod integrate;
mod polarizability;
mod radical;
mod void;

pub use cell::{BOUNDARY, Face, VoronoiCells};
pub use domain::{DomainAnalysis, DomainResult};
pub use integrate::{BOHR_TO_ANG, DensityGrid, MolecularMoments, VoronoiIntegration};
pub use polarizability::polarizability_finite_field;
pub use radical::RadicalVoronoi;
pub use void::{VoidAnalysis, VoidResult};

/// Minimal union-find (path compression + union by size) over cell indices,
/// shared by the domain and void consumers.
pub(crate) struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    pub(crate) fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    pub(crate) fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (big, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
    }
}
