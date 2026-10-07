//! Radical (Laguerre) Voronoi tessellation + its first two consumers.
//!
//! Gated behind the `voronoi` feature (which implies `compute`). The default
//! backend is a **native pure-Rust** cell-by-cell radical tessellation
//! ([`RadicalVoronoi`]) — no C/C++ FFI, WASM-clean — ported from voro++
//! (`src/v_cell.cpp`, `src/v_rad_option.h`, `src/v_container_prd.cpp`) as used
//! by the reference implementation (`vorowrapper.cpp`). Two real consumers ship with it:
//! [`VoronoiDomainAnalysis`] (microheterogeneity / ionic-liquid domains, `domain.cpp`)
//! and [`VoronoiVoidAnalysis`] (cavity / free-volume, `void.cpp`).
//!
//! Layer: `compute` → `core` (`SimBox`); no new dependency.

mod cell;
mod domain;
mod integrate;
mod polarizability;
mod radical;
mod void;

pub use cell::{VORONOI_BOUNDARY, VoronoiCells, VoronoiFace};
pub use domain::{VoronoiDomainAnalysis, VoronoiDomainResult};
pub use integrate::{DensityGrid, MolecularMoments, VoronoiIntegration};
pub use polarizability::polarizability_finite_field;
pub use radical::RadicalVoronoi;
pub use void::{VoronoiVoidAnalysis, VoronoiVoidResult};
