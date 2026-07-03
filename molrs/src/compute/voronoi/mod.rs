//! Radical (Laguerre) Voronoi tessellation + its first two consumers.
//!
//! Gated behind the `voronoi` feature (which implies `compute`). The default
//! backend is a **native pure-Rust** cell-by-cell radical tessellation
//! ([`RadicalVoronoi`]) — no C/C++ FFI, WASM-clean — ported from voro++
//! (`src/v_cell.cpp`, `src/v_rad_option.h`, `src/v_container_prd.cpp`) as used
//! by the reference implementation (`vorowrapper.cpp`). Two real consumers ship with it:
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
