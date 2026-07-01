//! Geometric hydrogen-bond criterion.
//!
//! Ported from TRAVIS `CHBond` (`src/hbond.cpp`, the `m_frAD` / `m_frAH` /
//! `m_fwinkel` gate around lines 914–961): a D–H···A bond requires the
//! acceptor–donor distance `rAD` below a cutoff, the acceptor–hydrogen distance
//! `rAH` below a cutoff, and an angle test on `cos(winkel)` where `winkel` is the
//! angle **at the donor** between the D→A and D→H vectors (TRAVIS default
//! ≤ 30° from collinear).
//!
//! **Documented deviation (per the spec's acceptance contract):** molrs gates on
//! the Luzar–Chandler **D–H···A angle at the hydrogen** (Luzar & Chandler,
//! *Nature* **1996**, 379, 55), with the common defaults r(D···A) ≤ 3.5 Å and
//! ∠(D–H···A) ≥ 150° (i.e. ≤ 30° from the linear 180°). Both conventions encode
//! the same near-linear-bond intent; molrs uses the angle-at-H form because that
//! is the quantity the spec and the downstream lifetime TCFs reference.

use molrs::types::F;

/// Which interatomic distance the cutoff applies to when pairing candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistKind {
    /// Cutoff on the donor–acceptor distance r(D···A) (TRAVIS `m_frAD`).
    DonorAcceptor,
    /// Cutoff on the hydrogen–acceptor distance r(H···A) (TRAVIS `m_frAH`).
    HydrogenAcceptor,
}

/// Frozen geometric parameters for hydrogen-bond detection.
///
/// `Default` is the Luzar–Chandler water convention: r(D···A) ≤ 3.5 Å,
/// ∠(D–H···A) ≥ 150°.
#[derive(Debug, Clone, Copy)]
pub struct HBondCriterion {
    /// Distance cutoff in ångström (applies to `dist_kind`).
    pub dist_cutoff: F,
    /// Which distance the cutoff gates.
    pub dist_kind: DistKind,
    /// Minimum D–H···A angle in **degrees** (measured at the hydrogen).
    pub angle_cutoff: F,
}

impl Default for HBondCriterion {
    fn default() -> Self {
        Self {
            dist_cutoff: 3.5,
            dist_kind: DistKind::DonorAcceptor,
            angle_cutoff: 150.0,
        }
    }
}

impl HBondCriterion {
    /// Construct an explicit criterion.
    pub fn new(dist_cutoff: F, dist_kind: DistKind, angle_cutoff: F) -> Self {
        Self {
            dist_cutoff,
            dist_kind,
            angle_cutoff,
        }
    }
}
