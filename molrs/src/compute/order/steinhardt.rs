//! Steinhardt bond-orientational order parameters `q_ℓ` and `w_ℓ`.
//!
//! Mirrors `freud.order.Steinhardt`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/Steinhardt.cc)).
//! Implements:
//!
//! - per-particle `q_ℓm(i) = (1/N_i) Σ_{j ∈ neigh(i)} Y_ℓm(r̂_ij)`
//! - the **averaged** variant (``average = true``) `q̄_ℓm(i) = (1/(N_i+1))
//!   (q_ℓm(i) + Σ_{j ∈ neigh(i)} q_ℓm(j))` — the "near-shell" Steinhardt
//! - the rotational invariant
//!   `q_ℓ(i) = √( (4π/(2ℓ+1)) Σ_m |q_ℓm(i)|² )`
//! - the cubic invariant
//!   `w_ℓ(i) = Σ_{m1+m2+m3=0} (ℓ ℓ ℓ; m1 m2 m3) q_ℓm1(i) q_ℓm2(i) q_ℓm3(i)`
//!   with an optional normalization
//!   `ŵ_ℓ(i) = w_ℓ(i) / ( Σ_m |q_ℓm(i)|² )^{3/2}`.
//!
//! # Conventions
//!
//! - Self-query [`NeighborList`]: each pair `(i, j)` with `i < j` carries the
//!   vector `r_j − r_i`. The Steinhardt accumulator visits each pair once and
//!   updates both particles, exploiting `Y_ℓm(−r̂) = (−1)^ℓ Y_ℓm(r̂)`.
//! - `Y_ℓm` follows the Condon-Shortley + physics-normalization convention
//!   (see [`molrs::math::spherical_harmonics`]).
//!
//! # References
//!
//! - Steinhardt, Nelson & Ronchetti, *Phys. Rev. B* **28**, 784 (1983).
//! - Lechner & Dellago, *J. Chem. Phys.* **129**, 114707 (2008) — averaged
//!   variant.

use crate::compute::result::ComputeResult;
use std::cmp::Ordering;

use molrs::math::complex::Complex;
use molrs::math::spherical_harmonics::ylm_all;
use molrs::math::wigner3j::wigner_3j;
use molrs::spatial::neighbors::NeighborList;
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use crate::compute::error::ComputeError;
use crate::compute::traits::Compute;
use crate::compute::util::get_positions_ref;

const FOUR_PI: F = 4.0 * std::f64::consts::PI;

/// Steinhardt order-parameter calculator.
///
/// Stateless parameter container: configured ℓ values + variant flags.
#[derive(Debug, Clone)]
pub struct Steinhardt {
    l: Vec<u32>,
    average: bool,
    wl: bool,
    wl_normalize: bool,
}

impl Steinhardt {
    /// Build a calculator for the listed ℓ values (must be non-empty).
    pub fn new(l: &[u32]) -> Result<Self, ComputeError> {
        if l.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "Steinhardt::l",
                value: "[]".into(),
            });
        }
        Ok(Self {
            l: l.to_vec(),
            average: false,
            wl: false,
            wl_normalize: false,
        })
    }

    /// Enable Lechner-Dellago averaged variant (`q̄_ℓm`).
    pub fn with_average(mut self, on: bool) -> Self {
        self.average = on;
        self
    }

    /// Also compute the third-order invariant `w_ℓ`.
    pub fn with_wl(mut self, on: bool) -> Self {
        self.wl = on;
        self
    }

    /// Normalize `w_ℓ` by `( Σ_m |q_ℓm|² )^{3/2}`.
    pub fn with_wl_normalize(mut self, on: bool) -> Self {
        self.wl_normalize = on;
        self
    }

    pub fn l(&self) -> &[u32] {
        &self.l
    }
}

/// Public helper used by `SolidLiquid` and `ContinuousCoordination`: compute
/// the raw `q_ℓm(i)` table for a single ℓ on a single frame.
///
/// Returns a row-major buffer of length `n_particles · (2ℓ+1)` with element
/// `[i, m+ℓ]` at index `i · (2ℓ+1) + (m + ℓ as i32) as usize`.
pub fn compute_qlm<FA: FrameAccess>(
    frame: &FA,
    nlist: &NeighborList,
    l: u32,
) -> Result<Vec<Complex>, ComputeError> {
    let (xs_p, _, _) = get_positions_ref(frame)?;
    let n = xs_p.slice().len();
    let m_count = (2 * l + 1) as usize;

    let mut qlm = vec![Complex::ZERO; n * m_count];
    let mut neighbor_count = vec![0_u32; n];

    let i_idx = nlist.query_point_indices();
    let j_idx = nlist.point_indices();
    let vectors = nlist.vectors();
    let n_pairs = nlist.n_pairs();

    let parity = if l & 1 == 0 { 1.0_f64 } else { -1.0 };
    let mut ylm_buf = vec![Complex::ZERO; m_count];

    for k in 0..n_pairs {
        let i = i_idx[k] as usize;
        let j = j_idx[k] as usize;
        let dx = vectors[[k, 0]];
        let dy = vectors[[k, 1]];
        let dz = vectors[[k, 2]];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r == 0.0 {
            continue;
        }
        let theta = (dz / r).clamp(-1.0, 1.0).acos();
        let phi = dy.atan2(dx);

        ylm_all(l, theta, phi, &mut ylm_buf);
        for m in 0..m_count {
            qlm[i * m_count + m] += ylm_buf[m];
            // Y_ℓm(-r̂) = (-1)^ℓ Y_ℓm(r̂): for the j-side, the bond vector
            // is r_i − r_j = −(r_j − r_i), so we accumulate the parity-flipped
            // term.
            qlm[j * m_count + m] += ylm_buf[m].scale(parity);
        }
        neighbor_count[i] += 1;
        neighbor_count[j] += 1;
    }

    // Normalise by neighbor count (skip isolated particles).
    for i in 0..n {
        let nc = neighbor_count[i];
        if nc == 0 {
            continue;
        }
        let inv = 1.0 / nc as F;
        for m in 0..m_count {
            qlm[i * m_count + m] = qlm[i * m_count + m].scale(inv);
        }
    }

    Ok(qlm)
}

/// Apply the Lechner-Dellago "near-shell" average over self + neighbors.
/// In place: `q̄_ℓm(i) = (q_ℓm(i) + Σ_{j ∈ neigh(i)} q_ℓm(j)) / (N_i + 1)`.
fn average_qlm(qlm: &[Complex], nlist: &NeighborList, n: usize, m_count: usize) -> Vec<Complex> {
    let mut acc = qlm.to_vec();
    let mut count = vec![1_u32; n]; // include self

    let i_idx = nlist.query_point_indices();
    let j_idx = nlist.point_indices();
    let n_pairs = nlist.n_pairs();

    for k in 0..n_pairs {
        let i = i_idx[k] as usize;
        let j = j_idx[k] as usize;
        for m in 0..m_count {
            acc[i * m_count + m] += qlm[j * m_count + m];
            acc[j * m_count + m] += qlm[i * m_count + m];
        }
        count[i] += 1;
        count[j] += 1;
    }
    for i in 0..n {
        let inv = 1.0 / count[i] as F;
        for m in 0..m_count {
            acc[i * m_count + m] = acc[i * m_count + m].scale(inv);
        }
    }
    acc
}

fn compute_ql_from_qlm(qlm: &[Complex], l: u32, n: usize) -> Vec<F> {
    let m_count = (2 * l + 1) as usize;
    let pref = FOUR_PI / (2.0 * l as F + 1.0);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc: F = 0.0;
        for m in 0..m_count {
            acc += qlm[i * m_count + m].norm_sqr();
        }
        out.push((pref * acc).sqrt());
    }
    out
}

fn compute_wl_from_qlm(qlm: &[Complex], l: u32, n: usize, normalize: bool) -> Vec<F> {
    let m_count = (2 * l + 1) as usize;
    let l_i32 = l as i32;
    let mut out = Vec::with_capacity(n);

    // Precompute the triples (m1, m2, m3=-m1-m2) along with the 3-j coefficient
    // so the per-particle loop is just complex products + a real coefficient.
    let mut triples: Vec<(usize, usize, usize, F)> = Vec::new();
    for m1 in -l_i32..=l_i32 {
        for m2 in -l_i32..=l_i32 {
            let m3 = -m1 - m2;
            if m3.abs() > l_i32 {
                continue;
            }
            let w = wigner_3j(l, l, l, m1, m2, m3);
            if w == 0.0 {
                continue;
            }
            triples.push((
                (m1 + l_i32) as usize,
                (m2 + l_i32) as usize,
                (m3 + l_i32) as usize,
                w,
            ));
        }
    }

    for i in 0..n {
        let off = i * m_count;
        let mut wl_re: F = 0.0;
        for &(im1, im2, im3, w) in &triples {
            let prod = qlm[off + im1] * qlm[off + im2] * qlm[off + im3];
            wl_re += w * prod.re;
        }
        if normalize {
            let mut sum_sq: F = 0.0;
            for m in 0..m_count {
                sum_sq += qlm[off + m].norm_sqr();
            }
            let denom = sum_sq.powf(1.5);
            out.push(if denom > 0.0 { wl_re / denom } else { 0.0 });
        } else {
            out.push(wl_re);
        }
    }
    out
}

impl Steinhardt {
    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &NeighborList,
    ) -> Result<SteinhardtResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let n = xs_p.slice().len();

        let mut qlm_per_l: Vec<Vec<Complex>> = Vec::with_capacity(self.l.len());
        let mut ql_per_l: Vec<Vec<F>> = Vec::with_capacity(self.l.len());
        let mut wl_per_l: Vec<Vec<F>> = Vec::with_capacity(self.l.len());

        for &l in &self.l {
            let m_count = (2 * l + 1) as usize;
            let qlm_raw = compute_qlm(frame, nlist, l)?;
            let qlm_used = if self.average {
                average_qlm(&qlm_raw, nlist, n, m_count)
            } else {
                qlm_raw
            };
            let ql = compute_ql_from_qlm(&qlm_used, l, n);
            if self.wl {
                let wl = compute_wl_from_qlm(&qlm_used, l, n, self.wl_normalize);
                wl_per_l.push(wl);
            }
            qlm_per_l.push(qlm_used);
            ql_per_l.push(ql);
        }

        Ok(SteinhardtResult {
            l: self.l.clone(),
            qlm: qlm_per_l,
            ql: ql_per_l,
            wl: if self.wl { Some(wl_per_l) } else { None },
        })
    }
}

impl Compute for Steinhardt {
    type Args<'a> = &'a Vec<NeighborList>;
    type Output = Vec<SteinhardtResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a Vec<NeighborList>,
    ) -> Result<Vec<SteinhardtResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        match frames.len().cmp(&nlists.len()) {
            Ordering::Equal => {}
            _ => {
                return Err(ComputeError::DimensionMismatch {
                    expected: frames.len(),
                    got: nlists.len(),
                    what: "neighbor-list count",
                });
            }
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .zip(nlists.par_iter())
                .map(|(frame, nl)| self.one_frame(*frame, nl))
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for (frame, nl) in frames.iter().zip(nlists.iter()) {
            out.push(self.one_frame(*frame, nl)?);
        }
        Ok(out)
    }
}

/// Per-frame Steinhardt result for one or more ℓ values.
///
/// Each `Vec` is parallel to `l`: `qlm[k]` has shape `(N, 2·l[k]+1)`,
/// `ql[k]` has shape `(N,)`.
#[derive(Debug, Clone, Default)]
pub struct SteinhardtResult {
    /// ℓ values, in the order requested.
    pub l: Vec<u32>,
    /// Per-ℓ `q_ℓm` table, flattened in row-major `[particle, m+ℓ]` order
    /// (length `N · (2ℓ+1)`).
    pub qlm: Vec<Vec<Complex>>,
    /// Per-ℓ scalar `q_ℓ` per particle (length `N`).
    pub ql: Vec<Vec<F>>,
    /// Per-ℓ `w_ℓ` per particle, present only if [`Steinhardt::with_wl`] was set.
    pub wl: Option<Vec<Vec<F>>>,
}

impl ComputeResult for SteinhardtResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::test_support::nlist_from_frame;
    use molrs::Frame;
    use molrs::spatial::simbox::SimBox;
    use molrs::store::block::Block;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], pbc).unwrap());
        frame
    }

    fn build_nlist(frame: &Frame, cutoff: F) -> NeighborList {
        nlist_from_frame(frame, cutoff)
    }

    // -- 1) Trivial single-pair --------------------------------------------------

    #[test]
    fn ql_isolated_particle_is_zero() {
        // Single particle: no neighbors → q_ℓ = 0
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.0);
        let s = Steinhardt::new(&[4, 6]).unwrap();
        let r = s.compute(&[&frame], &vec![nl]).unwrap();
        assert_eq!(r[0].ql[0][0], 0.0);
        assert_eq!(r[0].ql[1][0], 0.0);
    }

    // -- 2) Identical environments must give identical q_ℓ ----------------------
    //
    // For an FCC lattice's 12-coordinate environment, every particle has the same
    // q_ℓ values. We don't construct a full lattice here; instead, two particles
    // with identical *bond* distributions (a regular octahedron of neighbors
    // around each) suffice.

    /// Build an octahedron-coordinated central particle: 6 neighbors at
    /// distance 1 along ±x, ±y, ±z.
    fn octahedron(box_len: F) -> Frame {
        let c = box_len * 0.5;
        let positions = [
            [c, c, c],
            [c + 1.0, c, c],
            [c - 1.0, c, c],
            [c, c + 1.0, c],
            [c, c - 1.0, c],
            [c, c, c + 1.0],
            [c, c, c - 1.0],
        ];
        frame_with(&positions, box_len, [false; 3])
    }

    #[test]
    fn q6_octahedral_environment_finite_and_invariant_to_rotation() {
        // Central particle has 6 neighbors. Compute q_6 at center. Then
        // rotate the same octahedron by π/4 around z and confirm q_6 unchanged.
        let frame = octahedron(20.0);
        let nl = build_nlist(&frame, 1.2);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &s.compute(&[&frame], &vec![nl]).unwrap()[0];
        let q6_center = r.ql[0][0];
        assert!(
            q6_center > 0.0,
            "q_6(center) should be > 0; got {q6_center}"
        );

        // Rotated octahedron: same positions but apply yaw φ=π/4 around z.
        let theta = std::f64::consts::FRAC_PI_4;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let mut positions = vec![[10.0_f64, 10.0, 10.0]];
        for &(dx, dy, dz) in &[
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ] {
            positions.push([
                10.0 + cos_t * dx - sin_t * dy,
                10.0 + sin_t * dx + cos_t * dy,
                10.0 + dz,
            ]);
        }
        let frame2 = frame_with(&positions, 20.0, [false; 3]);
        let nl2 = build_nlist(&frame2, 1.2);
        let r2 = &s.compute(&[&frame2], &vec![nl2]).unwrap()[0];
        let q6_rotated = r2.ql[0][0];
        assert!(
            (q6_rotated - q6_center).abs() < 1e-10,
            "q_6 should be rotation-invariant; got {q6_center} vs {q6_rotated}"
        );
    }

    // -- 3) Multiple ℓ ---------------------------------------------------------

    #[test]
    fn multiple_l_values_independent() {
        let frame = octahedron(20.0);
        let nl = build_nlist(&frame, 1.2);
        let s_solo = Steinhardt::new(&[6]).unwrap();
        let s_pair = Steinhardt::new(&[4, 6]).unwrap();

        let r_solo = &s_solo.compute(&[&frame], &vec![nl.clone()]).unwrap()[0];
        let r_pair = &s_pair.compute(&[&frame], &vec![nl]).unwrap()[0];

        assert!((r_solo.ql[0][0] - r_pair.ql[1][0]).abs() < 1e-12);
        // q_4 and q_6 in general differ for an octahedron.
        assert!((r_pair.ql[0][0] - r_pair.ql[1][0]).abs() > 1e-3);
    }

    // -- 4) w_ℓ third-order invariant ------------------------------------------

    #[test]
    fn wl_present_when_requested() {
        let frame = octahedron(20.0);
        let nl = build_nlist(&frame, 1.2);
        let s = Steinhardt::new(&[6]).unwrap().with_wl(true);
        let r = &s.compute(&[&frame], &vec![nl]).unwrap()[0];
        assert!(r.wl.is_some());
        let wl = r.wl.as_ref().unwrap();
        assert_eq!(wl.len(), 1);
        assert_eq!(wl[0].len(), 7);
        // For a perfect octahedral environment w_6 is a definite (nonzero) number.
        assert!(
            wl[0][0].abs() > 1e-6,
            "w_6(octahedron) should be nonzero; got {}",
            wl[0][0]
        );
    }

    #[test]
    fn wl_normalize_scales_into_unit_range() {
        let frame = octahedron(20.0);
        let nl = build_nlist(&frame, 1.2);
        let s = Steinhardt::new(&[6])
            .unwrap()
            .with_wl(true)
            .with_wl_normalize(true);
        let r = &s.compute(&[&frame], &vec![nl]).unwrap()[0];
        let wl = r.wl.as_ref().unwrap();
        // Normalised ŵ_ℓ ∈ [-1, 1] for any ℓ ≥ 1 (Cauchy-Schwarz on triple sum).
        assert!(
            wl[0][0].abs() <= 1.0 + 1e-9,
            "|ŵ_6| should be ≤ 1; got {}",
            wl[0][0]
        );
    }

    #[test]
    fn wl_absent_by_default() {
        let frame = octahedron(20.0);
        let nl = build_nlist(&frame, 1.2);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &s.compute(&[&frame], &vec![nl]).unwrap()[0];
        assert!(r.wl.is_none());
    }

    // -- 5) Average variant (Lechner-Dellago) ----------------------------------

    #[test]
    fn average_variant_changes_ql() {
        let frame = octahedron(20.0);
        let nl = build_nlist(&frame, 1.2);
        let s_plain = Steinhardt::new(&[6]).unwrap();
        let s_avg = Steinhardt::new(&[6]).unwrap().with_average(true);
        let r_plain = &s_plain.compute(&[&frame], &vec![nl.clone()]).unwrap()[0];
        let r_avg = &s_avg.compute(&[&frame], &vec![nl]).unwrap()[0];
        // Outer-shell particles see different neighborhoods in averaged mode.
        let neighbor_idx = 1;
        assert!(
            (r_plain.ql[0][neighbor_idx] - r_avg.ql[0][neighbor_idx]).abs() > 1e-9
                || r_plain.ql[0][neighbor_idx] == 0.0
        );
    }

    // -- 6) Deterministic --------------------------------------------------------

    #[test]
    fn deterministic_across_calls() {
        let frame = octahedron(20.0);
        let nl = build_nlist(&frame, 1.2);
        let s = Steinhardt::new(&[4, 6]).unwrap().with_wl(true);
        let r1 = s.compute(&[&frame], &vec![nl.clone()]).unwrap();
        let r2 = s.compute(&[&frame], &vec![nl]).unwrap();
        for (a, b) in r1[0].ql.iter().zip(r2[0].ql.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-15);
            }
        }
    }

    // -- 7) Public compute_qlm helper -----------------------------------------

    #[test]
    fn compute_qlm_normalization_matches_internal() {
        let frame = octahedron(20.0);
        let nl = build_nlist(&frame, 1.2);
        let qlm_raw = compute_qlm(&frame, &nl, 6).unwrap();

        // Plain (non-averaged) Steinhardt should yield the same qlm.
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &s.compute(&[&frame], &vec![nl]).unwrap()[0];
        for (a, b) in qlm_raw.iter().zip(r.qlm[0].iter()) {
            assert!((a.re - b.re).abs() < 1e-14 && (a.im - b.im).abs() < 1e-14);
        }
    }

    // -- 8) Empty / error paths ------------------------------------------------

    #[test]
    fn empty_l_is_error() {
        assert!(Steinhardt::new(&[]).is_err());
    }

    #[test]
    fn empty_frames_is_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = Steinhardt::new(&[6])
            .unwrap()
            .compute(&frames, &Vec::<NeighborList>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn mismatched_nlist_count_is_error() {
        let frame = octahedron(20.0);
        let err = Steinhardt::new(&[6])
            .unwrap()
            .compute(&[&frame], &Vec::<NeighborList>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    // -- 9) Multi-frame --------------------------------------------------------

    #[test]
    fn multi_frame_returns_one_result_per_frame() {
        let frame1 = octahedron(20.0);
        let frame2 = octahedron(20.0);
        let nl1 = build_nlist(&frame1, 1.2);
        let nl2 = build_nlist(&frame2, 1.2);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = s.compute(&[&frame1, &frame2], &vec![nl1, nl2]).unwrap();
        assert_eq!(r.len(), 2);
        // Same frame, same nlist → same q_6 at center
        assert!((r[0].ql[0][0] - r[1].ql[0][0]).abs() < 1e-12);
    }

    // -- 10) q_ℓm shape sanity --------------------------------------------------

    #[test]
    fn qlm_shape_is_n_times_2lp1() {
        let frame = octahedron(20.0);
        let nl = build_nlist(&frame, 1.2);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &s.compute(&[&frame], &vec![nl]).unwrap()[0];
        assert_eq!(r.qlm[0].len(), 7 * (2 * 6 + 1));
    }

    // -- 11) Antiparallel pair: parity check -----------------------------------
    //
    // For two particles at ±x_hat (just one pair, antiparallel bonds), the
    // contributions on each particle differ only by (-1)^ℓ. For ℓ=6 (even),
    // both particles must end up with identical q_ℓm — and therefore identical q_6.

    #[test]
    fn parity_two_particle_pair() {
        let frame = frame_with(&[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &s.compute(&[&frame], &vec![nl]).unwrap()[0];
        assert!(
            (r.ql[0][0] - r.ql[0][1]).abs() < 1e-12,
            "even-ℓ q_ℓ must be parity-symmetric across an antiparallel pair"
        );
    }

    // -- 12) Antiparallel pair, odd ℓ: parity flip -----------------------------

    #[test]
    fn parity_odd_l_two_particle_pair() {
        // For ℓ=3 (odd), q_ℓm at particle 1 = -q_ℓm at particle 0 → same magnitude.
        let frame = frame_with(&[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let s = Steinhardt::new(&[3]).unwrap();
        let r = &s.compute(&[&frame], &vec![nl]).unwrap()[0];
        assert!((r.ql[0][0] - r.ql[0][1]).abs() < 1e-12);
    }

    // -- 13) Multi-l result struct ordering -----------------------------------

    #[test]
    fn result_l_field_preserves_input_order() {
        let frame = octahedron(20.0);
        let nl = build_nlist(&frame, 1.2);
        let s = Steinhardt::new(&[6, 4, 8]).unwrap();
        let r = &s.compute(&[&frame], &vec![nl]).unwrap()[0];
        assert_eq!(r.l, vec![6, 4, 8]);
    }

    // -- 14) PBC: same lattice in wrapped vs unwrapped boxes gives same q_ℓ ---

    #[test]
    fn pbc_consistent_with_open_box() {
        // Six-coordinate environment centred on the *origin* of a periodic box,
        // so neighbors at (±1, 0, 0) etc. straddle the boundary.
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [9.0, 0.0, 0.0], // wrapped equivalent of (-1, 0, 0)
            [0.0, 1.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 9.0],
        ];
        let frame = frame_with(&positions, 10.0, [true, true, true]);
        let nl = build_nlist(&frame, 1.5);

        let frame_open = octahedron(20.0);
        let nl_open = build_nlist(&frame_open, 1.2);

        let s = Steinhardt::new(&[6]).unwrap();
        let r_pbc = &s.compute(&[&frame], &vec![nl]).unwrap()[0];
        let r_open = &s.compute(&[&frame_open], &vec![nl_open]).unwrap()[0];
        assert!(
            (r_pbc.ql[0][0] - r_open.ql[0][0]).abs() < 1e-10,
            "q_6 should match between PBC-wrapped and open-box octahedra: got {} vs {}",
            r_pbc.ql[0][0],
            r_open.ql[0][0]
        );
    }
}
