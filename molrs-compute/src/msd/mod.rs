//! Mean squared displacement analysis (stateless).
//!
//! Given a slice of frames, treats `frames[0]` as the reference and produces
//! one [`MSDResult`] per frame. The struct is unit-sized — no hidden state,
//! same input → same output every time.

mod result;

pub use result::{MSDResult, MSDTimeSeries};

use molrs::frame_access::FrameAccess;
use molrs::types::F;
use ndarray::Array1;

use crate::error::ComputeError;
use crate::traits::Compute;
use crate::util::get_positions_ref;

/// Mean squared displacement analysis.
///
/// `MSD(t) = <|r(t) - r(0)|^2>` with the first frame as reference.
#[derive(Debug, Clone, Copy, Default)]
pub struct MSD;

impl MSD {
    pub fn new() -> Self {
        Self
    }
}

fn msd_vs_reference<FA: FrameAccess>(
    frame: &FA,
    ref_x: &[F],
    ref_y: &[F],
    ref_z: &[F],
) -> Result<MSDResult, ComputeError> {
    let (xs_p, ys_p, zs_p) = get_positions_ref(frame)?;
    let xs = xs_p.slice();
    let ys = ys_p.slice();
    let zs = zs_p.slice();
    let n = xs.len();
    if n != ref_x.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: ref_x.len(),
            got: n,
            what: "MSD particle count",
        });
    }
    let mut per_particle = Array1::<F>::zeros(n);
    let mut total: F = 0.0;
    let pp = per_particle.as_slice_mut().expect("zeros is contiguous");
    // Tight scalar loop — autovectorizes on AArch64/NEON and x86-64/AVX.
    for i in 0..n {
        let dx = xs[i] - ref_x[i];
        let dy = ys[i] - ref_y[i];
        let dz = zs[i] - ref_z[i];
        let d2 = dx * dx + dy * dy + dz * dz;
        pp[i] = d2;
        total += d2;
    }
    let mean = if n > 0 { total / n as F } else { 0.0 };
    Ok(MSDResult { per_particle, mean })
}

impl Compute for MSD {
    type Args<'a> = ();
    type Output = MSDTimeSeries;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _args: (),
    ) -> Result<MSDTimeSeries, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        // Own the reference positions so downstream frames can be processed
        // in parallel. Contiguous views on the first frame become slice
        // copies into owned Vecs once.
        let (rx_p, ry_p, rz_p) = get_positions_ref(frames[0])?;
        let ref_x: Vec<F> = rx_p.slice().to_vec();
        let ref_y: Vec<F> = ry_p.slice().to_vec();
        let ref_z: Vec<F> = rz_p.slice().to_vec();

        // Per-frame MSD is a single pass through N atoms with trivial
        // arithmetic, so rayon's task-submission overhead (~5-10 µs)
        // dominates until we have ~8+ frames. Fall back to serial below
        // that threshold.
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 8;

        #[cfg(feature = "rayon")]
        let results: Vec<MSDResult> = if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            frames
                .par_iter()
                .map(|frame| msd_vs_reference(*frame, &ref_x, &ref_y, &ref_z))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            frames
                .iter()
                .map(|frame| msd_vs_reference(*frame, &ref_x, &ref_y, &ref_z))
                .collect::<Result<Vec<_>, _>>()?
        };
        #[cfg(not(feature = "rayon"))]
        let results: Vec<MSDResult> = frames
            .iter()
            .map(|frame| msd_vs_reference(*frame, &ref_x, &ref_y, &ref_z))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(MSDTimeSeries::new(results))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::block::Block;
    use ndarray::Array1 as A1;

    fn make_frame(x: &[F], y: &[F], z: &[F]) -> Frame {
        let mut block = Block::new();
        block
            .insert("x", A1::from_vec(x.to_vec()).into_dyn())
            .unwrap();
        block
            .insert("y", A1::from_vec(y.to_vec()).into_dyn())
            .unwrap();
        block
            .insert("z", A1::from_vec(z.to_vec()).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame
    }

    #[test]
    fn reference_frame_msd_is_zero() {
        let f0 = make_frame(&[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]);
        let f1 = make_frame(&[1.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
        let series = MSD::new().compute(&[&f0, &f1], ()).unwrap();
        assert_eq!(series.len(), 2);
        assert!(series.data[0].mean.abs() < 1e-12);
        assert!((series.data[1].mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn deterministic_across_calls() {
        let f0 = make_frame(&[0.0, 0.0, 0.0], &[0.0; 3], &[0.0; 3]);
        let f1 = make_frame(&[1.0, 1.0, 1.0], &[0.0; 3], &[0.0; 3]);
        let f2 = make_frame(&[2.0, 2.0, 2.0], &[0.0; 3], &[0.0; 3]);
        let msd = MSD::new();
        let a = msd.compute(&[&f0, &f1, &f2], ()).unwrap();
        let b = msd.compute(&[&f0, &f1, &f2], ()).unwrap();
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!((a.data[i].mean - b.data[i].mean).abs() < 1e-12);
        }
    }

    #[test]
    fn linear_progression() {
        // Frame i: each particle at (i, 0, 0). MSD(i) = i² per particle.
        let frames_owned: Vec<Frame> = (0..4)
            .map(|i| make_frame(&[i as F; 3], &[0.0; 3], &[0.0; 3]))
            .collect();
        let frames: Vec<&Frame> = frames_owned.iter().collect();
        let series = MSD::new().compute(&frames, ()).unwrap();
        for i in 0..4 {
            let expected = (i as F) * (i as F);
            assert!(
                (series.data[i].mean - expected).abs() < 1e-12,
                "MSD[{i}] = {}, expected {expected}",
                series.data[i].mean
            );
        }
    }

    #[test]
    fn empty_input_errors() {
        let frames: Vec<&Frame> = Vec::new();
        let err = MSD::new().compute(&frames, ()).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn mismatched_particle_count_errors() {
        let f0 = make_frame(&[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]);
        let f1 = make_frame(&[1.0], &[0.0], &[0.0]); // one particle only
        let err = MSD::new().compute(&[&f0, &f1], ()).unwrap_err();
        assert!(matches!(
            err,
            ComputeError::DimensionMismatch {
                expected: 2,
                got: 1,
                ..
            }
        ));
    }
}
