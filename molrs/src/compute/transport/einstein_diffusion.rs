//! Einstein diffusion raw compute — the windowed-MSD route to D.

use molrs::store::frame_access::FrameAccess;
use ndarray::Array1;

use crate::compute::error::ComputeError;
use crate::compute::msd::MSD;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;

/// Raw self-MSD for the Einstein diffusion route.
#[derive(Debug, Clone)]
pub struct EinsteinDiffusionResult {
    /// Lag times τ = i·dt, length `n_frames`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// System-average MSD per lag, identical to
    /// [`MSD::windowed`](crate::compute::MSD)'s per-lag mean. Units: `[length]²`.
    pub msd: Array1<f64>,
}

impl ComputeResult for EinsteinDiffusionResult {}

/// Raw self-MSD compute. Delegates to
/// [`MSD::windowed`](crate::compute::MSD) — MSD math is **not** re-derived here.
/// `D = slope/(2d)` is then a [`LinearFit`](crate::compute::fit::LinearFit) +
/// scale step.
#[derive(Debug, Clone, Copy, Default)]
pub struct EinsteinDiffusion;

/// `EinsteinDiffusion` argument bundle: the frame spacing for the lag axis.
#[derive(Debug, Clone, Copy)]
pub struct EinsteinDiffusionArgs {
    /// Frame spacing for the lag-time axis. Units: time.
    pub dt: f64,
}

impl Compute for EinsteinDiffusion {
    /// Frame spacing only; positions come from `frames`.
    type Args<'a> = EinsteinDiffusionArgs;
    type Output = EinsteinDiffusionResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if args.dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: args.dt.to_string(),
            });
        }
        let series = MSD::windowed().compute(frames, ())?;
        let msd = Array1::from_iter(series.data.iter().map(|r| r.mean));
        let lag_times = Array1::from_iter((0..series.data.len()).map(|i| i as f64 * args.dt));
        Ok(EinsteinDiffusionResult { lag_times, msd })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::store::block::Block;
    use ndarray::Array1 as A1;

    fn make_frame(x: &[f64], y: &[f64], z: &[f64]) -> Frame {
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
    fn einstein_diffusion_delegates_to_msd_windowed() {
        // ac-012: EinsteinDiffusion.msd == MSD::windowed().compute means.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [0.1, -0.2, 0.3, -0.1, 0.05, 0.0];
        let frames_owned: Vec<Frame> = (0..6)
            .map(|t| make_frame(&[xs[t], xs[t] + 0.5], &[ys[t], ys[t] + 0.4], &[0.0, 0.0]))
            .collect();
        let frames: Vec<&Frame> = frames_owned.iter().collect();

        let series = MSD::windowed().compute(&frames, ()).unwrap();
        let raw = EinsteinDiffusion
            .compute(&frames, EinsteinDiffusionArgs { dt: 2.0 })
            .unwrap();
        assert_eq!(raw.msd.len(), series.data.len());
        for i in 0..raw.msd.len() {
            assert!((raw.msd[i] - series.data[i].mean).abs() < 1e-12, "i={i}");
            assert!((raw.lag_times[i] - i as f64 * 2.0).abs() < 1e-12);
        }
    }
}
