//! Green–Kubo diffusion raw compute — the velocity-ACF route to D.

use molrs::store::frame_access::FrameAccess;

use super::vacf::{VacfArgs, VacfResult, velocity_acf};
use crate::compute::error::ComputeError;
use crate::compute::traits::Compute;

/// Raw velocity ACF for the Green–Kubo diffusion route — the same raw curve as
/// [`VACF`](super::VACF), named for the diffusion workflow.
/// `D = (1/d)·∫ VACF dt` is then a
/// [`CumulativeTrapezoid`](crate::compute::fitting::CumulativeTrapezoid) + scale step.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreenKuboDiffusion;

impl Compute for GreenKuboDiffusion {
    type Args<'a> = VacfArgs<'a>;
    type Output = VacfResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (velocities, dt, resolution) = args;
        velocity_acf(velocities, dt, resolution)
    }
}

#[cfg(test)]
mod tests {
    use super::super::vacf::VACF;
    use super::*;
    use molrs::Frame;
    use ndarray::Array2;
    use rand::{RngExt, SeedableRng};

    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    fn rng_series(n: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Array2::zeros((n, cols));
        for t in 0..n {
            for c in 0..cols {
                s[[t, c]] = rng.random_range(-1.0..1.0);
            }
        }
        s
    }

    #[test]
    fn green_kubo_diffusion_equals_vacf() {
        let n = 64;
        let dt = 1.0;
        let v = rng_series(n, 3, 11);
        let a = VACF.compute(&no_frames(), (&v, dt, 20)).unwrap();
        let b = GreenKuboDiffusion
            .compute(&no_frames(), (&v, dt, 20))
            .unwrap();
        assert_eq!(a.acf, b.acf);
        assert_eq!(a.lag_times, b.lag_times);
    }
}
