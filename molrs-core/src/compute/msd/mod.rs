mod result;

pub use result::MSDResult;

use crate::Frame;
use crate::types::F;
use ndarray::Array1;

use super::accumulator::Accumulator;
use super::error::ComputeError;
use super::reducer::ConcatReducer;
use super::traits::Compute;
use super::util::get_f_slice;

/// Mean squared displacement analysis.
///
/// Computes MSD = |r(t) - r(0)|² for each particle and the system average.
/// The first frame fed is automatically used as the reference configuration.
///
/// Reference stored as SoA (three separate Vec) for cache-friendly access.
#[derive(Debug, Clone)]
pub struct MSD {
    ref_x: Vec<F>,
    ref_y: Vec<F>,
    ref_z: Vec<F>,
    n_particles: usize,
    results: Vec<MSDResult>,
}

impl MSD {
    /// Create an empty MSD analysis.
    ///
    /// The first frame passed to [`feed`](Self::feed) becomes the reference.
    pub fn new() -> Self {
        Self {
            ref_x: Vec::new(),
            ref_y: Vec::new(),
            ref_z: Vec::new(),
            n_particles: 0,
            results: Vec::new(),
        }
    }

    /// Create from an explicit reference frame.
    #[deprecated(note = "use MSD::new() + feed(); first frame is auto-reference")]
    pub fn from_reference(ref_frame: &Frame) -> Result<Self, ComputeError> {
        let mut msd = Self::new();
        msd.set_reference(ref_frame)?;
        Ok(msd)
    }

    /// Feed a frame. The first frame sets the reference; subsequent frames
    /// compute MSD relative to that reference.
    pub fn feed(&mut self, frame: &Frame) -> Result<(), ComputeError> {
        if self.n_particles == 0 {
            self.set_reference(frame)?;
            // Reference frame has MSD = 0
            let n = self.n_particles;
            self.results.push(MSDResult {
                per_particle: Array1::<F>::zeros(n),
                mean: 0.0,
            });
            return Ok(());
        }
        let result = self.compute_single(frame)?;
        self.results.push(result);
        Ok(())
    }

    /// Return all accumulated MSD results (one per frame fed).
    pub fn results(&self) -> &[MSDResult] {
        &self.results
    }

    /// Reset the analysis, clearing reference and all results.
    pub fn reset(&mut self) {
        self.ref_x.clear();
        self.ref_y.clear();
        self.ref_z.clear();
        self.n_particles = 0;
        self.results.clear();
    }

    /// Number of frames accumulated.
    pub fn count(&self) -> usize {
        self.results.len()
    }

    fn set_reference(&mut self, frame: &Frame) -> Result<(), ComputeError> {
        let atoms = frame
            .get("atoms")
            .ok_or(ComputeError::MissingBlock { name: "atoms" })?;
        let x = get_f_slice(atoms, "atoms", "x")?;
        let y = get_f_slice(atoms, "atoms", "y")?;
        let z = get_f_slice(atoms, "atoms", "z")?;
        self.n_particles = x.len();
        self.ref_x = x.to_vec();
        self.ref_y = y.to_vec();
        self.ref_z = z.to_vec();
        Ok(())
    }

    fn compute_single(&self, frame: &Frame) -> Result<MSDResult, ComputeError> {
        let atoms = frame
            .get("atoms")
            .ok_or(ComputeError::MissingBlock { name: "atoms" })?;
        let xs = get_f_slice(atoms, "atoms", "x")?;
        let ys = get_f_slice(atoms, "atoms", "y")?;
        let zs = get_f_slice(atoms, "atoms", "z")?;

        let n = xs.len();
        if n != self.n_particles {
            return Err(ComputeError::DimensionMismatch {
                expected: self.n_particles,
                got: n,
            });
        }

        let mut per_particle = Array1::<F>::zeros(n);
        let mut total: F = 0.0;

        for i in 0..n {
            let dx = xs[i] - self.ref_x[i];
            let dy = ys[i] - self.ref_y[i];
            let dz = zs[i] - self.ref_z[i];
            let d2 = dx * dx + dy * dy + dz * dz;
            per_particle[i] = d2;
            total += d2;
        }

        let mean = if n > 0 { total / n as F } else { 0.0 };

        Ok(MSDResult { per_particle, mean })
    }

    /// Convenience: wrap in `Accumulator<Self, ConcatReducer<MSDResult>>`.
    #[deprecated(note = "use MSD::new() + feed() instead")]
    pub fn accumulate_concat(self) -> Accumulator<Self, ConcatReducer<MSDResult>> {
        Accumulator::new(self, ConcatReducer::new())
    }
}

impl Default for MSD {
    fn default() -> Self {
        Self::new()
    }
}

impl Compute for MSD {
    type Output = MSDResult;

    fn compute(&self, frame: &Frame) -> Result<MSDResult, ComputeError> {
        if self.n_particles == 0 {
            return Err(ComputeError::MissingBlock { name: "atoms" });
        }
        self.compute_single(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
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
    fn feed_trajectory() {
        let f0 = make_frame(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let f1 = make_frame(&[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let f2 = make_frame(&[2.0, 2.0, 2.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);

        let mut msd = MSD::new();
        msd.feed(&f0).unwrap(); // auto-reference
        msd.feed(&f1).unwrap();
        msd.feed(&f2).unwrap();

        let results = msd.results();
        assert_eq!(results.len(), 3);
        assert!(results[0].mean.abs() < 1e-6); // frame 0 vs itself
        assert!((results[1].mean - 1.0).abs() < 1e-6); // dx=1 for each particle
        assert!((results[2].mean - 4.0).abs() < 1e-6); // dx=2 for each particle
    }

    #[test]
    fn zero_displacement() {
        let frame = make_frame(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]);
        let mut msd = MSD::new();
        msd.feed(&frame).unwrap();
        msd.feed(&frame).unwrap();

        assert!(msd.results()[1].mean.abs() < 1e-6);
    }

    #[test]
    fn dimension_mismatch_error() {
        let ref_frame = make_frame(&[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]);
        let bad_frame = make_frame(&[1.0], &[1.0], &[1.0]);

        let mut msd = MSD::new();
        msd.feed(&ref_frame).unwrap();
        let err = msd.feed(&bad_frame).unwrap_err();
        assert!(matches!(
            err,
            ComputeError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn reset_clears_state() {
        let f0 = make_frame(&[0.0], &[0.0], &[0.0]);
        let f1 = make_frame(&[1.0], &[0.0], &[0.0]);

        let mut msd = MSD::new();
        msd.feed(&f0).unwrap();
        msd.feed(&f1).unwrap();
        assert_eq!(msd.count(), 2);

        msd.reset();
        assert_eq!(msd.count(), 0);

        // Can re-feed after reset
        msd.feed(&f0).unwrap();
        assert_eq!(msd.count(), 1);
    }
}
