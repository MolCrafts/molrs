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

/// Mean squared displacement relative to a stored reference configuration.
///
/// Computes |r(t) - r_ref|² for each particle and the system average.
/// Reference stored as SoA (three separate Vec) for cache-friendly access.
#[derive(Debug, Clone)]
pub struct MSD {
    ref_x: Vec<F>,
    ref_y: Vec<F>,
    ref_z: Vec<F>,
    n_particles: usize,
}

impl MSD {
    /// Create from a reference frame (reads x, y, z from "atoms" block).
    pub fn from_reference(ref_frame: &Frame) -> Result<Self, ComputeError> {
        let atoms = ref_frame
            .get("atoms")
            .ok_or(ComputeError::MissingBlock { name: "atoms" })?;
        let x = get_f_slice(atoms, "atoms", "x")?;
        let y = get_f_slice(atoms, "atoms", "y")?;
        let z = get_f_slice(atoms, "atoms", "z")?;

        Ok(Self {
            n_particles: x.len(),
            ref_x: x.to_vec(),
            ref_y: y.to_vec(),
            ref_z: z.to_vec(),
        })
    }

    /// Convenience: wrap in `Accumulator<Self, ConcatReducer<MSDResult>>`.
    pub fn accumulate_concat(self) -> Accumulator<Self, ConcatReducer<MSDResult>> {
        Accumulator::new(self, ConcatReducer::new())
    }
}

impl Compute for MSD {
    type Output = MSDResult;

    fn compute(&self, frame: &Frame) -> Result<MSDResult, ComputeError> {
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
    fn uniform_displacement() {
        let ref_frame = make_frame(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let displaced = make_frame(&[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);

        let msd = MSD::from_reference(&ref_frame).unwrap();
        let result = msd.compute(&displaced).unwrap();

        assert!((result.mean - 1.0).abs() < 1e-6);
        for &d2 in result.per_particle.iter() {
            assert!((d2 - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_displacement() {
        let frame = make_frame(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]);
        let msd = MSD::from_reference(&frame).unwrap();
        let result = msd.compute(&frame).unwrap();

        assert!(result.mean.abs() < 1e-6);
    }

    #[test]
    fn dimension_mismatch_error() {
        let ref_frame = make_frame(&[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]);
        let bad_frame = make_frame(&[1.0], &[1.0], &[1.0]);

        let msd = MSD::from_reference(&ref_frame).unwrap();
        let err = msd.compute(&bad_frame).unwrap_err();
        assert!(matches!(
            err,
            ComputeError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn concat_accumulator() {
        let ref_frame = make_frame(&[0.0], &[0.0], &[0.0]);
        let msd = MSD::from_reference(&ref_frame).unwrap();
        let mut acc = msd.accumulate_concat();

        let f1 = make_frame(&[1.0], &[0.0], &[0.0]);
        let f2 = make_frame(&[2.0], &[0.0], &[0.0]);
        let f3 = make_frame(&[3.0], &[0.0], &[0.0]);

        acc.feed(&f1).unwrap();
        acc.feed(&f2).unwrap();
        acc.feed(&f3).unwrap();

        let history = acc.result();
        assert_eq!(history.len(), 3);
        assert!((history[0].mean - 1.0).abs() < 1e-6);
        assert!((history[1].mean - 4.0).abs() < 1e-6);
        assert!((history[2].mean - 9.0).abs() < 1e-6);
    }
}
