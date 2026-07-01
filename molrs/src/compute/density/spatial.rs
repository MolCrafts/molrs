//! Spatial Distribution Function (SDF): 3-D number density of a target species
//! in the body-fixed frame of a reference molecule, accumulated over a
//! trajectory.
//!
//! Unlike [`GaussianDensity`](super::gaussian_density::GaussianDensity), which
//! is lab-frame, the SDF first superimposes each frame's reference atoms onto a
//! canonical template (native Kabsch, [`super::kabsch`]), applies the same
//! rigid rotation to the surrounding target atoms — after minimum-image
//! unwrapping relative to the reference centre of mass — and only then bins
//! them into a grid centred on that COM. The result is the familiar "density
//! cloud" of *where, relative to this molecule*, a second species sits.
//!
//! # TRAVIS provenance
//!
//! - The reference-frame fix on a 3-atom reference set mirrors TRAVIS's global
//!   `g_iFixMol` / `g_iFixAtom[0..2]` alignment (the SDF "fix" in `engine.cpp`),
//!   here realized as a least-squares quaternion superposition so the whole
//!   reference set — not just three atoms — is used and no BLAS is needed.
//! - 3-D voxel accumulation follows TRAVIS's `C3DF::AddToBin` (`src/3df.cpp`):
//!   nearest-voxel deposition, out-of-grid samples skipped.
//! - The per-voxel mean-orientation field is `Σ value / count` exactly as
//!   `CSDFMap::Finish` averages its value bins (`src/sdfmap.cpp:418-424`).
//!
//! Density normalization: `ρ(voxel) = counts / (n_frames · ΔV)` in Å⁻³; the
//! optional bulk-normalized `g_SDF = ρ / ρ_bulk` is the SDF analogue of RDF's
//! `g(r)` and tends to 1 far from the reference for an unstructured target.

use molrs::store::frame_access::FrameAccess;
use molrs::types::F;
use ndarray::{Array2, Array3, Array4};

use super::kabsch::{kabsch, rotate};
use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use crate::compute::util::{get_positions_ref, mic_disp};

/// A regular axis-aligned voxel grid centred on the reference COM.
///
/// `extent[d]` is the **full** side length (Å) along axis `d`; the grid spans
/// `[−extent/2, +extent/2]` about the COM with `n[d]` voxels per axis.
#[derive(Debug, Clone, Copy)]
pub struct GridSpec {
    /// Voxels per axis.
    pub n: [usize; 3],
    /// Full side length per axis, Å.
    pub extent: [F; 3],
}

impl GridSpec {
    /// Voxel side lengths (Å) per axis.
    fn voxel_size(&self) -> [F; 3] {
        [
            self.extent[0] / self.n[0] as F,
            self.extent[1] / self.n[1] as F,
            self.extent[2] / self.n[2] as F,
        ]
    }

    /// Single voxel volume, Å³.
    fn voxel_volume(&self) -> F {
        let v = self.voxel_size();
        v[0] * v[1] * v[2]
    }

    /// Map a body-frame coordinate (relative to the COM) to a voxel index,
    /// or `None` if it falls outside the grid.
    fn index(&self, r: [F; 3]) -> Option<[usize; 3]> {
        let vs = self.voxel_size();
        let mut idx = [0usize; 3];
        for d in 0..3 {
            let shifted = r[d] + 0.5 * self.extent[d];
            if shifted < 0.0 {
                return None;
            }
            let i = (shifted / vs[d]).floor() as isize;
            if i < 0 || i >= self.n[d] as isize {
                return None;
            }
            idx[d] = i as usize;
        }
        Some(idx)
    }
}

/// Spatial Distribution Function calculator.
///
/// All inputs live on `&self` (the [`Compute::Args`] are `()`): a rigid
/// reference atom selection and its canonical `template`, a target selection,
/// the grid, and optional bulk density / orientation vectors.
#[derive(Debug, Clone)]
pub struct SpatialDistribution {
    reference: Vec<usize>,
    template: Array2<F>,
    target: Vec<usize>,
    grid: GridSpec,
    /// Optional `(tail, head)` atom-index pairs (parallel to `target`) defining
    /// a per-target vector whose body-frame mean is mapped per voxel.
    orientation: Option<Vec<(usize, usize)>>,
    /// Optional bulk number density (Å⁻³) for the `g_SDF` normalization.
    bulk_density: Option<F>,
}

impl SpatialDistribution {
    /// New SDF over a `reference` selection (with canonical `template`,
    /// `reference.len() × 3`) and a `target` selection, binned into `grid`.
    ///
    /// # Errors
    ///
    /// [`ComputeError::DimensionMismatch`] if `template` rows ≠ `reference`
    /// length; [`ComputeError::OutOfRange`] for a degenerate grid or fewer than
    /// 3 reference atoms. (Collinearity of the template is caught per-frame by
    /// [`kabsch`].)
    pub fn new(
        reference: Vec<usize>,
        template: Array2<F>,
        target: Vec<usize>,
        grid: GridSpec,
    ) -> Result<Self, ComputeError> {
        if template.nrows() != reference.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: reference.len(),
                got: template.nrows(),
                what: "SDF template rows",
            });
        }
        if reference.len() < 3 {
            return Err(ComputeError::OutOfRange {
                field: "SpatialDistribution::reference",
                value: reference.len().to_string(),
            });
        }
        if grid.n.contains(&0) || grid.extent.iter().any(|&e| e <= 0.0 || e.is_nan()) {
            return Err(ComputeError::OutOfRange {
                field: "SpatialDistribution::grid",
                value: format!("n={:?}, extent={:?}", grid.n, grid.extent),
            });
        }
        Ok(Self {
            reference,
            template,
            target,
            grid,
            orientation: None,
            bulk_density: None,
        })
    }

    /// Attach `(tail, head)` index pairs (one per target atom, same order) so
    /// the result carries a per-voxel mean body-frame orientation of the unit
    /// `head − tail` vector.
    pub fn with_orientation(mut self, pairs: Vec<(usize, usize)>) -> Self {
        self.orientation = Some(pairs);
        self
    }

    /// Set the bulk number density (Å⁻³) used to form `g_SDF = ρ / ρ_bulk`.
    pub fn with_bulk_density(mut self, rho: F) -> Self {
        self.bulk_density = Some(rho);
        self
    }

    fn reference_coords(&self, xs: &[F], ys: &[F], zs: &[F]) -> Array2<F> {
        let mut a = Array2::<F>::zeros((self.reference.len(), 3));
        for (row, &i) in self.reference.iter().enumerate() {
            a[[row, 0]] = xs[i];
            a[[row, 1]] = ys[i];
            a[[row, 2]] = zs[i];
        }
        a
    }
}

impl Compute for SpatialDistribution {
    type Args<'a> = ();
    type Output = SpatialDistributionResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _args: (),
    ) -> Result<Self::Output, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if let Some(pairs) = &self.orientation
            && pairs.len() != self.target.len()
        {
            return Err(ComputeError::DimensionMismatch {
                expected: self.target.len(),
                got: pairs.len(),
                what: "SDF orientation pairs",
            });
        }

        let [nx, ny, nz] = self.grid.n;
        let mut counts = Array3::<F>::zeros((nx, ny, nz));
        let mut orient_sum: Option<Array4<F>> = self
            .orientation
            .as_ref()
            .map(|_| Array4::zeros((nx, ny, nz, 3)));
        let mut orient_count: Option<Array3<F>> = self
            .orientation
            .as_ref()
            .map(|_| Array3::zeros((nx, ny, nz)));

        for frame in frames {
            let simbox = frame.simbox_ref();
            let (xs_p, ys_p, zs_p) = get_positions_ref(*frame)?;
            let xs = xs_p.slice();
            let ys = ys_p.slice();
            let zs = zs_p.slice();

            // Align this frame's reference set onto the template.
            let ref_coords = self.reference_coords(xs, ys, zs);
            let (r, _rmsd) = kabsch(self.template.view(), ref_coords.view())?;

            // Reference COM (lab frame) = centroid of the reference atoms.
            let com = centroid(&ref_coords);

            for (t, &ai) in self.target.iter().enumerate() {
                // Minimum-image vector COM → target, then rotate into body frame.
                let disp = mic_disp(simbox, com, [xs[ai], ys[ai], zs[ai]]);
                let body = rotate(&r, disp);
                let Some([ix, iy, iz]) = self.grid.index(body) else {
                    continue;
                };
                counts[[ix, iy, iz]] += 1.0;

                if let (Some(pairs), Some(osum), Some(ocount)) = (
                    &self.orientation,
                    orient_sum.as_mut(),
                    orient_count.as_mut(),
                ) {
                    let (tail, head) = pairs[t];
                    let v = mic_disp(
                        simbox,
                        [xs[tail], ys[tail], zs[tail]],
                        [xs[head], ys[head], zs[head]],
                    );
                    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    if n > 0.0 {
                        let u = [v[0] / n, v[1] / n, v[2] / n];
                        let bu = rotate(&r, u);
                        for d in 0..3 {
                            osum[[ix, iy, iz, d]] += bu[d];
                        }
                        ocount[[ix, iy, iz]] += 1.0;
                    }
                }
            }
        }

        let mut result = SpatialDistributionResult {
            counts,
            density: Array3::zeros((nx, ny, nz)),
            g_sdf: None,
            orientation: None,
            orient_sum,
            orient_count,
            n: self.grid.n,
            extent: self.grid.extent,
            voxel_volume: self.grid.voxel_volume(),
            n_frames: frames.len(),
            bulk_density: self.bulk_density,
            finalized: false,
        };
        result.finalize();
        Ok(result)
    }
}

/// Centroid of an `M × 3` coordinate array.
fn centroid(a: &Array2<F>) -> [F; 3] {
    let m = a.nrows().max(1) as F;
    let mut c = [0.0_f64; 3];
    for row in a.rows() {
        for d in 0..3 {
            c[d] += row[d];
        }
    }
    for cd in &mut c {
        *cd /= m;
    }
    c
}

/// Accumulated SDF grid plus optional bulk-normalized density and orientation.
#[derive(Debug, Clone)]
pub struct SpatialDistributionResult {
    /// Raw voxel counts, summed across frames.
    pub counts: Array3<F>,
    /// Number density per voxel (Å⁻³): `counts / (n_frames · ΔV)`.
    pub density: Array3<F>,
    /// Bulk-normalized `g_SDF = density / ρ_bulk`, if a bulk density was set.
    pub g_sdf: Option<Array3<F>>,
    /// Per-voxel mean body-frame orientation vector `(nx, ny, nz, 3)`, if an
    /// orientation selection was supplied. Zero on empty voxels.
    pub orientation: Option<Array4<F>>,
    orient_sum: Option<Array4<F>>,
    orient_count: Option<Array3<F>>,
    /// Voxels per axis.
    pub n: [usize; 3],
    /// Full grid side length per axis, Å.
    pub extent: [F; 3],
    /// Single-voxel volume, Å³.
    pub voxel_volume: F,
    /// Number of frames accumulated.
    pub n_frames: usize,
    bulk_density: Option<F>,
    finalized: bool,
}

impl ComputeResult for SpatialDistributionResult {
    fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        let denom = self.n_frames.max(1) as F * self.voxel_volume;
        self.density = self.counts.mapv(|c| c / denom);
        if let Some(rho) = self.bulk_density
            && rho > 0.0
        {
            self.g_sdf = Some(self.density.mapv(|d| d / rho));
        }
        if let (Some(sum), Some(count)) = (&self.orient_sum, &self.orient_count) {
            let [nx, ny, nz] = self.n;
            let mut mean = Array4::<F>::zeros((nx, ny, nz, 3));
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let c = count[[ix, iy, iz]];
                        if c > 0.0 {
                            for d in 0..3 {
                                mean[[ix, iy, iz, d]] = sum[[ix, iy, iz, d]] / c;
                            }
                        }
                    }
                }
            }
            self.orientation = Some(mean);
        }
        self.finalized = true;
    }
}
