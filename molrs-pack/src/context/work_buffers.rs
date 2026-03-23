//! Reusable temporary buffers for objective/gradient and movebad paths.

use molrs::types::F;
/// Reusable mutable buffers shared across packing iterations.
pub struct WorkBuffers {
    /// Cartesian gradient accumulator used by objective gradient evaluation.
    pub gxcar: Vec<[F; 3]>,
    /// Temporary radius backup used by movebad/radius scaling paths.
    pub radiuswork: Vec<F>,
    /// Per-molecule score buffer used by flashsort/movebad ranking.
    pub fmol: Vec<F>,
    /// Index permutation buffer reused by flashsort in movebad.
    pub flash_ind: Vec<usize>,
    /// Histogram bucket buffer reused by flashsort.
    pub flash_l: Vec<usize>,
}

impl WorkBuffers {
    pub fn new(ntotat: usize) -> Self {
        Self {
            gxcar: vec![[0.0; 3]; ntotat],
            radiuswork: vec![0.0; ntotat],
            fmol: Vec::new(),
            flash_ind: Vec::new(),
            flash_l: Vec::new(),
        }
    }

    pub fn ensure_atom_capacity(&mut self, ntotat: usize) {
        if self.gxcar.len() != ntotat {
            self.gxcar.resize(ntotat, [0.0; 3]);
        }
        if self.radiuswork.len() != ntotat {
            self.radiuswork.resize(ntotat, 0.0);
        }
    }
}
