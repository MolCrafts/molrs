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
    /// Last x-vector whose expanded Cartesian geometry is still resident in `PackContext`.
    pub cached_x: Vec<F>,
    /// Active-type mask associated with `cached_x`.
    pub cached_comptype: Vec<bool>,
    /// Whether the cached geometry was built in init1 mode.
    pub cached_init1: bool,
    /// Cell grid signature for the cached geometry.
    pub cached_ncells: [usize; 3],
    pub cached_cell_length: [F; 3],
    pub cached_pbc_min: [F; 3],
    pub cached_pbc_length: [F; 3],
    /// Whether the cached geometry metadata is valid.
    pub cached_geometry_valid: bool,
}

impl WorkBuffers {
    pub fn new(ntotat: usize) -> Self {
        Self {
            gxcar: vec![[0.0; 3]; ntotat],
            radiuswork: vec![0.0; ntotat],
            fmol: Vec::new(),
            flash_ind: Vec::new(),
            flash_l: Vec::new(),
            cached_x: Vec::new(),
            cached_comptype: Vec::new(),
            cached_init1: false,
            cached_ncells: [0; 3],
            cached_cell_length: [0.0; 3],
            cached_pbc_min: [0.0; 3],
            cached_pbc_length: [0.0; 3],
            cached_geometry_valid: false,
        }
    }

    pub fn ensure_atom_capacity(&mut self, ntotat: usize) {
        if self.gxcar.len() != ntotat {
            self.gxcar.resize(ntotat, [0.0; 3]);
        }
        if self.radiuswork.len() != ntotat {
            self.radiuswork.resize(ntotat, 0.0);
        }
        self.cached_geometry_valid = false;
    }

    pub fn matches_cached_geometry(
        &self,
        x: &[F],
        comptype: &[bool],
        init1: bool,
        ncells: [usize; 3],
        cell_length: [F; 3],
        pbc_min: [F; 3],
        pbc_length: [F; 3],
    ) -> bool {
        self.cached_geometry_valid
            && self.cached_init1 == init1
            && self.cached_ncells == ncells
            && self.cached_cell_length == cell_length
            && self.cached_pbc_min == pbc_min
            && self.cached_pbc_length == pbc_length
            && self.cached_x == x
            && self.cached_comptype == comptype
    }

    pub fn update_cached_geometry(
        &mut self,
        x: &[F],
        comptype: &[bool],
        init1: bool,
        ncells: [usize; 3],
        cell_length: [F; 3],
        pbc_min: [F; 3],
        pbc_length: [F; 3],
    ) {
        self.cached_x.clear();
        self.cached_x.extend_from_slice(x);
        self.cached_comptype.clear();
        self.cached_comptype.extend_from_slice(comptype);
        self.cached_init1 = init1;
        self.cached_ncells = ncells;
        self.cached_cell_length = cell_length;
        self.cached_pbc_min = pbc_min;
        self.cached_pbc_length = pbc_length;
        self.cached_geometry_valid = true;
    }
}
