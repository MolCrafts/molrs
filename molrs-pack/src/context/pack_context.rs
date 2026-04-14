//! Packing runtime context — mirrors Packmol `compute_data` behavior.

use std::sync::Arc;

use crate::cell::{cell_ind, icell_to_cell, index_cell};
use crate::constraints::{Constraints, EvalMode, EvalOutput};
use crate::restraint::Restraint;
use molrs::Element;
use molrs::types::F;

use super::model::ModelData;
use super::state::{RuntimeState, RuntimeStateMut};
use super::work_buffers::WorkBuffers;

/// Index of a restraint assigned to a specific atom.
pub type RestraintRef = usize;

/// Neighbor offsets used by `computef.f90` (13 forward neighbors).
const NEIGHBOR_OFFSETS_F: [(isize, isize, isize); 13] = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
    (0, 1, 1),
    (1, 1, 0),
    (1, 0, 1),
    (1, -1, -1),
    (1, -1, 1),
    (1, 1, -1),
    (1, 1, 1),
];

/// Neighbor offsets used by `computeg.f90` (13 forward neighbors, different order).
const NEIGHBOR_OFFSETS_G: [(isize, isize, isize); 13] = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 0),
    (1, 0, 1),
    (1, -1, 0),
    (1, 0, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
];

/// Full runtime context for one packing execution.
/// All arrays are 0-based; Fortran 1-based arrays are shifted by -1.
pub struct PackContext {
    // ---- Constraints facade ----
    pub constraints: Constraints,

    // ---- Atom Cartesian coordinates (updated every function evaluation) ----
    /// Current Cartesian positions: xcart[icart] = [x, y, z]. Size: ntotat.
    pub xcart: Vec<[F; 3]>,
    /// Element per atom: elements[icart]. Size: ntotat. `None` means unknown/"X".
    pub elements: Vec<Option<Element>>,

    // ---- Reference (centered) coordinates ----
    /// Reference coordinates coor[idatom] = [x, y, z]. Size: total atoms across all types.
    pub coor: Vec<[F; 3]>,

    // ---- Radii ----
    /// Current radii (may be scaled): radius[icart]. Size: ntotat.
    pub radius: Vec<F>,
    /// Original (unscaled) radii: radius_ini[icart]. Size: ntotat.
    pub radius_ini: Vec<F>,
    /// Function scaling per atom: fscale[icart]. Size: ntotat.
    pub fscale: Vec<F>,

    // ---- Short radius (optional secondary penalty) ----
    pub use_short_radius: Vec<bool>,
    pub short_radius: Vec<F>,
    pub short_radius_scale: Vec<F>,

    // ---- Objective function accumulators ----
    /// Maximum inter-molecular distance violation (fdist in Fortran).
    pub fdist: F,
    /// Maximum constraint violation (frest in Fortran).
    pub frest: F,
    /// Per-atom distance violation (for movebad).
    pub fdist_atom: Vec<F>,
    /// Per-atom constraint violation (for movebad).
    pub frest_atom: Vec<F>,

    // ---- Molecule topology ----
    /// Number of molecules per type: nmols[itype]. 0-based type index.
    pub nmols: Vec<usize>,
    /// Number of atoms per type: natoms[itype]. 0-based type index.
    pub natoms: Vec<usize>,
    /// First datum atom index (0-based) for each type: idfirst[itype].
    pub idfirst: Vec<usize>,
    /// Total number of types (free).
    pub ntype: usize,
    /// Total number of types including fixed types.
    pub ntype_with_fixed: usize,
    /// Total number of free molecules.
    pub ntotmol: usize,
    /// Total number of atoms (free + fixed).
    pub ntotat: usize,
    /// Number of fixed atoms.
    pub nfixedat: usize,

    // ---- Rotation constraints (Packmol constrain_rotation) ----
    /// Rotation constraint flags per free type in Euler variable order
    /// [beta(y), gama(z), teta(x)].
    pub constrain_rot: Vec<[bool; 3]>,
    /// Rotation bounds per free type and Euler variable:
    /// [center_rad, half_width_rad].
    pub rot_bound: Vec<[[F; 2]; 3]>,

    // ---- Restraints ----
    /// All restraints pool: restraints[irest].
    pub restraints: Vec<Arc<dyn Restraint>>,
    /// CSR offsets for per-atom restraint indices:
    /// restraints of atom `icart` are in `iratom_data[iratom_offsets[icart]..iratom_offsets[icart+1]]`.
    pub iratom_offsets: Vec<usize>,
    /// Flattened per-atom restraint indices.
    pub iratom_data: Vec<RestraintRef>,

    // ---- Cell list bookkeeping ----
    /// Type index per atom: ibtype[icart] (0-based type index).
    pub ibtype: Vec<usize>,
    /// Molecule index within its type: ibmol[icart] (0-based).
    pub ibmol: Vec<usize>,
    /// Is this a fixed atom?
    pub fixedatom: Vec<bool>,
    /// Is this type being computed in the current iteration?
    pub comptype: Vec<bool>,

    // ---- Cell geometry ----
    pub ncells: [usize; 3],
    pub cell_length: [F; 3],
    pub pbc_length: [F; 3],
    pub pbc_min: [F; 3],

    // ---- Linked cell lists ----
    /// latomfirst[i][j][k] = first atom index in cell (i,j,k), None means empty.
    /// Stored as flat Vec indexed by `index_cell`.
    pub latomfirst: Vec<Option<usize>>,
    /// latomnext[icart] = next atom in the same cell (None = end).
    pub latomnext: Vec<Option<usize>>,
    /// Fixed atom list per cell (permanent).
    pub latomfix: Vec<Option<usize>>,
    /// Occupied cell linked list: first cell.
    pub lcellfirst: Option<usize>,
    /// lcellnext[icell] = next occupied cell.
    pub lcellnext: Vec<Option<usize>>,
    /// Is cell empty?
    pub empty_cell: Vec<bool>,
    /// Cells that contain fixed atoms and must be restored on every reset.
    pub fixed_cells: Vec<usize>,
    /// Cells touched during the previous objective/gradient evaluation.
    pub active_cells: Vec<usize>,
    /// Precomputed 13 forward-neighbor cell indices per cell for `compute_f`.
    pub neighbor_cells_f: Vec<[usize; 13]>,
    /// Precomputed 13 forward-neighbor cell indices per cell for `compute_g`.
    pub neighbor_cells_g: Vec<[usize; 13]>,

    // ---- State flags ----
    /// If true, skip pair-distance computations (constraints only during init).
    pub init1: bool,
    /// If true, accumulate per-atom fdist/frest (movebad mode).
    pub move_flag: bool,

    // ---- Algorithm parameters ----
    pub scale: F,
    pub scale2: F,

    // ---- Bounding box ----
    pub sizemin: [F; 3],
    pub sizemax: [F; 3],

    // ---- Maximum internal distances per type ----
    pub dmax: Vec<F>,

    // ---- Work buffers ----
    pub work: WorkBuffers,

    // ---- Output frame (owned, built incrementally) ----
    /// Frame that accumulates constant columns (element, mol_id) during init
    /// and receives position columns at the end of packing.
    pub frame: molrs::Frame,

    // ---- Debug: call counters (zeroed per pgencan call) ----
    ncf: usize,
    ncg: usize,
}

impl PackContext {
    /// Allocate and zero-initialize all arrays.
    pub fn new(ntotat: usize, ntotmol: usize, ntype: usize) -> Self {
        let ncells = [1, 1, 1];
        let ncell_total = ncells[0] * ncells[1] * ncells[2];
        Self {
            constraints: Constraints,
            xcart: vec![[0.0; 3]; ntotat],
            elements: vec![None; ntotat],
            coor: Vec::new(),
            radius: vec![0.0; ntotat],
            radius_ini: vec![0.0; ntotat],
            fscale: vec![1.0; ntotat],
            use_short_radius: vec![false; ntotat],
            short_radius: vec![0.0; ntotat],
            short_radius_scale: vec![0.0; ntotat],
            fdist: 0.0,
            frest: 0.0,
            fdist_atom: vec![0.0; ntotat],
            frest_atom: vec![0.0; ntotat],
            nmols: Vec::new(),
            natoms: Vec::new(),
            idfirst: Vec::new(),
            ntype,
            ntype_with_fixed: ntype,
            ntotmol,
            ntotat,
            nfixedat: 0,
            constrain_rot: vec![[false; 3]; ntype],
            rot_bound: vec![[[0.0; 2]; 3]; ntype],
            restraints: Vec::new(),
            iratom_offsets: vec![0; ntotat + 1],
            iratom_data: Vec::new(),
            ibtype: vec![0; ntotat],
            ibmol: vec![0; ntotat],
            fixedatom: vec![false; ntotat],
            comptype: vec![true; ntype],
            ncells,
            cell_length: [1.0; 3],
            pbc_length: [1.0; 3],
            pbc_min: [0.0; 3],
            latomfirst: vec![None; ncell_total],
            latomnext: vec![None; ntotat],
            latomfix: vec![None; ncell_total],
            lcellfirst: None,
            lcellnext: vec![None; ncell_total],
            empty_cell: vec![true; ncell_total],
            fixed_cells: Vec::new(),
            active_cells: Vec::new(),
            neighbor_cells_f: vec![[0; 13]; ncell_total],
            neighbor_cells_g: vec![[0; 13]; ncell_total],
            init1: false,
            move_flag: false,
            scale: 1.0,
            scale2: 0.01,
            sizemin: [0.0; 3],
            sizemax: [0.0; 3],
            dmax: vec![0.0; ntype],
            work: WorkBuffers::new(ntotat),
            frame: molrs::Frame::new(),
            ncf: 0,
            ncg: 0,
        }
    }

    /// Context view for mostly static model data.
    #[inline]
    pub fn model(&self) -> ModelData<'_> {
        ModelData { ctx: self }
    }

    /// Read-only runtime state view.
    #[inline]
    pub fn runtime(&self) -> RuntimeState<'_> {
        RuntimeState { ctx: self }
    }

    /// Mutable runtime state view.
    #[inline]
    pub fn runtime_mut(&mut self) -> RuntimeStateMut<'_> {
        RuntimeStateMut { ctx: self }
    }

    /// Unified constraints evaluation entrypoint.
    #[inline]
    pub fn evaluate(&mut self, x: &[F], mode: EvalMode, gradient: Option<&mut [F]>) -> EvalOutput {
        let constraints = self.constraints;
        constraints.evaluate(x, self, mode, gradient)
    }

    /// Resize cell list arrays after ncells is set.
    pub fn resize_cell_arrays(&mut self) {
        let nc = self.ncells[0] * self.ncells[1] * self.ncells[2];
        self.latomfirst = vec![None; nc];
        self.latomfix = vec![None; nc];
        self.lcellnext = vec![None; nc];
        self.empty_cell = vec![true; nc];
        self.fixed_cells.clear();
        self.active_cells.clear();
        self.neighbor_cells_f = vec![[0; 13]; nc];
        self.neighbor_cells_g = vec![[0; 13]; nc];
        self.rebuild_neighbor_cells();
    }

    /// Reset cell lists (called at start of each compute_f/compute_g).
    /// Port of `resetcells.f90`.
    pub fn resetcells(&mut self) {
        self.lcellfirst = None;
        for &icell in &self.active_cells {
            self.latomfirst[icell] = None;
            self.lcellnext[icell] = None;
            self.empty_cell[icell] = true;
        }
        self.active_cells.clear();

        for &icell in &self.fixed_cells {
            self.latomfirst[icell] = self.latomfix[icell];
            self.empty_cell[icell] = false;
            self.lcellnext[icell] = self.lcellfirst;
            self.lcellfirst = Some(icell);
            self.active_cells.push(icell);
        }

        // Reset latomnext for free atoms only
        let free_atoms = self.ntotat - self.nfixedat;
        for i in 0..free_atoms {
            self.latomnext[i] = None;
        }
    }

    #[inline]
    pub fn reset_eval_counters(&mut self) {
        self.ncf = 0;
        self.ncg = 0;
    }

    #[inline]
    pub fn increment_ncf(&mut self) {
        self.ncf += 1;
    }

    #[inline]
    pub fn increment_ncg(&mut self) {
        self.ncg += 1;
    }

    #[inline]
    pub fn ncf(&self) -> usize {
        self.ncf
    }

    #[inline]
    pub fn ncg(&self) -> usize {
        self.ncg
    }

    fn rebuild_neighbor_cells(&mut self) {
        let (nx, ny, nz) = (self.ncells[0], self.ncells[1], self.ncells[2]);
        let nc = nx * ny * nz;
        for icell in 0..nc {
            let cell = icell_to_cell(icell, &self.ncells);
            let (ci, cj, ck) = (cell[0], cell[1], cell[2]);

            let mut nbs_f = [0usize; 13];
            for (idx, &(di, dj, dk)) in NEIGHBOR_OFFSETS_F.iter().enumerate() {
                let ncell = [
                    cell_ind(ci as isize + di, nx),
                    cell_ind(cj as isize + dj, ny),
                    cell_ind(ck as isize + dk, nz),
                ];
                nbs_f[idx] = index_cell(&ncell, &self.ncells);
            }
            self.neighbor_cells_f[icell] = nbs_f;

            let mut nbs_g = [0usize; 13];
            for (idx, &(di, dj, dk)) in NEIGHBOR_OFFSETS_G.iter().enumerate() {
                let ncell = [
                    cell_ind(ci as isize + di, nx),
                    cell_ind(cj as isize + dj, ny),
                    cell_ind(ck as isize + dk, nz),
                ];
                nbs_g[idx] = index_cell(&ncell, &self.ncells);
            }
            self.neighbor_cells_g[icell] = nbs_g;
        }
    }
}
