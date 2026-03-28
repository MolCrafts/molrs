//! Initialization procedures.
//! Port of `initial.f90`, `restmol.f90`, `cenmass.f90`.
//!
//! Call order matches Packmol's `initial.f90` exactly:
//!   1. compute_dmax
//!   2. restmol(type 0, solve=true) → sizemin/sizemax
//!   3. first random guess (within sizemin/sizemax)
//!   4. Phase 1: constraint-only GENCAN per type (reduced x!)
//!   5. Rescale sizemin/sizemax + compute cm_min/cm_max per type
//!   6. Setup cell grid + place fixed atoms
//!   7. Random initial point using cm_min/cm_max (restmol(false) only, up to 20 tries)
//!   8. Random angles
//!   9. Phase 2: constraint-only GENCAN per type (reduced x!)

use molrs::types::F;
use std::time::Instant;

use crate::cell::{index_cell, setcell};
use crate::constraints::EvalMode;
use crate::context::PackContext;
use crate::euler::{compcart, eulerrmat};
use crate::gencan::{GencanParams, GencanWorkspace, pgencan};
use crate::movebad::{MoveBadConfig, movebad};
use crate::random::uniform01;

use rand::Rng;

const TWO_PI: F = std::f64::consts::TAU as F;
/// Max tries for random placement per molecule (Packmol `max_guess_try = 20`).
const MAX_GUESS_TRY: usize = 20;

// ── swaptype state ─────────────────────────────────────────────────────────

/// Saved state for swaptype operations (mirrors `swaptypemod.f90`).
pub(crate) struct SwapState {
    /// Full x vector (all molecules of all types), indexed [COM...|euler...].
    xfull: Vec<F>,
    /// Original `sys.ntotmol` (all free molecules).
    ntotmol_full: usize,
}

impl SwapState {
    /// action=0: save full x.
    pub(crate) fn init(x: &[F], sys: &PackContext) -> Self {
        SwapState {
            xfull: x.to_vec(),
            ntotmol_full: sys.ntotmol,
        }
    }

    /// action=1: set up reduced x for `itype` only.
    ///
    /// Returns the compact x vector (length = nmols[itype] * 6).
    /// Also updates `sys.ntotmol` and `sys.comptype`.
    pub(crate) fn set_type(&self, itype: usize, sys: &mut PackContext) -> Vec<F> {
        // Byte-offsets in xfull for this type's COM/euler variables
        // (Packmol swaptype.f90 action 1, with 0-based indexing)
        let ilubar_start: usize = sys.nmols[0..itype].iter().sum::<usize>() * 3;
        let ilugan_start: usize = self.ntotmol_full * 3 + ilubar_start;
        let nm = sys.nmols[itype];

        let mut xtype = vec![0.0 as F; nm * 6];
        xtype[..nm * 3].copy_from_slice(&self.xfull[ilubar_start..ilubar_start + nm * 3]);
        xtype[nm * 3..nm * 6].copy_from_slice(&self.xfull[ilugan_start..ilugan_start + nm * 3]);

        // Reduce ntotmol + set comptype
        sys.ntotmol = nm;
        for i in 0..sys.ntype_with_fixed {
            sys.comptype[i] = i >= sys.ntype || i == itype;
        }

        xtype
    }

    /// action=2: save per-type results back into xfull.
    pub(crate) fn save_type(&mut self, itype: usize, xtype: &[F], sys: &PackContext) {
        let ilubar_start: usize = sys.nmols[0..itype].iter().sum::<usize>() * 3;
        let ilugan_start: usize = self.ntotmol_full * 3 + ilubar_start;
        let nm = sys.nmols[itype];

        self.xfull[ilubar_start..ilubar_start + nm * 3].copy_from_slice(&xtype[..nm * 3]);
        self.xfull[ilugan_start..ilugan_start + nm * 3].copy_from_slice(&xtype[nm * 3..nm * 6]);
    }

    /// action=3: restore full x and ntotmol.
    pub(crate) fn restore(&self, x: &mut [F], sys: &mut PackContext) {
        debug_assert_eq!(x.len(), self.xfull.len());
        x.copy_from_slice(&self.xfull);
        sys.ntotmol = self.ntotmol_full;
        for i in 0..sys.ntype_with_fixed {
            sys.comptype[i] = true;
        }
    }
}

// ── dmax ───────────────────────────────────────────────────────────────────

/// Compute maximum internal distance per molecule type.
pub fn compute_dmax(sys: &mut PackContext) {
    sys.dmax = vec![0.0 as F; sys.ntype];
    for itype in 0..sys.ntype {
        let idatom_base = sys.idfirst[itype];
        let na = sys.natoms[itype];
        for ia in 0..na {
            for ib in (ia + 1)..na {
                let a = sys.coor[idatom_base + ia];
                let b = sys.coor[idatom_base + ib];
                let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
                if d2 > sys.dmax[itype] {
                    sys.dmax[itype] = d2;
                }
            }
        }
        sys.dmax[itype] = sys.dmax[itype].sqrt();
        if sys.dmax[itype] == 0.0 {
            sys.dmax[itype] = 1.0;
        }
        log::debug!("  dmax type {itype}: {:.4}", sys.dmax[itype]);
    }
}

// ── restmol ────────────────────────────────────────────────────────────────

/// Scoped state override for `restmol`; restores context on drop.
struct RestmolScope<'a> {
    sys: &'a mut PackContext,
    itype: usize,
    ntotmol: usize,
    nmols_itype: usize,
    comptype: Vec<bool>,
    init1: bool,
}

impl<'a> RestmolScope<'a> {
    fn enter(sys: &'a mut PackContext, itype: usize) -> Self {
        let saved = Self {
            ntotmol: sys.ntotmol,
            nmols_itype: sys.nmols[itype],
            comptype: sys.comptype.clone(),
            init1: sys.init1,
            itype,
            sys,
        };

        saved.sys.ntotmol = 1;
        // Only reduce the active type to 1 molecule.
        // Other types keep their original nmols so compute_f's icart counter advances
        // correctly past them — preserving the constraint array index alignment.
        // (Packmol restmol.f90 line 34: only nmols(itype) = 1, others unchanged.)
        saved.sys.nmols[itype] = 1;
        for i in 0..saved.sys.ntype_with_fixed {
            saved.sys.comptype[i] = i == itype;
        }
        saved.sys.init1 = true; // constraint-only, no cell list

        saved
    }

    fn ctx_mut(&mut self) -> &mut PackContext {
        self.sys
    }
}

impl Drop for RestmolScope<'_> {
    fn drop(&mut self) {
        self.sys.ntotmol = self.ntotmol;
        self.sys.nmols[self.itype] = self.nmols_itype;
        self.sys.comptype.clone_from(&self.comptype);
        self.sys.init1 = self.init1;
    }
}

/// Run a single-molecule GENCAN solve (restmol).
/// Port of `restmol.f90`.
///
/// `ilubar` is the offset in `x` for the COM of this molecule.
/// Euler angles are at `x[ilubar + ntotmol*3 ..]`.
///
/// - `solve = false`: evaluate constraint function only (no optimization).
/// - `solve = true`: run GENCAN to minimize constraint violations.
///
/// On return, `sys.frest` holds the constraint violation for this molecule.
#[allow(clippy::too_many_arguments)]
pub fn restmol(
    itype: usize,
    ilubar: usize,
    x: &mut [F],
    sys: &mut PackContext,
    precision: F,
    gencan_maxit: usize,
    solve: bool,
    workspace: &mut GencanWorkspace,
) {
    let ilugan_offset = sys.ntotmol * 3;
    let mut xmol = vec![0.0 as F; 6];
    xmol[0] = x[ilubar];
    xmol[1] = x[ilubar + 1];
    xmol[2] = x[ilubar + 2];
    xmol[3] = x[ilubar + ilugan_offset];
    xmol[4] = x[ilubar + ilugan_offset + 1];
    xmol[5] = x[ilubar + ilugan_offset + 2];

    {
        let mut scope = RestmolScope::enter(sys, itype);
        let sys = scope.ctx_mut();
        if !solve {
            sys.evaluate(&xmol, EvalMode::FOnly, None);
        } else {
            let params = GencanParams {
                maxit: gencan_maxit,
                maxfc: gencan_maxit * 10,
                iprint: 0,
                ..Default::default()
            };
            pgencan(&mut xmol, sys, &params, precision, workspace);
        }
    }

    x[ilubar] = xmol[0];
    x[ilubar + 1] = xmol[1];
    x[ilubar + 2] = xmol[2];
    x[ilubar + ilugan_offset] = xmol[3];
    x[ilubar + ilugan_offset + 1] = xmol[4];
    x[ilubar + ilugan_offset + 2] = xmol[5];
    // sys.frest retains the value from the restmol compute_f
}

// ── gencan loop for one type ───────────────────────────────────────────────

/// Run the GENCAN init loop for one type on a **compact** xtype vector.
///
/// Equivalent to Packmol's per-type init loop:
/// ```text
/// do while (frest > precision .and. i < nloop0_type)
///   pgencan; computef; movebad if needed
/// done
/// ```
#[allow(clippy::too_many_arguments)]
fn init_loop_one_type(
    itype: usize,
    nloop0: usize,
    xtype: &mut [F],
    sys: &mut PackContext,
    precision: F,
    gencan_maxit: usize,
    movebad_cfg: &MoveBadConfig<'_>,
    rng: &mut impl Rng,
    t0: &Instant,
    workspace: &mut GencanWorkspace,
) {
    // Packmol calls pgencan with global maxit (default 20) inside each nloop0 loop.
    // nloop0 controls the number of outer loops, not the inner maxit.
    let params = GencanParams {
        maxit: gencan_maxit,
        maxfc: gencan_maxit * 10,
        iprint: 0,
        ..Default::default()
    };

    let mut iter = 0usize;
    sys.evaluate(xtype, EvalMode::FOnly, None);
    log::debug!(
        "[{:.3}s]     initial frest={:.4e}",
        t0.elapsed().as_secs_f64(),
        sys.frest
    );

    while sys.frest > precision && iter < nloop0 {
        iter += 1;
        pgencan(xtype, sys, &params, precision, workspace);
        sys.evaluate(xtype, EvalMode::FOnly, None);
        log::debug!(
            "[{:.3}s]     post-gencan frest={:.4e}",
            t0.elapsed().as_secs_f64(),
            sys.frest
        );
        if sys.frest > precision {
            log::debug!(
                "[{:.3}s]     movebad iter {iter}",
                t0.elapsed().as_secs_f64()
            );
            movebad(xtype, sys, precision, movebad_cfg, rng, workspace);
        }
    }
    log::debug!(
        "[{:.3}s]   type {itype} done (nloop0={nloop0}): frest={:.4e}",
        t0.elapsed().as_secs_f64(),
        sys.frest
    );
}

// ── full initialization ────────────────────────────────────────────────────

/// Full initialization procedure.
/// Faithful port of `initial.f90`.
#[allow(clippy::too_many_arguments)]
pub fn initial(
    x: &mut [F],
    sys: &mut PackContext,
    precision: F,
    discale: F,
    sidemax: F,
    nloop0: usize,
    pbc: Option<([F; 3], [F; 3])>,
    movebad_cfg: &MoveBadConfig<'_>,
    rng: &mut impl Rng,
) {
    let t0 = Instant::now();
    let mut workspace = GencanWorkspace::new();

    sys.move_flag = false;
    sys.init1 = false;
    sys.lcellfirst = None;

    for i in 0..sys.ntype_with_fixed {
        sys.comptype[i] = true;
    }

    // Packmol initial.f90 line 50-51
    sys.scale = 1.0;
    sys.scale2 = 0.01;

    // ── 1. compute dmax ──────────────────────────────────────────────────────
    log::debug!("[{:.3}s] computing dmax", t0.elapsed().as_secs_f64());
    compute_dmax(sys);

    // ── 2. restmol(type 0) → sizemin/sizemax ─────────────────────────────────
    // Packmol initial.f90 lines 77-84
    x.fill(0.0);
    log::debug!(
        "[{:.3}s] initial restmol → sizemin/sizemax",
        t0.elapsed().as_secs_f64()
    );
    restmol(
        0,
        0,
        x,
        sys,
        precision,
        movebad_cfg.gencan_maxit,
        true,
        &mut workspace,
    );
    let cm0 = [x[0], x[1], x[2]];
    sys.sizemin = [cm0[0] - sidemax, cm0[1] - sidemax, cm0[2] - sidemax];
    sys.sizemax = [cm0[0] + sidemax, cm0[1] + sidemax, cm0[2] + sidemax];
    log::debug!(
        "[{:.3}s] sizemin={:?}  sizemax={:?}",
        t0.elapsed().as_secs_f64(),
        sys.sizemin,
        sys.sizemax
    );

    // ── 3. first random guess ─────────────────────────────────────────────────
    // Packmol initial.f90 lines 88-117
    log::debug!(
        "[{:.3}s] generating first random guess",
        t0.elapsed().as_secs_f64()
    );
    {
        let mut ilubar = 0usize;
        let mut ilugan = sys.ntotmol * 3;
        for itype in 0..sys.ntype {
            for _imol in 0..sys.nmols[itype] {
                x[ilubar] = sys.sizemin[0] + uniform01(rng) * (sys.sizemax[0] - sys.sizemin[0]);
                x[ilubar + 1] = sys.sizemin[1] + uniform01(rng) * (sys.sizemax[1] - sys.sizemin[1]);
                x[ilubar + 2] = sys.sizemin[2] + uniform01(rng) * (sys.sizemax[2] - sys.sizemin[2]);
                x[ilugan] = random_angle_for_type(itype, 0, sys, rng);
                x[ilugan + 1] = random_angle_for_type(itype, 1, sys, rng);
                x[ilugan + 2] = random_angle_for_type(itype, 2, sys, rng);
                ilubar += 3;
                ilugan += 3;
            }
        }
    }

    // Init xcart (Packmol initial.f90 lines 121-138)
    init_xcart_from_x(x, sys);

    // Mark fixed atoms (Packmol initial.f90 lines 140-165)
    let free_atoms = sys.ntotat - sys.nfixedat;
    for icart in free_atoms..sys.ntotat {
        sys.fixedatom[icart] = true;
    }

    // ── 4. Phase 1: constraint-only GENCAN per type (reduced x) ──────────────
    // Packmol initial.f90 lines 174-224 (via swaptype)
    log::debug!(
        "[{:.3}s] Phase 1: constraint-only GENCAN ({} types, nloop0={})",
        t0.elapsed().as_secs_f64(),
        sys.ntype,
        nloop0,
    );
    sys.init1 = true;
    {
        let mut swap = SwapState::init(x, sys);
        for itype in 0..sys.ntype {
            let nm = sys.nmols[itype];
            log::debug!(
                "[{:.3}s]   type {itype}: {nm} mols × {} atoms  (n={})",
                t0.elapsed().as_secs_f64(),
                sys.natoms[itype],
                nm * 6
            );
            let mut xtype = swap.set_type(itype, sys);
            init_loop_one_type(
                itype,
                nloop0,
                &mut xtype,
                sys,
                precision,
                movebad_cfg.gencan_maxit,
                movebad_cfg,
                rng,
                &t0,
                &mut workspace,
            );
            swap.save_type(itype, &xtype, sys);
        }
        swap.restore(x, sys);
    }
    sys.init1 = false;

    // ── 5. Rescale sizemin/sizemax + compute cm_min/cm_max ───────────────────
    // Packmol initial.f90 lines 227-336
    log::debug!(
        "[{:.3}s] rescaling bounds + computing cm_min/cm_max",
        t0.elapsed().as_secs_f64()
    );

    // Update xcart from the Phase-1 result
    init_xcart_from_x(x, sys);

    // Packmol sets radmax as the maximum *diameter* (2 * radius),
    // not the maximum radius (packmol.f90 lines 532-534).
    let radmax = sys
        .radius_ini
        .iter()
        .copied()
        .map(|r| 2.0 * r)
        .fold(0.0 as F, F::max);

    let mut smin = [1.0e20 as F; 3];
    let mut smax = [-1.0e20 as F; 3];

    // Fixed atoms (Packmol lines 234-246)
    for icart in free_atoms..sys.ntotat {
        let pos = sys.xcart[icart];
        for k in 0..3 {
            smin[k] = smin[k].min(pos[k]);
            smax[k] = smax[k].max(pos[k]);
        }
    }

    // Free atoms + compute cm_min/cm_max per type (Packmol lines 248-336)
    let mut cm_min_per_type = vec![[1.0e20 as F; 3]; sys.ntype];
    let mut cm_max_per_type = vec![[-1.0e20 as F; 3]; sys.ntype];
    {
        let mut icart = 0usize;
        for itype in 0..sys.ntype {
            for _imol in 0..sys.nmols[itype] {
                let mut xcm = [0.0 as F; 3];
                for _iatom in 0..sys.natoms[itype] {
                    let pos = sys.xcart[icart];
                    for k in 0..3 {
                        smin[k] = smin[k].min(pos[k]);
                        smax[k] = smax[k].max(pos[k]);
                        xcm[k] += pos[k];
                    }
                    icart += 1;
                }
                let na = sys.natoms[itype] as F;
                for k in 0..3 {
                    xcm[k] /= na;
                    cm_min_per_type[itype][k] = cm_min_per_type[itype][k].min(xcm[k]);
                    cm_max_per_type[itype][k] = cm_max_per_type[itype][k].max(xcm[k]);
                }
            }
        }
    }

    // Guard: if no atoms were found, fall back to sizemin/sizemax
    for k in 0..3 {
        if smin[k] > 1.0e19 {
            smin[k] = sys.sizemin[k];
            smax[k] = sys.sizemax[k];
        }
    }

    // Apply 1.1*radmax padding (Packmol lines 266-267)
    for k in 0..3 {
        sys.sizemin[k] = smin[k] - 1.1 * radmax;
        sys.sizemax[k] = smax[k] + 1.1 * radmax;
    }

    log::debug!(
        "[{:.3}s] sizemin={:?}  sizemax={:?}  radmax={:.4}",
        t0.elapsed().as_secs_f64(),
        sys.sizemin,
        sys.sizemax,
        radmax
    );
    for itype in 0..sys.ntype {
        log::debug!(
            "[{:.3}s]   type {itype} cm range: min={:?}  max={:?}",
            t0.elapsed().as_secs_f64(),
            cm_min_per_type[itype],
            cm_max_per_type[itype]
        );
    }

    // ── 6. Setup periodic box + cell grid + fixed atoms ──────────────────────
    // Packmol initial.f90 lines 272-317
    if let Some((pbc_min, pbc_max)) = pbc {
        sys.pbc_min = pbc_min;
        sys.pbc_length = [
            pbc_max[0] - pbc_min[0],
            pbc_max[1] - pbc_min[1],
            pbc_max[2] - pbc_min[2],
        ];
    } else {
        sys.pbc_min = sys.sizemin;
        sys.pbc_length = [
            sys.sizemax[0] - sys.sizemin[0],
            sys.sizemax[1] - sys.sizemin[1],
            sys.sizemax[2] - sys.sizemin[2],
        ];
    }

    let cell_side = if radmax > 0.0 {
        discale * 1.01 * radmax
    } else {
        1.0
    };
    log::debug!(
        "[{:.3}s] setting up cell grid (cell_side={:.4})",
        t0.elapsed().as_secs_f64(),
        cell_side
    );
    for k in 0..3 {
        sys.ncells[k] = ((sys.pbc_length[k] / cell_side).floor() as usize).max(1);
        sys.cell_length[k] = sys.pbc_length[k] / sys.ncells[k] as F;
    }
    log::debug!(
        "[{:.3}s] ncells={:?}  cell_length={:?}",
        t0.elapsed().as_secs_f64(),
        sys.ncells,
        sys.cell_length
    );

    sys.resize_cell_arrays();

    // Add fixed atoms to latomfix (Packmol lines 303-318)
    for icart in free_atoms..sys.ntotat {
        let pos = sys.xcart[icart];
        let cell = setcell(
            &pos,
            &sys.pbc_min,
            &sys.pbc_length,
            &sys.cell_length,
            &sys.ncells,
        );
        let icell = index_cell(&cell, &sys.ncells);
        if sys.latomfix[icell].is_none() {
            sys.fixed_cells.push(icell);
        }
        sys.latomnext[icart] = sys.latomfix[icell];
        sys.latomfix[icell] = Some(icart);
    }

    // ── 7. Random initial point using cm_min/cm_max ───────────────────────────
    // Packmol initial.f90 lines 362-427
    // For each molecule: try up to MAX_GUESS_TRY random positions within the
    // per-type COM bounding box, calling restmol(false) to check constraints.
    // Packmol does NOT call restmol(true) per molecule here.
    log::debug!(
        "[{:.3}s] setting random initial point ({} types, {} total mols)",
        t0.elapsed().as_secs_f64(),
        sys.ntype,
        sys.ntotmol
    );
    {
        let mut ilubar = 0usize;
        for itype in 0..sys.ntype {
            let cm_lo = cm_min_per_type[itype];
            let cm_hi = cm_max_per_type[itype];
            let nmols = sys.nmols[itype];
            log::debug!(
                "[{:.3}s]   type {itype}: {nmols} mols, \
                 cm_x=[{:.2},{:.2}] cm_y=[{:.2},{:.2}] cm_z=[{:.2},{:.2}]",
                t0.elapsed().as_secs_f64(),
                cm_lo[0],
                cm_hi[0],
                cm_lo[1],
                cm_hi[1],
                cm_lo[2],
                cm_hi[2]
            );
            for _imol in 0..nmols {
                // Try up to MAX_GUESS_TRY random positions (restmol(false) only)
                let mut ntry = 0usize;
                let mut fmol = 1.0 as F;
                while fmol > precision && ntry < MAX_GUESS_TRY {
                    ntry += 1;
                    let rx: F = uniform01(rng);
                    let ry: F = uniform01(rng);
                    let rz: F = uniform01(rng);
                    x[ilubar] = cm_lo[0] + rx * (cm_hi[0] - cm_lo[0]);
                    x[ilubar + 1] = cm_lo[1] + ry * (cm_hi[1] - cm_lo[1]);
                    x[ilubar + 2] = cm_lo[2] + rz * (cm_hi[2] - cm_lo[2]);
                    restmol(
                        itype,
                        ilubar,
                        x,
                        sys,
                        precision,
                        movebad_cfg.gencan_maxit,
                        false,
                        &mut workspace,
                    );
                    fmol = sys.frest;
                }
                ilubar += 3;
            }
            log::debug!(
                "[{:.3}s]   type {itype} placement done",
                t0.elapsed().as_secs_f64()
            );
        }
    }

    // ── 8. Random angles ─────────────────────────────────────────────────────
    // Packmol initial.f90 lines 431-458
    log::debug!("[{:.3}s] setting random angles", t0.elapsed().as_secs_f64());
    {
        let mut ilugan = sys.ntotmol * 3;
        for itype in 0..sys.ntype {
            for _imol in 0..sys.nmols[itype] {
                x[ilugan] = random_angle_for_type(itype, 0, sys, rng);
                x[ilugan + 1] = random_angle_for_type(itype, 1, sys, rng);
                x[ilugan + 2] = random_angle_for_type(itype, 2, sys, rng);
                ilugan += 3;
            }
        }
    }

    // ── 9. Phase 2: constraint-only GENCAN per type (reduced x) ──────────────
    // Packmol initial.f90 lines 516-550
    log::debug!(
        "[{:.3}s] Phase 2: constraint-only GENCAN ({} types, nloop0={})",
        t0.elapsed().as_secs_f64(),
        sys.ntype,
        nloop0,
    );
    sys.init1 = true;
    {
        let mut swap = SwapState::init(x, sys);
        for itype in 0..sys.ntype {
            let nm = sys.nmols[itype];
            log::debug!(
                "[{:.3}s]   type {itype}: {nm} mols × {} atoms  (n={})",
                t0.elapsed().as_secs_f64(),
                sys.natoms[itype],
                nm * 6
            );
            let mut xtype = swap.set_type(itype, sys);
            init_loop_one_type(
                itype,
                nloop0,
                &mut xtype,
                sys,
                precision,
                movebad_cfg.gencan_maxit,
                movebad_cfg,
                rng,
                &t0,
                &mut workspace,
            );
            swap.save_type(itype, &xtype, sys);
        }
        swap.restore(x, sys);
    }
    sys.init1 = false;

    log::debug!("[{:.3}s] initial() complete", t0.elapsed().as_secs_f64());
}

fn random_angle_for_type(itype: usize, axis: usize, sys: &PackContext, rng: &mut impl Rng) -> F {
    if sys.constrain_rot[itype][axis] {
        let center = sys.rot_bound[itype][axis][0];
        let half_width = sys.rot_bound[itype][axis][1].abs();
        (center - half_width) + 2.0 * uniform01(rng) * half_width
    } else {
        TWO_PI * uniform01(rng)
    }
}

// ── init_xcart_from_x ──────────────────────────────────────────────────────

/// Initialize xcart from x (COM + Euler angles).
pub fn init_xcart_from_x(x: &[F], sys: &mut PackContext) {
    let mut ilubar = 0usize;
    let mut ilugan = sys.ntotmol * 3;
    let mut icart = 0usize;

    for itype in 0..sys.ntype {
        for _imol in 0..sys.nmols[itype] {
            let xcm = [x[ilubar], x[ilubar + 1], x[ilubar + 2]];
            let beta = x[ilugan];
            let gama = x[ilugan + 1];
            let teta = x[ilugan + 2];
            let (v1, v2, v3) = eulerrmat(beta, gama, teta);

            let idatom_base = sys.idfirst[itype];
            for iatom in 0..sys.natoms[itype] {
                let pos = compcart(&xcm, &sys.coor[idatom_base + iatom], &v1, &v2, &v3);
                sys.xcart[icart] = pos;
                sys.fixedatom[icart] = false;
                icart += 1;
            }

            ilugan += 3;
            ilubar += 3;
        }
    }
}
