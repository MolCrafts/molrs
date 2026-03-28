//! Objective function and gradient computation.
//! Exact port of `computef.f90`, `computeg.f90`, `fparc.f90`, `gparc.f90`.

use crate::cell::{index_cell, setcell};
use crate::context::PackContext;
use crate::euler::{compcart, eulerrmat, eulerrmat_derivatives};
use molrs::types::F;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Clone, Copy)]
enum ExpandMode {
    F,
    G,
    FG,
}

#[cfg(feature = "rayon")]
const PARALLEL_PAIR_CELL_THRESHOLD: usize = 64;

/// Compute objective function value and update `sys.xcart`.
/// Port of `computef.f90`.
///
/// `x` layout: [COM₀..COMₙ (3N)] ++ [euler₀..eulerₙ (3N)]
pub fn compute_f(x: &[F], sys: &mut PackContext) -> F {
    sys.increment_ncf();
    sys.fdist = 0.0;
    sys.frest = 0.0;

    if !sys.init1 {
        sys.resetcells();
    }

    let mut f = expand_molecules(x, sys, ExpandMode::F);

    if sys.init1 {
        update_cached_geometry(x, sys);
        return f;
    }

    f += accumulate_pair_f(sys);
    update_cached_geometry(x, sys);

    f
}

/// Compute objective function value and gradient in one pass over geometry/state.
/// This avoids rebuilding Cartesian coordinates and cell lists twice.
pub fn compute_fg(x: &[F], sys: &mut PackContext, g: &mut [F]) -> F {
    sys.increment_ncf();
    sys.increment_ncg();
    sys.fdist = 0.0;
    sys.frest = 0.0;
    sys.work.gxcar.fill([0.0; 3]);

    if !sys.init1 {
        sys.resetcells();
    }

    let mut f = expand_molecules(x, sys, ExpandMode::FG);

    if !sys.init1 {
        f += accumulate_pair_fg(sys);
    }

    update_cached_geometry(x, sys);
    project_cartesian_gradient(x, sys, g);

    f
}

/// Atom-pair distance penalty function.
/// Port of `fparc.f90`.
#[inline(always)]
fn fparc(icart: usize, first_jcart: Option<usize>, sys: &mut PackContext) -> F {
    let mut result = 0.0;
    let mut jcart_id = first_jcart;
    let xi = sys.xcart[icart];
    let ri = sys.radius[icart];
    let ri_ini = sys.radius_ini[icart];
    let fsi = sys.fscale[icart];
    let move_flag = sys.move_flag;

    while let Some(jcart) = jcart_id {
        // Skip same molecule
        if sys.ibmol[icart] == sys.ibmol[jcart] && sys.ibtype[icart] == sys.ibtype[jcart] {
            jcart_id = sys.latomnext[jcart];
            continue;
        }
        // Skip two fixed atoms
        if sys.fixedatom[icart] && sys.fixedatom[jcart] {
            jcart_id = sys.latomnext[jcart];
            continue;
        }

        let xj = sys.xcart[jcart];
        let mut dx = xi[0] - xj[0];
        let mut dy = xi[1] - xj[1];
        let mut dz = xi[2] - xj[2];
        if sys.pbc_length[0] > 0.0 {
            dx -= (dx / sys.pbc_length[0]).round() * sys.pbc_length[0];
        }
        if sys.pbc_length[1] > 0.0 {
            dy -= (dy / sys.pbc_length[1]).round() * sys.pbc_length[1];
        }
        if sys.pbc_length[2] > 0.0 {
            dz -= (dz / sys.pbc_length[2]).round() * sys.pbc_length[2];
        }
        let datom = dx * dx + dy * dy + dz * dz;
        let rsum = ri + sys.radius[jcart];
        let tol = rsum * rsum;

        if datom < tol {
            let penalty = datom - tol;
            result += fsi * sys.fscale[jcart] * penalty * penalty;
            // Short radius penalty
            if sys.use_short_radius[icart] || sys.use_short_radius[jcart] {
                let short_rsum = sys.short_radius[icart] + sys.short_radius[jcart];
                let short_tol = short_rsum * short_rsum;
                if datom < short_tol {
                    let penalty = datom - short_tol;
                    let mut sr_scale =
                        (sys.short_radius_scale[icart] * sys.short_radius_scale[jcart]).sqrt();
                    sr_scale *= (tol * tol) / (short_tol * short_tol);
                    result += fsi * sys.fscale[jcart] * sr_scale * penalty * penalty;
                }
            }
        }

        let rsum_ini = ri_ini + sys.radius_ini[jcart];
        let tol_ini = rsum_ini * rsum_ini;
        let violation = tol_ini - datom;
        if violation > sys.fdist {
            sys.fdist = violation;
        }
        if move_flag {
            if violation > sys.fdist_atom[icart] {
                sys.fdist_atom[icart] = violation;
            }
            if violation > sys.fdist_atom[jcart] {
                sys.fdist_atom[jcart] = violation;
            }
        }

        jcart_id = sys.latomnext[jcart];
    }

    result
}

/// Compute gradient `g` from current system state.
/// Port of `computeg.f90`.
pub fn compute_g(x: &[F], sys: &mut PackContext, g: &mut [F]) {
    sys.increment_ncg();
    // Zero Cartesian gradient
    sys.work.gxcar.fill([0.0; 3]);

    if matches_cached_geometry(x, sys) {
        accumulate_constraint_gradient_from_xcart(sys);
        if !sys.init1 {
            accumulate_pair_g(sys);
        }
        project_cartesian_gradient(x, sys, g);
        return;
    }

    if !sys.init1 {
        sys.resetcells();
    }

    expand_molecules(x, sys, ExpandMode::G);

    if !sys.init1 {
        accumulate_pair_g(sys);
    }

    update_cached_geometry(x, sys);
    project_cartesian_gradient(x, sys, g);
}

/// Expand all active molecules into `sys.xcart`, applying either constraint values
/// (`computef`) or constraint gradients (`computeg`), and rebuilding the cell list.
#[inline]
fn expand_molecules(x: &[F], sys: &mut PackContext, mode: ExpandMode) -> F {
    let mut f = 0.0;
    let mut ilubar = 0usize;
    let mut ilugan = sys.ntotmol * 3;
    let mut icart = 0usize;

    // Packmol computef/computeg loop only over free types (1..ntype).
    for itype in 0..sys.ntype {
        if !sys.comptype[itype] {
            icart += sys.nmols[itype] * sys.natoms[itype];
            continue;
        }

        for imol in 0..sys.nmols[itype] {
            let xcm = [x[ilubar], x[ilubar + 1], x[ilubar + 2]];
            let beta = x[ilugan];
            let gama = x[ilugan + 1];
            let teta = x[ilugan + 2];
            let (v1, v2, v3) = eulerrmat(beta, gama, teta);
            let idatom_base = sys.idfirst[itype];

            for iatom in 0..sys.natoms[itype] {
                let idatom = idatom_base + iatom;
                let pos = compcart(&xcm, &sys.coor[idatom], &v1, &v2, &v3);
                sys.xcart[icart] = pos;

                match mode {
                    ExpandMode::F => {
                        f += accumulate_constraint_value(icart, &pos, sys);
                    }
                    ExpandMode::G => {
                        accumulate_constraint_gradient(icart, &pos, sys);
                    }
                    ExpandMode::FG => {
                        f += accumulate_constraint_value(icart, &pos, sys);
                        accumulate_constraint_gradient(icart, &pos, sys);
                    }
                }

                if !sys.init1 {
                    insert_atom_in_cell(icart, itype, imol, &pos, sys);
                }

                icart += 1;
            }

            ilugan += 3;
            ilubar += 3;
        }
    }

    f
}

#[inline(always)]
fn accumulate_constraint_value(icart: usize, pos: &[F; 3], sys: &mut PackContext) -> F {
    let mut fplus = 0.0;
    let start = sys.iratom_offsets[icart];
    let end = sys.iratom_offsets[icart + 1];
    for &irest in &sys.iratom_data[start..end] {
        fplus += sys.restraints[irest].value(pos, sys.scale, sys.scale2);
    }
    if fplus > sys.frest {
        sys.frest = fplus;
    }
    if sys.move_flag {
        sys.frest_atom[icart] += fplus;
    }
    fplus
}

#[inline(always)]
fn accumulate_constraint_gradient(icart: usize, pos: &[F; 3], sys: &mut PackContext) {
    let start = sys.iratom_offsets[icart];
    let end = sys.iratom_offsets[icart + 1];
    let scale = sys.scale;
    let scale2 = sys.scale2;
    let gc = &mut sys.work.gxcar[icart];
    for &irest in &sys.iratom_data[start..end] {
        sys.restraints[irest].gradient(pos, scale, scale2, gc);
    }
}

#[inline]
fn accumulate_constraint_gradient_from_xcart(sys: &mut PackContext) {
    let mut icart = 0usize;

    for itype in 0..sys.ntype {
        if !sys.comptype[itype] {
            icart += sys.nmols[itype] * sys.natoms[itype];
            continue;
        }

        for _imol in 0..sys.nmols[itype] {
            for _iatom in 0..sys.natoms[itype] {
                let pos = sys.xcart[icart];
                accumulate_constraint_gradient(icart, &pos, sys);
                icart += 1;
            }
        }
    }
}

#[inline(always)]
fn insert_atom_in_cell(
    icart: usize,
    itype: usize,
    imol: usize,
    pos: &[F; 3],
    sys: &mut PackContext,
) {
    let cell = setcell(
        pos,
        &sys.pbc_min,
        &sys.pbc_length,
        &sys.cell_length,
        &sys.ncells,
    );
    let icell = index_cell(&cell, &sys.ncells);
    sys.latomnext[icart] = sys.latomfirst[icell];
    sys.latomfirst[icell] = Some(icart);

    if sys.empty_cell[icell] {
        sys.empty_cell[icell] = false;
        sys.active_cells.push(icell);
        sys.lcellnext[icell] = sys.lcellfirst;
        sys.lcellfirst = Some(icell);
    }

    sys.ibtype[icart] = itype;
    sys.ibmol[icart] = imol;
}

#[inline(always)]
fn accumulate_pair_f(sys: &mut PackContext) -> F {
    #[cfg(feature = "rayon")]
    if !sys.move_flag && sys.active_cells.len() >= PARALLEL_PAIR_CELL_THRESHOLD {
        let (f, fdist_max) = accumulate_pair_f_parallel(sys);
        sys.fdist = sys.fdist.max(fdist_max);
        return f;
    }

    let mut f = 0.0;
    let mut icell_id = sys.lcellfirst;
    while let Some(icell) = icell_id {
        let neighbors = sys.neighbor_cells_f[icell];

        let mut icart_id = sys.latomfirst[icell];
        while let Some(icart) = icart_id {
            f += fparc(icart, sys.latomnext[icart], sys);
            for &ncell in &neighbors {
                f += fparc(icart, sys.latomfirst[ncell], sys);
            }

            icart_id = sys.latomnext[icart];
        }

        icell_id = sys.lcellnext[icell];
    }

    f
}

#[cfg(feature = "rayon")]
fn accumulate_pair_f_parallel(sys: &PackContext) -> (F, F) {
    sys.active_cells
        .par_iter()
        .map(|&icell| {
            let mut f: F = 0.0;
            let mut fdist_max: F = 0.0;
            let neighbors = sys.neighbor_cells_f[icell];

            let mut icart_id = sys.latomfirst[icell];
            while let Some(icart) = icart_id {
                let (f_same, fdist_same) = fparc_stats(icart, sys.latomnext[icart], sys);
                f += f_same;
                fdist_max = fdist_max.max(fdist_same);

                for &ncell in &neighbors {
                    let (f_neigh, fdist_neigh) = fparc_stats(icart, sys.latomfirst[ncell], sys);
                    f += f_neigh;
                    fdist_max = fdist_max.max(fdist_neigh);
                }

                icart_id = sys.latomnext[icart];
            }

            (f, fdist_max)
        })
        .reduce(
            || (0.0 as F, 0.0 as F),
            |(f1, d1), (f2, d2)| (f1 + f2, d1.max(d2)),
        )
}

#[inline(always)]
fn accumulate_pair_g(sys: &mut PackContext) {
    let mut icell_id = sys.lcellfirst;
    while let Some(icell) = icell_id {
        let neighbors = sys.neighbor_cells_g[icell];

        let mut icart_id = sys.latomfirst[icell];
        while let Some(icart) = icart_id {
            gparc(icart, sys.latomnext[icart], sys);
            for &ncell in &neighbors {
                gparc(icart, sys.latomfirst[ncell], sys);
            }

            icart_id = sys.latomnext[icart];
        }

        icell_id = sys.lcellnext[icell];
    }
}

#[inline(always)]
fn accumulate_pair_fg(sys: &mut PackContext) -> F {
    let mut f = 0.0;
    let mut icell_id = sys.lcellfirst;
    while let Some(icell) = icell_id {
        let neighbors = sys.neighbor_cells_g[icell];

        let mut icart_id = sys.latomfirst[icell];
        while let Some(icart) = icart_id {
            f += fgparc(icart, sys.latomnext[icart], sys);
            for &ncell in &neighbors {
                f += fgparc(icart, sys.latomfirst[ncell], sys);
            }

            icart_id = sys.latomnext[icart];
        }

        icell_id = sys.lcellnext[icell];
    }

    f
}

/// Atom-pair gradient accumulation into `sys.work.gxcar`.
/// Port of `gparc.f90`.
#[inline(always)]
fn gparc(icart: usize, first_jcart: Option<usize>, sys: &mut PackContext) {
    let mut jcart_id = first_jcart;
    while let Some(jcart) = jcart_id {
        if sys.ibmol[icart] == sys.ibmol[jcart] && sys.ibtype[icart] == sys.ibtype[jcart] {
            jcart_id = sys.latomnext[jcart];
            continue;
        }
        if sys.fixedatom[icart] && sys.fixedatom[jcart] {
            jcart_id = sys.latomnext[jcart];
            continue;
        }

        let rsum = sys.radius[icart] + sys.radius[jcart];
        let tol = rsum * rsum;
        let xi = sys.xcart[icart];
        let xj = sys.xcart[jcart];
        let mut dx = xi[0] - xj[0];
        let mut dy = xi[1] - xj[1];
        let mut dz = xi[2] - xj[2];
        if sys.pbc_length[0] > 0.0 {
            dx -= (dx / sys.pbc_length[0]).round() * sys.pbc_length[0];
        }
        if sys.pbc_length[1] > 0.0 {
            dy -= (dy / sys.pbc_length[1]).round() * sys.pbc_length[1];
        }
        if sys.pbc_length[2] > 0.0 {
            dz -= (dz / sys.pbc_length[2]).round() * sys.pbc_length[2];
        }
        let datom = dx * dx + dy * dy + dz * dz;

        if datom < tol {
            let dtemp = sys.fscale[icart] * sys.fscale[jcart] * 4.0 * (datom - tol);
            let xdiff0 = dtemp * dx;
            let xdiff1 = dtemp * dy;
            let xdiff2 = dtemp * dz;
            sys.work.gxcar[icart][0] += xdiff0;
            sys.work.gxcar[icart][1] += xdiff1;
            sys.work.gxcar[icart][2] += xdiff2;
            sys.work.gxcar[jcart][0] -= xdiff0;
            sys.work.gxcar[jcart][1] -= xdiff1;
            sys.work.gxcar[jcart][2] -= xdiff2;
            // Short radius gradient
            if sys.use_short_radius[icart] || sys.use_short_radius[jcart] {
                let short_rsum = sys.short_radius[icart] + sys.short_radius[jcart];
                let short_tol = short_rsum * short_rsum;
                if datom < short_tol {
                    let mut sr_scale =
                        (sys.short_radius_scale[icart] * sys.short_radius_scale[jcart]).sqrt();
                    sr_scale *= (tol * tol) / (short_tol * short_tol);
                    let dtemp2 = sys.fscale[icart]
                        * sys.fscale[jcart]
                        * 4.0
                        * sr_scale
                        * (datom - short_tol);
                    let xdiff0 = dtemp2 * dx;
                    let xdiff1 = dtemp2 * dy;
                    let xdiff2 = dtemp2 * dz;
                    sys.work.gxcar[icart][0] += xdiff0;
                    sys.work.gxcar[icart][1] += xdiff1;
                    sys.work.gxcar[icart][2] += xdiff2;
                    sys.work.gxcar[jcart][0] -= xdiff0;
                    sys.work.gxcar[jcart][1] -= xdiff1;
                    sys.work.gxcar[jcart][2] -= xdiff2;
                }
            }
        }

        jcart_id = sys.latomnext[jcart];
    }
}

/// Atom-pair function and gradient accumulation into `sys.work.gxcar`.
#[inline(always)]
fn fgparc(icart: usize, first_jcart: Option<usize>, sys: &mut PackContext) -> F {
    let mut result = 0.0;
    let mut jcart_id = first_jcart;
    let xi = sys.xcart[icart];
    let ri = sys.radius[icart];
    let ri_ini = sys.radius_ini[icart];
    let fsi = sys.fscale[icart];
    let move_flag = sys.move_flag;

    while let Some(jcart) = jcart_id {
        if sys.ibmol[icart] == sys.ibmol[jcart] && sys.ibtype[icart] == sys.ibtype[jcart] {
            jcart_id = sys.latomnext[jcart];
            continue;
        }
        if sys.fixedatom[icart] && sys.fixedatom[jcart] {
            jcart_id = sys.latomnext[jcart];
            continue;
        }

        let xj = sys.xcart[jcart];
        let mut dx = xi[0] - xj[0];
        let mut dy = xi[1] - xj[1];
        let mut dz = xi[2] - xj[2];
        if sys.pbc_length[0] > 0.0 {
            dx -= (dx / sys.pbc_length[0]).round() * sys.pbc_length[0];
        }
        if sys.pbc_length[1] > 0.0 {
            dy -= (dy / sys.pbc_length[1]).round() * sys.pbc_length[1];
        }
        if sys.pbc_length[2] > 0.0 {
            dz -= (dz / sys.pbc_length[2]).round() * sys.pbc_length[2];
        }
        let datom = dx * dx + dy * dy + dz * dz;
        let fsj = sys.fscale[jcart];
        let rsum = ri + sys.radius[jcart];
        let tol = rsum * rsum;

        if datom < tol {
            let penalty = datom - tol;
            let scale = fsi * fsj;
            result += scale * penalty * penalty;

            let dtemp = scale * 4.0 * penalty;
            let xdiff0 = dtemp * dx;
            let xdiff1 = dtemp * dy;
            let xdiff2 = dtemp * dz;
            sys.work.gxcar[icart][0] += xdiff0;
            sys.work.gxcar[icart][1] += xdiff1;
            sys.work.gxcar[icart][2] += xdiff2;
            sys.work.gxcar[jcart][0] -= xdiff0;
            sys.work.gxcar[jcart][1] -= xdiff1;
            sys.work.gxcar[jcart][2] -= xdiff2;

            if sys.use_short_radius[icart] || sys.use_short_radius[jcart] {
                let short_rsum = sys.short_radius[icart] + sys.short_radius[jcart];
                let short_tol = short_rsum * short_rsum;
                if datom < short_tol {
                    let short_penalty = datom - short_tol;
                    let mut sr_scale =
                        (sys.short_radius_scale[icart] * sys.short_radius_scale[jcart]).sqrt();
                    sr_scale *= (tol * tol) / (short_tol * short_tol);
                    let sr_pair_scale = scale * sr_scale;
                    result += sr_pair_scale * short_penalty * short_penalty;

                    let dtemp2 = sr_pair_scale * 4.0 * short_penalty;
                    let xdiff0 = dtemp2 * dx;
                    let xdiff1 = dtemp2 * dy;
                    let xdiff2 = dtemp2 * dz;
                    sys.work.gxcar[icart][0] += xdiff0;
                    sys.work.gxcar[icart][1] += xdiff1;
                    sys.work.gxcar[icart][2] += xdiff2;
                    sys.work.gxcar[jcart][0] -= xdiff0;
                    sys.work.gxcar[jcart][1] -= xdiff1;
                    sys.work.gxcar[jcart][2] -= xdiff2;
                }
            }
        }

        let rsum_ini = ri_ini + sys.radius_ini[jcart];
        let tol_ini = rsum_ini * rsum_ini;
        let violation = tol_ini - datom;
        if violation > sys.fdist {
            sys.fdist = violation;
        }
        if move_flag {
            if violation > sys.fdist_atom[icart] {
                sys.fdist_atom[icart] = violation;
            }
            if violation > sys.fdist_atom[jcart] {
                sys.fdist_atom[jcart] = violation;
            }
        }

        jcart_id = sys.latomnext[jcart];
    }

    result
}

#[cfg(feature = "rayon")]
#[inline(always)]
fn fparc_stats(icart: usize, first_jcart: Option<usize>, sys: &PackContext) -> (F, F) {
    let mut result: F = 0.0;
    let mut fdist_max: F = 0.0;
    let mut jcart_id = first_jcart;
    let xi = sys.xcart[icart];
    let ri = sys.radius[icart];
    let ri_ini = sys.radius_ini[icart];
    let fsi = sys.fscale[icart];

    while let Some(jcart) = jcart_id {
        if sys.ibmol[icart] == sys.ibmol[jcart] && sys.ibtype[icart] == sys.ibtype[jcart] {
            jcart_id = sys.latomnext[jcart];
            continue;
        }
        if sys.fixedatom[icart] && sys.fixedatom[jcart] {
            jcart_id = sys.latomnext[jcart];
            continue;
        }

        let xj = sys.xcart[jcart];
        let mut dx = xi[0] - xj[0];
        let mut dy = xi[1] - xj[1];
        let mut dz = xi[2] - xj[2];
        if sys.pbc_length[0] > 0.0 {
            dx -= (dx / sys.pbc_length[0]).round() * sys.pbc_length[0];
        }
        if sys.pbc_length[1] > 0.0 {
            dy -= (dy / sys.pbc_length[1]).round() * sys.pbc_length[1];
        }
        if sys.pbc_length[2] > 0.0 {
            dz -= (dz / sys.pbc_length[2]).round() * sys.pbc_length[2];
        }
        let datom = dx * dx + dy * dy + dz * dz;
        let rsum = ri + sys.radius[jcart];
        let tol = rsum * rsum;

        if datom < tol {
            let penalty = datom - tol;
            result += fsi * sys.fscale[jcart] * penalty * penalty;
            if sys.use_short_radius[icart] || sys.use_short_radius[jcart] {
                let short_rsum = sys.short_radius[icart] + sys.short_radius[jcart];
                let short_tol = short_rsum * short_rsum;
                if datom < short_tol {
                    let penalty = datom - short_tol;
                    let mut sr_scale =
                        (sys.short_radius_scale[icart] * sys.short_radius_scale[jcart]).sqrt();
                    sr_scale *= (tol * tol) / (short_tol * short_tol);
                    result += fsi * sys.fscale[jcart] * sr_scale * penalty * penalty;
                }
            }
        }

        let rsum_ini = ri_ini + sys.radius_ini[jcart];
        let tol_ini = rsum_ini * rsum_ini;
        let violation = tol_ini - datom;
        fdist_max = fdist_max.max(violation);

        jcart_id = sys.latomnext[jcart];
    }

    (result, fdist_max)
}

fn project_cartesian_gradient(x: &[F], sys: &mut PackContext, g: &mut [F]) {
    g.iter_mut().for_each(|v| *v = 0.0);

    let mut k1 = 0usize;
    let mut k2 = sys.ntotmol * 3;
    let mut icart = 0usize;

    for itype in 0..sys.ntype {
        if !sys.comptype[itype] {
            icart += sys.nmols[itype] * sys.natoms[itype];
            continue;
        }

        for _imol in 0..sys.nmols[itype] {
            let beta = x[k2];
            let gama = x[k2 + 1];
            let teta = x[k2 + 2];

            let (dv1beta, dv1gama, dv1teta, dv2beta, dv2gama, dv2teta, dv3beta, dv3gama, dv3teta) =
                eulerrmat_derivatives(beta, gama, teta);

            let idatom_base = sys.idfirst[itype];
            for iatom in 0..sys.natoms[itype] {
                let idatom = idatom_base + iatom;
                let cr = sys.coor[idatom];

                for k in 0..3 {
                    g[k1 + k] += sys.work.gxcar[icart][k];
                }

                for k in 0..3 {
                    g[k2] += (cr[0] * dv1beta[k] + cr[1] * dv2beta[k] + cr[2] * dv3beta[k])
                        * sys.work.gxcar[icart][k];
                    g[k2 + 1] += (cr[0] * dv1gama[k] + cr[1] * dv2gama[k] + cr[2] * dv3gama[k])
                        * sys.work.gxcar[icart][k];
                    g[k2 + 2] += (cr[0] * dv1teta[k] + cr[1] * dv2teta[k] + cr[2] * dv3teta[k])
                        * sys.work.gxcar[icart][k];
                }

                icart += 1;
            }

            k2 += 3;
            k1 += 3;
        }
    }
}

#[inline]
fn matches_cached_geometry(x: &[F], sys: &PackContext) -> bool {
    sys.work.matches_cached_geometry(
        x,
        &sys.comptype,
        sys.init1,
        sys.ncells,
        sys.cell_length,
        sys.pbc_min,
        sys.pbc_length,
    )
}

#[inline]
fn update_cached_geometry(x: &[F], sys: &mut PackContext) {
    sys.work.update_cached_geometry(
        x,
        &sys.comptype,
        sys.init1,
        sys.ncells,
        sys.cell_length,
        sys.pbc_min,
        sys.pbc_length,
    );
}
