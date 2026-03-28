//! movebad heuristic and flashsort.
//! Port of `heuristics.f90` and `flashsort.f90`.

use crate::constraints::EvalMode;
use crate::context::PackContext;
use crate::gencan::GencanWorkspace;
use crate::initial::restmol;
use crate::random::uniform01;
use molrs::types::F;
use rand::Rng;

pub struct MoveBadConfig<'a> {
    pub movefrac: F,
    pub maxmove_per_type: &'a [usize],
    pub movebadrandom: bool,
    pub gencan_maxit: usize,
}

/// Move the worst molecules to better positions.
/// Port of `movebad` from `heuristics.f90`.
pub fn movebad(
    x: &mut [F],
    sys: &mut PackContext,
    precision: F,
    cfg: &MoveBadConfig<'_>,
    rng: &mut impl Rng,
    workspace: &mut GencanWorkspace,
) {
    // Zero per-atom accumulators
    let ntotat = sys.ntotat;
    for icart in 0..ntotat {
        sys.fdist_atom[icart] = 0.0;
        sys.frest_atom[icart] = 0.0;
    }

    // Compute f with move flag to collect per-atom scores.
    // Packmol keeps radius=radius_ini during the whole movebad routine and restores
    // the working radii only at the end (heuristics.f90 lines 45-47 and 145-147).
    sys.move_flag = true;
    sys.work.radiuswork.copy_from_slice(&sys.radius);
    for i in 0..sys.ntotat {
        sys.radius[i] = sys.radius_ini[i];
    }
    sys.evaluate(x, EvalMode::FOnly, None);
    sys.move_flag = false;

    let ntotmol = sys.ntotmol;

    let mut icart_offset = 0usize;
    for itype in 0..sys.ntype {
        if !sys.comptype[itype] {
            icart_offset += sys.nmols[itype] * sys.natoms[itype];
            continue;
        }

        let nmols_itype = sys.nmols[itype];
        let natoms_itype = sys.natoms[itype];

        // Compute per-molecule violation score
        sys.work.fmol.clear();
        sys.work.fmol.resize(nmols_itype, 0.0);
        let fmol = &mut sys.work.fmol;
        let mut nbad = 0usize;
        let mut icart = icart_offset;
        for score in fmol.iter_mut().take(nmols_itype) {
            let mut fdist_mol = 0.0 as F;
            let mut frest_mol = 0.0 as F;
            for _ in 0..natoms_itype {
                fdist_mol = fdist_mol.max(sys.fdist_atom[icart]);
                frest_mol = frest_mol.max(sys.frest_atom[icart]);
                icart += 1;
            }
            if fdist_mol > precision || frest_mol > precision {
                nbad += 1;
                *score = fdist_mol + frest_mol;
            }
        }

        if nbad == 0 {
            icart_offset += nmols_itype * natoms_itype;
            continue;
        }

        let frac = (nbad as F / nmols_itype as F).min(cfg.movefrac);
        let nmove_base = (nmols_itype as F * frac) as isize;
        let nmove = usize::min(
            cfg.maxmove_per_type[itype],
            isize::max(nmove_base, 1) as usize,
        );

        // Sort molecules by violation (flash1 — O(N) histogram sort)
        let mflash = 1 + nmols_itype / 10;
        sys.work.flash_ind.clear();
        sys.work.flash_ind.extend(0..nmols_itype);
        flash1(fmol, mflash, &mut sys.work.flash_ind, &mut sys.work.flash_l);

        // Molecule offset in x for this type.
        // Matches Packmol heuristics.f90 lines 105-108:
        //   if(comptype(i)) imol = imol + nmols(i)  [only for i < itype]
        // Only ACTIVE types contribute to the offset.
        // In Phase 1 (compact x, one type active), all earlier types are inactive
        // so mol_base = 0 — the active type's molecules start at x[0].
        // In the main loop (full x, all types active), mol_base = sum(nmols[0..itype]).
        let mol_base: usize = {
            let mut base = 0usize;
            for it in 0..itype {
                if sys.comptype[it] {
                    base += sys.nmols[it];
                }
            }
            base
        };

        // Pre-collect (bad, good) molecule index pairs to avoid borrowing
        // sys.work.flash_ind across the restmol mutable borrow of sys.
        let move_pairs: Vec<(usize, usize)> = (0..nmove)
            .map(|k| {
                let ibad_mol = sys.work.flash_ind[nmols_itype - 1 - k];
                let igood_mol = sys.work.flash_ind
                    [(uniform01(rng) * nmols_itype as F * frac) as usize % nmols_itype.max(1)];
                (ibad_mol, igood_mol)
            })
            .collect();

        for &(ibad_mol, igood_mol) in &move_pairs {
            let ilubar_bad = (mol_base + ibad_mol) * 3;
            let ilugan_bad = ntotmol * 3 + (mol_base + ibad_mol) * 3;
            let ilubar_good = (mol_base + igood_mol) * 3;
            let ilugan_good = ntotmol * 3 + (mol_base + igood_mol) * 3;

            let dmax = sys.dmax[itype];
            if cfg.movebadrandom {
                x[ilubar_bad] = sys.sizemin[0] + uniform01(rng) * (sys.sizemax[0] - sys.sizemin[0]);
                x[ilubar_bad + 1] =
                    sys.sizemin[1] + uniform01(rng) * (sys.sizemax[1] - sys.sizemin[1]);
                x[ilubar_bad + 2] =
                    sys.sizemin[2] + uniform01(rng) * (sys.sizemax[2] - sys.sizemin[2]);
            } else {
                // Move bad molecule near good molecule with random perturbation.
                x[ilubar_bad] = x[ilubar_good] - 0.3 * dmax + 0.6 * uniform01(rng) * dmax;
                x[ilubar_bad + 1] = x[ilubar_good + 1] - 0.3 * dmax + 0.6 * uniform01(rng) * dmax;
                x[ilubar_bad + 2] = x[ilubar_good + 2] - 0.3 * dmax + 0.6 * uniform01(rng) * dmax;
            }

            // Copy angles from good molecule
            x[ilugan_bad] = x[ilugan_good];
            x[ilugan_bad + 1] = x[ilugan_good + 1];
            x[ilugan_bad + 2] = x[ilugan_good + 2];

            // Fit the moved molecule into constraints
            restmol(
                itype,
                ilubar_bad,
                x,
                sys,
                precision,
                cfg.gencan_maxit,
                true,
                workspace,
            );
        }

        icart_offset += nmols_itype * natoms_itype;
    }

    sys.evaluate(x, EvalMode::FOnly, None);
    sys.radius.copy_from_slice(&sys.work.radiuswork);
}

/// Flashsort (O(N) histogram sort) — sort array `a` ascending,
/// updating index array `ind` to track permutation.
/// Port of `flash1` from `flashsort.f90`.
///
/// `l_buf` is a reusable histogram buffer (resized as needed).
pub fn flash1(a: &mut [F], m: usize, ind: &mut [usize], l_buf: &mut Vec<usize>) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    debug_assert_eq!(ind.len(), n);

    let anmin = a.iter().copied().fold(F::INFINITY, F::min);
    let nmax_idx = a
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    if a[0] == a[nmax_idx] {
        return; // all equal
    }

    let c1 = (m as F - 1.0) / (a[nmax_idx] - anmin);
    l_buf.clear();
    l_buf.resize(m, 0);
    let l = l_buf;

    for ai in a.iter().take(n) {
        let k = (c1 * (*ai - anmin)) as usize;
        let k = k.min(m - 1);
        l[k] += 1;
    }
    for k in 1..m {
        l[k] += l[k - 1];
    }

    // Swap a[nmax] and a[0]
    a.swap(nmax_idx, 0);
    ind.swap(nmax_idx, 0);

    // Permutation phase
    let mut nmove = 0usize;
    let mut j = 0usize;
    let mut k = m;

    while nmove < n - 1 {
        while j >= l[k - 1] {
            j += 1;
            if j >= n {
                break;
            }
            k = (c1 * (a[j] - anmin)) as usize;
            k = k.min(m - 1) + 1;
        }
        if j >= n {
            break;
        }
        let mut flash = a[j];
        let mut iflash = ind[j];
        while j != l[k - 1] {
            k = (c1 * (flash - anmin)) as usize;
            k = k.min(m - 1);
            let lk = l[k] - 1;
            l[k] -= 1;
            let hold = a[lk];
            let ihold = ind[lk];
            a[lk] = flash;
            ind[lk] = iflash;
            flash = hold;
            iflash = ihold;
            nmove += 1;
            // Recompute k for new flash
            k = (c1 * (flash - anmin)) as usize;
            k = k.min(m - 1) + 1;
        }
    }

    // Insertion sort for cleanup (exact port of Fortran DO I=N-2,1,-1)
    let mut i = n as isize - 2;
    while i >= 1 {
        let iu = i as usize;
        if a[iu + 1] < a[iu] {
            let hold = a[iu];
            let ihold = ind[iu];
            let mut j = iu;
            while j + 1 < n && a[j + 1] < hold {
                a[j] = a[j + 1];
                ind[j] = ind[j + 1];
                j += 1;
            }
            a[j] = hold;
            ind[j] = ihold;
        }
        i -= 1;
    }
}
