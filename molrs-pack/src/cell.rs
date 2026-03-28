//! 3D linked cell list implementation.
//! Exact port of `cell_indexing.f90`, `resetcells.f90` and `pbc.f90` from Packmol.

use molrs::types::F;
/// Wrap coordinate into periodic box, returning value in [pbc_min, pbc_min + pbc_length).
#[inline]
fn in_box(x: F, pbc_min: F, pbc_length: F) -> F {
    let mut v = x - pbc_min;
    v -= (v / pbc_length).floor() * pbc_length;
    v + pbc_min
}

/// Compute cell index (0-based) for a given position along one axis.
/// Port of Fortran `setcell`.
#[inline]
pub fn axis_cell(x: F, pbc_min: F, pbc_length: F, cell_length: F, ncells: usize) -> usize {
    let xt = in_box(x, pbc_min, pbc_length);
    let idx = ((xt - pbc_min) / cell_length).floor() as usize;
    idx.min(ncells - 1)
}

/// Compute the 3D cell indices (0-based) for a position.
#[inline(always)]
pub fn setcell(
    pos: &[F; 3],
    pbc_min: &[F; 3],
    pbc_length: &[F; 3],
    cell_length: &[F; 3],
    ncells: &[usize; 3],
) -> [usize; 3] {
    [
        axis_cell(pos[0], pbc_min[0], pbc_length[0], cell_length[0], ncells[0]),
        axis_cell(pos[1], pbc_min[1], pbc_length[1], cell_length[1], ncells[1]),
        axis_cell(pos[2], pbc_min[2], pbc_length[2], cell_length[2], ncells[2]),
    ]
}

/// Convert a 3D cell index (0-based) to a flat index.
/// Port of `index_cell` (Fortran 1-based → Rust 0-based).
#[inline]
pub fn index_cell(cell: &[usize; 3], ncells: &[usize; 3]) -> usize {
    cell[0] * ncells[1] * ncells[2] + cell[1] * ncells[2] + cell[2]
}

/// Convert a flat index to a 3D cell index (0-based).
/// Port of `icell_to_cell` (converted to 0-based).
pub fn icell_to_cell(icell: usize, ncells: &[usize; 3]) -> [usize; 3] {
    let k = icell % ncells[2];
    let rem = icell / ncells[2];
    let j = rem % ncells[1];
    let i = rem / ncells[1];
    [i, j, k]
}

/// Wrap a cell index with periodic boundary conditions (0-based).
/// Port of Fortran `cell_ind` (which uses 1-based, here we use 0-based).
#[inline]
pub fn cell_ind(idx: isize, ncells: usize) -> usize {
    idx.rem_euclid(ncells as isize) as usize
}

/// Compute PBC-corrected difference vector.
/// Port of `delta_vector` from `pbc.f90`.
#[inline]
pub fn delta_vector(xi: &[F; 3], xj: &[F; 3], pbc_length: &[F; 3]) -> [F; 3] {
    let mut d = [xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2]];
    for k in 0..3 {
        if pbc_length[k] > 0.0 {
            d[k] -= (d[k] / pbc_length[k]).round() * pbc_length[k];
        }
    }
    d
}
