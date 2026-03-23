//! `extern "C"` functions for SimBox (simulation cell) operations.
//!
//! A **SimBox** defines a periodic simulation cell via a 3x3 cell matrix
//! **H** (columns are lattice vectors), an origin point, and per-axis
//! periodic boundary condition (PBC) flags.
//!
//! All lengths are in Angstrom.  The cell matrix is stored and
//! transmitted in **row-major** order as a flat array of 9 floats.
//!
//! # Coordinate conventions
//!
//! * Cell matrix **H**: `h[i][j]` -- component `j` of lattice vector `i`.
//! * Origin: lower corner of the simulation box in Angstrom.
//! * PBC flags: `[px, py, pz]` -- `true` if periodic along that axis.
//! * Tilts: off-diagonal elements `[xy, xz, yz]` in Angstrom.

use ndarray::{Array1, Array2, ArrayView2, array};

use molrs::region::simbox::SimBox;

use crate::F;
use crate::error::{self, MolrsStatus};
use crate::handle::{MolrsSimBoxHandle, handle_to_simbox_key, simbox_key_to_handle};
use crate::store::lock_store;
use crate::{ffi_try, null_check};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Create a SimBox from a general 3x3 cell matrix.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_new(const molrs_float_t h9[9],
///                               const molrs_float_t origin3[3],
///                               const bool pbc3[3],
///                               MolrsSimBoxHandle* out);
/// ```
///
/// # Arguments
///
/// * `h9` -- 3x3 cell matrix in row-major order (9 floats) in Angstrom.
///   `h9[0..3]` is lattice vector **a**, `h9[3..6]` is **b**, `h9[6..9]` is **c**.
/// * `origin3` -- Box origin `[ox, oy, oz]` in Angstrom (3 floats).
/// * `pbc3` -- Per-axis periodic boundary flags `[px, py, pz]` (3 bools).
/// * `out` -- On success, receives the new SimBox handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if any pointer is null.
/// * `MolrsStatus::SingularCell` if the cell matrix is singular
///   (determinant = 0).
///
/// # Safety
///
/// * `h9` must point to at least 9 readable `molrs_float_t` values.
/// * `origin3` must point to at least 3 readable `molrs_float_t` values.
/// * `pbc3` must point to at least 3 readable `bool` values.
/// * `out` must point to a writable `MolrsSimBoxHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_new(
    h9: *const F,
    origin3: *const F,
    pbc3: *const bool,
    out: *mut MolrsSimBoxHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(h9);
        null_check!(origin3);
        null_check!(pbc3);
        null_check!(out);

        let h_slice = unsafe { std::slice::from_raw_parts(h9, 9) };
        let h = match Array2::from_shape_vec((3, 3), h_slice.to_vec()) {
            Ok(m) => m,
            Err(e) => {
                error::set_last_error(format!("invalid h matrix: {e}"));
                return MolrsStatus::InvalidArgument;
            }
        };
        let o_slice = unsafe { std::slice::from_raw_parts(origin3, 3) };
        let origin = Array1::from_vec(o_slice.to_vec());
        let pbc_slice = unsafe { std::slice::from_raw_parts(pbc3, 3) };
        let pbc = [pbc_slice[0], pbc_slice[1], pbc_slice[2]];

        let sb = match SimBox::new(h, origin, pbc) {
            Ok(sb) => sb,
            Err(_) => {
                error::set_last_error("singular cell matrix");
                return MolrsStatus::SingularCell;
            }
        };
        let mut store = lock_store();
        let key = store.simboxes.insert(sb);
        unsafe { *out = simbox_key_to_handle(key) };
        MolrsStatus::Ok
    })
}

/// Create a cubic SimBox with edge length `a`.
///
/// Equivalent to calling [`molrs_simbox_new`] with a diagonal cell
/// matrix `diag(a, a, a)`.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_cube(molrs_float_t a,
///                                const molrs_float_t origin3[3],
///                                const bool pbc3[3],
///                                MolrsSimBoxHandle* out);
/// ```
///
/// # Arguments
///
/// * `a` -- Cube edge length in Angstrom.
/// * `origin3` -- Box origin `[ox, oy, oz]` in Angstrom.
/// * `pbc3` -- Per-axis PBC flags.
/// * `out` -- On success, receives the new SimBox handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if any pointer is null.
/// * `MolrsStatus::InvalidArgument` if `a` is zero or negative.
///
/// # Safety
///
/// * `origin3` must point to 3 readable floats.
/// * `pbc3` must point to 3 readable bools.
/// * `out` must point to a writable `MolrsSimBoxHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_cube(
    a: F,
    origin3: *const F,
    pbc3: *const bool,
    out: *mut MolrsSimBoxHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(origin3);
        null_check!(pbc3);
        null_check!(out);

        let o_slice = unsafe { std::slice::from_raw_parts(origin3, 3) };
        let origin = Array1::from_vec(o_slice.to_vec());
        let pbc_slice = unsafe { std::slice::from_raw_parts(pbc3, 3) };
        let pbc = [pbc_slice[0], pbc_slice[1], pbc_slice[2]];

        let sb = match SimBox::cube(a, origin, pbc) {
            Ok(sb) => sb,
            Err(_) => {
                error::set_last_error("invalid cube parameters");
                return MolrsStatus::InvalidArgument;
            }
        };
        let mut store = lock_store();
        let key = store.simboxes.insert(sb);
        unsafe { *out = simbox_key_to_handle(key) };
        MolrsStatus::Ok
    })
}

/// Create an orthorhombic (rectangular) SimBox from axis lengths.
///
/// Equivalent to calling [`molrs_simbox_new`] with a diagonal cell
/// matrix `diag(lx, ly, lz)`.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_ortho(const molrs_float_t lengths3[3],
///                                 const molrs_float_t origin3[3],
///                                 const bool pbc3[3],
///                                 MolrsSimBoxHandle* out);
/// ```
///
/// # Arguments
///
/// * `lengths3` -- Box side lengths `[lx, ly, lz]` in Angstrom.
/// * `origin3` -- Box origin `[ox, oy, oz]` in Angstrom.
/// * `pbc3` -- Per-axis PBC flags.
/// * `out` -- On success, receives the new SimBox handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if any pointer is null.
/// * `MolrsStatus::InvalidArgument` if any length is zero or negative.
///
/// # Safety
///
/// * `lengths3`, `origin3` must each point to 3 readable floats.
/// * `pbc3` must point to 3 readable bools.
/// * `out` must point to a writable `MolrsSimBoxHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_ortho(
    lengths3: *const F,
    origin3: *const F,
    pbc3: *const bool,
    out: *mut MolrsSimBoxHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(lengths3);
        null_check!(origin3);
        null_check!(pbc3);
        null_check!(out);

        let l_slice = unsafe { std::slice::from_raw_parts(lengths3, 3) };
        let lengths = Array1::from_vec(l_slice.to_vec());
        let o_slice = unsafe { std::slice::from_raw_parts(origin3, 3) };
        let origin = Array1::from_vec(o_slice.to_vec());
        let pbc_slice = unsafe { std::slice::from_raw_parts(pbc3, 3) };
        let pbc = [pbc_slice[0], pbc_slice[1], pbc_slice[2]];

        let sb = match SimBox::ortho(lengths, origin, pbc) {
            Ok(sb) => sb,
            Err(_) => {
                error::set_last_error("invalid ortho parameters");
                return MolrsStatus::InvalidArgument;
            }
        };
        let mut store = lock_store();
        let key = store.simboxes.insert(sb);
        unsafe { *out = simbox_key_to_handle(key) };
        MolrsStatus::Ok
    })
}

/// Destroy a SimBox and invalidate its handle.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_drop(MolrsSimBoxHandle handle);
/// ```
///
/// # Arguments
///
/// * `handle` -- The SimBox to destroy.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::InvalidSimBoxHandle` if `handle` is stale.
///
/// # Safety
///
/// The caller must not use `handle` after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_drop(handle: MolrsSimBoxHandle) -> MolrsStatus {
    ffi_try!({
        let mut store = lock_store();
        let key = handle_to_simbox_key(handle);
        match store.simboxes.remove(key) {
            Some(_) => MolrsStatus::Ok,
            None => {
                error::set_last_error("invalid simbox handle");
                MolrsStatus::InvalidSimBoxHandle
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// Helper: get a reference to a SimBox by handle, returning error if invalid.
macro_rules! get_simbox {
    ($store:expr, $handle:expr) => {
        match $store.simboxes.get(handle_to_simbox_key($handle)) {
            Some(sb) => sb,
            None => {
                error::set_last_error("invalid simbox handle");
                return MolrsStatus::InvalidSimBoxHandle;
            }
        }
    };
}

/// Copy the 3x3 cell matrix into a caller buffer (row-major, 9 floats).
///
/// The values are in Angstrom.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_h(MolrsSimBoxHandle h, molrs_float_t out9[9]);
/// ```
///
/// # Arguments
///
/// * `h` -- SimBox handle.
/// * `out9` -- Buffer of at least 9 floats; receives the cell matrix.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out9` is null.
/// * `MolrsStatus::InvalidSimBoxHandle` if `h` is stale.
///
/// # Safety
///
/// * `h` must be a live SimBox handle.
/// * `out9` must point to at least 9 writable `molrs_float_t` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_h(h: MolrsSimBoxHandle, out9: *mut F) -> MolrsStatus {
    ffi_try!({
        null_check!(out9);
        let store = lock_store();
        let sb = get_simbox!(store, h);
        let hv = sb.h_view();
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out9, 9) };
        for (i, &val) in hv.iter().enumerate() {
            out_slice[i] = val;
        }
        MolrsStatus::Ok
    })
}

/// Copy the box origin into a caller buffer (3 floats, in Angstrom).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_origin(MolrsSimBoxHandle h,
///                                  molrs_float_t out3[3]);
/// ```
///
/// # Arguments
///
/// * `h` -- SimBox handle.
/// * `out3` -- Buffer of at least 3 floats; receives `[ox, oy, oz]`.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out3` is null.
/// * `MolrsStatus::InvalidSimBoxHandle` if `h` is stale.
///
/// # Safety
///
/// * `h` must be a live SimBox handle.
/// * `out3` must point to at least 3 writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_origin(h: MolrsSimBoxHandle, out3: *mut F) -> MolrsStatus {
    ffi_try!({
        null_check!(out3);
        let store = lock_store();
        let sb = get_simbox!(store, h);
        let ov = sb.origin_view();
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out3, 3) };
        for (i, &val) in ov.iter().enumerate() {
            out_slice[i] = val;
        }
        MolrsStatus::Ok
    })
}

/// Copy the per-axis periodic boundary condition flags (3 bools).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_pbc(MolrsSimBoxHandle h, bool out3[3]);
/// ```
///
/// # Arguments
///
/// * `h` -- SimBox handle.
/// * `out3` -- Buffer of at least 3 bools; receives `[px, py, pz]`.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out3` is null.
/// * `MolrsStatus::InvalidSimBoxHandle` if `h` is stale.
///
/// # Safety
///
/// * `h` must be a live SimBox handle.
/// * `out3` must point to at least 3 writable `bool` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_pbc(h: MolrsSimBoxHandle, out3: *mut bool) -> MolrsStatus {
    ffi_try!({
        null_check!(out3);
        let store = lock_store();
        let sb = get_simbox!(store, h);
        let pbc = sb.pbc_view();
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out3, 3) };
        for (i, &val) in pbc.iter().enumerate() {
            out_slice[i] = val;
        }
        MolrsStatus::Ok
    })
}

/// Compute the cell volume in Angstrom^3.
///
/// The volume is `|det(H)|` where H is the 3x3 cell matrix.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_volume(MolrsSimBoxHandle h,
///                                  molrs_float_t* out);
/// ```
///
/// # Arguments
///
/// * `h` -- SimBox handle.
/// * `out` -- On success, receives the volume in Angstrom^3.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::InvalidSimBoxHandle` if `h` is stale.
///
/// # Safety
///
/// * `h` must be a live SimBox handle.
/// * `out` must point to a writable float.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_volume(h: MolrsSimBoxHandle, out: *mut F) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let store = lock_store();
        let sb = get_simbox!(store, h);
        unsafe { *out = sb.volume() };
        MolrsStatus::Ok
    })
}

/// Get lattice vector lengths `[|a|, |b|, |c|]` in Angstrom.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_lengths(MolrsSimBoxHandle h,
///                                   molrs_float_t out3[3]);
/// ```
///
/// # Arguments
///
/// * `h` -- SimBox handle.
/// * `out3` -- Buffer of at least 3 floats; receives the lengths.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out3` is null.
/// * `MolrsStatus::InvalidSimBoxHandle` if `h` is stale.
///
/// # Safety
///
/// * `h` must be a live SimBox handle.
/// * `out3` must point to at least 3 writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_lengths(h: MolrsSimBoxHandle, out3: *mut F) -> MolrsStatus {
    ffi_try!({
        null_check!(out3);
        let store = lock_store();
        let sb = get_simbox!(store, h);
        let lengths = sb.lengths();
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out3, 3) };
        for (i, &val) in lengths.iter().enumerate() {
            out_slice[i] = val;
        }
        MolrsStatus::Ok
    })
}

/// Get off-diagonal tilt factors `[xy, xz, yz]` in Angstrom.
///
/// For an orthorhombic box all tilts are zero.  Non-zero tilts indicate
/// a triclinic cell (e.g. for crystal simulations).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_tilts(MolrsSimBoxHandle h,
///                                 molrs_float_t out3[3]);
/// ```
///
/// # Arguments
///
/// * `h` -- SimBox handle.
/// * `out3` -- Buffer of at least 3 floats; receives `[xy, xz, yz]`.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out3` is null.
/// * `MolrsStatus::InvalidSimBoxHandle` if `h` is stale.
///
/// # Safety
///
/// * `h` must be a live SimBox handle.
/// * `out3` must point to at least 3 writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_tilts(h: MolrsSimBoxHandle, out3: *mut F) -> MolrsStatus {
    ffi_try!({
        null_check!(out3);
        let store = lock_store();
        let sb = get_simbox!(store, h);
        let tilts = sb.tilts();
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out3, 3) };
        for (i, &val) in tilts.iter().enumerate() {
            out_slice[i] = val;
        }
        MolrsStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Batch coordinate operations
// ---------------------------------------------------------------------------

/// Wrap Cartesian coordinates into the unit cell (minimum-image).
///
/// Atoms outside the periodic box are folded back in.  Non-periodic
/// axes are unaffected.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_wrap(MolrsSimBoxHandle h,
///                                const molrs_float_t* xyz_in,
///                                molrs_float_t* xyz_out,
///                                size_t n_atoms);
/// ```
///
/// # Arguments
///
/// * `h` -- SimBox handle.
/// * `xyz_in` -- Input coordinates, flat `[x0,y0,z0, x1,y1,z1, ...]`
///   (`n_atoms * 3` floats, in Angstrom).
/// * `xyz_out` -- Output buffer (`n_atoms * 3` floats).  May alias
///   `xyz_in` for in-place wrapping.
/// * `n_atoms` -- Number of atoms.  If 0, the function is a no-op.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `xyz_in` or `xyz_out` is null.
/// * `MolrsStatus::InvalidSimBoxHandle` if `h` is stale.
/// * `MolrsStatus::InvalidArgument` if the input array shape is bad.
///
/// # Safety
///
/// * `h` must be a live SimBox handle.
/// * `xyz_in` must point to at least `n_atoms * 3` readable floats.
/// * `xyz_out` must point to at least `n_atoms * 3` writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_wrap(
    h: MolrsSimBoxHandle,
    xyz_in: *const F,
    xyz_out: *mut F,
    n_atoms: usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(xyz_in);
        null_check!(xyz_out);
        if n_atoms == 0 {
            return MolrsStatus::Ok;
        }
        let store = lock_store();
        let sb = get_simbox!(store, h);

        let in_slice = unsafe { std::slice::from_raw_parts(xyz_in, n_atoms * 3) };
        let in_arr = match ArrayView2::from_shape((n_atoms, 3), in_slice) {
            Ok(a) => a,
            Err(e) => {
                error::set_last_error(format!("invalid input array: {e}"));
                return MolrsStatus::InvalidArgument;
            }
        };
        let wrapped = sb.wrap(in_arr);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(xyz_out, n_atoms * 3) };
        let wrapped_slice = wrapped
            .as_slice_memory_order()
            .expect("wrapped result is contiguous");
        out_slice.copy_from_slice(wrapped_slice);
        MolrsStatus::Ok
    })
}

/// Compute minimum-image displacement vectors for `n_pairs` atom pairs.
///
/// For each pair `(r1[i], r2[i])`, computes the shortest displacement
/// vector `dr = r2 - r1` under periodic boundary conditions.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_simbox_shortest_vector(
///     MolrsSimBoxHandle h,
///     const molrs_float_t* r1,
///     const molrs_float_t* r2,
///     molrs_float_t* dr_out,
///     size_t n_pairs);
/// ```
///
/// # Arguments
///
/// * `h` -- SimBox handle.
/// * `r1` -- First set of positions, flat `[x0,y0,z0, ...]`
///   (`n_pairs * 3` floats, in Angstrom).
/// * `r2` -- Second set of positions (`n_pairs * 3` floats, in Angstrom).
/// * `dr_out` -- Output displacements (`n_pairs * 3` floats, in Angstrom).
/// * `n_pairs` -- Number of pairs.  If 0, the function is a no-op.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if any pointer is null.
/// * `MolrsStatus::InvalidSimBoxHandle` if `h` is stale.
///
/// # Safety
///
/// * `h` must be a live SimBox handle.
/// * `r1`, `r2` must each point to at least `n_pairs * 3` readable floats.
/// * `dr_out` must point to at least `n_pairs * 3` writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_simbox_shortest_vector(
    h: MolrsSimBoxHandle,
    r1: *const F,
    r2: *const F,
    dr_out: *mut F,
    n_pairs: usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(r1);
        null_check!(r2);
        null_check!(dr_out);
        if n_pairs == 0 {
            return MolrsStatus::Ok;
        }
        let store = lock_store();
        let sb = get_simbox!(store, h);

        let r1_slice = unsafe { std::slice::from_raw_parts(r1, n_pairs * 3) };
        let r2_slice = unsafe { std::slice::from_raw_parts(r2, n_pairs * 3) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(dr_out, n_pairs * 3) };

        for i in 0..n_pairs {
            let a = array![r1_slice[i * 3], r1_slice[i * 3 + 1], r1_slice[i * 3 + 2]];
            let b = array![r2_slice[i * 3], r2_slice[i * 3 + 1], r2_slice[i * 3 + 2]];
            let dr = sb.shortest_vector(a.view(), b.view());
            out_slice[i * 3] = dr[0];
            out_slice[i * 3 + 1] = dr[1];
            out_slice[i * 3 + 2] = dr[2];
        }
        MolrsStatus::Ok
    })
}
