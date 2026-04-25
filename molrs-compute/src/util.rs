//! Internal helpers shared across compute kernels.
//!
//! Two hot-path optimizations live here:
//!
//! 1. [`MicHelper`] — hoisted minimum-image-convention state. Box kind and PBC
//!    flags are resolved **once per frame**; the per-atom [`disp`](MicHelper::disp)
//!    call is pure scalar arithmetic with no allocations or dynamic dispatch.
//! 2. [`get_positions_generic`] — returns views when the underlying columns are
//!    contiguous, falling back to owned vectors only for non-contiguous layouts
//!    (extremely rare). The [`PositionsRef`] enum lets callers dereference into
//!    `&[F]` without cloning.

use molrs::Block;
use molrs::Frame;
use molrs::block::BlockDtype;
use molrs::frame_access::FrameAccess;
use molrs::region::simbox::{BoxKind, SimBox};
use molrs::types::F;
use ndarray::array;

use super::error::ComputeError;

/// Extract a float column from a Block as a contiguous slice,
/// respecting compile-time precision (`F` = f32 or f64).
pub fn get_f_slice<'a>(
    block: &'a Block,
    block_name: &'static str,
    col_name: &'static str,
) -> Result<&'a [F], ComputeError> {
    let col = block.get(col_name).ok_or(ComputeError::MissingColumn {
        block: block_name,
        col: col_name,
    })?;
    let arr = <F as BlockDtype>::from_column(col).ok_or(ComputeError::MissingColumn {
        block: block_name,
        col: col_name,
    })?;
    arr.as_slice().ok_or(ComputeError::MissingColumn {
        block: block_name,
        col: col_name,
    })
}

// ---------------------------------------------------------------------------
// MIC helper — hoisted, allocation-free
// ---------------------------------------------------------------------------

/// Precomputed minimum-image-convention state.
///
/// Build one per frame with [`MicHelper::from_simbox`], then call
/// [`disp`](MicHelper::disp) in the per-atom loop. `disp` is `#[inline]` and
/// performs no allocations; `rustc` will typically inline it into the caller.
#[derive(Debug, Clone, Copy)]
pub enum MicHelper<'a> {
    /// Free boundaries or no box — plain Euclidean difference.
    Free,
    /// Orthorhombic cell; per-axis PBC mask decides which axes wrap.
    Ortho {
        len: [F; 3],
        inv_len: [F; 3],
        pbc: [bool; 3],
    },
    /// Triclinic fallback. Keeps a borrow on the `SimBox` so the hot path can
    /// delegate to [`SimBox::shortest_vector`] for the rare non-ortho case.
    Triclinic(&'a SimBox),
}

impl<'a> MicHelper<'a> {
    /// Extract the hot-loop state from an optional `SimBox`.
    #[inline]
    pub fn from_simbox(sb: Option<&'a SimBox>) -> Self {
        match sb {
            None => MicHelper::Free,
            Some(sb) => {
                let pbc = sb.pbc();
                if !pbc.iter().any(|&p| p) {
                    return MicHelper::Free;
                }
                match sb.kind() {
                    BoxKind::Ortho { len, inv_len } => MicHelper::Ortho {
                        len: [len[0], len[1], len[2]],
                        inv_len: [inv_len[0], inv_len[1], inv_len[2]],
                        pbc,
                    },
                    BoxKind::Triclinic => MicHelper::Triclinic(sb),
                }
            }
        }
    }

    /// MIC displacement vector `to − from`.
    #[inline(always)]
    pub fn disp(&self, from: [F; 3], to: [F; 3]) -> [F; 3] {
        match *self {
            MicHelper::Free => [to[0] - from[0], to[1] - from[1], to[2] - from[2]],
            MicHelper::Ortho { len, inv_len, pbc } => {
                let mut dr = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
                // Unrolled manually — the compiler tends to leave this as a
                // branchy loop otherwise.
                if pbc[0] {
                    dr[0] -= (dr[0] * inv_len[0]).round() * len[0];
                }
                if pbc[1] {
                    dr[1] -= (dr[1] * inv_len[1]).round() * len[1];
                }
                if pbc[2] {
                    dr[2] -= (dr[2] * inv_len[2]).round() * len[2];
                }
                dr
            }
            MicHelper::Triclinic(sb) => {
                let a = array![from[0], from[1], from[2]];
                let b = array![to[0], to[1], to[2]];
                let dr = sb.shortest_vector(a.view(), b.view());
                [dr[0], dr[1], dr[2]]
            }
        }
    }
}

/// Compute MIC displacement vector from `from` to `to`.
///
/// **Prefer [`MicHelper::from_simbox`] + [`MicHelper::disp`] in hot loops** —
/// this free function resolves the box kind on every call. Kept for
/// convenience at cold call sites.
#[inline]
pub fn mic_disp(simbox: Option<&SimBox>, from: [F; 3], to: [F; 3]) -> [F; 3] {
    MicHelper::from_simbox(simbox).disp(from, to)
}

// ---------------------------------------------------------------------------
// Position access — borrow when contiguous, copy only if forced
// ---------------------------------------------------------------------------

/// Positional slices: (x, y, z) each of length N.
pub type PositionSlices<'a> = (&'a [F], &'a [F], &'a [F]);

/// Extract x, y, z slices from the "atoms" block of a Frame.
pub fn get_positions(frame: &Frame) -> Result<PositionSlices<'_>, ComputeError> {
    let atoms = frame
        .get("atoms")
        .ok_or(ComputeError::MissingBlock { name: "atoms" })?;
    let xs = get_f_slice(atoms, "atoms", "x")?;
    let ys = get_f_slice(atoms, "atoms", "y")?;
    let zs = get_f_slice(atoms, "atoms", "z")?;
    Ok((xs, ys, zs))
}

/// Position storage: either a borrow from a contiguous column (zero copy) or
/// an owned `Vec` (required when the view is non-contiguous). Callers reach
/// the raw data via `AsRef<[F]>` / [`Positions::slice`].
#[derive(Debug)]
pub enum Positions<'a> {
    Borrowed(&'a [F]),
    Owned(Vec<F>),
}

impl<'a> Positions<'a> {
    #[inline]
    pub fn slice(&self) -> &[F] {
        match self {
            Positions::Borrowed(s) => s,
            Positions::Owned(v) => v.as_slice(),
        }
    }
}

fn column_to_positions<'a, FA: FrameAccess>(
    frame: &'a FA,
    col: &'static str,
) -> Result<Positions<'a>, ComputeError> {
    let view = frame
        .get_float("atoms", col)
        .ok_or(ComputeError::MissingColumn {
            block: "atoms",
            col,
        })?;
    match view.as_slice() {
        Some(s) => {
            // SAFETY: the slice refers to the same buffer backing the view,
            // which lives for 'a (tied to `&'a frame`). The `as_slice()`
            // return's own lifetime is a local reborrow; extending it is
            // sound because ArrayView in `ViewRepr<&'a A>` owns its data
            // pointer for 'a.
            let ptr = s.as_ptr();
            let len = s.len();
            let slice: &'a [F] = unsafe { std::slice::from_raw_parts(ptr, len) };
            Ok(Positions::Borrowed(slice))
        }
        None => Ok(Positions::Owned(view.iter().copied().collect())),
    }
}

/// Owned position vectors: (x, y, z) each of length N.
pub type PositionVecs = (Vec<F>, Vec<F>, Vec<F>);

/// Extract x, y, z positions from the "atoms" block of any [`FrameAccess`] type.
///
/// **Legacy path** — always materializes owned `Vec`s. Prefer
/// [`get_positions_ref`] in hot loops; it avoids the three 8N-byte copies
/// when the underlying columns are contiguous (the common case for both
/// `Frame` and `FrameView`).
pub fn get_positions_generic(frame: &impl FrameAccess) -> Result<PositionVecs, ComputeError> {
    let xs = frame
        .get_float("atoms", "x")
        .ok_or(ComputeError::MissingColumn {
            block: "atoms",
            col: "x",
        })?;
    let ys = frame
        .get_float("atoms", "y")
        .ok_or(ComputeError::MissingColumn {
            block: "atoms",
            col: "y",
        })?;
    let zs = frame
        .get_float("atoms", "z")
        .ok_or(ComputeError::MissingColumn {
            block: "atoms",
            col: "z",
        })?;
    Ok((
        xs.iter().copied().collect(),
        ys.iter().copied().collect(),
        zs.iter().copied().collect(),
    ))
}

/// Zero-copy position triplet when the frame exposes contiguous columns.
///
/// Returns a triple of [`Positions`]. Each can be dereferenced to `&[F]` via
/// [`slice`](Positions::slice). For `Frame` and `FrameView` the inner
/// variant is always [`Positions::Borrowed`], so the per-atom loop sees the
/// same cache-friendly layout as the stored block.
pub fn get_positions_ref<'a, FA: FrameAccess>(
    frame: &'a FA,
) -> Result<(Positions<'a>, Positions<'a>, Positions<'a>), ComputeError> {
    let xs = column_to_positions(frame, "x")?;
    let ys = column_to_positions(frame, "y")?;
    let zs = column_to_positions(frame, "z")?;
    Ok((xs, ys, zs))
}
