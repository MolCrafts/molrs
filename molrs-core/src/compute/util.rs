use crate::Block;
use crate::Frame;
use crate::block::BlockDtype;
use crate::frame_access::FrameAccess;
use crate::region::simbox::SimBox;
use crate::types::F;
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

/// Compute MIC displacement vector from `from` to `to`.
///
/// Uses `SimBox::shortest_vector_fast` when periodic boundaries are present;
/// falls back to plain subtraction for free-boundary or no SimBox.
#[inline]
pub fn mic_disp(simbox: Option<&SimBox>, from: [F; 3], to: [F; 3]) -> [F; 3] {
    match simbox {
        Some(sb) if sb.pbc().iter().any(|&p| p) => {
            let a = array![from[0], from[1], from[2]];
            let b = array![to[0], to[1], to[2]];
            let dr = sb.shortest_vector_fast(a.view(), b.view());
            [dr[0], dr[1], dr[2]]
        }
        _ => [to[0] - from[0], to[1] - from[1], to[2] - from[2]],
    }
}

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

/// Owned position vectors: (x, y, z) each of length N.
///
/// Like [`get_positions`] but works with any [`FrameAccess`] implementor by
/// copying data into owned `Vec`s.
pub type PositionVecs = (Vec<F>, Vec<F>, Vec<F>);

/// Extract x, y, z positions from the "atoms" block of any [`FrameAccess`] type.
///
/// Returns owned `Vec`s rather than borrowed slices, so it works uniformly with
/// both `Frame` and `FrameView`.
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
