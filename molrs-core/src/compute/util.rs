use crate::Block;
use crate::block::BlockDtype;
use crate::types::F;

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
