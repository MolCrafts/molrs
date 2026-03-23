//! Internal column representation for heterogeneous data.

use ndarray::ArrayD;

use super::dtype::DType;
use crate::types::{F, I, U};

/// Internal enum representing a column of data in a Block.
///
/// This type is exposed in the public API but users typically don't need to
/// interact with it directly. Instead, use the type-specific getters like
/// `get_float()`, `get_int()`, etc.
#[derive(Clone)]
pub enum Column {
    /// Floating point column using the compile-time scalar type [`F`].
    Float(ArrayD<F>),
    /// Signed integer column using the compile-time scalar type [`I`].
    Int(ArrayD<I>),
    /// Boolean column
    Bool(ArrayD<bool>),
    /// Unsigned integer column using the compile-time scalar type [`U`].
    UInt(ArrayD<U>),
    /// 8-bit unsigned integer column
    U8(ArrayD<u8>),
    /// String column
    String(ArrayD<String>),
}

impl Column {
    /// Returns the number of rows (axis-0 length) of this column.
    ///
    /// Returns `None` if the array has rank 0 (which should never happen
    /// in a valid Block, as rank-0 arrays are rejected during insertion).
    pub fn nrows(&self) -> Option<usize> {
        match self {
            Column::Float(a) => a.shape().first().copied(),
            Column::Int(a) => a.shape().first().copied(),
            Column::Bool(a) => a.shape().first().copied(),
            Column::UInt(a) => a.shape().first().copied(),
            Column::U8(a) => a.shape().first().copied(),
            Column::String(a) => a.shape().first().copied(),
        }
    }

    /// Returns the data type of this column.
    pub fn dtype(&self) -> DType {
        match self {
            Column::Float(_) => DType::Float,
            Column::Int(_) => DType::Int,
            Column::Bool(_) => DType::Bool,
            Column::UInt(_) => DType::UInt,
            Column::U8(_) => DType::U8,
            Column::String(_) => DType::String,
        }
    }

    /// Returns the shape of the underlying array.
    pub fn shape(&self) -> &[usize] {
        match self {
            Column::Float(a) => a.shape(),
            Column::Int(a) => a.shape(),
            Column::Bool(a) => a.shape(),
            Column::UInt(a) => a.shape(),
            Column::U8(a) => a.shape(),
            Column::String(a) => a.shape(),
        }
    }

    // Type-specific accessors for the compile-time float scalar.
    pub fn as_float(&self) -> Option<&ArrayD<F>> {
        match self {
            Column::Float(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_float_mut(&mut self) -> Option<&mut ArrayD<F>> {
        match self {
            Column::Float(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for the compile-time signed integer scalar.
    pub fn as_int(&self) -> Option<&ArrayD<I>> {
        match self {
            Column::Int(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_int_mut(&mut self) -> Option<&mut ArrayD<I>> {
        match self {
            Column::Int(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for bool
    pub fn as_bool(&self) -> Option<&ArrayD<bool>> {
        match self {
            Column::Bool(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_bool_mut(&mut self) -> Option<&mut ArrayD<bool>> {
        match self {
            Column::Bool(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for the compile-time unsigned integer scalar.
    pub fn as_uint(&self) -> Option<&ArrayD<U>> {
        match self {
            Column::UInt(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_uint_mut(&mut self) -> Option<&mut ArrayD<U>> {
        match self {
            Column::UInt(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for u8
    pub fn as_u8(&self) -> Option<&ArrayD<u8>> {
        match self {
            Column::U8(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_u8_mut(&mut self) -> Option<&mut ArrayD<u8>> {
        match self {
            Column::U8(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for String
    pub fn as_string(&self) -> Option<&ArrayD<String>> {
        match self {
            Column::String(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_string_mut(&mut self) -> Option<&mut ArrayD<String>> {
        match self {
            Column::String(a) => Some(a),
            _ => None,
        }
    }
}

impl std::fmt::Debug for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Column::Float(a) => write!(f, "Column::Float(shape={:?})", a.shape()),
            Column::Int(a) => write!(f, "Column::Int(shape={:?})", a.shape()),
            Column::Bool(a) => write!(f, "Column::Bool(shape={:?})", a.shape()),
            Column::UInt(a) => write!(f, "Column::UInt(shape={:?})", a.shape()),
            Column::U8(a) => write!(f, "Column::U8(shape={:?})", a.shape()),
            Column::String(a) => write!(f, "Column::String(shape={:?})", a.shape()),
        }
    }
}
