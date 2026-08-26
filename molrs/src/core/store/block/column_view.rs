//! Zero-copy borrowed view of a [`Column`].
//!
//! `ColumnView<'a>` borrows the underlying ndarray data from a [`Column`] without
//! copying, providing read-only access with the same API surface as `Column`.

use ndarray::ArrayViewD;
use num_complex::Complex;

use super::column::Column;
use super::dtype::DType;
use crate::types::{F, I, Idx};

macro_rules! map_view {
    ($view:expr, $arr:ident => $body:expr) => {
        match $view {
            ColumnView::Float16($arr) => $body,
            ColumnView::Float32($arr) => $body,
            ColumnView::Float($arr) => $body,
            ColumnView::Int8($arr) => $body,
            ColumnView::Int16($arr) => $body,
            ColumnView::Int($arr) => $body,
            ColumnView::Int64($arr) => $body,
            ColumnView::U8($arr) => $body,
            ColumnView::UInt16($arr) => $body,
            ColumnView::UInt32($arr) => $body,
            ColumnView::UInt($arr) => $body,
            ColumnView::Bool($arr) => $body,
            ColumnView::String($arr) => $body,
            ColumnView::Complex64($arr) => $body,
            ColumnView::Complex128($arr) => $body,
        }
    };
}

/// A borrowed, read-only view of a [`Column`].
///
/// Each variant holds an `ArrayViewD` that borrows from the corresponding
/// `ArrayD` inside an owned `Column`. No data is copied.
pub enum ColumnView<'a> {
    /// Borrowed `f16` column.
    Float16(ArrayViewD<'a, half::f16>),
    /// Borrowed `f32` column.
    Float32(ArrayViewD<'a, f32>),
    /// Borrowed float column.
    Float(ArrayViewD<'a, F>),
    /// Borrowed `i8` column.
    Int8(ArrayViewD<'a, i8>),
    /// Borrowed `i16` column.
    Int16(ArrayViewD<'a, i16>),
    /// Borrowed signed integer column.
    Int(ArrayViewD<'a, I>),
    /// Borrowed `i64` column.
    Int64(ArrayViewD<'a, i64>),
    /// Borrowed boolean column.
    Bool(ArrayViewD<'a, bool>),
    /// Borrowed unsigned integer column.
    UInt(ArrayViewD<'a, Idx>),
    /// Borrowed u8 column.
    U8(ArrayViewD<'a, u8>),
    /// Borrowed `u16` column.
    UInt16(ArrayViewD<'a, u16>),
    /// Borrowed `u32` column.
    UInt32(ArrayViewD<'a, u32>),
    /// Borrowed string column.
    String(ArrayViewD<'a, String>),
    /// Borrowed `complex64` column.
    Complex64(ArrayViewD<'a, Complex<f32>>),
    /// Borrowed `complex128` column.
    Complex128(ArrayViewD<'a, Complex<f64>>),
}

impl<'a> ColumnView<'a> {
    /// Returns the number of rows (axis-0 length) of this column view.
    ///
    /// Returns `None` if the array has rank 0.
    pub fn nrows(&self) -> Option<usize> {
        map_view!(self, a => a.shape().first().copied())
    }

    /// Returns the data type of this column view.
    pub fn dtype(&self) -> DType {
        match self {
            ColumnView::Float16(_) => DType::Float16,
            ColumnView::Float32(_) => DType::Float32,
            ColumnView::Float(_) => DType::Float,
            ColumnView::Int8(_) => DType::Int8,
            ColumnView::Int16(_) => DType::Int16,
            ColumnView::Int(_) => DType::Int,
            ColumnView::Int64(_) => DType::Int64,
            ColumnView::Bool(_) => DType::Bool,
            ColumnView::UInt(_) => DType::UInt,
            ColumnView::U8(_) => DType::U8,
            ColumnView::UInt16(_) => DType::UInt16,
            ColumnView::UInt32(_) => DType::UInt32,
            ColumnView::String(_) => DType::String,
            ColumnView::Complex64(_) => DType::Complex64,
            ColumnView::Complex128(_) => DType::Complex128,
        }
    }

    /// Returns the shape of the underlying array view.
    pub fn shape(&self) -> &[usize] {
        map_view!(self, a => a.shape())
    }

    /// Returns a view of the float data, or `None` if this column view is not `Float`.
    pub fn as_float(&self) -> Option<ArrayViewD<'a, F>> {
        match self {
            ColumnView::Float(a) => Some(a.clone()),
            _ => None,
        }
    }

    /// Returns a view of the integer data, or `None` if not `Int`.
    pub fn as_int(&self) -> Option<ArrayViewD<'a, I>> {
        match self {
            ColumnView::Int(a) => Some(a.clone()),
            _ => None,
        }
    }

    /// Returns a view of the boolean data, or `None` if not `Bool`.
    pub fn as_bool(&self) -> Option<ArrayViewD<'a, bool>> {
        match self {
            ColumnView::Bool(a) => Some(a.clone()),
            _ => None,
        }
    }

    /// Returns a view of the unsigned integer data, or `None` if not `UInt`.
    pub fn as_uint(&self) -> Option<ArrayViewD<'a, Idx>> {
        match self {
            ColumnView::UInt(a) => Some(a.clone()),
            _ => None,
        }
    }

    /// Returns a view of the u8 data, or `None` if not `U8`.
    pub fn as_u8(&self) -> Option<ArrayViewD<'a, u8>> {
        match self {
            ColumnView::U8(a) => Some(a.clone()),
            _ => None,
        }
    }

    /// Returns a view of the string data, or `None` if not `String`.
    pub fn as_string(&self) -> Option<ArrayViewD<'a, String>> {
        match self {
            ColumnView::String(a) => Some(a.clone()),
            _ => None,
        }
    }

    /// Creates an owned [`Column`] by cloning the viewed data.
    /// Format one row as EXTXYZ property tokens.
    pub fn xyz_tokens(&self, row: usize) -> Vec<String> {
        use ndarray::Axis;
        match self {
            ColumnView::Bool(a) => a
                .index_axis(Axis(0), row)
                .iter()
                .map(|v| if *v { "T" } else { "F" }.to_string())
                .collect(),
            ColumnView::String(a) => a.index_axis(Axis(0), row).iter().cloned().collect(),
            _ => map_view!(self, a => {
                a.index_axis(Axis(0), row)
                    .iter()
                    .map(|v| v.to_string())
                    .collect()
            }),
        }
    }

    pub fn to_owned(&self) -> Column {
        match self {
            ColumnView::Float16(a) => Column::from_f16(a.to_owned()),
            ColumnView::Float32(a) => Column::from_f32(a.to_owned()),
            ColumnView::Float(a) => Column::from_float(a.to_owned()),
            ColumnView::Int8(a) => Column::from_i8(a.to_owned()),
            ColumnView::Int16(a) => Column::from_i16(a.to_owned()),
            ColumnView::Int(a) => Column::from_int(a.to_owned()),
            ColumnView::Int64(a) => Column::from_i64(a.to_owned()),
            ColumnView::Bool(a) => Column::from_bool(a.to_owned()),
            ColumnView::UInt(a) => Column::from_uint(a.to_owned()),
            ColumnView::U8(a) => Column::from_u8(a.to_owned()),
            ColumnView::UInt16(a) => Column::from_u16(a.to_owned()),
            ColumnView::UInt32(a) => Column::from_u32(a.to_owned()),
            ColumnView::String(a) => Column::from_string(a.to_owned()),
            ColumnView::Complex64(a) => Column::from_c64(a.to_owned()),
            ColumnView::Complex128(a) => Column::from_c128(a.to_owned()),
        }
    }
}

impl<'a> From<&'a Column> for ColumnView<'a> {
    fn from(col: &'a Column) -> Self {
        match col {
            Column::Float16(a) => ColumnView::Float16(a.view()),
            Column::Float32(a) => ColumnView::Float32(a.view()),
            Column::Float(a) => ColumnView::Float(a.view()),
            Column::Int8(a) => ColumnView::Int8(a.view()),
            Column::Int16(a) => ColumnView::Int16(a.view()),
            Column::Int(a) => ColumnView::Int(a.view()),
            Column::Int64(a) => ColumnView::Int64(a.view()),
            Column::Bool(a) => ColumnView::Bool(a.view()),
            Column::UInt(a) => ColumnView::UInt(a.view()),
            Column::U8(a) => ColumnView::U8(a.view()),
            Column::UInt16(a) => ColumnView::UInt16(a.view()),
            Column::UInt32(a) => ColumnView::UInt32(a.view()),
            Column::String(a) => ColumnView::String(a.view()),
            Column::Complex64(a) => ColumnView::Complex64(a.view()),
            Column::Complex128(a) => ColumnView::Complex128(a.view()),
        }
    }
}

impl std::fmt::Debug for ColumnView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ColumnView::{:?}(shape={:?})",
            self.dtype(),
            self.shape()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_from_column_float() {
        let col = Column::from_float(Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::Float);
        assert_eq!(view.nrows(), Some(3));
        assert_eq!(view.shape(), &[3]);
        assert!(view.as_float().is_some());
        assert!(view.as_int().is_none());
    }

    #[test]
    fn test_from_column_int() {
        let col = Column::from_int(Array1::from_vec(vec![1 as I, 2, 3]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::Int);
        assert!(view.as_int().is_some());
        assert!(view.as_float().is_none());
    }

    #[test]
    fn test_from_column_bool() {
        let col = Column::from_bool(Array1::from_vec(vec![true, false]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::Bool);
        assert!(view.as_bool().is_some());
    }

    #[test]
    fn test_from_column_uint() {
        let col = Column::from_uint(Array1::from_vec(vec![1 as Idx, 2]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::UInt);
        assert!(view.as_uint().is_some());
    }

    #[test]
    fn test_from_column_u8() {
        let col = Column::from_u8(Array1::from_vec(vec![1u8, 2]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::U8);
        assert!(view.as_u8().is_some());
    }

    #[test]
    fn test_from_column_string() {
        let col = Column::from_string(
            Array1::from_vec(vec!["a".to_string(), "b".to_string()]).into_dyn(),
        );
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::String);
        assert!(view.as_string().is_some());
    }

    #[test]
    fn test_to_owned_roundtrip() {
        let col = Column::from_float(Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn());
        let view = ColumnView::from(&col);
        let owned = view.to_owned();
        assert_eq!(owned.dtype(), DType::Float);
        assert_eq!(owned.nrows(), Some(3));
        assert_eq!(
            owned.as_float().unwrap().as_slice_memory_order().unwrap(),
            &[1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_zero_copy() {
        let col = Column::from_float(Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn());
        let view = ColumnView::from(&col);
        // The view's data pointer should match the original column's data pointer
        let orig_ptr = col.as_float().unwrap().as_ptr();
        let view_ptr = view.as_float().unwrap().as_ptr();
        assert_eq!(orig_ptr, view_ptr);
    }
}
