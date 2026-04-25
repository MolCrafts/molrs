//! Zero-copy borrowed view of a [`Column`].
//!
//! `ColumnView<'a>` borrows the underlying ndarray data from a [`Column`] without
//! copying, providing read-only access with the same API surface as `Column`.

use ndarray::ArrayViewD;

use super::column::Column;
use super::dtype::DType;
use crate::types::{F, I, U};

/// A borrowed, read-only view of a [`Column`].
///
/// Each variant holds an `ArrayViewD` that borrows from the corresponding
/// `ArrayD` inside an owned `Column`. No data is copied.
pub enum ColumnView<'a> {
    /// Borrowed float column.
    Float(ArrayViewD<'a, F>),
    /// Borrowed signed integer column.
    Int(ArrayViewD<'a, I>),
    /// Borrowed boolean column.
    Bool(ArrayViewD<'a, bool>),
    /// Borrowed unsigned integer column.
    UInt(ArrayViewD<'a, U>),
    /// Borrowed u8 column.
    U8(ArrayViewD<'a, u8>),
    /// Borrowed string column.
    String(ArrayViewD<'a, String>),
}

impl<'a> ColumnView<'a> {
    /// Returns the number of rows (axis-0 length) of this column view.
    ///
    /// Returns `None` if the array has rank 0.
    pub fn nrows(&self) -> Option<usize> {
        match self {
            ColumnView::Float(a) => a.shape().first().copied(),
            ColumnView::Int(a) => a.shape().first().copied(),
            ColumnView::Bool(a) => a.shape().first().copied(),
            ColumnView::UInt(a) => a.shape().first().copied(),
            ColumnView::U8(a) => a.shape().first().copied(),
            ColumnView::String(a) => a.shape().first().copied(),
        }
    }

    /// Returns the data type of this column view.
    pub fn dtype(&self) -> DType {
        match self {
            ColumnView::Float(_) => DType::Float,
            ColumnView::Int(_) => DType::Int,
            ColumnView::Bool(_) => DType::Bool,
            ColumnView::UInt(_) => DType::UInt,
            ColumnView::U8(_) => DType::U8,
            ColumnView::String(_) => DType::String,
        }
    }

    /// Returns the shape of the underlying array view.
    pub fn shape(&self) -> &[usize] {
        match self {
            ColumnView::Float(a) => a.shape(),
            ColumnView::Int(a) => a.shape(),
            ColumnView::Bool(a) => a.shape(),
            ColumnView::UInt(a) => a.shape(),
            ColumnView::U8(a) => a.shape(),
            ColumnView::String(a) => a.shape(),
        }
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
    pub fn as_uint(&self) -> Option<ArrayViewD<'a, U>> {
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
    pub fn to_owned(&self) -> Column {
        match self {
            ColumnView::Float(a) => Column::from_float(a.to_owned()),
            ColumnView::Int(a) => Column::from_int(a.to_owned()),
            ColumnView::Bool(a) => Column::from_bool(a.to_owned()),
            ColumnView::UInt(a) => Column::from_uint(a.to_owned()),
            ColumnView::U8(a) => Column::from_u8(a.to_owned()),
            ColumnView::String(a) => Column::from_string(a.to_owned()),
        }
    }
}

impl<'a> From<&'a Column> for ColumnView<'a> {
    fn from(col: &'a Column) -> Self {
        match col {
            Column::Float(a) => ColumnView::Float(a.view()),
            Column::Int(a) => ColumnView::Int(a.view()),
            Column::Bool(a) => ColumnView::Bool(a.view()),
            Column::UInt(a) => ColumnView::UInt(a.view()),
            Column::U8(a) => ColumnView::U8(a.view()),
            Column::String(a) => ColumnView::String(a.view()),
        }
    }
}

impl std::fmt::Debug for ColumnView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColumnView::Float(a) => write!(f, "ColumnView::Float(shape={:?})", a.shape()),
            ColumnView::Int(a) => write!(f, "ColumnView::Int(shape={:?})", a.shape()),
            ColumnView::Bool(a) => write!(f, "ColumnView::Bool(shape={:?})", a.shape()),
            ColumnView::UInt(a) => write!(f, "ColumnView::UInt(shape={:?})", a.shape()),
            ColumnView::U8(a) => write!(f, "ColumnView::U8(shape={:?})", a.shape()),
            ColumnView::String(a) => write!(f, "ColumnView::String(shape={:?})", a.shape()),
        }
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
        let col = Column::from_uint(Array1::from_vec(vec![1 as U, 2]).into_dyn());
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
