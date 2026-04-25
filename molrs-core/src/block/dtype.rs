//! Data type enumeration and trait for Block columns.

use ndarray::ArrayD;

use super::column::Column;
use crate::types::{F, I, U};

/// Supported data types for Block columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// Floating point using the compile-time scalar type [`F`].
    Float,
    /// Signed integer using the compile-time scalar type [`I`].
    Int,
    /// Boolean
    Bool,
    /// Unsigned integer using the compile-time scalar type [`U`].
    UInt,
    /// 8-bit unsigned integer
    U8,
    /// String
    String,
}

impl DType {
    /// Returns the name of the data type as a string.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Float => "float",
            DType::Int => "int",
            DType::Bool => "bool",
            DType::UInt => "uint",
            DType::U8 => "u8",
            DType::String => "string",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Trait for types that can be stored in a Block column.
///
/// This trait provides the mechanism for generic dispatch when inserting
/// arrays into a Block. Users don't need to interact with this trait directly.
pub trait BlockDtype: Sized + 'static {
    /// Returns the DType for this type.
    fn dtype() -> DType;

    /// Converts an ArrayD of this type into a Column.
    fn into_column(arr: ArrayD<Self>) -> Column;

    /// Tries to extract a reference to an ArrayD of this type from a Column.
    fn from_column(col: &Column) -> Option<&ArrayD<Self>>;

    /// Tries to extract a mutable reference to an ArrayD of this type from a Column.
    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>>;
}

impl BlockDtype for F {
    fn dtype() -> DType {
        DType::Float
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::from_float(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_float()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_float_mut()
    }
}

impl BlockDtype for I {
    fn dtype() -> DType {
        DType::Int
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::from_int(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_int()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_int_mut()
    }
}

impl BlockDtype for bool {
    fn dtype() -> DType {
        DType::Bool
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::from_bool(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_bool()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_bool_mut()
    }
}

impl BlockDtype for U {
    fn dtype() -> DType {
        DType::UInt
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::from_uint(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_uint()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_uint_mut()
    }
}

impl BlockDtype for u8 {
    fn dtype() -> DType {
        DType::U8
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::from_u8(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_u8()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_u8_mut()
    }
}

impl BlockDtype for String {
    fn dtype() -> DType {
        DType::String
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::from_string(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_string()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_string_mut()
    }
}
