//! Data type enumeration and trait for Block columns.

use ndarray::ArrayD;
use num_complex::Complex;

use super::column::Column;
use crate::types::{F, I, Idx};

/// Supported data types for Block columns.
///
/// Domain aliases [`F`] / [`I`] / [`Idx`] stay on [`DType::Float`] /
/// [`DType::Int`] / [`DType::UInt`]. Every other variant is a storage width:
/// a column that arrived as `f32` or `i64` has to leave as that width, not as
/// the compute scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// IEEE binary16.
    Float16,
    /// IEEE binary32.
    Float32,
    /// Floating point using the compute scalar [`F`] (`f64`).
    Float,
    /// Signed 8-bit integer.
    Int8,
    /// Signed 16-bit integer.
    Int16,
    /// Signed integer using the domain scalar [`I`] (`i32`).
    Int,
    /// Signed 64-bit integer.
    Int64,
    /// Boolean
    Bool,
    /// Unsigned integer using the identifier scalar [`Idx`] (`u64`).
    UInt,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer.
    UInt16,
    /// 32-bit unsigned integer.
    UInt32,
    /// String
    String,
    /// Complex pair of `f32` (numpy `complex64`).
    Complex64,
    /// Complex pair of `f64` (numpy `complex128`).
    Complex128,
}

impl DType {
    /// Returns the name of the data type as a string.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Float16 => "f16",
            DType::Float32 => "f32",
            DType::Float => "float",
            DType::Int8 => "i8",
            DType::Int16 => "i16",
            DType::Int => "int",
            DType::Int64 => "i64",
            DType::Bool => "bool",
            DType::UInt => "uint",
            DType::U8 => "u8",
            DType::UInt16 => "u16",
            DType::UInt32 => "u32",
            DType::String => "string",
            DType::Complex64 => "c64",
            DType::Complex128 => "c128",
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

macro_rules! impl_block_dtype {
    ($ty:ty, $dtype:expr, $into:ident, $as_ref:ident, $as_mut:ident) => {
        impl BlockDtype for $ty {
            fn dtype() -> DType {
                $dtype
            }
            fn into_column(arr: ArrayD<Self>) -> Column {
                Column::$into(arr)
            }
            fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
                col.$as_ref()
            }
            fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
                col.$as_mut()
            }
        }
    };
}

impl_block_dtype!(half::f16, DType::Float16, from_f16, as_f16, as_f16_mut);
impl_block_dtype!(f32, DType::Float32, from_f32, as_f32, as_f32_mut);
impl_block_dtype!(F, DType::Float, from_float, as_float, as_float_mut);
impl_block_dtype!(i8, DType::Int8, from_i8, as_i8, as_i8_mut);
impl_block_dtype!(i16, DType::Int16, from_i16, as_i16, as_i16_mut);
impl_block_dtype!(I, DType::Int, from_int, as_int, as_int_mut);
impl_block_dtype!(i64, DType::Int64, from_i64, as_i64, as_i64_mut);
impl_block_dtype!(bool, DType::Bool, from_bool, as_bool, as_bool_mut);
impl_block_dtype!(Idx, DType::UInt, from_uint, as_uint, as_uint_mut);
impl_block_dtype!(u8, DType::U8, from_u8, as_u8, as_u8_mut);
impl_block_dtype!(u16, DType::UInt16, from_u16, as_u16, as_u16_mut);
impl_block_dtype!(u32, DType::UInt32, from_u32, as_u32, as_u32_mut);
impl_block_dtype!(String, DType::String, from_string, as_string, as_string_mut);
impl_block_dtype!(Complex<f32>, DType::Complex64, from_c64, as_c64, as_c64_mut);
impl_block_dtype!(
    Complex<f64>,
    DType::Complex128,
    from_c128,
    as_c128,
    as_c128_mut
);
