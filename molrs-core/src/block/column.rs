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

    /// Returns a reference to the float data, or `None` if this column is not `Float`.
    pub fn as_float(&self) -> Option<&ArrayD<F>> {
        match self {
            Column::Float(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a mutable reference to the float data, or `None` if not `Float`.
    pub fn as_float_mut(&mut self) -> Option<&mut ArrayD<F>> {
        match self {
            Column::Float(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a reference to the integer data, or `None` if not `Int`.
    pub fn as_int(&self) -> Option<&ArrayD<I>> {
        match self {
            Column::Int(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a mutable reference to the integer data, or `None` if not `Int`.
    pub fn as_int_mut(&mut self) -> Option<&mut ArrayD<I>> {
        match self {
            Column::Int(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a reference to the boolean data, or `None` if not `Bool`.
    pub fn as_bool(&self) -> Option<&ArrayD<bool>> {
        match self {
            Column::Bool(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a mutable reference to the boolean data, or `None` if not `Bool`.
    pub fn as_bool_mut(&mut self) -> Option<&mut ArrayD<bool>> {
        match self {
            Column::Bool(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a reference to the unsigned integer data, or `None` if not `UInt`.
    pub fn as_uint(&self) -> Option<&ArrayD<U>> {
        match self {
            Column::UInt(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a mutable reference to the unsigned integer data, or `None` if not `UInt`.
    pub fn as_uint_mut(&mut self) -> Option<&mut ArrayD<U>> {
        match self {
            Column::UInt(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a reference to the u8 data, or `None` if not `U8`.
    pub fn as_u8(&self) -> Option<&ArrayD<u8>> {
        match self {
            Column::U8(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a mutable reference to the u8 data, or `None` if not `U8`.
    pub fn as_u8_mut(&mut self) -> Option<&mut ArrayD<u8>> {
        match self {
            Column::U8(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a reference to the string data, or `None` if not `String`.
    pub fn as_string(&self) -> Option<&ArrayD<String>> {
        match self {
            Column::String(a) => Some(a),
            _ => None,
        }
    }

    /// Returns a mutable reference to the string data, or `None` if not `String`.
    pub fn as_string_mut(&mut self) -> Option<&mut ArrayD<String>> {
        match self {
            Column::String(a) => Some(a),
            _ => None,
        }
    }

    /// Resize this column along axis 0 to `new_nrows`.
    ///
    /// - If `new_nrows` < current nrows, slices to keep only the first `new_nrows` rows.
    /// - If `new_nrows` > current nrows, extends with default values
    ///   (0.0 for Float, 0 for Int/UInt/U8, false for Bool, empty string for String).
    /// - If `new_nrows` == current nrows, this is a no-op.
    ///
    /// Only axis 0 is modified; trailing dimensions are preserved.
    pub fn resize(&mut self, new_nrows: usize) {
        use ndarray::{Axis, IxDyn, concatenate};

        let current = self.shape()[0];
        if new_nrows == current {
            return;
        }

        match self {
            Column::Float(a) => {
                if new_nrows < current {
                    *a = a.slice_axis(Axis(0), (..new_nrows).into()).to_owned();
                } else {
                    let mut pad_shape = a.shape().to_vec();
                    pad_shape[0] = new_nrows - current;
                    let pad = ArrayD::<F>::zeros(IxDyn(&pad_shape));
                    *a = concatenate(Axis(0), &[a.view(), pad.view()]).unwrap();
                }
            }
            Column::Int(a) => {
                if new_nrows < current {
                    *a = a.slice_axis(Axis(0), (..new_nrows).into()).to_owned();
                } else {
                    let mut pad_shape = a.shape().to_vec();
                    pad_shape[0] = new_nrows - current;
                    let pad = ArrayD::<I>::zeros(IxDyn(&pad_shape));
                    *a = concatenate(Axis(0), &[a.view(), pad.view()]).unwrap();
                }
            }
            Column::UInt(a) => {
                if new_nrows < current {
                    *a = a.slice_axis(Axis(0), (..new_nrows).into()).to_owned();
                } else {
                    let mut pad_shape = a.shape().to_vec();
                    pad_shape[0] = new_nrows - current;
                    let pad = ArrayD::<U>::zeros(IxDyn(&pad_shape));
                    *a = concatenate(Axis(0), &[a.view(), pad.view()]).unwrap();
                }
            }
            Column::U8(a) => {
                if new_nrows < current {
                    *a = a.slice_axis(Axis(0), (..new_nrows).into()).to_owned();
                } else {
                    let mut pad_shape = a.shape().to_vec();
                    pad_shape[0] = new_nrows - current;
                    let pad = ArrayD::<u8>::zeros(IxDyn(&pad_shape));
                    *a = concatenate(Axis(0), &[a.view(), pad.view()]).unwrap();
                }
            }
            Column::Bool(a) => {
                if new_nrows < current {
                    *a = a.slice_axis(Axis(0), (..new_nrows).into()).to_owned();
                } else {
                    let mut pad_shape = a.shape().to_vec();
                    pad_shape[0] = new_nrows - current;
                    let pad = ArrayD::<bool>::default(IxDyn(&pad_shape));
                    *a = concatenate(Axis(0), &[a.view(), pad.view()]).unwrap();
                }
            }
            Column::String(a) => {
                if new_nrows < current {
                    *a = a.slice_axis(Axis(0), (..new_nrows).into()).to_owned();
                } else {
                    let mut pad_shape = a.shape().to_vec();
                    pad_shape[0] = new_nrows - current;
                    let pad = ArrayD::<String>::default(IxDyn(&pad_shape));
                    *a = concatenate(Axis(0), &[a.view(), pad.view()]).unwrap();
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{F, I, U};
    use ndarray::{Array1, ArrayD};

    // ---- helpers ----

    fn float_col(n: usize) -> Column {
        Column::Float(Array1::from_vec(vec![0.0 as F; n]).into_dyn())
    }

    fn int_col(n: usize) -> Column {
        Column::Int(Array1::from_vec(vec![0 as I; n]).into_dyn())
    }

    fn bool_col(n: usize) -> Column {
        Column::Bool(Array1::from_vec(vec![false; n]).into_dyn())
    }

    fn uint_col(n: usize) -> Column {
        Column::UInt(Array1::from_vec(vec![0 as U; n]).into_dyn())
    }

    fn u8_col(n: usize) -> Column {
        Column::U8(Array1::from_vec(vec![0u8; n]).into_dyn())
    }

    fn string_col(n: usize) -> Column {
        Column::String(Array1::from_vec(vec![String::new(); n]).into_dyn())
    }

    // ---- 1. nrows ----

    #[test]
    fn test_nrows() {
        assert_eq!(float_col(5).nrows(), Some(5));
        assert_eq!(int_col(3).nrows(), Some(3));
        assert_eq!(bool_col(7).nrows(), Some(7));
        assert_eq!(uint_col(2).nrows(), Some(2));
        assert_eq!(u8_col(4).nrows(), Some(4));
        assert_eq!(string_col(1).nrows(), Some(1));

        // rank-0 array has no axis-0 dimension
        let rank0 = Column::Float(ArrayD::<F>::from_elem(vec![], 1.0));
        assert_eq!(rank0.nrows(), None);
    }

    // ---- 2. dtype ----

    #[test]
    fn test_dtype() {
        assert_eq!(float_col(1).dtype(), DType::Float);
        assert_eq!(int_col(1).dtype(), DType::Int);
        assert_eq!(bool_col(1).dtype(), DType::Bool);
        assert_eq!(uint_col(1).dtype(), DType::UInt);
        assert_eq!(u8_col(1).dtype(), DType::U8);
        assert_eq!(string_col(1).dtype(), DType::String);
    }

    // ---- 3. shape ----

    #[test]
    fn test_shape() {
        // 1-D
        assert_eq!(float_col(4).shape(), &[4]);
        // 2-D
        let col2d = Column::Int(ArrayD::<I>::from_elem(vec![3, 2], 0));
        assert_eq!(col2d.shape(), &[3, 2]);
    }

    // ---- 4. as_float on Float ----

    #[test]
    fn test_as_float_on_float() {
        let col = float_col(3);
        assert!(col.as_float().is_some());
        assert_eq!(col.as_float().unwrap().len(), 3);
    }

    // ---- 5. as_float on wrong type ----

    #[test]
    fn test_as_float_on_wrong_type() {
        assert!(int_col(2).as_float().is_none());
        assert!(bool_col(2).as_float().is_none());
        assert!(uint_col(2).as_float().is_none());
        assert!(u8_col(2).as_float().is_none());
        assert!(string_col(2).as_float().is_none());
    }

    // ---- 6. as_int ----

    #[test]
    fn test_as_int() {
        let col = int_col(4);
        assert!(col.as_int().is_some());
        assert_eq!(col.as_int().unwrap().len(), 4);
        // wrong types return None
        assert!(float_col(1).as_int().is_none());
        assert!(bool_col(1).as_int().is_none());
    }

    // ---- 7. as_bool ----

    #[test]
    fn test_as_bool() {
        let col = bool_col(2);
        assert!(col.as_bool().is_some());
        assert_eq!(col.as_bool().unwrap().len(), 2);
        // wrong types return None
        assert!(float_col(1).as_bool().is_none());
        assert!(int_col(1).as_bool().is_none());
    }

    // ---- 8. as_uint ----

    #[test]
    fn test_as_uint() {
        let col = uint_col(6);
        assert!(col.as_uint().is_some());
        assert_eq!(col.as_uint().unwrap().len(), 6);
        // wrong types return None
        assert!(float_col(1).as_uint().is_none());
        assert!(int_col(1).as_uint().is_none());
    }

    // ---- 9. as_u8 ----

    #[test]
    fn test_as_u8() {
        let col = u8_col(3);
        assert!(col.as_u8().is_some());
        assert_eq!(col.as_u8().unwrap().len(), 3);
        // wrong types return None
        assert!(float_col(1).as_u8().is_none());
        assert!(uint_col(1).as_u8().is_none());
    }

    // ---- 10. as_string ----

    #[test]
    fn test_as_string() {
        let col = string_col(2);
        assert!(col.as_string().is_some());
        assert_eq!(col.as_string().unwrap().len(), 2);
        // wrong types return None
        assert!(float_col(1).as_string().is_none());
        assert!(int_col(1).as_string().is_none());
    }

    // ---- 11. as_float_mut ----

    #[test]
    fn test_as_float_mut() {
        let mut col =
            Column::Float(Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn());
        {
            let arr = col.as_float_mut().unwrap();
            arr[0] = 99.0;
        }
        let arr = col.as_float().unwrap();
        assert!((arr[0] - 99.0).abs() < F::EPSILON);
        // wrong variant returns None
        let mut int = int_col(1);
        assert!(int.as_float_mut().is_none());
    }

    // ---- 12. Debug format ----

    #[test]
    fn test_debug_format() {
        let dbg = format!("{:?}", float_col(3));
        assert!(dbg.contains("Column::Float"));
        assert!(dbg.contains("shape="));

        let dbg = format!("{:?}", int_col(2));
        assert!(dbg.contains("Column::Int"));

        let dbg = format!("{:?}", bool_col(1));
        assert!(dbg.contains("Column::Bool"));

        let dbg = format!("{:?}", uint_col(4));
        assert!(dbg.contains("Column::UInt"));

        let dbg = format!("{:?}", u8_col(5));
        assert!(dbg.contains("Column::U8"));

        let dbg = format!("{:?}", string_col(1));
        assert!(dbg.contains("Column::String"));
    }
}
