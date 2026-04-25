//! Unified access traits for owned and borrowed column/block types.
//!
//! [`ColumnAccess`] and [`BlockAccess`] provide a common read-only interface
//! that is implemented by both the owned types ([`Column`], [`Block`]) and
//! their zero-copy view counterparts ([`ColumnView`], [`BlockView`]).

use ndarray::ArrayViewD;

use super::Block;
use super::block_view::BlockView;
use super::column::Column;
use super::column_view::ColumnView;
use super::dtype::DType;
use crate::types::{F, I, U};

/// Unified read-only access for [`Column`] and [`ColumnView`].
pub trait ColumnAccess {
    /// Returns a float array view, or `None` if the column is not `Float`.
    fn as_float_view(&self) -> Option<ArrayViewD<'_, F>>;
    /// Returns an int array view, or `None` if the column is not `Int`.
    fn as_int_view(&self) -> Option<ArrayViewD<'_, I>>;
    /// Returns a bool array view, or `None` if the column is not `Bool`.
    fn as_bool_view(&self) -> Option<ArrayViewD<'_, bool>>;
    /// Returns a uint array view, or `None` if the column is not `UInt`.
    fn as_uint_view(&self) -> Option<ArrayViewD<'_, U>>;
    /// Returns a u8 array view, or `None` if the column is not `U8`.
    fn as_u8_view(&self) -> Option<ArrayViewD<'_, u8>>;
    /// Returns a string array view, or `None` if the column is not `String`.
    fn as_string_view(&self) -> Option<ArrayViewD<'_, String>>;
    /// Returns the number of rows (axis-0 length), or `None` if rank 0.
    fn nrows(&self) -> Option<usize>;
    /// Returns the data type of this column.
    fn dtype(&self) -> DType;
    /// Returns the shape of the underlying array as an owned `Vec`.
    fn shape(&self) -> Vec<usize>;
}

impl ColumnAccess for Column {
    fn as_float_view(&self) -> Option<ArrayViewD<'_, F>> {
        self.as_float().map(|a| a.view())
    }

    fn as_int_view(&self) -> Option<ArrayViewD<'_, I>> {
        self.as_int().map(|a| a.view())
    }

    fn as_bool_view(&self) -> Option<ArrayViewD<'_, bool>> {
        self.as_bool().map(|a| a.view())
    }

    fn as_uint_view(&self) -> Option<ArrayViewD<'_, U>> {
        self.as_uint().map(|a| a.view())
    }

    fn as_u8_view(&self) -> Option<ArrayViewD<'_, u8>> {
        self.as_u8().map(|a| a.view())
    }

    fn as_string_view(&self) -> Option<ArrayViewD<'_, String>> {
        self.as_string().map(|a| a.view())
    }

    fn nrows(&self) -> Option<usize> {
        Column::nrows(self)
    }

    fn dtype(&self) -> DType {
        Column::dtype(self)
    }

    fn shape(&self) -> Vec<usize> {
        Column::shape(self).to_vec()
    }
}

impl ColumnAccess for ColumnView<'_> {
    fn as_float_view(&self) -> Option<ArrayViewD<'_, F>> {
        self.as_float()
    }

    fn as_int_view(&self) -> Option<ArrayViewD<'_, I>> {
        self.as_int()
    }

    fn as_bool_view(&self) -> Option<ArrayViewD<'_, bool>> {
        self.as_bool()
    }

    fn as_uint_view(&self) -> Option<ArrayViewD<'_, U>> {
        self.as_uint()
    }

    fn as_u8_view(&self) -> Option<ArrayViewD<'_, u8>> {
        self.as_u8()
    }

    fn as_string_view(&self) -> Option<ArrayViewD<'_, String>> {
        self.as_string()
    }

    fn nrows(&self) -> Option<usize> {
        ColumnView::nrows(self)
    }

    fn dtype(&self) -> DType {
        ColumnView::dtype(self)
    }

    fn shape(&self) -> Vec<usize> {
        ColumnView::shape(self).to_vec()
    }
}

/// Unified read-only access for [`Block`] and [`BlockView`].
pub trait BlockAccess {
    /// Gets a float array view for `key` if present and of correct type.
    fn get_float_view(&self, key: &str) -> Option<ArrayViewD<'_, F>>;
    /// Gets an int array view for `key` if present and of correct type.
    fn get_int_view(&self, key: &str) -> Option<ArrayViewD<'_, I>>;
    /// Gets a bool array view for `key` if present and of correct type.
    fn get_bool_view(&self, key: &str) -> Option<ArrayViewD<'_, bool>>;
    /// Gets a uint array view for `key` if present and of correct type.
    fn get_uint_view(&self, key: &str) -> Option<ArrayViewD<'_, U>>;
    /// Gets a u8 array view for `key` if present and of correct type.
    fn get_u8_view(&self, key: &str) -> Option<ArrayViewD<'_, u8>>;
    /// Gets a string array view for `key` if present and of correct type.
    fn get_string_view(&self, key: &str) -> Option<ArrayViewD<'_, String>>;
    /// Returns the common axis-0 length, or `None` if empty.
    fn nrows(&self) -> Option<usize>;
    /// Number of columns.
    fn len(&self) -> usize;
    /// Returns `true` if there are no columns.
    fn is_empty(&self) -> bool;
    /// Returns `true` if the block contains the specified key.
    fn contains_key(&self, key: &str) -> bool;
    /// Returns column keys as a `Vec`.
    fn column_keys(&self) -> Vec<&str>;
    /// Returns the data type of the column with the given key, if it exists.
    fn column_dtype(&self, key: &str) -> Option<DType>;
    /// Returns the shape of the column with the given key, if it exists.
    fn column_shape(&self, key: &str) -> Option<Vec<usize>>;
}

impl BlockAccess for Block {
    fn get_float_view(&self, key: &str) -> Option<ArrayViewD<'_, F>> {
        self.get_float(key).map(|a| a.view())
    }

    fn get_int_view(&self, key: &str) -> Option<ArrayViewD<'_, I>> {
        self.get_int(key).map(|a| a.view())
    }

    fn get_bool_view(&self, key: &str) -> Option<ArrayViewD<'_, bool>> {
        self.get_bool(key).map(|a| a.view())
    }

    fn get_uint_view(&self, key: &str) -> Option<ArrayViewD<'_, U>> {
        self.get_uint(key).map(|a| a.view())
    }

    fn get_u8_view(&self, key: &str) -> Option<ArrayViewD<'_, u8>> {
        self.get_u8(key).map(|a| a.view())
    }

    fn get_string_view(&self, key: &str) -> Option<ArrayViewD<'_, String>> {
        self.get_string(key).map(|a| a.view())
    }

    fn nrows(&self) -> Option<usize> {
        Block::nrows(self)
    }

    fn len(&self) -> usize {
        Block::len(self)
    }

    fn is_empty(&self) -> bool {
        Block::is_empty(self)
    }

    fn contains_key(&self, key: &str) -> bool {
        Block::contains_key(self, key)
    }

    fn column_keys(&self) -> Vec<&str> {
        self.keys().collect()
    }

    fn column_dtype(&self, key: &str) -> Option<DType> {
        self.get(key).map(|col| col.dtype())
    }

    fn column_shape(&self, key: &str) -> Option<Vec<usize>> {
        self.get(key).map(|col| col.shape().to_vec())
    }
}

impl BlockAccess for BlockView<'_> {
    fn get_float_view(&self, key: &str) -> Option<ArrayViewD<'_, F>> {
        self.get_float(key)
    }

    fn get_int_view(&self, key: &str) -> Option<ArrayViewD<'_, I>> {
        self.get_int(key)
    }

    fn get_bool_view(&self, key: &str) -> Option<ArrayViewD<'_, bool>> {
        self.get_bool(key)
    }

    fn get_uint_view(&self, key: &str) -> Option<ArrayViewD<'_, U>> {
        self.get_uint(key)
    }

    fn get_u8_view(&self, key: &str) -> Option<ArrayViewD<'_, u8>> {
        self.get_u8(key)
    }

    fn get_string_view(&self, key: &str) -> Option<ArrayViewD<'_, String>> {
        self.get_string(key)
    }

    fn nrows(&self) -> Option<usize> {
        BlockView::nrows(self)
    }

    fn len(&self) -> usize {
        BlockView::len(self)
    }

    fn is_empty(&self) -> bool {
        BlockView::is_empty(self)
    }

    fn contains_key(&self, key: &str) -> bool {
        BlockView::contains_key(self, key)
    }

    fn column_keys(&self) -> Vec<&str> {
        self.keys().copied().collect()
    }

    fn column_dtype(&self, key: &str) -> Option<DType> {
        BlockView::dtype(self, key)
    }

    fn column_shape(&self, key: &str) -> Option<Vec<usize>> {
        self.get(key).map(|col_view| col_view.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_block() -> Block {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn())
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![10 as I, 20, 30]).into_dyn())
            .unwrap();
        block
    }

    #[test]
    fn test_column_access_on_column() {
        let col = Column::from_float(Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn());
        assert!(ColumnAccess::as_float_view(&col).is_some());
        assert!(ColumnAccess::as_int_view(&col).is_none());
        assert_eq!(ColumnAccess::nrows(&col), Some(2));
        assert_eq!(ColumnAccess::dtype(&col), DType::Float);
        assert_eq!(ColumnAccess::shape(&col), vec![2]);
    }

    #[test]
    fn test_column_access_on_column_view() {
        let col = Column::from_int(Array1::from_vec(vec![1 as I, 2, 3]).into_dyn());
        let view = ColumnView::from(&col);
        assert!(ColumnAccess::as_int_view(&view).is_some());
        assert!(ColumnAccess::as_float_view(&view).is_none());
        assert_eq!(ColumnAccess::nrows(&view), Some(3));
        assert_eq!(ColumnAccess::dtype(&view), DType::Int);
    }

    #[test]
    fn test_block_access_on_block() {
        let block = make_block();
        assert!(BlockAccess::get_float_view(&block, "x").is_some());
        assert!(BlockAccess::get_int_view(&block, "id").is_some());
        assert_eq!(BlockAccess::nrows(&block), Some(3));
        assert_eq!(BlockAccess::len(&block), 2);
        assert!(!BlockAccess::is_empty(&block));
        assert!(BlockAccess::contains_key(&block, "x"));
        assert!(!BlockAccess::contains_key(&block, "missing"));
    }

    #[test]
    fn test_block_access_on_block_view() {
        let block = make_block();
        let view = BlockView::from(&block);
        assert!(BlockAccess::get_float_view(&view, "x").is_some());
        assert!(BlockAccess::get_int_view(&view, "id").is_some());
        assert_eq!(BlockAccess::nrows(&view), Some(3));
        assert_eq!(BlockAccess::len(&view), 2);
        assert!(!BlockAccess::is_empty(&view));
        assert!(BlockAccess::contains_key(&view, "x"));
    }

    #[test]
    fn test_generic_function_with_block_access() {
        fn count_float_columns(b: &impl BlockAccess) -> usize {
            b.column_keys()
                .iter()
                .filter(|k| b.get_float_view(k).is_some())
                .count()
        }

        let block = make_block();
        assert_eq!(count_float_columns(&block), 1);

        let view = BlockView::from(&block);
        assert_eq!(count_float_columns(&view), 1);
    }
}
