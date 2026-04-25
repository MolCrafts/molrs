//! Zero-copy borrowed view of a [`Block`].
//!
//! `BlockView<'a>` borrows columns from a [`Block`] as [`ColumnView`]s without
//! copying any array data, providing read-only access with the same API surface
//! as `Block`.

use std::collections::HashMap;

use ndarray::ArrayViewD;

use super::Block;
use super::column::Column;
use super::column_view::ColumnView;
use super::dtype::DType;
use crate::types::{F, I, U};

/// A borrowed, read-only view of a [`Block`].
///
/// Keys are `&str` references into the original `Block`'s key strings.
/// Values are [`ColumnView`]s that borrow the underlying array data.
pub struct BlockView<'a> {
    map: HashMap<&'a str, ColumnView<'a>>,
    nrows: Option<usize>,
}

impl<'a> BlockView<'a> {
    /// Creates an empty `BlockView`.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            nrows: None,
        }
    }

    /// Inserts a column view under the given key.
    ///
    /// If the `BlockView` was empty, `nrows` is set from the column's axis-0
    /// length. Subsequent insertions are not validated for consistency (caller
    /// is responsible).
    pub fn insert(&mut self, key: &'a str, col: ColumnView<'a>) {
        if self.nrows.is_none() {
            self.nrows = col.nrows();
        }
        self.map.insert(key, col);
    }

    /// Number of columns in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the view contains no columns.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the common axis-0 length, or `None` if empty.
    #[inline]
    pub fn nrows(&self) -> Option<usize> {
        self.nrows
    }

    /// Returns `true` if the view contains the specified key.
    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Gets an immutable reference to the column view for `key` if present.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&ColumnView<'a>> {
        self.map.get(key)
    }

    /// Gets a float array view for `key` if present and of correct type.
    pub fn get_float(&self, key: &str) -> Option<ArrayViewD<'a, F>> {
        self.get(key).and_then(|c| c.as_float())
    }

    /// Gets an int array view for `key` if present and of correct type.
    pub fn get_int(&self, key: &str) -> Option<ArrayViewD<'a, I>> {
        self.get(key).and_then(|c| c.as_int())
    }

    /// Gets a bool array view for `key` if present and of correct type.
    pub fn get_bool(&self, key: &str) -> Option<ArrayViewD<'a, bool>> {
        self.get(key).and_then(|c| c.as_bool())
    }

    /// Gets a uint array view for `key` if present and of correct type.
    pub fn get_uint(&self, key: &str) -> Option<ArrayViewD<'a, U>> {
        self.get(key).and_then(|c| c.as_uint())
    }

    /// Gets a u8 array view for `key` if present and of correct type.
    pub fn get_u8(&self, key: &str) -> Option<ArrayViewD<'a, u8>> {
        self.get(key).and_then(|c| c.as_u8())
    }

    /// Gets a string array view for `key` if present and of correct type.
    pub fn get_string(&self, key: &str) -> Option<ArrayViewD<'a, String>> {
        self.get(key).and_then(|c| c.as_string())
    }

    /// Returns an iterator over `(&str, &ColumnView)`.
    pub fn iter(&self) -> impl Iterator<Item = (&&'a str, &ColumnView<'a>)> {
        self.map.iter()
    }

    /// Returns an iterator over column keys.
    pub fn keys(&self) -> impl Iterator<Item = &&'a str> {
        self.map.keys()
    }

    /// Returns an iterator over column view references.
    pub fn values(&self) -> impl Iterator<Item = &ColumnView<'a>> {
        self.map.values()
    }

    /// Returns the data type of the column with the given key, if it exists.
    pub fn dtype(&self, key: &str) -> Option<DType> {
        self.get(key).map(|c| c.dtype())
    }

    /// Creates an owned [`Block`] by cloning all viewed data.
    pub fn to_owned(&self) -> Block {
        let mut block = Block::new();
        for (&key, col_view) in &self.map {
            let col: Column = col_view.to_owned();
            let _ = block.insert_column(key, col);
        }
        block
    }
}

impl<'a> Default for BlockView<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> From<&'a Block> for BlockView<'a> {
    fn from(block: &'a Block) -> Self {
        let mut view = BlockView {
            map: HashMap::with_capacity(block.len()),
            nrows: block.nrows(),
        };
        for (key, col) in block.iter() {
            view.map.insert(key, ColumnView::from(col));
        }
        view
    }
}

impl std::fmt::Debug for BlockView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (k, v) in &self.map {
            map.entry(k, &format!("{}(shape={:?})", v.dtype(), v.shape()));
        }
        map.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_from_block() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn())
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![10 as I, 20, 30]).into_dyn())
            .unwrap();

        let view = BlockView::from(&block);
        assert_eq!(view.len(), 2);
        assert_eq!(view.nrows(), Some(3));
        assert!(view.contains_key("x"));
        assert!(view.contains_key("id"));
        assert!(!view.is_empty());
    }

    #[test]
    fn test_typed_getters() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn())
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![10 as I, 20, 30]).into_dyn())
            .unwrap();

        let view = BlockView::from(&block);

        // Correct type
        assert!(view.get_float("x").is_some());
        assert!(view.get_int("id").is_some());

        // Wrong type
        assert!(view.get_int("x").is_none());
        assert!(view.get_float("id").is_none());

        // Missing key
        assert!(view.get_float("missing").is_none());
    }

    #[test]
    fn test_to_owned_roundtrip() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn())
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![10 as I, 20, 30]).into_dyn())
            .unwrap();

        let view = BlockView::from(&block);
        let owned = view.to_owned();

        assert_eq!(owned.nrows(), Some(3));
        assert_eq!(owned.len(), 2);
        assert_eq!(
            owned
                .get_float("x")
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[1.0, 2.0, 3.0]
        );
        assert_eq!(
            owned
                .get_int("id")
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[10, 20, 30]
        );
    }

    #[test]
    fn test_zero_copy() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn())
            .unwrap();

        let view = BlockView::from(&block);
        let orig_ptr = block.get_float("x").unwrap().as_ptr();
        let view_ptr = view.get_float("x").unwrap().as_ptr();
        assert_eq!(orig_ptr, view_ptr);
    }

    #[test]
    fn test_empty_view() {
        let view = BlockView::new();
        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
        assert_eq!(view.nrows(), None);
    }

    #[test]
    fn test_iter_keys() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn())
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![10 as I, 20]).into_dyn())
            .unwrap();

        let view = BlockView::from(&block);
        let keys: Vec<&&str> = view.keys().collect();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_dtype_query() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn())
            .unwrap();

        let view = BlockView::from(&block);
        assert_eq!(view.dtype("x"), Some(DType::Float));
        assert_eq!(view.dtype("missing"), None);
    }
}
