//! Block: dict-like keyed arrays with consistent axis-0 length and heterogeneous types.
//!
//! A Block stores heterogeneous arrays (float, int, bool) keyed by strings,
//! enforcing that all stored arrays share the same axis-0 length (nrows).
//!
//! # Examples
//!
//! ```
//! use molrs_core::block::Block;
//! use molrs_core::types::{F, I};
//! use ndarray::{Array1, ArrayD};
//!
//! let mut block = Block::new();
//!
//! // Insert different types - generic dispatch handles the conversion
//! let pos = Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn();
//! let ids = Array1::from_vec(vec![10 as I, 20 as I, 30 as I]).into_dyn();
//!
//! block.insert("pos", pos).unwrap();
//! block.insert("id", ids).unwrap();
//!
//! // Type-safe retrieval
//! let pos_ref = block.get_float("pos").unwrap();
//! let ids_ref = block.get_int("id").unwrap();
//!
//! assert_eq!(block.nrows(), Some(3));
//! assert_eq!(block.len(), 2);
//! ```

mod column;
mod dtype;
mod error;

pub mod access;
pub mod block_view;
pub mod column_view;

pub use access::{BlockAccess, ColumnAccess};
pub use block_view::BlockView;
pub use column::{Column, ColumnHolder};
pub use column_view::ColumnView;
pub use dtype::{BlockDtype, DType};
pub use error::BlockError;

use ndarray::ArrayD;
use std::collections::HashMap;
use std::ops::{Index, IndexMut};

/// A dictionary from string keys to ndarray arrays with a consistent axis-0 length.
///
/// This Block supports heterogeneous column types (float, int, bool).
#[derive(Default, Clone)]
pub struct Block {
    map: HashMap<String, Column>,
    nrows: Option<usize>,
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (k, v) in &self.map {
            map.entry(k, &format!("{}(shape={:?})", v.dtype(), v.shape()));
        }
        map.finish()
    }
}

impl Block {
    /// Creates an empty Block.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            nrows: None,
        }
    }

    /// Creates an empty Block with the specified capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: HashMap::with_capacity(cap),
            nrows: None,
        }
    }

    /// Number of keys (columns).
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if there are no arrays in the block.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the common axis-0 length of all arrays, or `None` if empty.
    #[inline]
    pub fn nrows(&self) -> Option<usize> {
        self.nrows
    }

    /// Returns true if the Block contains the specified key.
    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Inserts an array under `key`, enforcing consistent axis-0 length.
    ///
    /// This method uses generic dispatch via the `BlockDtype` trait to accept
    /// any supported type (float, int, bool) without requiring users to
    /// manually construct Column enums.
    ///
    /// # Errors
    ///
    /// - Returns `BlockError::RankZero` if the array has rank 0
    /// - Returns `BlockError::RaggedAxis0` if the array's axis-0 length doesn't
    ///   match the Block's existing `nrows`
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs_core::block::Block;
    /// use molrs_core::types::{F, I};
    /// use ndarray::Array1;
    ///
    /// let mut block = Block::new();
    ///
    /// // Insert float array
    /// let arr_float = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
    /// block.insert("x", arr_float).unwrap();
    ///
    /// // Insert int array with same nrows
    /// let arr_int = Array1::from_vec(vec![10 as I, 20 as I]).into_dyn();
    /// block.insert("id", arr_int).unwrap();
    ///
    /// // This would error - different nrows
    /// let arr_bad = Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn();
    /// assert!(block.insert("bad", arr_bad).is_err());
    /// ```
    pub fn insert<T: BlockDtype>(
        &mut self,
        key: impl Into<String>,
        arr: ArrayD<T>,
    ) -> Result<(), BlockError> {
        let key = key.into();
        let shape = arr.shape();

        // Check rank >= 1
        if shape.is_empty() {
            return Err(BlockError::RankZero { key });
        }

        let len0 = shape[0];

        // Check axis-0 consistency
        match self.nrows {
            None => {
                // First insertion defines nrows
                self.nrows = Some(len0);
            }
            Some(expected) => {
                if len0 != expected {
                    return Err(BlockError::RaggedAxis0 {
                        key,
                        expected,
                        got: len0,
                    });
                }
            }
        }

        let col = T::into_column(arr);
        self.map.insert(key, col);
        Ok(())
    }

    /// Insert a pre-built [`Column`] under `key`, validating axis-0 length.
    ///
    /// This is the zero-copy insert path: the caller owns a [`Column`]
    /// (which internally holds an `Arc<ArrayD<T>>`) and hands it over
    /// without unwrapping the Arc. Useful when moving a column between
    /// blocks or re-inserting a clone.
    pub fn insert_column(&mut self, key: impl Into<String>, col: Column) -> Result<(), BlockError> {
        let key = key.into();
        let shape = col.shape();

        if shape.is_empty() {
            return Err(BlockError::RankZero { key });
        }

        let len0 = shape[0];

        match self.nrows {
            None => {
                self.nrows = Some(len0);
            }
            Some(expected) => {
                if len0 != expected {
                    return Err(BlockError::RaggedAxis0 {
                        key,
                        expected,
                        got: len0,
                    });
                }
            }
        }

        self.map.insert(key, col);
        Ok(())
    }

    /// Gets an immutable reference to the column for `key` if present.
    ///
    /// For type-safe access, prefer using `get_float()`, `get_int()`, etc.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&Column> {
        self.map.get(key)
    }

    /// Gets a mutable reference to the column for `key` if present.
    ///
    /// For type-safe access, prefer using `get_float_mut()`, `get_int_mut()`, etc.
    ///
    /// # Warning
    ///
    /// Mutating the column's shape through this reference is allowed but NOT
    /// revalidated. It's the caller's responsibility to maintain axis-0 consistency.
    #[inline]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Column> {
        self.map.get_mut(key)
    }

    // Type-specific getters for the compile-time float scalar.

    /// Gets an immutable reference to a float array for `key` if present and of correct type.
    pub fn get_float(&self, key: &str) -> Option<&ArrayD<crate::types::F>> {
        self.get(key).and_then(|c| c.as_float())
    }

    /// Gets a mutable reference to a float array for `key` if present and of correct type.
    pub fn get_float_mut(&mut self, key: &str) -> Option<&mut ArrayD<crate::types::F>> {
        self.get_mut(key).and_then(|c| c.as_float_mut())
    }

    // Type-specific getters for the compile-time signed integer scalar.

    /// Gets an immutable reference to an int array for `key` if present and of correct type.
    pub fn get_int(&self, key: &str) -> Option<&ArrayD<crate::types::I>> {
        self.get(key).and_then(|c| c.as_int())
    }

    /// Gets a mutable reference to an int array for `key` if present and of correct type.
    pub fn get_int_mut(&mut self, key: &str) -> Option<&mut ArrayD<crate::types::I>> {
        self.get_mut(key).and_then(|c| c.as_int_mut())
    }

    // Type-specific getters for bool

    /// Gets an immutable reference to a bool array for `key` if present and of correct type.
    pub fn get_bool(&self, key: &str) -> Option<&ArrayD<bool>> {
        self.get(key).and_then(|c| c.as_bool())
    }

    /// Gets a mutable reference to a bool array for `key` if present and of correct type.
    pub fn get_bool_mut(&mut self, key: &str) -> Option<&mut ArrayD<bool>> {
        self.get_mut(key).and_then(|c| c.as_bool_mut())
    }

    // Type-specific getters for the compile-time unsigned integer scalar.

    /// Gets an immutable reference to a uint array for `key` if present and of correct type.
    pub fn get_uint(&self, key: &str) -> Option<&ArrayD<crate::types::U>> {
        self.get(key).and_then(|c| c.as_uint())
    }

    /// Gets a mutable reference to a uint array for `key` if present and of correct type.
    pub fn get_uint_mut(&mut self, key: &str) -> Option<&mut ArrayD<crate::types::U>> {
        self.get_mut(key).and_then(|c| c.as_uint_mut())
    }

    // Type-specific getters for u8

    /// Gets an immutable reference to a u8 array for `key` if present and of correct type.
    pub fn get_u8(&self, key: &str) -> Option<&ArrayD<u8>> {
        self.get(key).and_then(|c| c.as_u8())
    }

    /// Gets a mutable reference to a u8 array for `key` if present and of correct type.
    pub fn get_u8_mut(&mut self, key: &str) -> Option<&mut ArrayD<u8>> {
        self.get_mut(key).and_then(|c| c.as_u8_mut())
    }

    // Type-specific getters for String

    /// Gets an immutable reference to a String array for `key` if present and of correct type.
    pub fn get_string(&self, key: &str) -> Option<&ArrayD<String>> {
        self.get(key).and_then(|c| c.as_string())
    }

    /// Gets a mutable reference to a String array for `key` if present and of correct type.
    pub fn get_string_mut(&mut self, key: &str) -> Option<&mut ArrayD<String>> {
        self.get_mut(key).and_then(|c| c.as_string_mut())
    }

    /// Removes and returns the column for `key`, if present.
    ///
    /// If the Block becomes empty after removal, resets `nrows` to `None`.
    pub fn remove(&mut self, key: &str) -> Option<Column> {
        let out = self.map.remove(key);
        if self.map.is_empty() {
            self.nrows = None;
        }
        out
    }

    /// Renames a column from `old_key` to `new_key`.
    ///
    /// Returns `true` if the column was successfully renamed, `false` if `old_key` doesn't exist
    /// or `new_key` already exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs_core::block::Block;
    /// use molrs_core::types::F;
    /// use ndarray::Array1;
    ///
    /// let mut block = Block::new();
    /// block.insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn()).unwrap();
    ///
    /// assert!(block.rename_column("x", "position_x"));
    /// assert!(!block.contains_key("x"));
    /// assert!(block.contains_key("position_x"));
    /// ```
    pub fn rename_column(&mut self, old_key: &str, new_key: &str) -> bool {
        // Check if old_key exists and new_key doesn't exist
        if !self.map.contains_key(old_key) || self.map.contains_key(new_key) {
            return false;
        }

        // Remove the old key and re-insert with new key
        if let Some(column) = self.map.remove(old_key) {
            self.map.insert(new_key.to_string(), column);
            true
        } else {
            false
        }
    }

    /// Clears the Block, removing all keys and resetting `nrows`.
    pub fn clear(&mut self) {
        self.map.clear();
        self.nrows = None;
    }

    /// Returns an iterator over (&str, &Column).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns an iterator over keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(|k| k.as_str())
    }

    /// Returns an iterator over column references.
    pub fn values(&self) -> impl Iterator<Item = &Column> {
        self.map.values()
    }

    /// Returns the data type of the column with the given key, if it exists.
    pub fn dtype(&self, key: &str) -> Option<DType> {
        self.get(key).map(|c| c.dtype())
    }

    /// Resize all columns along axis 0 to `new_nrows`.
    ///
    /// - **Shrink** (`new_nrows` < current): slices each column to keep the first `new_nrows` rows.
    /// - **Grow** (`new_nrows` > current): extends each column with default values
    ///   (0.0 for Float, 0 for Int/UInt/U8, false for Bool, empty string for String).
    /// - **Same size**: no-op, returns `Ok(())`.
    /// - **Empty block** (no columns): sets `nrows` without touching columns.
    ///
    /// Multi-dimensional columns (e.g. Nx3 positions) are resized only along
    /// axis 0; trailing dimensions are preserved.
    ///
    /// # Arguments
    /// * `new_nrows` - The desired number of rows after resize.
    ///
    /// # Returns
    /// * `Ok(())` on success.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs_core::block::Block;
    /// use molrs_core::types::F;
    /// use ndarray::Array1;
    ///
    /// let mut block = Block::new();
    /// block.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn()).unwrap();
    ///
    /// block.resize(4).unwrap();
    /// assert_eq!(block.nrows(), Some(4));
    /// let x = block.get_float("x").unwrap();
    /// assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 0.0, 0.0]);
    /// ```
    pub fn resize(&mut self, new_nrows: usize) -> Result<(), crate::error::MolRsError> {
        if self.is_empty() {
            self.nrows = Some(new_nrows);
            return Ok(());
        }

        let current = self.nrows.unwrap_or(0);
        if new_nrows == current {
            return Ok(());
        }

        for col in self.map.values_mut() {
            col.resize(new_nrows);
        }
        self.nrows = Some(new_nrows);
        Ok(())
    }

    /// Merge another block into this one by concatenating columns along axis-0.
    ///
    /// Both blocks must have the same set of column keys and matching dtypes.
    /// The resulting block will have nrows = self.nrows + other.nrows.
    ///
    /// # Arguments
    /// * `other` - The block to merge into this one
    ///
    /// # Returns
    /// * `Ok(())` if merge succeeds
    /// * `Err(BlockError)` if blocks have incompatible columns
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs_core::block::Block;
    /// use molrs_core::types::F;
    /// use ndarray::Array1;
    ///
    /// let mut block1 = Block::new();
    /// block1.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn()).unwrap();
    ///
    /// let mut block2 = Block::new();
    /// block2.insert("x", Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn()).unwrap();
    ///
    /// block1.merge(&block2).unwrap();
    /// assert_eq!(block1.nrows(), Some(4));
    /// ```
    pub fn merge(&mut self, other: &Block) -> Result<(), BlockError> {
        use ndarray::Axis;
        use ndarray::concatenate;

        // If other is empty, nothing to do
        if other.is_empty() {
            return Ok(());
        }

        // If self is empty, clone other
        if self.is_empty() {
            self.map = other.map.clone();
            self.nrows = other.nrows;
            return Ok(());
        }

        // Check that both blocks have the same keys
        let self_keys: std::collections::HashSet<_> = self.keys().collect();
        let other_keys: std::collections::HashSet<_> = other.keys().collect();

        if self_keys != other_keys {
            return Err(BlockError::validation(format!(
                "Cannot merge blocks with different keys. Self has {:?}, other has {:?}",
                self_keys, other_keys
            )));
        }

        // Merge each column
        let mut new_map = HashMap::new();
        for key in self.keys() {
            let self_col = &self.map[key];
            let other_col = &other.map[key];

            // Check dtype compatibility
            if self_col.dtype() != other_col.dtype() {
                return Err(BlockError::validation(format!(
                    "Column '{}' has incompatible dtypes: {:?} vs {:?}",
                    key,
                    self_col.dtype(),
                    other_col.dtype()
                )));
            }

            // Concatenate based on dtype
            let merged_col = match (self_col, other_col) {
                (Column::Float(a), Column::Float(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate float column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::from_float(merged)
                }
                (Column::Int(a), Column::Int(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate int column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::from_int(merged)
                }
                (Column::UInt(a), Column::UInt(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate uint column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::from_uint(merged)
                }
                (Column::U8(a), Column::U8(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate u8 column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::from_u8(merged)
                }
                (Column::Bool(a), Column::Bool(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate bool column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::from_bool(merged)
                }
                (Column::String(a), Column::String(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate string column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::from_string(merged)
                }
                _ => unreachable!("dtype mismatch already checked"),
            };

            new_map.insert(key.to_string(), merged_col);
        }

        // Update nrows
        let new_nrows = self.nrows.unwrap() + other.nrows.unwrap();
        self.map = new_map;
        self.nrows = Some(new_nrows);

        Ok(())
    }
}

// Index trait for convenient access: block["key"]
impl Index<&str> for Block {
    type Output = Column;

    fn index(&self, key: &str) -> &Self::Output {
        self.get(key)
            .unwrap_or_else(|| panic!("key '{}' not found in Block", key))
    }
}

impl IndexMut<&str> for Block {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.get_mut(key)
            .unwrap_or_else(|| panic!("key '{}' not found in Block", key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{F, I};
    use ndarray::Array1;

    #[test]
    fn test_insert_mixed_dtypes() {
        let mut block = Block::new();

        let arr_float = Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn();
        let arr_float_2 = Array1::from_vec(vec![4.0 as F, 5.0 as F, 6.0 as F]).into_dyn();
        let arr_i64 = Array1::from_vec(vec![10 as I, 20, 30]).into_dyn();
        let arr_bool = Array1::from_vec(vec![true, false, true]).into_dyn();

        assert!(block.insert("x", arr_float).is_ok());
        assert!(block.insert("y", arr_float_2).is_ok());
        assert!(block.insert("id", arr_i64).is_ok());
        assert!(block.insert("mask", arr_bool).is_ok());

        assert_eq!(block.len(), 4);
        assert_eq!(block.nrows(), Some(3));
    }

    #[test]
    fn test_axis0_mismatch_error() {
        let mut block = Block::new();

        let arr1 = Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn();
        let arr2 = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();

        block.insert("x", arr1).unwrap();
        let result = block.insert("y", arr2);

        assert!(result.is_err());
        match result {
            Err(BlockError::RaggedAxis0 { expected, got, .. }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("Expected RaggedAxis0 error"),
        }
    }

    #[test]
    fn test_typed_getters() {
        let mut block = Block::new();

        let arr_float = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        let arr_i64 = Array1::from_vec(vec![10 as I, 20]).into_dyn();

        block.insert("x", arr_float).unwrap();
        block.insert("id", arr_i64).unwrap();

        // Correct type access
        assert!(block.get_float("x").is_some());
        assert!(block.get_int("id").is_some());

        // Wrong type access returns None
        assert!(block.get_int("x").is_none());
        assert!(block.get_float("id").is_none());

        // Mutable access
        if let Some(x_mut) = block.get_float_mut("x") {
            x_mut[[0]] = 99.0;
        }
        assert_eq!(block.get_float("x").unwrap()[[0]], 99.0);
    }

    #[test]
    fn test_index_access() {
        let mut block = Block::new();

        let arr = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        block.insert("x", arr).unwrap();

        // Immutable index
        let col = &block["x"];
        assert_eq!(col.dtype(), DType::Float);

        // Mutable index
        let col_mut = &mut block["x"];
        if let Some(arr_mut) = col_mut.as_float_mut() {
            arr_mut[[0]] = 42.0;
        }
        assert_eq!(block.get_float("x").unwrap()[[0]], 42.0);
    }

    #[test]
    #[should_panic(expected = "key 'missing' not found")]
    fn test_index_panic_on_missing_key() {
        let block = Block::new();
        let _ = &block["missing"];
    }

    #[test]
    fn test_remove_resets_nrows() {
        let mut block = Block::new();

        let arr = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        block.insert("x", arr).unwrap();

        assert_eq!(block.nrows(), Some(2));

        block.remove("x");
        assert_eq!(block.nrows(), None);
        assert!(block.is_empty());
    }

    #[test]
    fn test_iter_keys_values() {
        let mut block = Block::new();

        let arr1 = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        let arr2 = Array1::from_vec(vec![10 as I, 20]).into_dyn();

        block.insert("x", arr1).unwrap();
        block.insert("id", arr2).unwrap();

        let keys: Vec<&str> = block.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"x"));
        assert!(keys.contains(&"id"));

        let dtypes: Vec<DType> = block.values().map(|c| c.dtype()).collect();
        assert!(dtypes.contains(&DType::Float));
        assert!(dtypes.contains(&DType::Int));
    }

    #[test]
    fn test_rank_zero_error() {
        let mut block = Block::new();

        // Create a rank-0 array (scalar)
        let arr = ArrayD::<F>::zeros(vec![]);

        let result = block.insert("scalar", arr);
        assert!(result.is_err());
        match result {
            Err(BlockError::RankZero { key }) => {
                assert_eq!(key, "scalar");
            }
            _ => panic!("Expected RankZero error"),
        }
    }

    #[test]
    fn test_dtype_query() {
        let mut block = Block::new();

        let arr_float = Array1::from_vec(vec![1.0 as F]).into_dyn();
        let arr_i64 = Array1::from_vec(vec![10 as I]).into_dyn();

        block.insert("x", arr_float).unwrap();
        block.insert("id", arr_i64).unwrap();

        assert_eq!(block.dtype("x"), Some(DType::Float));
        assert_eq!(block.dtype("id"), Some(DType::Int));
        assert_eq!(block.dtype("missing"), None);
    }

    #[test]
    fn test_merge_basic() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        let arr1 = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        let arr2 = Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn();

        block1.insert("x", arr1).unwrap();
        block2.insert("x", arr2).unwrap();

        block1.merge(&block2).unwrap();

        assert_eq!(block1.nrows(), Some(4));
        let x = block1.get_float("x").unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_merge_empty_blocks() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        // Merge empty into empty
        block1.merge(&block2).unwrap();
        assert_eq!(block1.nrows(), None);

        // Merge non-empty into empty
        let arr = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        block2.insert("x", arr).unwrap();
        block1.merge(&block2).unwrap();
        assert_eq!(block1.nrows(), Some(2));

        // Merge empty into non-empty
        let block3 = Block::new();
        block1.merge(&block3).unwrap();
        assert_eq!(block1.nrows(), Some(2));
    }

    #[test]
    fn test_merge_incompatible_keys() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        let arr1 = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        let arr2 = Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn();

        block1.insert("x", arr1).unwrap();
        block2.insert("y", arr2).unwrap();

        let result = block1.merge(&block2);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_incompatible_dtypes() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        let arr1 = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        let arr2 = Array1::from_vec(vec![3 as I, 4]).into_dyn();

        block1.insert("x", arr1).unwrap();
        block2.insert("x", arr2).unwrap();

        let result = block1.merge(&block2);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_multiple_columns() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        block1
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn())
            .unwrap();
        block1
            .insert("id", Array1::from_vec(vec![10 as I, 20]).into_dyn())
            .unwrap();

        block2
            .insert("x", Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn())
            .unwrap();
        block2
            .insert("id", Array1::from_vec(vec![30 as I, 40]).into_dyn())
            .unwrap();

        block1.merge(&block2).unwrap();

        assert_eq!(block1.nrows(), Some(4));
        let x = block1.get_float("x").unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        let id = block1.get_int("id").unwrap();
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20, 30, 40]);
    }

    #[test]
    fn test_rename_column() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn())
            .unwrap();
        block
            .insert("y", Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn())
            .unwrap();

        // Successful rename
        assert!(block.rename_column("x", "position_x"));
        assert!(!block.contains_key("x"));
        assert!(block.contains_key("position_x"));
        assert_eq!(
            block
                .get_float("position_x")
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[1.0, 2.0]
        );

        // Try to rename non-existent column
        assert!(!block.rename_column("nonexistent", "new_name"));

        // Try to rename to existing column name
        assert!(!block.rename_column("position_x", "y"));
    }

    #[test]
    fn test_resize_shrink() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0, 3.0, 4.0]).into_dyn(),
            )
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![10 as I, 20, 30, 40]).into_dyn())
            .unwrap();

        block.resize(2).unwrap();

        assert_eq!(block.nrows(), Some(2));
        let x = block.get_float("x").unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0]);
        let id = block.get_int("id").unwrap();
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20]);
    }

    #[test]
    fn test_resize_grow() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn())
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![10 as I, 20]).into_dyn())
            .unwrap();

        block.resize(4).unwrap();

        assert_eq!(block.nrows(), Some(4));
        let x = block.get_float("x").unwrap();
        // Original data preserved, new rows are 0.0
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 0.0, 0.0]);
        let id = block.get_int("id").unwrap();
        // Original data preserved, new rows are 0
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20, 0, 0]);
    }

    #[test]
    fn test_resize_same() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn())
            .unwrap();

        block.resize(3).unwrap();

        assert_eq!(block.nrows(), Some(3));
        let x = block.get_float("x").unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_resize_empty() {
        let mut block = Block::new();

        block.resize(5).unwrap();
        assert_eq!(block.nrows(), Some(5));
        assert!(block.is_empty());
    }

    #[test]
    fn test_resize_multidim() {
        use ndarray::Array2;

        let mut block = Block::new();
        // 4x3 position array
        let pos = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0 as F, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap()
        .into_dyn();
        block.insert("pos", pos).unwrap();

        // Shrink 4x3 -> 2x3
        block.resize(2).unwrap();
        assert_eq!(block.nrows(), Some(2));
        let pos = block.get_float("pos").unwrap();
        assert_eq!(pos.shape(), &[2, 3]);
        assert_eq!(
            pos.as_slice_memory_order().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        // Grow 2x3 -> 5x3
        block.resize(5).unwrap();
        assert_eq!(block.nrows(), Some(5));
        let pos = block.get_float("pos").unwrap();
        assert_eq!(pos.shape(), &[5, 3]);
        // Original data followed by zeros
        assert_eq!(
            pos.as_slice_memory_order().unwrap(),
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
    }

    #[test]
    fn test_resize_mixed_dtypes() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn())
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![10 as I, 20, 30]).into_dyn())
            .unwrap();
        block
            .insert("mask", Array1::from_vec(vec![true, false, true]).into_dyn())
            .unwrap();
        block
            .insert(
                "name",
                Array1::from_vec(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                    .into_dyn(),
            )
            .unwrap();

        // Grow from 3 to 5
        block.resize(5).unwrap();
        assert_eq!(block.nrows(), Some(5));

        let x = block.get_float("x").unwrap();
        assert_eq!(
            x.as_slice_memory_order().unwrap(),
            &[1.0, 2.0, 3.0, 0.0, 0.0]
        );
        let id = block.get_int("id").unwrap();
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20, 30, 0, 0]);
        let mask = block.get_bool("mask").unwrap();
        assert_eq!(
            mask.as_slice_memory_order().unwrap(),
            &[true, false, true, false, false]
        );
        let name = block.get_string("name").unwrap();
        assert_eq!(name[[0]], "a");
        assert_eq!(name[[1]], "b");
        assert_eq!(name[[2]], "c");
        assert_eq!(name[[3]], "");
        assert_eq!(name[[4]], "");

        // Shrink from 5 to 2
        block.resize(2).unwrap();
        assert_eq!(block.nrows(), Some(2));

        let x = block.get_float("x").unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0]);
        let id = block.get_int("id").unwrap();
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20]);
        let mask = block.get_bool("mask").unwrap();
        assert_eq!(mask.as_slice_memory_order().unwrap(), &[true, false]);
        let name = block.get_string("name").unwrap();
        assert_eq!(name[[0]], "a");
        assert_eq!(name[[1]], "b");
    }
}
