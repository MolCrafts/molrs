//! Frame: a dictionary mapping string keys to heterogeneous [`Block`]s.
//!
//! A Frame groups multiple [`Block`]s under string keys. Each `Block` may contain
//! heterogeneous columns (different scalar dtypes like f32, f64, i64, bool), and
//! manages its own `nrows` invariant. `Frame` itself only manages the mapping from
//! names to blocks and does **not** enforce cross-block axis-0 consistency.
//!
//! # Examples
//!
//! ```
//! use molrs::frame::Frame;
//! use molrs::block::Block;
//! use molrs::types::{F, I};
//! use ndarray::Array1;
//!
//! let mut frame = Frame::new();
//!
//! // Create an atoms block
//! let mut atoms = Block::new();
//! atoms.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn()).unwrap();
//! atoms.insert("y", Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn()).unwrap();
//! atoms.insert("id", Array1::from_vec(vec![1 as I, 2 as I, 3 as I]).into_dyn()).unwrap();
//!
//! frame.insert("atoms", atoms);
//!
//! // Access via Index trait
//! let atoms_ref = &frame["atoms"];
//! assert_eq!(atoms_ref.nrows(), Some(3));
//!
//! // Add metadata
//! frame.meta.insert("title".into(), "My Molecule".into());
//! ```

use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use super::block::Block;
use super::region::simbox::SimBox;
use crate::error::MolRsError;
use crate::grid::Grid;

/// A dictionary from string keys to [`Block`]s.
///
/// Frame provides a simple container for organizing multiple blocks of data,
/// typically representing different aspects of a molecular system (e.g., atoms,
/// bonds, velocities). Each block can have different numbers of rows and different
/// column types.
#[derive(Default, Clone)]
pub struct Frame {
    map: HashMap<String, Block>,
    grids: HashMap<String, Grid>,
    /// Arbitrary key-value metadata associated with the frame.
    pub meta: HashMap<String, String>,
    /// Simulation box defining periodic boundary conditions.
    pub simbox: Option<SimBox>,
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Frame");

        // Format blocks as a map of name -> (nrows, ncols)
        let mut blocks_map = std::collections::BTreeMap::new();
        for (k, b) in &self.map {
            blocks_map.insert(k.as_str(), (b.nrows(), b.len()));
        }
        debug_struct.field("blocks", &blocks_map);

        // Show metadata if non-empty
        if !self.meta.is_empty() {
            debug_struct.field("meta", &self.meta);
        }
        if !self.grids.is_empty() {
            let mut grid_names: Vec<_> = self.grids.keys().cloned().collect();
            grid_names.sort();
            debug_struct.field("grids", &grid_names);
        }

        debug_struct.finish()
    }
}

impl Frame {
    /// Creates an empty Frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    ///
    /// let frame = Frame::new();
    /// assert!(frame.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            grids: HashMap::new(),
            meta: HashMap::new(),
            simbox: None,
        }
    }

    /// Creates an empty Frame with the specified capacity for blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    ///
    /// let frame = Frame::with_capacity(10);
    /// assert!(frame.is_empty());
    /// ```
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: HashMap::with_capacity(cap),
            grids: HashMap::new(),
            meta: HashMap::new(),
            simbox: None,
        }
    }

    /// Creates a Frame from an existing HashMap of blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("atoms".to_string(), Block::new());
    ///
    /// let frame = Frame::from_map(map);
    /// assert_eq!(frame.len(), 1);
    /// ```
    pub fn from_map(map: HashMap<String, Block>) -> Self {
        Self {
            map,
            grids: HashMap::new(),
            meta: HashMap::new(),
            simbox: None,
        }
    }

    /// Consumes the Frame and returns the inner HashMap of blocks, grids, metadata, and simbox.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    ///
    /// let mut frame = Frame::new();
    /// frame.insert("atoms", Block::new());
    /// frame.meta.insert("title".into(), "Test".into());
    ///
    /// let (blocks, grids, meta, simbox) = frame.into_inner();
    /// assert_eq!(blocks.len(), 1);
    /// assert!(grids.is_empty());
    /// assert_eq!(meta.get("title").unwrap(), "Test");
    /// assert!(simbox.is_none());
    /// ```
    pub fn into_inner(
        self,
    ) -> (
        HashMap<String, Block>,
        HashMap<String, Grid>,
        HashMap<String, String>,
        Option<SimBox>,
    ) {
        (self.map, self.grids, self.meta, self.simbox)
    }

    /// Number of blocks (keys) in the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    ///
    /// let mut frame = Frame::new();
    /// assert_eq!(frame.len(), 0);
    ///
    /// frame.insert("atoms", Block::new());
    /// assert_eq!(frame.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if the frame contains no blocks.
    ///
    /// Note: This only checks blocks, not metadata.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns true if the frame contains the specified key.
    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Gets an immutable reference to the block for `key` if present.
    ///
    /// For a panicking version, use the `Index` trait: `&frame["key"]`.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&Block> {
        self.map.get(key)
    }

    /// Gets a mutable reference to the block for `key` if present.
    ///
    /// For a panicking version, use the `IndexMut` trait: `&mut frame["key"]`.
    #[inline]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Block> {
        self.map.get_mut(key)
    }

    /// Inserts a block under `key`. Returns the previous block if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    ///
    /// let mut frame = Frame::new();
    /// let old = frame.insert("atoms", Block::new());
    /// assert!(old.is_none());
    ///
    /// let old = frame.insert("atoms", Block::new());
    /// assert!(old.is_some());
    /// ```
    pub fn insert(&mut self, key: impl Into<String>, block: Block) -> Option<Block> {
        self.map.insert(key.into(), block)
    }

    /// Removes and returns the block for `key`, if present.
    pub fn remove(&mut self, key: &str) -> Option<Block> {
        self.map.remove(key)
    }

    /// Clears the frame, removing all blocks.
    ///
    /// **Note**: This does NOT clear metadata. Use `clear_all()` to clear both.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    ///
    /// let mut frame = Frame::new();
    /// frame.insert("atoms", Block::new());
    /// frame.meta.insert("title".into(), "Test".into());
    ///
    /// frame.clear();
    /// assert!(frame.is_empty());
    /// assert!(!frame.meta.is_empty()); // metadata preserved
    /// ```
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Clears both blocks and metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    ///
    /// let mut frame = Frame::new();
    /// frame.insert("atoms", Block::new());
    /// frame.meta.insert("title".into(), "Test".into());
    ///
    /// frame.clear_all();
    /// assert!(frame.is_empty());
    /// assert!(frame.meta.is_empty());
    /// ```
    pub fn clear_all(&mut self) {
        self.map.clear();
        self.grids.clear();
        self.meta.clear();
        self.simbox = None;
    }

    /// Insert or replace a named grid.
    pub fn insert_grid(&mut self, name: impl Into<String>, grid: Grid) -> Option<Grid> {
        self.grids.insert(name.into(), grid)
    }

    /// Remove a named grid.
    pub fn remove_grid(&mut self, name: &str) -> Option<Grid> {
        self.grids.remove(name)
    }

    /// Borrow a grid by name.
    pub fn get_grid(&self, name: &str) -> Option<&Grid> {
        self.grids.get(name)
    }

    /// Borrow a grid mutably by name.
    pub fn get_grid_mut(&mut self, name: &str) -> Option<&mut Grid> {
        self.grids.get_mut(name)
    }

    /// Returns true if the frame contains a named grid.
    pub fn has_grid(&self, name: &str) -> bool {
        self.grids.contains_key(name)
    }

    /// Returns an iterator over `(name, grid)` pairs.
    pub fn grids(&self) -> impl Iterator<Item = (&str, &Grid)> {
        self.grids.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns an iterator over grid names.
    pub fn grid_keys(&self) -> impl Iterator<Item = &str> {
        self.grids.keys().map(|k| k.as_str())
    }

    /// Renames a column in the specified block.
    ///
    /// Returns `true` if the column was successfully renamed, `false` if the block doesn't exist,
    /// the old column key doesn't exist, or the new column key already exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    /// use molrs::types::F;
    /// use ndarray::Array1;
    ///
    /// let mut frame = Frame::new();
    /// let mut atoms = Block::new();
    /// atoms.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn()).unwrap();
    /// frame.insert("atoms", atoms);
    ///
    /// assert!(frame.rename_column("atoms", "x", "position_x"));
    /// assert!(!frame["atoms"].contains_key("x"));
    /// assert!(frame["atoms"].contains_key("position_x"));
    /// ```
    pub fn rename_column(&mut self, block_key: &str, old_col_key: &str, new_col_key: &str) -> bool {
        if let Some(block) = self.map.get_mut(block_key) {
            block.rename_column(old_col_key, new_col_key)
        } else {
            false
        }
    }

    /// Renames a block in the frame.
    ///
    /// Returns `true` if the block was successfully renamed, `false` if the old block
    /// doesn't exist or the new block name already exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    /// use molrs::types::F;
    /// use ndarray::Array1;
    ///
    /// let mut frame = Frame::new();
    /// let mut atoms = Block::new();
    /// atoms.insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn()).unwrap();
    /// frame.insert("atoms", atoms);
    ///
    /// assert!(frame.rename_block("atoms", "molecules"));
    /// assert!(!frame.contains_key("atoms"));
    /// assert!(frame.contains_key("molecules"));
    /// ```
    pub fn rename_block(&mut self, old_key: &str, new_key: &str) -> bool {
        // Check if old_key exists and new_key doesn't exist
        if !self.map.contains_key(old_key) || self.map.contains_key(new_key) {
            return false;
        }

        // Remove the old key and re-insert with new key
        if let Some(block) = self.map.remove(old_key) {
            self.map.insert(new_key.to_string(), block);
            true
        } else {
            false
        }
    }

    /// Returns an iterator over (&str, &Block).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Block)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns a mutable iterator over (&str, &mut Block).
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    /// use molrs::types::F;
    /// use ndarray::Array1;
    ///
    /// let mut frame = Frame::new();
    /// let mut atoms = Block::new();
    /// atoms.insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn()).unwrap();
    /// frame.insert("atoms", atoms);
    ///
    /// for (_name, block) in frame.iter_mut() {
    ///     // Can mutate blocks
    ///     if let Some(x) = block.get_float_mut("x") {
    ///         x[[0]] = 99.0 as F;
    ///     }
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut Block)> {
        self.map.iter_mut().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns an iterator over keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(|k| k.as_str())
    }

    /// Returns an iterator over block references.
    pub fn values(&self) -> impl Iterator<Item = &Block> {
        self.map.values()
    }

    /// Returns a mutable iterator over block references.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Block> {
        self.map.values_mut()
    }

    /// Validates cross-block consistency.
    ///
    /// This method checks for common consistency issues:
    /// - All blocks with "atoms" prefix should have the same nrows
    /// - Bond indices (if present) should reference valid atoms
    ///
    /// # Returns
    /// - `Ok(())` if validation passes
    /// - `Err(MolRsError::Validation)` with details if validation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    /// use molrs::types::F;
    /// use ndarray::Array1;
    ///
    /// let mut frame = Frame::new();
    /// let mut atoms = Block::new();
    /// atoms.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn()).unwrap();
    /// atoms.insert("y", Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn()).unwrap();
    /// frame.insert("atoms", atoms);
    ///
    /// assert!(frame.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), MolRsError> {
        // Check atoms blocks have consistent nrows
        let atoms_blocks: Vec<_> = self
            .map
            .iter()
            .filter(|(k, _)| k.starts_with("atoms"))
            .collect();

        if !atoms_blocks.is_empty() {
            let first_nrows = atoms_blocks[0].1.nrows();
            for (key, block) in &atoms_blocks {
                if block.nrows() != first_nrows {
                    return Err(MolRsError::validation(format!(
                        "Inconsistent atom block sizes: '{}' has {:?} rows but expected {:?}",
                        key,
                        block.nrows(),
                        first_nrows
                    )));
                }
            }
        }

        // Check bond indices if bonds block exists
        if let Some(bonds) = self.get("bonds")
            && let Some(atoms) = self.get("atoms")
        {
            let natoms = atoms.nrows().unwrap_or(0);

            // Check atomi indices
            if let Some(i_col) = bonds.get_uint("atomi") {
                for &idx in i_col.iter() {
                    if idx as usize >= natoms {
                        return Err(MolRsError::validation(format!(
                            "Bond atomi index {} out of range [0, {})",
                            idx, natoms
                        )));
                    }
                }
            }

            // Check atomj indices
            if let Some(j_col) = bonds.get_uint("atomj") {
                for &idx in j_col.iter() {
                    if idx as usize >= natoms {
                        return Err(MolRsError::validation(format!(
                            "Bond atomj index {} out of range [0, {})",
                            idx, natoms
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Checks if the frame is consistent without returning an error.
    ///
    /// This is a non-panicking version of `validate()` that returns a boolean.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::frame::Frame;
    /// use molrs::block::Block;
    ///
    /// let frame = Frame::new();
    /// assert!(frame.is_consistent());
    /// ```
    pub fn is_consistent(&self) -> bool {
        self.validate().is_ok()
    }
}

// Index trait for convenient access: frame["atoms"]
impl Index<&str> for Frame {
    type Output = Block;

    fn index(&self, key: &str) -> &Self::Output {
        self.get(key)
            .unwrap_or_else(|| panic!("Frame does not contain block '{}'", key))
    }
}

impl IndexMut<&str> for Frame {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.get_mut(key)
            .unwrap_or_else(|| panic!("Frame does not contain block '{}'", key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{F, I};
    use ndarray::Array1;

    #[test]
    fn test_frame_new() {
        let frame = Frame::new();
        assert!(frame.is_empty());
        assert_eq!(frame.len(), 0);
    }

    #[test]
    fn test_frame_insert_get() {
        let mut frame = Frame::new();
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn())
            .unwrap();

        frame.insert("atoms", block);
        assert_eq!(frame.len(), 1);
        assert!(frame.contains_key("atoms"));

        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(2));
    }

    #[test]
    fn test_frame_index_access() {
        let mut frame = Frame::new();
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn())
            .unwrap();
        frame.insert("atoms", block);

        // Immutable index
        let atoms = &frame["atoms"];
        assert_eq!(atoms.nrows(), Some(1));

        // Mutable index
        let atoms_mut = &mut frame["atoms"];
        if let Some(x) = atoms_mut.get_float_mut("x") {
            x[[0]] = 99.0;
        }
        assert_eq!(frame["atoms"].get_float("x").unwrap()[[0]], 99.0);
    }

    #[test]
    #[should_panic(expected = "Frame does not contain block 'missing'")]
    fn test_frame_index_panic() {
        let frame = Frame::new();
        let _ = &frame["missing"];
    }

    #[test]
    fn test_frame_iter() {
        let mut frame = Frame::new();
        frame.insert("atoms", Block::new());
        frame.insert("bonds", Block::new());

        let keys: Vec<&str> = frame.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"atoms"));
        assert!(keys.contains(&"bonds"));

        let mut count = 0;
        for (_name, _block) in frame.iter() {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_frame_iter_mut() {
        let mut frame = Frame::new();
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn())
            .unwrap();
        frame.insert("atoms", block);

        for (_name, block) in frame.iter_mut() {
            if let Some(x) = block.get_float_mut("x") {
                x[[0]] = 42.0;
            }
        }

        assert_eq!(frame["atoms"].get_float("x").unwrap()[[0]], 42.0);
    }

    #[test]
    fn test_frame_values_mut() {
        let mut frame = Frame::new();
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn())
            .unwrap();
        frame.insert("atoms", block);

        for block in frame.values_mut() {
            if let Some(x) = block.get_float_mut("x") {
                x[[0]] = 77.0;
            }
        }

        assert_eq!(frame["atoms"].get_float("x").unwrap()[[0]], 77.0);
    }

    #[test]
    fn test_frame_from_map() {
        let mut map = HashMap::new();
        map.insert("atoms".to_string(), Block::new());
        map.insert("bonds".to_string(), Block::new());

        let frame = Frame::from_map(map);
        assert_eq!(frame.len(), 2);
        assert!(frame.contains_key("atoms"));
        assert!(frame.contains_key("bonds"));
    }

    #[test]
    fn test_frame_into_inner() {
        let mut frame = Frame::new();
        frame.insert("atoms", Block::new());
        frame.meta.insert("title".into(), "Test".into());

        let (blocks, grids, meta, simbox) = frame.into_inner();
        assert_eq!(blocks.len(), 1);
        assert!(grids.is_empty());
        assert!(blocks.contains_key("atoms"));
        assert_eq!(meta.get("title").unwrap(), "Test");
        assert!(simbox.is_none());
    }

    #[test]
    fn test_frame_clear_preserves_meta() {
        let mut frame = Frame::new();
        frame.insert("atoms", Block::new());
        frame.meta.insert("title".into(), "Test".into());

        frame.clear();
        assert!(frame.is_empty());
        assert!(!frame.meta.is_empty());
        assert_eq!(frame.meta.get("title").unwrap(), "Test");
    }

    #[test]
    fn test_frame_clear_all() {
        let mut frame = Frame::new();
        frame.insert("atoms", Block::new());
        frame.meta.insert("title".into(), "Test".into());

        frame.clear_all();
        assert!(frame.is_empty());
        assert!(frame.meta.is_empty());
    }

    #[test]
    fn test_frame_debug() {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn(),
            )
            .unwrap();
        frame.insert("atoms", atoms);
        frame.meta.insert("title".into(), "Test".into());

        let debug_str = format!("{:?}", frame);
        assert!(debug_str.contains("Frame"));
        assert!(debug_str.contains("atoms"));
        assert!(debug_str.contains("title"));
    }

    #[test]
    fn test_rename_column() {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);

        // Successful rename
        assert!(frame.rename_column("atoms", "x", "position_x"));
        assert!(!frame["atoms"].contains_key("x"));
        assert!(frame["atoms"].contains_key("position_x"));
        assert_eq!(
            frame["atoms"]
                .get_float("position_x")
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[1.0, 2.0]
        );

        // Try to rename in non-existent block
        assert!(!frame.rename_column("nonexistent", "x", "new_x"));

        // Try to rename non-existent column
        assert!(!frame.rename_column("atoms", "nonexistent", "new_name"));
    }

    #[test]
    fn test_rename_block() {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);

        // Successful rename
        assert!(frame.rename_block("atoms", "molecules"));
        assert!(!frame.contains_key("atoms"));
        assert!(frame.contains_key("molecules"));
        assert_eq!(
            frame["molecules"]
                .get_float("x")
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[1.0, 2.0]
        );

        // Try to rename non-existent block
        assert!(!frame.rename_block("nonexistent", "new_block"));

        // Try to rename to existing block name
        let mut bonds = Block::new();
        bonds
            .insert("type", Array1::from_vec(vec![1 as I]).into_dyn())
            .unwrap();
        frame.insert("bonds", bonds);
        assert!(!frame.rename_block("molecules", "bonds"));
    }
}
