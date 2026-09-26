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
//! use molrs::store::frame::Frame;
//! use molrs::store::block::Block;
//! use molrs::types::{F, Idx};
//! use ndarray::Array1;
//!
//! let mut frame = Frame::new();
//!
//! // Create an atoms block
//! let mut atoms = Block::new();
//! atoms.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn()).unwrap();
//! atoms.insert("y", Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn()).unwrap();
//! atoms.insert("id", Array1::from_vec(vec![1 as Idx, 2 as Idx, 3 as Idx]).into_dyn()).unwrap();
//!
//! frame.insert("atoms", atoms);
//!
//! // Access via Index trait
//! let atoms_ref = &frame["atoms"];
//! assert_eq!(atoms_ref.nrows(), Some(3));
//!
//! // Add metadata
//! frame.meta.insert("title", "My Molecule");
//! ```

use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use crate::error::MolRsError;
use crate::spatial::simbox::SimBox;
use crate::store::block::{Block, DType};
use crate::store::meta::MetaMap;
use crate::store::schema::ColumnDim;
use crate::types::F;
use crate::units::preset::PresetDim;
use crate::units::{UnitPreset, UnitRegistry, UnitsError};

/// Exact schema version of serialized frames and per-frame Zarr groups.
pub const FRAME_SCHEMA_VERSION: u32 = 2;

/// A dictionary from string keys to [`Block`]s.
///
/// Frame provides a simple container for organizing multiple blocks of data,
/// typically representing different aspects of a molecular system (e.g., atoms,
/// bonds, velocities). Each block can have different numbers of rows and different
/// column types.
#[derive(Default, Clone)]
pub struct Frame {
    map: HashMap<String, Block>,
    /// Exact-dtype metadata associated with the frame.
    pub meta: MetaMap,
    /// Simulation box defining periodic boundary conditions.
    pub simbox: Option<SimBox>,
}

/// Type alias for the result of into_inner().
type IntoInnerResult = (HashMap<String, Block>, MetaMap, Option<SimBox>);

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

        debug_struct.finish()
    }
}

impl Frame {
    /// Creates an empty Frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::store::frame::Frame;
    ///
    /// let frame = Frame::new();
    /// assert!(frame.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            meta: MetaMap::new(),
            simbox: None,
        }
    }

    /// Creates an empty Frame with the specified capacity for blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::store::frame::Frame;
    ///
    /// let frame = Frame::with_capacity(10);
    /// assert!(frame.is_empty());
    /// ```
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: HashMap::with_capacity(cap),
            meta: MetaMap::new(),
            simbox: None,
        }
    }

    /// Creates a Frame from an existing HashMap of blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
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
            meta: MetaMap::new(),
            simbox: None,
        }
    }

    /// Consumes the Frame and returns the inner HashMap of blocks, metadata, and simbox.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
    ///
    /// let mut frame = Frame::new();
    /// frame.insert("atoms", Block::new());
    /// frame.meta.insert("title", "Test");
    ///
    /// let (blocks, meta, simbox) = frame.into_inner();
    /// assert_eq!(blocks.len(), 1);
    /// assert_eq!(meta.get("title").unwrap().as_str(), Some("Test"));
    /// assert!(simbox.is_none());
    /// ```
    pub fn into_inner(self) -> IntoInnerResult {
        (self.map, self.meta, self.simbox)
    }

    /// Number of blocks (keys) in the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
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
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
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
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
    ///
    /// let mut frame = Frame::new();
    /// frame.insert("atoms", Block::new());
    /// frame.meta.insert("title", "Test");
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
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
    ///
    /// let mut frame = Frame::new();
    /// frame.insert("atoms", Block::new());
    /// frame.meta.insert("title", "Test");
    ///
    /// frame.clear_all();
    /// assert!(frame.is_empty());
    /// assert!(frame.meta.is_empty());
    /// ```
    pub fn clear_all(&mut self) {
        self.map.clear();
        self.meta.clear();
        self.simbox = None;
    }

    /// Renames a column in the specified block.
    ///
    /// Returns `true` if the column was successfully renamed, `false` if the block doesn't exist,
    /// the old column key doesn't exist, or the new column key already exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
    /// use molrs::types::F;
    /// use ndarray::Array1;
    ///
    /// let mut frame = Frame::new();
    /// let mut atoms = Block::new();
    /// atoms.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn()).unwrap();
    /// frame.insert("atoms", atoms);
    ///
    /// frame.rename_column("atoms", "x", "position_x").unwrap();
    /// assert!(!frame["atoms"].contains_key("x"));
    /// assert!(frame["atoms"].contains_key("position_x"));
    /// ```
    pub fn rename_column(
        &mut self,
        block_key: &str,
        old_col_key: &str,
        new_col_key: &str,
    ) -> Result<(), crate::store::block::BlockError> {
        match self.map.get_mut(block_key) {
            Some(block) => block.rename_column(old_col_key, new_col_key),
            None => Err(crate::store::block::BlockError::Validation {
                message: format!("cannot rename: no block '{block_key}'"),
            }),
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
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
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
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
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

    /// Judge this frame against the canonical Frame schema.
    ///
    /// Delegates entirely to [`crate::store::schema::Validator::canonical`] —
    /// this method does **not** re-implement dtype, shape, required-column, or
    /// endpoint-range checks. Callers that need the full report (every
    /// violation at once) should use the `Validator` directly; this is the
    /// short form for the common "is this frame well-formed?" gate.
    ///
    /// # Returns
    /// - `Ok(())` if the frame conforms
    /// - `Err(MolRsError::Validation)` with the full report text otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
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
        crate::store::schema::Validator::canonical()
            .validate(self)
            .map_err(|report| MolRsError::validation(report.to_string()))
    }

    /// Checks if the frame is consistent without returning an error.
    ///
    /// This is a non-panicking version of `validate()` that returns a boolean.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::store::frame::Frame;
    /// use molrs::store::block::Block;
    ///
    /// let frame = Frame::new();
    /// assert!(frame.is_consistent());
    /// ```
    pub fn is_consistent(&self) -> bool {
        self.validate().is_ok()
    }

    /// Convert every physical column and the box from preset `from` to
    /// preset `to`, in place.
    ///
    /// The factor of each Float column comes from its key's
    /// [`ColumnSpec::dimension`](crate::store::schema::ColumnSpec::dimension):
    ///
    /// - `Of(d)` — `registry.parse(from.unit(d))` to `registry.parse(to.unit(d))`;
    /// - `Product(a, b)` — the product of the two factors;
    /// - `Dimensionless` — untouched.
    ///
    /// Int, UInt, String and Bool columns are never converted. The
    /// [`SimBox`] cell matrix and origin scale by the length factor (a box
    /// with no defined cell keeps its placeholder matrix; only its origin
    /// scales).
    ///
    /// `meta[`[`keys::UNITS`](crate::store::keys::UNITS)`]`, when present,
    /// must name `from`; after success it names `to`. An absent key is taken
    /// to mean `from` and is set to `to`.
    ///
    /// The conversion is atomic: every factor and the meta check are
    /// resolved before anything is written, so on refusal the frame is
    /// unchanged.
    ///
    /// # Errors
    ///
    /// [`UnitsError::Unconvertible`] naming `block.key`, `simbox` or
    /// `meta.units` when:
    ///
    /// - a Float column's key is undeclared in the schema, or declared
    ///   [`NotAQuantity`](ColumnDim::NotAQuantity);
    /// - a column stored as `Float16`, `Float32`, `Complex64` or `Complex128`
    ///   has a key that is not declared
    ///   [`Dimensionless`](ColumnDim::Dimensionless) — only `Float` storage
    ///   is scaled;
    /// - a dimension's unit does not parse in `registry` — e.g. `lj_mass`
    ///   in a registry built with only
    ///   [`define_lj_sigma`](UnitRegistry::define_lj_sigma), which is how a
    ///   σ-only conversion refuses mass, charge and velocity;
    /// - the scaled box is not a valid cell;
    /// - `meta.units` is not a string equal to `from.name()`.
    pub fn convert_units(
        &mut self,
        registry: &UnitRegistry,
        from: &UnitPreset,
        to: &UnitPreset,
    ) -> Result<(), UnitsError> {
        if let Some(value) = self.meta.get(crate::store::keys::UNITS)
            && value.as_str() != Some(from.name())
        {
            return Err(UnitsError::Unconvertible {
                column: "meta.units".to_string(),
                reason: format!(
                    "frame declares units {value:?}, but the source preset is `{}`",
                    from.name()
                ),
            });
        }

        let factor_of = |d: PresetDim, column: &str| -> Result<F, UnitsError> {
            let refuse = |reason: String| UnitsError::Unconvertible {
                column: column.to_string(),
                reason,
            };
            let scale = |e: UnitsError| {
                refuse(format!(
                    "no {} scale from `{}` to `{}`: {e}",
                    d.name(),
                    from.name(),
                    to.name()
                ))
            };
            let [src, dst] = [from, to].map(|preset| {
                preset.unit(d.name()).ok_or_else(|| {
                    refuse(format!(
                        "preset `{}` defines no {} unit",
                        preset.name(),
                        d.name()
                    ))
                })
            });
            let src = registry.parse(src?).map_err(scale)?;
            let dst = registry.parse(dst?).map_err(scale)?;
            src.factor_to(&dst).map_err(scale)
        };

        let mut plan: Vec<(String, String, F)> = Vec::new();
        for (block_name, block) in self.iter() {
            for key in block.keys() {
                let dtype = block.dtype(key).expect("key comes from block.keys()");
                let scalable = match dtype {
                    DType::Float => true,
                    DType::Float16 | DType::Float32 | DType::Complex64 | DType::Complex128 => false,
                    DType::Int8
                    | DType::Int16
                    | DType::Int
                    | DType::Int64
                    | DType::Bool
                    | DType::UInt
                    | DType::U8
                    | DType::UInt16
                    | DType::UInt32
                    | DType::String => continue,
                };
                let column = format!("{block_name}.{key}");
                let dimension = crate::store::schema::column(key).map(|spec| spec.dimension);
                if !scalable {
                    // Only `Float` storage is scaled in place; any other
                    // floating width may pass solely when it carries no unit.
                    if dimension == Some(ColumnDim::Dimensionless) {
                        continue;
                    }
                    return Err(UnitsError::Unconvertible {
                        column,
                        reason: format!("storage dtype {} not convertible", dtype.name()),
                    });
                }
                let dimension = dimension.ok_or_else(|| UnitsError::Unconvertible {
                    column: column.clone(),
                    reason: "Float column with no schema dimension".to_string(),
                })?;
                let factor = match dimension {
                    ColumnDim::Dimensionless => continue,
                    ColumnDim::NotAQuantity => {
                        return Err(UnitsError::Unconvertible {
                            column,
                            reason: "Float column declared as not a physical quantity".to_string(),
                        });
                    }
                    ColumnDim::Of(d) => factor_of(d, &column)?,
                    ColumnDim::Product(a, b) => factor_of(a, &column)? * factor_of(b, &column)?,
                };
                plan.push((block_name.to_string(), key.to_string(), factor));
            }
        }

        let simbox = match &self.simbox {
            None => None,
            Some(b) => {
                let length = factor_of(PresetDim::Length, "simbox")?;
                let h = if b.is_cell_defined() {
                    b.h_view().to_owned() * length
                } else {
                    // A no-cell box carries the identity as a placeholder.
                    b.h_view().to_owned()
                };
                let origin = b.origin_view().to_owned() * length;
                Some(
                    SimBox::new_cell(h, origin, b.pbc(), b.is_cell_defined()).map_err(|e| {
                        UnitsError::Unconvertible {
                            column: "simbox".to_string(),
                            reason: format!("scaled cell is invalid: {e:?}"),
                        }
                    })?,
                )
            }
        };

        for (block_name, key, factor) in plan {
            self.get_mut(&block_name)
                .and_then(|block| block.get_float_mut(&key))
                .expect("planned column exists")
                .mapv_inplace(|v| v * factor);
        }
        self.simbox = simbox;
        self.meta.insert(crate::store::keys::UNITS, to.name());
        Ok(())
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
        frame.meta.insert("title", "Test");

        let (blocks, meta, simbox) = frame.into_inner();
        assert_eq!(blocks.len(), 1);
        assert!(blocks.contains_key("atoms"));
        assert_eq!(meta.get("title").unwrap().as_str(), Some("Test"));
        assert!(simbox.is_none());
    }

    #[test]
    fn test_frame_clear_preserves_meta() {
        let mut frame = Frame::new();
        frame.insert("atoms", Block::new());
        frame.meta.insert("title", "Test");

        frame.clear();
        assert!(frame.is_empty());
        assert!(!frame.meta.is_empty());
        assert_eq!(frame.meta.get("title").unwrap().as_str(), Some("Test"));
    }

    #[test]
    fn test_frame_clear_all() {
        let mut frame = Frame::new();
        frame.insert("atoms", Block::new());
        frame.meta.insert("title", "Test");

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
        frame.meta.insert("title", "Test");

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
        frame.rename_column("atoms", "x", "position_x").unwrap();
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
        assert!(frame.rename_column("nonexistent", "x", "new_x").is_err());

        // Try to rename non-existent column
        assert!(
            frame
                .rename_column("atoms", "nonexistent", "new_name")
                .is_err()
        );
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
            .insert("count", Array1::from_vec(vec![1 as I]).into_dyn())
            .unwrap();
        frame.insert("bonds", bonds);
        assert!(!frame.rename_block("molecules", "bonds"));
    }

    // ---- convert_units ----------------------------------------------------
    //
    // Goldens hand-derived (spec assembly-02 Domain basis) for sigma = 4.2 A,
    // m = 100 g/mol, epsilon = 1 kcal/mol under the LAMMPS `lj` relations
    // (docs.lammps.org/units.html): x = x* sigma; v = v* sigma/tau;
    // q = q* e sqrt(sigma[A] eps[kcal/mol] / 332.06371).

    use crate::store::keys::UNITS;
    use crate::types::Idx;
    use crate::units::{UnitPreset, UnitRegistry, UnitsError};
    use ndarray::array;

    fn sigma_only_registry() -> UnitRegistry {
        let mut r = UnitRegistry::new();
        let sigma = r.quantity(4.2, "angstrom").unwrap();
        r.define_lj_sigma(&sigma).unwrap();
        r
    }

    fn full_lj_registry() -> UnitRegistry {
        let mut r = UnitRegistry::new();
        let mass = r.quantity(100.0, "gram_per_mole").unwrap();
        let sigma = r.quantity(4.2, "angstrom").unwrap();
        let epsilon = r.quantity(1.0, "kilocalorie_per_mole").unwrap();
        r.define_lj_units(&mass, &sigma, &epsilon).unwrap();
        r
    }

    fn float_col(block: &mut Block, key: &str, v: F) {
        block
            .insert(key, Array1::from_vec(vec![v]).into_dyn())
            .unwrap();
    }

    /// One atom at x* = 1.5 in a triclinic box with bounds [0, 10] and
    /// xy tilt -0.5, declared as `lj` in meta.
    fn lj_frame() -> Frame {
        let mut atoms = Block::new();
        float_col(&mut atoms, "x", 1.5);
        float_col(&mut atoms, "y", 0.5);
        float_col(&mut atoms, "z", -1.0);
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.simbox = Some(
            SimBox::new(
                SimBox::matrix_from_lengths_tilts([10.0, 10.0, 10.0], [-0.5, 0.0, 0.0]),
                array![0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
        );
        frame.meta.insert(UNITS, "lj");
        frame
    }

    fn with_float(key: &str, v: F) -> Frame {
        let mut frame = lj_frame();
        float_col(frame.get_mut("atoms").unwrap(), key, v);
        frame
    }

    /// Bit-level image of blocks, box and meta, to prove a refusal wrote nothing.
    #[derive(Debug, PartialEq)]
    struct Snapshot {
        columns: Vec<(String, String, String)>,
        simbox: Option<(Vec<String>, Vec<String>)>,
        meta: MetaMap,
    }

    fn snapshot(frame: &Frame) -> Snapshot {
        let mut columns = Vec::new();
        for (bname, block) in frame.iter() {
            for key in block.keys() {
                let image = if let Some(a) = block.get_float(key) {
                    let bits: Vec<String> =
                        a.iter().map(|v| format!("{:x}", v.to_bits())).collect();
                    format!("float{:?}{:?}", a.shape(), bits)
                } else if let Some(a) = block.get_int(key) {
                    format!("int{a:?}")
                } else if let Some(a) = block.get_uint(key) {
                    format!("uint{a:?}")
                } else {
                    panic!("snapshot: unexpected dtype for {bname}.{key}");
                };
                columns.push((bname.to_string(), key.to_string(), image));
            }
        }
        columns.sort();
        fn bits<'a>(it: impl Iterator<Item = &'a F>) -> Vec<String> {
            it.map(|v| format!("{:x}", v.to_bits())).collect()
        }
        let simbox = frame
            .simbox
            .as_ref()
            .map(|b| (bits(b.h_view().iter()), bits(b.origin_view().iter())));
        Snapshot {
            columns,
            simbox,
            meta: frame.meta.clone(),
        }
    }

    fn atom(frame: &Frame, key: &str) -> F {
        frame["atoms"].get_float(key).unwrap()[[0]]
    }

    fn meta_units(frame: &Frame) -> Option<&str> {
        frame.meta.get(UNITS).and_then(|v| v.as_str())
    }

    fn assert_close(got: F, want: F, tol: F) {
        assert!((got - want).abs() < tol, "got {got}, want {want}");
    }

    fn assert_rel(got: F, want: F) {
        let rel = ((got - want) / want).abs();
        assert!(rel < 1e-6, "got {got}, want {want} (rel {rel:e})");
    }

    /// Converting lj -> real must be refused naming `column`, leaving the
    /// frame bitwise unchanged (meta `units` included).
    fn assert_refused_unchanged(mut frame: Frame, reg: &UnitRegistry, column: &str) {
        let before = snapshot(&frame);
        let err = frame
            .convert_units(reg, &UnitPreset::lj(), &UnitPreset::real())
            .unwrap_err();
        match err {
            UnitsError::Unconvertible { column: named, .. } => {
                assert!(
                    named.contains(column),
                    "refusal names {named:?}, want {column:?}"
                )
            }
            other => panic!("expected Unconvertible for {column}, got {other:?}"),
        }
        assert_eq!(
            snapshot(&frame),
            before,
            "refusal must leave the frame unchanged"
        );
    }

    #[test]
    fn sigma_only_lj_to_real_converts_lengths_and_refuses_mass() {
        let reg = sigma_only_registry();
        let (lj, real) = (UnitPreset::lj(), UnitPreset::real());

        let mut frame = lj_frame();
        frame.convert_units(&reg, &lj, &real).unwrap();
        assert_close(atom(&frame, "x"), 6.3, 1e-12);
        assert_close(atom(&frame, "y"), 2.1, 1e-12);
        assert_close(atom(&frame, "z"), -4.2, 1e-12);
        let bx = frame.simbox.as_ref().unwrap();
        let h = bx.h_view();
        for i in 0..3 {
            assert_close(h[[i, i]], 42.0, 1e-12);
            assert_close(bx.origin_view()[i], 0.0, 1e-12);
        }
        assert_close(bx.tilts()[0], -2.1, 1e-12);
        assert_eq!(meta_units(&frame), Some("real"));

        let mut with_mass = with_float("mass", 1.0);
        let before = snapshot(&with_mass);
        let err = with_mass.convert_units(&reg, &lj, &real).unwrap_err();
        assert!(
            matches!(err, UnitsError::Unconvertible { ref column, .. } if column.contains("mass")),
            "got {err:?}"
        );
        assert_eq!(meta_units(&with_mass), Some("lj"));
        assert_eq!(snapshot(&with_mass), before);
    }

    #[test]
    fn convert_units_scales_box_origin_by_length_factor() {
        let mut frame = lj_frame();
        frame.simbox = Some(
            SimBox::new(
                SimBox::matrix_from_lengths_tilts([10.0, 10.0, 10.0], [0.0, 0.0, 0.0]),
                array![1.0, -2.0, 0.5],
                [true, true, true],
            )
            .unwrap(),
        );
        frame
            .convert_units(
                &sigma_only_registry(),
                &UnitPreset::lj(),
                &UnitPreset::real(),
            )
            .unwrap();
        let o = frame.simbox.as_ref().unwrap().origin_view().to_owned();
        assert_close(o[0], 4.2, 1e-12);
        assert_close(o[1], -8.4, 1e-12);
        assert_close(o[2], 2.1, 1e-12);
    }

    #[test]
    fn sigma_only_refuses_charge_leaving_frame_unchanged() {
        assert_refused_unchanged(with_float("charge", 1.0), &sigma_only_registry(), "charge");
    }

    #[test]
    fn sigma_only_refuses_velocity_leaving_frame_unchanged() {
        assert_refused_unchanged(with_float("vx", 1.0), &sigma_only_registry(), "vx");
    }

    #[test]
    fn undeclared_float_column_is_refused() {
        assert_refused_unchanged(with_float("foo", 1.0), &full_lj_registry(), "foo");
    }

    #[test]
    fn integer_id_type_and_image_columns_are_untouched() {
        let mut frame = lj_frame();
        let atoms = frame.get_mut("atoms").unwrap();
        atoms
            .insert("id", Array1::from_vec(vec![7 as Idx]).into_dyn())
            .unwrap();
        atoms
            .insert("type_id", Array1::from_vec(vec![2 as Idx]).into_dyn())
            .unwrap();
        atoms
            .insert("ix", Array1::from_vec(vec![-1 as I]).into_dyn())
            .unwrap();
        frame
            .convert_units(
                &sigma_only_registry(),
                &UnitPreset::lj(),
                &UnitPreset::real(),
            )
            .unwrap();
        let atoms = &frame["atoms"];
        assert_eq!(atoms.get_uint("id").unwrap()[[0]], 7);
        assert_eq!(atoms.get_uint("type_id").unwrap()[[0]], 2);
        assert_eq!(atoms.get_int("ix").unwrap()[[0]], -1);
        assert_close(atom(&frame, "x"), 6.3, 1e-12);
    }

    #[test]
    fn meta_units_other_than_the_source_preset_is_refused() {
        let mut frame = lj_frame();
        frame.meta.insert(UNITS, "metal");
        assert_refused_unchanged(frame, &full_lj_registry(), "meta.units");

        let mut frame = lj_frame();
        frame.meta.insert(UNITS, "metal");
        let err = frame
            .convert_units(&full_lj_registry(), &UnitPreset::lj(), &UnitPreset::real())
            .unwrap_err();
        assert!(
            matches!(err, UnitsError::Unconvertible { ref column, .. } if column == "meta.units"),
            "got {err:?}"
        );
    }

    #[test]
    fn absent_meta_units_is_set_to_the_target_preset() {
        let mut frame = lj_frame();
        frame.meta.remove(UNITS);
        frame
            .convert_units(
                &sigma_only_registry(),
                &UnitPreset::lj(),
                &UnitPreset::real(),
            )
            .unwrap();
        assert_eq!(meta_units(&frame), Some("real"));
    }

    #[test]
    fn full_lj_converts_velocity_and_charge_goldens() {
        let mut frame = with_float("vx", 1.0);
        float_col(frame.get_mut("atoms").unwrap(), "charge", 1.0);
        frame
            .convert_units(&full_lj_registry(), &UnitPreset::lj(), &UnitPreset::real())
            .unwrap();
        assert_rel(atom(&frame, "vx"), 2.045483e-3);
        assert_rel(atom(&frame, "charge"), 0.1124641);
    }

    #[test]
    fn full_lj_converts_mass_and_dipole_product() {
        // mux: Product(Charge, Length) -> 0.1124641 e * 4.2 A = 0.4723492 e*A.
        let mut frame = with_float("mass", 1.0);
        float_col(frame.get_mut("atoms").unwrap(), "mux", 1.0);
        frame
            .convert_units(&full_lj_registry(), &UnitPreset::lj(), &UnitPreset::real())
            .unwrap();
        assert_rel(atom(&frame, "mass"), 100.0);
        assert_rel(atom(&frame, "mux"), 0.4723492);
    }

    #[test]
    fn f32_length_column_is_refused_leaving_frame_unchanged() {
        // The Zarr reader stores an f32 array as `Column::from_f32`.
        let mut frame = lj_frame();
        let atoms = frame.get_mut("atoms").unwrap();
        atoms.remove("x");
        atoms
            .insert_column(
                "x",
                crate::store::block::Column::from_f32(Array1::from_vec(vec![1.5f32]).into_dyn()),
            )
            .unwrap();
        let before_h = frame.simbox.as_ref().unwrap().h_view().to_owned();

        let err = frame
            .convert_units(
                &sigma_only_registry(),
                &UnitPreset::lj(),
                &UnitPreset::real(),
            )
            .unwrap_err();
        match err {
            UnitsError::Unconvertible {
                ref column,
                ref reason,
            } => {
                assert_eq!(column, "atoms.x");
                assert!(reason.contains("f32"), "reason {reason:?}");
            }
            other => panic!("expected Unconvertible for atoms.x, got {other:?}"),
        }
        let x = frame["atoms"].get("x").unwrap().as_f32().unwrap()[[0]];
        assert_eq!(x.to_bits(), 1.5f32.to_bits());
        assert_eq!(atom(&frame, "y").to_bits(), (0.5 as F).to_bits());
        assert_eq!(frame.simbox.as_ref().unwrap().h_view().to_owned(), before_h);
        assert_eq!(meta_units(&frame), Some("lj"));
    }

    #[test]
    fn dimensionless_quaternion_is_untouched() {
        let mut frame = with_float("quatw", 0.3);
        frame
            .convert_units(&full_lj_registry(), &UnitPreset::lj(), &UnitPreset::real())
            .unwrap();
        assert_eq!(atom(&frame, "quatw").to_bits(), (0.3 as F).to_bits());
    }
}
