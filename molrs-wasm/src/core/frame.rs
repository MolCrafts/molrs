//! WASM bindings for [`Frame`] -- the top-level hierarchical data container.
//!
//! A `Frame` holds a collection of named [`Block`]s (e.g., `"atoms"`,
//! `"bonds"`, `"angles"`) and an optional [`SimBox`](super::region::simbox::Box)
//! defining periodic boundary conditions.
//!
//! # Typical block layout
//!
//! | Block key   | Expected columns | Column types |
//! |-------------|------------------|--------------|
//! | `"atoms"`   | `symbol` (string), `x`, `y`, `z` (F), optionally `mass`, `charge` (F) | string, F |
//! | `"bonds"`   | `atomi`, `atomj` (u32), `bond_type` (u32), `bond_number` (u32) | u32 |
//! | `"angles"`  | `i`, `j`, `k` (u32) | u32 |
//!
//! # Example (JavaScript)
//!
//! ```js
//! const frame = new Frame();
//! const atoms = frame.createBlock("atoms");
//! atoms.setColStr("symbol", ["C", "C", "O"]);
//! atoms.setColF("x", xCoords);
//! atoms.setColF("y", yCoords);
//! atoms.setColF("z", zCoords);
//!
//! const bonds = frame.createBlock("bonds");
//! bonds.setColU32("i", new Uint32Array([0, 1]));
//! bonds.setColU32("j", new Uint32Array([1, 2]));
//! bonds.setColU("bond_type", bondTypes);   // 4 = aromatic
//! bonds.setColU("bond_number", bondNumbers); // localized 1/2/3
//! ```

use js_sys::{Array as JsArray, Float32Array, Int32Array, Uint32Array};
use wasm_bindgen::prelude::*;

use molrs::store::block::Block as RsBlock;
use molrs::store::meta::MetaValue;
use molrs_ffi::{BlockRef, FrameRef};

use super::types::JsFloatArray;

use super::block::Block;
use super::js_err;

/// Hierarchical data container mapping string keys to typed [`Block`]s.
///
/// A `Frame` owns a set of named blocks (column stores) and an optional
/// simulation box ([`Box`](super::region::simbox::Box)). This is the
/// primary interchange type for molecular data in the WASM API.
///
/// # Conventions
///
/// - The `"atoms"` block should contain per-atom properties: `symbol`
///   (string), `x`/`y`/`z` (F, coordinates in angstrom), and optionally
///   `mass` (F, atomic mass units) and `charge` (F, elementary charges).
/// - The `"bonds"` block should contain bond topology: `atomi`/`atomj` (u32,
///   zero-based atom indices), `bond_type` (u32: 1 single, 2 double, 3 triple,
///   4 aromatic) and `bond_number` (u32: the localized Lewis/Kekulé integer).
///
/// # Example (JavaScript)
///
/// ```js
/// const frame = new Frame();
/// const atoms = frame.createBlock("atoms");
/// atoms.setColF("x", xCoords);
/// ```
#[wasm_bindgen]
pub struct Frame {
    /// Paired frame id + shared store. All lifetime management lives in
    /// the shared `molrs_ffi::FrameRef` type so each binding layer (wasm,
    /// python, capi) has only the attribute plumbing to write.
    pub(crate) inner: FrameRef,
}

#[wasm_bindgen]
impl Frame {
    /// Create a new, empty `Frame` with no blocks and no simulation box.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = new Frame();
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Frame {
            inner: FrameRef::new_standalone(),
        }
    }

    /// Create a new empty [`Block`] and register it under `key`.
    ///
    /// If a block with the same key already exists it is replaced.
    ///
    /// # Arguments
    ///
    /// * `key` - Block name (e.g., `"atoms"`, `"bonds"`)
    ///
    /// # Returns
    ///
    /// A mutable [`Block`] handle that can be used to add columns.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the underlying store operation fails
    /// (e.g., the frame has been dropped).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const atoms = frame.createBlock("atoms");
    /// atoms.setColF("x", xCoords);
    /// ```
    #[wasm_bindgen(js_name = createBlock)]
    pub fn create_block(&self, key: &str) -> Result<Block, JsValue> {
        let rs_block = RsBlock::new();
        self.inner
            .store
            .borrow_mut()
            .set_block(self.inner.id, key, rs_block)
            .map_err(js_err)?;
        let handle = self
            .inner
            .store
            .borrow()
            .get_block(self.inner.id, key)
            .map_err(js_err)?;
        Ok(Block {
            inner: BlockRef::new(self.inner.store.clone(), handle),
        })
    }

    /// Retrieve an existing [`Block`] by name.
    ///
    /// # Arguments
    ///
    /// * `key` - Block name to look up
    ///
    /// # Returns
    ///
    /// The [`Block`] if found, or `undefined` if no block with that key
    /// exists in this frame.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const atoms = frame.getBlock("atoms");
    /// if (atoms) {
    ///   const x = atoms.copyColF("x");
    /// }
    /// ```
    #[wasm_bindgen(js_name = getBlock)]
    pub fn get_block(&self, key: &str) -> Option<Block> {
        let handle = self
            .inner
            .store
            .borrow()
            .get_block(self.inner.id, key)
            .ok()?;
        Some(Block {
            inner: BlockRef::new(self.inner.store.clone(), handle),
        })
    }

    /// True when `block[key]` exists and is `f32`.
    #[wasm_bindgen(js_name = hasF32)]
    pub fn has_f32(&self, block: &str, key: &str) -> bool {
        self.get_block(block).is_some_and(|b| b.has_f32(key))
    }

    /// True when `block[key]` exists and is `f64`.
    #[wasm_bindgen(js_name = hasF64)]
    pub fn has_f64(&self, block: &str, key: &str) -> bool {
        self.get_block(block).is_some_and(|b| b.has_f64(key))
    }

    /// True when `block[key]` exists and is `i32`.
    #[wasm_bindgen(js_name = hasI32)]
    pub fn has_i32(&self, block: &str, key: &str) -> bool {
        self.get_block(block).is_some_and(|b| b.has_i32(key))
    }

    /// True when `block[key]` exists and is `u32`.
    #[wasm_bindgen(js_name = hasU32)]
    pub fn has_u32(&self, block: &str, key: &str) -> bool {
        self.get_block(block).is_some_and(|b| b.has_u32(key))
    }

    /// True when `block[key]` exists and is a string column.
    #[wasm_bindgen(js_name = hasStr)]
    pub fn has_str(&self, block: &str, key: &str) -> bool {
        self.get_block(block).is_some_and(|b| b.has_str(key))
    }

    /// Owned `f32` column from `block`. Missing with no `default` throws.
    #[wasm_bindgen(js_name = getF32)]
    pub fn get_f32(
        &self,
        block: &str,
        key: &str,
        default: Option<Float32Array>,
    ) -> Result<Float32Array, JsValue> {
        frame_block(self, block)?.get_f32(key, default)
    }

    /// Owned `f64` column from `block`. Missing with no `default` throws.
    #[wasm_bindgen(js_name = getF64)]
    pub fn get_f64(
        &self,
        block: &str,
        key: &str,
        default: Option<JsFloatArray>,
    ) -> Result<JsFloatArray, JsValue> {
        frame_block(self, block)?.get_f64(key, default)
    }

    /// Owned i32 column from `block`. Missing with no `default` throws.
    #[wasm_bindgen(js_name = getI32)]
    pub fn get_i32(
        &self,
        block: &str,
        key: &str,
        default: Option<Int32Array>,
    ) -> Result<Int32Array, JsValue> {
        frame_block(self, block)?.get_i32(key, default)
    }

    /// Owned u32 column from `block`. Missing with no `default` throws.
    #[wasm_bindgen(js_name = getU32)]
    pub fn get_u32(
        &self,
        block: &str,
        key: &str,
        default: Option<Uint32Array>,
    ) -> Result<Uint32Array, JsValue> {
        frame_block(self, block)?.get_u32(key, default)
    }

    /// Owned string column from `block`. Missing with no `default` throws.
    #[wasm_bindgen(js_name = getStr)]
    pub fn get_str(
        &self,
        block: &str,
        key: &str,
        default: Option<JsArray>,
    ) -> Result<JsArray, JsValue> {
        frame_block(self, block)?.get_str(key, default)
    }

    /// Insert a block by deep-copying its data into this frame's store.
    ///
    /// This is useful for transferring a block from one frame to another.
    /// The source block's data is cloned; subsequent modifications to the
    /// source will not affect this frame.
    ///
    /// # Arguments
    ///
    /// * `key` - Name under which to store the block
    /// * `block` - The source [`Block`] whose data will be copied
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if either the source block or the
    /// destination frame handle is invalid.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const otherFrame = new Frame();
    /// const atoms = otherFrame.createBlock("atoms");
    /// // ... populate atoms ...
    /// frame.insertBlock("atoms", atoms);
    /// ```
    #[wasm_bindgen(js_name = insertBlock)]
    pub fn insert_block(&self, key: &str, block: Block) -> Result<(), JsValue> {
        let rs_block = block.inner.clone_block().map_err(js_err)?;
        self.inner
            .store
            .borrow_mut()
            .set_block(self.inner.id, key, rs_block)
            .map_err(js_err)
    }

    /// Remove a block by name.
    ///
    /// # Arguments
    ///
    /// * `key` - Block name to remove
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has been dropped or the
    /// key does not exist.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.removeBlock("bonds");
    /// ```
    #[wasm_bindgen(js_name = removeBlock)]
    pub fn remove_block(&self, key: &str) -> Result<(), JsValue> {
        self.inner
            .store
            .borrow_mut()
            .remove_block(self.inner.id, key)
            .map_err(js_err)
    }

    /// Remove all blocks from this frame (but keep the frame alive).
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has already been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.clear();
    /// ```
    #[wasm_bindgen(js_name = clear)]
    pub fn clear(&self) -> Result<(), JsValue> {
        self.inner
            .store
            .borrow_mut()
            .clear_frame(self.inner.id)
            .map_err(js_err)
    }

    /// Rename a block from `old_key` to `new_key`.
    ///
    /// # Arguments
    ///
    /// * `old_key` - Current block name
    /// * `new_key` - New block name
    ///
    /// # Returns
    ///
    /// `true` if the block was found and renamed, `false` if `old_key`
    /// did not exist.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.renameBlock("atoms", "particles");
    /// ```
    #[wasm_bindgen(js_name = renameBlock)]
    pub fn rename_block(&self, old_key: &str, new_key: &str) -> Result<bool, JsValue> {
        self.inner
            .store
            .borrow_mut()
            .with_frame_mut(self.inner.id, |f| f.rename_block(old_key, new_key))
            .map_err(js_err)
    }

    /// Rename a column within a specific block.
    ///
    /// # Arguments
    ///
    /// * `block_key` - Name of the block containing the column
    /// * `old_col` - Current column name
    /// * `new_col` - New column name
    ///
    /// # Returns
    ///
    /// `true` if the column was found and renamed, `false` if
    /// `old_col` did not exist in the block.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame or block does not exist.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.renameColumn("atoms", "element", "symbol");
    /// ```
    #[wasm_bindgen(js_name = renameColumn)]
    pub fn rename_column(
        &self,
        block_key: &str,
        old_col: &str,
        new_col: &str,
    ) -> Result<(), JsValue> {
        self.inner
            .store
            .borrow_mut()
            .with_frame_mut(self.inner.id, |f| {
                f.rename_column(block_key, old_col, new_col)
            })
            .map_err(js_err)?
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Read a per-frame metadata value as a numeric scalar.
    ///
    /// Returns `Some(v)` if the meta key exists AND its string value parses
    /// as an `f64`. Returns `None` if the key is missing or the value is
    /// non-numeric (e.g., `config="trans"`).
    ///
    /// Frame meta is typed (`MetaValue`). This accessor accepts every numeric
    /// scalar dtype and preserves compatibility with numeric strings written
    /// through [`setMeta`](Self::set_meta).
    ///
    /// # Arguments
    ///
    /// * `name` — Meta key to look up (e.g., `"energy"`, `"temp"`).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const energy = frame.getMetaScalar("energy");
    /// if (energy !== undefined) {
    ///   console.log("Energy:", energy);
    /// }
    /// ```
    #[wasm_bindgen(js_name = getMetaScalar)]
    pub fn get_meta_scalar(&self, name: &str) -> Option<f64> {
        self.inner
            .store
            .borrow()
            .with_frame(self.inner.id, |frame| {
                frame.meta.get(name).and_then(|value| match value {
                    MetaValue::I32(value) => Some(f64::from(*value)),
                    MetaValue::I64(value) => Some(*value as f64),
                    MetaValue::U32(value) => Some(f64::from(*value)),
                    MetaValue::U64(value) => Some(*value as f64),
                    MetaValue::F32(value) => Some(f64::from(*value)),
                    MetaValue::F64(value) => Some(*value),
                    MetaValue::String(value) => value.parse::<f64>().ok(),
                    _ => None,
                })
            })
            .ok()?
    }

    /// Return the names of all metadata keys on this frame.
    ///
    /// Includes all keys regardless of whether their values are numeric
    /// or categorical. To filter to numeric keys, iterate and call
    /// [`getMetaScalar`](Self::get_meta_scalar) on each.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const names = frame.metaNames(); // e.g. ["energy", "config", "temp"]
    /// ```
    #[wasm_bindgen(js_name = metaNames)]
    pub fn meta_names(&self) -> Vec<String> {
        self.inner
            .store
            .borrow()
            .with_frame(self.inner.id, |frame| {
                frame.meta.keys().cloned().collect::<Vec<String>>()
            })
            .unwrap_or_default()
    }

    /// Return the names of all blocks attached to this frame.
    ///
    /// Iteration order matches the underlying `HashMap` and is therefore
    /// not stable across runs — callers that need a deterministic order
    /// must sort on the JS side. Returns an empty array if the frame
    /// has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const names = frame.blockNames(); // e.g. ["atoms", "bonds"]
    /// ```
    #[wasm_bindgen(js_name = blockNames)]
    pub fn block_names(&self) -> Vec<String> {
        self.inner
            .store
            .borrow()
            .with_frame(self.inner.id, |frame| {
                frame.keys().map(|k| k.to_string()).collect::<Vec<String>>()
            })
            .unwrap_or_default()
    }

    /// Set a per-frame metadata string value.
    ///
    /// Stores `value` as a typed `MetaValue::String` on `frame.meta`.
    /// Use [`setMetaScalar`](Self::set_meta_scalar) for numeric labels that
    /// [`getMetaScalar`](Self::get_meta_scalar) should return.
    ///
    /// # Arguments
    ///
    /// * `name` — Meta key (e.g., `"note"`, `"config"`).
    /// * `value` — String value.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.setMeta("note", "run-42");
    /// frame.setMetaScalar("energy", -3.14);
    /// ```
    #[wasm_bindgen(js_name = setMeta)]
    pub fn set_meta(&self, name: &str, value: &str) -> Result<(), JsValue> {
        self.inner
            .store
            .borrow_mut()
            .with_frame_mut(self.inner.id, |frame| {
                frame.meta.insert(
                    name.to_string(),
                    molrs::store::meta::MetaValue::String(value.to_string()),
                );
            })
            .map_err(js_err)
    }

    /// Set a per-frame numeric metadata value (`MetaValue::F64`).
    #[wasm_bindgen(js_name = setMetaScalar)]
    pub fn set_meta_scalar(&self, name: &str, value: f64) -> Result<(), JsValue> {
        self.inner
            .store
            .borrow_mut()
            .with_frame_mut(self.inner.id, |frame| {
                frame
                    .meta
                    .insert(name.to_string(), molrs::store::meta::MetaValue::F64(value));
            })
            .map_err(js_err)
    }

    /// Get the simulation box attached to this frame (if any).
    ///
    /// # Returns
    ///
    /// The [`Box`](super::region::simbox::Box) if one has been set,
    /// or `undefined` otherwise.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const box = frame.simbox;
    /// if (box) {
    ///   console.log("Volume:", box.volume());
    /// }
    /// ```
    #[wasm_bindgen(getter, js_name = box)]
    pub fn get_box(&self) -> Option<super::region::simbox::Box> {
        self.inner
            .store
            .borrow()
            .with_frame_box(self.inner.id, |sb| {
                sb.map(|s| super::region::simbox::Box { inner: s.clone() })
            })
            .ok()?
    }

    /// Attach or detach a simulation box.
    ///
    /// Pass a [`Box`](super::region::simbox::Box) to attach, or
    /// `undefined`/`null` to detach.
    ///
    /// # Arguments
    ///
    /// * `box` - The simulation box, or `undefined`/`null` to remove it
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const origin = originVec;
    /// frame.simbox = Box.cube(10.0, origin, true, true, true);
    /// ```
    #[wasm_bindgen(setter, js_name = box)]
    pub fn set_box(&self, simbox: Option<super::region::simbox::Box>) -> Result<(), JsValue> {
        self.inner
            .store
            .borrow_mut()
            .set_frame_box(self.inner.id, simbox.map(|b| b.inner))
            .map_err(js_err)
    }

    /// Explicitly release this frame and all its blocks from the store.
    ///
    /// After calling `drop()`, any subsequent operations on this frame
    /// or its blocks will throw. This is optional -- the frame will also
    /// be released when garbage-collected by the JS engine.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame was already dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.drop();
    /// // frame.clear() would now throw
    /// ```
    /// Judge this frame against the canonical Frame schema.
    ///
    /// Throws a string describing every violation when the frame does not
    /// conform. The rules live in molrs (`Validator::canonical`); JavaScript
    /// must not re-check endpoint ranges or column dtypes by hand.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string with the full schema report when validation
    /// fails, or when the frame handle has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.validate();  // throws if bonds.atomi is out of range, …
    /// ```
    #[wasm_bindgen]
    pub fn validate(&self) -> Result<(), JsValue> {
        self.with_frame(|f| f.validate().map_err(|e| JsValue::from_str(&e.to_string())))
    }

    #[wasm_bindgen(js_name = drop)]
    pub fn drop_frame(&self) -> Result<(), JsValue> {
        self.inner
            .store
            .borrow_mut()
            .frame_drop(self.inner.id)
            .map_err(js_err)
    }
}

fn frame_block(frame: &Frame, block: &str) -> Result<Block, JsValue> {
    frame
        .get_block(block)
        .ok_or_else(|| JsValue::from_str(&format!("block '{block}' not found")))
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal helpers (not exposed to JS).
impl Frame {
    pub(crate) fn from_rs(rs_frame: molrs::store::frame::Frame) -> Result<Self, JsValue> {
        let store = molrs_ffi::new_shared();
        let id = store.borrow_mut().frame_new();
        store.borrow_mut().set_frame(id, rs_frame).map_err(js_err)?;
        Ok(Frame {
            inner: FrameRef::new(store, id),
        })
    }

    /// Borrow the inner core frame for the duration of a closure.
    ///
    /// Zero-copy: no deep clone. The closure runs while the FFI store is
    /// immutably borrowed, so it must not attempt to mutate the store.
    pub(crate) fn with_frame<R>(
        &self,
        f: impl FnOnce(&molrs::store::frame::Frame) -> Result<R, JsValue>,
    ) -> Result<R, JsValue> {
        self.inner
            .store
            .borrow()
            .with_frame(self.inner.id, f)
            .map_err(js_err)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_frame_lifecycle() {
        let frame = Frame::new();
        assert!(frame.clear().is_ok());
        frame.drop_frame().unwrap();
        assert!(frame.clear().is_err());
    }

    /// Helper: build a wrapped `Frame` with two typed meta entries.
    fn frame_with_meta() -> Frame {
        use molrs::store::meta::MetaValue;
        let mut rs_frame = molrs::store::frame::Frame::new();
        rs_frame
            .meta
            .insert("energy".to_string(), MetaValue::F64(-1.23));
        rs_frame
            .meta
            .insert("config".to_string(), MetaValue::String("trans".into()));
        Frame::from_rs(rs_frame).unwrap()
    }

    #[wasm_bindgen_test]
    fn get_meta_scalar_parses_numeric() {
        let frame = frame_with_meta();
        let energy = frame.get_meta_scalar("energy").unwrap();
        assert!((energy - (-1.23)).abs() < 1e-10);
    }

    #[wasm_bindgen_test]
    fn get_meta_scalar_none_for_non_numeric() {
        let frame = frame_with_meta();
        assert!(frame.get_meta_scalar("config").is_none());
    }

    #[wasm_bindgen_test]
    fn get_meta_scalar_none_for_missing_key() {
        let frame = frame_with_meta();
        assert!(frame.get_meta_scalar("missing").is_none());
    }

    #[wasm_bindgen_test]
    fn meta_names_contains_all_keys() {
        let frame = frame_with_meta();
        let names = frame.meta_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"energy".to_string()));
        assert!(names.contains(&"config".to_string()));
    }
}
