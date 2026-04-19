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
//! | `"bonds"`   | `i`, `j` (u32 -- atom indices), `order` (F -- 1.0/1.5/2.0/3.0) | u32, F |
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
//! bonds.setColF("order", bondOrders);
//! ```

use js_sys::Array as JsArray;
use wasm_bindgen::prelude::*;

use molrs::block::Block as RsBlock;
use molrs_ffi::{BlockRef, FrameRef};

use super::block::Block;
use super::grid::Grid;
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
/// - The `"bonds"` block should contain bond topology: `i`/`j` (u32,
///   zero-based atom indices) and `order` (F, bond order: 1.0 = single,
///   1.5 = aromatic, 2.0 = double, 3.0 = triple).
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
        self.inner.store
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
        self.inner.store.borrow_mut().clear_frame(self.inner.id).map_err(js_err)
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
        self.inner.store
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
    ) -> Result<bool, JsValue> {
        self.inner.store
            .borrow_mut()
            .with_frame_mut(self.inner.id, |f| f.rename_column(block_key, old_col, new_col))
            .map_err(js_err)
    }

    /// Return the names of all grids attached to this frame.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const names = frame.gridNames(); // e.g. ["chgcar", "spin"]
    /// ```
    #[wasm_bindgen(js_name = gridNames)]
    pub fn grid_names(&self) -> Result<JsArray, JsValue> {
        self.inner.store
            .borrow()
            .with_frame(self.inner.id, |frame| {
                let names = JsArray::new();
                for name in frame.grid_keys() {
                    names.push(&JsValue::from_str(name));
                }
                names
            })
            .map_err(js_err)
    }

    /// Returns `true` if a named grid is attached to this frame.
    ///
    /// # Arguments
    ///
    /// * `name` — Grid name to look up.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.hasGrid("chgcar"); // true or false
    /// ```
    #[wasm_bindgen(js_name = hasGrid)]
    pub fn has_grid(&self, name: &str) -> Result<bool, JsValue> {
        self.inner.store
            .borrow()
            .with_frame(self.inner.id, |frame| frame.has_grid(name))
            .map_err(js_err)
    }

    /// Retrieve a named grid attached to this frame.
    ///
    /// Returns a cloned [`Grid`] wrapper, or `undefined` if the grid does
    /// not exist. The returned object is independent of the frame — mutations
    /// to it are not reflected in the frame without a subsequent
    /// [`insertGrid`](Frame::insert_grid) call.
    ///
    /// # Arguments
    ///
    /// * `name` — Grid name to retrieve.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const g = frame.getGrid("chgcar");
    /// if (g) {
    ///   const arr = g.getArray("rho");
    /// }
    /// ```
    #[wasm_bindgen(js_name = getGrid)]
    pub fn get_grid(&self, name: &str) -> Result<Option<Grid>, JsValue> {
        self.inner.store
            .borrow()
            .with_frame(self.inner.id, |frame| {
                frame.get_grid(name).map(|g| Grid::from_rs(g.clone()))
            })
            .map_err(js_err)
    }

    /// Attach a grid to this frame under the given name.
    ///
    /// If a grid with the same name already exists it is replaced. The grid
    /// data is moved into the frame; the JS `Grid` object becomes empty after
    /// this call and should not be reused.
    ///
    /// # Arguments
    ///
    /// * `name` — Name to store the grid under (e.g., `"chgcar"`).
    /// * `grid` — The [`Grid`] to attach.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const grid = new Grid(10, 10, 10, origin, cell, true, true, true);
    /// grid.insertArray("rho", rhoData);
    /// frame.insertGrid("chgcar", grid);
    /// ```
    #[wasm_bindgen(js_name = insertGrid)]
    pub fn insert_grid(&self, name: &str, grid: Grid) -> Result<(), JsValue> {
        self.inner.store
            .borrow_mut()
            .with_frame_mut(self.inner.id, |frame| {
                frame.insert_grid(name, grid.into_rs());
            })
            .map_err(js_err)
    }

    /// Remove a named grid from this frame.
    ///
    /// # Arguments
    ///
    /// * `name` — Grid name to remove.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.removeGrid("chgcar");
    /// ```
    #[wasm_bindgen(js_name = removeGrid)]
    pub fn remove_grid(&self, name: &str) -> Result<(), JsValue> {
        self.inner.store
            .borrow_mut()
            .with_frame_mut(self.inner.id, |frame| {
                frame.remove_grid(name);
            })
            .map_err(js_err)
    }

    /// Read a per-frame metadata value as a numeric scalar.
    ///
    /// Returns `Some(v)` if the meta key exists AND its string value parses
    /// as an `f64`. Returns `None` if the key is missing or the value is
    /// non-numeric (e.g., `config="trans"`).
    ///
    /// `frame.meta` is a `HashMap<String, String>`; the ExtXYZ parser stores
    /// all comment-line values as strings. This accessor reads numeric ones
    /// via `str::parse::<f64>`.
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
        self.inner.store
            .borrow()
            .with_frame(self.inner.id, |frame| {
                frame.meta.get(name).and_then(|s| s.parse::<f64>().ok())
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
        self.inner.store
            .borrow()
            .with_frame(self.inner.id, |frame| {
                frame.meta.keys().cloned().collect::<Vec<String>>()
            })
            .unwrap_or_default()
    }

    /// Set a per-frame metadata value.
    ///
    /// Stores `value` as the string backing for `name` on `frame.meta`.
    /// Numeric values are read back via
    /// [`getMetaScalar`](Self::get_meta_scalar) by parsing the string
    /// form. `frame.meta` is the single source of truth for per-frame
    /// scalars — no separate aggregation layer is needed on the JS side.
    ///
    /// # Arguments
    ///
    /// * `name` — Meta key (e.g., `"energy"`, `"temp"`).
    /// * `value` — String value. For numeric labels, the caller is
    ///   responsible for converting (e.g., `num.toString()`).
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.setMeta("energy", "-3.14");
    /// frame.setMeta("note", "run-42");
    /// ```
    #[wasm_bindgen(js_name = setMeta)]
    pub fn set_meta(&self, name: &str, value: &str) -> Result<(), JsValue> {
        self.inner.store
            .borrow_mut()
            .with_frame_mut(self.inner.id, |frame| {
                frame.meta.insert(name.to_string(), value.to_string());
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
    #[wasm_bindgen(getter, js_name = simbox)]
    pub fn get_simbox(&self) -> Option<super::region::simbox::Box> {
        self.inner.store
            .borrow()
            .with_frame_simbox(self.inner.id, |sb| {
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
    /// * `simbox` - The simulation box, or `undefined`/`null` to remove it
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
    #[wasm_bindgen(setter, js_name = simbox)]
    pub fn set_simbox(&self, simbox: Option<super::region::simbox::Box>) -> Result<(), JsValue> {
        self.inner.store
            .borrow_mut()
            .set_frame_simbox(self.inner.id, simbox.map(|b| b.inner))
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
    #[wasm_bindgen(js_name = drop)]
    pub fn drop_frame(&self) -> Result<(), JsValue> {
        self.inner.store.borrow_mut().frame_drop(self.inner.id).map_err(js_err)
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal helpers (not exposed to JS).
impl Frame {
    pub(crate) fn from_rs(rs_frame: molrs::frame::Frame) -> Result<Self, JsValue> {
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
        f: impl FnOnce(&molrs::frame::Frame) -> Result<R, JsValue>,
    ) -> Result<R, JsValue> {
        self.inner.store
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

    /// Helper: build a wrapped `Frame` with two meta entries mirroring what
    /// an ExtXYZ parser would emit for `energy=-1.23 config=trans`.
    fn frame_with_meta() -> Frame {
        let mut rs_frame = molrs::frame::Frame::new();
        rs_frame
            .meta
            .insert("energy".to_string(), "-1.23".to_string());
        rs_frame
            .meta
            .insert("config".to_string(), "trans".to_string());
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
