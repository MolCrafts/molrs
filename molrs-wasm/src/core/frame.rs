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

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::Array as JsArray;
use wasm_bindgen::prelude::*;

use molrs::block::Block as RsBlock;
use molrs_ffi::{FrameId as FFIFrameId, Store as FFIStore};

use super::block::Block;
use super::grid::Grid;
use super::{SharedStore, js_err};

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
    id: FFIFrameId,
    store: SharedStore,
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
        let store = Rc::new(RefCell::new(FFIStore::new()));
        let id = store.borrow_mut().frame_new();
        Frame { id, store }
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
        self.store
            .borrow_mut()
            .set_block(self.id, key, rs_block)
            .map_err(js_err)?;
        let handle = self
            .store
            .borrow()
            .get_block(self.id, key)
            .map_err(js_err)?;
        Ok(Block {
            handle,
            store: self.store.clone(),
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
        let handle = self.store.borrow().get_block(self.id, key).ok()?;
        Some(Block {
            handle,
            store: self.store.clone(),
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
        let rs_block = block
            .store
            .borrow()
            .clone_block(&block.handle)
            .map_err(js_err)?;
        self.store
            .borrow_mut()
            .set_block(self.id, key, rs_block)
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
        self.store
            .borrow_mut()
            .remove_block(self.id, key)
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
        self.store.borrow_mut().clear_frame(self.id).map_err(js_err)
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
        self.store
            .borrow_mut()
            .with_frame_mut(self.id, |f| f.rename_block(old_key, new_key))
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
        self.store
            .borrow_mut()
            .with_frame_mut(self.id, |f| f.rename_column(block_key, old_col, new_col))
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
        self.store
            .borrow()
            .with_frame(self.id, |frame| {
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
        self.store
            .borrow()
            .with_frame(self.id, |frame| frame.has_grid(name))
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
        self.store
            .borrow()
            .with_frame(self.id, |frame| {
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
        self.store
            .borrow_mut()
            .with_frame_mut(self.id, |frame| {
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
        self.store
            .borrow_mut()
            .with_frame_mut(self.id, |frame| {
                frame.remove_grid(name);
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
        self.store
            .borrow()
            .with_frame_simbox(self.id, |sb| {
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
        self.store
            .borrow_mut()
            .set_frame_simbox(self.id, simbox.map(|b| b.inner))
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
        self.store.borrow_mut().frame_drop(self.id).map_err(js_err)
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal helpers (not exposed to JS).
impl Frame {
    pub(crate) fn from_rs_frame(rs_frame: molrs::frame::Frame) -> Result<Self, JsValue> {
        let store = Rc::new(RefCell::new(FFIStore::new()));
        let id = store.borrow_mut().frame_new();
        store.borrow_mut().set_frame(id, rs_frame).map_err(js_err)?;
        Ok(Frame { id, store })
    }

    pub(crate) fn clone_core_frame(&self) -> Result<molrs::frame::Frame, JsValue> {
        self.store.borrow().clone_frame(self.id).map_err(js_err)
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
}
