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
//! | `"bonds"`   | `atomi`, `atomj` (u64), `bond_type` (u64), `bond_number` (u64) | u64 |
//! | `"angles"`  | `atomi`, `atomj`, `atomk` (u64) | u64 |
//!
//! # Example (JavaScript)
//!
//! ```js
//! const frame = new Frame();
//! const atoms = frame.createBlock("atoms");
//! atoms.set("element", ["C", "C", "O"]);
//! atoms.set("x", xCoords); // Float64Array
//! atoms.set("y", yCoords);
//! atoms.set("z", zCoords);
//!
//! const bonds = frame.createBlock("bonds");
//! bonds.set("atomi", new BigUint64Array([0n, 1n]));
//! bonds.set("atomj", new BigUint64Array([1n, 2n]));
//! bonds.set("bond_type", bondTypes);     // BigUint64Array; 4 = aromatic
//! bonds.set("bond_number", bondNumbers); // BigUint64Array; localized 1/2/3
//!
//! frame.get("atoms").get("x"); // Float64Array, like frame["atoms"]["x"]
//! ```

use wasm_bindgen::prelude::*;

use molrs::store::block::Block as RsBlock;
use molrs::store::meta::MetaValue;
use molrs_ffi::{BlockRef, FrameRef};

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
/// - The `"bonds"` block should contain bond topology: `atomi`/`atomj` (u64,
///   zero-based atom indices), `bond_type` (u64: 1 single, 2 double, 3 triple,
///   4 aromatic) and `bond_number` (u64: the localized Lewis/Kekulé integer).
///
/// # Example (JavaScript)
///
/// ```js
/// const frame = new Frame();
/// const atoms = frame.createBlock("atoms");
/// atoms.set("x", xCoords);
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
    /// atoms.set("x", xCoords);
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

    /// The [`Block`] named `key`: a live handle, so writes through it land
    /// in this frame. `frame.get("atoms").get("x")` mirrors Python's
    /// `frame["atoms"]["x"]`.
    ///
    /// # Errors
    ///
    /// Throws if no block is named `key` (test with [`has`](Self::has)), or
    /// if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const x = frame.get("atoms").get("x");
    /// const nBonds = frame.has("bonds") ? frame.get("bonds").nrows : 0;
    /// ```
    #[wasm_bindgen(js_name = get)]
    pub fn get(&self, key: &str) -> Result<Block, JsValue> {
        if !self.has(key) {
            return Err(JsValue::from_str(&format!("block '{key}' not found")));
        }
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

    /// Whether a block named `key` exists.
    #[wasm_bindgen(js_name = has)]
    pub fn has(&self, key: &str) -> bool {
        self.inner
            .store
            .borrow()
            .with_frame(self.inner.id, |f| f.contains_key(key))
            .unwrap_or(false)
    }

    /// Store a deep copy of `block` under `key`, replacing any block there.
    ///
    /// The source is copied: later writes to `block` do not reach this frame,
    /// and `block` stays usable.
    ///
    /// # Errors
    ///
    /// Throws if either the source block or this frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.set("atoms", other.get("atoms"));
    /// ```
    #[wasm_bindgen(js_name = set)]
    pub fn set(&self, key: &str, block: &Block) -> Result<(), JsValue> {
        let rs_block = block.inner.clone_block().map_err(js_err)?;
        self.inner
            .store
            .borrow_mut()
            .set_block(self.inner.id, key, rs_block)
            .map_err(js_err)
    }

    /// Remove the block named `key`.
    ///
    /// # Errors
    ///
    /// Throws if the frame has been dropped or `key` does not exist.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.remove("bonds");
    /// ```
    #[wasm_bindgen(js_name = remove)]
    pub fn remove(&self, key: &str) -> Result<(), JsValue> {
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
    /// Read a per-frame metadata value that is a string.
    ///
    /// The counterpart of [`setMeta`](Self::set_meta), and the accessor for
    /// labels that are words rather than numbers — a LAMMPS `dump local`
    /// section label (`dump_local_label`), say. Deliberately does not
    /// stringify numeric meta: those have their own accessor
    /// ([`getMetaScalar`](Self::get_meta_scalar)), and a getter that rendered
    /// every dtype would make `"1"` and `1` indistinguishable to the caller.
    ///
    /// Returns `undefined` when the key is missing, holds a non-string value,
    /// or the frame has been dropped.
    ///
    /// # Arguments
    ///
    /// * `name` — Meta key to look up (e.g., `"dump_local_label"`, `"note"`).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// if (frame.getMeta("dump_local_label") === "BONDS") {
    ///   // the local rows are bond topology
    /// }
    /// ```
    #[wasm_bindgen(js_name = getMeta)]
    pub fn get_meta(&self, name: &str) -> Option<String> {
        self.inner
            .store
            .borrow()
            .with_frame(self.inner.id, |frame| {
                frame
                    .meta
                    .get(name)
                    .and_then(MetaValue::as_str)
                    .map(str::to_owned)
            })
            .ok()?
    }

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
                    MetaValue::F64(value) => Some(*value),
                    MetaValue::String(value) => value.parse::<f64>().ok(),
                    _ => None,
                })
            })
            .ok()?
    }

    /// Return the names of all metadata keys on this frame, in insertion order.
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

    /// Names of the blocks in this frame, in insertion order (the order the
    /// file or the caller added them). Empty if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const names = frame.keys(); // e.g. ["atoms", "bonds"]
    /// ```
    #[wasm_bindgen(js_name = keys)]
    pub fn keys(&self) -> Vec<String> {
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
            .with_frame_meta_mut(self.inner.id, |meta| {
                meta.insert(
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
            .with_frame_meta_mut(self.inner.id, |meta| {
                meta.insert(name.to_string(), molrs::store::meta::MetaValue::F64(value));
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
    fn get_returns_a_live_block_and_throws_when_missing() {
        use wasm_bindgen::JsCast;
        let frame = Frame::new();
        assert!(!frame.has("atoms"));
        let e = frame.get("atoms").err().expect("missing block throws");
        assert!(e.as_string().unwrap().contains("'atoms' not found"));

        frame.create_block("atoms").unwrap();
        assert!(frame.has("atoms"));
        let mut atoms = frame.get("atoms").unwrap();
        let x = js_sys::Float64Array::from(&[1.0, 2.0][..]);
        atoms
            .set("x", JsValue::from(x).unchecked_into(), None)
            .unwrap();
        // A second handle sees the write: `get` is not a copy.
        assert_eq!(frame.get("atoms").unwrap().nrows().unwrap(), 2);
    }

    #[wasm_bindgen_test]
    fn set_copies_and_leaves_the_source_usable() {
        use wasm_bindgen::JsCast;
        let src = Frame::new();
        let mut atoms = src.create_block("atoms").unwrap();
        let x = js_sys::Float64Array::from(&[1.0][..]);
        atoms
            .set("x", JsValue::from(x).unchecked_into(), None)
            .unwrap();

        let dst = Frame::new();
        dst.set("atoms", &atoms).unwrap();
        let y = js_sys::Float64Array::from(&[2.0][..]);
        atoms
            .set("y", JsValue::from(y).unchecked_into(), None)
            .unwrap();
        assert!(!dst.get("atoms").unwrap().has("y"));
        assert_eq!(dst.keys(), vec!["atoms".to_string()]);

        dst.remove("atoms").unwrap();
        assert!(!dst.has("atoms"));
        assert!(dst.remove("atoms").is_err());
    }

    #[wasm_bindgen_test]
    fn meta_names_follow_insertion_order() {
        let frame = frame_with_meta();
        assert_eq!(
            frame.meta_names(),
            vec!["energy".to_string(), "config".to_string()]
        );
    }
}
