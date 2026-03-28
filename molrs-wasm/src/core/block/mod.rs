//! WASM bindings for [`Block`] -- typed columnar data container.
//!
//! A `Block` stores named columns, each backed by a homogeneously typed
//! array. All columns within a block share the same row count (`nrows`).
//!
//! # Column access API
//!
//! The API follows a consistent naming convention with type suffixes:
//!
//! | Method | JS signature | Semantics | Throws on missing key? |
//! |--------|-------------|-----------|------------------------|
//! | `setColF` | `(key: string, data: Float32Array|Float64Array, shape?: number[])` | Write float column | No |
//! | `setColI32` | `(key: string, data: Int32Array)` | Write i32 column | No |
//! | `setColU32` | `(key: string, data: Uint32Array)` | Write u32 column | No |
//! | `setColStr` | `(key: string, data: string[])` | Write string column | No |
//! | `viewCol{T}` | `(key: string) -> TypedArray` | Zero-copy view (invalidated on WASM memory growth) | Yes |
//! | `copyCol{T}` | `(key: string) -> TypedArray` | Owned JS copy (safe to keep) | Yes |
//!
//! # Memory safety note
//!
//! `viewCol*` methods return zero-copy typed array views backed by WASM
//! linear memory. These views become **invalid** if WASM memory grows
//! (e.g., due to any allocation). Use `copyCol*` if you need to keep
//! the data across allocations.

use js_sys::{Array as JsArray, Int32Array, Uint32Array};
use ndarray::Array1;
use wasm_bindgen::prelude::*;

use molrs::block::{Block as RsBlock, DType};
use molrs::types::F;
use molrs_ffi::BlockHandle as FFIBlockHandle;

use super::types::{FLOAT_DTYPE_NAME, JsFloatArray};
use super::{SharedStore, js_err};

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// Column-oriented data store with typed arrays.
///
/// Each column is identified by a string key and has a fixed data type
/// (`F`, `i32`, `u32`, `string`). All columns in a block must have
/// the same number of rows.
///
/// # Supported column types
///
/// | JS type | Rust type | dtype string | Setter | Getter (copy) | Getter (view) |
/// |---------|-----------|-------------|--------|---------------|---------------|
/// | `Float32Array` / `Float64Array` | `F` | `"f32"` / `"f64"` | `setColF` | `copyColF` | `viewColF` |
/// | `Int32Array` | `i32` | `"i32"` | `setColI32` | `copyColI32` | `viewColI32` |
/// | `Uint32Array` | `u32` | `"u32"` | `setColU32` | `copyColU32` | `viewColU32` |
/// | `string[]` | `String` | `"string"` | `setColStr` | `copyColStr` | -- |
///
/// # Example (JavaScript)
///
/// ```js
/// const block = new Block();
/// block.setColF("x", coordsX);
/// block.setColF("y", coordsY);
/// console.log(block.nrows()); // 3
/// console.log(block.keys());  // ["x", "y"]
///
/// const x = block.copyColF("x"); // owned copy, safe to keep
/// ```
#[wasm_bindgen]
pub struct Block {
    pub(crate) handle: FFIBlockHandle,
    pub(crate) store: SharedStore,
}

#[wasm_bindgen]
impl Block {
    /// Create a new, standalone empty `Block`.
    ///
    /// The block is backed by its own temporary store. Prefer
    /// [`Frame.createBlock()`](crate::Frame::create_block) to create
    /// blocks that are immediately attached to a frame.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the internal store allocation fails.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const block = new Block();
    /// block.setColF("values", values);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Block, JsValue> {
        let store = std::rc::Rc::new(std::cell::RefCell::new(molrs_ffi::Store::new()));
        let fid = store.borrow_mut().frame_new();
        store
            .borrow_mut()
            .set_block(fid, "temp", RsBlock::new())
            .map_err(js_err)?;
        let handle = store.borrow().get_block(fid, "temp").map_err(js_err)?;
        Ok(Block { handle, store })
    }

    // ---- metadata ----

    /// Return the number of columns in this block.
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(block.len()); // e.g., 3
    /// ```
    #[wasm_bindgen(js_name = len)]
    pub fn len(&self) -> Result<usize, JsValue> {
        self.with(|b| b.len())
    }

    /// Check whether this block has zero columns.
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// if (block.isEmpty()) { // no columns yet }
    /// ```
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }

    /// Return the number of rows (shared across all columns).
    ///
    /// Returns `0` if the block has no columns.
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(block.nrows()); // e.g., 100
    /// ```
    #[wasm_bindgen(js_name = nrows)]
    pub fn nrows(&self) -> Result<usize, JsValue> {
        self.with(|b| b.nrows().unwrap_or(0))
    }

    /// Return all column names as a JS `string[]`.
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const names = block.keys(); // ["x", "y", "z", "symbol"]
    /// ```
    #[wasm_bindgen(js_name = keys)]
    pub fn keys(&self) -> Result<JsArray, JsValue> {
        self.with(|b| {
            let arr = JsArray::new();
            for k in b.keys() {
                arr.push(&JsValue::from_str(k));
            }
            arr
        })
    }

    /// Return the data type string for a column.
    ///
    /// Possible return values: `"f32"` or `"f64"` for float columns,
    /// plus `"i32"`, `"u32"`, `"bool"`,
    /// `"string"`, `"u8"`. Returns `undefined` if the column does
    /// not exist.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name
    ///
    /// # Returns
    ///
    /// The dtype string, or `undefined` if the column is not found.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(block.dtype("x"));      // "f32" or "f64"
    /// console.log(block.dtype("symbol")); // "string"
    /// ```
    #[wasm_bindgen(js_name = dtype)]
    pub fn dtype(&self, key: &str) -> Option<String> {
        self.store
            .borrow()
            .with_block(&self.handle, |b| {
                b.dtype(key).map(|dt| {
                    match dt {
                        DType::Float => FLOAT_DTYPE_NAME,
                        DType::Int => "i32",
                        DType::UInt => "u32",
                        DType::U8 => "u8",
                        DType::Bool => "bool",
                        DType::String => "string",
                    }
                    .to_string()
                })
            })
            .ok()
            .flatten()
    }

    /// Rename a column from `old_key` to `new_key`.
    ///
    /// # Arguments
    ///
    /// * `old_key` - Current column name
    /// * `new_key` - New column name
    ///
    /// # Returns
    ///
    /// `true` if the column was found and renamed, `false` otherwise.
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// block.renameColumn("element", "symbol"); // true
    /// ```
    #[wasm_bindgen(js_name = renameColumn)]
    pub fn rename_column(&mut self, old_key: &str, new_key: &str) -> Result<bool, JsValue> {
        self.store
            .borrow_mut()
            .with_block_mut(&mut self.handle, |b| b.rename_column(old_key, new_key))
            .map_err(js_err)
    }

    // ---- Float ----

    /// Set a float column from a JS float typed array.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name (e.g., `"x"`, `"mass"`, `"charge"`)
    /// * `data` - JS float typed array with the column values
    /// * `shape` - Optional shape array for multi-dimensional data
    ///   (e.g., `[N, 3]` for an Nx3 matrix stored flat). If omitted,
    ///   the data is stored as a 1D column.
    ///
    /// # Errors
    ///
    /// Throws if `shape` product does not match `data.length`, or if
    /// the resulting row count is inconsistent with existing columns.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// block.setColF("x", xCoords);
    /// // Multi-dimensional: 2 rows x 3 columns
    /// block.setColF("pos", positions, [2, 3]);
    /// ```
    #[wasm_bindgen(js_name = setColF)]
    pub fn set_col_f(
        &mut self,
        key: &str,
        data: &JsFloatArray,
        shape: Option<Box<[usize]>>,
    ) -> Result<(), JsValue> {
        self.insert_float(key, data.to_vec(), shape)
    }

    /// Zero-copy JS float typed-array view into WASM linear memory.
    ///
    /// Returns a view backed directly by the block's storage in WASM
    /// memory. This avoids copying but the view becomes **invalid**
    /// if WASM linear memory grows (due to any allocation).
    ///
    /// Use [`copyColF`](Block::copy_col_f) for a safe, long-lived copy.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name
    ///
    /// # Returns
    ///
    /// A JS float typed-array view into WASM memory.
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist or is not of the active float type.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const view = block.viewColF("x"); // zero-copy, use immediately
    /// const copy = block.copyColF("x"); // safe to keep
    /// ```
    #[wasm_bindgen(js_name = viewColF)]
    pub fn view_col_f(&self, key: &str) -> Result<JsFloatArray, JsValue> {
        self.store
            .borrow()
            .borrow_col_F(&self.handle, key, |s, _| unsafe { JsFloatArray::view(s) })
            .map_err(|e| col_not_found_or(key, FLOAT_DTYPE_NAME, e))
    }

    /// Owned JS float typed-array copy of a column.
    ///
    /// Returns a new JS float typed array that is an independent copy of
    /// the column data. Safe to store and use across allocations.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name
    ///
    /// # Returns
    ///
    /// An owned JS float typed-array copy of the column.
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist or is not of the active float type.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const x = block.copyColF("x");
    /// console.log(x[0]); // 1.0
    /// ```
    #[wasm_bindgen(js_name = copyColF)]
    pub fn copy_col_f(&self, key: &str) -> Result<JsFloatArray, JsValue> {
        self.with(|b| {
            b.get_float(key)
                .and_then(|arr| arr.as_slice_memory_order())
                .map(JsFloatArray::from)
                .ok_or_else(|| col_err(key, FLOAT_DTYPE_NAME))
        })?
    }

    // ---- I32 ----

    /// Set a signed integer column from an `Int32Array`.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name
    /// * `data` - `Int32Array` with the column values
    ///
    /// # Errors
    ///
    /// Throws if the row count is inconsistent with existing columns.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// block.setColI32("charge_sign", new Int32Array([1, -1, 0]));
    /// ```
    #[wasm_bindgen(js_name = setColI32)]
    pub fn set_col_i32(&mut self, key: &str, data: &Int32Array) -> Result<(), JsValue> {
        let vec: Vec<i32> = data.to_vec();
        self.insert_col(key, Array1::from(vec).into_dyn())
    }

    /// Zero-copy `Int32Array` view into WASM linear memory.
    ///
    /// **Warning**: invalidated if WASM linear memory grows.
    /// Use [`copyColI32`](Block::copy_col_i32) for a safe copy.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist or is not of type `i32`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const view = block.viewColI32("type_id");
    /// ```
    #[wasm_bindgen(js_name = viewColI32)]
    pub fn view_col_i32(&self, key: &str) -> Result<Int32Array, JsValue> {
        self.store
            .borrow()
            .borrow_col_I(&self.handle, key, |s, _| unsafe { Int32Array::view(s) })
            .map_err(|e| col_not_found_or(key, "i32", e))
    }

    /// Owned `Int32Array` copy of a column.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist or is not of type `i32`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const types = block.copyColI32("type_id");
    /// ```
    #[wasm_bindgen(js_name = copyColI32)]
    pub fn copy_col_i32(&self, key: &str) -> Result<Int32Array, JsValue> {
        self.with(|b| {
            b.get_int(key)
                .and_then(|arr| arr.as_slice_memory_order())
                .map(Int32Array::from)
                .ok_or_else(|| col_err(key, "i32"))
        })?
    }

    // ---- U32 ----

    /// Set an unsigned integer column from a `Uint32Array`.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name (e.g., `"i"`, `"j"` for bond indices)
    /// * `data` - `Uint32Array` with the column values
    ///
    /// # Errors
    ///
    /// Throws if the row count is inconsistent with existing columns.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// // Bond topology: atom indices
    /// bonds.setColU32("i", new Uint32Array([0, 1]));
    /// bonds.setColU32("j", new Uint32Array([1, 2]));
    /// ```
    #[wasm_bindgen(js_name = setColU32)]
    pub fn set_col_u32(&mut self, key: &str, data: &Uint32Array) -> Result<(), JsValue> {
        let vec: Vec<u32> = data.to_vec();
        self.insert_col(key, Array1::from(vec).into_dyn())
    }

    /// Zero-copy `Uint32Array` view into WASM linear memory.
    ///
    /// **Warning**: invalidated if WASM linear memory grows.
    /// Use [`copyColU32`](Block::copy_col_u32) for a safe copy.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist or is not of type `u32`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const view = block.viewColU32("i");
    /// ```
    #[wasm_bindgen(js_name = viewColU32)]
    pub fn view_col_u32(&self, key: &str) -> Result<Uint32Array, JsValue> {
        self.store
            .borrow()
            .borrow_col_U(&self.handle, key, |s, _| unsafe { Uint32Array::view(s) })
            .map_err(|e| col_not_found_or(key, "u32", e))
    }

    /// Owned `Uint32Array` copy of a column.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist or is not of type `u32`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const bondI = block.copyColU32("i");
    /// const bondJ = block.copyColU32("j");
    /// ```
    #[wasm_bindgen(js_name = copyColU32)]
    pub fn copy_col_u32(&self, key: &str) -> Result<Uint32Array, JsValue> {
        self.with(|b| {
            b.get_uint(key)
                .and_then(|arr| arr.as_slice_memory_order())
                .map(Uint32Array::from)
                .ok_or_else(|| col_err(key, "u32"))
        })?
    }

    // ---- Str ----

    /// Set a string column from a JS `string[]`.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name (e.g., `"symbol"`, `"name"`)
    /// * `data` - JS `Array` where every element must be a string
    ///
    /// # Errors
    ///
    /// Throws if any element is not a string, or if the row count is
    /// inconsistent with existing columns.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// atoms.setColStr("symbol", ["C", "C", "O"]);
    /// ```
    #[wasm_bindgen(js_name = setColStr)]
    pub fn set_col_str(&mut self, key: &str, data: JsArray) -> Result<(), JsValue> {
        let mut strings = Vec::with_capacity(data.length() as usize);
        for (i, v) in data.iter().enumerate() {
            strings.push(v.as_string().ok_or_else(|| {
                JsValue::from_str(&format!("element at index {i} is not a string"))
            })?);
        }
        self.insert_col(key, Array1::from(strings).into_dyn())
    }

    /// Owned `string[]` copy of a string column.
    ///
    /// # Arguments
    ///
    /// * `key` - Column name
    ///
    /// # Returns
    ///
    /// A JS `Array` of strings.
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist or is not of type `string`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const symbols = block.copyColStr("symbol"); // ["C", "C", "O"]
    /// ```
    #[wasm_bindgen(js_name = copyColStr)]
    pub fn copy_col_str(&self, key: &str) -> Result<JsArray, JsValue> {
        self.with(|b| {
            b.get_string(key)
                .map(|arr| {
                    let js = JsArray::new();
                    for s in arr.iter() {
                        js.push(&JsValue::from_str(s));
                    }
                    js
                })
                .ok_or_else(|| col_err(key, "string"))
        })?
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new().expect("Block::new on fresh store")
    }
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn col_err(key: &str, dtype: &str) -> JsValue {
    JsValue::from_str(&format!("column '{key}' not found or not {dtype}"))
}

fn col_not_found_or(key: &str, dtype: &str, ffi_err: molrs_ffi::FfiError) -> JsValue {
    JsValue::from_str(&format!("column '{key}' ({dtype}): {ffi_err}"))
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl Block {
    fn with<R>(&self, f: impl FnOnce(&RsBlock) -> R) -> Result<R, JsValue> {
        self.store
            .borrow()
            .with_block(&self.handle, f)
            .map_err(js_err)
    }

    fn insert_col<T: molrs::block::BlockDtype>(
        &mut self,
        key: &str,
        array: ndarray::ArrayD<T>,
    ) -> Result<(), JsValue> {
        self.store
            .borrow_mut()
            .with_block_mut(&mut self.handle, |b| {
                b.insert(key, array)
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            })
            .map_err(js_err)?
    }

    fn insert_float(
        &mut self,
        key: &str,
        data: Vec<F>,
        shape: Option<Box<[usize]>>,
    ) -> Result<(), JsValue> {
        let array = if let Some(dims) = shape {
            let dims_vec = Vec::from(dims);
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims_vec), data)
                .map_err(|e| JsValue::from_str(&e.to_string()))?
        } else {
            Array1::from(data).into_dyn()
        };
        self.insert_col(key, array)
    }

    pub(crate) fn set_owned_column(
        &mut self,
        key: &str,
        data: Vec<F>,
        shape: Box<[usize]>,
    ) -> Result<(), JsValue> {
        self.insert_float(key, data, Some(shape))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::frame::Frame;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_block_set_copy_f() {
        let frame = Frame::new();
        let mut block = frame.create_block("atoms").unwrap();

        let data = JsFloatArray::from(&[1.0, 2.0, 3.0][..]);
        block.set_col_f("x", &data, None).unwrap();
        let copied = block.copy_col_f("x").unwrap();
        assert_eq!(copied.length(), 3);
        assert_eq!(copied.get_index(0), 1.0);
    }

    #[wasm_bindgen_test]
    fn test_block_set_copy_u32() {
        let frame = Frame::new();
        let mut block = frame.create_block("atoms").unwrap();

        let ids = Uint32Array::from(&[7_u32, 8_u32][..]);
        block.set_col_u32("id", &ids).unwrap();
        let copied = block.copy_col_u32("id").unwrap();
        assert_eq!(copied.length(), 2);
    }

    #[wasm_bindgen_test]
    fn test_block_set_copy_i32() {
        let frame = Frame::new();
        let mut block = frame.create_block("atoms").unwrap();

        let charges = Int32Array::from(&[1_i32, -2][..]);
        block.set_col_i32("charge", &charges).unwrap();
        let copied = block.copy_col_i32("charge").unwrap();
        assert_eq!(copied.length(), 2);
    }

    #[wasm_bindgen_test]
    fn test_missing_key_throws() {
        let frame = Frame::new();
        let block = frame.create_block("atoms").unwrap();
        assert!(block.copy_col_f("nonexistent").is_err());
    }
}
