//! Shared-ownership wrappers pairing handles with a SharedStore.
//!
//! Every language binding (wasm, python, capi) needs a way to hold a handle
//! alongside a shared reference to the [`Store`]. Before this module each
//! binding re-implemented the `(handle, Rc<RefCell<Store>>)` tuple together
//! with a grab-bag of borrow helpers — duplicated, subtly divergent glue.
//!
//! [`FrameRef`] and [`BlockRef`] replace that pattern: bindings just hold
//! one of these and forward every access through the shared helpers defined
//! here. The bindings stay thin (attribute macros + name mapping) and the
//! canonical schema lookup / dtype dispatch / slice borrowing live in one
//! place.
//!
//! ## Access semantics
//!
//! Column accessors return [`Result<Option<T>, FfiError>`], splitting two
//! semantically different conditions:
//!
//! * `Ok(None)` — the column does **not** exist. This is a normal, expected
//!   outcome because molrs schemas are flexible (e.g. LAMMPS data has no
//!   `element`). Callers handle it with a simple `if let Some(..)`.
//! * `Err(DTypeMismatch)` — the column **exists but has the wrong dtype**
//!   for the accessor called. Almost always a schema-level caller bug.
//! * `Err(InvalidBlockHandle)` / `Err(NonContiguous)` — structural errors
//!   (stale handle, non-memory-contiguous view). Propagate as-is.
//!
//! ## Threading
//!
//! [`SharedStore`] uses `Rc<RefCell<_>>`. All current FFI consumers (wasm
//! is single-threaded; python holds the GIL; capi is used from
//! single-threaded C/C++ today) satisfy this. Native multi-threaded Rust
//! consumers should hold the raw [`Store`] directly or wrap it themselves
//! in `Arc<Mutex<_>>`.

use std::cell::RefCell;
use std::rc::Rc;

use molrs::block::{Block, DType};
use molrs::frame::Frame;
use molrs::region::simbox::SimBox;
use molrs::types::{F, I, U};

use crate::error::FfiError;
use crate::handle::{BlockHandle, FrameId};
use crate::store::Store;

/// Single-threaded shared ownership of a [`Store`].
pub type SharedStore = Rc<RefCell<Store>>;

/// Create a new empty [`SharedStore`].
pub fn new_shared() -> SharedStore {
    Rc::new(RefCell::new(Store::new()))
}

// =====================================================================
// FrameRef
// =====================================================================

/// A paired [`FrameId`] + [`SharedStore`]. Cheap to clone (two `Rc` bumps).
///
/// Bindings wrap this as their `Frame` type and add language-specific
/// attributes (e.g. `#[wasm_bindgen]`, `#[pyclass]`).
#[derive(Clone)]
pub struct FrameRef {
    pub id: FrameId,
    pub store: SharedStore,
}

impl FrameRef {
    /// Wrap an existing store + id.
    pub fn new(store: SharedStore, id: FrameId) -> Self {
        Self { id, store }
    }

    /// Create a new empty frame inside a fresh [`SharedStore`].
    pub fn new_standalone() -> Self {
        let store = new_shared();
        let id = store.borrow_mut().frame_new();
        Self { id, store }
    }

    /// Run a closure with immutable access to the underlying [`Frame`].
    pub fn with<R>(&self, f: impl FnOnce(&Frame) -> R) -> Result<R, FfiError> {
        self.store.borrow().with_frame(self.id, f)
    }

    /// Run a closure with mutable access to the underlying [`Frame`].
    ///
    /// Conservatively invalidates every block handle on this frame when the
    /// closure returns, since `&mut Frame` permits arbitrary block
    /// modifications.
    pub fn with_mut<R>(&self, f: impl FnOnce(&mut Frame) -> R) -> Result<R, FfiError> {
        self.store.borrow_mut().with_frame_mut(self.id, f)
    }

    /// Resolve a child block key into a [`BlockRef`]. Returns
    /// `Err(KeyNotFound)` if the key is absent from this frame.
    pub fn block(&self, key: &str) -> Result<BlockRef, FfiError> {
        let handle = self.store.borrow().get_block(self.id, key)?;
        Ok(BlockRef::new(Rc::clone(&self.store), handle))
    }

    /// True if the frame has a block at `key`.
    pub fn has_block(&self, key: &str) -> bool {
        self.with(|f| f.contains_key(key)).unwrap_or(false)
    }

    /// Clone the simbox out of the frame (if any).
    pub fn simbox_clone(&self) -> Result<Option<SimBox>, FfiError> {
        self.store
            .borrow()
            .with_frame_simbox(self.id, |sb| sb.cloned())
    }

    /// Replace / clear the simbox.
    pub fn set_simbox(&self, simbox: Option<SimBox>) -> Result<(), FfiError> {
        self.store.borrow_mut().set_frame_simbox(self.id, simbox)
    }

    /// Deep-clone the frame's data out of the store.
    pub fn clone_frame(&self) -> Result<Frame, FfiError> {
        self.store.borrow().clone_frame(self.id)
    }

    /// Drop the frame from the store, invalidating every handle to it.
    pub fn drop_frame(self) -> Result<(), FfiError> {
        self.store.borrow_mut().frame_drop(self.id)
    }
}

// =====================================================================
// BlockRef
// =====================================================================

/// A paired [`BlockHandle`] + [`SharedStore`]. Cheap to clone.
///
/// Bindings wrap this as their `Block` type. All column-access helpers
/// live on this type so the bindings don't reinvent dtype dispatch.
#[derive(Clone)]
pub struct BlockRef {
    pub handle: BlockHandle,
    pub store: SharedStore,
}

impl BlockRef {
    pub fn new(store: SharedStore, handle: BlockHandle) -> Self {
        Self { handle, store }
    }

    /// Run a closure with immutable access to the underlying [`Block`].
    pub fn with<R>(&self, f: impl FnOnce(&Block) -> R) -> Result<R, FfiError> {
        self.store.borrow().with_block(&self.handle, f)
    }

    /// Run a closure with mutable access; bumps the handle version.
    pub fn with_mut<R>(&mut self, f: impl FnOnce(&mut Block) -> R) -> Result<R, FfiError> {
        self.store
            .borrow_mut()
            .with_block_mut(&mut self.handle, f)
    }

    // ---- Metadata ----

    /// Number of rows. Returns 0 for an empty block (never Err except on
    /// handle invalidation).
    pub fn nrows(&self) -> Result<usize, FfiError> {
        self.with(|b| b.nrows().unwrap_or(0))
    }

    /// Column names currently present on this block.
    pub fn keys(&self) -> Result<Vec<String>, FfiError> {
        self.with(|b| b.keys().map(|s| s.to_string()).collect())
    }

    /// `true` if the column exists.
    pub fn has(&self, key: &str) -> Result<bool, FfiError> {
        self.with(|b| b.contains_key(key))
    }

    /// Column dtype. `Ok(None)` if the column doesn't exist.
    pub fn dtype(&self, key: &str) -> Result<Option<DType>, FfiError> {
        self.with(|b| b.dtype(key))
    }

    /// Column shape. `Ok(None)` if the column doesn't exist. For molrs 1-D
    /// columns this is `[nrows]`; multi-dim columns (if any) carry their
    /// full shape.
    pub fn shape(&self, key: &str) -> Result<Option<Vec<usize>>, FfiError> {
        self.with(|b| match b.dtype(key) {
            None => None,
            Some(DType::Float) => b.get_float(key).map(|a| a.shape().to_vec()),
            Some(DType::Int) => b.get_int(key).map(|a| a.shape().to_vec()),
            Some(DType::UInt) => b.get_uint(key).map(|a| a.shape().to_vec()),
            Some(DType::U8) => b.get_u8(key).map(|a| a.shape().to_vec()),
            Some(DType::Bool) => b.get_bool(key).map(|a| a.shape().to_vec()),
            Some(DType::String) => b.get_string(key).map(|a| a.shape().to_vec()),
        })
    }

    /// Deep-clone the block data out of the store.
    pub fn clone_block(&self) -> Result<Block, FfiError> {
        self.store.borrow().clone_block(&self.handle)
    }

    // ---- Typed column borrows (zero-copy closures) ----

    /// Borrow an `F` column as a contiguous slice.
    ///
    /// * `Ok(None)` — column is absent (schema flexibility).
    /// * `Ok(Some(R))` — column is `F`, closure ran, result returned.
    /// * `Err(DTypeMismatch)` — column exists but isn't `F`.
    /// * `Err(NonContiguous)` — column is `F` but not contiguous.
    /// * `Err(InvalidBlockHandle)` — handle stale.
    pub fn borrow_f<R>(
        &self,
        key: &str,
        f: impl FnOnce(&[F], &[usize]) -> R,
    ) -> Result<Option<R>, FfiError> {
        self.with(|b| -> Result<Option<R>, FfiError> {
            match b.dtype(key) {
                None => Ok(None),
                Some(DType::Float) => {
                    let arr = b.get_float(key).ok_or(FfiError::InvalidBlockHandle)?;
                    let slice = arr.as_slice_memory_order().ok_or_else(|| {
                        FfiError::NonContiguous {
                            key: key.to_string(),
                        }
                    })?;
                    Ok(Some(f(slice, arr.shape())))
                }
                Some(actual) => Err(FfiError::DTypeMismatch {
                    key: key.to_string(),
                    expected: DType::Float,
                    actual,
                }),
            }
        })?
    }

    /// Borrow an `I` column as a contiguous slice. See [`borrow_f`](Self::borrow_f).
    pub fn borrow_i<R>(
        &self,
        key: &str,
        f: impl FnOnce(&[I], &[usize]) -> R,
    ) -> Result<Option<R>, FfiError> {
        self.with(|b| -> Result<Option<R>, FfiError> {
            match b.dtype(key) {
                None => Ok(None),
                Some(DType::Int) => {
                    let arr = b.get_int(key).ok_or(FfiError::InvalidBlockHandle)?;
                    let slice = arr.as_slice_memory_order().ok_or_else(|| {
                        FfiError::NonContiguous {
                            key: key.to_string(),
                        }
                    })?;
                    Ok(Some(f(slice, arr.shape())))
                }
                Some(actual) => Err(FfiError::DTypeMismatch {
                    key: key.to_string(),
                    expected: DType::Int,
                    actual,
                }),
            }
        })?
    }

    /// Borrow a `U` column as a contiguous slice. See [`borrow_f`](Self::borrow_f).
    pub fn borrow_u<R>(
        &self,
        key: &str,
        f: impl FnOnce(&[U], &[usize]) -> R,
    ) -> Result<Option<R>, FfiError> {
        self.with(|b| -> Result<Option<R>, FfiError> {
            match b.dtype(key) {
                None => Ok(None),
                Some(DType::UInt) => {
                    let arr = b.get_uint(key).ok_or(FfiError::InvalidBlockHandle)?;
                    let slice = arr.as_slice_memory_order().ok_or_else(|| {
                        FfiError::NonContiguous {
                            key: key.to_string(),
                        }
                    })?;
                    Ok(Some(f(slice, arr.shape())))
                }
                Some(actual) => Err(FfiError::DTypeMismatch {
                    key: key.to_string(),
                    expected: DType::UInt,
                    actual,
                }),
            }
        })?
    }

    // ---- String column (always owned copy — strings aren't in contiguous
    // scalar memory, so zero-copy isn't meaningful) ----

    /// Copy a `String` column. See [`borrow_f`](Self::borrow_f) for the
    /// Ok/Err semantics.
    pub fn col_str(&self, key: &str) -> Result<Option<Vec<String>>, FfiError> {
        self.with(|b| -> Result<Option<Vec<String>>, FfiError> {
            match b.dtype(key) {
                None => Ok(None),
                Some(DType::String) => {
                    let arr = b.get_string(key).ok_or(FfiError::InvalidBlockHandle)?;
                    Ok(Some(arr.iter().cloned().collect()))
                }
                Some(actual) => Err(FfiError::DTypeMismatch {
                    key: key.to_string(),
                    expected: DType::String,
                    actual,
                }),
            }
        })?
    }

    // ---- Owned copies of numeric columns ----

    /// Owned copy of an `F` column. Same dtype semantics as [`borrow_f`].
    pub fn copy_f(&self, key: &str) -> Result<Option<(Vec<F>, Vec<usize>)>, FfiError> {
        self.borrow_f(key, |slice, shape| (slice.to_vec(), shape.to_vec()))
    }

    /// Owned copy of an `I` column.
    pub fn copy_i(&self, key: &str) -> Result<Option<(Vec<I>, Vec<usize>)>, FfiError> {
        self.borrow_i(key, |slice, shape| (slice.to_vec(), shape.to_vec()))
    }

    /// Owned copy of a `U` column.
    pub fn copy_u(&self, key: &str) -> Result<Option<(Vec<U>, Vec<usize>)>, FfiError> {
        self.borrow_u(key, |slice, shape| (slice.to_vec(), shape.to_vec()))
    }
}
