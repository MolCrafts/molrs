//! Internal column representation for heterogeneous data.
//!
//! # Storage model
//!
//! `Column` variants wrap `Arc<ColumnHolder<T>>` so cloning a Column is an
//! O(1) refcount bump rather than a deep copy. The [`ColumnHolder`] is a thin
//! wrapper that makes the underlying `ArrayD<T>` either:
//!
//! * **Rust-owned** — the normal path; `ArrayD<T>`'s backing `Vec<T>` was
//!   allocated by Rust's allocator and is dropped normally when the holder
//!   drops.
//! * **Foreign-borrowed** — the buffer was allocated by *some other* allocator
//!   (e.g., numpy's). The holder fakes an `ArrayD<T>` pointing at that memory
//!   and skips the `Vec::drop` on holder drop. Instead, it holds an opaque
//!   "keep-alive" object (e.g., a `Py<PyArrayDyn<T>>`) whose own `Drop` is
//!   responsible for releasing the memory via the foreign allocator.
//!
//! Readers don't need to know which storage is active: `as_float()`,
//! `shape()`, `view()` etc work identically. Writers go through
//! [`ColumnHolder::realize_owned_mut`] which always detaches from foreign
//! storage (copy-on-write) before returning a mutable reference, so mutation
//! never reaches into foreign memory.

use std::any::Any;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::Arc;

use ndarray::ArrayD;

use num_complex::Complex;

use super::dtype::DType;
use crate::types::{F, I, Idx};

/// Walk every column variant, binding the inner holder.
macro_rules! map_column {
    ($col:expr, $holder:ident => $body:expr) => {
        match $col {
            Column::Float16($holder) => $body,
            Column::Float32($holder) => $body,
            Column::Float($holder) => $body,
            Column::Int8($holder) => $body,
            Column::Int16($holder) => $body,
            Column::Int($holder) => $body,
            Column::Int64($holder) => $body,
            Column::U8($holder) => $body,
            Column::UInt16($holder) => $body,
            Column::UInt32($holder) => $body,
            Column::UInt($holder) => $body,
            Column::Bool($holder) => $body,
            Column::String($holder) => $body,
            Column::Complex64($holder) => $body,
            Column::Complex128($holder) => $body,
        }
    };
}

/// Wrapper around `ArrayD<T>` that optionally defers buffer ownership to a
/// foreign allocator.
///
/// See the module-level docs for the full story. Readers access the inner
/// array via `Deref<Target = ArrayD<T>>`. Writers must go through
/// `realize_owned_mut` to ensure mutation never touches foreign memory.
pub struct ColumnHolder<T> {
    array: ManuallyDrop<ArrayD<T>>,
    /// Optional keep-alive for a foreign-allocated buffer.
    ///
    /// When `Some`, the `array` field's `Vec<T>` points at memory managed by
    /// some other allocator (e.g., numpy). On drop we must NOT run
    /// `Vec::drop` on that buffer; we drop the keeper instead, and trust that
    /// the keeper's own `Drop` releases the buffer through its native API.
    ///
    /// When `None`, the `array` is Rust-owned and drops normally.
    foreign_keeper: Option<Box<dyn Any + Send + Sync>>,
}

impl<T> ColumnHolder<T> {
    /// Create a holder owning a Rust-allocated `ArrayD<T>`. This is the normal
    /// path: the holder drops the inner `ArrayD` normally.
    pub fn from_owned(arr: ArrayD<T>) -> Self {
        Self {
            array: ManuallyDrop::new(arr),
            foreign_keeper: None,
        }
    }

    /// Create a holder borrowing a foreign-allocated buffer.
    ///
    /// # Safety
    ///
    /// The caller must guarantee:
    ///
    /// * The `arr`'s backing memory was allocated by the same allocator that
    ///   `keeper`'s `Drop` impl will call into. Typically the caller built
    ///   `arr` via `Vec::from_raw_parts(ptr, len, len)` where `ptr` was
    ///   obtained from `keeper` (e.g., numpy array data pointer).
    /// * The `keeper` keeps the foreign memory alive for at least as long as
    ///   this holder exists.
    /// * The foreign buffer will not be mutated or reallocated while any
    ///   reader holds a reference to it through this holder.
    ///
    /// When the holder drops, the inner `ArrayD`'s `Vec` is *not* dropped
    /// (which would invoke Rust's allocator on foreign memory — UB). Instead,
    /// `keeper` is dropped, and its own `Drop` impl releases the memory.
    pub unsafe fn from_foreign<K: Any + Send + Sync>(arr: ArrayD<T>, keeper: K) -> Self {
        Self {
            array: ManuallyDrop::new(arr),
            foreign_keeper: Some(Box::new(keeper)),
        }
    }

    /// Is this holder backed by foreign memory?
    pub fn is_foreign(&self) -> bool {
        self.foreign_keeper.is_some()
    }

    /// Direct reference to the inner `ArrayD<T>`.
    #[inline]
    pub fn array(&self) -> &ArrayD<T> {
        &self.array
    }
}

impl<T> Deref for ColumnHolder<T> {
    type Target = ArrayD<T>;
    #[inline]
    fn deref(&self) -> &ArrayD<T> {
        &self.array
    }
}

impl<T: Clone> Clone for ColumnHolder<T> {
    /// Cloning always produces a **Rust-owned** holder (deep-copies the inner
    /// array out of any foreign buffer). This is the fundamental guarantee
    /// that lets `Arc::make_mut`-style APIs return a `&mut ArrayD<T>` without
    /// risking mutation of foreign memory.
    fn clone(&self) -> Self {
        Self::from_owned(ArrayD::clone(&self.array))
    }
}

impl<T> Drop for ColumnHolder<T> {
    fn drop(&mut self) {
        if let Some(keeper) = self.foreign_keeper.take() {
            // Foreign-backed: release the keep-alive, let its Drop free the
            // underlying buffer via the foreign allocator.
            // The inner `ArrayD`'s `Vec` is NOT dropped — that would call
            // Rust's allocator on foreign memory.
            drop(keeper);
        } else {
            // Rust-owned: drop the inner ArrayD normally.
            // SAFETY: `foreign_keeper` is None, so `array` is a real owned
            // ArrayD allocated by Rust's global allocator. We have not dropped
            // `array` previously (this is the only place we do).
            unsafe { ManuallyDrop::drop(&mut self.array) }
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for ColumnHolder<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnHolder")
            .field("shape", &self.array.shape())
            .field("foreign", &self.is_foreign())
            .finish()
    }
}

/// Internal enum representing a column of data in a Block.
///
/// Inner values are `Arc<ColumnHolder<T>>` so `Column::clone()` and
/// `Block::clone()` are cheap: they bump refcounts instead of copying
/// scalar data. The [`Column::from_float`] / [`Column::from_int`] / …
/// constructors wrap an owned `ArrayD<T>` in a Rust-owned holder.
///
/// Type-specific getters (`as_float`, `as_int`, …) return plain `&ArrayD<T>`
/// references via deref; the Arc+holder layering is transparent to callers.
/// Mutable getters go through [`ColumnHolder::clone`] copy-on-write so the
/// returned `&mut ArrayD<T>` always refers to Rust-owned memory.
#[derive(Clone)]
pub enum Column {
    /// IEEE binary16.
    Float16(Arc<ColumnHolder<half::f16>>),
    /// IEEE binary32.
    Float32(Arc<ColumnHolder<f32>>),
    /// Floating point column using the compute scalar [`F`] (`f64`).
    Float(Arc<ColumnHolder<F>>),
    /// Signed 8-bit integer column.
    Int8(Arc<ColumnHolder<i8>>),
    /// Signed 16-bit integer column.
    Int16(Arc<ColumnHolder<i16>>),
    /// Signed integer column using the domain scalar [`I`] (`i32`).
    Int(Arc<ColumnHolder<I>>),
    /// Signed 64-bit integer column.
    Int64(Arc<ColumnHolder<i64>>),
    /// Boolean column
    Bool(Arc<ColumnHolder<bool>>),
    /// Unsigned integer column using the identifier scalar [`Idx`] (`u64`).
    UInt(Arc<ColumnHolder<Idx>>),
    /// 8-bit unsigned integer column
    U8(Arc<ColumnHolder<u8>>),
    /// 16-bit unsigned integer column.
    UInt16(Arc<ColumnHolder<u16>>),
    /// 32-bit unsigned integer column.
    UInt32(Arc<ColumnHolder<u32>>),
    /// String column
    String(Arc<ColumnHolder<String>>),
    /// Complex pair of `f32` (numpy `complex64`).
    Complex64(Arc<ColumnHolder<Complex<f32>>>),
    /// Complex pair of `f64` (numpy `complex128`).
    Complex128(Arc<ColumnHolder<Complex<f64>>>),
}

/// Force an `Arc<ColumnHolder<T>>` to be (1) Rust-owned and (2) uniquely
/// referenced, so callers can mutate the inner `ArrayD<T>` safely. Clones if
/// necessary. Returns `&mut ArrayD<T>`.
fn realize_owned_mut<T: Clone>(arc: &mut Arc<ColumnHolder<T>>) -> &mut ArrayD<T> {
    // Step 1: if holder is foreign-backed, clone to detach from foreign memory.
    // Cloning always produces a Rust-owned holder (see ColumnHolder::clone).
    if arc.is_foreign() {
        *arc = Arc::new((**arc).clone());
    }
    // Step 2: make Arc unique via Arc::make_mut (clones holder if shared, which
    // in turn deep-clones the ArrayD — preserving current CoW semantics).
    let holder = Arc::make_mut(arc);
    // Step 3: holder is now guaranteed Rust-owned AND uniquely referenced.
    // ManuallyDrop's DerefMut gives auto-deref to &mut ArrayD<T>.
    &mut holder.array
}

impl Column {
    /// Wrap an owned float ndarray in a Rust-owned `Column`.
    pub fn from_float(arr: ArrayD<F>) -> Self {
        Column::Float(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned `f16` ndarray in a Rust-owned `Column`.
    pub fn from_f16(arr: ArrayD<half::f16>) -> Self {
        Column::Float16(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned `f32` ndarray in a Rust-owned `Column`.
    pub fn from_f32(arr: ArrayD<f32>) -> Self {
        Column::Float32(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned int ndarray in a Rust-owned `Column`.
    pub fn from_int(arr: ArrayD<I>) -> Self {
        Column::Int(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned `i8` ndarray in a Rust-owned `Column`.
    pub fn from_i8(arr: ArrayD<i8>) -> Self {
        Column::Int8(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned `i16` ndarray in a Rust-owned `Column`.
    pub fn from_i16(arr: ArrayD<i16>) -> Self {
        Column::Int16(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned `i64` ndarray in a Rust-owned `Column`.
    pub fn from_i64(arr: ArrayD<i64>) -> Self {
        Column::Int64(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned bool ndarray in a Rust-owned `Column`.
    pub fn from_bool(arr: ArrayD<bool>) -> Self {
        Column::Bool(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned uint ndarray in a Rust-owned `Column`.
    pub fn from_uint(arr: ArrayD<Idx>) -> Self {
        Column::UInt(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned u8 ndarray in a Rust-owned `Column`.
    pub fn from_u8(arr: ArrayD<u8>) -> Self {
        Column::U8(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned `u16` ndarray in a Rust-owned `Column`.
    pub fn from_u16(arr: ArrayD<u16>) -> Self {
        Column::UInt16(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned `u32` ndarray in a Rust-owned `Column`.
    pub fn from_u32(arr: ArrayD<u32>) -> Self {
        Column::UInt32(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned string ndarray in a Rust-owned `Column`.
    pub fn from_string(arr: ArrayD<String>) -> Self {
        Column::String(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned `complex64` ndarray in a Rust-owned `Column`.
    pub fn from_c64(arr: ArrayD<Complex<f32>>) -> Self {
        Column::Complex64(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap an owned `complex128` ndarray in a Rust-owned `Column`.
    pub fn from_c128(arr: ArrayD<Complex<f64>>) -> Self {
        Column::Complex128(Arc::new(ColumnHolder::from_owned(arr)))
    }

    /// Wrap a foreign-backed `ColumnHolder<F>` directly. Zero-copy path for
    /// bindings that have the holder pre-built (see
    /// [`ColumnHolder::from_foreign`]).
    pub fn from_float_holder(holder: ColumnHolder<F>) -> Self {
        Column::Float(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_f16_holder(holder: ColumnHolder<half::f16>) -> Self {
        Column::Float16(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_f32_holder(holder: ColumnHolder<f32>) -> Self {
        Column::Float32(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_int_holder(holder: ColumnHolder<I>) -> Self {
        Column::Int(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_i8_holder(holder: ColumnHolder<i8>) -> Self {
        Column::Int8(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_i16_holder(holder: ColumnHolder<i16>) -> Self {
        Column::Int16(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_i64_holder(holder: ColumnHolder<i64>) -> Self {
        Column::Int64(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_bool_holder(holder: ColumnHolder<bool>) -> Self {
        Column::Bool(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_uint_holder(holder: ColumnHolder<Idx>) -> Self {
        Column::UInt(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_u8_holder(holder: ColumnHolder<u8>) -> Self {
        Column::U8(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_u16_holder(holder: ColumnHolder<u16>) -> Self {
        Column::UInt16(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_u32_holder(holder: ColumnHolder<u32>) -> Self {
        Column::UInt32(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_string_holder(holder: ColumnHolder<String>) -> Self {
        Column::String(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_c64_holder(holder: ColumnHolder<Complex<f32>>) -> Self {
        Column::Complex64(Arc::new(holder))
    }

    /// See [`Column::from_float_holder`].
    pub fn from_c128_holder(holder: ColumnHolder<Complex<f64>>) -> Self {
        Column::Complex128(Arc::new(holder))
    }

    /// Returns the number of rows (axis-0 length) of this column.
    ///
    /// Returns `None` if the array has rank 0 (which should never happen
    /// in a valid Block, as rank-0 arrays are rejected during insertion).
    pub fn nrows(&self) -> Option<usize> {
        map_column!(self, a => a.shape().first().copied())
    }

    /// Returns the data type of this column.
    pub fn dtype(&self) -> DType {
        match self {
            Column::Float16(_) => DType::Float16,
            Column::Float32(_) => DType::Float32,
            Column::Float(_) => DType::Float,
            Column::Int8(_) => DType::Int8,
            Column::Int16(_) => DType::Int16,
            Column::Int(_) => DType::Int,
            Column::Int64(_) => DType::Int64,
            Column::Bool(_) => DType::Bool,
            Column::UInt(_) => DType::UInt,
            Column::U8(_) => DType::U8,
            Column::UInt16(_) => DType::UInt16,
            Column::UInt32(_) => DType::UInt32,
            Column::String(_) => DType::String,
            Column::Complex64(_) => DType::Complex64,
            Column::Complex128(_) => DType::Complex128,
        }
    }

    /// Returns the shape of the underlying array.
    pub fn shape(&self) -> &[usize] {
        map_column!(self, a => a.shape())
    }

    /// Owned little-endian byte buffer of this column's numeric backing store,
    /// in row-major (standard) layout.
    ///
    /// Returns `None` only for the [`Column::String`] variant, whose
    /// variable-length elements have no fixed byte representation. A strided or
    /// sliced numeric column is materialized into standard layout first, so any
    /// valid numeric column yields `Some` with length
    /// `product(shape) * size_of::<element>()`. `Bool` is emitted as one byte
    /// per element (`0`/`1`).
    pub fn raw_bytes(&self) -> Option<Vec<u8>> {
        fn le_numeric<T: Clone, const N: usize>(
            h: &ColumnHolder<T>,
            to_bytes: impl Fn(&T) -> [u8; N],
        ) -> Vec<u8> {
            let a = h.array().as_standard_layout();
            let mut out = Vec::with_capacity(a.len() * N);
            for v in a.iter() {
                out.extend_from_slice(&to_bytes(v));
            }
            out
        }
        match self {
            Column::Float16(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::Float32(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::Float(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::Int8(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::Int16(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::Int(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::Int64(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::UInt(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::U8(h) => Some(h.array().as_standard_layout().iter().copied().collect()),
            Column::UInt16(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::UInt32(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::Bool(h) => Some(
                h.array()
                    .as_standard_layout()
                    .iter()
                    .map(|&b| b as u8)
                    .collect(),
            ),
            Column::String(_) => None,
            Column::Complex64(h) => Some(le_numeric(h, |v| {
                let mut bytes = [0u8; 8];
                bytes[..4].copy_from_slice(&v.re.to_le_bytes());
                bytes[4..].copy_from_slice(&v.im.to_le_bytes());
                bytes
            })),
            Column::Complex128(h) => Some(le_numeric(h, |v| {
                let mut bytes = [0u8; 16];
                bytes[..8].copy_from_slice(&v.re.to_le_bytes());
                bytes[8..].copy_from_slice(&v.im.to_le_bytes());
                bytes
            })),
        }
    }

    /// Gather rows at `indices` (along axis 0) into a new owned Column of the
    /// same dtype. Backs [`Block::select_rows`](crate::store::block::Block::select_rows)
    /// and the sort path. String rows are cloned.
    pub fn select_rows(&self, indices: &[usize]) -> Column {
        use ndarray::Axis;
        match self {
            Column::Float16(h) => Column::from_f16(h.array().select(Axis(0), indices)),
            Column::Float32(h) => Column::from_f32(h.array().select(Axis(0), indices)),
            Column::Float(h) => Column::from_float(h.array().select(Axis(0), indices)),
            Column::Int8(h) => Column::from_i8(h.array().select(Axis(0), indices)),
            Column::Int16(h) => Column::from_i16(h.array().select(Axis(0), indices)),
            Column::Int(h) => Column::from_int(h.array().select(Axis(0), indices)),
            Column::Int64(h) => Column::from_i64(h.array().select(Axis(0), indices)),
            Column::Bool(h) => Column::from_bool(h.array().select(Axis(0), indices)),
            Column::UInt(h) => Column::from_uint(h.array().select(Axis(0), indices)),
            Column::U8(h) => Column::from_u8(h.array().select(Axis(0), indices)),
            Column::UInt16(h) => Column::from_u16(h.array().select(Axis(0), indices)),
            Column::UInt32(h) => Column::from_u32(h.array().select(Axis(0), indices)),
            Column::String(h) => Column::from_string(h.array().select(Axis(0), indices)),
            Column::Complex64(h) => Column::from_c64(h.array().select(Axis(0), indices)),
            Column::Complex128(h) => Column::from_c128(h.array().select(Axis(0), indices)),
        }
    }

    /// Is this column backed by a foreign (non-Rust) buffer?
    pub fn is_foreign(&self) -> bool {
        map_column!(self, a => a.is_foreign())
    }

    /// Format one row as EXTXYZ property tokens.
    pub fn xyz_tokens(&self, row: usize) -> Vec<String> {
        use ndarray::Axis;
        match self {
            Column::Bool(h) => h
                .array()
                .index_axis(Axis(0), row)
                .iter()
                .map(|v| if *v { "T" } else { "F" }.to_string())
                .collect(),
            Column::String(h) => h.array().index_axis(Axis(0), row).iter().cloned().collect(),
            _ => map_column!(self, h => {
                h.array()
                    .index_axis(Axis(0), row)
                    .iter()
                    .map(|v| v.to_string())
                    .collect()
            }),
        }
    }

    /// Returns a reference to the float data, or `None` if this column is not `Float`.
    pub fn as_float(&self) -> Option<&ArrayD<F>> {
        match self {
            Column::Float(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the float data, or `None` if not `Float`.
    ///
    /// Copy-on-write: clones if shared or foreign-backed. The returned mut ref
    /// always refers to Rust-owned memory.
    pub fn as_float_mut(&mut self) -> Option<&mut ArrayD<F>> {
        match self {
            Column::Float(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the integer data, or `None` if not `Int`.
    pub fn as_int(&self) -> Option<&ArrayD<I>> {
        match self {
            Column::Int(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the integer data, or `None` if not `Int`.
    ///
    /// Copy-on-write: clones if shared or foreign-backed.
    pub fn as_int_mut(&mut self) -> Option<&mut ArrayD<I>> {
        match self {
            Column::Int(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the boolean data, or `None` if not `Bool`.
    pub fn as_bool(&self) -> Option<&ArrayD<bool>> {
        match self {
            Column::Bool(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the boolean data, or `None` if not `Bool`.
    ///
    /// Copy-on-write: clones if shared or foreign-backed.
    pub fn as_bool_mut(&mut self) -> Option<&mut ArrayD<bool>> {
        match self {
            Column::Bool(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the unsigned integer data, or `None` if not `UInt`.
    pub fn as_uint(&self) -> Option<&ArrayD<Idx>> {
        match self {
            Column::UInt(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the unsigned integer data, or `None` if not `UInt`.
    ///
    /// Copy-on-write: clones if shared or foreign-backed.
    pub fn as_uint_mut(&mut self) -> Option<&mut ArrayD<Idx>> {
        match self {
            Column::UInt(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the u8 data, or `None` if not `U8`.
    pub fn as_u8(&self) -> Option<&ArrayD<u8>> {
        match self {
            Column::U8(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the u8 data, or `None` if not `U8`.
    ///
    /// Copy-on-write: clones if shared or foreign-backed.
    pub fn as_u8_mut(&mut self) -> Option<&mut ArrayD<u8>> {
        match self {
            Column::U8(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the string data, or `None` if not `String`.
    pub fn as_string(&self) -> Option<&ArrayD<String>> {
        match self {
            Column::String(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the string data, or `None` if not `String`.
    ///
    /// Copy-on-write: clones if shared or foreign-backed.
    pub fn as_string_mut(&mut self) -> Option<&mut ArrayD<String>> {
        match self {
            Column::String(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the `f16` data, or `None` if not `Float16`.
    pub fn as_f16(&self) -> Option<&ArrayD<half::f16>> {
        match self {
            Column::Float16(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the `f16` data, or `None` if not `Float16`.
    pub fn as_f16_mut(&mut self) -> Option<&mut ArrayD<half::f16>> {
        match self {
            Column::Float16(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the `f32` data, or `None` if not `Float32`.
    pub fn as_f32(&self) -> Option<&ArrayD<f32>> {
        match self {
            Column::Float32(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the `f32` data, or `None` if not `Float32`.
    pub fn as_f32_mut(&mut self) -> Option<&mut ArrayD<f32>> {
        match self {
            Column::Float32(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the `i8` data, or `None` if not `Int8`.
    pub fn as_i8(&self) -> Option<&ArrayD<i8>> {
        match self {
            Column::Int8(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the `i8` data, or `None` if not `Int8`.
    pub fn as_i8_mut(&mut self) -> Option<&mut ArrayD<i8>> {
        match self {
            Column::Int8(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the `i16` data, or `None` if not `Int16`.
    pub fn as_i16(&self) -> Option<&ArrayD<i16>> {
        match self {
            Column::Int16(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the `i16` data, or `None` if not `Int16`.
    pub fn as_i16_mut(&mut self) -> Option<&mut ArrayD<i16>> {
        match self {
            Column::Int16(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the `i64` data, or `None` if not `Int64`.
    pub fn as_i64(&self) -> Option<&ArrayD<i64>> {
        match self {
            Column::Int64(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the `i64` data, or `None` if not `Int64`.
    pub fn as_i64_mut(&mut self) -> Option<&mut ArrayD<i64>> {
        match self {
            Column::Int64(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the `u16` data, or `None` if not `UInt16`.
    pub fn as_u16(&self) -> Option<&ArrayD<u16>> {
        match self {
            Column::UInt16(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the `u16` data, or `None` if not `UInt16`.
    pub fn as_u16_mut(&mut self) -> Option<&mut ArrayD<u16>> {
        match self {
            Column::UInt16(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the `u32` data, or `None` if not `UInt32`.
    pub fn as_u32(&self) -> Option<&ArrayD<u32>> {
        match self {
            Column::UInt32(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the `u32` data, or `None` if not `UInt32`.
    pub fn as_u32_mut(&mut self) -> Option<&mut ArrayD<u32>> {
        match self {
            Column::UInt32(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the `complex64` data, or `None` if not `Complex64`.
    pub fn as_c64(&self) -> Option<&ArrayD<Complex<f32>>> {
        match self {
            Column::Complex64(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the `complex64` data, or `None` if not `Complex64`.
    pub fn as_c64_mut(&mut self) -> Option<&mut ArrayD<Complex<f32>>> {
        match self {
            Column::Complex64(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a reference to the `complex128` data, or `None` if not `Complex128`.
    pub fn as_c128(&self) -> Option<&ArrayD<Complex<f64>>> {
        match self {
            Column::Complex128(a) => Some(a.array()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the `complex128` data, or `None` if not `Complex128`.
    pub fn as_c128_mut(&mut self) -> Option<&mut ArrayD<Complex<f64>>> {
        match self {
            Column::Complex128(a) => Some(realize_owned_mut(a)),
            _ => None,
        }
    }

    /// Returns a clone of the inner float holder Arc, or `None` if not `Float`.
    /// O(1) refcount bump; shares storage.
    pub fn float_arc(&self) -> Option<Arc<ColumnHolder<F>>> {
        match self {
            Column::Float(a) => Some(Arc::clone(a)),
            _ => None,
        }
    }

    /// See [`float_arc`](Self::float_arc).
    pub fn int_arc(&self) -> Option<Arc<ColumnHolder<I>>> {
        match self {
            Column::Int(a) => Some(Arc::clone(a)),
            _ => None,
        }
    }

    /// See [`float_arc`](Self::float_arc).
    pub fn bool_arc(&self) -> Option<Arc<ColumnHolder<bool>>> {
        match self {
            Column::Bool(a) => Some(Arc::clone(a)),
            _ => None,
        }
    }

    /// See [`float_arc`](Self::float_arc).
    pub fn uint_arc(&self) -> Option<Arc<ColumnHolder<Idx>>> {
        match self {
            Column::UInt(a) => Some(Arc::clone(a)),
            _ => None,
        }
    }

    /// See [`float_arc`](Self::float_arc).
    pub fn u8_arc(&self) -> Option<Arc<ColumnHolder<u8>>> {
        match self {
            Column::U8(a) => Some(Arc::clone(a)),
            _ => None,
        }
    }

    /// See [`float_arc`](Self::float_arc).
    pub fn string_arc(&self) -> Option<Arc<ColumnHolder<String>>> {
        match self {
            Column::String(a) => Some(Arc::clone(a)),
            _ => None,
        }
    }

    /// Resize this column along axis 0 to `new_nrows`.
    ///
    /// See the main doc on `Block::resize`. If the underlying holder is
    /// shared or foreign-backed, this replaces the holder with a fresh
    /// Rust-owned copy.
    pub fn resize(&mut self, new_nrows: usize) {
        let current = self.shape()[0];
        if new_nrows == current {
            return;
        }
        match self {
            Column::Float16(a) => resize_holder(a, current, new_nrows),
            Column::Float32(a) => resize_holder(a, current, new_nrows),
            Column::Float(a) => resize_holder(a, current, new_nrows),
            Column::Int8(a) => resize_holder(a, current, new_nrows),
            Column::Int16(a) => resize_holder(a, current, new_nrows),
            Column::Int(a) => resize_holder(a, current, new_nrows),
            Column::Int64(a) => resize_holder(a, current, new_nrows),
            Column::UInt(a) => resize_holder(a, current, new_nrows),
            Column::U8(a) => resize_holder(a, current, new_nrows),
            Column::UInt16(a) => resize_holder(a, current, new_nrows),
            Column::UInt32(a) => resize_holder(a, current, new_nrows),
            Column::Bool(a) => resize_holder(a, current, new_nrows),
            Column::String(a) => resize_holder(a, current, new_nrows),
            Column::Complex64(a) => resize_holder(a, current, new_nrows),
            Column::Complex128(a) => resize_holder(a, current, new_nrows),
        }
    }
}

fn resize_holder<T: Clone + Default>(
    a: &mut Arc<ColumnHolder<T>>,
    current: usize,
    new_nrows: usize,
) {
    use ndarray::{Axis, IxDyn, concatenate};
    let view = a.view();
    let new_arr = if new_nrows < current {
        view.slice_axis(Axis(0), (..new_nrows).into()).to_owned()
    } else {
        let mut pad_shape = a.shape().to_vec();
        pad_shape[0] = new_nrows - current;
        let pad = ArrayD::<T>::default(IxDyn(&pad_shape));
        concatenate(Axis(0), &[view, pad.view()]).unwrap()
    };
    *a = Arc::new(ColumnHolder::from_owned(new_arr));
}

impl std::fmt::Debug for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Column::{:?}(shape={:?})", self.dtype(), self.shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{F, I, Idx};
    use ndarray::{Array1, ArrayD};

    // ---- helpers ----

    fn float_col(n: usize) -> Column {
        Column::from_float(Array1::from_vec(vec![0.0 as F; n]).into_dyn())
    }

    fn int_col(n: usize) -> Column {
        Column::from_int(Array1::from_vec(vec![0 as I; n]).into_dyn())
    }

    fn bool_col(n: usize) -> Column {
        Column::from_bool(Array1::from_vec(vec![false; n]).into_dyn())
    }

    fn uint_col(n: usize) -> Column {
        Column::from_uint(Array1::from_vec(vec![0 as Idx; n]).into_dyn())
    }

    fn u8_col(n: usize) -> Column {
        Column::from_u8(Array1::from_vec(vec![0u8; n]).into_dyn())
    }

    fn string_col(n: usize) -> Column {
        Column::from_string(Array1::from_vec(vec![String::new(); n]).into_dyn())
    }

    // ---- nrows / dtype / shape ----

    #[test]
    fn test_nrows() {
        assert_eq!(float_col(5).nrows(), Some(5));
        assert_eq!(int_col(3).nrows(), Some(3));
        assert_eq!(bool_col(7).nrows(), Some(7));
        assert_eq!(uint_col(2).nrows(), Some(2));
        assert_eq!(u8_col(4).nrows(), Some(4));
        assert_eq!(string_col(1).nrows(), Some(1));

        let rank0 = Column::from_float(ArrayD::<F>::from_elem(vec![], 1.0));
        assert_eq!(rank0.nrows(), None);
    }

    #[test]
    fn test_dtype() {
        assert_eq!(float_col(1).dtype(), DType::Float);
        assert_eq!(int_col(1).dtype(), DType::Int);
        assert_eq!(bool_col(1).dtype(), DType::Bool);
        assert_eq!(uint_col(1).dtype(), DType::UInt);
        assert_eq!(u8_col(1).dtype(), DType::U8);
        assert_eq!(string_col(1).dtype(), DType::String);
    }

    #[test]
    fn test_shape() {
        assert_eq!(float_col(4).shape(), &[4]);
        let col2d = Column::from_int(ArrayD::<I>::from_elem(vec![3, 2], 0));
        assert_eq!(col2d.shape(), &[3, 2]);
    }

    // ---- typed accessors ----

    #[test]
    fn test_as_float_on_float() {
        let col = float_col(3);
        assert!(col.as_float().is_some());
        assert_eq!(col.as_float().unwrap().len(), 3);
    }

    #[test]
    fn test_as_float_on_wrong_type() {
        assert!(int_col(2).as_float().is_none());
        assert!(bool_col(2).as_float().is_none());
        assert!(uint_col(2).as_float().is_none());
        assert!(u8_col(2).as_float().is_none());
        assert!(string_col(2).as_float().is_none());
    }

    #[test]
    fn test_as_int() {
        let col = int_col(4);
        assert!(col.as_int().is_some());
        assert_eq!(col.as_int().unwrap().len(), 4);
        assert!(float_col(1).as_int().is_none());
        assert!(bool_col(1).as_int().is_none());
    }

    #[test]
    fn test_as_bool() {
        let col = bool_col(2);
        assert!(col.as_bool().is_some());
        assert_eq!(col.as_bool().unwrap().len(), 2);
        assert!(float_col(1).as_bool().is_none());
        assert!(int_col(1).as_bool().is_none());
    }

    #[test]
    fn test_as_uint() {
        let col = uint_col(6);
        assert!(col.as_uint().is_some());
        assert_eq!(col.as_uint().unwrap().len(), 6);
        assert!(float_col(1).as_uint().is_none());
        assert!(int_col(1).as_uint().is_none());
    }

    #[test]
    fn test_as_u8() {
        let col = u8_col(3);
        assert!(col.as_u8().is_some());
        assert_eq!(col.as_u8().unwrap().len(), 3);
        assert!(float_col(1).as_u8().is_none());
        assert!(uint_col(1).as_u8().is_none());
    }

    #[test]
    fn test_as_string() {
        let col = string_col(2);
        assert!(col.as_string().is_some());
        assert_eq!(col.as_string().unwrap().len(), 2);
        assert!(float_col(1).as_string().is_none());
        assert!(int_col(1).as_string().is_none());
    }

    #[test]
    fn test_as_float_mut() {
        let mut col =
            Column::from_float(Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn());
        {
            let arr = col.as_float_mut().unwrap();
            arr[0] = 99.0;
        }
        let arr = col.as_float().unwrap();
        assert!((arr[0] - 99.0).abs() < F::EPSILON);
        let mut int = int_col(1);
        assert!(int.as_float_mut().is_none());
    }

    #[test]
    fn test_debug_format() {
        let dbg = format!("{:?}", float_col(3));
        assert!(dbg.contains("Column::Float"));
        assert!(dbg.contains("shape="));
    }

    // ---- Arc semantics ----

    #[test]
    fn test_clone_shares_buffer() {
        let col = Column::from_float(Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn());
        let cloned = col.clone();
        let p1 = col.as_float().unwrap().as_ptr();
        let p2 = cloned.as_float().unwrap().as_ptr();
        assert_eq!(p1, p2, "clone must share the buffer");
    }

    #[test]
    fn test_as_float_mut_cow_when_shared() {
        let col = Column::from_float(Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn());
        let mut cloned = col.clone();
        cloned.as_float_mut().unwrap()[0] = 99.0;
        assert_eq!(col.as_float().unwrap()[0], 1.0);
        assert_eq!(cloned.as_float().unwrap()[0], 99.0);
    }

    // ---- Foreign holder correctness ----

    /// Test that a foreign-backed holder reads the right data without
    /// dropping the foreign memory (we use `Vec<f64>` itself as the foreign
    /// "allocator" — its Drop is what will free memory when the keeper drops).
    #[test]
    fn test_foreign_holder_readable() {
        // "Foreign" keeper = a Rust Vec, but we exercise the ManuallyDrop +
        // Drop path (keeper drops at end, Vec::drop runs via the keeper's own
        // Drop, not via ManuallyDrop::drop).
        let source: Vec<F> = vec![1.0, 2.0, 3.0, 4.0];
        let ptr = source.as_ptr() as *mut F;
        let len = source.len();
        // Forge ArrayD pointing at `source`'s memory. SAFETY: `source` outlives
        // the holder (kept alive as keeper). We construct cap=len so ndarray
        // won't try to grow/realloc.
        let forged = unsafe {
            let vec = Vec::from_raw_parts(ptr, len, len);
            ArrayD::from_shape_vec(ndarray::IxDyn(&[len]), vec).unwrap()
        };
        let holder = unsafe { ColumnHolder::from_foreign(forged, source) };
        let col = Column::from_float_holder(holder);
        assert!(col.is_foreign());
        let arr = col.as_float().unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_foreign_holder_cow_on_mut() {
        // Set up a foreign holder, then mutate. The mutation should detach
        // (CoW) and the source Vec should remain intact.
        let source: Vec<F> = vec![10.0, 20.0, 30.0];
        let ptr = source.as_ptr() as *mut F;
        let len = source.len();
        let forged = unsafe {
            let vec = Vec::from_raw_parts(ptr, len, len);
            ArrayD::from_shape_vec(ndarray::IxDyn(&[len]), vec).unwrap()
        };
        // Keep the source alive for the assertion below: store a clone in the
        // keeper, so the original Vec isn't consumed when we check it.
        let source_clone = source.clone();
        let holder = unsafe { ColumnHolder::from_foreign(forged, source_clone) };
        let mut col = Column::from_float_holder(holder);
        assert!(col.is_foreign());

        // Mutate the column. This should CoW into a Rust-owned holder.
        col.as_float_mut().unwrap()[0] = 999.0;
        assert!(!col.is_foreign(), "after mut, holder must be Rust-owned");
        assert_eq!(col.as_float().unwrap()[0], 999.0);

        // Source Vec is untouched.
        assert_eq!(source[0], 10.0);
    }

    #[test]
    fn test_foreign_holder_clone_detaches() {
        let source: Vec<F> = vec![7.0, 8.0];
        let ptr = source.as_ptr() as *mut F;
        let len = source.len();
        let forged = unsafe {
            let vec = Vec::from_raw_parts(ptr, len, len);
            ArrayD::from_shape_vec(ndarray::IxDyn(&[len]), vec).unwrap()
        };
        let source_clone = source.clone();
        let holder = unsafe { ColumnHolder::from_foreign(forged, source_clone) };
        // Clone produces a Rust-owned holder.
        let holder_clone = holder.clone();
        assert!(holder.is_foreign());
        assert!(!holder_clone.is_foreign());
        assert_eq!(holder_clone.array().as_slice().unwrap(), &[7.0, 8.0]);
    }

    // ---- raw_bytes ----

    #[test]
    fn test_raw_bytes_lengths() {
        assert_eq!(float_col(3).raw_bytes().map(|b| b.len()), Some(24));
        assert_eq!(int_col(4).raw_bytes().map(|b| b.len()), Some(16));
        // Identity columns are 8 bytes wide: an identifier that wraps is not
        // an identifier, so `Idx` is 64-bit.
        assert_eq!(uint_col(2).raw_bytes().map(|b| b.len()), Some(16));
        assert_eq!(u8_col(5).raw_bytes().map(|b| b.len()), Some(5));
        assert_eq!(bool_col(7).raw_bytes().map(|b| b.len()), Some(7));
        assert!(string_col(3).raw_bytes().is_none());
    }

    #[test]
    fn test_raw_bytes_little_endian_values() {
        let col = Column::from_float(Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn());
        let b = col.raw_bytes().unwrap();
        assert_eq!(&b[0..8], &1.0f64.to_le_bytes());
        assert_eq!(&b[8..16], &2.0f64.to_le_bytes());
    }

    #[test]
    fn test_raw_bytes_multidim_length() {
        // A [2,3] int column flattens to 6 elements * 4 bytes.
        let col = Column::from_int(
            ArrayD::<I>::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![0 as I; 6]).unwrap(),
        );
        assert_eq!(col.raw_bytes().map(|b| b.len()), Some(24));
    }

    #[test]
    fn test_raw_bytes_strided_materializes() {
        // A strided (sliced) numeric column still yields Some, in logical order.
        let base = Array1::from_vec(vec![1.0 as F, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let strided = base.slice_move(ndarray::s![..;2]).into_dyn(); // [1, 3, 5]
        let col = Column::from_float(strided);
        let b = col.raw_bytes().unwrap();
        assert_eq!(b.len(), 24);
        assert_eq!(&b[0..8], &1.0f64.to_le_bytes());
        assert_eq!(&b[8..16], &3.0f64.to_le_bytes());
        assert_eq!(&b[16..24], &5.0f64.to_le_bytes());
    }
}
