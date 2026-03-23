//! WASM bindings for Zarr V3 simulation store.
//!
//! Provides [`SimulationReader`] for reading trajectory data from
//! an in-memory Zarr V3 archive. Since WASM has no filesystem access,
//! the archive is supplied as a `Map<string, Uint8Array>` of
//! path-to-content pairs.
//!
//! # Zarr archive layout (expected)
//!
//! ```text
//! /zarr.json               -- root group metadata
//! /system/                  -- system topology (atoms, bonds, box)
//! /trajectory/              -- per-frame arrays (positions, velocities, ...)
//! /forcefield/              -- force field parameters (optional)
//! ```
//!
//! # Example (JavaScript)
//!
//! ```js
//! // files: Map<string, Uint8Array> loaded from a .zarr archive
//! const reader = new SimulationReader(files);
//! console.log(reader.countFrames()); // e.g., 1000
//! console.log(reader.countAtoms());  // e.g., 256
//!
//! const frame = reader.readFrame(42);
//! const atoms = frame.getBlock("atoms");
//! ```

use crate::core::frame::Frame;
use molrs::SimulationStore;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::storage::WritableStorageTraits;
use zarrs::storage::store::MemoryStore;

/// Reader for Zarr V3 simulation archives.
///
/// Wraps [`SimulationStore`] from `molrs-core` for reading trajectory
/// frames and system metadata from an in-memory Zarr store. The store
/// is populated from a `Map<string, Uint8Array>` of file paths to
/// binary content.
///
/// # Example (JavaScript)
///
/// ```js
/// const files = new Map();
/// files.set("zarr.json", new Uint8Array([...]));
/// files.set("system/.zarray", new Uint8Array([...]));
/// // ... etc.
///
/// const reader = new SimulationReader(files);
/// const frame = reader.readFrame(0);
/// ```
#[wasm_bindgen(js_name = SimulationReader)]
pub struct SimulationReader {
    inner: SimulationStore,
}

#[wasm_bindgen(js_class = SimulationReader)]
impl SimulationReader {
    /// Create a reader from a map of file paths to binary content.
    ///
    /// The map keys are relative paths within the Zarr archive
    /// (e.g., `"zarr.json"`, `"system/.zarray"`). Values are the
    /// raw bytes of each file as `Uint8Array`.
    ///
    /// # Arguments
    ///
    /// * `files` - `Map<string, Uint8Array>` mapping archive paths
    ///   to their binary content
    ///
    /// # Returns
    ///
    /// A new `SimulationReader` ready to read frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the archive cannot be opened
    /// (e.g., missing required metadata files).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new SimulationReader(filesMap);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(files: js_sys::Map) -> Result<SimulationReader, JsValue> {
        let store = Arc::new(MemoryStore::new());

        for key_res in files.keys() {
            let key = key_res.map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
            let path = key
                .as_string()
                .ok_or_else(|| JsValue::from_str("Invalid path key"))?;
            let content_value = files.get(&key);
            let content = js_sys::Uint8Array::new(&content_value).to_vec();

            let store_path = path.strip_prefix('/').unwrap_or(&path);
            let skey = zarrs::storage::StoreKey::new(store_path)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            store
                .set(&skey, content.into())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        let inner = SimulationStore::open_store(store as ReadableWritableListableStorage)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(SimulationReader { inner })
    }

    /// Read a trajectory frame at the given time index.
    ///
    /// The returned [`Frame`] merges the static system topology
    /// (atoms, bonds) with the per-frame trajectory data (positions,
    /// velocities, etc.) at time step `t`.
    ///
    /// # Arguments
    ///
    /// * `t` - Zero-based time step index
    ///
    /// # Returns
    ///
    /// A [`Frame`] with merged system + trajectory data, or `undefined`
    /// if no trajectory is present.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on I/O or deserialization errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = reader.readFrame(0);
    /// if (frame) {
    ///   const x = frame.getBlock("atoms").copyColF32("x");
    /// }
    /// ```
    #[wasm_bindgen(js_name = readFrame)]
    pub fn read_frame(&self, t: usize) -> Result<Option<Frame>, JsValue> {
        let traj = self
            .inner
            .open_trajectory()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let Some(reader) = traj else {
            return Ok(None);
        };

        let system = self
            .inner
            .read_system()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let rs_frame = reader
            .read_frame(t as u64, &system)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Some(Frame::from_rs_frame(rs_frame)?))
    }

    /// Return the number of trajectory frames in the archive.
    ///
    /// Returns `0` if no trajectory data is present.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on I/O errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(reader.countFrames()); // e.g., 1000
    /// ```
    #[wasm_bindgen(js_name = countFrames)]
    pub fn count_frames(&self) -> Result<usize, JsValue> {
        let traj = self
            .inner
            .open_trajectory()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        match traj {
            Some(reader) => Ok(reader.count_frames() as usize),
            None => Ok(0),
        }
    }

    /// Return the number of atoms in the system topology.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(reader.countAtoms()); // e.g., 256
    /// ```
    #[wasm_bindgen(js_name = countAtoms)]
    pub fn count_atoms(&self) -> usize {
        self.inner.count_atoms() as usize
    }
}
