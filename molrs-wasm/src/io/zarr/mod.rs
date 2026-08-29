//! WASM bindings for frame-sequence Zarr v3 archives.

use crate::core::frame::Frame;
use molrs::io::reader::TrajectoryReader;
use molrs::io::zarr::FrameSequence;
use std::cell::{RefCell, RefMut};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::storage::WritableStorageTraits;
use zarrs::storage::store::MemoryStore;

/// Reader for frame-sequence Zarr v3 archives.
///
/// The sequence is opened **once**, in the constructor. `FrameSequence` is the
/// lazy store cursor, so `readFrame` decodes exactly the frame it was asked
/// for instead of the whole record, and `countFrames` answers off the index
/// the open already cached.
///
/// Reading advances that cursor, so it lives behind a `RefCell` and every JS
/// method keeps its `&self` signature. A re-entrant call from JS is a borrow
/// failure, and a borrow failure is an exception — a wasm export on a fallible
/// path never panics.
#[wasm_bindgen(js_name = RecordReader)]
pub struct RecordReader {
    sequence: RefCell<FrameSequence>,
    n_atoms: usize,
}

impl RecordReader {
    /// The cursor, or the re-entrancy error.
    ///
    /// `try_borrow_mut`, never `borrow_mut`: the panicking form would abort the
    /// wasm instance on a caller mistake that an exception describes.
    fn sequence(&self) -> Result<RefMut<'_, FrameSequence>, JsValue> {
        self.sequence
            .try_borrow_mut()
            .map_err(|_| JsError::new("RecordReader is busy: re-entrant call").into())
    }
}

#[wasm_bindgen(js_class = RecordReader)]
impl RecordReader {
    #[wasm_bindgen(constructor)]
    pub fn new(files: js_sys::Map) -> Result<RecordReader, JsValue> {
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

        // The reader is a read door, so it gets the store's read-only view.
        let store = (store as ReadableWritableListableStorage).readable_listable();
        // Index-only: the schema plus each section's step_index and offset.
        let mut sequence =
            FrameSequence::open(store).map_err(|e| JsValue::from_str(&e.to_string()))?;
        // Cached, because it cannot change: the archive is a fixed snapshot,
        // so answering `countAtoms` per call would decode a frame to learn
        // something already known.
        let n_atoms = sequence
            .frame(0)
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .and_then(|frame| frame.get("atoms").and_then(|block| block.nrows()))
            .unwrap_or(0);
        Ok(RecordReader {
            sequence: RefCell::new(sequence),
            n_atoms,
        })
    }

    #[wasm_bindgen(js_name = readFrame)]
    pub fn read_frame(&self, t: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .sequence()?
            .frame(t as u64)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        match rs_frame {
            Some(frame) => Ok(Some(Frame::from_rs(frame)?)),
            None => Ok(None),
        }
    }

    #[wasm_bindgen(js_name = countFrames)]
    pub fn count_frames(&self) -> Result<usize, JsValue> {
        self.sequence()?
            .len()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = countAtoms)]
    pub fn count_atoms(&self) -> usize {
        self.n_atoms
    }

    #[wasm_bindgen(js_name = free)]
    pub fn free(&self) {}
}
