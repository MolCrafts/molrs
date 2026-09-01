//! WASM bindings for frame-sequence Zarr v3 archives.

use crate::core::frame::Frame;
use molrs::io::mrec::FrameSequence;
use molrs::io::reader::TrajectoryReader;
use std::cell::{RefCell, RefMut};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::storage::WritableStorageTraits;

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
#[wasm_bindgen(js_name = TrajectoryReader)]
pub struct RecordReader {
    sequence: RefCell<FrameSequence>,
}

impl RecordReader {
    /// The cursor, or the re-entrancy error.
    ///
    /// `try_borrow_mut`, never `borrow_mut`: the panicking form would abort the
    /// wasm instance on a caller mistake that an exception describes.
    fn sequence(&self) -> Result<RefMut<'_, FrameSequence>, JsValue> {
        self.sequence
            .try_borrow_mut()
            .map_err(|_| JsError::new("TrajectoryReader is busy: re-entrant call").into())
    }
}

#[wasm_bindgen(js_class = TrajectoryReader)]
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
        // Index-only: the schema plus each section's step_index and offset. No
        // frame is decoded here — `countAtomsAtFirstFrame` decodes lazily, so a
        // ragged store's large frame 0 is not paid for just to open the reader.
        let sequence = FrameSequence::open(store).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(RecordReader {
            sequence: RefCell::new(sequence),
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

    /// Atom count of frame 0, decoded on demand.
    ///
    /// Named for what it is: a ragged store grows its atom count per frame, so
    /// this is the *first* frame's count, not the maximum or the current one. A
    /// consumer sizing a buffer for the whole run must not trust it.
    #[wasm_bindgen(js_name = countAtomsAtFirstFrame)]
    pub fn count_atoms_at_first_frame(&self) -> Result<usize, JsValue> {
        Ok(self
            .sequence()?
            .frame(0)
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .and_then(|frame| frame.get("atoms").and_then(|block| block.nrows()))
            .unwrap_or(0))
    }

    /// Step numbers of the committed frames — frame labels for a replay UI.
    #[wasm_bindgen(js_name = steps)]
    pub fn steps(&self) -> Result<Vec<i64>, JsValue> {
        Ok(self.sequence()?.steps().to_vec())
    }

    /// Physical times (fs) of the committed frames, when the run wrote any.
    #[wasm_bindgen(js_name = times)]
    pub fn times(&self) -> Result<Option<Vec<f64>>, JsValue> {
        Ok(self.sequence()?.times().map(<[f64]>::to_vec))
    }

    /// Whether the store carries a block section of this name.
    ///
    /// Lets the stage decide once — e.g. `hasBlock("bonds")` — whether a
    /// per-frame section exists, instead of attaching a recompute modifier that
    /// forces a full rebuild on every frame.
    #[wasm_bindgen(js_name = hasBlock)]
    pub fn has_block(&self, name: &str) -> Result<bool, JsValue> {
        Ok(self.sequence()?.has_block(name))
    }

    /// Names of every block section present in the store.
    #[wasm_bindgen(js_name = blockNames)]
    pub fn block_names(&self) -> Result<Vec<String>, JsValue> {
        Ok(self.sequence()?.block_names().map(str::to_string).collect())
    }
}

#[cfg(test)]
mod export_pin;
