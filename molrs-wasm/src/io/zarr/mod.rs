//! WASM bindings for MolRec Zarr v3 archives.

use crate::core::frame::Frame;
use molrs::MolRec;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::storage::WritableStorageTraits;
use zarrs::storage::store::MemoryStore;

/// Reader for MolRec Zarr v3 archives.
#[wasm_bindgen(js_name = MolRecReader)]
pub struct MolRecReader {
    store: ReadableWritableListableStorage,
}

#[wasm_bindgen(js_class = MolRecReader)]
impl MolRecReader {
    #[wasm_bindgen(constructor)]
    pub fn new(files: js_sys::Map) -> Result<MolRecReader, JsValue> {
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

        let store = store as ReadableWritableListableStorage;
        MolRec::read_zarr_store(store.clone()).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(MolRecReader { store })
    }

    #[wasm_bindgen(js_name = readFrame)]
    pub fn read_frame(&self, t: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = molrs::read_molrec_frame_from_store(self.store.clone(), t)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        match rs_frame {
            Some(frame) => Ok(Some(Frame::from_rs_frame(frame)?)),
            None => Ok(None),
        }
    }

    #[wasm_bindgen(js_name = countFrames)]
    pub fn count_frames(&self) -> Result<usize, JsValue> {
        Ok(molrs::count_molrec_frames_in_store(self.store.clone())
            .map_err(|e| JsValue::from_str(&e.to_string()))? as usize)
    }

    #[wasm_bindgen(js_name = countAtoms)]
    pub fn count_atoms(&self) -> Result<usize, JsValue> {
        let rec = MolRec::read_zarr_store(self.store.clone())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rec
            .frame
            .get("atoms")
            .and_then(|block| block.nrows())
            .unwrap_or(0))
    }

    #[wasm_bindgen(js_name = free)]
    pub fn free(&self) {}
}
