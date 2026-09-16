//! WASM bindings for frame-sequence Zarr v3 stores (`*.mrec`).
//!
//! Three ways to hand the reader its bytes, in increasing laziness:
//!
//! - **`new TrajectoryReader(files)`** — a `Map<path, Uint8Array>` of every
//!   file, copied once into an in-memory store. Simple; the whole store is
//!   resident.
//! - **`TrajectoryReader.fromZip(bytes)`** — one packed `*.mrec.zip`, whose
//!   stored entries are unpacked into the same in-memory store.
//! - **`TrajectoryReader.fromStore(host)`** — a host object that serves keys
//!   on demand: `get(key)`, `getRange(key, offset, length)`, `size(key)` and
//!   `list(prefix)`. Only the chunks a frame touches ever cross into wasm, so
//!   a multi-gigabyte run opens in a worker that reads its files (or an HTTP
//!   range server) synchronously.

use crate::core::frame::Frame;
use crate::core::region::simbox::Box as JsBox;
use molrs::io::mrec::{FrameSequence, read_frame_section_store, section_names_store};
use molrs::io::reader::TrajectoryReader;
use std::io::Read;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, ReadableStorageTraits,
    ReadableWritableListableStorage, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes,
    StorePrefix, WritableStorageTraits,
};

fn js_string_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Reader for frame-sequence Zarr v3 stores.
///
/// The sequence is opened **once**, in the constructor. `FrameSequence` is the
/// lazy store cursor, so `readFrame` decodes exactly the frame it was asked
/// for (keeping the last decoded chunk of every column, so consecutive frames
/// are slices), and `countFrames` answers off the index the open already
/// cached. Every method takes `&self`: the cursor holds caches, not a
/// position.
#[wasm_bindgen(js_name = TrajectoryReader)]
pub struct RecordReader {
    sequence: FrameSequence,
}

impl RecordReader {
    fn open<S>(store: Arc<S>) -> Result<RecordReader, JsValue>
    where
        S: ?Sized
            + zarrs::storage::ReadableStorageTraits
            + zarrs::storage::ListableStorageTraits
            + 'static,
    {
        // Index-only: the schema plus each section's step_index and offset (or
        // their hints). No frame is decoded here.
        let sequence = FrameSequence::open(store).map_err(js_string_err)?;
        Ok(RecordReader { sequence })
    }
}

#[wasm_bindgen(js_class = TrajectoryReader)]
impl RecordReader {
    /// Open a store handed over as a `Map<path, Uint8Array>` of every file.
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
            let skey = StoreKey::new(store_path).map_err(js_string_err)?;
            store.set(&skey, content.into()).map_err(js_string_err)?;
        }
        // The reader is a read door, so it gets the store's read-only view.
        let store = (store as ReadableWritableListableStorage).readable_listable();
        Self::open(store)
    }

    /// Open a packed `*.mrec.zip` from its bytes.
    ///
    /// Every entry of a packed store is *stored* (never deflated), so this is
    /// a central-directory walk plus one copy per entry into an in-memory
    /// store. A zip with compressed entries is refused by name.
    #[wasm_bindgen(js_name = fromZip)]
    pub fn from_zip(bytes: &[u8]) -> Result<RecordReader, JsValue> {
        let cursor = std::io::Cursor::new(bytes.to_vec());
        let mut archive = zip::ZipArchive::new(cursor).map_err(js_string_err)?;
        let store = Arc::new(MemoryStore::new());
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).map_err(|e| {
                JsValue::from_str(&format!(
                    "zip entry {index}: {e} (packed stores use stored entries only)"
                ))
            })?;
            if entry.is_dir() {
                continue;
            }
            let name = entry.name().to_string();
            let mut content = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut content).map_err(js_string_err)?;
            let skey = StoreKey::new(name.trim_start_matches('/')).map_err(js_string_err)?;
            store.set(&skey, content.into()).map_err(js_string_err)?;
        }
        let store = (store as ReadableWritableListableStorage).readable_listable();
        Self::open(store)
    }

    /// Open a store served on demand by `host`.
    ///
    /// `host` is any object with these methods, all **synchronous**:
    ///
    /// - `get(key: string): Uint8Array | null` — the whole value;
    /// - `getRange(key: string, offset: number, length: number): Uint8Array | null`
    ///   — optional; `length` may be `-1` for "to the end". Without it the
    ///   reader falls back to `get` and slices, which reads whole shards;
    /// - `size(key: string): number | null` — the value's byte length;
    /// - `list(prefix: string): string[]` — every key under `prefix`
    ///   (`""` for all).
    ///
    /// Keys are store-relative paths without a leading slash
    /// (`trajectory/step/zarr.json`).
    #[wasm_bindgen(js_name = fromStore)]
    pub fn from_store(host: JsValue) -> Result<RecordReader, JsValue> {
        let store = Arc::new(HostStore::new(host)?);
        Self::open(store)
    }

    #[wasm_bindgen(js_name = readFrame)]
    pub fn read_frame(&self, t: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self.sequence.frame(t as u64).map_err(js_string_err)?;
        match rs_frame {
            Some(frame) => Ok(Some(Frame::from_rs(frame)?)),
            None => Ok(None),
        }
    }

    /// Read frame `t` carrying only the named `"block/column"` pairs.
    ///
    /// A viewer that needs coordinates asks for `["atoms/x", "atoms/y",
    /// "atoms/z"]` and decodes nothing else. The cell and per-step metadata
    /// always come along.
    #[wasm_bindgen(js_name = readColumns)]
    pub fn read_columns(&self, t: usize, columns: Vec<String>) -> Result<Option<Frame>, JsValue> {
        let pairs: Vec<(&str, &str)> = columns
            .iter()
            .map(|spec| {
                spec.split_once('/').ok_or_else(|| {
                    JsValue::from_str(&format!("column spec {spec:?} is not \"block/column\""))
                })
            })
            .collect::<Result<_, JsValue>>()?;
        let rs_frame = self
            .sequence
            .frame_columns(t as u64, &pairs)
            .map_err(js_string_err)?;
        match rs_frame {
            Some(frame) => Ok(Some(Frame::from_rs(frame)?)),
            None => Ok(None),
        }
    }

    /// The update of block `name` that frame `t` resolves to, or `undefined`
    /// when the block is absent there.
    ///
    /// Two consecutive frames resolving to the same update carry the same
    /// rows — a stage can keep its bond buffers when `blockUpdateAt("bonds",
    /// t)` did not change, without comparing a single value.
    #[wasm_bindgen(js_name = blockUpdateAt)]
    pub fn block_update_at(&self, name: &str, t: usize) -> Result<Option<f64>, JsValue> {
        Ok(self
            .sequence
            .block_update_at(name, t as u64)
            .map_err(js_string_err)?
            .map(|update| update as f64))
    }

    /// The cell at frame `t`, or `undefined` before any cell was written.
    #[wasm_bindgen(js_name = boxAt)]
    pub fn box_at(&self, t: usize) -> Result<Option<JsBox>, JsValue> {
        Ok(self
            .sequence
            .box_at(t as u64)
            .map_err(js_string_err)?
            .map(|inner| JsBox { inner }))
    }

    #[wasm_bindgen(js_name = countFrames)]
    pub fn count_frames(&self) -> Result<usize, JsValue> {
        Ok(self.sequence.steps().len())
    }

    /// Atom count of frame 0, decoded on demand.
    ///
    /// Named for what it is: a ragged store grows its atom count per frame, so
    /// this is the *first* frame's count, not the maximum or the current one.
    #[wasm_bindgen(js_name = countAtomsAtFirstFrame)]
    pub fn count_atoms_at_first_frame(&self) -> Result<usize, JsValue> {
        Ok(self
            .sequence
            .frame(0)
            .map_err(js_string_err)?
            .and_then(|frame| frame.get("atoms").and_then(|block| block.nrows()))
            .unwrap_or(0))
    }

    /// Step numbers of the committed frames — frame labels for a replay UI.
    #[wasm_bindgen(js_name = steps)]
    pub fn steps(&self) -> Vec<i64> {
        self.sequence.steps().to_vec()
    }

    /// Physical times (fs) of the committed frames, when the run wrote any.
    #[wasm_bindgen(js_name = times)]
    pub fn times(&self) -> Option<Vec<f64>> {
        self.sequence.times().map(<[f64]>::to_vec)
    }

    /// Whether the store carries a block section of this name with at least
    /// one committed update.
    #[wasm_bindgen(js_name = hasBlock)]
    pub fn has_block(&self, name: &str) -> bool {
        self.sequence.has_block(name)
    }

    /// Names of every block section present in the store.
    #[wasm_bindgen(js_name = blockNames)]
    pub fn block_names(&self) -> Vec<String> {
        self.sequence.block_names().map(str::to_string).collect()
    }
}

impl TrajectoryReader for RecordReader {
    fn build_index(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    fn read_step(&mut self, step: usize) -> std::io::Result<Option<molrs::store::frame::Frame>> {
        self.sequence
            .frame(step as u64)
            .map_err(std::io::Error::other)
    }

    fn len(&mut self) -> std::io::Result<usize> {
        Ok(self.sequence.steps().len())
    }
}

// ---------------------------------------------------------------------------
// Host-served store
// ---------------------------------------------------------------------------

/// A read-only Zarr store whose keys are served by synchronous JS callbacks.
///
/// wasm32 has one thread, so the `Send`/`Sync` the store traits ask for are
/// vacuous here; the `unsafe impl`s below say exactly that and nothing more.
struct HostStore {
    host: JsValue,
    get: js_sys::Function,
    get_range: Option<js_sys::Function>,
    size: js_sys::Function,
    list: js_sys::Function,
}

// SAFETY: wasm32-unknown-unknown is single-threaded; no JsValue ever crosses
// a thread boundary because there is none.
unsafe impl Send for HostStore {}
unsafe impl Sync for HostStore {}

fn storage_err(context: &str, e: JsValue) -> StorageError {
    StorageError::Other(format!(
        "{context}: {}",
        e.as_string().unwrap_or_else(|| format!("{e:?}"))
    ))
}

impl HostStore {
    fn new(host: JsValue) -> Result<Self, JsValue> {
        let method = |name: &str| -> Result<Option<js_sys::Function>, JsValue> {
            let value = js_sys::Reflect::get(&host, &JsValue::from_str(name))?;
            if value.is_undefined() || value.is_null() {
                return Ok(None);
            }
            value.dyn_into::<js_sys::Function>().map(Some).map_err(|_| {
                JsValue::from_str(&format!("store host property {name:?} is not a function"))
            })
        };
        let required = |name: &str| -> Result<js_sys::Function, JsValue> {
            method(name)?.ok_or_else(|| JsValue::from_str(&format!("store host lacks {name}(…)")))
        };
        Ok(Self {
            get: required("get")?,
            get_range: method("getRange")?,
            size: required("size")?,
            list: required("list")?,
            host,
        })
    }

    fn bytes_of(value: JsValue) -> Option<Bytes> {
        if value.is_undefined() || value.is_null() {
            return None;
        }
        Some(js_sys::Uint8Array::new(&value).to_vec().into())
    }

    fn call_get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let value = self
            .get
            .call1(&self.host, &JsValue::from_str(key.as_str()))
            .map_err(|e| storage_err("get", e))?;
        Ok(Self::bytes_of(value))
    }

    fn call_size(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let value = self
            .size
            .call1(&self.host, &JsValue::from_str(key.as_str()))
            .map_err(|e| storage_err("size", e))?;
        Ok(value.as_f64().map(|size| size as u64))
    }

    fn call_list(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let value = self
            .list
            .call1(&self.host, &JsValue::from_str(prefix))
            .map_err(|e| storage_err("list", e))?;
        let array = js_sys::Array::from(&value);
        Ok(array.iter().filter_map(|item| item.as_string()).collect())
    }

    fn range(&self, key: &StoreKey, range: &ByteRange) -> Result<MaybeBytes, StorageError> {
        match &self.get_range {
            Some(get_range) => {
                let (offset, length) = match range {
                    ByteRange::FromStart(offset, Some(length)) => (*offset as f64, *length as f64),
                    ByteRange::FromStart(offset, None) => (*offset as f64, -1.0),
                    ByteRange::Suffix(length) => {
                        let Some(size) = self.call_size(key)? else {
                            return Ok(None);
                        };
                        (size.saturating_sub(*length) as f64, *length as f64)
                    }
                };
                let value = get_range
                    .call3(
                        &self.host,
                        &JsValue::from_str(key.as_str()),
                        &JsValue::from_f64(offset),
                        &JsValue::from_f64(length),
                    )
                    .map_err(|e| storage_err("getRange", e))?;
                Ok(Self::bytes_of(value))
            }
            None => {
                let Some(whole) = self.call_get(key)? else {
                    return Ok(None);
                };
                let slice = range.to_range_usize(whole.len() as u64);
                Ok(Some(whole.slice(slice)))
            }
        }
    }
}

impl ReadableStorageTraits for HostStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.call_get(key)
    }

    fn get_partial(
        &self,
        key: &StoreKey,
        byte_range: ByteRange,
    ) -> Result<MaybeBytes, StorageError> {
        self.range(key, &byte_range)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let mut out: Vec<Result<Bytes, StorageError>> = Vec::new();
        for range in byte_ranges {
            match self.range(key, &range)? {
                Some(bytes) => out.push(Ok(bytes)),
                None => return Ok(None),
            }
        }
        Ok(Some(Box::new(out.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.call_size(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.get_range.is_some()
    }
}

impl ListableStorageTraits for HostStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.list_prefix(&StorePrefix::root())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let mut keys: Vec<StoreKey> = self
            .call_list(prefix.as_str())?
            .into_iter()
            .filter_map(|name| StoreKey::new(name).ok())
            .collect();
        keys.sort();
        Ok(keys)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let mut keys = Vec::new();
        let mut prefixes = std::collections::BTreeSet::new();
        for key in self.list_prefix(prefix)? {
            let rest = key
                .as_str()
                .strip_prefix(prefix.as_str())
                .unwrap_or(key.as_str());
            match rest.find('/') {
                Some(slash) => {
                    let child = format!("{}{}", prefix.as_str(), &rest[..=slash]);
                    if let Ok(child) = StorePrefix::new(child) {
                        prefixes.insert(child);
                    }
                }
                None => keys.push(key),
            }
        }
        Ok(StoreKeysPrefixes::new(keys, prefixes.into_iter().collect()))
    }

    fn size(&self) -> Result<u64, StorageError> {
        self.size_prefix(&StorePrefix::root())
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let mut total = 0u64;
        for key in self.list_prefix(prefix)? {
            total += self.call_size(&key)?.unwrap_or(0);
        }
        Ok(total)
    }
}

#[cfg(test)]
mod export_pin;

/// Load a `Map<path, Uint8Array>` of a record's files into an in-memory store.
///
/// Shared by the record-shape doors below and shaped like the
/// `TrajectoryReader` constructor: a record that is not a frame sequence is a
/// snapshot, and a snapshot is small enough to hand over whole.
fn memory_store_from(files: &js_sys::Map) -> Result<ReadableWritableListableStorage, JsValue> {
    let store = Arc::new(MemoryStore::new());
    for key_res in files.keys() {
        let key = key_res.map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
        let path = key
            .as_string()
            .ok_or_else(|| JsValue::from_str("Invalid path key"))?;
        let content_value = files.get(&key);
        let content = js_sys::Uint8Array::new(&content_value).to_vec();
        let store_path = path.strip_prefix('/').unwrap_or(&path);
        let skey = StoreKey::new(store_path).map_err(js_string_err)?;
        store.set(&skey, content.into()).map_err(js_string_err)?;
    }
    Ok(store as ReadableWritableListableStorage)
}

/// The record's top-level sections (`"meta"`, `"frame"`, `"trajectory"`, …).
///
/// Listed, never decoded — a record's sections are independent, and asking
/// which ones exist must not cost a read of any of them. A caller holding the
/// store's keys already knows this and needs no call at all; this is for one
/// holding only an opaque store.
#[wasm_bindgen(js_name = mrecSections)]
pub fn mrec_sections(files: js_sys::Map) -> Result<Vec<String>, JsValue> {
    section_names_store(memory_store_from(&files)?).map_err(js_string_err)
}

/// Unpack a packed `*.mrec.zip` into an in-memory store.
///
/// Stored entries only, like [`RecordReader::from_zip`] — a packed record is
/// written without compression so a reader is a container walk.
fn memory_store_from_zip(bytes: &[u8]) -> Result<ReadableWritableListableStorage, JsValue> {
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let mut archive = zip::ZipArchive::new(cursor).map_err(js_string_err)?;
    let store = Arc::new(MemoryStore::new());
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(js_string_err)?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_string();
        let mut content = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut content).map_err(js_string_err)?;
        let skey = StoreKey::new(name.trim_start_matches('/')).map_err(js_string_err)?;
        store.set(&skey, content.into()).map_err(js_string_err)?;
    }
    Ok(store as ReadableWritableListableStorage)
}

/// The `frame` section of a packed `*.mrec.zip`, or `undefined`.
///
/// The packed twin of [`readMrecFrame`](read_mrec_frame).
///
/// # Errors
///
/// Throws when the bytes are not a readable packed record.
#[wasm_bindgen(js_name = readMrecFrameFromZip)]
pub fn read_mrec_frame_from_zip(bytes: &[u8]) -> Result<Option<Frame>, JsValue> {
    match read_frame_section_store(memory_store_from_zip(bytes)?, "frame").map_err(js_string_err)? {
        Some(frame) => Ok(Some(Frame::from_rs(frame)?)),
        None => Ok(None),
    }
}

/// The `frame` section of a record — its snapshot — or `undefined`.
///
/// The door for a record written by [`writeFrame`-shaped producers][molpack]:
/// molpack writes a packed configuration as `meta` + `frame/`, which
/// `TrajectoryReader` reads as a sequence of length zero. This reads the
/// snapshot it actually carries.
///
/// [molpack]: https://github.com/MolCrafts/molpack
///
/// # Errors
///
/// Throws when the files are not a readable record.
#[wasm_bindgen(js_name = readMrecFrame)]
pub fn read_mrec_frame(files: js_sys::Map) -> Result<Option<Frame>, JsValue> {
    match read_frame_section_store(memory_store_from(&files)?, "frame").map_err(js_string_err)? {
        Some(frame) => Ok(Some(Frame::from_rs(frame)?)),
        None => Ok(None),
    }
}
