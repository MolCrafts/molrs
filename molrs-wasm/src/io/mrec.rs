//! WASM bindings for frame-sequence Zarr v3 stores (`*.mrec`).
//!
//! Three ways to hand the reader its bytes, in increasing laziness:
//!
//! - **`new MrecReader(files)`** — a `Map<path, Uint8Array>` of every
//!   file, copied once into an in-memory store. Simple; the whole store is
//!   resident.
//! - **`MrecReader.fromZip(bytes)`** — one packed `*.mrec.zip`, whose
//!   stored entries are unpacked into the same in-memory store.
//! - **`MrecReader.fromStorage(host)`** — a host object that serves keys
//!   on demand: `get(key)`, `getRange(key, offset, length)`, `size(key)` and
//!   `list(prefix)`. Only the chunks a frame touches ever cross into wasm, so
//!   a multi-gigabyte run opens in a worker that reads its files (or an HTTP
//!   range server) synchronously.
//!
//! A record that is not a frame sequence is read whole: `sectionNames(source)`
//! lists its sections and `readMrecFrame(source)` reads its `frame` snapshot,
//! where `source` is the files `Map` or the packed zip's bytes.

use crate::core::frame::Frame;
use crate::core::simbox::Box as JsBox;
use molrs::io::mrec::{MrecReader as RsMrecReader, section_names_storage};
use molrs::io::read_mrec_frame_storage;
use molrs::io::reader::TrajectoryReader;
use std::io::Read;
use std::sync::Arc;
use wasm_bindgen::JsCast;
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
/// The sequence is opened **once**, in the constructor. `MrecReader` is the
/// lazy store cursor, so `readFrame` decodes exactly the frame it was asked
/// for (keeping the last decoded chunk of every column, so consecutive frames
/// are slices), and `nFrames` answers off the index the open already
/// cached. Every method takes `&self`: the cursor holds caches, not a
/// position.
#[wasm_bindgen(js_name = MrecReader)]
pub struct MrecReader {
    sequence: RsMrecReader,
}

impl MrecReader {
    fn open<S>(store: Arc<S>) -> Result<MrecReader, JsValue>
    where
        S: ?Sized
            + zarrs::storage::ReadableStorageTraits
            + zarrs::storage::ListableStorageTraits
            + 'static,
    {
        // Index-only: the schema plus each section's step_index and offset (or
        // their hints). No frame is decoded here.
        let sequence = RsMrecReader::from_storage(store).map_err(js_string_err)?;
        Ok(MrecReader { sequence })
    }
}

#[wasm_bindgen(js_class = MrecReader)]
impl MrecReader {
    /// Open a store handed over as a `Map<path, Uint8Array>` of every file.
    #[wasm_bindgen(constructor)]
    pub fn new(files: js_sys::Map) -> Result<MrecReader, JsValue> {
        // The reader is a read door, so it gets the store's read-only view.
        Self::open(memory_storage_from_files(&files)?.readable_listable())
    }

    /// Open a packed `*.mrec.zip` from its bytes.
    ///
    /// Every entry of a packed store is *stored* (never deflated), so this is
    /// a central-directory walk plus one copy per entry into an in-memory
    /// store. A zip with compressed entries is refused by name.
    #[wasm_bindgen(js_name = fromZip)]
    pub fn from_zip(bytes: &[u8]) -> Result<MrecReader, JsValue> {
        Self::open(memory_storage_from_zip(bytes)?.readable_listable())
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
    #[wasm_bindgen(js_name = fromStorage)]
    pub fn from_storage(host: JsValue) -> Result<MrecReader, JsValue> {
        let store = Arc::new(HostStorage::new(host)?);
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

    #[wasm_bindgen(js_name = nFrames)]
    pub fn n_frames(&self) -> Result<usize, JsValue> {
        Ok(self.sequence.steps().len())
    }

    /// Atom count of frame 0, decoded on demand.
    ///
    /// Named for what it is: a ragged store grows its atom count per frame, so
    /// this is the *first* frame's count, not the maximum or the current one.
    #[wasm_bindgen(js_name = nAtomsAtFirstFrame)]
    pub fn n_atoms_at_first_frame(&self) -> Result<usize, JsValue> {
        Ok(self
            .sequence
            .frame(0)
            .map_err(js_string_err)?
            .and_then(|frame| frame.get("atoms").and_then(|block| block.n_rows()))
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

impl TrajectoryReader for MrecReader {
    fn build_index(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    fn read_step(&mut self, step: usize) -> std::io::Result<Option<molrs::core::Frame>> {
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
struct HostStorage {
    host: JsValue,
    get: js_sys::Function,
    get_range: Option<js_sys::Function>,
    size: js_sys::Function,
    list: js_sys::Function,
}

// SAFETY: wasm32-unknown-unknown is single-threaded; no JsValue ever crosses
// a thread boundary because there is none.
unsafe impl Send for HostStorage {}
unsafe impl Sync for HostStorage {}

fn storage_err(context: &str, e: JsValue) -> StorageError {
    StorageError::Other(format!(
        "{context}: {}",
        e.as_string().unwrap_or_else(|| format!("{e:?}"))
    ))
}

impl HostStorage {
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

impl ReadableStorageTraits for HostStorage {
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

impl ListableStorageTraits for HostStorage {
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

/// Load a `Map<path, Uint8Array>` of a record's files into an in-memory store
/// (the `MrecReader` constructor and the doors below).
fn memory_storage_from_files(
    files: &js_sys::Map,
) -> Result<ReadableWritableListableStorage, JsValue> {
    let store = Arc::new(MemoryStore::new());
    for key_res in files.keys() {
        let key = key_res.map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
        let path = key
            .as_string()
            .ok_or_else(|| JsValue::from_str("Invalid path key"))?;
        let content_value = files.get(&key);
        let content = js_sys::Uint8Array::new(&content_value).to_vec();
        let storage_path = path.strip_prefix('/').unwrap_or(&path);
        let skey = StoreKey::new(storage_path).map_err(js_string_err)?;
        store.set(&skey, content.into()).map_err(js_string_err)?;
    }
    Ok(store as ReadableWritableListableStorage)
}

/// Unpack a packed `*.mrec.zip` into an in-memory store (`MrecReader.fromZip`
/// and the doors below).
///
/// Every entry of a packed store is *stored* (never deflated), so this is a
/// central-directory walk plus one copy per entry. A zip with compressed
/// entries is refused by name.
fn memory_storage_from_zip(bytes: &[u8]) -> Result<ReadableWritableListableStorage, JsValue> {
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
    Ok(store as ReadableWritableListableStorage)
}

#[wasm_bindgen]
extern "C" {
    /// A record handed over whole: a `Map<path, Uint8Array>` of its files, or
    /// the bytes of a packed `*.mrec.zip`.
    #[wasm_bindgen(typescript_type = "Map<string, Uint8Array> | Uint8Array")]
    pub type MrecSource;
}

/// The in-memory store of a record handed over whole (see [`MrecSource`]).
fn memory_storage(source: &MrecSource) -> Result<ReadableWritableListableStorage, JsValue> {
    if let Some(files) = source.dyn_ref::<js_sys::Map>() {
        memory_storage_from_files(files)
    } else if let Some(bytes) = source.dyn_ref::<js_sys::Uint8Array>() {
        memory_storage_from_zip(&bytes.to_vec())
    } else {
        Err(JsValue::from_str(
            "a record is a Map<path, Uint8Array> of its files or the bytes of a packed *.mrec.zip",
        ))
    }
}

/// The record's top-level sections (`"meta"`, `"frame"`, `"trajectory"`, …):
/// molrs `io::mrec::section_names`.
///
/// Listed, never decoded — a record's sections are independent, and asking
/// which ones exist must not cost a read of any of them.
///
/// `source` is a `Map<path, Uint8Array>` of the record's files or the bytes
/// of a packed `*.mrec.zip`.
#[wasm_bindgen(js_name = sectionNames)]
pub fn section_names(source: &MrecSource) -> Result<Vec<String>, JsValue> {
    section_names_storage(memory_storage(source)?).map_err(js_string_err)
}

/// The `frame` section of a record — its snapshot — or `undefined`: molrs
/// `io::read_mrec_frame`.
///
/// molpack writes a packed configuration as `meta` + `frame/`, which
/// `MrecReader` reads as a sequence of length zero; this reads the snapshot
/// it actually carries. `source` is a `Map<path, Uint8Array>` of the record's
/// files or the bytes of a packed `*.mrec.zip`.
///
/// # Errors
///
/// Throws when the source is not a readable record.
#[wasm_bindgen(js_name = readMrecFrame)]
pub fn read_mrec_frame(source: &MrecSource) -> Result<Option<Frame>, JsValue> {
    match read_mrec_frame_storage(memory_storage(source)?, "frame").map_err(js_string_err)? {
        Some(frame) => Ok(Some(Frame::from_rs(frame)?)),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{quantize, quantum};
    use wasm_bindgen_test::*;

    /// A record whose `frame/atoms/x` and `trajectory/atoms/x` declare
    /// precision 1e-3 and were written natively with `numcodecs.shuffle` +
    /// `zstd` (C encoder). Regenerated by molrs's ignored test
    /// `regenerate_the_wasm_precision_fixture`.
    const FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/precision.mrec.zip");

    /// The values the fixture's `x` columns were presented with.
    const X: [[f64; 3]; 2] = [[0.123_456_789, -1.000_488, 7.3], [0.2, 1.75, -3.062_57]];

    fn rounded(values: &[f64]) -> Vec<f64> {
        let q = quantum(1e-3).unwrap();
        values.iter().map(|&x| quantize(x, q)).collect()
    }

    fn x_of(frame: &molrs::core::Frame) -> Vec<f64> {
        frame
            .get("atoms")
            .and_then(|atoms| atoms.get("x"))
            .and_then(|x| x.as_float())
            .expect("atoms.x is f64")
            .iter()
            .copied()
            .collect()
    }

    /// zstd decodes on wasm32 through the pure-Rust plugin, with no C.
    #[wasm_bindgen_test]
    fn a_precision_trajectory_written_with_zstd_reads_back() {
        let reader = MrecReader::from_zip(FIXTURE).unwrap();
        assert_eq!(reader.n_frames().unwrap(), X.len());
        for (index, values) in X.iter().enumerate() {
            let frame = reader.sequence.frame(index as u64).unwrap().unwrap();
            assert_eq!(x_of(&frame), rounded(values));
        }
        let frame = reader.read_frame(0).unwrap().unwrap();
        assert_eq!(
            frame.get("atoms").unwrap().precision("x").unwrap(),
            Some(1e-3)
        );
    }

    #[wasm_bindgen_test]
    fn a_precision_frame_written_with_zstd_reads_back() {
        let store = memory_storage_from_zip(FIXTURE).unwrap();
        let frame = read_mrec_frame_storage(store, "frame").unwrap().unwrap();
        assert_eq!(x_of(&frame), rounded(&X[0]));
        assert_eq!(frame.get("atoms").unwrap().precision("x"), Some(1e-3));
    }
}
