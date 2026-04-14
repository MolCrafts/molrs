//! Global store: mutex-protected singleton owning all C-API state.
//!
//! All FFI objects (frames, blocks, SimBoxes, force fields) and interned
//! key strings are owned by a single [`CStore`] instance, accessed via
//! the [`lock_store`] function.
//!
//! The store is lazily initialised on first access and lives for the
//! entire process lifetime.  [`molrs_shutdown`](crate::molrs_shutdown)
//! clears all contents but does not destroy the mutex.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{LazyLock, Mutex, MutexGuard};

use molrs::region::simbox::SimBox;
use molrs_ff::ForceField;
use slotmap::SlotMap;

use crate::handle::{FFKey, SimBoxKey};

/// Central store owning all C-API state.
///
/// This struct is the single source of truth for every object
/// reachable via a C-API handle.  It is wrapped in a `Mutex` and
/// accessed through [`lock_store`].
pub(crate) struct CStore {
    /// Frame/block management delegated to `molrs-ffi`.
    pub inner: molrs_ffi::Store,

    /// Interned key strings, stored null-terminated for direct return
    /// to C callers via [`molrs_key_name`](crate::molrs_key_name).
    /// The vector index is the `key_id`.
    pub interned_keys: Vec<CString>,

    /// Reverse lookup: Rust `String` to interned `key_id` (`u32`).
    pub key_to_id: HashMap<String, u32>,

    /// Standalone SimBox instances, keyed by [`SimBoxKey`].
    pub simboxes: SlotMap<SimBoxKey, SimBox>,

    /// Standalone ForceField instances, keyed by [`FFKey`].
    pub forcefields: SlotMap<FFKey, ForceField>,
}

impl CStore {
    fn new() -> Self {
        Self {
            inner: molrs_ffi::Store::new(),
            interned_keys: Vec::new(),
            key_to_id: HashMap::new(),
            simboxes: SlotMap::with_key(),
            forcefields: SlotMap::with_key(),
        }
    }

    /// Intern a key string, returning its id. Idempotent.
    pub fn intern(&mut self, key: &str) -> u32 {
        if let Some(&id) = self.key_to_id.get(key) {
            return id;
        }
        let id = self.interned_keys.len() as u32;
        let cstr = CString::new(key).unwrap_or_default();
        self.interned_keys.push(cstr);
        self.key_to_id.insert(key.to_owned(), id);
        id
    }

    /// Look up the Rust string for an interned key_id.
    pub fn key_str(&self, id: u32) -> Option<&str> {
        self.interned_keys
            .get(id as usize)
            .and_then(|cs| cs.to_str().ok())
    }

    /// Reset all state.
    pub fn clear(&mut self) {
        self.inner = molrs_ffi::Store::new();
        self.interned_keys.clear();
        self.key_to_id.clear();
        self.simboxes.clear();
        self.forcefields.clear();
    }
}

/// Global singleton store.
pub(crate) static GLOBAL_STORE: LazyLock<Mutex<CStore>> =
    LazyLock::new(|| Mutex::new(CStore::new()));

/// Lock the global store, returning a guard.
pub(crate) fn lock_store() -> MutexGuard<'static, CStore> {
    GLOBAL_STORE.lock().expect("GLOBAL_STORE poisoned")
}
