//! Archive — groups multiple simulations in one Zarr store.
//!
//! Each child group of the root is a self-contained simulation with
//! `system/`, `forcefield/`, `trajectory/`, and `provenance/` sub-groups.

use std::path::Path;
use std::sync::Arc;

use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::ReadableWritableListableStorage;

use crate::error::MolRsError;
use crate::forcefield::ForceField;
use crate::frame::Frame;

use super::simulation::SimulationStore;
use super::{Provenance, UnitSystem};

/// A collection of named simulations in a single Zarr V3 store.
pub struct Archive {
    store: ReadableWritableListableStorage,
}

impl Archive {
    /// Create a new, empty archive on the filesystem.
    pub fn create_file(path: impl AsRef<Path>) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);

        let mut root_attrs = serde_json::Map::new();
        root_attrs.insert("molrs_format".into(), "archive".into());
        root_attrs.insert("version".into(), 1.into());
        GroupBuilder::new()
            .attributes(root_attrs)
            .build(store.clone(), "/")?
            .store_metadata()?;

        Ok(Self { store })
    }

    /// Open an existing archive from the filesystem.
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);

        // Validate root format
        let root = zarrs::group::Group::open(store.clone(), "/")?;
        let fmt = root
            .attributes()
            .get("molrs_format")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if fmt != "archive" {
            return Err(MolRsError::zarr(format!(
                "expected molrs_format='archive', got '{}'",
                fmt
            )));
        }

        Ok(Self { store })
    }

    /// List all simulation names in the archive.
    pub fn list_simulations(&self) -> Result<Vec<String>, MolRsError> {
        let root = Node::open(&self.store, "/")?;
        let mut names = Vec::new();
        for child in root.children() {
            if !matches!(child.metadata(), NodeMetadata::Group(_)) {
                continue;
            }
            let name = child.path().as_str().trim_start_matches('/');
            if name.is_empty() {
                continue;
            }
            // Only include groups that contain a system/ sub-group
            let system_path = format!("/{}/system", name);
            if zarrs::group::Group::open(self.store.clone(), &system_path).is_ok() {
                names.push(name.to_owned());
            }
        }
        names.sort();
        Ok(names)
    }

    /// Open an existing simulation by name.
    pub fn open_simulation(&self, name: &str) -> Result<SimulationStore, MolRsError> {
        let prefix = format!("/{}", name);
        SimulationStore::open_in_store(self.store.clone(), &prefix)
    }

    /// Create a new simulation entry in the archive.
    pub fn create_simulation(
        &mut self,
        name: &str,
        system: &Frame,
        forcefield: Option<&ForceField>,
        units: UnitSystem,
        provenance: Provenance,
    ) -> Result<SimulationStore, MolRsError> {
        let prefix = format!("/{}", name);
        SimulationStore::create_in_store(
            self.store.clone(),
            &prefix,
            system,
            forcefield,
            units,
            provenance,
        )
    }

    /// Remove a simulation from the archive.
    ///
    /// Note: Zarr V3 directory stores do not support atomic deletion.
    /// This removes the group metadata but chunk files may remain on disk
    /// until the directory is manually cleaned.
    pub fn remove_simulation(&mut self, name: &str) -> Result<(), MolRsError> {
        let prefix = format!("/{}", name);
        // Erase the group by overwriting it as an empty group, effectively
        // marking it as removed.  A full cleanup would require recursive
        // filesystem deletion, which is out of scope for the Zarr store
        // abstraction.
        let _ = zarrs::group::Group::open(self.store.clone(), &prefix)
            .map_err(|_| MolRsError::zarr(format!("simulation '{}' not found", name)))?;

        // For filesystem stores the simplest approach is to remove the directory.
        // The zarrs crate does not expose a recursive delete, so we use std::fs.
        // This is acceptable because Archive only works with filesystem feature.
        use zarrs::storage::WritableStorageTraits;
        let prefix_key =
            zarrs::storage::StorePrefix::new(name).map_err(|e| MolRsError::zarr(e.to_string()))?;
        let _ = self.store.erase_prefix(&prefix_key);
        Ok(())
    }
}

fn zerr(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(e.to_string())
}
