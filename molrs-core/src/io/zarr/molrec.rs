//! MolRec Zarr v3 backend.

#[cfg(feature = "filesystem")]
use std::path::Path;
#[cfg(feature = "filesystem")]
use std::sync::Arc;

#[cfg(feature = "filesystem")]
use zarrs::filesystem::FilesystemStore;
#[cfg(feature = "filesystem")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::ReadableWritableListableStorage;

use crate::error::MolRsError;
use crate::molrec::MolRec;

/// Internal Zarr v3 backend state for MolRec.
pub(crate) struct MolRecZarrBackend {
    store: ReadableWritableListableStorage,
    prefix: String,
}

impl MolRecZarrBackend {
    #[cfg(feature = "filesystem")]
    pub fn create_file(path: impl AsRef<Path>, molrec: &MolRec) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
        Self::create_in_store(store, "/", molrec)
    }

    #[cfg(feature = "filesystem")]
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
        Self::open_in_store(store, "/")
    }

    pub fn open_store(store: ReadableWritableListableStorage) -> Result<Self, MolRsError> {
        Self::open_in_store(store, "/")
    }

    #[cfg(feature = "filesystem")]
    pub(crate) fn create_in_store(
        store: ReadableWritableListableStorage,
        prefix: &str,
        molrec: &MolRec,
    ) -> Result<Self, MolRsError> {
        let mut attrs = serde_json::Map::new();
        attrs.insert("molrs_format".into(), "molrec".into());
        attrs.insert("version".into(), 1.into());
        attrs.insert("frame_count".into(), (molrec.count_frames() as u64).into());
        GroupBuilder::new()
            .attributes(attrs)
            .build(store.clone(), prefix)?
            .store_metadata()?;

        write_string_map(&store, &super::frame_io::join_path(prefix, "meta"), &molrec.meta)?;
        write_string_map(
            &store,
            &super::frame_io::join_path(prefix, "method"),
            &molrec.method,
        )?;

        let mut base_frame = molrec.frame.clone();
        let field_names: Vec<String> = base_frame.field_names().map(str::to_string).collect();
        for name in field_names {
            let _ = base_frame.remove_field(&name);
        }
        super::frame_io::write_system(
            &store,
            &super::frame_io::join_path(prefix, "frame"),
            &base_frame,
        )?;

        if molrec.frame.field_names().next().is_some() {
            let observable_prefix = super::frame_io::join_path(prefix, "observable");
            GroupBuilder::new()
                .build(store.clone(), &observable_prefix)?
                .store_metadata()?;
            for (name, field) in molrec.frame.fields() {
                super::frame_io::write_field_observable(
                    &store,
                    &super::frame_io::join_path(&observable_prefix, name),
                    field,
                )?;
            }
        }

        if !molrec.trajectory.is_empty() {
            let traj_prefix = super::frame_io::join_path(prefix, "trajectory");
            GroupBuilder::new()
                .build(store.clone(), &traj_prefix)?
                .store_metadata()?;
            for (index, frame) in molrec.trajectory.iter().enumerate() {
                super::frame_io::write_system(
                    &store,
                    &super::frame_io::join_path(&traj_prefix, &index.to_string()),
                    frame,
                )?;
            }
        }

        Ok(Self {
            store,
            prefix: prefix.to_string(),
        })
    }

    pub(crate) fn open_in_store(
        store: ReadableWritableListableStorage,
        prefix: &str,
    ) -> Result<Self, MolRsError> {
        let root = zarrs::group::Group::open(store.clone(), prefix)?;
        let format = root
            .attributes()
            .get("molrs_format")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if format != "molrec" {
            return Err(MolRsError::zarr(format!(
                "expected MolRec zarr v3 store, found '{}'",
                format
            )));
        }
        Ok(Self {
            store,
            prefix: prefix.to_string(),
        })
    }

    pub fn read(&self) -> Result<MolRec, MolRsError> {
        let mut rec = MolRec::new(super::frame_io::read_system(
            &self.store,
            &super::frame_io::join_path(&self.prefix, "frame"),
        )?);
        rec.meta = read_string_map(&self.store, &super::frame_io::join_path(&self.prefix, "meta"))?;
        rec.method = read_string_map(
            &self.store,
            &super::frame_io::join_path(&self.prefix, "method"),
        )?;

        let observable_prefix = super::frame_io::join_path(&self.prefix, "observable");
        if let Ok(node) = Node::open(&self.store, &observable_prefix) {
            for child in node.children() {
                if !matches!(child.metadata(), NodeMetadata::Group(_)) {
                    continue;
                }
                let name = child.path().as_str().rsplit('/').next().unwrap_or("");
                if name.is_empty() {
                    continue;
                }
                let field = super::frame_io::read_field_observable(&self.store, child.path().as_str())?;
                rec.frame.add_field(name.to_string(), field);
            }
        }

        let traj_prefix = super::frame_io::join_path(&self.prefix, "trajectory");
        if let Ok(node) = Node::open(&self.store, &traj_prefix) {
            let mut children: Vec<_> = node
                .children()
                .into_iter()
                .filter(|child| matches!(child.metadata(), NodeMetadata::Group(_)))
                .collect();
            children.sort_by_key(|child| {
                child
                    .path()
                    .as_str()
                    .rsplit('/')
                    .next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(usize::MAX)
            });
            for child in children {
                rec.trajectory.push(super::frame_io::read_system(
                    &self.store,
                    child.path().as_str(),
                )?);
            }
        }

        Ok(rec)
    }

    pub fn count_frames(&self) -> Result<u64, MolRsError> {
        let root = zarrs::group::Group::open(self.store.clone(), &self.prefix)?;
        Ok(root
            .attributes()
            .get("frame_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(1))
    }

    pub fn read_frame(&self, index: usize) -> Result<Option<crate::frame::Frame>, MolRsError> {
        let rec = self.read()?;
        Ok(rec.frame_at(index))
    }
}

#[cfg(feature = "filesystem")]
pub fn write_molrec_file(path: impl AsRef<Path>, molrec: &MolRec) -> Result<(), MolRsError> {
    let _ = MolRecZarrBackend::create_file(path, molrec)?;
    Ok(())
}

#[cfg(feature = "filesystem")]
pub fn read_molrec_file(path: impl AsRef<Path>) -> Result<MolRec, MolRsError> {
    MolRecZarrBackend::open_file(path)?.read()
}

pub fn read_molrec_store(
    store: ReadableWritableListableStorage,
) -> Result<MolRec, MolRsError> {
    MolRecZarrBackend::open_store(store)?.read()
}

pub fn read_molrec_frame_from_store(
    store: ReadableWritableListableStorage,
    index: usize,
) -> Result<Option<crate::frame::Frame>, MolRsError> {
    MolRecZarrBackend::open_store(store)?.read_frame(index)
}

pub fn count_molrec_frames_in_store(
    store: ReadableWritableListableStorage,
) -> Result<u64, MolRsError> {
    MolRecZarrBackend::open_store(store)?.count_frames()
}

#[cfg(feature = "filesystem")]
fn write_string_map(
    store: &ReadableWritableListableStorage,
    path: &str,
    map: &std::collections::BTreeMap<String, String>,
) -> Result<(), MolRsError> {
    let mut attrs = serde_json::Map::new();
    for (k, v) in map {
        attrs.insert(k.clone(), serde_json::Value::String(v.clone()));
    }
    GroupBuilder::new()
        .attributes(attrs)
        .build(store.clone(), path)?
        .store_metadata()?;
    Ok(())
}

fn read_string_map(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<std::collections::BTreeMap<String, String>, MolRsError> {
    let mut out = std::collections::BTreeMap::new();
    let group = match zarrs::group::Group::open(store.clone(), path) {
        Ok(group) => group,
        Err(_) => return Ok(out),
    };
    for (k, v) in group.attributes() {
        if let Some(s) = v.as_str() {
            out.insert(k.clone(), s.to_string());
        } else {
            out.insert(k.clone(), v.to_string());
        }
    }
    Ok(out)
}

#[cfg(feature = "filesystem")]
fn zerr<E: std::fmt::Display>(e: E) -> MolRsError {
    MolRsError::zarr(e.to_string())
}

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use ndarray::Array1;

    use crate::block::Block;
    use crate::field::{FieldObservable, UniformGridField};
    use crate::types::F;

    use super::*;

    #[test]
    fn molrec_store_roundtrip_preserves_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("molrec.zarr");

        let mut atoms = Block::new();
        atoms.insert(
            "element",
            Array1::from_vec(vec!["H".to_string(), "H".to_string()]).into_dyn(),
        )
        .unwrap();
        atoms.insert("x", Array1::from_vec(vec![0.0 as F, 0.74 as F]).into_dyn())
            .unwrap();
        atoms.insert("y", Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn())
            .unwrap();
        atoms.insert("z", Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn())
            .unwrap();

        let mut frame = crate::Frame::new();
        frame.insert("atoms", atoms);
        frame.add_field(
            "electron_density",
            FieldObservable::uniform_grid(
                "electron_density",
                "electron_density",
                "e/Angstrom^3",
                UniformGridField {
                    shape: [2, 2, 2],
                    origin: [0.0, 0.0, 0.0],
                    cell: [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                    pbc: [false, false, false],
                    values: vec![0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.0, 0.2],
                },
            ),
        );

        let mut rec = MolRec::new(frame);
        rec.meta.insert("creator".into(), "test".into());

        write_molrec_file(&path, &rec).unwrap();
        let loaded = read_molrec_file(&path).unwrap();

        assert_eq!(loaded.count_frames(), 1);
        assert_eq!(loaded.meta.get("creator").map(String::as_str), Some("test"));
        assert!(loaded.frame.has_field("electron_density"));
        assert_eq!(
            loaded
                .frame
                .get("atoms")
                .and_then(|atoms| atoms.nrows()),
            Some(2)
        );
    }
}
