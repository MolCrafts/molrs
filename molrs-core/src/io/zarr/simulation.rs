//! SimulationStore — one Zarr V3 directory per simulation.
//!
//! Contains system (topology Frame), force field (optional), trajectory
//! (optional), and provenance/unit metadata.

#[cfg(feature = "filesystem")]
use std::path::Path;
#[cfg(feature = "filesystem")]
use std::sync::Arc;

#[cfg(feature = "filesystem")]
use zarrs::filesystem::FilesystemStore;
#[cfg(feature = "filesystem")]
use zarrs::group::GroupBuilder;
use zarrs::storage::ReadableWritableListableStorage;

use crate::error::MolRsError;
use crate::forcefield::ForceField;
use crate::frame::Frame;

use super::trajectory::TrajectoryReader;
#[cfg(feature = "filesystem")]
use super::trajectory::{TrajectoryConfig, TrajectoryWriter};
use super::{Provenance, UnitSystem};

/// A single simulation stored as a Zarr V3 directory.
///
/// Holds system topology, optional force field, optional trajectory, and
/// provenance metadata.  Operates relative to a `prefix` path so the same
/// code works standalone (`"/"`) or inside an [`super::Archive`].
pub struct SimulationStore {
    store: ReadableWritableListableStorage,
    prefix: String,
    n_atoms: u64,
}

impl SimulationStore {
    // -- Create / Open -------------------------------------------------------

    /// Create a new simulation store on the filesystem.
    #[cfg(feature = "filesystem")]
    pub fn create_file(
        path: impl AsRef<Path>,
        system: &Frame,
        forcefield: Option<&ForceField>,
        units: UnitSystem,
        provenance: Provenance,
    ) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
        Self::create_in_store(store, "/", system, forcefield, units, provenance)
    }

    /// Open an existing simulation store from the filesystem.
    #[cfg(feature = "filesystem")]
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
        Self::open_in_store(store, "/")
    }

    /// Open from a generic Zarr backend (e.g., MemoryStore for WASM).
    pub fn open_store(store: ReadableWritableListableStorage) -> Result<Self, MolRsError> {
        Self::open_in_store(store, "/")
    }

    // -- Internal constructors -----------------------------------------------

    #[cfg(feature = "filesystem")]
    pub(crate) fn create_in_store(
        store: ReadableWritableListableStorage,
        prefix: &str,
        system: &Frame,
        forcefield: Option<&ForceField>,
        units: UnitSystem,
        provenance: Provenance,
    ) -> Result<Self, MolRsError> {
        let n_atoms = system
            .get("atoms")
            .and_then(|b| b.nrows())
            .ok_or_else(|| MolRsError::zarr("system frame has no atoms block"))?
            as u64;

        // Root group (or prefix group for archive)
        let mut root_attrs = serde_json::Map::new();
        root_attrs.insert("molrs_format".into(), "simulation".into());
        root_attrs.insert("version".into(), 1.into());
        root_attrs.insert(
            "unit_system".into(),
            serde_json::Value::String(units.as_str().to_owned()),
        );
        GroupBuilder::new()
            .attributes(root_attrs)
            .build(store.clone(), prefix)?
            .store_metadata()?;

        // System
        let system_prefix = format_prefix(prefix, "system");
        super::frame_io::write_system(&store, &system_prefix, system)?;

        // ForceField
        if let Some(ff) = forcefield {
            super::forcefield::write_forcefield(&store, prefix, ff)?;
        }

        // Provenance
        write_provenance(&store, prefix, &provenance)?;

        Ok(Self {
            store,
            prefix: prefix.to_owned(),
            n_atoms,
        })
    }

    pub(crate) fn open_in_store(
        store: ReadableWritableListableStorage,
        prefix: &str,
    ) -> Result<Self, MolRsError> {
        // Read n_atoms from system/atoms
        let system_prefix = format_prefix(prefix, "system");
        let system = super::frame_io::read_system(&store, &system_prefix)?;
        let n_atoms = system
            .get("atoms")
            .and_then(|b| b.nrows())
            .ok_or_else(|| MolRsError::zarr("system frame has no atoms block"))?
            as u64;

        Ok(Self {
            store,
            prefix: prefix.to_owned(),
            n_atoms,
        })
    }

    // -- System --------------------------------------------------------------

    /// Read the system topology/structure as a Frame.
    pub fn read_system(&self) -> Result<Frame, MolRsError> {
        let system_prefix = format_prefix(&self.prefix, "system");
        super::frame_io::read_system(&self.store, &system_prefix)
    }

    /// Number of atoms in the system.
    pub fn count_atoms(&self) -> u64 {
        self.n_atoms
    }

    // -- ForceField ----------------------------------------------------------

    /// Read the force field, if stored.
    pub fn read_forcefield(&self) -> Result<Option<ForceField>, MolRsError> {
        super::forcefield::read_forcefield(&self.store, &self.prefix)
    }

    // -- Trajectory ----------------------------------------------------------

    /// Create a streaming trajectory writer.
    #[cfg(feature = "filesystem")]
    pub fn create_trajectory(
        &mut self,
        config: TrajectoryConfig,
    ) -> Result<TrajectoryWriter, MolRsError> {
        TrajectoryWriter::create(&self.store, &self.prefix, self.n_atoms, &config)
    }

    /// Get a trajectory reader, if a trajectory exists.
    pub fn open_trajectory(&self) -> Result<Option<TrajectoryReader>, MolRsError> {
        let traj_prefix = format_prefix(&self.prefix, "trajectory");
        let step_path = format!("{}/step", traj_prefix);
        if !super::frame_io::array_exists(&self.store, &step_path) {
            return Ok(None);
        }
        Ok(Some(TrajectoryReader::open(
            self.store.clone(),
            &self.prefix,
            self.n_atoms,
        )?))
    }

    // -- Provenance / Units --------------------------------------------------

    /// Read provenance metadata.
    pub fn read_provenance(&self) -> Result<Provenance, MolRsError> {
        read_provenance(&self.store, &self.prefix)
    }

    /// Read the unit system.
    pub fn read_units(&self) -> Result<UnitSystem, MolRsError> {
        let group = zarrs::group::Group::open(self.store.clone(), &self.prefix)?;
        let us = group
            .attributes()
            .get("unit_system")
            .and_then(|v| v.as_str())
            .and_then(UnitSystem::parse)
            .unwrap_or_default();
        Ok(us)
    }
}

// ---------------------------------------------------------------------------
// Provenance helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
fn write_provenance(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    prov: &Provenance,
) -> Result<(), MolRsError> {
    let path = format_prefix(prefix, "provenance");
    let mut attrs = serde_json::Map::new();
    if let Some(ref p) = prov.program {
        attrs.insert("program".into(), serde_json::Value::String(p.clone()));
    }
    if let Some(ref v) = prov.version {
        attrs.insert("version".into(), serde_json::Value::String(v.clone()));
    }
    if let Some(ref m) = prov.method {
        attrs.insert("method".into(), serde_json::Value::String(m.clone()));
    }
    if let Some(s) = prov.seed {
        attrs.insert("seed".into(), serde_json::Value::from(s));
    }
    GroupBuilder::new()
        .attributes(attrs)
        .build(store.clone(), &path)?
        .store_metadata()?;
    Ok(())
}

fn read_provenance(
    store: &ReadableWritableListableStorage,
    prefix: &str,
) -> Result<Provenance, MolRsError> {
    let path = format_prefix(prefix, "provenance");
    let group = match zarrs::group::Group::open(store.clone(), &path) {
        Ok(g) => g,
        Err(_) => return Ok(Provenance::default()),
    };
    let attrs = group.attributes();
    Ok(Provenance {
        program: attrs
            .get("program")
            .and_then(|v| v.as_str())
            .map(String::from),
        version: attrs
            .get("version")
            .and_then(|v| v.as_str())
            .map(String::from),
        method: attrs
            .get("method")
            .and_then(|v| v.as_str())
            .map(String::from),
        seed: attrs.get("seed").and_then(|v| v.as_u64()),
    })
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Alias for shared path-join utility.
fn format_prefix(prefix: &str, child: &str) -> String {
    super::frame_io::join_path(prefix, child)
}

#[cfg(feature = "filesystem")]
fn zerr(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(e.to_string())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use super::*;
    use ndarray::{Array1, array};
    use std::collections::HashMap;

    use crate::block::Block;
    use crate::region::simbox::SimBox;

    use super::super::trajectory::{TrajectoryConfig, TrajectoryFrame};

    fn make_test_system() -> Frame {
        use crate::types::F;
        let mut frame = Frame::new();

        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![5.0 as F, 6.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert(
                "mass",
                Array1::from_vec(vec![12.0 as F, 1.0 as F]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["C".to_string(), "H".to_string()]).into_dyn(),
            )
            .unwrap();
        frame.insert("atoms", atoms);

        let mut bonds = Block::new();
        bonds
            .insert("i", Array1::from_vec(vec![0u32]).into_dyn())
            .unwrap();
        bonds
            .insert("j", Array1::from_vec(vec![1u32]).into_dyn())
            .unwrap();
        frame.insert("bonds", bonds);

        frame.meta.insert("title".into(), "test molecule".into());
        frame.simbox = Some(
            SimBox::cube(
                10.0 as crate::types::F,
                array![
                    0.0 as crate::types::F,
                    0.0 as crate::types::F,
                    0.0 as crate::types::F
                ],
                [true, true, true],
            )
            .unwrap(),
        );

        frame
    }

    fn make_test_ff() -> ForceField {
        let mut ff = ForceField::new("test_ff");
        ff.def_atomstyle("full")
            .def_atomtype("CT", &[("mass", 12.011), ("charge", -0.12)]);
        ff.def_bondstyle("harmonic")
            .def_bondtype("CT", "HC", &[("k0", 340.0), ("r0", 1.09)]);
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_pairtype("CT", None, &[("epsilon", 0.066), ("sigma", 3.5)]);
        ff
    }

    // -- System round-trip ---------------------------------------------------

    #[test]
    fn test_system_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sim.zarr");
        let system = make_test_system();

        SimulationStore::create_file(
            &path,
            &system,
            None,
            UnitSystem::Real,
            Provenance::default(),
        )
        .unwrap();

        let store = SimulationStore::open_file(&path).unwrap();
        let loaded = store.read_system().unwrap();

        // atoms block
        let atoms = loaded.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(2));
        let x = atoms.get_float("x").unwrap();
        assert_eq!(
            x.as_slice().unwrap(),
            &[1.0 as crate::types::F, 2.0 as crate::types::F]
        );
        let elem = atoms.get_string("element").unwrap();
        assert_eq!(elem[0], "C");

        // bonds block
        let bonds = loaded.get("bonds").unwrap();
        assert_eq!(bonds.nrows(), Some(1));

        // meta
        assert_eq!(loaded.meta.get("title").unwrap(), "test molecule");

        // simbox
        let sb = loaded.simbox.as_ref().unwrap();
        assert!((sb.volume() - 1000.0).abs() < 1e-3);
    }

    // -- ForceField round-trip -----------------------------------------------

    #[test]
    fn test_forcefield_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sim.zarr");
        let system = make_test_system();
        let ff = make_test_ff();

        SimulationStore::create_file(
            &path,
            &system,
            Some(&ff),
            UnitSystem::Real,
            Provenance::default(),
        )
        .unwrap();

        let store = SimulationStore::open_file(&path).unwrap();
        let loaded_ff = store.read_forcefield().unwrap().unwrap();

        assert_eq!(loaded_ff.name, "test_ff");

        // atom types
        let at = loaded_ff.get_atomtypes();
        assert_eq!(at.len(), 1);
        assert_eq!(at[0].name, "CT");
        assert_eq!(at[0].params.get("mass"), Some(12.011));
        assert_eq!(at[0].params.get("charge"), Some(-0.12));

        // bond types
        let bt = loaded_ff.get_bondtypes();
        assert_eq!(bt.len(), 1);
        assert_eq!(bt[0].itom, "CT");
        assert_eq!(bt[0].jtom, "HC");
        assert_eq!(bt[0].params.get("k0"), Some(340.0));

        // pair types
        let pt = loaded_ff.get_pairtypes();
        assert_eq!(pt.len(), 1);
        let style = loaded_ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(style.params.get("cutoff"), Some(10.0));
    }

    // -- No FF ---------------------------------------------------------------

    #[test]
    fn test_no_forcefield() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sim.zarr");
        let system = make_test_system();

        SimulationStore::create_file(
            &path,
            &system,
            None,
            UnitSystem::Real,
            Provenance::default(),
        )
        .unwrap();

        let store = SimulationStore::open_file(&path).unwrap();
        assert!(store.read_forcefield().unwrap().is_none());
    }

    // -- Trajectory round-trip -----------------------------------------------

    #[test]
    fn test_trajectory_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sim.zarr");
        let system = make_test_system();

        let mut store = SimulationStore::create_file(
            &path,
            &system,
            None,
            UnitSystem::Real,
            Provenance::default(),
        )
        .unwrap();

        let config = TrajectoryConfig {
            positions: true,
            velocities: true,
            scalars: vec!["PE".into(), "KE".into()],
            chunk_size: 100,
            ..TrajectoryConfig::default()
        };

        let mut writer = store.create_trajectory(config).unwrap();

        for i in 0..10u64 {
            let pos: Vec<f32> = vec![
                1.0 + i as f32 * 0.1,
                3.0,
                5.0,
                2.0,
                4.0,
                6.0 + i as f32 * 0.1,
            ];
            let vel: Vec<f32> = vec![0.1, 0.0, 0.0, -0.1, 0.0, 0.0];
            let mut scalars = HashMap::new();
            scalars.insert("PE".to_string(), -10.0 + i as f64);
            scalars.insert("KE".to_string(), 5.0 - i as f64 * 0.1);

            writer
                .append_frame(&TrajectoryFrame {
                    step: i as i64,
                    time: i as f64 * 0.001,
                    positions: Some(&pos),
                    velocities: Some(&vel),
                    forces: None,
                    box_h: None,
                    scalars,
                })
                .unwrap();
        }
        assert_eq!(writer.count_frames(), 10);
        writer.close().unwrap();
        drop(store);

        // Read back
        let store = SimulationStore::open_file(&path).unwrap();
        let traj = store.open_trajectory().unwrap().unwrap();
        assert_eq!(traj.count_frames(), 10);

        // steps
        let steps = traj.read_steps().unwrap();
        assert_eq!(steps.len(), 10);
        assert_eq!(steps[0], 0);
        assert_eq!(steps[9], 9);

        // positions at frame 5
        let pos5 = traj.read_positions(5).unwrap();
        assert_eq!(pos5.len(), 6);
        assert!((pos5[0] - 1.5 as crate::types::F).abs() < 1e-5 as crate::types::F);

        // scalar
        let pe5 = traj.read_scalar("PE", 5).unwrap();
        assert!((pe5 - (-5.0)).abs() < 1e-10);

        // scalar series
        let pe_all = traj.read_scalar_series("PE").unwrap();
        assert_eq!(pe_all.len(), 10);
        assert!((pe_all[0] - (-10.0)).abs() < 1e-10);

        // read_frame overlay
        let sys = store.read_system().unwrap();
        let frame5 = traj.read_frame(5, &sys).unwrap();
        let x = frame5.get("atoms").unwrap().get_float("x").unwrap();
        assert!((x[0] - 1.5 as crate::types::F).abs() < 1e-5 as crate::types::F);
    }

    // -- No trajectory -------------------------------------------------------

    #[test]
    fn test_no_trajectory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sim.zarr");
        let system = make_test_system();

        SimulationStore::create_file(
            &path,
            &system,
            None,
            UnitSystem::Real,
            Provenance::default(),
        )
        .unwrap();

        let store = SimulationStore::open_file(&path).unwrap();
        assert!(store.open_trajectory().unwrap().is_none());
    }

    // -- Provenance / Units --------------------------------------------------

    #[test]
    fn test_provenance_units() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sim.zarr");
        let system = make_test_system();

        let prov = Provenance {
            program: Some("molrs-md".into()),
            version: Some("0.5.0".into()),
            method: Some("NVT Langevin".into()),
            seed: Some(42),
        };

        SimulationStore::create_file(&path, &system, None, UnitSystem::Metal, prov).unwrap();

        let store = SimulationStore::open_file(&path).unwrap();

        let p = store.read_provenance().unwrap();
        assert_eq!(p.program.as_deref(), Some("molrs-md"));
        assert_eq!(p.version.as_deref(), Some("0.5.0"));
        assert_eq!(p.method.as_deref(), Some("NVT Langevin"));
        assert_eq!(p.seed, Some(42));

        assert_eq!(store.read_units().unwrap(), UnitSystem::Metal);
    }

    // -- count_atoms ---------------------------------------------------------

    #[test]
    fn test_count_atoms() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sim.zarr");
        let system = make_test_system();

        SimulationStore::create_file(
            &path,
            &system,
            None,
            UnitSystem::Real,
            Provenance::default(),
        )
        .unwrap();

        let store = SimulationStore::open_file(&path).unwrap();
        assert_eq!(store.count_atoms(), 2);
    }

    // -- MemoryStore ---------------------------------------------------------

    #[test]
    fn test_memory_store() {
        use zarrs::storage::WritableStorageTraits;
        use zarrs::storage::store::MemoryStore;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sim.zarr");
        let system = make_test_system();
        let ff = make_test_ff();

        let mut store = SimulationStore::create_file(
            &path,
            &system,
            Some(&ff),
            UnitSystem::Real,
            Provenance::default(),
        )
        .unwrap();

        let config = TrajectoryConfig {
            positions: true,
            chunk_size: 10,
            ..TrajectoryConfig::default()
        };
        let mut writer = store.create_trajectory(config).unwrap();
        for i in 0..5u64 {
            let pos: Vec<f32> = vec![1.0 + i as f32 * 0.5, 3.0, 5.0, 2.0, 4.0, 6.0];
            writer
                .append_frame(&TrajectoryFrame {
                    step: i as i64,
                    time: i as f64 * 0.01,
                    positions: Some(&pos),
                    velocities: None,
                    forces: None,
                    box_h: None,
                    scalars: HashMap::new(),
                })
                .unwrap();
        }
        writer.close().unwrap();
        drop(store);

        // Load into MemoryStore
        let mem = Arc::new(MemoryStore::new());
        fn visit(dir: &std::path::Path, root: &std::path::Path, store: &MemoryStore) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                let p = entry.path();
                if p.is_dir() {
                    visit(&p, root, store);
                } else {
                    let rel = p.strip_prefix(root).unwrap().to_string_lossy().to_string();
                    let content = std::fs::read(&p).unwrap();
                    let key = zarrs::storage::StoreKey::new(&rel).unwrap();
                    store.set(&key, content.into()).unwrap();
                }
            }
        }
        visit(&path, &path, &mem);

        let sim = SimulationStore::open_store(mem as ReadableWritableListableStorage).unwrap();
        assert_eq!(sim.count_atoms(), 2);

        let sys = sim.read_system().unwrap();
        assert_eq!(sys.get("atoms").unwrap().nrows(), Some(2));

        let ff2 = sim.read_forcefield().unwrap().unwrap();
        assert_eq!(ff2.name, "test_ff");

        let traj = sim.open_trajectory().unwrap().unwrap();
        assert_eq!(traj.count_frames(), 5);
        let pos3 = traj.read_positions(3).unwrap();
        assert!((pos3[0] - 2.5 as crate::types::F).abs() < 1e-5 as crate::types::F);
    }

    // -- Chunk boundary ------------------------------------------------------

    #[test]
    fn test_chunk_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sim.zarr");
        let system = make_test_system();

        let mut store = SimulationStore::create_file(
            &path,
            &system,
            None,
            UnitSystem::Real,
            Provenance::default(),
        )
        .unwrap();

        let config = TrajectoryConfig {
            positions: true,
            chunk_size: 10,
            ..TrajectoryConfig::default()
        };
        let mut writer = store.create_trajectory(config).unwrap();

        for i in 0..25u64 {
            let pos: Vec<f32> = vec![i as f32, 0.0, 0.0, 0.0, 0.0, i as f32];
            writer
                .append_frame(&TrajectoryFrame {
                    step: i as i64,
                    time: i as f64,
                    positions: Some(&pos),
                    velocities: None,
                    forces: None,
                    box_h: None,
                    scalars: HashMap::new(),
                })
                .unwrap();
        }
        writer.close().unwrap();
        drop(store);

        let store = SimulationStore::open_file(&path).unwrap();
        let traj = store.open_trajectory().unwrap().unwrap();

        // At chunk boundary
        let pos9 = traj.read_positions(9).unwrap();
        assert!((pos9[0] - 9.0 as crate::types::F).abs() < 1e-5 as crate::types::F);

        let pos10 = traj.read_positions(10).unwrap();
        assert!((pos10[0] - 10.0 as crate::types::F).abs() < 1e-5 as crate::types::F);

        let pos11 = traj.read_positions(11).unwrap();
        assert!((pos11[0] - 11.0 as crate::types::F).abs() < 1e-5 as crate::types::F);
    }

    // -- Archive tests -------------------------------------------------------

    #[test]
    fn test_archive_list_and_open() {
        use super::super::archive::Archive;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("study.zarr");

        let mut archive = Archive::create_file(&path).unwrap();

        let sys1 = make_test_system();
        let sys2 = {
            let mut f = make_test_system();
            f.meta.insert("title".into(), "system 2".into());
            f
        };

        archive
            .create_simulation(
                "nvt_300k",
                &sys1,
                None,
                UnitSystem::Real,
                Provenance::default(),
            )
            .unwrap();
        archive
            .create_simulation(
                "nvt_400k",
                &sys2,
                None,
                UnitSystem::Metal,
                Provenance::default(),
            )
            .unwrap();

        let archive = Archive::open_file(&path).unwrap();
        let names = archive.list_simulations().unwrap();
        assert_eq!(names, vec!["nvt_300k", "nvt_400k"]);

        let sim1 = archive.open_simulation("nvt_300k").unwrap();
        let s1 = sim1.read_system().unwrap();
        assert_eq!(s1.meta.get("title").unwrap(), "test molecule");

        let sim2 = archive.open_simulation("nvt_400k").unwrap();
        let s2 = sim2.read_system().unwrap();
        assert_eq!(s2.meta.get("title").unwrap(), "system 2");
        assert_eq!(sim2.read_units().unwrap(), UnitSystem::Metal);
    }

    #[test]
    fn test_archive_with_trajectory() {
        use super::super::archive::Archive;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("study.zarr");

        let mut archive = Archive::create_file(&path).unwrap();
        let system = make_test_system();

        let mut sim = archive
            .create_simulation(
                "run1",
                &system,
                None,
                UnitSystem::Real,
                Provenance::default(),
            )
            .unwrap();

        let config = TrajectoryConfig {
            positions: true,
            scalars: vec!["PE".into()],
            chunk_size: 10,
            ..TrajectoryConfig::default()
        };
        let mut writer = sim.create_trajectory(config).unwrap();
        for i in 0..5u64 {
            let pos: Vec<f32> = vec![i as f32, 0.0, 0.0, 0.0, 0.0, 0.0];
            let mut scalars = HashMap::new();
            scalars.insert("PE".into(), -(i as f64));
            writer
                .append_frame(&TrajectoryFrame {
                    step: i as i64,
                    time: i as f64,
                    positions: Some(&pos),
                    velocities: None,
                    forces: None,
                    box_h: None,
                    scalars,
                })
                .unwrap();
        }
        writer.close().unwrap();

        // Re-open archive and read
        let archive = Archive::open_file(&path).unwrap();
        let sim = archive.open_simulation("run1").unwrap();
        let traj = sim.open_trajectory().unwrap().unwrap();
        assert_eq!(traj.count_frames(), 5);

        let pe = traj.read_scalar("PE", 3).unwrap();
        assert!((pe - (-3.0)).abs() < 1e-10);
    }
}
