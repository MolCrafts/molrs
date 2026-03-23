//! Trajectory streaming writer and random-access reader.
//!
//! A trajectory lives under a `{prefix}/trajectory/` Zarr group and stores
//! time-varying per-atom arrays (positions, velocities, forces) plus scalar
//! observables (PE, KE, temperature, …).

use std::collections::HashMap;

use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "filesystem")]
use zarrs::array::{ArrayBuilder, data_type};
#[cfg(feature = "filesystem")]
use zarrs::group::GroupBuilder;
use zarrs::storage::ReadableWritableListableStorage;

use crate::error::MolRsError;
use crate::frame::Frame;
use crate::types::F;

use super::frame_io;

// ---------------------------------------------------------------------------
// Public config / data types
// ---------------------------------------------------------------------------

/// Configuration for trajectory arrays.
#[derive(Clone, Debug)]
pub struct TrajectoryConfig {
    pub positions: bool,
    pub velocities: bool,
    pub forces: bool,
    pub box_h: bool,
    pub scalars: Vec<String>,
    pub chunk_size: u64,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            positions: true,
            velocities: false,
            forces: false,
            box_h: false,
            scalars: Vec::new(),
            chunk_size: 10_000,
        }
    }
}

/// Per-frame data for trajectory append.
pub struct TrajectoryFrame<'a> {
    pub step: i64,
    pub time: f64,
    pub positions: Option<&'a [f32]>,
    pub velocities: Option<&'a [f32]>,
    pub forces: Option<&'a [f32]>,
    pub box_h: Option<&'a [f32]>,
    pub scalars: HashMap<String, f64>,
}

// ---------------------------------------------------------------------------
// TrajectoryWriter
// ---------------------------------------------------------------------------

type ZarrArray = Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>;

/// Streaming trajectory writer.
pub struct TrajectoryWriter {
    n_atoms: u64,
    n_frames: u64,
    step_arr: ZarrArray,
    time_arr: ZarrArray,
    pos_arrs: Option<[ZarrArray; 3]>,
    vel_arrs: Option<[ZarrArray; 3]>,
    frc_arrs: Option<[ZarrArray; 3]>,
    scalar_arrs: HashMap<String, ZarrArray>,
    box_h_arr: Option<ZarrArray>,
}

impl TrajectoryWriter {
    /// Create trajectory arrays under `{prefix}/trajectory/`.
    #[cfg(feature = "filesystem")]
    pub(crate) fn create(
        store: &ReadableWritableListableStorage,
        prefix: &str,
        n_atoms: u64,
        config: &TrajectoryConfig,
    ) -> Result<Self, MolRsError> {
        let traj = super::frame_io::join_path(prefix, "trajectory");
        let c = if config.chunk_size == 0 {
            u32::MAX as u64
        } else {
            config.chunk_size
        };

        GroupBuilder::new()
            .build(store.clone(), &traj)?
            .store_metadata()?;

        // step: [0] i64
        let step_arr = ArrayBuilder::new(vec![0u64], vec![c], data_type::int64(), 0i64)
            .build(store.clone(), &format!("{}/step", traj))?;
        step_arr.store_metadata()?;

        // time: [0] f64
        let time_arr = ArrayBuilder::new(vec![0u64], vec![c], data_type::float64(), 0.0f64)
            .build(store.clone(), &format!("{}/time", traj))?;
        time_arr.store_metadata()?;

        let pos_arrs = if config.positions {
            Some(make_xyz_arrays(store, &traj, ["x", "y", "z"], n_atoms, c)?)
        } else {
            None
        };

        let vel_arrs = if config.velocities {
            Some(make_xyz_arrays(
                store,
                &traj,
                ["vx", "vy", "vz"],
                n_atoms,
                c,
            )?)
        } else {
            None
        };

        let frc_arrs = if config.forces {
            Some(make_xyz_arrays(
                store,
                &traj,
                ["fx", "fy", "fz"],
                n_atoms,
                c,
            )?)
        } else {
            None
        };

        let mut scalar_arrs = HashMap::new();
        for name in &config.scalars {
            let a = ArrayBuilder::new(vec![0u64], vec![c], data_type::float64(), 0.0f64)
                .build(store.clone(), &format!("{}/{}", traj, name))?;
            a.store_metadata()?;
            scalar_arrs.insert(name.clone(), a);
        }

        let box_h_arr = if config.box_h {
            let a = ArrayBuilder::new(vec![0, 3, 3], vec![c, 3, 3], data_type::float32(), 0.0f32)
                .build(store.clone(), &format!("{}/box_h", traj))?;
            a.store_metadata()?;
            Some(a)
        } else {
            None
        };

        Ok(Self {
            n_atoms,
            n_frames: 0,
            step_arr,
            time_arr,
            pos_arrs,
            vel_arrs,
            frc_arrs,
            scalar_arrs,
            box_h_arr,
        })
    }

    /// Append one frame of trajectory data.
    #[allow(clippy::single_range_in_vec_init)]
    pub fn append_frame(&mut self, frame: &TrajectoryFrame) -> Result<(), MolRsError> {
        let t = self.n_frames;
        let n = self.n_atoms;
        let new_t = t + 1;

        // step
        self.step_arr.set_shape(vec![new_t])?;
        self.step_arr.store_metadata()?;
        self.step_arr
            .store_array_subset(&ArraySubset::new_with_ranges(&[t..new_t]), &[frame.step])?;

        // time
        self.time_arr.set_shape(vec![new_t])?;
        self.time_arr.store_metadata()?;
        self.time_arr
            .store_array_subset(&ArraySubset::new_with_ranges(&[t..new_t]), &[frame.time])?;

        // positions
        if let (Some(arrs), Some(flat)) = (&mut self.pos_arrs, frame.positions) {
            append_xyz(arrs, flat, n, t, new_t)?;
        }

        // velocities
        if let (Some(arrs), Some(flat)) = (&mut self.vel_arrs, frame.velocities) {
            append_xyz(arrs, flat, n, t, new_t)?;
        }

        // forces
        if let (Some(arrs), Some(flat)) = (&mut self.frc_arrs, frame.forces) {
            append_xyz(arrs, flat, n, t, new_t)?;
        }

        // scalars
        for (name, arr) in &mut self.scalar_arrs {
            if let Some(&val) = frame.scalars.get(name) {
                arr.set_shape(vec![new_t])?;
                arr.store_metadata()?;
                arr.store_array_subset(&ArraySubset::new_with_ranges(&[t..new_t]), &[val])?;
            }
        }

        // box_h
        if let (Some(arr), Some(bh)) = (&mut self.box_h_arr, frame.box_h) {
            arr.set_shape(vec![new_t, 3, 3])?;
            arr.store_metadata()?;
            arr.store_array_subset(&ArraySubset::new_with_ranges(&[t..new_t, 0..3, 0..3]), bh)?;
        }

        self.n_frames = new_t;
        Ok(())
    }

    /// Number of frames written so far.
    pub fn count_frames(&self) -> u64 {
        self.n_frames
    }

    /// Finalize and flush (currently a no-op, but consumes self).
    pub fn close(self) -> Result<(), MolRsError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TrajectoryReader
// ---------------------------------------------------------------------------

/// Random-access trajectory reader.
pub struct TrajectoryReader {
    store: ReadableWritableListableStorage,
    prefix: String,
    n_atoms: u64,
    n_frames: u64,
}

impl TrajectoryReader {
    /// Open an existing trajectory at `{prefix}/trajectory/`.
    pub(crate) fn open(
        store: ReadableWritableListableStorage,
        prefix: &str,
        n_atoms: u64,
    ) -> Result<Self, MolRsError> {
        let traj = super::frame_io::join_path(prefix, "trajectory");
        let step_arr = Array::open(store.clone(), &format!("{}/step", traj))?;
        let n_frames = step_arr.shape()[0];
        Ok(Self {
            store,
            prefix: traj,
            n_atoms,
            n_frames,
        })
    }

    pub fn count_frames(&self) -> u64 {
        self.n_frames
    }

    /// Read a full Frame at index `t` (system frame with positions overlaid).
    pub fn read_frame(&self, t: u64, system: &Frame) -> Result<Frame, MolRsError> {
        self.check_bounds(t)?;
        let mut frame = system.clone();

        let atoms = frame
            .get_mut("atoms")
            .ok_or_else(|| MolRsError::zarr("system frame missing atoms block"))?;

        // overlay positions
        for key in ["x", "y", "z"] {
            let path = format!("{}/{}", self.prefix, key);
            if frame_io::array_exists(&self.store, &path) {
                let col = frame_io::read_row_f32(&self.store, &path, t, self.n_atoms)?;
                atoms
                    .insert(key, ndarray::Array1::from_vec(col).into_dyn())
                    .map_err(|e| MolRsError::zarr(format!("overlay {}: {}", key, e)))?;
            }
        }

        // overlay velocities
        for key in ["vx", "vy", "vz"] {
            let path = format!("{}/{}", self.prefix, key);
            if frame_io::array_exists(&self.store, &path) {
                let col = frame_io::read_row_f32(&self.store, &path, t, self.n_atoms)?;
                atoms
                    .insert(key, ndarray::Array1::from_vec(col).into_dyn())
                    .map_err(|e| MolRsError::zarr(format!("overlay {}: {}", key, e)))?;
            }
        }

        // overlay box_h
        let box_h_path = format!("{}/box_h", self.prefix);
        if frame_io::array_exists(&self.store, &box_h_path) {
            let arr = Array::open(self.store.clone(), &box_h_path)?;
            let subset = ArraySubset::new_with_ranges(&[t..t + 1, 0..3, 0..3]);
            let h_data: Vec<f32> = arr.retrieve_array_subset(&subset)?;
            let h = ndarray::Array2::from_shape_vec((3, 3), h_data)
                .map_err(|e| MolRsError::zarr(format!("box_h reshape: {}", e)))?
                .mapv(|v| v as F);

            let (origin, pbc) = match &frame.simbox {
                Some(sb) => {
                    let o = sb.origin_view().to_owned();
                    let p = [sb.pbc_view()[0], sb.pbc_view()[1], sb.pbc_view()[2]];
                    (o, p)
                }
                None => (ndarray::arr1(&[0.0, 0.0, 0.0]), [true, true, true]),
            };
            frame.simbox = Some(
                crate::region::simbox::SimBox::new(h, origin, pbc)
                    .map_err(|e| MolRsError::zarr(format!("box_h simbox: {:?}", e)))?,
            );
        }

        Ok(frame)
    }

    /// Read positions at frame `t` as flat `[3N]`.
    pub fn read_positions(&self, t: u64) -> Result<Vec<F>, MolRsError> {
        self.check_bounds(t)?;
        interleave_xyz(&self.store, &self.prefix, ["x", "y", "z"], t, self.n_atoms)
    }

    /// Read velocities at frame `t` as flat `[3N]`.
    pub fn read_velocities(&self, t: u64) -> Result<Vec<F>, MolRsError> {
        self.check_bounds(t)?;
        interleave_xyz(
            &self.store,
            &self.prefix,
            ["vx", "vy", "vz"],
            t,
            self.n_atoms,
        )
    }

    /// Read a named scalar at frame `t`.
    #[allow(clippy::single_range_in_vec_init)]
    pub fn read_scalar(&self, name: &str, t: u64) -> Result<f64, MolRsError> {
        self.check_bounds(t)?;
        let arr = Array::open(self.store.clone(), &format!("{}/{}", self.prefix, name))?;
        let subset = ArraySubset::new_with_ranges(&[t..t + 1]);
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        data.into_iter()
            .next()
            .ok_or_else(|| MolRsError::zarr("empty scalar read"))
    }

    /// Read all step values.
    pub fn read_steps(&self) -> Result<Vec<i64>, MolRsError> {
        read_1d(&self.store, &format!("{}/step", self.prefix))
    }

    /// Read all time values.
    pub fn read_times(&self) -> Result<Vec<f64>, MolRsError> {
        read_1d(&self.store, &format!("{}/time", self.prefix))
    }

    /// Read a scalar for all frames.
    pub fn read_scalar_series(&self, name: &str) -> Result<Vec<f64>, MolRsError> {
        read_1d(&self.store, &format!("{}/{}", self.prefix, name))
    }

    /// Read positions for a range of frames, each as flat `[3N]`.
    pub fn read_positions_range(
        &self,
        range: std::ops::Range<u64>,
    ) -> Result<Vec<Vec<F>>, MolRsError> {
        let mut out = Vec::with_capacity((range.end - range.start) as usize);
        for t in range {
            out.push(self.read_positions(t)?);
        }
        Ok(out)
    }

    fn check_bounds(&self, t: u64) -> Result<(), MolRsError> {
        if t >= self.n_frames {
            return Err(MolRsError::zarr(format!(
                "frame index {} out of range [0, {})",
                t, self.n_frames
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
fn make_xyz_arrays(
    store: &ReadableWritableListableStorage,
    traj_prefix: &str,
    names: [&str; 3],
    n_atoms: u64,
    chunk_size: u64,
) -> Result<[ZarrArray; 3], MolRsError> {
    let mut out = Vec::with_capacity(3);
    for name in names {
        let a = ArrayBuilder::new(
            vec![0, n_atoms],
            vec![chunk_size, n_atoms],
            data_type::float32(),
            0.0f32,
        )
        .build(store.clone(), &format!("{}/{}", traj_prefix, name))?;
        a.store_metadata()?;
        out.push(a);
    }
    Ok([out.remove(0), out.remove(0), out.remove(0)])
}

fn append_xyz(
    arrs: &mut [ZarrArray; 3],
    flat: &[f32],
    n: u64,
    t: u64,
    new_t: u64,
) -> Result<(), MolRsError> {
    let nu = n as usize;
    let mut bufs = [
        Vec::with_capacity(nu),
        Vec::with_capacity(nu),
        Vec::with_capacity(nu),
    ];
    for i in 0..nu {
        bufs[0].push(flat[3 * i]);
        bufs[1].push(flat[3 * i + 1]);
        bufs[2].push(flat[3 * i + 2]);
    }
    for (arr, buf) in arrs.iter_mut().zip(bufs.iter()) {
        arr.set_shape(vec![new_t, n])?;
        arr.store_metadata()?;
        arr.store_array_subset(&ArraySubset::new_with_ranges(&[t..new_t, 0..n]), buf)?;
    }
    Ok(())
}

fn interleave_xyz(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    names: [&str; 3],
    t: u64,
    n_atoms: u64,
) -> Result<Vec<F>, MolRsError> {
    let xs = frame_io::read_row_f32(store, &format!("{}/{}", prefix, names[0]), t, n_atoms)?;
    let ys = frame_io::read_row_f32(store, &format!("{}/{}", prefix, names[1]), t, n_atoms)?;
    let zs = frame_io::read_row_f32(store, &format!("{}/{}", prefix, names[2]), t, n_atoms)?;
    let mut out = Vec::with_capacity(3 * n_atoms as usize);
    for i in 0..n_atoms as usize {
        out.push(xs[i]);
        out.push(ys[i]);
        out.push(zs[i]);
    }
    Ok(out)
}

fn read_1d<T: zarrs::array::ElementOwned>(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<Vec<T>, MolRsError> {
    let arr = Array::open(store.clone(), path)?;
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let data: Vec<T> = arr.retrieve_array_subset(&subset)?;
    Ok(data)
}
