//! Pack a closed directory store into a single stored-entry .zarr.zip.
//!
//! `pack` takes the path of a **closed** store — a directory nobody is
//! appending to any more — and leaves one file where the directory was. The
//! name is derived, never chosen: the sibling is the store path with
//! `.zarr.zip` appended, except that a path already ending in `.zarr` only
//! gains `.zip`, so `traj.zarr` packs to `traj.zarr.zip` and never to
//! `traj.zarr.zarr.zip`. Every entry is written STORED (method 0): the chunks
//! arrive already gzipped, so packing is concatenation plus a central
//! directory.
//!
//! The parameter is a path and nothing else. There is no door that accepts a
//! live `FrameSequenceWriter` — packing a store still being appended to would
//! be the live single-file write this design rejects, and the signature is the
//! only proof that needs to exist. `close(writer)` then `pack(path)` is the
//! caller's composition.
//!
//! Reading is the mirror door `open_packed`, which opens the archive through
//! `zarrs_zip`'s read-only `ZipStorageAdapter` and hands back a store to give
//! to `FrameSequence::open` (or any other read door). No hand-written zip
//! parser lives here.

use std::ffi::OsStr;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use zarrs::filesystem::FilesystemStore;
use zarrs::storage::{ReadableListableStorage, StoreKey};

use molrs::MolRsError;

use super::record_io::zerr;

/// The suffix a packed store carries.
const ZIP_SUFFIX: &str = ".zip";
/// The suffix a directory store is expected to carry already.
const ZARR_SUFFIX: &str = "zarr";

/// Pack the closed directory store at `store_path` into a sibling
/// `.zarr.zip` and remove the directory, returning the path of the archive.
///
/// The archive's name is derived from the directory's: `traj.zarr` becomes
/// `traj.zarr.zip`, and a directory without the `.zarr` suffix gains the whole
/// `.zarr.zip`. Entries are walked in sorted order and written STORED
/// (method 0) — the chunks arrived gzipped, so a second compression pass would
/// buy nothing and cost a full re-encode.
///
/// The parameter is a path, never a live [`FrameSequenceWriter`]: packing a
/// store still being appended to is the live single-file write this backend
/// rejects. `writer.close()` followed by `pack(path)` is the caller's
/// composition.
///
/// # Errors
///
/// Returns a [`MolRsError::Zarr`] naming `store_path` when no directory is
/// there — which is also what a second `pack` of the same store meets, since
/// the first one removed the directory. Nothing is created in that case, so a
/// refused pack leaves no half-written archive behind.
///
/// Every other failure is also a [`MolRsError::Zarr`], and it names the path it
/// happened on, since neither `std::fs` nor the zip writer puts the path in its
/// own message. `store_path` naming no directory component at all fails before
/// anything is created. After that the archive is open, and a failure while
/// walking the directory, reading an entry, encoding an entry's name as a store
/// key (its path must be valid UTF-8) or writing into the archive **leaves a
/// partial `.zarr.zip` beside the still-intact directory store**: the directory
/// is removed only once the archive is closed, so the run's data always exists
/// in at least one of the two places, but the caller deletes the partial
/// archive before retrying. The removal itself can fail last of all, and then
/// both the finished archive and the directory survive, the directory being the
/// redundant copy.
///
/// [`FrameSequenceWriter`]: super::FrameSequenceWriter
pub fn pack(store_path: impl AsRef<Path>) -> Result<PathBuf, MolRsError> {
    let store_path = store_path.as_ref();
    if !store_path.is_dir() {
        return Err(at(store_path, "pack", "no directory store is there"));
    }
    let zip_path = packed_path(store_path)?;

    let file = std::fs::File::create(&zip_path).map_err(|e| at(store_path, "pack", e))?;
    let mut writer = zip::ZipWriter::new(file);
    for entry in sorted_files(store_path).map_err(|e| at(store_path, "pack", e))? {
        let name = entry
            .strip_prefix(store_path)
            .map_err(zerr)?
            .to_str()
            .ok_or_else(|| {
                at(
                    store_path,
                    "pack",
                    format!("{} is not a valid store key", entry.display()),
                )
            })?
            .to_string();
        let bytes = std::fs::read(&entry).map_err(|e| at(&entry, "pack", e))?;
        // `large_file` writes the zip64 extra field, which is required — and
        // only legal to omit — below 4 GiB. Sized per entry rather than set
        // once, so an ordinary store keeps the plainest possible archive.
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(bytes.len() as u64 >= u64::from(u32::MAX));
        writer
            .start_file(name, options)
            .map_err(|e| at(&entry, "pack", e))?;
        writer
            .write_all(&bytes)
            .map_err(|e| at(&entry, "pack", e))?;
    }
    writer.finish().map_err(|e| at(store_path, "pack", e))?;

    // The directory goes last: until the archive is closed, the store the
    // caller handed over is still the only copy of its data.
    std::fs::remove_dir_all(store_path).map_err(|e| at(store_path, "pack", e))?;
    Ok(zip_path)
}

/// Open a packed `.zarr.zip` read-only, through `zarrs_zip`'s
/// `ZipStorageAdapter`.
///
/// The returned store is readable and listable and nothing more, which is
/// exactly what a read door such as [`FrameSequence::open`] asks for. Stored
/// entries are read through the adapter's byte-range fast path, so a frame
/// costs the bytes of that frame rather than the bytes of the archive.
///
/// # Errors
///
/// Returns a [`MolRsError::Zarr`] if `path` has no file name, if its directory
/// cannot be opened, or if the file is not a readable zip archive.
///
/// [`FrameSequence::open`]: super::FrameSequence::open
pub fn open_packed(path: impl AsRef<Path>) -> Result<ReadableListableStorage, MolRsError> {
    let path = path.as_ref();
    let name = path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| at(path, "open", "it names no file"))?;
    // The adapter reads the archive as one key of the directory holding it.
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let directory = Arc::new(FilesystemStore::new(parent).map_err(|e| at(path, "open", e))?);
    let key = StoreKey::new(name).map_err(|e| at(path, "open", e))?;
    Ok(Arc::new(
        zarrs_zip::ZipStorageAdapter::new(directory, key).map_err(|e| at(path, "open", e))?,
    ))
}

/// An error naming the path it happened on.
///
/// Neither `std::fs` nor `zarrs_zip` puts the path in its message, and a door
/// whose whole argument is a path must not report a bare "permission denied".
fn at(path: &Path, verb: &str, cause: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(format!("cannot {verb} {}: {cause}", path.display()))
}

/// The archive name derived from a directory store's: `.zarr` gains only
/// `.zip`, anything else gains the whole `.zarr.zip`.
fn packed_path(store_path: &Path) -> Result<PathBuf, MolRsError> {
    let name = store_path
        .file_name()
        .ok_or_else(|| at(store_path, "pack", "it names no directory"))?;
    let mut packed = name.to_os_string();
    if Path::new(name).extension() == Some(OsStr::new(ZARR_SUFFIX)) {
        packed.push(ZIP_SUFFIX);
    } else {
        packed.push(format!(".{ZARR_SUFFIX}{ZIP_SUFFIX}"));
    }
    Ok(store_path.with_file_name(packed))
}

/// Every file under `root`, sorted by path.
///
/// The walk order is a stack's and therefore arbitrary; the sort at the end is
/// what makes it deterministic, so two packs of the same store lay their
/// entries down in the same order.
fn sorted_files(root: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut files = Vec::new();
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(&directory)? {
            let path = entry?.path();
            if path.is_dir() {
                directories.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use std::path::{Path, PathBuf};

    use molrs::store::block::{Block, Column};
    use molrs::store::frame::Frame;
    use molrs::store::trajectory::Trajectory;
    use ndarray::ArrayD;
    use tempfile::tempdir;

    use super::{open_packed, pack};
    use crate::io::zarr::{FrameSequence, read_trajectory_file, write_trajectory_file};

    /// The one block every fixture frame carries.
    const ATOMS: &str = "atoms";
    /// Its one `f64` column.
    const X: &str = "x";

    fn float_column(values: &[f64]) -> Column {
        Column::from_float(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn atoms_frame(values: &[f64]) -> Frame {
        let mut block = Block::new();
        block.insert_column(X, float_column(values)).unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, block);
        frame
    }

    /// Three frames of 3, 5 and 4 rows: ragged, so the CSR row pointer has to
    /// survive the pack, with every value distinct across frames so a row range
    /// taken from the wrong frame cannot match by accident. Step numbers and
    /// times of its own, so both index arrays are exercised too.
    fn trajectory() -> Trajectory {
        Trajectory {
            frames: vec![
                atoms_frame(&[1.0, -2.5, 1.0e-300]),
                atoms_frame(&[0.1, 0.2, 0.3, 4.5, -6.25]),
                atoms_frame(&[f64::MIN_POSITIVE, 2.0e300, 3.5, -0.125]),
            ],
            step: Some(vec![0, 5, 10]),
            time: Some(vec![0.0, 0.5, 1.25]),
        }
    }

    /// Column `x` of the `atoms` block as raw bits.
    ///
    /// Bits, not values: the trajectory dtype contract admits no tolerance, and
    /// comparing `f64` by value would let a sign-of-zero flip through.
    fn atoms_x_bits(frame: &Frame) -> Vec<u64> {
        frame
            .get(ATOMS)
            .expect("the frame carries an atoms block")
            .get(X)
            .expect("the atoms block carries column x")
            .as_float()
            .expect("column x arrived as f64")
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    /// Write [`trajectory`] into a fresh directory store `name` under `parent`.
    fn write_store(parent: &Path, name: &str) -> PathBuf {
        let path = parent.join(name);
        write_trajectory_file(&path, &trajectory()).expect("the fixture store writes");
        path
    }

    /// The pinned name: `<dir>.zarr` packs to `<dir>.zarr.zip`, one file, and
    /// the directory store is gone.
    #[test]
    fn pack_produces_one_zip_and_removes_the_directory() {
        let dir = tempdir().unwrap();
        let store_path = write_store(dir.path(), "traj.zarr");

        let zip_path = pack(&store_path).expect("packing a closed store succeeds");

        assert_eq!(
            zip_path,
            dir.path().join("traj.zarr.zip"),
            "a path already ending in .zarr gains only .zip"
        );
        assert!(zip_path.is_file(), "the packed store is a single file");
        assert!(
            !store_path.exists(),
            "pack removes the directory store it consumed"
        );
        let left: Vec<PathBuf> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(
            left,
            vec![zip_path],
            "the zip is the only thing left at rest"
        );
    }

    /// A directory with no `.zarr` suffix gains the whole suffix.
    #[test]
    fn packing_a_directory_without_the_zarr_suffix_appends_the_whole_suffix() {
        let dir = tempdir().unwrap();
        let store_path = write_store(dir.path(), "traj");

        let zip_path = pack(&store_path).expect("packing a closed store succeeds");

        assert_eq!(zip_path, dir.path().join("traj.zarr.zip"));
    }

    /// Every entry is STORED (method 0) — the chunks are already gzipped, so
    /// packing must not spend a second compression pass on them.
    #[test]
    fn every_zip_entry_is_stored_method_zero() {
        let dir = tempdir().unwrap();
        let store_path = write_store(dir.path(), "traj.zarr");
        let zip_path = pack(&store_path).expect("packing a closed store succeeds");

        let mut archive =
            zip::ZipArchive::new(std::fs::File::open(&zip_path).unwrap()).expect("a readable zip");

        assert!(!archive.is_empty(), "a packed store carries entries");
        for index in 0..archive.len() {
            let entry = archive.by_index(index).unwrap();
            assert_eq!(
                entry.compression(),
                zip::CompressionMethod::Stored,
                "{} is not a stored entry",
                entry.name()
            );
        }
    }

    /// Every frame, step and time read out of the packed zip is bit identical
    /// to what the directory store gave before packing.
    #[test]
    fn frames_read_back_from_the_zip_bit_exact() {
        let dir = tempdir().unwrap();
        let store_path = write_store(dir.path(), "traj.zarr");
        // The reference is taken from the directory form, before it is gone.
        let before = read_trajectory_file(&store_path).expect("the directory store reads");

        let zip_path = pack(&store_path).expect("packing a closed store succeeds");
        let mut sequence = FrameSequence::open(open_packed(&zip_path).expect("the zip opens"))
            .expect("the packed sequence opens");
        let after = sequence.to_trajectory().expect("the packed sequence reads");

        assert_eq!(after.step, before.step, "step numbers survive the pack");
        assert_eq!(after.time, before.time, "times survive the pack");
        assert_eq!(
            after.frames.len(),
            before.frames.len(),
            "frame count survives the pack"
        );
        for (index, (packed, directory)) in after.frames.iter().zip(&before.frames).enumerate() {
            assert_eq!(
                atoms_x_bits(packed),
                atoms_x_bits(directory),
                "frame {index} column x came back with different bits"
            );
        }
    }

    /// Packing a path that is not there is an error that names the path.
    #[test]
    fn packing_a_missing_directory_errs_naming_it() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("absent.zarr");

        let error = pack(&missing).expect_err("a missing store cannot be packed");

        let message = error.to_string();
        assert!(
            message.contains(&missing.display().to_string()),
            "the error must name the path it could not pack: {message}"
        );
        assert!(
            !dir.path().join("absent.zarr.zip").exists(),
            "a failed pack leaves no half-written zip behind"
        );
    }

    /// Packing twice is not idempotent and does not pretend to be: the second
    /// call meets the directory `pack` itself removed and errs naming it, the
    /// same way any other missing path does.
    #[test]
    fn packing_twice_errs_naming_the_removed_directory() {
        let dir = tempdir().unwrap();
        let store_path = write_store(dir.path(), "traj.zarr");
        let zip_path = pack(&store_path).expect("the first pack succeeds");

        let error = pack(&store_path).expect_err("the directory is gone after the first pack");

        let message = error.to_string();
        assert!(
            message.contains(&store_path.display().to_string()),
            "the error must name the path it could not pack: {message}"
        );
        assert!(
            zip_path.is_file(),
            "the failed second pack must not disturb the first pack's zip"
        );
    }
}
