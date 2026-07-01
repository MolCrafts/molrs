//! Multi-frame Gaussian-Cube trajectory reader.
//!
//! AIMD spectroscopy workflows emit one volumetric electron-density cube per
//! step. molrs reads a single cube in [`crate::io::data::cube`]; this adds the
//! trajectory case as TRAVIS consumes it (`src/bqb_cubeframe.cpp` reads a
//! sequence of cube frames from a concatenated stream). Two ingest shapes:
//!
//! - [`read_cube_trajectory`] — one file holding N cubes back-to-back (no blank
//!   separator lines, the conventional concatenation), each parsed by the
//!   single-cube reader.
//! - [`read_cube_trajectory_files`] — an ordered list of single-cube files.
//!
//! Every frame is normalised to Å at the reader boundary by the underlying
//! single-cube parser; the volumetric block stays in its native `e/Bohr³` (the
//! Voronoi integrator converts to `e/Å³`).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use molrs::error::MolRsError;
use molrs::store::frame::Frame;

use crate::io::data::cube::read_cube_from_reader;

/// Read a concatenated multi-cube trajectory file into one [`Frame`] per cube.
///
/// Cubes must be back-to-back with no blank separator lines (the standard way
/// AIMD drivers concatenate per-step cubes). Reading stops cleanly at EOF.
pub fn read_cube_trajectory<P: AsRef<Path>>(path: P) -> Result<Vec<Frame>, MolRsError> {
    let file = File::open(path).map_err(MolRsError::Io)?;
    let mut reader = BufReader::new(file);
    let mut frames = Vec::new();
    loop {
        // EOF when the buffered reader has nothing more to hand out.
        let at_eof = reader.fill_buf().map_err(MolRsError::Io)?.is_empty();
        if at_eof {
            break;
        }
        let frame = read_cube_from_reader(&mut reader)?;
        frames.push(frame);
    }
    Ok(frames)
}

/// Read an ordered sequence of single-cube files into one [`Frame`] each.
pub fn read_cube_trajectory_files<P: AsRef<Path>>(paths: &[P]) -> Result<Vec<Frame>, MolRsError> {
    let mut frames = Vec::with_capacity(paths.len());
    for p in paths {
        let file = File::open(p).map_err(MolRsError::Io)?;
        let reader = BufReader::new(file);
        frames.push(read_cube_from_reader(reader)?);
    }
    Ok(frames)
}
