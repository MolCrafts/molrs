//! Integration tests for the GROMACS TRR reader and writer.
//!
//! Every file in `tests-data/trr/` is read and structurally verified per the
//! IO testing rule in `molrs/CLAUDE.md`. Cross-format and single-vs-double
//! consistency checks anchor the decoded coordinates to ground truth.

use molrs::io::reader::TrajReader;
use molrs::io::trajectory::trr::{open_trr, read_trr, write_trr};
use molrs::store::frame::Frame;
use std::path::{Path, PathBuf};

fn all_trr_files() -> Vec<PathBuf> {
    let dir = crate::common::data_path("trr");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read trr test-data dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            (p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("trr"))
                    .unwrap_or(false))
            .then_some(p)
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "No TRR files found in tests-data/trr/");
    paths
}

fn file_name(p: &Path) -> &str {
    p.file_name().unwrap().to_str().unwrap()
}

fn col(frame: &Frame, axis: &str) -> Vec<f64> {
    frame
        .get("atoms")
        .unwrap()
        .get_float(axis)
        .unwrap()
        .iter()
        .copied()
        .collect()
}

/// Every TRR file parses to ≥ 1 frame; each frame has finite x/y/z columns.
#[test]
fn test_all_trr_files_parse() {
    for path in all_trr_files() {
        let frames =
            read_trr(&path).unwrap_or_else(|e| panic!("{}: read failed: {}", file_name(&path), e));
        assert!(!frames.is_empty(), "{}: zero frames", file_name(&path));
        for (i, frame) in frames.iter().enumerate() {
            let atoms = frame
                .get("atoms")
                .unwrap_or_else(|| panic!("{}: frame {} missing atoms", file_name(&path), i));
            assert!(
                atoms.nrows().unwrap_or(0) > 0,
                "{}: frame {} empty",
                file_name(&path),
                i
            );
            for axis in ["x", "y", "z"] {
                let c = atoms
                    .get_float(axis)
                    .unwrap_or_else(|| panic!("{}: missing {}", file_name(&path), axis));
                assert!(
                    c.iter().all(|v| v.is_finite()),
                    "{}: non-finite {}",
                    file_name(&path),
                    axis
                );
            }
        }
    }
}

/// `read_step(n)` (random access) returns the same coordinates as the
/// sequential `read_trr` walk for every fixture.
#[test]
fn test_trr_random_access_matches_sequential() {
    for path in all_trr_files() {
        let name = file_name(&path).to_owned();
        let sequential = read_trr(&path).expect("sequential read");
        let mut reader = open_trr(&path).expect("open");
        let n_total = reader.len().unwrap();
        assert_eq!(n_total, sequential.len(), "{}: frame count", name);

        let probes: Vec<usize> = if n_total <= 3 {
            (0..n_total).collect()
        } else {
            vec![0, n_total / 2, n_total - 1]
        };
        for &n in &probes {
            let frame = reader.read_step(n).unwrap().unwrap();
            let xs_a = col(&frame, "x");
            let xs_b = col(&sequential[n], "x");
            assert_eq!(xs_a.len(), xs_b.len(), "{}: frame {} length", name, n);
            for (a, b) in xs_a.iter().zip(xs_b.iter()) {
                assert!((a - b).abs() < 1e-9, "{}: frame {} x mismatch", name, n);
            }
        }
    }
}

/// Single- and double-precision encodings of the same simulation must agree
/// to f32 precision — proves the precision auto-detection is correct.
#[test]
fn test_trr_single_vs_double_precision_agree() {
    for (single, double) in [
        ("ubiquitin.trr", "ubiquitin_d.trr"),
        ("cell_shapes.trr", "cell_shapes_d.trr"),
    ] {
        let ps = crate::common::require_fixture(&format!("trr/{single}"));
        let pd = crate::common::require_fixture(&format!("trr/{double}"));
        let fs = read_trr(&ps).expect("read single");
        let fd = read_trr(&pd).expect("read double");
        assert_eq!(fs.len(), fd.len(), "{single} vs {double}: frame count");
        for (i, (a, b)) in fs.iter().zip(fd.iter()).enumerate() {
            for axis in ["x", "y", "z"] {
                let ca = col(a, axis);
                let cb = col(b, axis);
                assert_eq!(ca.len(), cb.len(), "{single}: frame {i} {axis} length");
                for (va, vb) in ca.iter().zip(cb.iter()) {
                    assert!(
                        (va - vb).abs() < 1e-4,
                        "{single} vs {double}: frame {i} {axis} {va} vs {vb}"
                    );
                }
            }
        }
    }
}

/// Write → read round-trip preserves frame count, atom count, and coordinates
/// to single-precision tolerance.
#[test]
fn test_trr_writer_roundtrip() {
    use tempfile::NamedTempFile;
    for path in all_trr_files() {
        let name = file_name(&path).to_owned();
        let frames1 = read_trr(&path).expect("read original");
        let temp = NamedTempFile::new().expect("temp");
        write_trr(temp.path(), &frames1).expect("write");
        let frames2 = read_trr(temp.path()).expect("read round-trip");

        assert_eq!(frames1.len(), frames2.len(), "{}: frame count", name);
        for (i, (a, b)) in frames1.iter().zip(frames2.iter()).enumerate() {
            for axis in ["x", "y", "z"] {
                let ca = col(a, axis);
                let cb = col(b, axis);
                assert_eq!(ca.len(), cb.len(), "{}: frame {} {} length", name, i, axis);
                for (va, vb) in ca.iter().zip(cb.iter()) {
                    let scale = va.abs().max(1.0);
                    assert!(
                        (va - vb).abs() <= 1e-3 * scale,
                        "{}: frame {} {} {} vs {}",
                        name,
                        i,
                        axis,
                        va,
                        vb
                    );
                }
            }
        }
    }
}

/// Box round-trips through the writer: a fixture with a box keeps its lattice
/// lengths within single-precision tolerance after write → read.
#[test]
fn test_trr_box_roundtrip() {
    use tempfile::NamedTempFile;
    for path in all_trr_files() {
        let name = file_name(&path).to_owned();
        let frames1 = read_trr(&path).expect("read");
        let Some(box1) = frames1[0].simbox.as_ref().map(|b| b.lengths().to_vec()) else {
            continue;
        };
        let temp = NamedTempFile::new().expect("temp");
        write_trr(temp.path(), &frames1).expect("write");
        let frames2 = read_trr(temp.path()).expect("read round-trip");
        let box2 = frames2[0]
            .simbox
            .as_ref()
            .expect("box preserved")
            .lengths()
            .to_vec();
        for i in 0..3 {
            assert!(
                (box1[i] - box2[i]).abs() < 1e-3 * box1[i].abs().max(1.0),
                "{}: box length {} {} vs {}",
                name,
                i,
                box1[i],
                box2[i]
            );
        }
    }
}
