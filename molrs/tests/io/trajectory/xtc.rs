//! Integration tests for the GROMACS XTC reader and writer.
//!
//! Every file in `tests-data/xtc/` is read and structurally verified per the
//! IO testing rule in `molrs/CLAUDE.md`. XTC is lossily compressed, so value
//! checks use the `1/precision` tolerance. Cross-format checks against the GRO
//! structure and the TRR trajectory anchor decoded coordinates to ground truth.

use molrs::io::reader::TrajReader;
use molrs::io::trajectory::trr::read_trr;
use molrs::io::trajectory::xtc::{open_xtc, read_xtc, write_xtc};
use molrs::store::frame::Frame;
use std::path::{Path, PathBuf};

/// Tolerance for XTC's lossy 1000-precision quantization (nm).
const XTC_TOL: f64 = 2e-3;

fn all_xtc_files() -> Vec<PathBuf> {
    let dir = crate::common::data_path("xtc");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read xtc test-data dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            (p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("xtc"))
                    .unwrap_or(false))
            .then_some(p)
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "No XTC files found in tests-data/xtc/");
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

/// Every XTC file parses to ≥ 1 frame with finite x/y/z. Includes
/// `ubiquitin_faux2023magic.xtc`, whose 2023 magic must be accepted.
#[test]
fn test_all_xtc_files_parse() {
    for path in all_xtc_files() {
        let frames =
            read_xtc(&path).unwrap_or_else(|e| panic!("{}: read failed: {}", file_name(&path), e));
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

/// The faux-2023-magic fixture is the same simulation as `ubiquitin.xtc`; both
/// must decode to identical coordinates.
#[test]
fn test_faux2023magic_matches_classic() {
    let pc = crate::common::require_fixture("xtc/ubiquitin.xtc");
    let pf = crate::common::require_fixture("xtc/ubiquitin_faux2023magic.xtc");
    let fc = read_xtc(&pc).expect("classic");
    let ff = read_xtc(&pf).expect("faux2023magic");
    assert_eq!(fc.len(), ff.len(), "frame count");
    for axis in ["x", "y", "z"] {
        let a = col(&fc[0], axis);
        let b = col(&ff[0], axis);
        assert_eq!(a.len(), b.len());
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-9, "{axis}: {va} vs {vb}");
        }
    }
}

/// `read_step(n)` matches the sequential walk for every fixture.
#[test]
fn test_xtc_random_access_matches_sequential() {
    for path in all_xtc_files() {
        let name = file_name(&path).to_owned();
        let sequential = read_xtc(&path).expect("sequential read");
        let mut reader = open_xtc(&path).expect("open");
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

/// Write → read round-trip preserves frame/atom counts and coordinates to the
/// XTC quantization tolerance.
#[test]
fn test_xtc_writer_roundtrip() {
    use tempfile::NamedTempFile;
    for path in all_xtc_files() {
        let name = file_name(&path).to_owned();
        let frames1 = read_xtc(&path).expect("read original");
        let temp = NamedTempFile::new().expect("temp");
        write_xtc(temp.path(), &frames1).expect("write");
        let frames2 = read_xtc(temp.path()).expect("read round-trip");

        assert_eq!(frames1.len(), frames2.len(), "{}: frame count", name);
        for (i, (a, b)) in frames1.iter().zip(frames2.iter()).enumerate() {
            assert_eq!(
                a.get("atoms").unwrap().nrows(),
                b.get("atoms").unwrap().nrows(),
                "{}: frame {} atom count",
                name,
                i
            );
            for axis in ["x", "y", "z"] {
                let ca = col(a, axis);
                let cb = col(b, axis);
                for (va, vb) in ca.iter().zip(cb.iter()) {
                    assert!(
                        (va - vb).abs() <= XTC_TOL,
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

/// Cross-format ground truth: the same `ubiquitin` simulation in lossless TRR
/// and lossy XTC must agree to the XTC tolerance — validates the compressed
/// decoder against the independent TRR decoder on a real ~20k-atom system.
#[test]
fn test_ubiquitin_trr_matches_xtc() {
    let ptrr = crate::common::require_fixture("trr/ubiquitin.trr");
    let pxtc = crate::common::require_fixture("xtc/ubiquitin.xtc");
    let trr = &read_trr(&ptrr).expect("read trr")[0];
    let xtc = &read_xtc(&pxtc).expect("read xtc")[0];
    let n_trr = trr.get("atoms").unwrap().nrows().unwrap();
    let n_xtc = xtc.get("atoms").unwrap().nrows().unwrap();
    // Same simulation: atom counts must match, else the fixtures diverged.
    assert_eq!(n_trr, n_xtc, "ubiquitin atom count trr vs xtc");
    for axis in ["x", "y", "z"] {
        let ct = col(trr, axis);
        let cx = col(xtc, axis);
        for i in 0..ct.len() {
            assert!(
                (ct[i] - cx[i]).abs() < XTC_TOL,
                "ubiquitin {axis}[{i}] trr {} vs xtc {}",
                ct[i],
                cx[i]
            );
        }
    }
}

/// The `cell_shapes` simulation in TRR (lossless) and XTC (lossy) must agree
/// to the XTC tolerance on the first frame.
#[test]
fn test_cell_shapes_trr_matches_xtc() {
    let ptrr = crate::common::require_fixture("trr/cell_shapes.trr");
    let pxtc = crate::common::require_fixture("xtc/cell_shapes.xtc");
    let trr = &read_trr(&ptrr).expect("read trr")[0];
    let xtc = &read_xtc(&pxtc).expect("read xtc")[0];
    for axis in ["x", "y", "z"] {
        let ct = col(trr, axis);
        let cx = col(xtc, axis);
        assert_eq!(ct.len(), cx.len(), "cell_shapes {axis} length");
        for i in 0..ct.len() {
            assert!(
                (ct[i] - cx[i]).abs() < XTC_TOL,
                "cell_shapes {axis}[{i}] trr {} vs xtc {}",
                ct[i],
                cx[i]
            );
        }
    }
}
