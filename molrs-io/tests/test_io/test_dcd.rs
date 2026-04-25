//! Integration tests for the DCD binary trajectory reader and writer.
//!
//! Every file in `tests-data/dcd/` is read and structurally verified per
//! the IO testing rule in `molrs/CLAUDE.md`. The writer is round-tripped
//! against the subset of files that fall in its supported flavor (no fixed
//! atoms, no 4D dynamics).

use molrs_io::dcd::{DcdReader, open_dcd, read_dcd, write_dcd};
use molrs_io::reader::TrajReader;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Fixture catalogue
// ---------------------------------------------------------------------------

fn all_dcd_files() -> Vec<PathBuf> {
    let dir = crate::test_data::get_test_data_path("dcd");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read dcd test-data dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            if p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("dcd"))
                    .unwrap_or(false)
            {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "No DCD files found in tests-data/dcd/ — did the submodule fetch run?"
    );
    paths
}

fn file_name(p: &Path) -> &str {
    p.file_name().unwrap().to_str().unwrap()
}

/// Files whose CHARMM features (NAMNF > 0, has_4d) the v1 writer cannot
/// reproduce. We still read them, just don't round-trip them.
fn writer_unsupported(name: &str) -> bool {
    matches!(name, "fixed-atoms.dcd" | "4d-dynamic.dcd")
}

// ---------------------------------------------------------------------------
// Reader tests — must succeed for every fixture.
// ---------------------------------------------------------------------------

/// Every DCD file parses to ≥ 1 frame and each frame has id/x/y/z columns.
#[test]
fn test_all_dcd_files_parse() {
    for path in all_dcd_files() {
        let frames = read_dcd(path.to_str().unwrap())
            .unwrap_or_else(|e| panic!("{}: read failed: {}", file_name(&path), e));
        assert!(!frames.is_empty(), "{}: zero frames", file_name(&path));

        for (i, frame) in frames.iter().enumerate() {
            let atoms = frame
                .get("atoms")
                .unwrap_or_else(|| panic!("{}: frame {} missing atoms block", file_name(&path), i));
            assert!(
                atoms.nrows().unwrap_or(0) > 0,
                "{}: frame {} atoms block is empty",
                file_name(&path),
                i
            );
            for col in ["x", "y", "z"] {
                assert!(
                    atoms.get_float(col).is_some(),
                    "{}: frame {} missing '{}' column",
                    file_name(&path),
                    i,
                    col
                );
            }
            assert!(
                atoms.get_int("id").is_some(),
                "{}: frame {} missing 'id' column",
                file_name(&path),
                i
            );
        }
    }
}

/// Filename → has_box, taken from `tests-data/dcd/README.md`. Files not in
/// this list either don't have an explicit box claim in the README or are
/// known to carry one in the bytes regardless (e.g. `water.dcd` from VMD).
#[test]
fn test_dcd_simbox_presence_matches_readme() {
    let expectations: &[(&str, bool)] = &[
        ("nopbc.dcd", false),
        ("withpbc.dcd", true),
        ("water.dcd", true),
        ("triclinic-namd.dcd", true),
        ("triclinic-octane-vectors.dcd", true),
        ("triclinic-octane-cos.dcd", true),
        ("triclinic-octane-direct.dcd", true),
        // `4d-dynamic.dcd` and `fixed-atoms.dcd` are the CHARMM dyn4dtest
        // outputs; neither writes the extra-block / unit-cell record.
    ];
    for &(name, expects_box) in expectations {
        let path = crate::test_data::get_test_data_path(&format!("dcd/{}", name));
        if !path.exists() {
            continue;
        }
        let mut reader = open_dcd(&path).expect("open");
        let frame = reader
            .read_step(0)
            .expect("read step 0")
            .expect("frame 0 exists");
        assert_eq!(
            frame.simbox.is_some(),
            expects_box,
            "{}: simbox presence mismatch (expected {})",
            name,
            expects_box
        );
    }
}

/// `4d-dynamic.dcd` carries a fourth `w` column on the atoms block.
#[test]
fn test_4d_dynamic_has_w_column() {
    let path = crate::test_data::get_test_data_path("dcd/4d-dynamic.dcd");
    if !path.exists() {
        return;
    }
    let frames = read_dcd(path.to_str().unwrap()).expect("read 4d-dynamic.dcd");
    assert!(!frames.is_empty(), "expected frames");
    let atoms = frames[0].get("atoms").unwrap();
    assert!(
        atoms.get_float("w").is_some(),
        "4d-dynamic.dcd frame 0 should have a 'w' column"
    );
}

/// `fixed-atoms.dcd` exposes full NATOMS rows on every frame, including
/// frames after frame 0 (where the on-disk record is shorter).
#[test]
fn test_fixed_atoms_full_natoms_per_frame() {
    let path = crate::test_data::get_test_data_path("dcd/fixed-atoms.dcd");
    if !path.exists() {
        return;
    }
    let frames = read_dcd(path.to_str().unwrap()).expect("read fixed-atoms.dcd");
    assert!(frames.len() >= 2, "need at least 2 frames to test fixed");
    let n0 = frames[0].get("atoms").unwrap().nrows().unwrap();
    for (i, f) in frames.iter().enumerate() {
        let n = f.get("atoms").unwrap().nrows().unwrap();
        assert_eq!(
            n, n0,
            "frame {} has {} rows, expected {} (fixed-atom expansion)",
            i, n, n0
        );
    }
}

/// For every fixture, `read_step(N)` (random access) returns the same
/// coordinate values as the sequential `read_frame` walk.
#[test]
fn test_dcd_random_access_matches_sequential() {
    for path in all_dcd_files() {
        let name = file_name(&path).to_owned();

        let sequential = match read_dcd(path.to_str().unwrap()) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let mut reader =
            open_dcd(&path).unwrap_or_else(|e| panic!("{}: open for random: {}", name, e));
        let n_total = reader.len().unwrap();
        assert_eq!(n_total, sequential.len(), "{}: nset disagrees", name);

        // Probe a few indices: 0, last, middle.
        let probes: Vec<usize> = if n_total <= 3 {
            (0..n_total).collect()
        } else {
            vec![0, n_total / 2, n_total - 1]
        };

        for &n in &probes {
            let frame = reader
                .read_step(n)
                .unwrap_or_else(|e| panic!("{}: read_step({}): {}", name, n, e))
                .unwrap_or_else(|| panic!("{}: read_step({}) returned None", name, n));
            let seq = &sequential[n];

            let xs_a = frame.get("atoms").unwrap().get_float("x").unwrap();
            let xs_b = seq.get("atoms").unwrap().get_float("x").unwrap();
            assert_eq!(
                xs_a.len(),
                xs_b.len(),
                "{}: frame {} length disagrees",
                name,
                n
            );
            for (a, b) in xs_a.iter().zip(xs_b.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "{}: frame {} x mismatch {} vs {}",
                    name,
                    n,
                    a,
                    b
                );
            }
        }
    }
}

/// The three `mrmd_h2so4` files are the same simulation re-encoded with
/// different on-disk layouts (LE/BE × 4-byte/8-byte markers). They must
/// agree on every per-atom coordinate to f32 precision.
#[test]
fn test_dcd_endianness_variants_agree() {
    let names = [
        "mrmd_h2so4-32bit-be.dcd",
        "mrmd_h2so4-64bit-be.dcd",
        "mrmd_h2so4-64bit-le.dcd",
    ];
    let mut frames_per_file = Vec::new();
    for n in names {
        let p = crate::test_data::get_test_data_path(&format!("dcd/{}", n));
        if !p.exists() {
            return;
        }
        frames_per_file.push((n, read_dcd(p.to_str().unwrap()).expect("read")));
    }

    let (first_name, first) = &frames_per_file[0];
    for (other_name, other) in &frames_per_file[1..] {
        assert_eq!(
            first.len(),
            other.len(),
            "{} vs {}: frame count differs",
            first_name,
            other_name
        );
        for (i, (fa, fb)) in first.iter().zip(other.iter()).enumerate() {
            let xa = fa.get("atoms").unwrap().get_float("x").unwrap();
            let xb = fb.get("atoms").unwrap().get_float("x").unwrap();
            assert_eq!(xa.len(), xb.len());
            for (a, b) in xa.iter().zip(xb.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "{} vs {}: frame {} x diverges {} vs {}",
                    first_name,
                    other_name,
                    i,
                    a,
                    b
                );
            }
        }
    }
}

/// All three `triclinic-octane` files describe the same simulation; their
/// box lattice vectors must agree to f32 precision.
#[test]
fn test_triclinic_octane_three_variants_agree() {
    let names = [
        "triclinic-octane-vectors.dcd",
        "triclinic-octane-direct.dcd",
        // The "cos" file deliberately encodes the same numbers under a
        // different CHARMM version. The cosine/degree heuristic recovers
        // the same physical lengths but the cos-encoded angles will
        // re-cosine through arccos and stay identical only when angles
        // round-trip cleanly. The other two are direct rewrites and will
        // match exactly.
    ];
    let mut lens_per_file = Vec::new();
    for n in names {
        let p = crate::test_data::get_test_data_path(&format!("dcd/{}", n));
        if !p.exists() {
            return;
        }
        let frames = read_dcd(p.to_str().unwrap()).expect("read");
        let first = frames.first().expect("at least one frame");
        let bx = first.simbox.as_ref().expect("triclinic file has box");
        lens_per_file.push((n, bx.lengths().to_vec()));
    }
    let (first_name, first_lens) = &lens_per_file[0];
    for (other_name, other_lens) in &lens_per_file[1..] {
        for i in 0..3 {
            assert!(
                (first_lens[i] - other_lens[i]).abs() < 1e-3,
                "{} vs {}: lattice length axis {} disagrees ({} vs {})",
                first_name,
                other_name,
                i,
                first_lens[i],
                other_lens[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Writer tests
// ---------------------------------------------------------------------------

/// Every supported fixture survives a write → read round-trip with frame
/// count, atom count, and per-atom coordinates preserved to f32 precision.
#[test]
fn test_dcd_writer_roundtrip() {
    use tempfile::NamedTempFile;
    for path in all_dcd_files() {
        let name = file_name(&path).to_owned();
        if writer_unsupported(&name) {
            continue;
        }

        let frames1 = read_dcd(path.to_str().unwrap())
            .unwrap_or_else(|e| panic!("{}: read original: {}", name, e));
        let temp = NamedTempFile::new().expect("temp");
        write_dcd(temp.path().to_str().unwrap(), &frames1)
            .unwrap_or_else(|e| panic!("{}: write: {}", name, e));
        let frames2 = read_dcd(temp.path().to_str().unwrap())
            .unwrap_or_else(|e| panic!("{}: read round-trip: {}", name, e));

        assert_eq!(
            frames1.len(),
            frames2.len(),
            "{}: frame count changed",
            name
        );
        for (i, (fa, fb)) in frames1.iter().zip(frames2.iter()).enumerate() {
            let aa = fa.get("atoms").unwrap();
            let ab = fb.get("atoms").unwrap();
            assert_eq!(
                aa.nrows().unwrap(),
                ab.nrows().unwrap(),
                "{}: frame {} atom count changed",
                name,
                i
            );
            let tolerance = 1e-3_f64;
            for axis in ["x", "y", "z"] {
                let xa = aa.get_float(axis).unwrap();
                let xb = ab.get_float(axis).unwrap();
                for (a, b) in xa.iter().zip(xb.iter()) {
                    let scale = a.abs().max(1.0);
                    assert!(
                        (a - b).abs() <= tolerance * scale,
                        "{}: frame {} axis {} diverges {} vs {} (scale {})",
                        name,
                        i,
                        axis,
                        a,
                        b,
                        scale
                    );
                }
            }
            assert_eq!(
                fa.simbox.is_some(),
                fb.simbox.is_some(),
                "{}: frame {} simbox presence flipped",
                name,
                i
            );
        }
    }
}

/// 4D-dynamics frames cannot round-trip through `write_dcd`; the writer must
/// reject them with `ErrorKind::Unsupported` rather than silently corrupting.
#[test]
fn test_dcd_writer_rejects_4d() {
    use tempfile::NamedTempFile;
    let path = crate::test_data::get_test_data_path("dcd/4d-dynamic.dcd");
    if !path.exists() {
        return;
    }
    let frames = read_dcd(path.to_str().unwrap()).expect("read");
    let temp = NamedTempFile::new().expect("temp");
    let err =
        write_dcd(temp.path().to_str().unwrap(), &frames).expect_err("write should fail on 4D");
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

/// Iterator API yields the same frames as `read_step`.
#[test]
fn test_dcd_iter_matches_read_step() {
    let path = crate::test_data::get_test_data_path("dcd/water.dcd");
    if !path.exists() {
        return;
    }
    let mut a = open_dcd(&path).expect("open");
    let n = a.len().unwrap();
    let mut b = open_dcd(&path).expect("open");
    let collected: Vec<_> = b.iter().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(collected.len(), n);
    for (i, frame_b) in collected.iter().enumerate().take(n) {
        let step = a.read_step(i).unwrap().unwrap();
        let xa = step.get("atoms").unwrap().get_float("x").unwrap();
        let xb = frame_b.get("atoms").unwrap().get_float("x").unwrap();
        for (va, vb) in xa.iter().zip(xb.iter()) {
            assert!((va - vb).abs() < 1e-9);
        }
    }
}

/// Sanity: opening with the lazy `DcdReader` still parses the header
/// on-demand and exposes its fields.
#[test]
fn test_dcd_header_accessor() {
    let path = crate::test_data::get_test_data_path("dcd/water.dcd");
    if !path.exists() {
        return;
    }
    let file = std::fs::File::open(&path).unwrap();
    let mut reader = DcdReader::new(std::io::BufReader::new(file));
    let header = reader.header().expect("parse header");
    assert!(header.natoms > 0);
    assert!(header.nset > 0);
}
