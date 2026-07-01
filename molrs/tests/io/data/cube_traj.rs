//! Integration tests for the multi-frame Cube trajectory reader
//! (travis-parity-07, ac-001).
//!
//! Per the mandatory IO Testing Rule (CLAUDE.md), these iterate **every real
//! file** under `tests-data/cube_traj/` — never synthetic cubes. The fixtures
//! are multi-frame AIMD electron-density cube streams that must be added to the
//! `MolCrafts/tests-data` repo (`cube_traj/` subdirectory) before this test can
//! run; until then `cargo test --test io` reports the missing directory exactly
//! as for every other format whose fixtures are not cloned.

use molrs::io::trajectory::cube_traj::read_cube_trajectory;

/// Every concatenated cube-trajectory fixture must parse into ≥ 2 frames, each
/// with a `grid` block and a consistent grid shape + atom count across frames.
#[test]
fn test_all_cube_traj_files_parse() {
    let files = crate::common::format_files("cube_traj");
    assert!(
        !files.is_empty(),
        "No cube_traj fixtures in tests-data/cube_traj/ — add real multi-frame \
         cube streams to the MolCrafts/tests-data repo (see CLAUDE.md IO rule)"
    );
    for path in files {
        let frames = read_cube_trajectory(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        assert!(
            frames.len() >= 2,
            "{path:?}: expected a multi-frame trajectory, got {} frame(s)",
            frames.len()
        );
        let grid0 = frames[0].get("grid").expect("frame 0 has a grid block");
        let shape0 = grid0.shape();
        let natoms0 = frames[0].get("atoms").and_then(|b| b.nrows());
        for (i, f) in frames.iter().enumerate() {
            let g = f
                .get("grid")
                .unwrap_or_else(|| panic!("frame {i} grid block"));
            assert_eq!(
                g.shape(),
                shape0,
                "{path:?}: grid shape drifts at frame {i}"
            );
            assert_eq!(
                f.get("atoms").and_then(|b| b.nrows()),
                natoms0,
                "{path:?}: atom count drifts at frame {i}"
            );
        }
    }
}
