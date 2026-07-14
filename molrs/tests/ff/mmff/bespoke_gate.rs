//! ac-011 — the bespoke MMFF path is UNTOUCHED.
//!
//! `molrs/src/ff/mmff/` is the only implementation that reproduces RDKit exactly
//! (max |delta| = 0.00000 on all 11 fixtures), and mmff-orthogonal-01 fixes the
//! *generic* path by measuring it against that one. A spec that quietly edited
//! its own reference frame would be measuring nothing.
//!
//! So this is the ordering discipline in machine-checkable form:
//!
//! > NEVER delete or disturb the only correct implementation before the
//! > replacement is proven.
//!
//! `BESPOKE.sha256` pins the SHA-256 of every file under `src/ff/mmff/`, taken
//! at the start of the spec. The deletion of the bespoke path belongs to
//! `mmff-orthogonal-02-delete-bespoke`, which removes this gate along with the
//! tree it guards — deliberately, in one commit, not by accident in this one.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("molrs/ has a parent")
}

fn bespoke_dir() -> PathBuf {
    repo_root().join("molrs/src/ff/mmff")
}

fn manifest_path() -> PathBuf {
    repo_root().join("molrs/tests/ff/mmff/BESPOKE.sha256")
}

/// Every file under `src/ff/mmff/`, relative to it, sorted.
fn bespoke_files() -> Vec<String> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).expect("read bespoke dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, root, out);
            } else {
                out.push(
                    path.strip_prefix(root)
                        .expect("under root")
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    let mut out = Vec::new();
    walk(&bespoke_dir(), &bespoke_dir(), &mut out);
    out.sort();
    out
}

#[test]
fn bespoke_mmff_path_is_byte_for_byte_unchanged() {
    let manifest = std::fs::read_to_string(manifest_path())
        .expect("BESPOKE.sha256 is committed next to this test");

    let pinned: Vec<(&str, &str)> = manifest
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            (
                it.next().expect("manifest row has a hash"),
                it.next().expect("manifest row has a file name"),
            )
        })
        .collect();
    assert!(
        !pinned.is_empty(),
        "BESPOKE.sha256 pins nothing — an empty manifest would make this gate vacuous"
    );

    let on_disk = bespoke_files();
    let pinned_names: Vec<&str> = pinned.iter().map(|(_, n)| *n).collect();
    assert_eq!(
        on_disk, pinned_names,
        "the file list under molrs/src/ff/mmff/ changed (files added or removed). \
         mmff-orthogonal-01 must not touch the bespoke path; deleting it is spec 02's job"
    );

    let mut changed = Vec::new();
    for (want, name) in pinned {
        let bytes = std::fs::read(bespoke_dir().join(name)).expect("read bespoke file");
        let got = format!("{:x}", Sha256::digest(&bytes));
        if got != want {
            changed.push(name.to_owned());
        }
    }
    assert!(
        changed.is_empty(),
        "molrs/src/ff/mmff/ was modified: {changed:?}\n\
         This is the ONLY MMFF implementation that reproduces RDKit exactly, and it is the \
         reference frame the generic-path fix is measured against (`<name>.breakdown.json` was \
         frozen from it). mmff-orthogonal-01 fixes the GENERIC path only. If a change here is \
         genuinely required, it is a different spec — and the frozen breakdowns must be \
         re-derived and re-validated against RDKit first."
    );
}
