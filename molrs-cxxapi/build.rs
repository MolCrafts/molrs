use std::path::PathBuf;

/// CXX bridge interface schema — single source of truth.
///
/// `{F}` is replaced by `f32` or `f64` based on the `f64` feature,
/// matching Atomiverse's `ATV_REAL`.  Only bulk-data ingestion
/// (coordinates, per-atom fields, distances) uses `{F}`.
/// Scalars cross as `f64` always.
const CXX_BRIDGE_SCHEMA: &str = r#"
use super::*;

#[cxx::bridge(namespace = "molrs")]
pub mod ffi {
    extern "Rust" {
        type AtvMolRec;
        type Rdf;

        // ── MolRec container ─────────────────────────────────────
        fn molrec_new() -> Box<AtvMolRec>;
        fn molrec_set_geometry(rec: &mut AtvMolRec, type_id: &[i32],
            x: &[{F}], y: &[{F}], z: &[{F}], box_mat: &[{F}]);
        fn molrec_add_field(rec: &mut AtvMolRec, name: &str, values: &[{F}]);
        fn molrec_add_scalar(rec: &mut AtvMolRec, name: &str, value: f64);
        fn molrec_add_string(rec: &mut AtvMolRec, name: &str, value: &str);
        fn molrec_commit_frame(rec: &mut AtvMolRec);
        fn molrec_clear(rec: &mut AtvMolRec);
        fn molrec_n_frames(rec: &AtvMolRec) -> i32;

        // ── XYZ I/O ─────────────────────────────────────────────
        fn xyz_write(path: &str, rec: &AtvMolRec);
        fn xyz_write_ext(path: &str, rec: &AtvMolRec);
        fn xyz_append(path: &str, rec: &AtvMolRec);
        fn xyz_append_ext(path: &str, rec: &AtvMolRec);
        fn trajectory_append(path: &str, type_id: &[i32],
            x: &[{F}], y: &[{F}], z: &[{F}], step: i32);
        fn molrec_print_summary(rec: &AtvMolRec);

        // ── RDF (opaque handle wraps molrs::compute::rdf::RDF) ──
        fn rdf_new(n_bins: i32, r_max: f64) -> Box<Rdf>;
        fn rdf_accumulate(rdf: &mut Rdf, distances: &[{F}], n_pairs: i32);
        fn rdf_write(rdf: &Rdf, path: &str,
            n_samples: i32, n_atoms: i32, box_vol: f64);

        // TODO: Mulliken analysis (pending molrs-core implementation)
    }
}
"#;

fn main() {
    let real = if cfg!(feature = "f64") { "f64" } else { "f32" };
    let bridge_src = CXX_BRIDGE_SCHEMA.replace("{F}", real);

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let bridge_path = out_dir.join("bridge.rs");
    std::fs::write(&bridge_path, &bridge_src).unwrap();

    // Also write to src/ so corrosion_add_cxxbridge can find it
    let src_path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("bridge.rs");
    std::fs::write(&src_path, &bridge_src).unwrap();

    cxx_build::bridge(&src_path)
        .std("c++17")
        .compile("molrs_cxxapi");

    println!("cargo::rerun-if-changed=build.rs");
}
