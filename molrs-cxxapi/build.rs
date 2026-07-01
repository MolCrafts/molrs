use std::path::PathBuf;

/// CXX bridge interface schema — single source of truth.
///
/// The merged `molcrafts-molrs` crate hardcodes `F = f64`, so every coordinate,
/// per-atom field, and distance crosses the bridge as `f64` unconditionally
/// (matching Atomiverse's `ATV_REAL`). There is no precision-switching feature.
const CXX_BRIDGE_SCHEMA: &str = r#"use super::*;

#[cxx::bridge(namespace = "molrs")]
pub mod ffi {
    extern "Rust" {
        type AtvMolRec;

        // ── MolRec container ─────────────────────────────────────
        fn molrec_new() -> Box<AtvMolRec>;
        fn molrec_set_geometry(
            rec: &mut AtvMolRec,
            type_id: &[i32],
            x: &[f64],
            y: &[f64],
            z: &[f64],
            box_mat: &[f64],
        );
        fn molrec_add_field(rec: &mut AtvMolRec, name: &str, values: &[f64]);
        fn molrec_add_scalar(rec: &mut AtvMolRec, name: &str, value: f64);
        fn molrec_add_string(rec: &mut AtvMolRec, name: &str, value: &str);
        fn molrec_commit_frame(rec: &mut AtvMolRec);
        fn molrec_clear(rec: &mut AtvMolRec);
        fn molrec_n_frames(rec: &AtvMolRec) -> i32;

        // ── Frame bridge (molrs.Frame via molrs-ffi FrameRef) ─────
        type FrameRef;

        fn frame_new() -> Box<FrameRef>;

        // Cross-extension ingress: rebuild a bridge handle from the raw
        // address of a molrs-python `*mut molrs_ffi::FrameRef` (carried by
        // `Frame._ffi_frameref_capsule()`). `unsafe` — caller must pass a
        // live pointer; see the `# Safety` note on the Rust impl.
        unsafe fn frame_clone_from_addr(addr: usize) -> Box<FrameRef>;

        // introspection
        fn frame_block_names(fref: &FrameRef) -> Vec<String>;
        fn frame_has_block(fref: &FrameRef, block: &str) -> bool;
        fn frame_block_columns(fref: &FrameRef, block: &str) -> Vec<String>;
        fn frame_block_nrows(fref: &FrameRef, block: &str) -> i64;

        // readers — owned copies (RefCell precludes returning borrowed slices)
        fn frame_column_f64(fref: &FrameRef, block: &str, col: &str) -> Vec<f64>;
        fn frame_column_i32(fref: &FrameRef, block: &str, col: &str) -> Vec<i32>;
        fn frame_column_u32(fref: &FrameRef, block: &str, col: &str) -> Vec<u32>;
        fn frame_column_str(fref: &FrameRef, block: &str, col: &str) -> Vec<String>;
        fn frame_simbox(fref: &FrameRef) -> Vec<f64>;

        // create-or-update writers
        fn frame_set_column_f64(fref: &mut FrameRef, block: &str, col: &str, data: &[f64]);
        fn frame_set_column_i32(fref: &mut FrameRef, block: &str, col: &str, data: &[i32]);
        fn frame_set_column_u32(fref: &mut FrameRef, block: &str, col: &str, data: &[u32]);
        fn frame_set_column_str(fref: &mut FrameRef, block: &str, col: &str, data: &[String]);
        fn frame_set_simbox(fref: &mut FrameRef, h: &[f64]);

        // ── I/O ──────────────────────────────────────────────────
        fn xyz_write(path: &str, rec: &AtvMolRec);
        fn xyz_write_ext(path: &str, rec: &AtvMolRec);
        fn xyz_append(path: &str, rec: &AtvMolRec);
        fn xyz_append_ext(path: &str, rec: &AtvMolRec);
        fn trajectory_append(
            path: &str,
            type_id: &[i32],
            x: &[f64],
            y: &[f64],
            z: &[f64],
            step: i32,
        );
        fn molrec_write_zarr(path: &str, rec: &AtvMolRec);
        fn molrec_read_zarr_first_frame(path: &str) -> Box<FrameRef>;
        // Read the first frame of an (ext)XYZ file into a materialize-ready
        // FrameRef (atoms.{x,y,z,type} + simbox). `type` is derived from the
        // species/element symbol column (Z). All XYZ parsing lives in molrs.
        fn xyz_read_first_frame(path: &str) -> Box<FrameRef>;
        fn molrec_print_summary(rec: &AtvMolRec);

        // ── Trajectory analysis (molrs `compute` kernels) ────────────
        // The C++ engine measures (accumulates frames into the MolRec); molrs
        // analyzes. No analysis math in C++. See molrs/src/compute/.
        //
        // Mean squared displacement, MsdMode::Direct: MSD(t) = <|r(t) - r(0)|^2>
        // with the first committed frame as reference (LAMMPS `compute msd`
        // semantics). Returns the mean-MSD curve, one value per committed frame
        // (element 0 is the reference, always 0). Empty when < 2 frames or on a
        // compute error -- never panics/aborts.
        fn analyze_msd(rec: &AtvMolRec) -> Vec<f64>;

        // Einstein self-diffusion coefficient D from the windowed-MSD slope:
        // D = slope / (2 * dims), where slope is an OLS linear fit of MSD(t) over
        // the lag window [fit_lo, fit_hi] (fractions of the last lag, 0<=lo<hi<=1).
        // `dt` = time between committed frames. Returns NaN on < 2 frames or a
        // fit error -- never panics/aborts. All math is molrs (EinsteinDiffusion
        // + LinearFit).
        fn analyze_diffusion(rec: &AtvMolRec, dt: f64, dims: i32, fit_lo: f64, fit_hi: f64) -> f64;

        // Mulliken stays in C++ — depends on electronic structure context (basis
        // sets, overlap matrix) that is not available from raw simulation data.
        // TODO: RDF via molrs (currently pure C++ in Atomiverse)
    }
}
"#;

fn main() {
    // molcrafts-molrs hardcodes F = f64.  No precision substitution needed.
    let bridge_src = CXX_BRIDGE_SCHEMA;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let bridge_path = out_dir.join("bridge.rs");
    std::fs::write(&bridge_path, bridge_src).unwrap();

    // Also write to src/ so corrosion_add_cxxbridge can find it
    let src_path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("bridge.rs");
    std::fs::write(&src_path, bridge_src).unwrap();

    cxx_build::bridge(&src_path)
        .std("c++17")
        .compile("molrs_cxxapi");

    println!("cargo::rerun-if-changed=build.rs");
}
