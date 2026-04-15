
use super::*;

#[cxx::bridge(namespace = "molrs")]
pub mod ffi {
    extern "Rust" {
        type AtvMolRec;

        // ── MolRec container ─────────────────────────────────────
        fn molrec_new() -> Box<AtvMolRec>;
        fn molrec_set_geometry(rec: &mut AtvMolRec, type_id: &[i32],
            x: &[f64], y: &[f64], z: &[f64], box_mat: &[f64]);
        fn molrec_add_field(rec: &mut AtvMolRec, name: &str, values: &[f64]);
        fn molrec_add_scalar(rec: &mut AtvMolRec, name: &str, value: f64);
        fn molrec_add_string(rec: &mut AtvMolRec, name: &str, value: &str);
        fn molrec_commit_frame(rec: &mut AtvMolRec);
        fn molrec_clear(rec: &mut AtvMolRec);
        fn molrec_n_frames(rec: &AtvMolRec) -> i32;

        // ── I/O ──────────────────────────────────────────────────
        fn xyz_write(path: &str, rec: &AtvMolRec);
        fn xyz_write_ext(path: &str, rec: &AtvMolRec);
        fn xyz_append(path: &str, rec: &AtvMolRec);
        fn xyz_append_ext(path: &str, rec: &AtvMolRec);
        fn trajectory_append(path: &str, type_id: &[i32],
            x: &[f64], y: &[f64], z: &[f64], step: i32);
        fn molrec_write_zarr(path: &str, rec: &AtvMolRec);
        fn molrec_print_summary(rec: &AtvMolRec);

        // Mulliken stays in C++ — depends on electronic structure context (basis
        // sets, overlap matrix) that is not available from raw simulation data.
        // TODO: RDF via molrs (currently pure C++ in Atomiverse)
    }
}
