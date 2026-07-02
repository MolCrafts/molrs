use super::*;

#[cxx::bridge(namespace = "molrs")]
pub mod ffi {
    extern "Rust" {
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
        // Write one frame to an XYZ file (standard element+coords). append=false
        // truncates (create); append=true appends the frame.
        fn write_frame_xyz(
            path: &str,
            type_id: &[i32],
            x: &[f64],
            y: &[f64],
            z: &[f64],
            box_mat: &[f64],
            append: bool,
        );
        fn trajectory_append(
            path: &str,
            type_id: &[i32],
            x: &[f64],
            y: &[f64],
            z: &[f64],
            step: i32,
        );
        // Write one frame + named per-atom fields (field_data reshaped
        // [n_fields, n_atoms]) to a single-frame Zarr store.
        fn write_frame_zarr(
            path: &str,
            type_id: &[i32],
            x: &[f64],
            y: &[f64],
            z: &[f64],
            box_mat: &[f64],
            field_names: Vec<String>,
            field_data: &[f64],
        );
        fn read_frame_zarr_first(path: &str) -> Box<FrameRef>;
        // Read the first frame of an (ext)XYZ file into a materialize-ready
        // FrameRef (atoms.{x,y,z,type} + simbox). `type` is derived from the
        // species/element symbol column (Z). All XYZ parsing lives in molrs.
        fn xyz_read_first_frame(path: &str) -> Box<FrameRef>;

        // ── Trajectory analysis (molrs `compute` kernels) ────────────
        // Raw-array in, curve out — no C++ accumulator. Each fn rebuilds
        // transient molrs frames from flat buffers and delegates the math to
        // molrs (compute::*). No analysis math in C++. See molrs/src/compute/.
        //
        // Mean squared displacement, MsdMode::Direct: MSD(t) = <|r(t) - r(0)|^2>
        // over a row-major [n_frames, n_dof] position buffer (per frame blocked
        // x|y|z, n_dof = 3*n_atoms), first frame = reference (LAMMPS `compute
        // msd`). Returns the mean-MSD curve (index 0 = 0). Empty on < 2 frames /
        // bad shape.
        fn analyze_msd(positions: &[f64], n_frames: i64, n_dof: i64) -> Vec<f64>;

        // Einstein self-diffusion coefficient D = slope / (2*dims) from the
        // windowed-MSD slope over [fit_lo, fit_hi] (fractions of the last lag),
        // over the same [n_frames, n_dof] position buffer as analyze_msd. `dt` =
        // time between frames. Returns NaN on < 2 frames / bad shape / fit error.
        // All math is molrs (EinsteinDiffusion + LinearFit).
        fn analyze_diffusion(
            positions: &[f64],
            n_frames: i64,
            n_dof: i64,
            dt: f64,
            dims: i32,
            fit_lo: f64,
            fit_hi: f64,
        ) -> f64;

        // Velocity autocorrelation function (VDOS / Green-Kubo-diffusion input).
        // `velocities` is a row-major [n_frames, n_dof] flat buffer (n_dof =
        // 3*n_atoms); `resolution` caps the max lag. Returns the DOF-averaged VACF
        // curve, one value per lag (index 0 = zero lag). Empty on < 2 frames / bad
        // args. molrs owns the FFT-ACF math (compute::VACF). frames are unused.
        fn analyze_vacf(
            velocities: &[f64],
            n_frames: i64,
            n_dof: i64,
            dt: f64,
            resolution: i64,
        ) -> Vec<f64>;

        // Radial distribution function g(r) over raw [n_frames, 3*n_atoms]
        // positions (blocked x|y|z per frame) + [n_frames, 9] per-frame cell
        // matrices (supports NPT). Rebuilds a LinkCell self-neighbor list per
        // frame (cutoff = r_max), then batch-accumulates pair distances into
        // `n_bins` bins over [r_min, r_max] (Å), normalized by the ideal-gas
        // shell volume. Returns the g(r) curve, one value per bin; the caller
        // derives bin-center radii (r_min + (i+0.5)*bin_width). Empty on bad
        // args/shape, a missing SimBox, or a compute error. All math is molrs
        // (compute::RDF).
        fn analyze_rdf(
            positions: &[f64],
            boxes: &[f64],
            n_frames: i64,
            n_atoms: i64,
            r_max: f64,
            n_bins: i64,
            r_min: f64,
        ) -> Vec<f64>;
    }
}
