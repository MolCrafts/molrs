use super::*;

#[cxx::bridge(namespace = "molrs")]
pub mod ffi {
    /// Chemical element exported from molrs' canonical Rust periodic table.
    ///
    /// The variants are injected by build.rs from
    /// `molrs/src/core/system/element.rs`; this bridge never owns a second
    /// hand-maintained element table.
    #[repr(u8)]
    enum Element {
        H = 1,
        He = 2,
        Li = 3,
        Be = 4,
        B = 5,
        C = 6,
        N = 7,
        O = 8,
        F = 9,
        Ne = 10,
        Na = 11,
        Mg = 12,
        Al = 13,
        Si = 14,
        P = 15,
        S = 16,
        Cl = 17,
        Ar = 18,
        K = 19,
        Ca = 20,
        Sc = 21,
        Ti = 22,
        V = 23,
        Cr = 24,
        Mn = 25,
        Fe = 26,
        Co = 27,
        Ni = 28,
        Cu = 29,
        Zn = 30,
        Ga = 31,
        Ge = 32,
        As = 33,
        Se = 34,
        Br = 35,
        Kr = 36,
        Rb = 37,
        Sr = 38,
        Y = 39,
        Zr = 40,
        Nb = 41,
        Mo = 42,
        Tc = 43,
        Ru = 44,
        Rh = 45,
        Pd = 46,
        Ag = 47,
        Cd = 48,
        In = 49,
        Sn = 50,
        Sb = 51,
        Te = 52,
        I = 53,
        Xe = 54,
        Cs = 55,
        Ba = 56,
        La = 57,
        Ce = 58,
        Pr = 59,
        Nd = 60,
        Pm = 61,
        Sm = 62,
        Eu = 63,
        Gd = 64,
        Tb = 65,
        Dy = 66,
        Ho = 67,
        Er = 68,
        Tm = 69,
        Yb = 70,
        Lu = 71,
        Hf = 72,
        Ta = 73,
        W = 74,
        Re = 75,
        Os = 76,
        Ir = 77,
        Pt = 78,
        Au = 79,
        Hg = 80,
        Tl = 81,
        Pb = 82,
        Bi = 83,
        Po = 84,
        At = 85,
        Rn = 86,
        Fr = 87,
        Ra = 88,
        Ac = 89,
        Th = 90,
        Pa = 91,
        U = 92,
        Np = 93,
        Pu = 94,
        Am = 95,
        Cm = 96,
        Bk = 97,
        Cf = 98,
        Es = 99,
        Fm = 100,
        Md = 101,
        No = 102,
        Lr = 103,
        Rf = 104,
        Db = 105,
        Sg = 106,
        Bh = 107,
        Hs = 108,
        Mt = 109,
        Ds = 110,
        Rg = 111,
        Cn = 112,
        Nh = 113,
        Fl = 114,
        Mc = 115,
        Lv = 116,
        Ts = 117,
        Og = 118,
    }

    /// Exact frame-metadata dtype.
    #[repr(u8)]
    enum MetaType {
        Bool,
        I32,
        I64,
        U32,
        U64,
        F32,
        F64,
        String,
        Bool3,
        I32x3,
        I64x3,
        U32x3,
        U64x3,
        F32x3,
        F64x3,
        F32x6,
        F64x6,
        F32x9,
        F64x9,
    }

    /// One exact-dtype metadata entry. Only the payload selected by `dtype` is used.
    struct MetaEntry {
        key: String,
        dtype: MetaType,
        bool_value: bool,
        i32_value: i32,
        i64_value: i64,
        u32_value: u32,
        u64_value: u64,
        f32_value: f32,
        f64_value: f64,
        string_value: String,
        bool_values: Vec<u8>,
        i32_values: Vec<i32>,
        i64_values: Vec<i64>,
        u32_values: Vec<u32>,
        u64_values: Vec<u64>,
        f32_values: Vec<f32>,
        f64_values: Vec<f64>,
    }

    extern "Rust" {
        // ── Exact consumer contract ───────────────────────────────
        // Version changes whenever an existing declaration, semantic dtype,
        // or ownership rule changes incompatibly. Capabilities let consumers
        // fail loudly when a required surface was compiled out or omitted.
        fn cxx_api_version() -> u32;
        fn cxx_api_capabilities() -> u64;

        // ── Frame bridge (molrs.Frame via molrs-ffi FrameRef) ─────
        type FrameRef;

        fn frame_schema_version() -> u32;
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
        fn frame_meta_entries(fref: &FrameRef) -> Vec<MetaEntry>;

        // readers — owned copies (RefCell precludes returning borrowed slices)
        fn frame_column_f64(fref: &FrameRef, block: &str, col: &str) -> Vec<f64>;
        fn frame_column_i32(fref: &FrameRef, block: &str, col: &str) -> Vec<i32>;
        fn frame_column_u32(fref: &FrameRef, block: &str, col: &str) -> Vec<u32>;
        fn frame_column_str(fref: &FrameRef, block: &str, col: &str) -> Vec<String>;
        fn frame_box(fref: &FrameRef) -> Vec<f64>;

        // create-or-update writers
        fn frame_set_column_f64(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            data: &[f64],
        ) -> Result<()>;
        fn frame_set_column_i32(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            data: &[i32],
        ) -> Result<()>;
        fn frame_set_column_u32(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            data: &[u32],
        ) -> Result<()>;
        fn frame_set_column_str(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            data: &[String],
        ) -> Result<()>;
        fn frame_set_box(fref: &mut FrameRef, h: &[f64]) -> Result<()>;
        fn frame_set_meta_entry(fref: &mut FrameRef, entry: MetaEntry) -> Result<()>;

        // AM1-BCC: Atomiverse supplies AM1 base charges; molrs owns BCC typing.
        //
        // `parameter_set` selects the correction family by name — "bcc"
        // (BCCPARM.DAT, model id `"bcc"`) or `"abcg2"` (BCCPARM_ABCG2.DAT,
        // `-c abcg2`). A name molrs does not know is refused, never defaulted.
        //
        // Returns `Result`, and that is load-bearing: cxx marks every non-Result
        // `extern "Rust"` fn `noexcept`, so a Rust panic could only abort the
        // calling process. The errors this can raise are the caller's CHEMISTRY —
        // a molecule with no BCC correction row (boron), a missing atom type, a
        // missing bond order — not programmer bugs, so they cross as a catchable
        // `rust::Error` and leave the engine alive to handle them.
        fn am1_bcc_assign_frame_from_base(
            fref: &mut FrameRef,
            am1_charges: &[f64],
            parameter_set: &str,
        ) -> Result<Vec<f64>>;

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
        ) -> Result<()>;
        // Typed-metadata writer. Every MetaEntry carries an explicit dtype;
        // malformed vector lengths are returned as errors.
        fn write_frame_xyz_typed(
            path: &str,
            type_id: &[i32],
            x: &[f64],
            y: &[f64],
            z: &[f64],
            box_mat: &[f64],
            meta: Vec<MetaEntry>,
            append: bool,
        ) -> Result<()>;
        // Write one frame + named per-atom fields (field_data reshaped
        // [n_fields, n_atoms]) to a single-frame Zarr store.
        fn write_frame(
            path: &str,
            type_id: &[i32],
            x: &[f64],
            y: &[f64],
            z: &[f64],
            box_mat: &[f64],
            field_names: Vec<String>,
            field_data: &[f64],
        ) -> Result<()>;
        fn read_first_frame(path: &str) -> Result<Box<FrameRef>>;
        // Read the first frame of an (ext)XYZ file into a materialize-ready
        // FrameRef (atoms.{x,y,z,type} + simbox). `type` is derived from the
        // required ExtXYZ species column (Z). All XYZ parsing lives in molrs.
        fn xyz_read_first_frame(path: &str) -> Result<Box<FrameRef>>;

        // ── Trajectory-analysis compute objects (mirror molrs::compute::*) ──
        // molrs's own calling convention, bridged as CXX opaque types:
        // instantiate a compute with its config (`*_compute_new`), then call
        // `compute(...)` over the whole accumulated raw trajectory. One-shot and
        // stateless — matching the molrs `Compute` trait, there is no
        // frame-by-frame accumulator. Each `compute` rebuilds transient molrs
        // frames from the flat buffers and delegates the math to molrs
        // (`compute::*`); no analysis math lives in C++. See molrs/src/compute/.

        // Mean squared displacement (MsdMode::Direct): MSD(t) = <|r(t) - r(0)|^2>
        // over a row-major [n_frames, n_dof] position buffer (per frame blocked
        // x|y|z, n_dof = 3*n_atoms), first frame = reference (LAMMPS `compute
        // msd`). `compute` returns the mean-MSD curve (index 0 = 0); empty on < 2
        // frames / bad shape.
        type MsdCompute;
        fn msd_compute_new() -> Box<MsdCompute>;
        fn compute(self: &MsdCompute, positions: &[f64], n_frames: i64, n_dof: i64) -> Vec<f64>;

        // Einstein self-diffusion coefficient D = slope / (2*dims) from the
        // windowed-MSD slope over [fit_lo, fit_hi] (fractions of the last lag).
        // `dt` = time between frames; `dims` = spatial dimensionality. `compute`
        // takes the same [n_frames, n_dof] position buffer as MsdCompute and
        // returns D; NaN on < 2 frames / bad shape / fit error. All math is molrs
        // (EinsteinDiffusion + LinearFit).
        type DiffusionCompute;
        fn diffusion_compute_new(
            dt: f64,
            dims: i32,
            fit_lo: f64,
            fit_hi: f64,
        ) -> Box<DiffusionCompute>;
        fn compute(self: &DiffusionCompute, positions: &[f64], n_frames: i64, n_dof: i64) -> f64;

        // Velocity autocorrelation function (VDOS / Green-Kubo input). `dt` = time
        // between frames; `resolution` caps the max lag. `compute` takes a
        // row-major [n_frames, n_dof] velocity buffer (n_dof = 3*n_atoms) and
        // returns the DOF-averaged VACF curve (index 0 = zero lag); empty on < 2
        // frames / bad args. molrs owns the FFT-ACF math (compute::VACF).
        type VacfCompute;
        fn vacf_compute_new(dt: f64, resolution: i64) -> Box<VacfCompute>;
        fn compute(self: &VacfCompute, velocities: &[f64], n_frames: i64, n_dof: i64) -> Vec<f64>;

        // Radial distribution function g(r). Config: n_bins, r_max, r_min (Å).
        // `compute` takes raw [n_frames, 3*n_atoms] positions (blocked x|y|z per
        // frame) + [n_frames, 9] per-frame cell matrices (supports NPT), rebuilds
        // a LinkCell self-neighbor list per frame (cutoff = r_max), and returns
        // the g(r) curve (one value per bin; the caller derives bin-center radii
        // r_min + (i+0.5)*bin_width). Empty on bad args/shape, a missing SimBox,
        // or a compute error. All math is molrs (compute::RDF).
        type RdfCompute;
        fn rdf_compute_new(n_bins: i64, r_max: f64, r_min: f64) -> Box<RdfCompute>;
        fn compute(
            self: &RdfCompute,
            positions: &[f64],
            boxes: &[f64],
            n_frames: i64,
            n_atoms: i64,
        ) -> Vec<f64>;

        // ── Streaming trajectory-analysis accumulators (bounded memory) ──
        // Frame-by-frame counterparts of the batch computes above: construct
        // with the same config, feed ONE frame per `accumulate` call, read the
        // result once at the end. State is O(bins / window·n_dof /
        // resolution·n_dof) — never O(trajectory) — so arbitrarily long MD
        // runs stream through without growing memory. All math is molrs
        // (compute::{RDFAccumulator, MSDAccumulator},
        // compute::transport::VACFAccumulator); `accumulate` returns false when a
        // frame is rejected (shape/DOF mismatch, bad box), leaving state
        // unchanged.

        // Streaming g(r): one flat blocked-x|y|z position frame + row-major
        // 3x3 cell per call; `finalize` returns the normalized g(r) (empty
        // before the first accepted frame). Identical numerics to RdfCompute
        // over the same frames.
        type RdfAccumulator;
        fn rdf_accumulator_new(n_bins: i64, r_max: f64, r_min: f64) -> Box<RdfAccumulator>;
        fn accumulate(self: &mut RdfAccumulator, positions: &[f64], box9: &[f64]) -> bool;
        fn n_frames(self: &RdfAccumulator) -> i64;
        fn finalize(self: &RdfAccumulator) -> Vec<f64>;

        // Streaming MSD: Direct-mode curve (frame 0 = reference, exact) plus
        // windowed-MSD sums capped at `window` lags (ring buffer). `diffusion`
        // = LinearFit slope / (2*dims) over the windowed curve within
        // [fit_lo, fit_hi] fractions of the max resolved lag; NaN on bad
        // args / too few frames / window = 0.
        type MsdAccumulator;
        fn msd_accumulator_new(window: i64) -> Box<MsdAccumulator>;
        fn accumulate(self: &mut MsdAccumulator, positions: &[f64]) -> bool;
        fn n_frames(self: &MsdAccumulator) -> i64;
        fn direct_curve(self: &MsdAccumulator) -> Vec<f64>;
        fn diffusion(self: &MsdAccumulator, dt: f64, dims: i32, fit_lo: f64, fit_hi: f64) -> f64;

        // Streaming DOF-averaged velocity ACF, lags 0..=resolution
        // (resolution >= 1). Matches VacfCompute (FFT batch path) to FFT
        // round-off over the same frames.
        type VacfAccumulator;
        fn vacf_accumulator_new(resolution: i64) -> Box<VacfAccumulator>;
        fn accumulate(self: &mut VacfAccumulator, velocities: &[f64]) -> bool;
        fn n_frames(self: &VacfAccumulator) -> i64;
        fn finalize(self: &VacfAccumulator) -> Vec<f64>;
    }
}
