use super::*;

#[cxx::bridge(namespace = "molrs")]
pub mod ffi {
    /// Chemical element exported from molrs' canonical Rust periodic table.
    ///
    /// The variants are injected by build.rs from
    /// `molrs/src/core/element.rs`; this bridge never owns a second
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
        F64,
        String,
        Bool3,
        I32x3,
        I64x3,
        U32x3,
        U64x3,
        F64x3,
        F64x6,
        F64x9,
    }

    /// One frame-metadata value with its key and exact dtype. Only the payload
    /// selected by `dtype` is used.
    struct KeyedMetaValue {
        key: String,
        dtype: MetaType,
        bool_value: bool,
        i32_value: i32,
        i64_value: i64,
        u32_value: u32,
        u64_value: u64,
        f64_value: f64,
        string_value: String,
        bool_values: Vec<u8>,
        i32_values: Vec<i32>,
        i64_values: Vec<i64>,
        u32_values: Vec<u32>,
        u64_values: Vec<u64>,
        f64_values: Vec<f64>,
    }

    extern "Rust" {
        // ── Exact consumer contract ───────────────────────────────
        // Capabilities let consumers fail loudly when a required surface was
        // compiled out or omitted.
        fn cxx_api_capabilities() -> u64;

        // ── Frame bridge (molrs.Frame via molrs-ffi FrameRef) ─────
        type FrameRef;

        // ── Region bridge (molrs.Region via molrs-ffi RegionRef) ──
        // A region answers a signed distance; `contains` is its sign and
        // `bounds` the box it fits in. Compositions are ordinary handles, so
        // a shell is `region_and(outer, region_not(inner))`.
        type RegionRef;

        // Every vector argument is exactly 3 values; anything else is an error.
        fn region_sphere(center: &[f64], radius: f64) -> Result<Box<RegionRef>>;
        fn region_cuboid(origin: &[f64], lengths: &[f64]) -> Result<Box<RegionRef>>;
        fn region_half_space(normal: &[f64], point: &[f64]) -> Result<Box<RegionRef>>;
        fn region_cylinder(
            base: &[f64],
            axis: &[f64],
            radius: f64,
            length: f64,
        ) -> Result<Box<RegionRef>>;
        fn region_ellipsoid(center: &[f64], semi_axes: &[f64]) -> Result<Box<RegionRef>>;

        fn region_and(a: &RegionRef, b: &RegionRef) -> Box<RegionRef>;
        fn region_or(a: &RegionRef, b: &RegionRef) -> Box<RegionRef>;
        fn region_not(a: &RegionRef) -> Box<RegionRef>;

        // `points` is flat [x, y, z, ...]; a ragged length is an error.
        fn region_distance(rref: &RegionRef, points: &[f64]) -> Result<Vec<f64>>;
        fn region_contains(rref: &RegionRef, points: &[f64]) -> Result<Vec<u8>>;
        fn region_bounds(rref: &RegionRef) -> Vec<f64>;

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
        fn frame_block_n_rows(fref: &FrameRef, block: &str) -> i64;

        // metadata: keys in insertion order, one value by key, insert/replace
        fn frame_meta_keys(fref: &FrameRef) -> Vec<String>;
        fn frame_get_meta(fref: &FrameRef, key: &str) -> Result<KeyedMetaValue>;
        fn frame_set_meta(fref: &mut FrameRef, value: KeyedMetaValue) -> Result<()>;

        // readers — owned copies (RefCell precludes returning borrowed slices);
        // an absent block or column reads as empty. The box is its cell
        // matrix H, 9 row-major values (empty when the frame has no box).
        fn frame_column_f64(fref: &FrameRef, block: &str, col: &str) -> Vec<f64>;
        fn frame_column_i32(fref: &FrameRef, block: &str, col: &str) -> Vec<i32>;
        fn frame_column_u64(fref: &FrameRef, block: &str, col: &str) -> Vec<u64>;
        fn frame_column_str(fref: &FrameRef, block: &str, col: &str) -> Vec<String>;
        fn frame_box_h(fref: &FrameRef) -> Vec<f64>;

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
        fn frame_set_column_u64(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            data: &[u64],
        ) -> Result<()>;
        fn frame_set_column_str(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            data: &[String],
        ) -> Result<()>;
        // a periodic box at the origin from its 9-value row-major H
        fn frame_set_box_h(fref: &mut FrameRef, h: &[f64]) -> Result<()>;
        // declared precision of an f64 column: a `*.mrec` writer stores it
        // rounded to the largest power of two <= precision (molrec)
        fn frame_set_precision(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            precision: f64,
        ) -> Result<()>;

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
        fn assign_am1_bcc_charges(
            fref: &mut FrameRef,
            am1_charges: &[f64],
            parameter_set: &str,
        ) -> Result<Vec<f64>>;

        // ── I/O ──────────────────────────────────────────────────
        // The writers take atomic numbers + blocked coordinates; `h` is empty
        // (no box) or a 9-value row-major H. A length mismatch, an unknown
        // atomic number or a malformed / singular H is an error.

        // Write one frame (element + coords + typed metadata) to an XYZ file
        // (molrs XyzWriter). append=false truncates; append=true appends.
        fn write_xyz(
            path: &str,
            atomic_number: &[i32],
            x: &[f64],
            y: &[f64],
            z: &[f64],
            h: &[f64],
            meta: Vec<KeyedMetaValue>,
            append: bool,
        ) -> Result<()>;
        // Read the first frame of an (ext)XYZ file into a materialize-ready
        // FrameRef (atoms.{x,y,z,atomic_number} + box). `atomic_number` is
        // derived from the required ExtXYZ species column. All XYZ parsing
        // lives in molrs (io::read_xyz).
        fn read_xyz(path: &str) -> Result<Box<FrameRef>>;

        // Write one frame + named per-atom fields (field_data is
        // [n_fields, n_atoms]) as a `*.mrec` record whose `frame` section is
        // that frame (molrs io::write_mrec_frame); read it back with
        // read_mrec_frame (molrs io::read_mrec_frame).
        fn write_mrec_frame(
            path: &str,
            atomic_number: &[i32],
            x: &[f64],
            y: &[f64],
            z: &[f64],
            h: &[f64],
            field_names: Vec<String>,
            field_data: &[f64],
        ) -> Result<()>;
        fn read_mrec_frame(path: &str) -> Result<Box<FrameRef>>;
        // Frame `index` of the trajectory in a `*.mrec` record (molrs
        // MrecReader::frame); reads what an MrecWriterRef wrote.
        fn read_mrec_trajectory_frame(path: &str, index: u64) -> Result<Box<FrameRef>>;

        // ── Streaming `*.mrec` trajectory writer (molrs MrecWriter) ──
        // The engine's output path: one writer per run, one frame per
        // append; complete inner chunks land on their own, `flush` commits
        // (durably when `durable`), `close` ends the run. `flush_every == 0`
        // leaves the landing cadence to the writer. `schema_from` pins the
        // blocks/columns every later frame must stay inside.
        type MrecWriterRef;
        fn mrec_writer_create(
            path: &str,
            schema_from: &FrameRef,
            flush_every: u64,
            durable: bool,
        ) -> Result<Box<MrecWriterRef>>;
        fn mrec_writer_open(
            path: &str,
            flush_every: u64,
            durable: bool,
        ) -> Result<Box<MrecWriterRef>>;
        fn mrec_writer_append(
            writer: &mut MrecWriterRef,
            fref: &FrameRef,
            step: i64,
            time: f64,
            has_time: bool,
        ) -> Result<()>;
        fn mrec_writer_flush(writer: &mut MrecWriterRef) -> Result<()>;
        fn mrec_writer_committed(writer: &MrecWriterRef) -> u64;
        fn mrec_writer_close(writer: Box<MrecWriterRef>) -> Result<()>;

        // ── Trajectory analysis (mirrors molrs::compute) ─────────
        // Two shapes, named as molrs names them. One-shot analyses (`Msd`,
        // `EinsteinDiffusion`, `Vacf`, `Rdf`): construct with the config
        // (`*_new`), then `compute(...)` over the whole raw trajectory. Their
        // streaming counterparts (`RdfAccumulator`, `MsdAccumulator`,
        // `VacfAccumulator`) further down take one frame per call. Each
        // rebuilds transient molrs frames from the flat buffers and delegates
        // the math to molrs; no analysis math lives in C++.

        // Mean squared displacement (MsdMode::Direct): MSD(t) = <|r(t) - r(0)|^2>
        // over a row-major [n_frames, n_dof] position buffer (per frame blocked
        // x|y|z, n_dof = 3*n_atoms), first frame = reference (LAMMPS `compute
        // msd`). `compute` returns the mean-MSD curve (index 0 = 0); empty on < 2
        // frames / bad shape.
        type Msd;
        fn msd_new() -> Box<Msd>;
        fn compute(self: &Msd, positions: &[f64], n_frames: i64, n_dof: i64) -> Vec<f64>;

        // Einstein self-diffusion coefficient D (molrs
        // EinsteinDiffusionResult::diffusion_coefficient) from the windowed-MSD
        // slope over [fit_lo, fit_hi] (fractions of the last lag). `dt` = time
        // between frames; `dims` = spatial dimensionality. `compute` takes the
        // same [n_frames, n_dof] position buffer as Msd and returns D; NaN on
        // < 2 frames / bad shape / fit error.
        type EinsteinDiffusion;
        fn einstein_diffusion_new(
            dt: f64,
            dims: i32,
            fit_lo: f64,
            fit_hi: f64,
        ) -> Box<EinsteinDiffusion>;
        fn compute(self: &EinsteinDiffusion, positions: &[f64], n_frames: i64, n_dof: i64) -> f64;

        // Velocity autocorrelation function (VDOS / Green-Kubo input). `dt` = time
        // between frames; `resolution` caps the max lag. `compute` takes a
        // row-major [n_frames, n_dof] velocity buffer (n_dof = 3*n_atoms) and
        // returns the DOF-averaged VACF curve (index 0 = zero lag); empty on < 2
        // frames / bad args. molrs owns the FFT-ACF math (compute::Vacf).
        type Vacf;
        fn vacf_new(dt: f64, resolution: i64) -> Box<Vacf>;
        fn compute(self: &Vacf, velocities: &[f64], n_frames: i64, n_dof: i64) -> Vec<f64>;

        // Radial distribution function g(r). Config: n_bins, r_max, r_min (Å).
        // `compute` takes raw [n_frames, 3*n_atoms] positions (blocked x|y|z per
        // frame) + [n_frames, 9] per-frame cell matrices H (supports NPT),
        // builds a self-neighbor list per frame (cutoff = r_max), and returns
        // the g(r) curve (one value per bin; the caller derives bin-center radii
        // r_min + (i+0.5)*bin_width). Empty on bad args/shape, a singular H,
        // or a compute error. All math is molrs (compute::Rdf).
        type Rdf;
        fn rdf_new(n_bins: i64, r_max: f64, r_min: f64) -> Box<Rdf>;
        fn compute(
            self: &Rdf,
            positions: &[f64],
            boxes: &[f64],
            n_frames: i64,
            n_atoms: i64,
        ) -> Vec<f64>;

        // ── Streaming accumulators (bounded memory) ──────────────
        // Frame-by-frame counterparts of the analyses above: construct with
        // the same config, feed ONE frame per `accumulate` call, read the
        // result once at the end. State is O(bins / window·n_dof /
        // resolution·n_dof) — never O(trajectory) — so arbitrarily long MD
        // runs stream through without growing memory. All math is molrs
        // (compute::{RdfAccumulator, MsdAccumulator, VacfAccumulator});
        // `accumulate` returns false when a frame is rejected (shape/DOF
        // mismatch, bad box), leaving state unchanged.

        // Streaming g(r): one flat blocked-x|y|z position frame + row-major
        // 3x3 H per call; `finalize` returns the normalized g(r) (empty
        // before the first accepted frame). Identical numerics to Rdf over
        // the same frames.
        type RdfAccumulator;
        fn rdf_accumulator_new(n_bins: i64, r_max: f64, r_min: f64) -> Box<RdfAccumulator>;
        fn accumulate(self: &mut RdfAccumulator, positions: &[f64], box9: &[f64]) -> bool;
        fn n_frames(self: &RdfAccumulator) -> i64;
        fn finalize(self: &RdfAccumulator) -> Vec<f64>;

        // Streaming MSD: Direct-mode curve (frame 0 = reference, exact) plus
        // windowed-MSD sums capped at `window` lags (ring buffer). `diffusion`
        // is molrs EinsteinDiffusionResult::diffusion_coefficient over the
        // windowed curve within [fit_lo, fit_hi] fractions of the max resolved
        // lag; NaN on bad args / too few frames / window = 0.
        type MsdAccumulator;
        fn msd_accumulator_new(window: i64) -> Box<MsdAccumulator>;
        fn accumulate(self: &mut MsdAccumulator, positions: &[f64]) -> bool;
        fn n_frames(self: &MsdAccumulator) -> i64;
        fn direct_curve(self: &MsdAccumulator) -> Vec<f64>;
        fn diffusion(self: &MsdAccumulator, dt: f64, dims: i32, fit_lo: f64, fit_hi: f64) -> f64;

        // Streaming DOF-averaged velocity ACF, lags 0..=resolution
        // (resolution >= 1). Matches Vacf (FFT batch path) to FFT round-off
        // over the same frames.
        type VacfAccumulator;
        fn vacf_accumulator_new(resolution: i64) -> Box<VacfAccumulator>;
        fn accumulate(self: &mut VacfAccumulator, velocities: &[f64]) -> bool;
        fn n_frames(self: &VacfAccumulator) -> i64;
        fn finalize(self: &VacfAccumulator) -> Vec<f64>;
    }
}
