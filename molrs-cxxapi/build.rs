use std::path::{Path, PathBuf};

/// CXX bridge interface schema — single source of truth.
///
/// The merged `molcrafts-molrs` crate hardcodes `F = f64`, so every coordinate,
/// per-atom field, and distance crosses the bridge as `f64` unconditionally
/// Atomiverse converts at its Frame materialization boundary when its storage
/// tier is not f64. There is no molrs precision-switching feature.
const CXX_BRIDGE_SCHEMA: &str = r#"use super::*;

#[cxx::bridge(namespace = "molrs")]
pub mod ffi {
    /// Chemical element exported from molrs' canonical Rust periodic table.
    ///
    /// The variants are injected by build.rs from
    /// `molrs/src/core/system/element.rs`; this bridge never owns a second
    /// hand-maintained element table.
    #[repr(u8)]
    enum Element {
__MOLRS_ELEMENT_VARIANTS__    }

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
        fn write_frame_zarr(
            path: &str,
            type_id: &[i32],
            x: &[f64],
            y: &[f64],
            z: &[f64],
            box_mat: &[f64],
            field_names: Vec<String>,
            field_data: &[f64],
        ) -> Result<()>;
        fn read_frame_zarr_first(path: &str) -> Result<Box<FrameRef>>;
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
"#;

/// Render the canonical Rust `Element` variants into the CXX shared enum.
///
/// The parser intentionally accepts only the exact `Name = Z,` representation
/// used by the canonical enum and requires the complete contiguous periodic
/// table. A source-format drift therefore fails the build instead of silently
/// exporting a partial or differently numbered C++ type.
fn cxx_element_variants(element_source: &str) -> String {
    let marker = "pub enum Element {";
    let body = element_source
        .split_once(marker)
        .unwrap_or_else(|| panic!("canonical Element declaration `{marker}` not found"))
        .1;

    let mut expected_z = 1_u8;
    let mut rendered = String::new();
    for raw in body.lines() {
        let line = raw.trim();
        if line == "}" {
            break;
        }
        if line.is_empty() || line.starts_with("//") || line.starts_with("#") {
            continue;
        }

        let (name, value) = line
            .strip_suffix(',')
            .and_then(|line| line.split_once('='))
            .unwrap_or_else(|| panic!("invalid canonical Element variant line: `{line}`"));
        let name = name.trim();
        let z: u8 = value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("invalid atomic number in Element variant: `{line}`"));
        assert_eq!(
            z, expected_z,
            "canonical Element must remain contiguous: expected Z={expected_z}, found `{line}`"
        );
        assert!(
            name.chars().next().is_some_and(char::is_uppercase)
                && name.chars().all(|ch| ch.is_ascii_alphabetic()),
            "invalid canonical Element variant name: `{name}`"
        );
        rendered.push_str(&format!("        {name} = {z},\n"));
        expected_z = expected_z
            .checked_add(1)
            .expect("canonical Element has more than 255 variants");
    }
    assert_eq!(
        expected_z, 119,
        "canonical Element must export exactly H=1 through Og=118"
    );
    rendered
}

fn main() {
    // molcrafts-molrs hardcodes F = f64.  No precision substitution needed.
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let api_version_path = manifest_dir.join("CXX_API_VERSION");
    let api_version_source = std::fs::read_to_string(&api_version_path)
        .unwrap_or_else(|err| panic!("read {}: {err}", api_version_path.display()));
    let api_version: u32 = api_version_source.trim().parse().unwrap_or_else(|_| {
        panic!(
            "{} must contain one unsigned decimal CXX API version",
            api_version_path.display()
        )
    });
    assert!(api_version > 0, "CXX API version zero is reserved");
    let element_path = manifest_dir
        .join("..")
        .join("molrs")
        .join("src")
        .join("core")
        .join("system")
        .join("element.rs");
    let element_source = std::fs::read_to_string(&element_path).unwrap_or_else(|err| {
        panic!(
            "read canonical Element source {}: {err}",
            element_path.display()
        )
    });
    let variants = cxx_element_variants(&element_source);
    let bridge_src = CXX_BRIDGE_SCHEMA.replace("__MOLRS_ELEMENT_VARIANTS__", &variants);
    assert!(
        !bridge_src.contains("__MOLRS_ELEMENT_VARIANTS__"),
        "CXX Element variant placeholder was not replaced"
    );

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(
        out_dir.join("cxx_api_version.rs"),
        format!("const MOLRS_CXX_API_VERSION: u32 = {api_version};\n"),
    )
    .unwrap();
    let bridge_path = out_dir.join("bridge.rs");
    std::fs::write(&bridge_path, &bridge_src).unwrap();

    // Also write to src/ so corrosion_add_cxxbridge can find it
    let src_path = manifest_dir.join("src").join("bridge.rs");
    std::fs::write(&src_path, &bridge_src).unwrap();

    cxx_build::bridge(&src_path)
        .std("c++20")
        .compile("molrs_cxxapi");

    compile_test_probe(&out_dir);

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={}", api_version_path.display());
    println!("cargo::rerun-if-changed={}", element_path.display());
}

/// Compile `tests/cxx/bridge_probe.cc` — the C++ caller the integration tests
/// need — against the header cxx just generated.
///
/// The tests own a criterion that is a claim about C++ (chem-perceive-12 ac-001:
/// a chemistry error must arrive on the C++ side as a catchable `rust::Error`,
/// and must not abort the process), so a C++ translation unit has to exist for a
/// Rust test to observe it. This is that unit's build step; it is test support
/// only, and nothing in `src/` calls into it.
///
/// **The compile is allowed to fail.** On failure the probe is simply absent, the
/// `cxx_probe` cfg is not set, and the tests that need it fail with an
/// explanation (`tests/am1bcc_bridge.rs`). The alternative — letting a bad probe
/// abort the build script — would take the whole workspace down with it: no crate
/// would compile and no other test could run, which is the wrong failure for a
/// test fixture that has drifted from the bridge.
fn compile_test_probe(out_dir: &Path) {
    println!("cargo::rustc-check-cfg=cfg(cxx_probe)");

    let probe = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("tests")
        .join("cxx")
        .join("bridge_probe.cc");
    println!("cargo::rerun-if-changed={}", probe.display());
    if !probe.is_file() {
        return; // e.g. a packaged crate built without its tests
    }

    // cxx nests the generated header under a path derived from the (absolute)
    // bridge source path, so the include dir is found rather than reconstructed.
    let include_root = out_dir.join("cxxbridge").join("include");
    let Some(header) = find_file(&include_root, "bridge.rs.h") else {
        println!(
            "cargo::warning=molrs-cxxapi: generated bridge.rs.h not found; C++ test probe skipped"
        );
        return;
    };

    // `cargo_metadata(false)`: this archive must NOT become a `-l` for every
    // target of the crate. The crate's headline artifact is a `staticlib`, and
    // rustc bundles native static libs into it — the test probe would be shipped
    // inside the very library Atomiverse links against. Instead the archive is
    // handed to the linker of the TEST targets only, below.
    const LIB: &str = "molrs_cxxapi_test_probe";
    let result = cc::Build::new()
        .cpp(true)
        .std("c++20")
        .cargo_metadata(false)
        .include(header.parent().expect("the header has a parent dir"))
        .include(&include_root)
        .file(&probe)
        .try_compile(LIB);

    match result {
        Ok(()) => {
            // The C++ stdlib is already linked for every target by the cxx_build
            // compile above, so the archive alone is enough here.
            println!(
                "cargo::rustc-link-arg-tests={}",
                out_dir.join(format!("lib{LIB}.a")).display()
            );
            println!("cargo::rustc-cfg=cxx_probe");
        }
        Err(err) => {
            println!(
                "cargo::warning=molrs-cxxapi: the C++ test probe did not compile against the \
                 generated bridge header ({err}). The AM1-BCC bridge tests will fail with an \
                 explanation; rebuild with `-vv` for the compiler diagnostic."
            );
        }
    }
}

/// The first file named `name` anywhere under `dir`.
fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut dirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
        } else if path.file_name().and_then(|f| f.to_str()) == Some(name) {
            return Some(path);
        }
    }
    dirs.iter().find_map(|d| find_file(d, name))
}
