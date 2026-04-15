//! 3D coordinate generation (distance geometry + force-field refinement).
//!
//! Generates realistic 3D molecular geometries from a molecular graph
//! (2D connectivity). The pipeline uses distance geometry embedding
//! followed by MMFF94-based energy minimization.
//!
//! # Pipeline stages
//!
//! 1. **Distance geometry** -- embed atoms in 3D using bounds-matrix
//!    smoothing and random distance sampling.
//! 2. **Fragment assembly** -- overlay known fragment templates.
//! 3. **Coarse minimization** -- quick energy minimization.
//! 4. **Rotor search** -- systematic search over rotatable bonds.
//! 5. **Final minimization** -- full energy minimization.
//! 6. **Stereo guards** -- verify stereochemistry is preserved.
//!
//! # References
//!
//! - Halgren, T.A. (1996). Merck Molecular Force Field (MMFF94).
//!   *J. Comput. Chem.*, 17(5-6), 490-519.

use wasm_bindgen::prelude::*;

use molrs::atomistic::Atomistic;
use molrs_embed::{EmbedOptions, EmbedSpeed, generate_3d};
use molrs::molgraph::MolGraph;

use crate::core::frame::Frame;

/// Generate 3D coordinates for a molecular [`Frame`].
///
/// The input frame must have an `"atoms"` block with a `"element"`
/// string column (element symbols like `"C"`, `"N"`, `"O"`). A
/// `"bonds"` block with `i`, `j` (u32) and `order` (F) columns
/// is required for correct geometry.
///
/// Returns a **new** [`Frame`] with 3D coordinates added as `x`, `y`,
/// `z` (F, angstrom) columns in the `"atoms"` block.
///
/// # Arguments
///
/// * `frame` - Input molecular frame with atoms and bonds (from
///   [`parseSMILES`](crate::parse_smiles) or file readers)
/// * `speed` - Quality/speed preset:
///   - `"fast"` -- minimal refinement, suitable for visualization
///   - `"medium"` (default) -- balanced quality/speed
///   - `"better"` -- thorough conformer search, best geometry
/// * `seed` - Optional RNG seed (`u32`) for reproducibility. If
///   omitted, a random seed is used.
///
/// # Returns
///
/// A new [`Frame`] with 3D coordinates. The original frame is
/// not modified.
///
/// # Errors
///
/// Throws a `JsValue` string if:
/// - The frame has no `"atoms"` block or is missing required columns
/// - The molecular graph has invalid valences or topology
/// - The 3D embedding fails to converge
///
/// # Example (JavaScript)
///
/// ```js
/// const ir = parseSMILES("c1ccccc1"); // benzene
/// const frame2d = ir.toFrame();
/// const frame3d = generate3D(frame2d, "fast", 42);
///
/// const atoms = frame3d.getBlock("atoms");
/// const x = atoms.copyColF("x"); // Float32Array or Float64Array with 3D x-coords
/// const y = atoms.copyColF("y");
/// const z = atoms.copyColF("z");
/// ```
#[wasm_bindgen(js_name = generate3D)]
pub fn generate_3d_wasm(
    frame: &Frame,
    speed: Option<String>,
    seed: Option<u32>,
) -> Result<Frame, JsValue> {
    let opts = parse_opts(speed.as_deref(), seed)?;
    let atomistic = frame.with_frame(|rs_frame| {
        let molgraph = MolGraph::from_frame(rs_frame)
            .map_err(|e| JsValue::from_str(&format!("Frame → MolGraph: {e}")))?;
        Atomistic::try_from_molgraph(molgraph)
            .map_err(|e| JsValue::from_str(&format!("MolGraph → Atomistic: {e}")))
    })?;

    let (result, _report) =
        generate_3d(&atomistic, &opts).map_err(|e| JsValue::from_str(&format!("embed: {e}")))?;

    Frame::from_rs(result.to_frame())
}

fn parse_opts(speed: Option<&str>, seed: Option<u32>) -> Result<EmbedOptions, JsValue> {
    let sp = match speed.unwrap_or("medium") {
        "fast" => EmbedSpeed::Fast,
        "medium" => EmbedSpeed::Medium,
        "better" => EmbedSpeed::Better,
        other => {
            return Err(JsValue::from_str(&format!(
                "unknown speed '{other}', expected 'fast', 'medium', or 'better'"
            )));
        }
    };
    Ok(EmbedOptions {
        speed: sp,
        rng_seed: seed.map(u64::from),
        ..EmbedOptions::default()
    })
}
