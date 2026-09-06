//! Surface-mesh readers for the WASM API.
//!
//! | Function | Format |
//! |----------|--------|
//! | [`readSTL`](read_stl_export) | STL, ASCII or binary (auto-detected) |
//!
//! Unlike the readers in [`reader`](super::reader), these produce a
//! [`Mesh`] rather than a `Frame`: an STL carries triangles, not atoms.

use crate::core::mesh::Mesh;
use molrs::io::mesh::parse_stl;
use wasm_bindgen::prelude::*;

/// Read an STL file's bytes into a [`Mesh`].
///
/// Both shapes of the format are accepted and told apart by length, not by
/// the leading keyword — a binary STL's 80-byte header is free text and
/// routinely starts with `solid` too.
///
/// # Arguments
///
/// * `bytes` - The full binary content of an STL file. An internal copy is
///   taken, so the caller may discard the buffer as soon as this returns.
///
/// # Errors
///
/// Throws a `JsValue` string when the bytes are neither a length-matched
/// binary STL nor readable ASCII. A file with no facets is **not** an error:
/// it reads as a mesh with zero faces, and whether that is acceptable is the
/// caller's decision.
///
/// # Example (JavaScript)
///
/// ```js
/// const mesh = readSTL(new Uint8Array(await file.arrayBuffer()));
/// console.log(`${mesh.nFaces()} triangles, watertight: ${mesh.isWatertight()}`);
/// ```
#[wasm_bindgen(js_name = readSTL)]
pub fn read_stl_export(bytes: &[u8]) -> Result<Mesh, JsValue> {
    parse_stl(bytes)
        .map(Mesh::new)
        .map_err(|e| JsValue::from_str(&format!("STL reading error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    /// One ASCII facet, wound so the normal is +z and the recorded one is not
    /// consulted at all (the writer left it at zero, as molpack's does).
    const TRIANGLE: &str = "solid one\n\
          facet normal 0 0 0\n\
            outer loop\n\
              vertex 0 0 0\n\
              vertex 2 0 0\n\
              vertex 0 2 0\n\
            endloop\n\
          endfacet\n\
        endsolid one\n";

    #[wasm_bindgen_test]
    fn reads_a_mesh_for_javascript() {
        let mesh = read_stl_export(TRIANGLE.as_bytes()).unwrap();
        assert_eq!(mesh.n_faces(), 1);
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.faces(), vec![0, 1, 2]);
        assert_eq!(mesh.face_normals_f32(), vec![0.0, 0.0, 1.0]);
        assert_eq!(mesh.aabb(), vec![0.0, 0.0, 0.0, 2.0, 2.0, 0.0]);
        assert!(!mesh.is_watertight());
    }

    #[wasm_bindgen_test]
    fn surfaces_a_parse_failure_as_a_js_string() {
        assert!(read_stl_export(b"ITEM: TIMESTEP\n0\n").is_err());
    }
}
