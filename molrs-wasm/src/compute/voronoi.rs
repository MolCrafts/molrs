//! Radical (Laguerre) Voronoi and its domain / void consumers — WASM face
//! of the `molrs::compute` voronoi family.

use super::js_value;
use crate::core::frame::Frame;
use crate::core::frame::positions_from_frame;
use molrs::op::types::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Per-atom covalent radii from the frame's `element` column.
fn covalent_radii_from_frame(frame: &molrs::core::Frame) -> Result<Vec<F>, JsValue> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| JsValue::from_str("Frame has no 'atoms' block"))?;
    let column = atoms
        .get("element")
        .and_then(|c| c.as_string())
        .ok_or_else(|| {
            JsValue::from_str(
                "radii weighting needs a string 'element' column on the atoms block; \
             construct with useAtomRadii = false for a plain Voronoi diagram",
            )
        })?;
    column
        .iter()
        .map(|symbol| {
            molrs::core::Element::by_symbol(symbol)
                .map(|el| F::from(el.covalent_radius()))
                .ok_or_else(|| JsValue::from_str(&format!("unknown element symbol {symbol}")))
        })
        .collect()
}

/// Tessellate a frame and report the box volume used for normalization.
///
/// With `use_atom_radii` the cells are Laguerre-weighted by covalent radius;
/// without it every generator has radius zero, which is a plain Voronoi diagram.
fn voronoi_cells(
    frame: &molrs::core::Frame,
    use_atom_radii: bool,
) -> Result<(molrs::compute::VoronoiCells, F), JsValue> {
    let positions = positions_from_frame(frame)?;
    let n = positions.nrows();
    let simbox = frame.simbox.as_ref().ok_or_else(|| {
        JsValue::from_str("Radical Voronoi needs a periodic simulation box (frame.simbox is unset)")
    })?;
    let radii = if use_atom_radii {
        covalent_radii_from_frame(frame)?
    } else {
        vec![0.0; n]
    };
    if radii.len() != n {
        return Err(JsValue::from_str(
            "Radical Voronoi: the element column length does not match the atom count",
        ));
    }
    let cells = molrs::compute::RadicalVoronoi
        .build(positions.view(), &radii, simbox)
        .map_err(|e| JsValue::from_str(&format!("RadicalVoronoi: {e}")))?;
    Ok((cells, simbox.volume()))
}

#[wasm_bindgen(js_name = RadicalVoronoi)]
pub struct RadicalVoronoi {
    use_atom_radii: bool,
}

#[wasm_bindgen(js_class = RadicalVoronoi)]
impl RadicalVoronoi {
    /// With `use_atom_radii` the tessellation is Laguerre-weighted by each
    /// atom's covalent radius, read from the frame's `element` column.
    #[wasm_bindgen(constructor)]
    pub fn new(use_atom_radii: bool) -> Self {
        Self { use_atom_radii }
    }

    /// Cell volumes plus the face graph. `faceNeighbors` / `faceAreas` are the
    /// concatenation of every cell's faces; cell `i` owns the slice
    /// `[faceOffsets[i], faceOffsets[i + 1])`. A negative neighbour id is a box
    /// boundary rather than another cell.
    pub fn compute(&self, frame: &Frame) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            volumes: Vec<F>,
            total_volume: F,
            box_volume: F,
            face_neighbors: Vec<i64>,
            face_areas: Vec<F>,
            face_offsets: Vec<usize>,
        }
        frame.with_frame(|rs_frame| {
            let (cells, box_volume) = voronoi_cells(rs_frame, self.use_atom_radii)?;
            let mut face_neighbors = Vec::new();
            let mut face_areas = Vec::new();
            let mut face_offsets = Vec::with_capacity(cells.len() + 1);
            face_offsets.push(0);
            for faces in &cells.faces {
                for face in faces {
                    face_neighbors.push(face.neighbor);
                    face_areas.push(face.area);
                }
                face_offsets.push(face_neighbors.len());
            }
            js_value(&Out {
                total_volume: cells.total_volume(),
                volumes: cells.volumes.clone(),
                box_volume,
                face_neighbors,
                face_areas,
                face_offsets,
            })
        })
    }
}

#[wasm_bindgen(js_name = VoronoiDomainAnalysis)]
pub struct VoronoiDomainAnalysis {
    use_atom_radii: bool,
}

#[wasm_bindgen(js_class = VoronoiDomainAnalysis)]
impl VoronoiDomainAnalysis {
    #[wasm_bindgen(constructor)]
    pub fn new(use_atom_radii: bool) -> Self {
        Self { use_atom_radii }
    }

    /// Merge face-adjacent cells that share a `labels` value into domains.
    pub fn compute(&self, frame: &Frame, labels: &[i32]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            sizes: Vec<usize>,
            count: usize,
            largest_fraction: F,
            domain_of: Vec<usize>,
        }
        frame.with_frame(|rs_frame| {
            let (cells, _) = voronoi_cells(rs_frame, self.use_atom_radii)?;
            if labels.len() != cells.len() {
                return Err(JsValue::from_str(
                    "VoronoiDomainAnalysis: labels must have one entry per atom",
                ));
            }
            let labels: Vec<i64> = labels.iter().map(|&v| i64::from(v)).collect();
            let r = molrs::compute::DomainAnalysis
                .analyze(&cells, &labels)
                .map_err(|e| JsValue::from_str(&format!("VoronoiDomainAnalysis: {e}")))?;
            js_value(&Out {
                sizes: r.sizes,
                count: r.count,
                largest_fraction: r.largest_fraction,
                domain_of: r.domain_of,
            })
        })
    }
}

#[wasm_bindgen(js_name = VoronoiVoidAnalysis)]
pub struct VoronoiVoidAnalysis {
    use_atom_radii: bool,
    box_volume: Option<F>,
}

#[wasm_bindgen(js_class = VoronoiVoidAnalysis)]
impl VoronoiVoidAnalysis {
    /// `box_volume` overrides the frame's box volume when normalizing the void
    /// fraction; pass `null` to use the frame's own box.
    #[wasm_bindgen(constructor)]
    pub fn new(use_atom_radii: bool, box_volume: Option<F>) -> Self {
        Self {
            use_atom_radii,
            box_volume,
        }
    }

    /// A non-zero `isVoid[i]` marks cell `i` a void probe; adjacent probe cells
    /// merge into one cavity. `boxVolume` defaults to the frame's box volume.
    pub fn compute(&self, frame: &Frame, is_void: &[u8]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            cavity_volumes: Vec<F>,
            total_void_volume: F,
            void_fraction: F,
        }
        frame.with_frame(|rs_frame| {
            let (cells, frame_volume) = voronoi_cells(rs_frame, self.use_atom_radii)?;
            if is_void.len() != cells.len() {
                return Err(JsValue::from_str(
                    "VoronoiVoidAnalysis: isVoid must have one entry per atom",
                ));
            }
            let mask: Vec<bool> = is_void.iter().map(|&v| v != 0).collect();
            let r = molrs::compute::VoidAnalysis
                .analyze(&cells, &mask, self.box_volume.unwrap_or(frame_volume))
                .map_err(|e| JsValue::from_str(&format!("VoronoiVoidAnalysis: {e}")))?;
            js_value(&Out {
                cavity_volumes: r.cavity_volumes,
                total_void_volume: r.total_void_volume,
                void_fraction: r.void_fraction,
            })
        })
    }
}
