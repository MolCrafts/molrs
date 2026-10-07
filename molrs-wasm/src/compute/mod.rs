//! WASM bindings for trajectory analysis — the face of `molrs::compute`.
//!
//! One file per analysis family, mirroring the Rust owner (`rdf`, `msd`,
//! `cluster`, `shape`, `transport`, `spectroscopy`, …), plus [`catalog`], the
//! table a downstream UI reads to present and dispatch them. Neighbor search is
//! not analysis: [`NeighborList`](crate::core::NeighborList) and the
//! [`Neighbors`](crate::core::Neighbors) table it produces live in
//! `core`, as in Rust.
//!
//! The classes here are freud-style: configure once, then call `compute` on
//! [`Frame`](crate::core::Frame)s.
//!
//! # Example (JavaScript)
//!
//! ```js
//! // Build the pair table
//! const nl = new NeighborList(5.0);
//! nl.build(frame);
//! const nlist = nl.neighbors();
//!
//! // Cluster on it
//! const result = new Cluster(1).compute(frame, nlist);
//!
//! // RDF streams its own neighbor search
//! const gr = new RDF(100, 5.0).compute(frame);
//! console.log(gr.binCenters(), gr.rdf());
//!
//! // MSD needs no neighbor table
//! const msd = new MSD();
//! for (const frame of trajectory) {
//!     msd.feed(frame);
//! }
//! console.log(msd.results()[1].mean); // MSD at frame 1 in Å²
//! ```
//!
//! # References
//!
//! - Ramasubramani, V. et al. (2020). freud: A software suite for
//!   high throughput analysis of particle simulation data. *Computer
//!   Physics Communications*, 254, 107275.

use molrs::op::F;
use ndarray::{Array1, Array2};
use serde::Serialize;
use wasm_bindgen::prelude::*;

mod catalog;
mod cluster;
mod density;
mod dielectric;
mod diffraction;
mod distribution;
mod dynamics;
mod environment;
mod fitting;
mod hbond;
mod ml;
mod msd;
mod order;
mod pmft;
mod rdf;
mod shape;
mod spectroscopy;
mod transport;
#[cfg(feature = "voronoi")]
mod voronoi;

pub use catalog::*;
pub use cluster::*;
pub use density::*;
pub use dielectric::*;
pub use diffraction::*;
pub use distribution::*;
pub use dynamics::*;
pub use environment::*;
pub use fitting::*;
pub use hbond::*;
pub use ml::*;
pub use msd::*;
pub use order::*;
pub use pmft::*;
pub use rdf::*;
pub use shape::*;
pub use spectroscopy::*;
pub use transport::*;
#[cfg(feature = "voronoi")]
pub use voronoi::*;

// ---------------------------------------------------------------------------
// Wire helpers shared by several families
// ---------------------------------------------------------------------------

fn js_value<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value)
        .map_err(|e| JsValue::from_str(&format!("serialize wasm result: {e}")))
}

fn array1(data: &[F]) -> Array1<F> {
    Array1::from_vec(data.to_vec())
}

fn array2(data: &[F], rows: usize, cols: usize, name: &str) -> Result<Array2<F>, JsValue> {
    if data.len() != rows * cols {
        return Err(JsValue::from_str(&format!(
            "{name}: data length {} != rows * cols = {} * {}",
            data.len(),
            rows,
            cols
        )));
    }
    Array2::from_shape_vec((rows, cols), data.to_vec())
        .map_err(|e| JsValue::from_str(&format!("{name}: {e}")))
}

fn quats(data: &[F], name: &str) -> Result<Vec<[F; 4]>, JsValue> {
    if !data.len().is_multiple_of(4) {
        return Err(JsValue::from_str(&format!(
            "{name}: expected flat [w,x,y,z,...] length divisible by 4"
        )));
    }
    Ok(data.as_chunks::<4>().0.to_vec())
}

fn u32_pairs(data: &[u32], name: &str) -> Result<Vec<(u32, u32)>, JsValue> {
    if !data.len().is_multiple_of(2) {
        return Err(JsValue::from_str(&format!("{name}: expected pairs")));
    }
    Ok(data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| (p[0], p[1]))
        .collect())
}

fn usize_pairs(data: &[u32], name: &str) -> Result<Vec<(usize, usize)>, JsValue> {
    Ok(u32_pairs(data, name)?
        .into_iter()
        .map(|(a, b)| (a as usize, b as usize))
        .collect())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SeriesOut {
    lag_times: Vec<F>,
    values: Vec<F>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Grid2Out {
    data: Vec<F>,
    shape: [usize; 2],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Grid3Out<T: Serialize> {
    data: Vec<T>,
    shape: [usize; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::NeighborList;
    use crate::core::frame::Frame;
    use wasm_bindgen_test::*;

    /// Helper: create a Frame with N particles at given positions + cubic simbox.
    fn make_frame(positions: &[[F; 3]], box_len: F) -> Frame {
        use molrs::core::Block;
        use molrs::core::SimBox;
        use ndarray::{Array1, array};

        let x = Array1::from_iter(positions.iter().map(|p| p[0]));
        let y = Array1::from_iter(positions.iter().map(|p| p[1]));
        let z = Array1::from_iter(positions.iter().map(|p| p[2]));

        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();

        let mut rs_frame = molrs::core::Frame::new();
        rs_frame.insert("atoms", block);
        rs_frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0, 0.0], [false, false, false]).unwrap());

        Frame::from_rs(rs_frame).unwrap()
    }

    /// Half-shell pairs within `cutoff`, both columns kept.
    fn neighbors(frame: &Frame, cutoff: F) -> crate::core::Neighbors {
        let mut nl = NeighborList::new(cutoff).unwrap();
        nl.build(frame).unwrap();
        nl.neighbors(None).unwrap()
    }

    #[wasm_bindgen_test]
    fn rdf_runs() {
        let positions: Vec<[F; 3]> = (0..50)
            .map(|i| {
                let v = i as F * 0.2;
                [v % 10.0, (v * 1.3) % 10.0, (v * 1.7) % 10.0]
            })
            .collect();
        let frame = make_frame(&positions, 10.0);

        // Streaming path: no NeighborList materialization.
        let rdf = Rdf::new(20, 4.0, None, None).unwrap();
        let result = rdf.compute(&frame).unwrap();

        assert_eq!(result.bin_centers().length(), 20);
        assert_eq!(result.rdf().length(), 20);
        assert_eq!(result.bin_edges().length(), 21);
    }

    #[wasm_bindgen_test]
    fn msd_feed_trajectory() {
        let ref_pos = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let cur_pos = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let ref_frame = make_frame(&ref_pos, 20.0);
        let cur_frame = make_frame(&cur_pos, 20.0);

        let mut msd = Msd::new();
        msd.feed(&ref_frame).unwrap();
        msd.feed(&cur_frame).unwrap();

        assert_eq!(msd.count(), 2);
        let results = msd.results().unwrap();
        assert_eq!(results.len(), 2);

        // frame 0 vs itself = 0
        assert!(results[0].mean() < 1e-6);

        // frame 1: particle 0: d^2 = 1, particle 1: d^2 = 4, mean = 2.5
        assert!((results[1].mean() - 2.5).abs() < 1e-5);
    }

    #[wasm_bindgen_test]
    fn cluster_two_groups() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [8.0, 8.0, 8.0],
            [8.5, 8.0, 8.0],
        ];
        let frame = make_frame(&positions, 20.0);
        let nbrs = neighbors(&frame, 2.0);

        let cluster = Cluster::new(1);
        let result = cluster.compute(&frame, &nbrs).unwrap();

        assert_eq!(result.num_clusters(), 2);
        let idx = result.cluster_idx();
        assert_eq!(idx.len(), 4);
        assert_eq!(idx[0], idx[1]);
        assert_eq!(idx[2], idx[3]);
        assert_ne!(idx[0], idx[2]);
    }

    #[wasm_bindgen_test]
    fn cluster_min_size_filters() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [8.0, 8.0, 8.0], // isolated
        ];
        let frame = make_frame(&positions, 20.0);
        let nbrs = neighbors(&frame, 2.0);

        let cluster = Cluster::new(2);
        let result = cluster.compute(&frame, &nbrs).unwrap();

        assert_eq!(result.num_clusters(), 1);
        let idx = result.cluster_idx();
        assert_eq!(idx[2], -1); // filtered out
        assert!(idx[0] >= 0);
    }
}
