//! Surface-mesh file formats.
//!
//! The odd module out in [`crate::io`]: [`data`](crate::io::data) and
//! [`trajectory`](crate::io::trajectory) read atoms and hand back a
//! [`Frame`](crate::store::frame::Frame), while these files carry no atoms at
//! all. They read into a [`TriMesh`](crate::spatial::TriMesh) — the container a
//! packing run is confined to, the geometry a viewer paints around it.
//!
//! Currently:
//! - [`stl`] — STL, both the ASCII and the binary shape

pub mod stl;

pub use stl::{parse_stl, read_stl};
