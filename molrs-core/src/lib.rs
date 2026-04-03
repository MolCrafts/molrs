//! # molrs
//!
//! A Rust library providing core molecular modeling functionality.
//!
//! ## Features
//!
//! - Element data and lookup by atomic number or symbol
//! - Core data structures, geometry, IO, and neighbor-list algorithms
//! - Type-safe molecular representations
//!
//! ## Examples
//!
//! ### Element lookup
//!
//! ```
//! use molrs::Element;
//!
//! // Look up elements by atomic number
//! let hydrogen = Element::by_number(1).unwrap();
//! assert_eq!(hydrogen.symbol(), "H");
//!
//! // Or by symbol (case-insensitive)
//! let h = Element::by_symbol("h").unwrap();
//! assert_eq!(h.name(), "Hydrogen");
//! ```
//!
//! ### Packing (moved to `molrs-pack`)
//!
//! ```ignore
//! use molrs_pack::Molpack;
//!
//! let mut packer = Molpack::new(None).with_precision(1e-3);
//! let _ = &mut packer;
//! ```

#![allow(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

// Embedded data files
pub mod data;

// Modules formerly in core/
pub mod atomistic;
pub mod block;
pub mod coarsegrain;
pub mod element;
pub mod forcefield;
pub mod grid;
pub mod frame;
pub mod gasteiger;
pub mod gen3d;
pub mod hydrogens;
pub mod mapping;
pub mod molgraph;
pub mod molrec;
pub mod potential;
pub mod region;
pub mod rings;
pub mod rotatable;
pub mod stereo;
pub mod topology;
pub mod types;
pub mod typifier;

// Analysis compute modules
pub mod compute;

// Other top-level modules
pub mod error;
pub mod io;
pub mod math;
pub mod neighbors;
pub mod smiles;

// Public re-exports for common types
pub use atomistic::Atomistic;
pub use block::Block;
pub use coarsegrain::CoarseGrain;
pub use element::Element;
pub use error::MolRsError;
pub use frame::Frame;
pub use grid::Grid;
pub use gasteiger::{GasteigerCharges, compute_gasteiger_charges};
pub use gen3d::{
    EmbedAlgorithm, ForceFieldKind, Gen3DOptions, Gen3DReport, Gen3DSpeed, StageKind, StageReport,
    generate_3d,
};
pub use hydrogens::{add_hydrogens, implicit_h_count, remove_hydrogens};
pub use mapping::{CGMapping, WeightScheme};
pub use molgraph::{Atom, AtomId, Bead, Bond, BondId, MolGraph, PropValue};
pub use rings::{RingInfo, find_rings};
pub use stereo::{
    BondStereo, TetrahedralStereo, assign_bond_stereo_from_3d, assign_stereo_from_3d,
    chiral_volume, find_chiral_centers,
};
pub use topology::{Topology, TopologyRingInfo};

// Typifier re-exports
pub use typifier::Typifier;
pub use typifier::mmff::{MMFFAtomProp, MMFFEquiv, MMFFParams, MMFFTypifier};

// SMILES/SMARTS re-exports
pub use smiles::{parse_smarts, parse_smiles, to_atomistic};

// Re-export IO functions for convenience
pub use forcefield::xml::{read_forcefield_xml, read_forcefield_xml_str};
pub use io::chgcar::read_chgcar;
pub use io::lammps_data::{read_lammps_data, write_lammps_data};
pub use io::lammps_dump::{open_lammps_dump, read_lammps_dump, write_lammps_dump};
pub use io::pdb::{read_pdb_frame, write_pdb_frame};
pub use io::xyz::{read_xyz_frame, read_xyz_traj, write_xyz_frame};
pub use molrec::{
    CellBox, MolRec, ObservableData, ObservableKind, ObservableRecord, RecordBox, SchemaValue,
    Trajectory,
};

// Zarr I/O re-exports (only with "zarr" feature)
#[cfg(feature = "zarr")]
pub use io::zarr::{
    count_molrec_frames_in_store, read_molrec_frame_from_store, read_molrec_store,
};
#[cfg(all(feature = "zarr", feature = "filesystem"))]
pub use io::zarr::read_molrec_file;
#[cfg(all(feature = "zarr", feature = "filesystem"))]
pub use io::zarr::write_molrec_file;
