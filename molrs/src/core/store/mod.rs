//! Columnar data containers: [`Block`] column store,
//! [`Frame`] hierarchical container, the
//! [`Trajectory`] frame-sequence carrier, the
//! [`MolRec`] record aggregate, canonical column keys, and
//! type-name construction and the type-id contract ([`type_labels`]).

mod block;
pub(crate) mod forcefield_section;
mod frame;
mod frame_access;
mod frame_view;
pub mod keys;
pub(crate) mod meta;
pub mod precision;
mod record;
pub mod schema;
mod trajectory;
pub mod type_labels;
pub mod typed_json;

pub use block::{
    Block, BlockAccess, BlockDtype, BlockError, BlockView, Column, ColumnHolder, ColumnView, DType,
};
pub use forcefield_section::{EndpointKey, ForceFieldSection, StyleEntry, style_block_name};
pub use frame::Frame;
pub use frame_access::FrameAccess;
pub use frame_view::FrameView;
pub use meta::{MetaIter, MetaMap, MetaValue};
pub use record::{MOLREC_VERSION, MolRec, Observables, RESERVED_META_KEYS};
pub use trajectory::{ObservableData, ObservableKind, ObservableRecord, Trajectory};
