//! Columnar data containers: [`Block`](block::Block) column store,
//! [`Frame`](frame::Frame) hierarchical container, the
//! [`Trajectory`](trajectory::Trajectory) frame-sequence carrier, the
//! [`MolRec`](record::MolRec) record aggregate, canonical column keys, and
//! type-name construction and the type-id contract ([`type_labels`]).

pub mod block;
pub mod forcefield_section;
pub mod frame;
pub mod frame_access;
pub mod frame_view;
pub mod keys;
pub mod meta;
pub mod precision;
pub mod record;
pub mod schema;
pub mod trajectory;
pub mod type_labels;
pub mod typed_json;
