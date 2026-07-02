//! Columnar data containers: [`Block`](block::Block) column store,
//! [`Frame`](frame::Frame) hierarchical container, the
//! [`Trajectory`](trajectory::Trajectory) frame-sequence carrier, and canonical
//! column keys.

pub mod block;
pub mod frame;
pub mod frame_access;
pub mod frame_view;
pub mod keys;
pub mod trajectory;
