//! Spatial region types for the WASM API.
//!
//! [`shapes`] holds the nine solids — the same names the Python binder uses —
//! and the boolean composition between them. [`simbox::Box`] is the periodic
//! cell, which lives here for historical reasons rather than because it is a
//! region.

pub mod shapes;
pub mod simbox;
