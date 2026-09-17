//! Core regression benchmarks — one module per `src/core/` type.
//!
//! Each holds one small representative input per operation: these exist to
//! catch a regression, not to draw a scaling curve (`.claude/notes/performance.md`).

mod frame;
mod graph;
mod helpers;
mod neighbor_list;
mod region;
mod simbox;
mod topology;

use criterion::criterion_main;

criterion_main!(
    frame::benches,
    graph::benches,
    topology::benches,
    simbox::benches,
    region::benches,
    neighbor_list::benches,
);
