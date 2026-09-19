//! MD regression benchmarks — the force path, one shape per thing that can
//! regress differently.
//!
//! These are **regression** benchmarks in the sense `.claude/notes/performance.md`
//! means: one small representative input each, to notice a change, not to draw
//! a scaling curve. They exist because the whole force path — two providers,
//! nine pair kernels, the halo and its pair table — shipped without one, and
//! the first time any of it was measured it turned out that 94% of a ghost step
//! was rediscovering a pair list that had not changed.
//!
//! The four step shapes are the four things that fail differently:
//!
//! * `mic/plain` — the minimum image with nothing to weight.
//! * `mic/weighted` — the same with special-bonds weights, the path that used
//!   to rebuild the pair table once per distinct weight.
//! * `ghost/plain` — copies, no bonded terms.
//! * `ghost/bonded_fold` — copies, bonded terms, and an atom crossing a face.
//!   Without the fold the bonded re-resolution never runs and the case that was
//!   O(N^5/3) is invisible.

mod pair;
mod step;

use criterion::criterion_main;

criterion_main!(step::benches, pair::benches);
