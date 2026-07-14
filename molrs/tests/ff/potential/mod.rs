//! Integration tests for the `potential` module tree.
//!
//! Mirrors `src/potential/`. Each leaf builds inputs in code, evaluates the
//! public `Potential` API, and checks energy / forces against analytical
//! values plus finite-difference gradients and Newton's third law.

#[path = "geometry.rs"]
mod geometry;

#[path = "angle.rs"]
mod angle;
#[path = "bond.rs"]
mod bond;
#[path = "mmff.rs"]
mod mmff;
#[path = "opls.rs"]
mod opls;

/// mmff-orthogonal-02 ac-003 / ac-004 — what `ParamSource` relaxes (per-instance
/// styles compile from zero type rows) and what it must NOT relax (a `TypeRows`
/// style with no type defs still errors). RED until `ParamSource` lands.
#[path = "param_source.rs"]
mod param_source;

#[path = "pair.rs"]
mod pair;
/// mmff-orthogonal-02 ac-002 — "a registered kernel constructor that ignores `tp`
/// is not a Style", asserted in both directions off the source. RED until every
/// tp-ignoring kernel is registered `ParamSource::PerInstance`.
#[path = "param_source_gate.rs"]
mod param_source_gate;
#[path = "pme.rs"]
mod pme;
