# molrs-pack Behavior Alignment with Packmol

## Scope
- Alignment target: `/Users/roykid/work/packmol` source behavior.
- Practical acceptance scope: 5 production-size examples:
`pack_mixture`, `pack_bilayer`, `pack_interface`, `pack_solvprotein`, `pack_spherical`.

## Packmol Behaviors Matched
- Objective structure:
  - Geometric restraint penalties (`comprest`/`gwalls` equivalent).
  - Minimum-distance conflict penalty (`computef`/`fparc` equivalent).
- Optimization workflow:
  - Initialization with constraint-only fitting (`initial`/`restmol`/`swaptype` flow).
  - Main phased optimization (per-type then all-types).
  - `movebad` heuristic and radius scaling schedule.
  - GENCAN/SPG/CG loop and precision gate (`fdist`/`frest` style convergence).
- Constraint semantics used by the 5 examples:
  - `inside box`
  - `inside sphere`
  - `outside sphere`
  - `above/below plane`
  - fixed molecule placement
- Determinism:
  - Explicit seed support; same seed used for Packmol and molrs-pack comparison runs.

## How Alignment is Verified

### 1. Batch Example Validation
- Test file: `molrs-pack/tests/examples_batch.rs`
- Runs all 5 examples (expensive test, marked `#[ignore]`).
- Validates:
  - atom count consistency with expanded target specs
  - molecule-count consistency
  - XYZ output format sanity (first-line atom count)
  - quantified violation metrics under tolerance/precision

### 2. Packmol vs molrs-pack Timing + Metrics
- Command:
```bash
cargo run -p molrs-pack --release --bin compare_examples
```
- Output tables:
  - `example | packmol_time_s | molrs-pack_time_s | ratio`
  - violation metrics table for both tools:
    - `max_distance_violation`
    - `max_constraint_penalty`
    - `violating_pairs`
    - `violating_atoms`
- Tool exits non-zero if:
  - any side fails validation
  - any example ratio is `> 1.5`

## Unavoidable Differences
- Coordinates are not expected to be byte-identical to Packmol even with same seed.
- Acceptance basis is functional equivalence:
  - same constraints enforced
  - same conflict criteria class
  - same tolerance/precision magnitude
  - quantified violation metrics remain valid and comparable
