# Architecture

Developer-oriented view of the crate's structure, data flow, and
internal invariants. Read [`concepts`](crate::concepts) first for the
abstractions this chapter assumes.

## Module map

```text
molrs-pack/
├── src/
│   ├── lib.rs              — public re-exports + doc modules
│   ├── packer.rs           — Molpack builder + pack() driver + phase loop
│   ├── target.rs           — Target = one molecule type + its restraints/relaxers
│   ├── restraint.rs        — Restraint trait + 14 concrete *Restraint structs
│   ├── region.rs           — Region trait + And/Or/Not + FromRegion
│   ├── relaxer.rs          — Relaxer/RelaxerRunner + TorsionMcRelaxer
│   ├── handler.rs          — Handler trait + 4 built-in observers
│   ├── objective.rs        — compute_f/g/fg + Objective trait
│   ├── context/            — PackContext (single owner of mutable state)
│   ├── constraints/        — EvalMode/EvalOutput + Constraints ZST facade
│   ├── gencan/             — GENCAN optimizer (pgencan/tn_ls/spg/cg)
│   ├── initial.rs          — initial placement + swap state
│   ├── movebad.rs          — movebad heuristic (perturb worst molecules)
│   ├── euler.rs            — Euler angle ↔ rotation matrix
│   ├── cell.rs             — cell-list indexing
│   ├── frame.rs            — PackContext ↔ molrs::Frame
│   ├── validation.rs       — post-pack correctness check
│   ├── cases.rs            — ExampleCase enum for 5 canonical examples
│   ├── numerics.rs, error.rs, random.rs
│   └── api/                — builder-style facade (thin)
├── tests/                  — 7 integration test files
├── benches/                — 5 criterion benches
├── examples/               — 5 canonical + 2 ad-hoc demos
└── docs/                   — this documentation
```

## Module dependency graph

```text
                    ┌───────┐
                    │ lib.rs│  (public re-exports + doc landing page)
                    └───┬───┘
                        │
                    ┌───┴────────────────────────────────┐
                    │        packer.rs  (driver)         │
                    │   Molpack, pack(), run_phase,      │
                    │   run_iteration, evaluate_unscaled │
                    └─┬────┬─────┬──────┬──────┬─────────┘
                      │    │     │      │      │
            ┌─────────┘    │     │      │      └──────────┐
            │          ┌───┘     │      └───┐             │
            ▼          ▼         ▼          ▼             ▼
       ┌────────┐ ┌────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐
       │ target │ │ initial│ │ gencan/ │ │ movebad │ │handler │
       └───┬────┘ └───┬────┘ └────┬────┘ └────┬────┘ └────────┘
           │          │           │           │
     ┌─────┴──────┐   │           │           │
     ▼            ▼   │           │           │
 ┌────────┐  ┌────────┴┐ ┌────────┴─────────────────────┐
 │restraint│ │ relaxer │ │  context/PackContext          │
 │   +     │ │  +      │ │   ModelData / RuntimeState /  │
 │ region  │ │TorsionMc│ │   WorkBuffers                 │
 └────────┘  └─────────┘ └──┬────────────────────────────┘
                            │
                            ▼
                      ┌────────────┐
                      │ objective  │◄───── constraints/ (EvalMode facade)
                      │   .rs      │
                      │ (hot path) │
                      └────────────┘
```

Ordering rule: `target.rs` / `restraint.rs` / `region.rs` are pure
data types with no dependencies on the driver. The `packer.rs` driver
depends on everything downstream. `objective.rs` is the narrow waist
through which all per-atom work flows.

## Core-type relationships

```text
User space                              │ Crate internals
                                        │
Target ----with_restraint()---          │ Vec<Arc<dyn Restraint>> ─┐
  │                                     │                          │
  │                                     │                          │  Arc::clone
  │     ┌── 14 *Restraint structs ──    │                          │  (refcount
  │     │                               │                          │   bump) into
  │     ├── user structs impl Restraint │                          │   per-atom
  │     │                               │                          │   CSR pool
  │     └── FromRegion<R>           ──  │                          ▼
  │                                     │   PackContext.restraints:
  │                                     │   Vec<Arc<dyn Restraint>>
  │                                     │     +
  ├── with_relaxer(impl Relaxer)     ── │   PackContext.iratom_offsets / iratom_data
  │                                     │   (CSR: atom idx → restraint indices)
  ├── fixed_at()                        │
  ├── constrain_rotation_*()            │   PackContext.rot_bound[][] Euler bounds
  └── count / ref_coords / radii        │   PackContext.coor / radius / nmols / natoms

Molpack
  ├── .add_handler(impl Handler)        │   Vec<Box<dyn Handler>>   (observers)
  └── .add_restraint(impl Restraint)    │   Vec<Arc<dyn Restraint>>
          (scope = global)              │   broadcast at pack() entry
```

Two important things:

1. **`PackContext` is the single owner of mutable state.** Everything
   else either reads from it, borrows `&mut` exclusively, or reads a
   snapshot. GENCAN, movebad, handlers, and the phase driver all take
   `&mut PackContext`.
2. **`Arc<dyn Restraint>` for polymorphic storage.** Cheap clone
   (refcount bump) into the CSR pool. The hot path
   (`objective.rs::accumulate_constraint_value/gradient`) does one
   virtual call per restraint per atom.

## Lifecycle of `pack()`

```text
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER ASSEMBLY                                                │
│    Molpack::new()                                               │
│      .tolerance(2.0).precision(0.01)                            │
│      .add_handler(ProgressHandler::new())                       │
│      .add_restraint(InsideBoxRestraint::new(...))  [global]     │
│      .pack(&[target_a, target_b], max_loops=400, seed=42)       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. PACK SETUP (packer.rs::pack)                                 │
│    a. Validate inputs (non-empty, non-zero atoms, valid PBC)    │
│    b. BROADCAST Molpack.global_restraints into each Target's    │
│       molecule_restraints via Arc::clone (scope equivalence)    │
│    c. Split targets into free / fixed                           │
│    d. Build PackContext with:                                   │
│         - flat restraints pool (Vec<Arc<dyn Restraint>>)        │
│         - CSR iratom_offsets/iratom_data per atom               │
│         - fixed atom positions via eulerfixed()                 │
│         - cell list initialized for current PBC box             │
│    e. initial::initial() produces starting x[0..6*ntotmol]      │
│         layout: [com0(3), com1(3), ..., eul0(3), eul1(3), ...]  │
│    f. handlers.on_start / on_initial                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. PHASE LOOP    for phase in 0..=ntype:                        │
│                                                                 │
│   phase <  ntype : PER-TYPE PRE-COMPACTION                      │
│       comptype[i] = (i == phase)  — optimize only this type     │
│                                                                 │
│   phase == ntype : ALL-TYPES MAIN LOOP (terminal phase)         │
│       comptype[i] = true for every i                            │
│                                                                 │
│   Each iteration of run_phase() does:                           │
│      ┌──────────────────────────────────────────────────────┐   │
│      │ handlers.on_phase_start(info)                        │   │
│      │ swap in per-type state; reset radscale → discale     │   │
│      │ precision short-circuit (unscaled evaluate first)    │   │
│      │                                                      │   │
│      │ for loop_idx in 0..max_loops:                        │   │
│      │    run_iteration():                                  │   │
│      │       [movebad if enabled]                           │   │
│      │       for (type, runners) in relaxers:               │   │
│      │         runners.on_iter(&ref_coords, f, &mut fn)     │   │
│      │       pgencan(x, &mut sys, params, precision)        │   │
│      │       evaluate_unscaled(&mut sys, xwork)             │   │
│      │       compute fimp = %Δf; radscale decay             │   │
│      │       handlers.on_step(step_info, sys)               │   │
│      │       if converged: return Converged                 │   │
│      │       if should_stop: return EarlyStop               │   │
│      │                                                      │   │
│      │ handlers.on_phase_end(info, report) [default no-op]  │   │
│      └──────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. FINALIZE                                                     │
│    a. init_xcart_from_x() — expand variables → Cartesian coords │
│    b. handlers.on_finish(sys)                                   │
│    c. context_to_frame(sys) — build output molrs::Frame         │
│    d. return PackResult { frame, converged, fdist, frest, ... } │
└─────────────────────────────────────────────────────────────────┘
```

Key invariants:

- `comptype[i]` gates which molecule types contribute to the
  objective in the current phase.
- `radscale` starts at `discale` (default 1.1) and decays toward 1.0
  — inflates atomic radii at start of phase, then shrinks to real
  tolerance.
- `precision` is the convergence threshold on both `fdist`
  (inter-molecule overlap) and `frest` (restraint violation).

## Hot path: one `evaluate()` call

Called O(10³–10⁴) times per `pack()` run. Performance lives here.

```text
pgencan → tn_ls (line search) → PackContext::evaluate(x, mode, g)
                                       │
                                       │   FOnly           → compute_f
                                       │   GradientOnly    → compute_g
                                       │   FAndGradient    → compute_fg
                                       │   RestMol         → compute_fg
                                       ▼
                   ┌───────────────────────────────────────────┐
                   │ objective.rs — compute_f / g / fg         │
                   │                                           │
                   │  1. expand_molecules():                   │
                   │       for type t, mol m, atom a:          │
                   │         com + eulerrmat(angles) · ref     │
                   │         → xcart[icart] = rotated position │
                   │                                           │
                   │  2. accumulate_constraint_value/gradient: │
                   │       for each atom icart:                │
                   │         range = iratom_offsets[icart]     │
                   │                 ..offsets[icart+1]         │
                   │         for &irest in iratom_data[range]: │
                   │           sys.restraints[irest]           │
                   │             .f / .fg(pos, s, s2)          │
                   │         accumulate f_total, frest         │
                   │         accumulate grad into gxcar[icart] │
                   │                                           │
                   │  3. insert_atom_in_cell():                │
                   │       place icart into latomfirst[icell]  │
                   │       linked-list (cell list build)       │
                   │                                           │
                   │  4. accumulate_pair_f/_fg_parallel:       │
                   │       for each non-empty cell:            │
                   │         for each neighbor cell (13):      │
                   │           for each (i, j) pair:           │
                   │             d = pbc_distance(xi, xj)      │
                   │             penalty = (σ − d)² if d < σ   │
                   │           → f_pair, grad_pair[i, j]       │
                   │                                           │
                   │  5. project_cartesian_gradient():         │
                   │       for each atom a in molecule m:      │
                   │         grad_com   += gxcar[icart]        │
                   │         grad_euler += Jᵀ · gxcar[icart]   │
                   │         (J = ∂xcart/∂euler)               │
                   │                                           │
                   │  return EvalOutput { f_total, fdist,frest}│
                   └───────────────────────────────────────────┘
```

Performance notes:

- Steps 1–3 are O(N_atoms); step 4 is
  O(N_atoms × avg_neighbors) ≈ O(N_atoms × 32).
- Step 4 is `rayon`-parallelized per cell
  (`accumulate_pair_fg_parallel`).
- `Arc<dyn Restraint>` in step 2 does a virtual call per restraint.
  Measured cost post-Phase-B: +0.22% e2e vs pre-refactor — well under
  the +5% soft gate. If this ever regresses, a crate-private
  `PackedRestraint` tagged-union fast path is queued in the spec.
- `Cell<f64>` would not be `Sync`; the parallel reduction in step 4
  uses `AtomicU64` with `f64::to_bits` / `f64::from_bits` shims.

## Invariants and conventions

**Gradient convention.** `Restraint::fg` accumulates the TRUE
gradient (∂penalty/∂x) INTO `g` with `+=`. Optimizer negates for
descent.

**Two-scale contract.** Linear penalties (kinds 2/3/6/7/10/11)
consume `scale`; quadratic penalties (kinds 4/5/8/9/12/13/14/15)
consume `scale2`. Each `impl Restraint` picks one internally.

**Rotation convention.** `apply_scaled_step` uses LEFT multiplication
`R_new = δR · R_old`. Single-atom tests cannot detect LEFT/RIGHT
bugs — always test with ≥ 2 atoms.

**Coordinate layout.** GENCAN variable vector `x` is
`[com0(3), com1(3), ..., eul0(3), eul1(3), ...]` of length
`6 * ntotmol`. Cartesian coords `xcart` are `Vec<[F; 3]>` of length
`ntotat`.

**Thread safety.** All trait objects are `Send + Sync`.
`Cell<f64>` is NOT `Sync` — internal interior mutability uses
`AtomicU64`.

**Scope equivalence law** (spec §4):

```text
molpack.add_restraint(r)
    ≡  for t in targets { t.with_restraint(r.clone()) }
```

No separate global-restraint storage path in `PackContext` — the
broadcast at `pack()` entry IS the implementation.

## Design decisions

- **Restraint vs Constraint.** Packmol implements all 15 "constraints"
  as soft penalties. Name reflects mechanism, not user intent →
  `Restraint`. A `Constraint` trait is reserved for future hard
  constraint work.
- **Direction-3 extension pattern.** User plugins and built-ins must
  be type-equal. Earlier drafts used a tagged-union
  `BuiltinConstraint { kind, params[9] }` blob exposed publicly — this
  created second-class citizenship for user types. Phase B.0 replaced
  this with 15 independent concrete structs. Internal AoS fast-path
  (if ever needed for perf) stays `pub(crate)` behind a `try_pack()`
  hook.
- **Arc over Box.** `Vec<Arc<dyn Restraint>>` instead of
  `Vec<Box<dyn Restraint>>` so packer init can clone restraints
  cheaply (refcount bump) into the per-atom CSR pool.
- **Objective trait.** GENCAN takes `&mut dyn Objective`, not
  `&mut PackContext`. This decouples the optimizer from the packing
  state — it can be tested against synthetic objectives (Rosenbrock /
  Booth / Beale) without the full packing pipeline.
- **Three-phase schedule.** Per-type compaction before the final
  all-types phase is a Packmol-essential heuristic: it prevents early
  phases from getting trapped by cross-type interference.
- **`init1` short-circuit.** `PackContext.init1` is a bool that skips
  the pair-kernel step during geometric pre-fitting — the initial
  phase only needs to push atoms into their regions, pair overlaps
  are ignored until the main loop.

## Cheatsheet: where to look for specific behavior

| Question | File : function |
|---|---|
| How is a restraint's penalty computed for one atom? | `restraint.rs::*::f` / `*::fg` |
| Where does the global `add_restraint` broadcast happen? | `packer.rs::pack` (top of function) |
| How are per-atom restraints indexed into the pool? | `packer.rs::pack` (CSR build loop) |
| What does the GENCAN variable vector look like? | `initial.rs::init_xcart_from_x` |
| How do Euler angles become Cartesian positions? | `euler.rs::compcart` / `eulerrmat` |
| Where is the pair-overlap kernel? | `objective.rs::accumulate_pair_fg_parallel` |
| How is precision-based termination tested? | `gencan/mod.rs::packmolprecision` |
| What does `movebad` do? | `movebad.rs::movebad` |
| How is torsion MC wired in? | `relaxer.rs::TorsionMcRelaxer::on_iter` |
| Where does periodic boundary wrap? | `context/pack_context.rs::pbc_distance` |
