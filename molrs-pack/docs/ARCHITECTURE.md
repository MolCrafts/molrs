# molrs-pack Architecture

Developer-oriented overview of the `molrs-pack` crate. Read this before touching the packer if you want to understand what goes where and why. Companion document: [`CODE_TOUR.md`](./CODE_TOUR.md) for step-by-step navigation + extension tutorials.

## 1. What the crate does

Given `N` molecule types with copy counts and geometric restraints, produce a non-overlapping arrangement of all atoms in free or periodic space. The algorithm is a faithful Rust port of **Packmol** (Martínez et al. 2009) driven by a **GENCAN** quasi-Newton optimizer with bound constraints. Correctness is benchmarked against Packmol's reference output for 5 canonical workloads (see `examples/pack_*`).

Scope:

- **In**: soft-penalty geometric restraints (box / sphere / plane / ellipsoid / cylinder / gaussian / user-defined), torsion-MC relaxation of flexible chains, free or periodic boundaries, fixed molecule placement, per-atom-subset restraint scoping.
- **Out** (future work): hard constraints (SHAKE/RATTLE/LINCS), Python bindings, batched GEMM / SIMD inner kernels.

## 2. Module map

```
molrs-pack/
├── src/
│   ├── lib.rs              — public re-exports only
│   ├── api/                — (builder-style facade; thin)
│   ├── packer.rs           — Molpack builder + pack() driver + phase loop
│   ├── target.rs           — Target = one molecule type + its restraints/relaxers
│   ├── restraint.rs        — Restraint trait + 14 concrete *Restraint structs
│   ├── region.rs           — Region trait + And/Or/Not combinators + FromRegion
│   ├── relaxer.rs          — Relaxer/RelaxerRunner traits + TorsionMcRelaxer
│   ├── handler.rs          — Handler observer trait + 4 built-ins
│   ├── objective.rs        — compute_f/g/fg + Objective trait (GENCAN's view)
│   ├── context/            — PackContext: all mutable state lives here
│   │   ├── pack_context.rs — fields, constructors, helpers
│   │   ├── model.rs        — immutable topology data
│   │   ├── state.rs        — mutable per-iteration state
│   │   └── work_buffers.rs — scratch buffers (xcart, gxcar, ...)
│   ├── constraints/        — EvalMode/EvalOutput + Constraints ZST facade
│   │                         (misleadingly named — it's the evaluate-dispatch
│   │                          facade, not a restraint store)
│   ├── gencan/             — GENCAN optimizer (pgencan/tn_ls/spg/cg)
│   ├── initial.rs          — init1 / initial placement / swap state
│   ├── movebad.rs          — movebad heuristic (perturb worst molecules)
│   ├── euler.rs            — Euler angle ↔ rotation matrix
│   ├── cell.rs             — cell-list indexing
│   ├── frame.rs            — PackContext ↔ molrs::Frame conversion
│   ├── validation.rs       — post-pack correctness check (validate_from_targets)
│   ├── cases.rs            — ExampleCase enum wiring the 5 canonical examples
│   ├── numerics.rs         — constants (objective floor, small thresholds)
│   ├── error.rs            — PackError variants
│   └── random.rs           — RNG helper
├── benches/
│   ├── pack_end_to_end.rs  — catastrophic-regression alarm (5 workloads)
│   ├── evaluate_unscaled.rs— fn + caller microbench
│   ├── run_iteration.rs    — fn + caller microbench
│   ├── run_phase.rs        — fn + caller microbench
│   └── objective_dispatch.rs — dyn-dispatch cost isolation
├── tests/
│   ├── restraint.rs  — all 14 Restraint kinds + user-plugin type equality
│   ├── target.rs     — Target builder semantics
│   ├── packer.rs     — pack() end-to-end + Molpack::add_restraint broadcast
│   ├── gradient.rs   — finite-difference validation of analytic gradients
│   ├── relaxer.rs    — Relaxer / TorsionMcRelaxer integration
│   ├── euler.rs      — Euler math round-trip
│   └── examples_batch.rs — --ignored; Packmol equivalence on all 5 examples
├── examples/               — 5 canonical + 2 ad-hoc demos
└── docs/
    ├── ARCHITECTURE.md     — this file
    ├── CODE_TOUR.md        — navigation + extension tutorials
    └── packmol_alignment.md— how kind numbers map to comprest.f90 / gwalls.f90
```

## 3. Module dependency graph

```
                   ┌───────┐
                   │ lib.rs│  (public re-exports only)
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
      │  .rs   │ │   .rs  │ │         │ │   .rs   │ │  .rs   │
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

Ordering rule: `target.rs` / `restraint.rs` / `region.rs` have **no dependencies** on `packer.rs` or `objective.rs` — the data structures are pure. The `packer.rs` driver depends on everything downstream. `objective.rs` is the narrow waist through which all per-atom work flows.

## 4. Core types — relationships

```
 User space                         │ Crate internals
                                    │
 Target -------with_restraint()---- │ Vec<Arc<dyn Restraint>>  ─┐
    │                               │                           │
    │                               │                           │  copied
    │       ┌──── 15 *Restraint ────┤                           │  (Arc::clone,
    │       │     structs           │                           │   refcount
    │       │                       │                           │   bump)
    │       ├──── user structs      │                           │
    │       │     impl Restraint    │                           │
    │       │                       │                           ▼
    │       └──── FromRegion<R>  ───┤  PackContext.restraints: Vec<Arc<dyn Restraint>>
    │                               │                      +
    ├── with_relaxer()              │  PackContext.iratom_offsets / iratom_data
    │       └──── impl Relaxer ─────┤  (CSR: atom idx → indices into .restraints)
    │                               │
    ├── fixed_at()                  │  PackContext.fixedatom[] bitmask
    ├── constrain_rotation_*()      │  PackContext.rot_bound[][] Euler bounds
    └── count / ref_coords / radii  │  PackContext.coor / radius / nmols / natoms

 Molpack
    ├── .add_handler(impl Handler)  │  Vec<Box<dyn Handler>> (observers)
    └── .add_restraint(impl R.)     │  Vec<Arc<dyn Restraint>>
            (scope = global)        │  (broadcast to every target at pack() time)
```

Two important things:

1. **`PackContext` is the single owner of mutable state.** Everything else either reads from it, borrows `&mut` exclusively, or reads a snapshot. GENCAN, movebad, handlers, and the phase driver all take `&mut PackContext`.
2. **`Arc<dyn Restraint>` for polymorphic storage.** Cheap clone (refcount bump) into the CSR pool. The hot path (`objective.rs::accumulate_constraint_value/gradient`) does one virtual call per restraint per atom.

## 5. Lifecycle — one `pack()` call

```
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
│         - fixed atom positions placed via eulerfixed()          │
│         - cell list initialized for current PBC box             │
│    e. initial::initial() produces starting x[0..3*ntotmol*3]    │
│         variable layout: [com0(3), com1(3), ..., eul0(3), ...]  │
│    f. handlers.on_start / on_initial                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. PHASE LOOP    for phase in 0..=ntype:                        │
│                                                                 │
│   phase < ntype :  PER-TYPE PRE-COMPACTION                      │
│       comptype[i] = (i == phase) — optimize only one type       │
│                                                                 │
│   phase == ntype:  ALL-TYPES MAIN LOOP (terminal phase)         │
│       comptype[i] = true for every i                            │
│                                                                 │
│   Each iteration of run_phase() does:                           │
│                                                                 │
│      ┌──────────────────────────────────────────────────────┐   │
│      │ handlers.on_phase_start(info)                        │   │
│      │ swap in per-type state; reset radscale → discale     │   │
│      │ precision short-circuit (unscaled evaluate first)    │   │
│      │                                                      │   │
│      │ for loop_idx in 0..max_loops:                        │   │
│      │    run_iteration():                                  │   │
│      │       [movebad if enabled]                           │   │
│      │       for (type, runners) in relaxers:               │   │
│      │          runners.on_iter(&ref_coords, f, &mut fn)    │   │
│      │       pgencan(x, &mut sys, gencan_params, precision) │   │
│      │       evaluate_unscaled(&mut sys, xwork)             │   │
│      │       compute fimp = %Δf; radscale decay             │   │
│      │       handlers.on_step(step_info, sys)               │   │
│      │       if converged: return Converged                 │   │
│      │       if should_stop: return EarlyStop               │   │
│      │                                                      │   │
│      │ handlers.on_phase_end(info, report)    [default noop]│   │
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

**Key invariants**:

- `comptype[i]` gates which molecule types contribute to the objective in the current phase. Per-type phases evaluate just one type; the final phase evaluates all.
- `radscale` starts at `discale` (default 1.1) and decays toward 1.0 — inflates atomic radii at start of phase, then shrinks to real tolerance.
- `precision` is the convergence threshold on both `fdist` (inter-molecule overlap) and `frest` (restraint violation).

## 6. Hot-path — one `evaluate()` call

Called O(10³–10⁴) times per `pack()` run. This is where performance lives.

```
pgencan → tn_ls (line search) → PackContext::evaluate(x, EvalMode, opt g)
                                       │
                                       │   FOnly → compute_f
                                       │   GradientOnly → compute_g
                                       │   FAndGradient / RestMol → compute_fg
                                       ▼
                   ┌───────────────────────────────────────────┐
                   │ objective.rs  — compute_f/g/fg            │
                   │                                           │
                   │  1. expand_molecules():                   │
                   │       for type t, mol m, atom a:          │
                   │         com + eulerrmat(angles) * ref     │
                   │         → xcart[icart] = rotated position │
                   │                                           │
                   │  2. accumulate_constraint_value/gradient: │
                   │       for each atom icart:                │
                   │         range = iratom_offsets[icart]     │
                   │                 ..offsets[icart+1]         │
                   │         for &irest in iratom_data[range]: │
                   │           sys.restraints[irest]           │
                   │              .f / .fg(pos, s, s2)         │
                   │         → accumulate into f_total, frest  │
                   │         → accumulate grad into gxcar[icart]│
                   │                                           │
                   │  3. insert_atom_in_cell():                │
                   │       place icart into latomfirst[icell]  │
                   │       linked-list (cell list build)       │
                   │                                           │
                   │  4. accumulate_pair_f / _fg_parallel:     │
                   │       for each non-empty cell:            │
                   │         for each neighbor cell (13):      │
                   │           for each (i, j) pair:           │
                   │             d = pbc_distance(xi, xj)      │
                   │             penalty = (σ - d)² if d < σ   │
                   │             → f_pair, grad_pair[i,j]      │
                   │                                           │
                   │  5. project_cartesian_gradient():         │
                   │       for each atom a in molecule m:      │
                   │         grad_com += gxcar[icart]          │
                   │         grad_euler += J^T · gxcar[icart]  │
                   │         (J = ∂xcart/∂euler)               │
                   │                                           │
                   │  return EvalOutput { f_total, fdist,frest}│
                   └───────────────────────────────────────────┘
```

**Performance notes**:

- Steps 1–3 are O(N_atoms); step 4 is O(N_atoms × avg_neighbors) ≈ O(N_atoms × 32).
- Step 4 is `rayon`-parallelized per cell (`accumulate_pair_fg_parallel`).
- `Arc<dyn Restraint>` in step 2 does a virtual call per restraint. Measured cost post-B.0: +0.22% e2e vs pre-refactor — well under the +5% soft gate. If this ever regresses a crate-private `PackedRestraint` tagged-union fast path is queued in spec §6.3.
- `Cell<f64>` would not be `Sync`; the parallel reduction in step 4 uses `AtomicU64` + `f64::to_bits`/`from_bits` shims.

## 7. Extension points (direction 3)

Every public trait follows the same pattern: `pub trait X` + N concrete `pub struct` types that `impl X`; user code `impl X` sits in the same type slot as the built-ins.

| Trait | User implements | Used by |
|---|---|---|
| `Restraint` | geometric or biasing penalty function (`f`/`fg`) | `objective.rs::accumulate_constraint_*` |
| `Region` | geometric predicate (`contains`/`signed_distance`) that composes via `And`/`Or`/`Not` | `FromRegion<R>` wraps it into a Restraint |
| `Relaxer` | reference-geometry modifier that runs between outer iterations (MC, MD, gradient descent) | `packer.rs::run_iteration` (relaxer_runners loop) |
| `Handler` | observer callbacks for progress / early stop / trajectory output | `packer.rs::run_iteration`, `run_phase`, `pack` |

**Hard rules** (spec §0 bullet 9):

- No `Builtin*` / `Native*` wrapper types in the public API.
- No tagged-union enum published as the trait's concrete representative.
- No builder pattern — concrete structs are constructed directly with `StructName { field1, field2 }` or `StructName::new(...)`.
- Combinators (e.g. `.and()` for region intersection) live on their own auxiliary trait (e.g. `Region`), never on the primary evaluation trait.
- Per-atom-subset scope is a method parameter pair `(&[idx], restraint)`, not a wrapper type.

## 8. Invariants and conventions

**Trait contracts**:

- `Restraint::fg` accumulates the TRUE gradient (∂penalty/∂x) INTO `g` with `+=`. Optimizer negates for descent.
- Two-scale contract: linear penalties (box / cube / plane — Packmol kinds 2/3/6/7/10/11) consume `scale`; quadratic penalties (sphere / ellipsoid / cylinder / gaussian — kinds 4/5/8/9/12/13/14/15) consume `scale2`.
- `Region::signed_distance`: negative inside, positive outside, zero on the boundary.
- `Region::signed_distance_grad`: default is 3-point central FD (ε=1e-6, slow); concrete regions override analytically for the hot path.
- `Handler` methods default to no-op except `on_step` (required). Observer callbacks must not mutate `sys` (hence `&PackContext`).

**Rotation convention**: `apply_scaled_step` uses **LEFT** multiplication `R_new = δR · R_old`. Single-atom tests cannot detect LEFT/RIGHT bugs — always test with ≥2 atoms.

**Coordinate layout**: GENCAN variable vector `x` is `[com0(3), com1(3), ..., eul0(3), eul1(3), ...]` of length `6 * ntotmol`. Cartesian coords `xcart` are `Vec<[F; 3]>` of length `ntotat`.

**Thread safety**: all trait objects are `Send + Sync`. `Cell<f64>` is NOT `Sync` — internal interior mutability uses `AtomicU64`.

**Scope equivalence law** (spec §4):

```rust
molpack.add_restraint(r) ≡ for t in targets { t.with_restraint(r.clone()) }
```

No separate "global-restraint" storage path in `PackContext` — the broadcast at `pack()` entry IS the implementation.

## 9. Design decisions worth understanding

- **Restraint vs Constraint.** Packmol implements all 15 "constraints" as soft penalties (no Lagrange / SHAKE / RATTLE machinery exists). Name reflects mechanism, not user intent → `Restraint`. A `Constraint` trait is reserved for a future phase when actual hard constraints are added.
- **Direction 3 extension pattern.** User plugins and built-ins MUST be type-equal. Earlier drafts used a tagged-union `BuiltinConstraint { kind, params[9] }` blob for internal AoS — exposed publicly, it created second-class citizenship for user types. Phase B.0 replaced this with 15 independent concrete structs each holding its own semantically-named fields. Internal AoS fast-path (if needed for perf) stays crate-private behind a `try_pack()` hook.
- **Arc over Box.** `Vec<Arc<dyn Restraint>>` instead of `Vec<Box<dyn Restraint>>` so packer init can clone restraints cheaply (refcount bump) into the per-atom CSR pool. Also lets `Target` hold its restraints as `Arc` while the packer also holds references to them without ownership transfer.
- **Objective trait** (GENCAN interface). Abstracts over `PackContext::evaluate` — GENCAN takes `&mut dyn Objective`, not `&mut PackContext`. This makes the optimizer testable against synthetic objectives (Rosenbrock / Booth / Beale) without the full packing state.
- **Three-phase schedule.** Per-type compaction before the final all-types phase is a Packmol-essential heuristic: it prevents early phases from getting trapped by cross-type interference. Each per-type phase runs its own `run_phase` call with `comptype[i] = (i == phase)`.
- **`init1` short-circuit.** `PackContext.init1` is a bool that skips the pair-kernel step during geometric pre-fitting — the initial phase only needs to push atoms into their regions, pair overlaps are ignored until the main loop. Keeps init1 phase ~10× faster.

## 10. Where to look for specific behavior

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

## 11. Further reading

- `CODE_TOUR.md` — step-by-step walkthrough + extension tutorials
- `packmol_alignment.md` — Packmol kind number ↔ Rust struct mapping
- `.claude/specs/molrs-pack-plugin-arch.md` — full spec including v2-r4 direction-3 rules, performance gates, risk register
- `.claude/skills/learn-packmol/SKILL.md` — Packmol Fortran reference discipline
