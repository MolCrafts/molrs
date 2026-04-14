# Core Concepts

This chapter defines each abstraction in the crate in one place.
Cross-link to the types for full API details.

## Restraint

A [`Restraint`](crate::Restraint) is a **soft penalty** applied per
atom: `f(x, scale, scale2) -> F` and `fg(x, scale, scale2, g) -> F`.
It contributes to the packing objective and — in all current
implementations — derives from Packmol's `comprest.f90` / `gwalls.f90`.

```text
pub trait Restraint: Send + Sync + std::fmt::Debug {
    fn f (&self, x: &[F; 3], scale: F, scale2: F) -> F;
    fn fg(&self, x: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F;
    fn is_parallel_safe(&self) -> bool { true }
    fn name(&self) -> &'static str { std::any::type_name::<Self>() }
}
```

The crate ships 14 concrete `*Restraint` structs (one per Packmol
`kind` 2..=15), each holding its own semantically-named geometric
fields. User types `impl Restraint` sit in the same type slot —
there is no `Builtin*` wrapper in the public API. See
[`extending`](crate::extending) for a tutorial.

### Gradient convention

`fg` accumulates the TRUE gradient (∂penalty/∂x) INTO `g` with `+=`.
Do not overwrite: many restraints may touch the same atom. The
optimizer negates for descent.

### Two-scale contract

Packmol convention (mirrored in the port):

- Linear penalties — kinds 2, 3, 6, 7, 10, 11 (box / cube / plane) —
  consume `scale`.
- Quadratic penalties — kinds 4, 5, 8, 9, 12, 13, 14, 15 (sphere /
  ellipsoid / cylinder / gaussian) — consume `scale2`.

Each `impl Restraint` picks one internally. User-defined restraints
may ignore both knobs and use their own stiffness coefficient as an
instance field.

## Region

A [`Region`](crate::Region) is a **geometric predicate** with a signed
distance function:

```text
pub trait Region: Send + Sync + std::fmt::Debug {
    fn contains(&self, x: &[F; 3]) -> bool;
    fn signed_distance(&self, x: &[F; 3]) -> F;
    fn signed_distance_grad(&self, x: &[F; 3]) -> [F; 3] { /* default FD */ }
    fn bounding_box(&self) -> Option<BBox> { None }
}
```

Regions compose via the zero-cost combinators
[`And`](crate::And) / [`Or`](crate::Or) / [`Not`](crate::Not), with
analytic chain-rule gradients (max / min / negate). The
[`RegionExt`](crate::RegionExt) trait gives every `Region` ergonomic
`.and(...)` / `.or(...)` / `.not()` methods.

Any `Region` lifts to a `Restraint` via
[`FromRegion<R>`](crate::FromRegion):

```text
penalty(x) = scale2 * max(0, signed_distance(x))²
```

Use `Region` when you want compositional geometry (intersection /
union / complement). Use `Restraint` directly when you want a specific
penalty shape (linear vs quadratic, custom stiffness, multi-atom).

## Relaxer

A [`Relaxer`](crate::Relaxer) modifies a target's **reference geometry**
between outer optimizer calls. Use cases: torsion-MC sampling for
flexible chains, local MD relaxation, gradient descent on bond-angle
targets.

Two-part design — builder + runner:

- `Relaxer::build(&self, ref_coords) -> Box<dyn RelaxerRunner>` is
  called once at `pack()` entry.
- `RelaxerRunner::on_iter(&mut self, coords, f_current, evaluate, rng)`
  runs between movebad and GENCAN each outer iteration; returns
  `Some(new_coords)` on accept, `None` on reject.

Relaxers require `count == 1` because all copies share the same
reference coords.

Built-in: [`TorsionMcRelaxer`](crate::TorsionMcRelaxer) (Metropolis
torsion sampling with self-avoidance).

## Handler

A [`Handler`](crate::Handler) is an observer invoked at well-defined
lifecycle points:

```text
pub trait Handler: Send {
    fn on_start        (&mut self, ntotat, ntotmol)       {}
    fn on_initial      (&mut self, sys: &PackContext)     {}
    fn on_step         (&mut self, info: &StepInfo, sys);   // required
    fn on_phase_start  (&mut self, info: &PhaseInfo)      {}
    fn on_phase_end    (&mut self, info, report: &PhaseReport) {}
    fn on_inner_iter   (&mut self, iter, f, sys)          {}
    fn on_finish       (&mut self, sys: &PackContext)     {}
    fn should_stop     (&self) -> bool                    { false }
}
```

Observer contract: `sys` is always `&PackContext`, never `&mut`.
Handlers cannot modify packer state — use a `Relaxer` if you need to.

Built-ins: [`NullHandler`](crate::NullHandler),
[`ProgressHandler`](crate::ProgressHandler),
[`EarlyStopHandler`](crate::EarlyStopHandler),
[`XYZHandler`](crate::XYZHandler).

## Objective

The [`Objective`](crate::objective::Objective) trait abstracts over
what GENCAN sees. `PackContext` implements it; synthetic test
objectives (Rosenbrock / Booth / Beale) can implement it to exercise
the optimizer in isolation.

```text
pub trait Objective {
    fn evaluate(&mut self, x: &[F], mode: EvalMode, g: Option<&mut [F]>) -> EvalOutput;
    fn fdist(&self) -> F;
    fn frest(&self) -> F;
    fn ncf(&self) -> u32;
    fn ncg(&self) -> u32;
    fn reset_eval_counters(&mut self);
    fn bounds(&self, l: &mut [F], u: &mut [F]);
}
```

GENCAN (`pgencan`, `gencan`, `tn_ls`, `spg`, `cg`) takes `&mut dyn
Objective` rather than `&mut PackContext` — the optimizer is
decoupled from the packing state.

## Target

A [`Target`](crate::Target) describes one molecule type:

- Input coordinates + centered reference coordinates.
- Van der Waals radii, element symbols, copy count, name.
- Its attached restraints (per-target + per-atom-subset).
- Its attached relaxers.
- Optional fixed placement (Euler + translation).
- Optional Euler-angle bounds (`constrain_rotation_x/y/z`).

Targets are snapshotted at `pack()` entry — mutating a `Target` after
passing it to the packer has no effect.

## Molpack

[`Molpack`](crate::Molpack) is the builder facade:

```text
Molpack::new()
    .tolerance(2.0)
    .precision(0.01)
    .maxit(20)
    .pbc(min, max)              // or .pbc_box(lengths)
    .add_handler(...)
    .add_restraint(...)         // global; broadcast to every target
    .pack(&[targets], max_loops, seed)
```

Every setter consumes and returns `self`. `pack` takes `&mut self`
(handlers are invoked through it).

## PackContext

[`PackContext`](crate::PackContext) is the single owner of mutable
packing state — coordinates, cell lists, restraint pool, rotation
buffers, counters. All optimizer / movebad / handler code paths take
`&mut PackContext` (for writers) or `&PackContext` (for observers).

Structure (`molrs-pack/src/context/`):

- `ModelData` — topology and inputs (immutable after init).
- `RuntimeState` — mutable per-iteration state (x, coor, radius).
- `WorkBuffers` — scratch arrays (xcart, gxcar, radiuswork).

Users rarely touch `PackContext` directly — it's passed through
handlers and relaxers. Power users implementing a custom `Objective`
against synthetic test problems will interact with it.

## Scope equivalence law

Formal statement (spec §4):

```text
molpack.add_restraint(r)
    ≡  for t in targets { t.with_restraint(r.clone()) }
```

There is no separate "global-restraint" storage path in `PackContext`.
The broadcast happens inside `pack()`; clones each target and pushes
the global restraints into its `molecule_restraints` via
`Arc::clone` (refcount bump, not a deep copy).

Per-atom-subset scope is a method-argument pair
`(indices: &[usize], restraint: impl Restraint)`, not a wrapper
struct. There is no `AtomRestraint` public type.

## Restraint versus Constraint

- **Restraint** = soft penalty (violable; pays energy cost).
- **Constraint** = hard constraint (must satisfy; SHAKE / RATTLE /
  LINCS / Lagrange-multiplier mechanisms).

Packmol implements all 15 "constraints" as soft penalties
(`scale * max(0, d)` or `scale2 * max(0, d)²`). Honest naming ⇒
`Restraint`. This crate does not currently define a `Constraint`
trait; adding hard constraints is future work.

## Direction-3 extension pattern

Every extension trait in this crate follows the same shape:

1. Public trait: `pub trait X`.
2. N concrete `pub struct` types that `impl X`, each holding its own
   semantically-named fields.
3. User types `impl X` identically — zero type-level distinction from
   built-ins.

Forbidden in the public API:

- `Builtin*` / `Native*` / `Packmol*` prefixed wrapper types.
- Tagged-union enums that package N built-ins as a single exposed
  type.
- Builder pattern (`X::new().add(...).add(...)`).
- Injection of composition operators into the main trait
  (composition lives on separate traits, e.g. `Region` vs `Restraint`).
- Wrapper types for per-atom-subset scope (that's a method-argument
  pair, not a type).

If you need internal AoS performance structures (e.g. tagged unions
for hot-path match dispatch), they go behind `pub(crate)` and opt-in
via a crate-private hook — invisible to users.
