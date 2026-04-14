# Getting Started

This chapter walks through the minimum you need to run a packing job
and the most common customization points. Read the reference docs of
individual types on the crate landing page for full details.

## Adding the dependency

```toml
[dependencies]
molcrafts-molrs-pack = "0.0"
```

## First pack: one molecule type in a box

```rust,no_run
use molrs_pack::{InsideBoxRestraint, Molpack, Target};

let water_positions = [
    [0.0, 0.0, 0.0],
    [0.96, 0.0, 0.0],
    [-0.24, 0.93, 0.0],
];
let water_radii = [1.52, 1.20, 1.20];

let target = Target::from_coords(&water_positions, &water_radii, 100)
    .with_name("water")
    .with_restraint(InsideBoxRestraint::new([0.0; 3], [40.0; 3]));

let result = Molpack::new()
    .tolerance(2.0)
    .precision(0.01)
    .pack(&[target], 200, Some(42))?;

println!("converged = {}, natoms = {}", result.converged, result.natoms());
# Ok::<(), molrs_pack::PackError>(())
```

Arguments to [`Molpack::pack`](crate::Molpack::pack):

- `&[Target]` — one entry per molecule type.
- `max_loops: usize` — outer-iteration budget per phase.
- `seed: Option<u64>` — pass `Some(n)` for reproducible placements.

Returns a [`PackResult`](crate::PackResult) with the output coordinates
as a `molrs::Frame` plus convergence diagnostics.

## Restraint scopes

Restraints can be attached at three granularities. All of them use the
same [`Restraint`](crate::Restraint) trait underneath.

**Per-target, all atoms** — the most common case:

```rust,no_run
# use molrs_pack::{InsideBoxRestraint, Target};
# let (pos, rad) = (&[[0.0; 3]][..], &[1.0][..]);
let target = Target::from_coords(pos, rad, 100)
    .with_restraint(InsideBoxRestraint::new([0.0; 3], [40.0; 3]));
```

**Per-target, atom subset** — restrict to specific atoms of every
molecule copy (1-based indexing, Packmol convention):

```rust,no_run
# use molrs_pack::{BelowPlaneRestraint, Target};
# let (pos, rad) = (&[[0.0; 3]][..], &[1.0][..]);
let target = Target::from_coords(pos, rad, 100)
    .with_restraint_for_atoms(&[1, 2], BelowPlaneRestraint::new([0.0, 0.0, 1.0], 5.0));
```

**Global, all targets** — broadcast. Semantically equivalent to
calling `with_restraint` on every target:

```rust,no_run
# use molrs_pack::{InsideSphereRestraint, Molpack, Target};
# let (pos, rad) = (&[[0.0; 3]][..], &[1.0][..]);
# let t_a = Target::from_coords(pos, rad, 100);
# let t_b = Target::from_coords(pos, rad, 100);
let result = Molpack::new()
    .add_restraint(InsideSphereRestraint::new([20.0; 3], 30.0))
    .pack(&[t_a, t_b], 200, Some(42))?;
# Ok::<(), molrs_pack::PackError>(())
```

The scope-equivalence law is formal: `molpack.add_restraint(r)` is
implemented as `for t in targets { t.with_restraint(r.clone()) }`.
There is no separate "global-restraint" storage path.

## Handlers (progress observation)

Attach [`Handler`](crate::Handler) implementations for progress reporting,
trajectory output, or early-stop logic. Built-ins:

```rust,no_run
# use molrs_pack::{EarlyStopHandler, InsideBoxRestraint, Molpack, ProgressHandler, Target, XYZHandler};
# let (pos, rad) = (&[[0.0; 3]][..], &[1.0][..]);
# let target = Target::from_coords(pos, rad, 100).with_restraint(InsideBoxRestraint::new([0.0; 3], [40.0; 3]));
let mut packer = Molpack::new()
    .add_handler(ProgressHandler::new())
    .add_handler(XYZHandler::new("traj.xyz", /* every = */ 10))
    .add_handler(EarlyStopHandler::new(/* threshold = */ 1e-4));
# let _ = packer.pack(&[target], 200, Some(42));
```

Write your own — see the [`extending`](crate::extending) module.

## Relaxers (in-loop conformation sampling)

Flexible molecules benefit from torsion-MC relaxation between outer
optimizer calls. Attach a [`Relaxer`](crate::Relaxer) to a target:

```rust,no_run
# use molrs::molgraph::MolGraph;
# use molrs_pack::{InsideSphereRestraint, Target, TorsionMcRelaxer};
# let (pos, rad) = (&[[0.0; 3]][..], &[1.0][..]);
# let graph = MolGraph::new();
let target = Target::from_coords(pos, rad, 1)  // relaxers require count == 1
    .with_restraint(InsideSphereRestraint::new([0.0; 3], 20.0))
    .with_relaxer(
        TorsionMcRelaxer::new(&graph)
            .with_temperature(0.5)
            .with_steps(20)
    );
```

## Free versus periodic boundary

Free boundary is the default. For PBC:

```rust,no_run
# use molrs_pack::Molpack;
let packer = Molpack::new()
    .pbc_box([30.0, 30.0, 30.0]);              // origin at zero
// or
let packer = Molpack::new()
    .pbc([0.0, 0.0, 0.0], [30.0, 30.0, 30.0]);  // explicit min / max
```

Zero-length axes return [`PackError::InvalidPBCBox`](crate::PackError).

## Running the canonical examples

Five Packmol-equivalent workloads are checked in under `examples/`:

```bash
cargo run -p molcrafts-molrs-pack --release --example pack_mixture
cargo run -p molcrafts-molrs-pack --release --example pack_bilayer
cargo run -p molcrafts-molrs-pack --release --example pack_interface
cargo run -p molcrafts-molrs-pack --release --example pack_solvprotein
cargo run -p molcrafts-molrs-pack --release --example pack_spherical
```

Two heavier ad-hoc demonstrations (not in the regression suite):

```bash
cargo run -p molcrafts-molrs-pack --release --example mc_fold_chain
cargo run -p molcrafts-molrs-pack --release --example pack_polymer_vesicle
```

Set `MOLRS_PACK_EXAMPLE_PROGRESS=1` to enable the built-in
`ProgressHandler`. Set `MOLRS_PACK_EXAMPLE_XYZ=1` to dump a trajectory
under `out/`.

## Where to go next

- [`concepts`](crate::concepts) — the key abstractions in one place.
- [`architecture`](crate::architecture) — system design, lifecycle,
  hot path.
- [`extending`](crate::extending) — tutorials for writing your own
  `Restraint` / `Region` / `Handler` / `Relaxer`.
