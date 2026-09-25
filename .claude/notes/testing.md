# Testing (molrs)

## Where science lives

**Numerical / chemical correctness is tested only in Rust unit tests**
(`#[cfg(test)]` next to code under `molrs/src/**`):

```bash
cargo mrs-test                       # the gate: all 2513 unit tests, ~5 s
cargo mrs-doctest                    # rustdoc examples (not covered by --lib)
cargo mrs-test -- ff::potential      # inner loop, filtered: same binary
```

Narrow the **filter**, never the feature list. Every `cargo mrs-test` run shares
one test binary; after an edit cargo rebuilds it incrementally (~6 s) and the
whole suite runs in ~5 s, so filtering saves at most seconds. A hand-written `cargo test <module>` (or `--features md`) resolves a
different feature set and recompiles all 293k lines — that is a 67 s rebuild
and another ~420 MB of incremental cache, not a saving. See
`.claude/notes/build.md`.

A unit test asserts one behaviour of one unit with hand-written inputs: a
closed form, a limit, an invariant (`F = -dE/dx`, symmetry, exactness), a
hand-derived expectation. Not in the suite: end-to-end pipelines
(parse → typify → energy, thousand-step MD runs), numbers captured from an
external program (RDKit, antechamber, parmchk2, LAMMPS), tables asserting
their own rows, source-text gates. `BruteForce` is the neighbour-search
oracle and carries its own tests, so a backend test compares against it
rather than against a second hand-written double loop.

Perception tests may build their molecules with `io::smiles` fixtures; that
is a test-only dependency and the one exception to "`perceive` reaches only
`core`".

### Rayon in unit tests

`rayon` is a default feature. Tests must **not** let each case spawn
`available_parallelism()` workers — that races `build_global` and hits
`EAGAIN`. The test pool is installed once in `core::test_rayon`:

- default **2** workers (the parallel path still runs)
- override with `MOLRS_TEST_THREADS` (clamped to 2..=8)

`core::test_rayon::test_pool_is_multithreaded` asserts more than one worker
actually executed a `par_iter`. Do not set `MOLRS_TEST_THREADS=1`.

There is no `molrs/tests/` tree; behaviour is tested next to the code.

## Language bindings (Python / C / WASM)

Bindings only prove the **seam**:

- symbols import / construct
- types and dtypes at the boundary (e.g. float64 arrays)
- non-mutating contracts, error mapping, column order

They must **not** re-derive numerics the Rust suite proves or run
multi-stage pipelines. `molrs-python/tests/test_stream.py` is the Python-side
coverage of the WebSocket publisher and stays.

```bash
uv --directory molrs-python sync --no-install-project --extra dev
uv --directory molrs-python run --no-sync tox -e py
```

## Owed coverage (2026-09-20)

Public surface still without a unit test of its own; fill with hand-built
minimal molecules, not oracle dumps:

- `ff/mmff/{atomtype,resolve}.rs`, `ff/typifier/estimate/cascade.rs`,
  `ff/typifier/atd/*`, `ff/typifier/am1bcc.rs` (only the cxxapi seam test
  reaches it)
- `core/store/schema/{validator,violation}.rs` (exercised through
  `schema/mod.rs` only)
- `perceive/builder.rs`
- `stream/publisher.rs` on the Rust side beyond the socket smoke tests
