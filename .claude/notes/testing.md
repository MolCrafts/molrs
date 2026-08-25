# Testing (molrs)

## Where science lives

**Numerical / chemical correctness is tested only in Rust unit tests**
(`#[cfg(test)]` next to code under `molrs/src/**`):

```bash
cargo test -p molcrafts-molrs --lib --features md   # example: only the subsystem under test
```

### Rayon in unit tests

`rayon` is a default feature. Tests must **not** let each case spawn
`available_parallelism()` workers — that races `build_global` and hits
`EAGAIN`. The test pool is installed once in `core::test_rayon`:

- default **2** workers (the parallel path still runs)
- override with `MOLRS_TEST_THREADS` (clamped to 2..=8)

`core::test_rayon::test_pool_is_multithreaded` asserts more than one worker
actually executed a `par_iter`. Do not set `MOLRS_TEST_THREADS=1`.

There is **no** `molrs/tests/` integration-binary tree.

## Language bindings (Python / C / WASM)

Bindings only prove the **seam**:

- symbols import / construct
- types and dtypes at the boundary (e.g. float64 arrays)
- non-mutating contracts, error mapping, column order

They must **not** re-run antechamber/RDKit oracles or multi-molecule
numerical parity. That work belongs in core unit tests.

```bash
cd molrs-python && maturin develop && pytest -q
```
