# Testing (molrs)

## Where science lives

**Numerical / chemical correctness is tested only in Rust unit tests**
(`#[cfg(test)]` next to code under `molrs/src/**`):

```bash
cargo test -p molcrafts-molrs --lib --features full
```

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
