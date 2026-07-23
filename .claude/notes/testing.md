# Testing (molrs)

## Layout

- Unit tests: `#[cfg(test)]` modules next to code in `molrs/src/**`
- Default gate: `cargo test -p molcrafts-molrs --lib --features full`
- No `molrs/tests/` integration-binary tree (removed)
- Binders: `molrs-python` pytest, `molrs-capi` cmake, `molrs-wasm` wasm-pack
- Oracle data for C++ AM1-BCC bridge: `molrs-cxxapi/tests/antechamber_oracle.rs`

## Commands

```bash
cargo test -p molcrafts-molrs --lib --features full
# binders (also pre-push)
cd molrs-python && maturin develop && pytest -q
```

## Iron laws

- One concern per unit test function
- No silent skips of known failures
