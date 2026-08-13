# Release — molrs before molpy

## Rule

1. Land on **master**, tag **`vX.Y.Z`**, wait for **Publish** (crates.io + npm + PyPI including Pyodide wheel).
2. Only then bump **molpy** to the same **major.minor** and tag.
3. Shared pin: consumers use `molcrafts-molrs>=X.Y.0,<X.(Y+1)`.

## Publish (tag push)

Workflow `.github/workflows/publish.yml`:

| Job | Registry |
|-----|----------|
| `publish-molrs` | crates.io (`molcrafts-molrs`) |
| `publish-wasm` | npm (`@molcrafts/molrs`) |
| `build-python` | desktop wheels → artifact |
| `build-python-pyodide` | Emscripten/Pyodide wheel → artifact |
| `publish-python` | PyPI (all wheels, trusted publishing) |

Re-run: Actions → Publish → `workflow_dispatch` (idempotent skips).

## scripts/

Only `scripts/fetch-test-data.sh` lives in-tree. No publish helper scripts.

## v0.12.1 (2026-08-05)

- SMILES/SMARTS emit: `write_smiles` / `from_atomistic` / `write_smarts` (io surface only)
- smiles-emit-01..04 closed

## v0.13.1 (2026-08-13)

Patch on the 0.13 line. Land on master, tag `v0.13.1`, wait for Publish, then molpy 0.13.1.

- `Frame.meta` is a write-through `FrameMeta` mapping: assign `frame.meta["timestep"] = 0` (plain Python scalars). `MetaValue` remains for explicit typed writes.
- Python `Block` / `Frame` columns accept `molrs.keys.Key` as well as `str`.
- NeighborList / Neighbors binders (Python + WASM) and DRS correlators already on this line since 0.13.0.

## v0.12.2 (2026-08-05)

- Public Python names only: `write_smiles` / `write_smarts` (removed `write_local_smarts` export)
