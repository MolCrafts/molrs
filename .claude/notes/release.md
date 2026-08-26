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

## v0.12.2 (2026-08-05)

- Public Python names only: `write_smiles` / `write_smarts` (removed `write_local_smarts` export)

## v0.13.0 (2026-08-09)

- `stream::Publisher` (was `FrameServer` / `FramePublisher`); serialization routed through `io`
- Structure and force-field readers/writers for the molpy sink
- Publish matrix: 3 OS × Python 3.12/3.13/3.14; split macOS arm64/x64 wheels

## v0.13.1 (2026-08-13)

- NeighborList engine + Neighbors table (Python / WASM)
- `molrs.keys` as Key-typed constants
- DRS correlators; shared FFT/flux primitives
- `Frame.meta` is dict-like (`FrameMeta`)

## v0.13.2 (2026-08-19)

- DCD / XTC / TRR on the `FrameIndexBuilder` streaming surface

## v0.14.0 (untagged — stay on `dev`; tag is 08)

- `UnitPreset` / `UnitPresetRegistry` in `core::units`; zero unit conversion inside MD
- `Potential` and `Compute` as `runtime_checkable` Protocols (Python)
- `MD(dtype=)` experimental (`import molrs.md` emits `FutureWarning`)
- Public record API is `Record` / `Trajectory.read` / `Trajectory.write` (not `MolRec` / `read_zarr`)
- One LJ pair kernel; `VerletSkin::pairs_at` is the only MIC site; PME as pair style `coul/long/pme`
- Identity scalar `Idx = u64` (retired `U = u32`); column storage widths preserved (no f32→f64 / i64→i32 / u64→u32 narrowing)
- WASM domain-uint columns are `BigUint64Array`; JS names stay `setColU32` / `copyColU32` / `viewColU32` / `hasU32`
- wasm `NeighborQuery` symmetry deferred to 0.15 (binder-surface-symmetry note)
