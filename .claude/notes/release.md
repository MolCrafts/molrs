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

Re-run the failed tag workflow, or dispatch Publish against the same tag
(idempotent skips). Branch dispatches run CI and build artifacts without
publishing. Tags must match the root package version and be on master.
Registry publications wait for CI. The public checklist is [docs/releasing.md](../../docs/releasing.md).

## scripts/

Fixture fetching and the optional shared-library verification scripts live
here. No publish helper scripts; publishing stays in the workflow.

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
- `frame.meta` is a live write-through mapping of **plain** Python values (`FrameMeta`), not a `dict` snapshot of `MetaValue` boxes. The dtype belongs to the key: writing a plain value to an existing key keeps that key's dtype and refuses one it cannot hold, so `m[k] = m[k]` is an identity; assign a `MetaValue` to give a key a different dtype and read the tag back with `meta.dtype(k)`. A JSON document is returned decoded, so nested edits are read-modify-write. `None` stores as JSON null (it used to raise). `Frame.from_dict` is removed — `Frame(blocks=..., meta=...)` covers it, and `dict(frame.meta)` round-trips.
- One LJ pair kernel; `VerletSkin::pairs_at` is the only MIC site; PME as pair style `coul/long/pme`
- Identity scalar `Idx = u64` (retired `U = u32`); column storage widths preserved (no f32→f64 / i64→i32 / u64→u32 narrowing)
- WASM domain-uint columns are `BigUint64Array`; JS names stay `setColU32` / `copyColU32` / `viewColU32` / `hasU32`
- wasm `NeighborQuery` symmetry deferred to 0.15 (binder-surface-symmetry note)
- `Region` trait gains `distance` / `distance_grad` (negative inside); one type per shape, outside is `NotRegion` — `HollowSphere` removed (`Sphere & ~Sphere`; molpy's `__init__` re-export dropped in lockstep, 2026-09-14); boundaries closed (`Parallelepiped` was half-open); new `HalfSpace`, `Cylinder`, `Ellipsoid`, `Polyhedron` (watertight `TriMesh`), `SphereUnion` (atoms as a region, periodic minimum image); Python `TriMesh`, `io.read_stl`, `distance` on every region class, composed-`Region` pickling as an object tree (was a JSON recipe)
- `molrs_ffi::RegionRef` + capsule `molrs.RegionRef/<abi_line>` (`_ffi_regionref_capsule()` on every region class); `_ffi_abi_token()` is a 5-tuple (consumers read indices 0–1)
- prmtop-derived force fields declare `lj/cut` + `coul/cut` with explicit `coulomb`/`dielectric`/`cutoff` (`AMBER_COULOMB = 18.2223²`). A LAMMPS include written **with** its header changes from `pair_style lj/cut/coul/long 10 10` to `pair_style lj/cut/coul/cut 9 10`; `pair_coeff` lines are unchanged. NBFIX / 12-6-4 / multi-term-improper / non-uniform-SCEE prmtops are refused.
