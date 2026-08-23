# Consuming molrs from another project

molrs (`molcrafts-molrs`) exposes its data and force-field types through **two
as-built paths**. Pick by what your project is:

| Your project | Path | Crate | Cost |
|---|---|---|---|
| A Rust crate / binary | **Native** | `molcrafts-molrs` (direct dep) | zero-copy by construction |
| A Python / WASM binding | **Handle API** | `molrs-ffi` | zero-copy column borrows across the language boundary |

There is no marshalling layer and no `to_dict`/`from_dict` round-trip — a consumer
holds molrs data directly (native) or through a stable handle (FFI). The reference
Rust consumer, [`molcrafts-molpack`](https://github.com/MolCrafts/molpack), uses the
native path: its `Cargo.toml` depends on `molcrafts-molrs` directly and operates on
`molrs::Frame` / `molrs::ff::ForceField` natively.

---

## Path A — native Rust (depend on `molcrafts-molrs`)

Add the crate, enabling only the sub-systems you need (`core` is always on).
Downstream packages that co-release with molrs (e.g. molpy) pin the shared
**major.minor** line (`>=X.Y.0,<X.(Y+1)`), not an exact patch.

```toml
[dependencies]
molrs = { package = "molcrafts-molrs", version = "0.12", default-features = false, features = ["ff"] }
```

Then use the native types directly — no FFI, no copies. For example, building
evaluable MMFF94 potentials from a molecule (the pattern molpack's relaxer follows):

```rust,no_run
use molrs::Atomistic;
use molrs::ff::potential::intramolecular_pairs;
use molrs::ff::typifier::mmff::MMFF94Typifier;
// UFF: use molrs::ff::typifier::uff::UFFTypifier  (same composition)

let mol = Atomistic::new();                              // build or load your molecule
let typifier = MMFF94Typifier::new();

let mut frame = typifier.typify(&mol)?.to_frame();       // labels + charges
frame.insert("pairs", intramolecular_pairs(&frame));     // the consumer's neighbour list
let potentials = typifier.ff().to_potentials(&frame)?;   // the standard compile path

let coords: Vec<f64> = Vec::new();                       // flat [x,y,z, ...]
let (energy, _forces) = potentials.calc_energy_forces(&coords);
println!("MMFF94 energy = {energy} kcal/mol");
# Ok::<(), String>(())
```

There is no MMFF/UFF shortcut, and that is the point: a force field read from a
file is consumed by exactly these three lines. The typifier's contract is
`typify` — labels and charges — and compiling is `ForceField::to_potentials(&frame)`.
The neighbour list is *yours* because you are the one who knows when it goes
stale: a minimizer that moves atoms decides when to rebuild it, and molrs will
not guess. (WASM `LBFGS` may install a topology pair list when no neighbor list
is supplied — that still lives on the optimizer, not as a free-floating
`optimizeGeometry`.)

(A `MMFF94Typifier::build(&mol)` convenience used to fold all three into one call.
It was deleted — it had, for its whole life, compiled potentials with **no
electrostatic style at all**, because no `ForceField` ever defined
`pair/mmff_ele`; caffeine came out 150 kcal/mol low and nothing noticed, because
the shortcut hid the `Frame` where the missing term would have been visible.
This is the pattern the molpack relaxer follows.)

This exact snippet is compile-checked as the module doctest on
`molrs::ff::typifier::mmff`.

MMFF ships **two** named front doors — `MMFF94Typifier` and `MMFF94STypifier` —
over one engine. Swap the type to swap the parameter set; there is no variant flag.
MMFF94s (Halgren 1999, the "static" set) re-parameterises 11 out-of-plane rows and
42 torsion rows so that delocalised trivalent nitrogen (MMFF types 10 `NC=O` /
40 `NC=C`) minimizes planar; everything else is shared, so a molecule without such
a nitrogen gets bit-for-bit identical potentials from both.

`UFFTypifier` is the third named front door (RDKit-aligned Universal Force
Field). Same composition; no electrostatics.

## Path B — Python / WASM via the `molrs-ffi` handle API

Language binders (`molrs-python`, `molrs-wasm`, the C API) link `molrs-ffi` and hold
a **handle** — a `FrameRef` (a `FrameId` paired with a shared `Store`) — forwarding
every column access through the shared helpers. Numeric columns are borrowed as
contiguous slices (zero-copy); strings are copied (they aren't contiguous scalars).

```rust,no_run
use molrs_ffi::FrameRef;

let frame = FrameRef::new_standalone();          // a frame inside a fresh SharedStore
// ... populate it via frame.with_mut(|f| ...) ...
if let Ok(atoms) = frame.block("atoms") {
    // zero-copy borrow of the uint atom-id column (see the uint-index contract below)
    let n_ids = atoms.borrow_u("id", |ids, _shape| ids.len()).ok().flatten();
    let _ = n_ids;
}
```

`molrs-ffi` exposes `FrameRef`, `BlockRef`, `ForceFieldRef` (under the `ff` feature),
`SharedStore` / `new_shared`, `FrameId`, `BlockHandle`, and one error type `FfiError`.
This snippet is compile-checked as the `molrs-ffi` crate-level doctest.

### ABI contract (cross-extension handle exchange)

Two separately compiled extensions (e.g. the `molcrafts-molrs` wheel and the
`molcrafts-molpack` wheel) may exchange raw `molrs_ffi` handles through
PyCapsules. That is a pointer bridge, so both sides must embed a
**layout-identical** molrs core. The rule — decided project-wide — is:

> **Minor-line = ABI version.** Every downstream shares one molrs minor line.
> Within a minor line the layout of every FFI-crossing type is frozen; a
> layout change requires a minor bump. When molrs moves to a new minor,
> downstream is obliged to re-align.

`molrs_ffi::abi` is the single source of the contract; **never hard-code the
capsule names**:

- `abi::abi_line()` — `major.minor` of the embedded molrs (e.g. `"0.14"`).
- `abi::frameref_capsule_name()` / `abi::forcefield_capsule_name()` —
  `molrs.FrameRef/<line>` / `molrs.ForceFieldRef/<line>`. Versioned since
  0.14 (older lines used the unversioned `molrs.FrameRef`), so a cross-minor
  exchange fails the capsule *name check* — a clean `ValueError` — instead of
  dereferencing a possibly drifted layout.
- `molrs._ffi_abi_token()` (Python) — returns
  `(abi_line, version, frameref_name, forcefield_name)`. A consumer extension
  calls it once at import and raises a clear `ImportError` on a line mismatch
  (molpack's `interop::check_abi` is the reference implementation).

Enforcement on the supply side: `molrs-ffi/src/abi.rs` carries a **layout
snapshot test** (size / align / field offsets of every FFI-crossing type,
committed as `src/layout.snapshot`). Changing any of those layouts within a
minor fails CI; a toolchain update that alone changes the report is treated
the same way (the bridge crosses compiled layouts, not source).

Version combinations:

| producer (molrs wheel) | consumer (e.g. molpack) | outcome |
|---|---|---|
| same minor, any patch | same minor, any patch | **supported** — layout frozen by the snapshot gate |
| ≥0.14 line X | line Y ≠ X | `ImportError` at consumer import (token mismatch) |
| ≥0.14 | pre-handshake (≤0.13) consumer | capsule name mismatch → clean `ValueError` at first resolve |
| ≤0.13 | ≥0.14 consumer | `ImportError` at import (wheel lacks `_ffi_abi_token`) |

Release ordering is unchanged: molrs ships a new minor first; molpy / molpack
re-align and ship after ("Release before molpy" iron law).

---

## Path C — C ABI (`libmolrs_capi`)

The **only sanctioned dynamic-linking deliverable**. External C / C++ / HPC
consumers link `libmolrs_capi` (cdylib or staticlib) against the
cbindgen-generated `molrs.h` — a flat, handle-based C API over frames,
blocks, sim boxes, and force fields (feature surface: always-on core +
perceive, plus `ff`, `io`, `smiles`; storage is a global mutex-protected
store, so treat the library as single-threaded per process).

- **Download**: `molrs-capi-<version>-<platform>.tar.gz` (lib + `molrs.h` +
  LICENSE + sha256) attached to each GitHub Release on `v*` tags.
- **Handshake**: before any other call, compare `molrs_c_api_version()`
  against the `MOLRS_C_API_VERSION` your header was compiled with; the
  constant increments on any breaking signature / handle-semantics change
  (mirrors molrs-cxxapi's `CXX_API_VERSION`). `molrs_version()` reports the
  embedded molrs release for diagnostics.

In-house Rust consumers (molpack, the binders) do **not** link this — Rust
has no stable ABI, so they stay statically linked on Path A/B; the shared
build cache (one `<repo>/target` across all workspace roots) is what removes
the duplicate compilation cost, not dynamic linking.

## Data contract (both paths)

Whichever path you take, molrs data follows these conventions:

- **Atom indices are unsigned** (`u32`, the `UInt` dtype). Index columns —
  `atoms.id`, the `atomi`/`atomj`/`atomk`/`atoml` columns on bond/angle/dihedral
  blocks — are read via `get_uint` (native) / `borrow_u` (handle). Do **not** read
  them as signed.
- **Pairs block schema.** A non-bonded pair list is a block with `atomi`, `atomj`
  (uint) and `is_14` (bool) columns. This is the single pairs convention across the
  force field.
- **`special_bonds` weights live on the `ForceField`**, not in the neighbour list.
  The force field carries the 1-2 / 1-3 / 1-4 LJ and Coulomb scale factors
  (e.g. amber `0/0/0.5` LJ, `0/0/0.8333` Coulomb); a reader fills them.
- **The neighbour list is the consumer's job.** `ForceField` holds parameters +
  `special_bonds` only; the optimizer / integrator builds the intramolecular pair
  list (`molrs::ff::potential::intramolecular_pairs(&frame) → atomi/atomj/is_14`)
  and inserts it before calling `to_potentials`.

## Which path?

- Writing Rust → **Path A**. You get molrs types natively with no boundary cost;
  there is no reason to route through `molrs-ffi`.
- Writing a Python/WASM binding → **Path B**. Hold a `FrameRef`, borrow columns
  through `BlockRef`, and map names with your binding's attribute macros.
