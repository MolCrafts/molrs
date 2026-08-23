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

In-house Rust consumers (molpack, the binders) do **not** go through this C
ABI — they take Path A or Path B directly. How those consumers are *linked*
— static by default, dynamic only on request — is the next section.

---

## Local link form — static by default, dynamic on request

**A zero-argument build is static, everywhere.** Nothing committed selects a
link form: local, CI and publish hand cargo the same set of link flags — namely
none — so the same command produces the same artifact shape on a laptop and in
the release workflow. Every consumer embeds its own molrs image, and every
wheel / crate is self-contained.

The **dynamic** form still exists, as a pure command-line opt-in. Under it,
Path B's crate is not only the handle API but also the physical shared object:
molrs stays an rlib and is linked *into* `libmolrs_ffi`, so every build graph
that reaches molrs-ffi resolves molrs out of that one dylib instead of
embedding its own copy — in one venv the molrs and molpack extensions then
hold one molrs image, not two.

| | static (default) | dynamic (opt-in) |
|---|---|---|
| How you get it | nothing — zero arguments | two flags on the command line, by two different transports — see below (`cargo` and `maturin` do not take them the same way) |
| LTO | `lto = "thin"`, from the manifests | must be turned off |
| Distribution | every wheel / crate self-contained; each downstream publishes independently | lockstep — molrs and *every* downstream rebuild and republish together |
| Compat contract | **major.minor** — patch-compatible, as the ABI section above states | **exact build** — ANY change, a patch included, breaks the pair |

### Asking for the dynamic form

Two things must reach rustc, and they travel by **two different transports**:

| What must reach rustc | Kind | bare `cargo` | `maturin build` |
|---|---|---|---|
| `-C prefer-dynamic` | rustflags | `--config "target.…rustflags=[…]"` | inline `RUSTFLAGS` on that one invocation — `--config` does **not** reach it |
| `lto = false` | profile key | `--config 'profile.release.lto=false'` | the same `--config`, unchanged |

Bare cargo — both by `--config`:

```
cargo build --release \
  --config "target.'cfg(not(target_arch = \"wasm32\"))'.rustflags=['-C','prefer-dynamic']" \
  --config 'profile.release.lto=false'
```

maturin — rustflags by `RUSTFLAGS`, profile key by `--config`:

```
RUSTFLAGS='-C prefer-dynamic' maturin build --release \
  --config 'profile.release.lto=false'
```

Set `RUSTFLAGS` *inline* on the maturin command, never exported: unlike the
`--config` form it carries no `cfg(not(target_arch = "wasm32"))` guard, so its
whole safety is that it lives and dies with one native wheel build.

Neither of the two is optional. `prefer-dynamic` is what makes consumers
resolve molrs out of the one dylib rather than embedding an rlib copy. Clearing
`lto` is forced by it: the seven native manifests carry `lto = "thin"` in
`[profile.release]`, and rustc refuses LTO the moment a Rust dylib enters a
cdylib's graph — `only 'staticlib', 'bin', and 'cdylib' outputs are supported
with LTO`. A cargo `--config` profile key is the only thing that overrides a
*manifest* profile in place.

**Why the transports differ — measured, do not "simplify" it back into one
line.** maturin unconditionally sets `CARGO_ENCODED_RUSTFLAGS` for the build it
launches, and env-level rustflags *replace* config-level rustflags wholesale —
including rustflags supplied by `--config`. With maturin 1.13.3:

- `scripts/verify-shared-dylib.sh` carrying both `--config` flags on both
  `maturin build --release` calls exits 1 with `FAIL — no libmolrs_ffi
  dynamic-link entry`: **both** extensions came out static.
- `maturin build --release -v` with those flags shows the `molrs_python` rustc
  invocation carrying **no** `-C prefer-dynamic`, while being offered both
  `--extern molrs_ffi=…libmolrs_ffi.dylib` and `…libmolrs_ffi.rlib`. Without
  `prefer-dynamic` rustc takes the rlib — that is the static outcome. maturin's
  own `-C link-arg=-undefined -C link-arg=dynamic_lookup` *are* on that line:
  the injection that did the replacing.
- Counter-probe: `RUSTFLAGS='-C prefer-dynamic' maturin build --release
  --config 'profile.release.lto=false' -v` puts `-C prefer-dynamic` on the
  `molrs_python` invocation. maturin **appends** its link-args to an inherited
  `RUSTFLAGS` instead of replacing it.
- `maturin build --help` offers `--config <KEY=VALUE>` and no rustflags /
  extra-args entry point, so there is no third transport to look for.

This is the same precedence rule the régime warning below describes, and it did
not bite while the flag lived in a committed `.cargo/config.toml`: maturin
*reads* that file and merges it into the rustflags it computes, whereas a
`--config` override on the command line never enters that computation.

`--config <KEY=VALUE|PATH>` is a stable, official cargo flag and maturin 1.13.3
exposes it directly (no `--` passthrough); it simply cannot carry rustflags.
The **static default** needs no environment variable at all — it takes zero
arguments — and no project-invented variable exists anywhere in either repo.
The dynamic opt-in borrows cargo's own `RUSTFLAGS`, for maturin builds only,
confined to `scripts/verify-shared-dylib.sh`: never exported, never in a
workflow, never written back into a `.cargo/config.toml`. There is still no
cargo alias, no wrapper script and no second target dir — all artifacts stay in
the one shared `target/`. Do not re-commit any of this: a committed switch is
the drift this arrangement removes.

`scripts/verify-shared-dylib.sh` is the canonical runner and carries the split
itself (`DYN_RUSTFLAGS` for the rustflags transport, `DYN_CONFIG` for the
profile key). The pre-push gate builds both wheels under the opt-in and proves
the shared dylib is still real, so a developer normally invokes the gate rather
than typing any of the above.

### Why the dynamic contract is "exact build", not "minor line"

Measured with cargo 1.96: an empty crate, source unchanged, version bumped
`0.1.0` → `0.1.1`. The exported symbol hash moved from
`…17h7f515c415611beecE` to `…17h7f9d024041394482E`. The version string is an
input to the symbol hash, so a *patch* release renames every exported symbol.
"Dynamic is compatible along the minor line" is therefore false: a dynamically
linked pair is only valid for the exact builds it was produced from, which is
why the dynamic column above demands a coordinated rebuild-and-republish of
every downstream.

A runtime handshake cannot rescue a mismatched dynamic pair either. dyld fails
at *load*, so no self-check of ours ever runs — the `_ffi_abi_token()` /
capsule-name handshake described above can only fire in the static form, where
both extensions import successfully in the first place. If the dynamic form is
ever used for publishing, the only correct refusal point is **pip install
time**: molrs would carry a build-id local version (e.g. `0.14.1+abi.<hash>`)
that every downstream pins with `==`, making a mismatched pair *unresolvable*
instead of unloadable.

> **TODO — NOT WIRED.** No build-id local version is emitted today and nothing
> pins one; this is deliberately out of scope. Until it is built, the dynamic
> form is a local / gate-only form and must not be published.

### Preconditions for the opt-in

One dylib is real only if every native root resolves the *same* molrs unit:

- identical feature set including the literal `default` label — hence the
  `molrs-default` forward feature on `molrs-ffi` / `molrs-python`;
- `[profile.release]` byte-identical across the seven native roots enumerated
  by `scripts/sync-dylib-locks.sh`'s `ROOTS` array; the profile is part of
  cargo's unit fingerprint, so one stray key mints a second molrs unit behind
  the single hashless dylib name;
- no lockfile skew across those roots — repaired by
  `scripts/sync-dylib-locks.sh`, which deletes all seven `Cargo.lock`s and
  regenerates them in one registry sweep;
- third-party feature widening anchored by `molrs-ffi`'s default-on `unify`
  pins.

**Exemptions**, all deliberate and all static: wasm32 (`molrs-wasm`) is
cfg-exempt — the opt-in's `cfg(not(target_arch = "wasm32"))` key does not match
it, and wasm has no dynamic linking anyway; Pyodide builds
`--no-default-features` and is not part of the dylib graph; `molrs-cxxapi` is a
`staticlib` (Atomiverse's delivery form), so it takes the rlib path by
construction. Released wheels and crates are static because static is the
default now, not because anything overrides it.

**Under the opt-in, do not mix a bare `cargo build --release` of `molrs-ffi`
with a maturin wheel build.** maturin still injects its own
`CARGO_ENCODED_RUSTFLAGS` — on macOS it appends
`-C link-arg=-undefined -C link-arg=dynamic_lookup` to whatever rustflags it
inherits from the environment and from `.cargo/config.toml`, which is exactly
why the opt-in above hands it `RUSTFLAGS` rather than a `--config` rustflags
key — and rustflags are a fingerprint input, so the two regimes are two
units fighting over the same *hashless*
`target/release/deps/libmolrs_ffi.dylib`. Last writer wins. The failure is
loud, not silent: the loser's consumer stops with `error[E0463]: can't find
crate for molrs_ffi`, and cargo does not self-heal (it considers its own unit
fresh). Recovery is `touch molrs-ffi/src/lib.rs`, then rebuild under whichever
regime you want. What changed with the static default is the *scope*, not the
mechanism: a zero-argument build resolves molrs-ffi through a hash-suffixed
rlib, and rlibs from different rustflags regimes coexist in the shared target
untouched. Only the dynamic form routes through the one hashless dylib, so this
now bites the developer who opted in rather than everyone by default.

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
