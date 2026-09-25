# Build & cache (molrs)

Project standard for how this repo is compiled and tested locally. Measured on
the 4-core Lustre workstation this repo lives on, 2026-09-22, `molcrafts-molrs`
at 293k lines / 2513 unit tests.

## The numbers that set the rules

| | long-lived `target/` (101 GB) | clean `target/` | clean + `debug = "line-tables-only"` |
|---|---|---|---|
| cold build of the lib test binary | — | 65 s | 59 s |
| edit `core/system/molgraph.rs` | 44-49 s | 6.9 s | 6.2 s |
| edit `ff/params/mmff.rs` (51.7k lines) | 79 s | 7.4 s | — |
| edit `compute/rdf/` | 7.2 s | — | 5.8 s |
| run all 2513 unit tests | 5.3 s | 5.3 s | 5.3 s |
| incremental cache, per feature set | 1.6 GB | 1.4 GB | **420 MB** |

Three things follow, and they are the whole of this note:

1. **Running the tests was never the problem** (5.3 s). Compiling was.
2. **A cluttered `target/` is slower than no `target/` at all.** Reading a
   1.4 GB incremental cache back over Lustre when it is not in page cache costs
   ~40 s; the same command run twice in a row costs 48.7 s then 7.4 s.
3. **A second feature set is a second full compile** of all 293k lines, plus
   another ~420 MB of cache. Feature discipline is a build-time feature.

## Rules

### One command set

Every root-workspace invocation goes through a `cargo mrs-*` alias in
`.cargo/config.toml` — `mrs-build`, `mrs-check`, `mrs-clippy`, `mrs-test`,
`mrs-doctest`, `mrs-doc`, and the OPLS table generator `mrs-gen-opls`. Hooks, CI and CLAUDE.md's `mol_project.build` all
call the aliases, so the feature string exists in exactly one place.

The rule has teeth: `test_single` used to be `cargo test {path}`, which drops
`stream`/`serde` and adds the doctest and bin targets. "Run one test" therefore
cost a full rebuild of the crate every time and doubled the cache. It is now
`cargo mrs-test -- {module}`: the same binary, filtered. (`scripts/test-scope.sh`,
a wrapper that mapped changed files to filters, was deleted 2026-09-25: filtering
saves at most the 5 s suite run, never compile time.)

### Four legitimate builds of `molcrafts-molrs`

1. `full,filesystem,stream` (+ default `rayon`) — every root-workspace command,
   `cargo mrs-gen-opls` included. The OPLS generator links this same library
   unit and adds exactly one example unit (a fingerprint directory holding
   `example-gen_opls_params*`); `mrs-check` / `mrs-clippy --all-targets` add
   their check units for it. It adds no library fingerprint.
2. `full,filesystem,rayon` — what `molrs-capi` and `molrs-cxxapi` link. They
   must not ship `stream` (tokio + tungstenite inside a C archive), so this set
   is separate **on purpose**; do not "unify" it by moving `stream` into `full`.
3. `molrs-ffi`'s `default-features = false` minimal set.
4. the wasm32 build under `molrs-wasm`.

Anything else showing up in `target/debug/.fingerprint/molcrafts-molrs-*` is
drift — find the invocation that spelled its features by hand.

### Debug info

`[profile.dev] debug = "line-tables-only"` and `[profile.dev.package."*"]
debug = false`, repeated in all six roots (a standalone workspace inherits
nothing). Line tables are what a failing unit test's backtrace needs;
full DWARF on a crate this size is what makes the cache 1.4 GB. If you need to
step through molrs in a debugger, override for that run:
`CARGO_PROFILE_DEV_DEBUG=2 cargo mrs-test` — and expect the next build to be a
cold one, because the profile is part of the fingerprint.

### `target/` hygiene

`target/` is a cache. The `target-sweep` pre-push hook measures it (~1 s) and
runs `cargo sweep -t 14` only once it passes 20 GB, so artifacts nothing has
read for two weeks get collected without anyone remembering to do it. To prune
by hand: `cargo sweep -t 14`, or `rm -rf target` — a cold rebuild is 65 s.
Watch for one-shot artifacts the sweep does not reach: `llvm-cov-target/`,
`x86_64-pc-windows-gnu/`, `criterion/` and `link-static-venv/` accounted for
~8 GB of the 101 GB.

If a build feels inexplicably slow, check the cache before blaming the code:

```bash
du -sh target target/debug/incremental
ls target/debug/.fingerprint | wc -l        # ~270 for one feature set
ls target/debug/.fingerprint/molcrafts-molrs-* -d | wc -l
```

Artifacts on node-local disk (`CARGO_TARGET_DIR=/tmp/$USER/molrs-target`) are
immune to the page-cache effect — measured 5.7-11 s for the same edits, never
40 s. It is an env-var-only choice, never committed: `/tmp` is node-local, so
the cache does not follow you to another node, and `.cargo/config.toml` is
shared with the molpack repo.

### One build configuration per phase (2026-09-25)

Inner loop: `cargo mrs-test [-- <module>]` only. At commit: rustfmt + `cargo
mrs-clippy` (molrs only). At push: rustdoc, binder and wasm clippy, doctests,
binder tests, tox, wasm-pack, capi. Every binder links molrs under its own feature
set, target or profile, i.e. one more full compile of molrs each; that is why
they never run in the loop or at commit. Cargo keeps one artifact set per
(features × profile × mode × target) and never deletes stale ones, which is how
`target/` reached 26 GB; there is exactly one `target/` (on this machine a symlink
to local disk, `/tmp/$USER/molrs-target`), never a second `CARGO_TARGET_DIR`,
worktree or copied crate.

### Hook scoping

`.pre-commit-config.yaml` scopes each hook with `files:` instead of
`always_run: true`, so a change confined to one binder does not build the other
four. `molrs/`, the root `Cargo.toml`, `.cargo/` and `rust-toolchain.toml` are
in every scope, because every binder links molrs. `prek run --all-files`
(what `/mol:ship` runs at the push tier) matches every scope, so the CI-parity
gate is unchanged.
