---
slug: link-mode-static-default-01-invert
criteria:
  - id: ac-001
    summary: Both .cargo/config.toml keep only [build] target-dir
    type: code
    pass_when: |
      `grep -rn -E "rustflags|prefer-dynamic" molrs/.cargo/config.toml
      molpack/.cargo/config.toml` returns 0 hits, and each file still
      contains its `[build]` + `target-dir` lines.
    status: verified
    last_checked: 2026-08-23
  - id: ac-002
    summary: Zero-argument build emits no prefer-dynamic (with liveness proof)
    type: runtime
    pass_when: |
      TWO MEASURED DEFECTS in the naive form, both fixed here.

      (1) VACUITY (grill): on an up-to-date tree `cargo build -v` prints ZERO
      rustc invocations, so `grep -c prefer-dynamic == 0` passes while the
      tree still carries the flag. The check must prove it looked.

      (2) FALSE GOLDEN (impl, full-graph rebuild): cargo passes
      `-C prefer-dynamic` to EVERY proc-macro unit unconditionally — they are
      dlopened into the compiler process. Measured on a from-scratch build of
      this graph: 187 rustc invocations, 12 carrying the flag, and all 12 are
      `--crate-type proc-macro` (async_trait, auto_impl, bytemuck_derive,
      derive_more_impl, futures_macro, monostate_impl, paste, serde_derive,
      serde_repr, thiserror_impl, tokio_macros, zerocopy_derive). A bare
      count-must-be-zero golden is therefore UNSATISFIABLE, whatever the
      config says. Exclude cargo's own behaviour instead of relaxing the
      golden:

        LOG=$(mktemp)
        cargo build -v --target-dir "$(mktemp -d)" >"$LOG" 2>&1
        RUSTC=$(grep -c 'Running .*rustc' "$LOG" || true)
        DYN=$(grep 'prefer-dynamic' "$LOG" | grep -vc 'crate-type proc-macro' || true)

      passes only when RUSTC > 0 (liveness) AND DYN == 0 (no NON-proc-macro
      unit gets the flag). A scratch --target-dir forces the whole graph so
      the liveness count is meaningful; from the molrs root, no flags, no env.
      VERIFIED 2026-08-23: RUSTC=187, DYN=0.
    status: verified
    last_checked: 2026-08-23
  - id: ac-003
    summary: Seven native roots share one [profile.release] incl. lto = "thin"
    type: code
    pass_when: |
      Extracting the `[profile.release]` block from each of the seven roots
      listed in scripts/sync-dylib-locks.sh ROOTS (molrs, molrs-ffi,
      molrs-python, molrs-capi, molrs-cxxapi, molpack, molpack/python) and
      hashing it yields exactly ONE distinct sha256, and that block contains
      `lto = "thin"`. molrs-wasm/Cargo.toml still has no [profile.release].
    status: verified
    last_checked: 2026-08-23
  - id: ac-004
    summary: No env regime left in either repo's workflows
    type: code
    pass_when: |
      From /Users/roykid/work/molcrafts:

        grep -rn -E 'RUSTFLAGS|CARGO_PROFILE_RELEASE_LTO' \
             molrs/.github/workflows molpack/.github/workflows

      returns 0 hits (grep errors on a bad path, so a typo cannot pass as
      zero). Liveness: the same command at HEAD before the fix reports the 10
      known env blocks — record that count. Separately, the two tox
      `pass_env` lists (molrs-python/pyproject.toml,
      molpack/python/pyproject.toml) still contain `RUSTFLAGS` and no longer
      contain `CARGO_PROFILE_RELEASE_LTO`, and
      a scan of added lines (`git diff -U0`) shows no newly introduced
      PROJECT-OWNED environment variable. Prose that names the standard cargo
      variable RUSTFLAGS (e.g. verify-shared-dylib.sh's remediation text
      warning that an exported RUSTFLAGS replaces config rustflags wholesale)
      is NOT a breach — the constraint is about variables this project
      invents, not about mentioning cargo's own.
    status: verified
    last_checked: 2026-08-23
  - id: ac-005
    summary: Static regression proves the default extension has no dylib entry
    type: runtime
    pass_when: |
      A molrs wheel built with NO flags (`maturin build --release`), installed
      into a clean venv, makes
      `python regressions/link-mode-static-default.py` exit 0: it reports
      `libmolrs_ffi` dynamic-link entry count == 0 for molrs._lib (otool -L on
      Darwin, ldd on Linux) and its hard-coded goldens hold
      (ELEMENTS == ["C","C","O"], x[1] == 1.5). No third-party tool is invoked
      at runtime beyond the platform linker inspector.
    status: verified
    last_checked: 2026-08-23
  - id: ac-006
    summary: Dynamic opt-in still provably works from the gate script
    type: runtime
    pass_when: |
      `bash scripts/verify-shared-dylib.sh` exits 0 with no env overrides; the
      script itself passes both `--config` flags on both `maturin build
      --release` calls; its output shows a libmolrs_ffi entry for BOTH
      molrs._lib and molpack.molpack, and one identical sha256 after the molrs
      wheel and after the molpack wheel.
    status: verified
    last_checked: 2026-08-23
  - id: ac-007
    summary: Opt-in flags exist in exactly two committed places
    type: code
    pass_when: |
      MEASURED DEFECT (2026-08-23, grill): `molrs/**/Cargo.toml` does NOT
      expand under bash (no globstar) — the original form searched nothing and
      would have passed silently. Use one recursive search over both repos
      with target/ and .git excluded, and prove liveness by requiring the two
      allowed sites to be FOUND:

        grep -rn --exclude-dir=target --exclude-dir=.git 'prefer-dynamic' \
             /Users/roykid/work/molcrafts/molrs /Users/roykid/work/molcrafts/molpack

      passes only when every hit is in exactly two files —
      molrs/scripts/verify-shared-dylib.sh and molrs/docs/interop.md — with
      BOTH present (a zero-hit result is a FAILURE, not a pass), and no hit in
      any .cargo/config.toml, any .github/workflows/*.yml, or any Cargo.toml.
      Hits inside .claude/specs/ are excluded (the spec text quotes the flag).
    status: verified
    last_checked: 2026-08-23
  - id: ac-008
    summary: All 17 dangling ffi-shared-dylib.md citations are gone
    type: code
    pass_when: |
      MEASURED DEFECT (2026-08-23, grill): the unscoped search hits 16 sites,
      three of which are THIS spec's own prose naming the deleted file — the
      criterion could never pass before close. Scope it out:

        grep -rn --exclude-dir=target --exclude-dir=.git --exclude-dir=specs \
             'ffi-shared-dylib\.md' \
             /Users/roykid/work/molcrafts/molrs /Users/roykid/work/molcrafts/molpack

      passes at 0 hits. Liveness baseline MEASURED at HEAD (git grep, specs
      excluded): **17**, not the 13 the draft claimed — molrs 13
      (CLAUDE.md:227, Cargo.toml:7 + :42, molrs-capi/Cargo.toml:23,
      molrs-capi/build.rs:100, molrs-capi/tests/cpp/CMakeLists.txt:38,
      molrs-cxxapi/Cargo.toml:22 + :35, molrs-ffi/Cargo.toml:37 + :79,
      molrs-python/Cargo.toml:92, molrs-python/build.rs:7,
      scripts/verify-shared-dylib.sh:237) and molpack 4 (Cargo.toml:205,
      build.rs:7, python/Cargo.toml:49, python/build.rs:7). A zero-hit pass
      is only credible against this recorded baseline.
    status: verified
    last_checked: 2026-08-23
  - id: ac-009
    summary: Four build.rs still emit the libstd rpath, with corrected prose
    type: code
    pass_when: |
      molrs-python/build.rs, molrs-capi/build.rs, molpack/build.rs and
      molpack/python/build.rs each still contain
      `cargo::rustc-link-arg=-Wl,-rpath,` and their comments no longer claim a
      prefer-dynamic default in .cargo/config.toml; molrs-cxxapi/build.rs is
      unchanged.
    status: verified
    last_checked: 2026-08-23
  - id: ac-010
    summary: Both repos' default suites are green under the static default
    type: runtime
    pass_when: |
      With no flags/env: molrs `cargo test -p molcrafts-molrs --lib --features
      full,filesystem` and `cargo test --doc -p molcrafts-molrs --features
      full,filesystem` pass; molpack `cargo test --features io --lib --tests
      --examples` passes; `tox -e py` passes in molrs-python and in
      molpack/python.
    status: verified
    last_checked: 2026-08-23
    deviation: |
      molpack `tox -e py` could NOT be run: it fails at dependency RESOLUTION
      (`No matching distribution found for molcrafts-molpy<0.15,>=0.14.0`;
      PyPI tops out at 0.13.2) — the un-published molpy 0.14 of the release
      train, identical to the ffi-shared-dylib spec's ac-009 deviation.
      Proof it is unrelated to this change: the log contains ZERO
      "test session starts" lines, i.e. pytest never began. Every other leg
      ran and passed: molrs fmt/clippy x2 green, molrs 1586 lib + 66 doc,
      molpack 226 rust, molrs-python tox 572 passed. Re-run this one leg when
      molcrafts-molpy 0.14 publishes.
  - id: ac-011
    summary: interop.md states the two-form contract and the un-wired pip TODO
    type: docs
    pass_when: |
      docs/interop.md § "Local link form" table has a static row (zero
      arguments, LTO'd, patch-compatible) and a dynamic row naming both
      --config flags; the section states the measured 0.1.0->0.1.1 symbol-hash
      fact, states that dynamic requires every downstream rebuild+republish on
      ANY change, keeps an opt-in-scoped version of the maturin-regime warning,
      and marks the pip build-id `0.14.1+abi.<hash>` `==` mechanism as NOT
      WIRED.
    status: pending
  - id: ac-012
    summary: CLAUDE.md and INDEX.md no longer assert the dynamic default
    type: docs
    pass_when: |
      molrs/CLAUDE.md's link-form paragraph describes static as the
      zero-argument default and dynamic as a command-line --config opt-in
      (frontmatter build.check/build.test/ci.local unchanged), and
      .claude/specs/INDEX.md line 37 no longer says CI/发布 env 钉回静态.
    status: pending
---

# Acceptance criteria

ac-002 / ac-005 / ac-006 是三条互为反向的产物断言:没有参数时**不能**出现
动态条目(ac-002、ac-005),带上两个 flag 时**必须**出现(ac-006)。任一方向缺失
都会让另一方向变成不可能失败的测试。ac-007 是"不得提交为开关"这条硬约束的
唯一可执行表达。
