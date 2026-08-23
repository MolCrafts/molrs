---
slug: ffi-shared-dylib
criteria:
  - id: ac-001
    summary: molrs-ffi is a dylib host with ff on by default
    type: code
    pass_when: |
      molrs-ffi/Cargo.toml declares `crate-type = ["dylib", "rlib"]`,
      `default = ["ff"]`, and its molrs dependency carries no
      `default-features = false`; `cargo build --manifest-path
      molrs-ffi/Cargo.toml --release` produces
      target/release/libmolrs_ffi.dylib (or .so on Linux).
    status: pending
  - id: ac-002
    summary: every native build graph resolves the same molrs feature union
    type: code
    pass_when: |
      `cargo metadata --manifest-path molrs/Cargo.toml --format-version 1`
      shows default = ["full","stream","filesystem","rayon"]; and
      `rg 'default-features = false' -g '*/Cargo.toml'` in both repos returns
      exactly three hits — molrs-wasm/Cargo.toml lines 41-42 (its molrs AND
      molrs-ffi deps) and molrs-python/Cargo.toml:38 (its molrs dep) — each
      site carrying an inline comment naming its exemption (browser bundle /
      Pyodide).
    status: pending
  - id: ac-003
    summary: three link probes run and their branch decisions are recorded
    type: runtime
    pass_when: |
      The spec's Design "probe 与两个回退分支" section names the outcome of all
      three probes with the exact command used: (a) an externally launched
      `python -c "import molrs, molpack"` against release-built wheels,
      (b) `cargo build --manifest-path molrs-cxxapi/...` and
      `--manifest-path molrs-capi/Cargo.toml` producing their staticlibs,
      (c) the same under `--release` with lto = "thin"; and each of A/B/C is
      marked 1 (as designed) or 2 (fallback applied, with the applied change).
    status: pending
  - id: ac-004
    summary: build scripts inject the runtime rpath and hard-fail on breach
    type: code
    pass_when: |
      molrs-capi/build.rs, molrs-python/build.rs, molpack/build.rs and
      molpack/python/build.rs each emit `cargo::rustc-link-arg=-Wl,-rpath,<dir>`
      derived from `rustc --print target-libdir` (plus the shared target profile
      dir under branch A2), and `panic!` with a copy-pasteable fix command when
      the probe fails — no `cargo::warning` fallback on that path.
    status: pending
  - id: ac-005
    summary: both Python extensions load one shared libmolrs_ffi
    type: runtime
    pass_when: |
      `bash scripts/verify-shared-dylib.sh` exits 0 after building both wheels
      with the release profile into one venv; the realpath of the libmolrs_ffi
      entry in `otool -L` / `ldd` is identical and non-empty for
      `molrs/_lib*.so` and `molpack/molpack*.so`. Running it with
      MOLPACK_ROOT pointing at a non-existent dir exits non-zero and prints the
      override usage.
    status: pending
  - id: ac-006
    summary: regression example round-trips a Frame capsule across the boundary
    type: runtime
    pass_when: |
      `python regressions/ffi-shared-dylib.py` (run inside the harness venv)
      exits 0: it imports molrs and molpack in one interpreter, builds a
      3-atom Frame via molrs.Block/Frame, passes it to molpack.Target(frame, 1),
      and asserts natoms == 3, elements == ["C","C","O"], count == 1,
      molrs._ffi_abi_token()[0] == "0.14" and the capsule repr contains
      "molrs.FrameRef/0.14" — all hard-coded, no third-party import.
    status: pending
  - id: ac-007
    summary: ABI layout snapshot unchanged and its gate is unconditional
    type: code
    pass_when: |
      `git diff --exit-code molrs-ffi/src/layout.snapshot` is empty, and
      `cargo test --manifest-path molrs-ffi/Cargo.toml
      layout_matches_committed_snapshot` passes with no `--features ff` flag;
      abi.rs carries the `#[cfg(not(feature = "ff"))] compile_error!` guard.
    status: pending
  - id: ac-008
    summary: molrs suites green; the stream delta is accounted for
    type: runtime
    pass_when: |
      `cargo test -p molcrafts-molrs --lib --features full,filesystem` reports
      0 failed and >= 1566 passed; `cargo test --doc -p molcrafts-molrs
      --features full,filesystem` 0 failed and >= 66; `cargo test
      --manifest-path molrs-ffi/Cargo.toml` 18 passed; `uv --directory
      molrs-python run --no-sync tox -e py` 572 passed. The count delta over
      1566 is attributed test-by-test to molrs/src/stream/** in the impl
      summary.
    status: pending
  - id: ac-009
    summary: molpack suites green against the converged graph
    type: runtime
    pass_when: |
      In molpack: `cargo test --features io --lib --tests --examples` +
      `cargo test --test cli --features cli` report 230 passed / 0 failed, and
      `uv run --directory python --group dev tox -e py` reports 132 passed /
      0 failed.
    status: pending
  - id: ac-010
    summary: capi ctest passes against a SHARED import; MOLRS_F64 is gone
    type: runtime
    pass_when: |
      `cargo build --manifest-path molrs-capi/Cargo.toml && cmake -S
      molrs-capi/tests/cpp -B molrs-capi/build-test && cmake --build
      molrs-capi/build-test && ctest --test-dir molrs-capi/build-test
      --output-on-failure` is green, the CMake target is
      `add_library(molrs_capi SHARED IMPORTED)` with BUILD_RPATH set and no
      `-lc`/`-lm` interface list, and `rg -i 'molrs_f64|feature.*\bf64\b'`
      over molrs-capi/ returns nothing.
    status: pending
  - id: ac-011
    summary: every cargo-running workflow pins the static form
    type: code
    pass_when: |
      ci-rust.yml, ci-python.yml, ci-capi.yml, bench.yml, nightly.yml,
      publish.yml (molrs) and ci.yml, bench.yml, publish-crate.yml,
      publish-pypi.yml (molpack) each declare
      `RUSTFLAGS: -C prefer-dynamic=no` at workflow or job level;
      ci-wasm.yml is deliberately unchanged with a comment saying why.
    status: pending
  - id: ac-012
    summary: static override reproduces today's self-contained artifacts
    type: runtime
    pass_when: |
      With `RUSTFLAGS='-C prefer-dynamic=no'`, a release build of
      molrs-python and molpack/python succeeds and `otool -L` / `ldd` on both
      extensions shows no libmolrs_ffi entry; `cargo build --release
      --manifest-path molrs-capi/Cargo.toml` still produces libmolrs_capi.a.
    status: pending
  - id: ac-013
    summary: wasm and Pyodide builds are untouched
    type: runtime
    pass_when: |
      `cargo check --manifest-path molrs-wasm/Cargo.toml --target
      wasm32-unknown-unknown` emits only the already-known warnings (no new
      error), `wasm-pack build --release --target bundler` in molrs-wasm is
      green, and `maturin build --no-default-features` resolves molrs to
      exactly ["full","stream"] (no rayon / filesystem / zarrs in
      `cargo tree`).
    status: pending
  - id: ac-014
    summary: interop and build-cache docs state the reversed contract
    type: docs
    pass_when: |
      docs/interop.md no longer claims in-house Rust consumers "do not link
      this ... they stay statically linked" (old lines 167-170) and instead
      describes the shared libmolrs_ffi host with the local-dynamic /
      CI-static matrix; CLAUDE.md's "Build cache" section names the same
      switch; and `git ls-files .cargo/config.toml` lists the file (it is
      untracked today while CLAUDE.md calls it committed).
    status: pending
  - id: ac-015
    summary: molpack stays publishable to crates.io
    type: code
    pass_when: |
      molpack/Cargo.toml lists molrs-ffi only under [dev-dependencies] with a
      `path` and NO `version` key (path-only dev-deps are stripped at publish),
      and a comment records that the molpack CLI binary stays statically linked
      until molcrafts-molrs-ffi joins the crates.io release train.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** 是收敛的两块地基:没有并集,dylib 只会被静默复制成两份,ac-005 便是空转绿。ac-002 用 grep 把"只剩两处豁免"变成可判定条件,而不是靠人读。
- **ac-003** 是本 spec 唯一带未知的条目:三个 probe 的结论必须落在纸上,分支 A2/B2/C2 一旦触发,ac-004/ac-012 的判定内容随之改变。probe 结果为空 = 不通过。
- **ac-005 + ac-006** 是同一件事的两半:前者断链接身份(同一份 dylib 文件),后者断运行期身份(capsule 跨边界零拷贝且 ABI 线一致)。两者都绿才等于"共享 dylib 真的在工作";只绿其一说明要么共用了库却握手失败,要么握手成功却各嵌一份。
- **ac-007** 保持 minor 线的 ABI 冻结语义不被本 spec 稀释:期望是**字节不变**,任何"刷新快照"的做法都是把 break 洗白。
- **ac-008** 的下界写成 `>=` 且要求逐条归因,是因为 `stream` 首次进入默认门必然抬高计数——不写归因就等于给"多了/少了几条"留了模糊空间。
- **ac-011 + ac-012** 一起守住发布安全:前者是声明,后者是可复现的产物证据。maturin-action 容器透传的一次性验证写在 spec 的 Testing strategy 里,发版前必须执行。
- **ac-015** 是核查中发现、需求原文未覆盖的硬约束(`molcrafts-molrs-ffi` 不在 crates.io);它把"下一次 molpack 打 tag 才炸"的雷提前变成一条可判定条件。
