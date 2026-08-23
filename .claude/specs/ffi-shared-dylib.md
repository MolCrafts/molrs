---
title: molrs-ffi 升级为共享动态库宿主(单一并集 feature 收敛)
status: in-progress
created: 2026-08-22
---

# molrs-ffi 升级为共享动态库宿主(单一并集 feature 收敛)

## Summary

把 `molcrafts-molrs-ffi` 从纯 rlib 变成 `["dylib","rlib"]` 的共享动态库宿主,并让所有原生构建图(molrs 根 workspace、molrs-capi、molrs-cxxapi、molrs-python、molpack 根 + molpack/python)收敛到**同一个 molrs feature 并集**(`full + stream + filesystem + rayon`,仅刨除 `blas` / `slow-tests`),配合两仓 `.cargo/config.toml` 里的 `-C prefer-dynamic` 与消费方 build.rs 注入的 libstd rpath,使本地开发中 molrs 与 molpack 的两个 Python 扩展在同一 venv 内加载**同一份** `libmolrs_ffi.dylib`,而不是各自静态嵌入一份 molrs。CI 与发布侧用 `RUSTFLAGS='-C prefer-dynamic=no'` 覆盖回今天的静态形态,产物与现状等价。本 spec 带一个**必须写清的行为变化**:今天发布的 molrs wheel 因 `molrs-python/Cargo.toml:38` 的 `default-features = false, features = ["full","stream"]` 从未启用 `rayon`,10 处 `#[cfg(not(feature = "rayon"))]` 顺序回退路径一直在跑;并集收敛后原生 wheel 转并行(线程数、归约顺序、末位浮点求和次序随之改变),同时 `stream` 模块首次进入默认 clippy/test 门。ABI 侧完全不动:capsule 名、minor-line 握手、`layout.snapshot` 字节均保持不变——验收断言的是 **unchanged**,不是 refresh。

## Domain basis(工程事实依据;本 spec 无物理内容,故无量纲/文献引用)

1. **Rust 无稳定 ABI**,所以一份 Rust dylib 只在同一个 `(rustc, features, profile)` 指纹圈内可复用。因此"feature 并集收敛"不是优化,而是共享 dylib 的**前置条件**:两个构建图若 molrs feature 不同,rustc 会各自生成不同 SVH 的 molrs 实例,退化成静默的双份静态嵌入(不报错、不冲突,只是白做)。
2. **rustc dependency-format 规则**:当图中某个 dylib(`libmolrs_ffi.dylib`)已把上游 rlib(molrs)静态包含进去,下游 bin/cdylib 不会再嵌入第二份,而是从该 dylib 解析这些符号。这正是"molpack 加 molrs-ffi 依赖即可让其测试/例子入圈"的机制。
3. **已实测(沙盒)**:(i) `wasm32` 目标下 `dylib` crate-type 只产生警告,不报错;(ii) bin 产物 + `-C prefer-dynamic` 正确链接 Rust dylib,并对 libstd 生成 `@rpath/libstd-*.dylib` 引用;(iii) 用 `rustc --print target-libdir` 注入 rpath 后可正常运行。
4. **未实测(本 spec 第一任务必须先 probe)**:(a) cdylib 产物(maturin 的 `molrs/_lib.abi3.so`、molpack 扩展)在 prefer-dynamic 下是否动态链 Rust dylib **且能被外部 python 进程加载**;(b) staticlib 产物(`molrs-cxxapi`、`molrs-capi` 的 `.a`)在默认 prefer-dynamic 下能否构建(staticlib 要求全静态自包含);(c) `[profile.release]` 的 `lto = "thin"` 与 `-C prefer-dynamic` 是否兼容(rustc 有 "cannot prefer dynamic linking when performing LTO" 这一类拒绝路径),而验收恰恰要求 release profile。三个 probe 的结果决定后续任务的最终形态,分支写在 Design。
5. **cargo 优先级**:环境变量 `RUSTFLAGS` 整体覆盖 `target.<cfg>.rustflags`。本 spec 一律用**显式** `RUSTFLAGS='-C prefer-dynamic=no'` 而非空串覆盖,避免依赖"空字符串算不算已设置"的实现细节。
6. **rayon 债(实测计数)**:`rg 'cfg\(not\(feature = "rayon"\)\)' molrs/src` = **10 处 / 10 文件**(`optimize/mod.rs`、`compute/{msd,van_hove,debye,density/spatial,hbond/detect,hbond/lifetime,order/reorientation_legendre,voronoi/radical,voronoi/integrate}`);`rg 'feature = "rayon"' molrs/src` = **77 处 / 37 文件**(含 `core/spatial/neighbors/{mod,linkcell}.rs`、`core/math/mod.rs`)。现行 wheel 全部走的是 `not(rayon)` 分支。
7. **动态链接的代价**:跨 dylib 边界没有跨 crate 内联/LTO。因此两仓的 bench workflow 必须保持静态形态,否则性能历史不可比;本地动态构建的**运行时性能不代表发布形态**。

## Design

### 目标形态

`molrs-ffi` 是仓内唯一的 ABI 属主,现在多一种**链接形态**(dylib),不新增任何 Rust 模块、不新增公开符号。molrs 本身仍是 rlib——它的代码被静态收进 `libmolrs_ffi.dylib`,下游经由第 2 条机制复用。所有原生消费者的 molrs feature 集必须逐字相同:

- `molcrafts-molrs` 的 `default` 从 `["rayon"]` 改为 `["full","stream","filesystem","rayon"]`(`filesystem → zarr`)。`blas`(需系统 BLAS)与 `slow-tests` 明确不进并集;`blas` 的 `cfg(not(...))` 分支行为不变。
- `molrs-ffi`:`[lib] crate-type = ["dylib","rlib"]`,`default = ["ff"]`,其 molrs 依赖删 `default-features = false`。
- `molrs-capi` / `molrs-cxxapi` / `molpack` 根 / `molpack/python`:删 `default-features = false`(各自原有的 `features = [...]` 保留,与并集相加是幂等的)。
- **两处豁免**:`molrs-wasm/Cargo.toml:41-42` 原样保留 `default-features = false`(浏览器包体);`molrs-python/Cargo.toml:38` 原样保留 `default-features = false, features = ["full","stream"]`,并集里缺的 `rayon`/`filesystem` 改由 molrs-python **自有的 default-on forward feature** 表达(`default = ["fs","rayon"]`,新增 `rayon = ["molrs/rayon"]`)。于是:原生 wheel 的 molrs feature 集 **== 并集**;Pyodide 的 `maturin --no-default-features` 回落到最小集 `full+stream`,libz-sys/rayon 都不进 emscripten 构建。

### 静 / 动开关矩阵

| 场景 | rustflags 来源 | 链接形态 |
|---|---|---|
| 本地 `cargo` / `maturin`(非 wasm32) | 两仓 `.cargo/config.toml` 的 `[target.'cfg(not(target_arch = "wasm32"))'] rustflags = ["-C","prefer-dynamic"]` | **动态**:`libmolrs_ffi.dylib` + `@rpath` libstd |
| 本地 wasm32(molrs-wasm / Pyodide) | cfg 不匹配 → 无 | 静态,与今天完全一致 |
| CI(ci-rust / ci-python / ci-capi / bench) | workflow 级 `env: RUSTFLAGS: -C prefer-dynamic=no` | **静态**,等价现状 |
| 发布(publish.yml / nightly.yml / molpack publish-*) | 同上(tox 已在 `molrs-python/pyproject.toml:79` 放行 `RUSTFLAGS`) | **静态**,wheel/tarball 自包含 |
| molrs-capi 对外产物、molrs-cxxapi staticlib | 同上 | 静态,`docs/interop.md` Path C 契约不变 |

### probe 与两个回退分支

任务 1 先跑三个 probe,结果**写回本 spec 的本节**再继续:

- **(a) cdylib 外部加载** → 分支 **A1**(通过):build.rs 只注入 libstd rpath(`rustc --print target-libdir`)。分支 **A2**(外部 python 进程找不到 `@rpath/libmolrs_ffi.dylib`,因为 cargo 只为自己 spawn 的进程设 `DYLD_FALLBACK_LIBRARY_PATH`):同一个 build.rs 追加第二条 `-rpath`,指向共享 target 的 profile 目录(从 `OUT_DIR` 上溯到 `<target>/<profile>`)。A2 是预期概率更高的分支。
- **(b) staticlib 构建** → 分支 **B1**(通过,因为 molrs-ffi 同时产出 rlib,staticlib 可取 rlib 一路静态):不动。分支 **B2**(rustc 拒绝):`molrs-cxxapi` / `molrs-capi` 的本地构建命令与其 pre-push hook 前缀 `RUSTFLAGS='-C prefer-dynamic=no'`;若仍不可行,则 `molrs-capi` 去掉 `staticlib` crate-type(发布侧本来就静态构建,不受影响),`molrs-cxxapi` 的 staticlib 不可去——它是 Atomiverse 的交付形态,那就以命令前缀方式豁免并在 `docs/interop.md` 与 wasm 并列文档化。
- **(c) release + `lto = "thin"` + prefer-dynamic** → 分支 **C1**(通过):profile 保持 lto。分支 **C2**(rustc 拒绝):本地动态形态的 `[profile.release]` 去掉 `lto`(CI/发布走 RUSTFLAGS 静态路径,lto 依然生效),并在四个 root 的 profile 注释里写明原因。

### probe 结论(任务 1 完成后回填)

- (a): 待填
- (b): 待填
- (c): 待填

### profile 必须逐字对齐(否则共享 dylib 会互相顶掉)

五个 workspace root(molrs 根、`molrs-python`、`molrs-capi`、molpack 根、`molpack/python`)各自声明 `[profile.release]`;profile 参与指纹,不一致会让 `target/release/deps/libmolrs_ffi-<hash>.dylib` 出现两份,而被 uplift 到 `target/release/libmolrs_ffi.dylib` 的那份取决于谁最后构建——`@rpath` 在运行时解析的正是这个无 hash 名字,于是"同一份 dylib"的不变量会随机破裂。因此把 `lto` / `codegen-units` / `opt-level` / `strip` 四项在五处对齐(以 `molrs-python` 现有值为准:`thin` / `1` / `3` / `symbols`,C2 分支下去掉 `lto`),`molpack/python/Cargo.toml` 需**新增** `[profile.release]`。

### molpack 根依赖:一个 librarian 未覆盖的硬约束(iron law 上报)

原始需求是"molpack 根加 molrs-ffi 依赖,使 CLI/测试入圈"。核查发现:**`molcrafts-molrs-ffi` 从未发布到 crates.io**(`.github/workflows/publish.yml` 只发 `molcrafts-molrs`;`.claude/notes/release.md` 的发布表同样只有一行 crates.io)。而 molpack 根是本链条里**唯一**会 `cargo publish` 的 crate(`publish-crate.yml`)。若加成普通依赖,下一次 molpack 打 tag 会直接 publish 失败——一个只在发布日才炸的静默雷。

决策:molpack 根以 **path-only `[dev-dependencies]`**(带 `path`、**不带** `version`)引入 molrs-ffi。cargo 在打包时会剥离 path-only dev-dependency,molpack 的可发布性零风险;`cargo test` / `cargo bench` / `--examples` 全部入圈(molpack pre-push 的 `cargo test --features io --lib --tests --examples` 与 `cargo build --benches` 即刻覆盖)。**代价必须写明**:`cargo build --features cli` 产出的 `molpack` CLI 二进制**不入圈**,仍静态嵌 molrs,直到 `molcrafts-molrs-ffi` 进入 crates.io 发布火车(已在 Out of scope,届时一行改成普通依赖即可)。

### 影响面(逐条,不许沉默)

- **`stream` 首次进入默认门**:今天 `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem` 与 `cargo test --lib --features full,filesystem` 都**没有**覆盖 `molrs/src/stream/**`(`stream` 不在 `full` 里)。并集后它进入 clippy(`-D warnings`)与单测门,实测该目录有 **20 个 `#[test]`/`#[tokio::test]`**。首跑若冒出 clippy 警告,按 iron law **就地修**,不得放宽门。
- **原生 wheel 转并行**:见 Domain basis 6。归约次序改变意味着浮点末位可能变化,`molrs-python` 572 条 pytest 与 molpack 132 条须逐条绿;任何"因并行而失败"的断言是真缺陷,不得放宽容差(先查该断言是否本就依赖顺序)。
- **编译面变宽**:所有原生消费者现在都编译 `stream`(tokio / tungstenite)与 `filesystem`(zarrs)。`molrs-cxxapi` 是根 workspace 成员,`cargo clippy -p molcrafts-molrs-cxxapi --all-targets` 的**编译**时间随之上升(它只 lint cxxapi 自身代码,不 lint 依赖,故新增 lint 面主要来自 molrs 自身的 `stream`)。这是编译成本,不是 break。
- **`.cargo/config.toml` 当前 UNTRACKED**:`CLAUDE.md` "Build cache" 一节写着"the committed `.cargo/config.toml`",而该文件今天并未被 git 跟踪——即任何新 clone 都**没有**共享 target 缓存,文档陈述为假。本 spec 顺手 `git add` 并修正该节措辞(iron law:发现即修)。
- **molpack CI 把 sibling 钉在 `MOLRS_GIT_REF: v0.14.0`**,所以 molpack CI 侧的收敛要等 molrs 发布该线后才生效;本地(dev sibling)立即生效。CI 的 `MOLRS_GIT_REF` bump 属发布火车,已在 Out of scope。
- **molrs 的 pre-push 将依赖 sibling molpack**(共享 dylib 是跨仓不变量,单仓无法观测)。脚本采取 refuse-and-explain:缺 sibling 时**非零退出**并打印 `MOLPACK_ROOT=` 覆盖用法与 clone 命令,**不做静默 skip**(静默 skip 正是本仓 `feedback_tests_that_cannot_fail` 明令禁止的空转绿)。
- **bench 必须留静态**:见 Domain basis 7。
- **maturin-action / cibuildwheel 容器透传**:`publish.yml` 的 linux wheel leg 在 manylinux 容器内构建。落地时必须确认 `RUSTFLAGS` 透传进容器(否则容器内仍读到仓内 config → 动态 wheel)。这是发布安全项,写进 Testing strategy 的发布前置动作。

### Reuse decision(逐条解决 librarian_report)

- `.cargo/config.toml`(两仓)— **generalize**:在已有文件追加 `[target.'cfg(not(target_arch = "wasm32"))'] rustflags`,不新建文件;molrs 侧同时 `git add`。
- `molrs-capi/build.rs` — **generalize**:rpath 注入扩展这个已有的 cbindgen build script,不新建第二个 build script。
- `molrs-cxxapi/build.rs:461` 的 `cargo::rustc-link-arg-tests` 双冒号形式 + 双层错误策略 — **pattern**:四个 build script 一律用 `cargo::` 双冒号指令;**契约违规(拿不到 target-libdir、路径不存在)直接 `panic!` 并给出修复命令**,可选支持才降级为 `cargo::warning`。明确**不抄** `molrs-capi/build.rs:73-84` 那种"header 已存在就 warning 放行"的容错——那对 rpath 是错的语义。
- `molrs-ffi/src/abi.rs:155` 的 `#[cfg(feature = "ff")]` — **generalize**:测试门改为无条件。注意 `report()` 内部 `ff` 门(行 145-149)必须保留(`ForceFieldRef` 本身是 `ff`-gated),因此同时在测试模块加一行 `#[cfg(not(feature = "ff"))] compile_error!("molrs-ffi's layout snapshot is defined with `ff` on (now default); build tests with default features")`——把"`--no-default-features` 跑测试"从**误导性的 snapshot drift 红**变成一条准确的编译期消息。snapshot 字节预期**不变**:快照内类型无任何 feature-gated 字段,且 CI 今天已用 `--features ff` 生成同一份报告。
- `ci-rust.yml:57` 的 `--features ff` — **generalize**:`ff` 进 default 后该 flag 冗余,删除。
- `molrs-capi/tests/cpp/CMakeLists.txt:37,56` — **generalize**:`STATIC IMPORTED` → `SHARED IMPORTED`,`IMPORTED_LOCATION` 指向 `${MOLRS_TARGET_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}molrs_capi${CMAKE_SHARED_LIBRARY_SUFFIX}`(平台后缀不能硬写 `.dylib`,ci-capi 跑 ubuntu),加 `BUILD_RPATH ${MOLRS_TARGET_DIR}`,删 62-72 的 `-lc/-lm/...` 链接列表(那是 staticlib 才需要的),并删除死开关 `MOLRS_F64`(行 10、39-43、78-80)——molrs 已无 `f64` feature。同一 iron law 顺带清掉 `molrs-capi/build.rs:13-14` 里"With feature flags f64/i64/u64"的失效注释。
- `scripts/fetch-test-data.sh` — **pattern**:新脚本沿用其形制(`set -euo pipefail`、`PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"`、env 覆盖、refuse-and-explain 而非静默继续)。
- `molrs-python/pyproject.toml:79` `pass_env = [... "RUSTFLAGS" ...]` — **reuse**:静态覆盖通道已存在,直接引用,不改。
- `molrs-python/tests/test_ffi_abi.py` — **pattern**:`regressions/ffi-shared-dylib.py` 的握手断言沿用它的形状(token 四元组、capsule 名内嵌 ABI 线)。**边界写明**:链接层面的"同一份 dylib"断言需要两个 wheel 共存于一个 venv,pytest 单包环境表达不了,故留在 shell 脚本里。
- 放置 — **placement**:不新增 Rust 模块;`molrs-ffi` 仍是唯一 ABI 属主,dylib 只是其新增链接形态;脚本入根 `scripts/`;跨仓文件在 Files 里分列(实现时两仓分别改、分别提交)。
- 已核实非问题:`blas` 的 `cfg(not(...))` 分支保持;molrs-python 自有 `fs` 门不受影响;`architecture_gate` 的 `required-features = ["full"]` 仍跑;全仓没有"断言某 feature 关闭"的测试。
- **新符号**:本 spec 不新增任何公开 API。唯一的新代码单元是四个 build script 里各一个私有函数(`fn emit_runtime_rpath()`),无工厂函数、无 context blob、无一体化门面。**四份近似重复是刻意的**:build script 之间共享代码需要一个 build-dependency crate,而 molpack 要发布到 crates.io,那意味着再发一个新 crate(发布火车,已 Out of scope);20 行重复优于新增跨仓发布依赖。build script 不可单测,这是事实,不用假单测掩盖(见 Testing strategy)。

## Files to create or modify

### molrs 仓(`/Users/roykid/work/molcrafts/molrs`)

- `molrs-ffi/Cargo.toml` — `crate-type = ["dylib","rlib"]`;`default = ["ff"]`;molrs dep 删 `default-features = false`
- `molrs-ffi/src/abi.rs` — 删测试上的 `#[cfg(feature = "ff")]`,加 `compile_error!` 守卫
- `molrs/Cargo.toml` — `default = ["full","stream","filesystem","rayon"]`
- `molrs-capi/Cargo.toml` — 删 `default-features = false`;profile 对齐
- `molrs-cxxapi/Cargo.toml` — 删 `default-features = false`
- `molrs-python/Cargo.toml` — 新增 `rayon = ["molrs/rayon"]` 并入 `default`;第 38 行 molrs dep **不动**(Pyodide 豁免);profile 对齐
- `molrs-wasm/Cargo.toml` — 仅补注释:`default-features = false` 是**刻意的浏览器豁免**,不随并集收敛
- `Cargo.toml`(根)— `[profile.release]` 对齐
- `.cargo/config.toml` — 追加 rustflags;**`git add`(当前 untracked)**
- `molrs-capi/build.rs` — 扩展:注入 runtime rpath;删失效的 f64/i64/u64 注释
- `molrs-python/build.rs` (new) — rpath 注入
- `molrs-capi/tests/cpp/CMakeLists.txt` — SHARED IMPORTED + `BUILD_RPATH`;删链接列表与 `MOLRS_F64`
- `scripts/verify-shared-dylib.sh` (new) — 跨仓共享 dylib 校验器
- `regressions/ffi-shared-dylib.py` (new) — 回归例子(公开 API + 硬编码 golden)
- `.pre-commit-config.yaml` — 新增 pre-push local hook `verify-shared-dylib`
- `.github/workflows/ci-rust.yml` — 删 `--features ff`;加 `RUSTFLAGS` 静态 env
- `.github/workflows/ci-python.yml` — 加 `RUSTFLAGS` 静态 env
- `.github/workflows/ci-capi.yml` — 加 `RUSTFLAGS` 静态 env
- `.github/workflows/bench.yml` — 加 `RUSTFLAGS` 静态 env(性能历史可比性)
- `.github/workflows/nightly.yml` — 加 `RUSTFLAGS` 静态 env
- `.github/workflows/publish.yml` — 加 `RUSTFLAGS` 静态 env(含 maturin-action / cibuildwheel 容器透传确认)
- `docs/interop.md` — 改写 167-170 行(Path C 的"in-house consumers do not link this / static only"论断被本 spec 直接反转)
- `CLAUDE.md` — "Build cache" 节同步(共享 target + 本地动态 / CI 静态 + 已 committed 的 config)
- **不改**:`.github/workflows/ci-wasm.yml`(wasm32 被 cfg 排除,无需 env)

### molpack 仓(`/Users/roykid/work/molcrafts/molpack`)

- `Cargo.toml` — molrs dep 删 `default-features = false`;新增 path-only `[dev-dependencies] molrs-ffi`;`[profile.release]` 对齐
- `python/Cargo.toml` — molrs dep 删 `default-features = false`;新增 `[profile.release]`
- `.cargo/config.toml` — 追加 rustflags
- `build.rs` (new) — rpath 注入
- `python/build.rs` (new) — rpath 注入
- `.github/workflows/ci.yml` — 加 `RUSTFLAGS` 静态 env
- `.github/workflows/bench.yml` — 加 `RUSTFLAGS` 静态 env
- `.github/workflows/publish-crate.yml` — 加 `RUSTFLAGS` 静态 env
- `.github/workflows/publish-pypi.yml` — 加 `RUSTFLAGS` 静态 env

## Tasks

- [ ] Probe cdylib load / staticlib build / release-LTO under `-C prefer-dynamic` and record branches A1|A2, B1|B2, C1|C2 in this spec's Design
- [ ] Write failing `scripts/verify-shared-dylib.sh` (RED today: the two extensions share no `libmolrs_ffi`; refuse-and-explain on missing `MOLPACK_ROOT`, print both resolved dylib paths on failure)
- [ ] Convert `molrs-ffi` into the dylib host in `molrs-ffi/Cargo.toml` + `molrs-ffi/src/abi.rs` (`crate-type`, `default = ["ff"]`, unconditional layout test + `compile_error!` guard) and drop the now-redundant `--features ff` from `.github/workflows/ci-rust.yml`
- [ ] Widen `molcrafts-molrs` default to `full,stream,filesystem,rayon` and converge every consumer graph (`molrs-capi`, `molrs-cxxapi`, `molrs-python` rayon forward feature, `molpack/Cargo.toml` + path-only molrs-ffi dev-dep, `molpack/python/Cargo.toml`; document the molrs-wasm exemption)
- [ ] Land the static/dynamic switch: append prefer-dynamic rustflags to both `.cargo/config.toml` (and `git add` molrs's), align `[profile.release]` across the five roots, and set `RUSTFLAGS: -C prefer-dynamic=no` in every cargo-running workflow of both repos
- [ ] Emit the runtime rpath from build scripts: extend `molrs-capi/build.rs`, add `molrs-python/build.rs`, `molpack/build.rs`, `molpack/python/build.rs` (panic with the fix command on failure, per the `molrs-cxxapi/build.rs` two-tier policy)
- [ ] Convert `molrs-capi/tests/cpp/CMakeLists.txt` to a SHARED import with `BUILD_RPATH`, and delete the dead `MOLRS_F64` switch, the `-lc/-lm` interface list, and the stale f64/i64/u64 comment in `molrs-capi/build.rs`
- [ ] Wire `verify-shared-dylib` into `.pre-commit-config.yaml` as a pre-push local hook and rewrite `docs/interop.md:167-170` plus the `CLAUDE.md` "Build cache" section
- [ ] Add regression example `regressions/ffi-shared-dylib.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite in both repos and confirm `molrs-ffi/src/layout.snapshot` is byte-identical

## Testing strategy

**本 spec 不新增 Rust 单元测试,且这是刻意的**:它不新增任何函数/模块——变更全部落在 Cargo manifest、cargo 配置、build script 与 CMake 上,而"两个扩展加载同一份 dylib"是一个**链接期/进程期**属性,只有双 wheel 共存的进程外 harness 能观测。写一个 `assert!(cfg!(feature = "ff"))` 之类的"配置自证"单测属于本仓明令禁止的"结构上无法暴露缺陷的测试",不写。已有的 `layout_matches_committed_snapshot` 从 `ff`-gated 变为无条件,是本 spec 唯一的测试**门**变更。

- **Happy path(链接层)**:`bash scripts/verify-shared-dylib.sh` — 在一个临时 venv 里以 **release** profile 构建并安装 molrs 与 molpack 两个 wheel,用 `importlib.util.find_spec("molrs._lib").origin` / `find_spec("molpack.molpack").origin` 定位两个扩展,`otool -L`(macOS)/ `ldd`(Linux)取出各自的 `libmolrs_ffi` 条目并 `realpath`,断言**两者相等且非空**;失败时打印两条解析出的路径与原始 `otool` 输出。最后一步调用回归例子。
- **Happy path(运行期)**:`regressions/ffi-shared-dylib.py` — 同一解释器内 `import molrs` + `import molpack`(后者在扩展初始化时经 `interop::check_abi` 调 `molrs._ffi_abi_token()` 完成握手),然后走真正的零拷贝 capsule:`molrs.Block()` 插入 `element` / `x` / `y` / `z` 三原子 → `frame["atoms"] = block` → `molpack.Target(frame, 1)`,断言 `natoms == 3`、`elements == ["C","C","O"]`、`count == 1`,并断言 `molrs._ffi_abi_token()` 的 line 段 == `"0.14"`、capsule 名 == `"molrs.FrameRef/0.14"`(**硬编码 golden**,与 `molrs-python/tests/test_ffi_abi.py` 同形)。全程只用两个包的公开 API,零第三方运行期依赖。
- **Edge cases**:(1) `MOLPACK_ROOT` 未设且 `../molpack` 不存在 → 脚本非零退出并打印覆盖用法(refuse-and-explain,非静默 skip);(2) `MOLPACK_ROOT` 指向非 molpack 目录 → 同样非零退出;(3) 静态形态等价性:`RUSTFLAGS='-C prefer-dynamic=no'` 下重跑构建,断言两个扩展的 `otool -L` **不含** `libmolrs_ffi`;(4) wasm/Pyodide 不受影响:`cargo check --manifest-path molrs-wasm/Cargo.toml --target wasm32-unknown-unknown` 仅出既知警告,`wasm-pack build --release --target bundler` 绿。
- **回归基线(全部必须绿,0 failed)**:`cargo test -p molcrafts-molrs --lib --features full,filesystem` ≥ **1566** passed(并集后新增 `molrs/src/stream/**` 的 ~20 条;差额必须能逐条归因到 stream/rayon,不得是"多出来一些");`cargo test --doc -p molcrafts-molrs --features full,filesystem` ≥ **66**;`cargo test --manifest-path molrs-ffi/Cargo.toml` == **18**(不再传 `--features ff`);molrs-python pytest **572**;molpack rust **230** + python **132**;`molrs-ffi/src/layout.snapshot` `git diff --exit-code` 为空。
- **发布前置动作(不属本 spec 的验收,但必须执行一次并记录)**:对 `publish.yml` 跑一次 `workflow_dispatch`,确认 manylinux 容器 / cibuildwheel 内确实读到 `RUSTFLAGS='-C prefer-dynamic=no'`(产物 `readelf -d` 无 `libmolrs_ffi`)。若透传失败,必须在打 tag 前改为 action 级 env 传入——否则会发出引用本机 dylib 的坏 wheel。
- **单测门**:`$META.build.test_single`,例如 `cargo test --manifest-path molrs-ffi/Cargo.toml layout_matches_committed_snapshot`。

## Out of scope

- **CI 的 python-dynamic job**(在 CI 里另跑一条动态形态的构建/测试)——本 spec 只保证 CI 静态等价现状。
- **wheel 内分发共享 `.so`**($ORIGIN / `@loader_path` rpath、auditwheel/delocate 的 dylib 打包策略)——发布形态保持自包含静态。
- **Atomiverse / molrs-cxxapi 翻转为 cdylib**——cxxapi 的 staticlib 与 wasm 并列作为文档化例外。
- **发布火车**:`molcrafts-molrs-ffi` 上 crates.io、molrs→molpack 的发版顺序、molpack CI 的 `MOLRS_GIT_REF` bump、以及随之把 molpack 根的 molrs-ffi 从 dev-dep 提升为普通依赖(让 CLI 二进制入圈)。
- **动态形态的性能测量**(跨 crate 内联损失的量化)——bench 一律留在静态形态,量化另开 `/mol:perf`。
- **`blas` / `slow-tests` 进并集**——前者需系统 BLAS(见 `.claude/notes/notes.md` 2026-05-28),后者是慢测开关,均刻意排除。
- 已考虑并否决的替代方案:(a) 让 `molcrafts-molrs` 自身也产出 dylib —— 无必要,它的代码已随 molrs-ffi 进入同一镜像,且会多出一份需要版本管理的 ABI 面;(b) 抽一个 build-script 辅助 crate 消除四份 rpath 重复 —— 需新增跨仓发布依赖,代价大于 20 行重复。
