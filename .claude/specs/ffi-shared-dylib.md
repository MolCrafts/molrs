---
title: molrs-ffi 升级为共享动态库宿主(单一并集 feature 收敛)
status: code-complete
created: 2026-08-22
revised: 2026-08-23
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

8. **cargo 的单元指纹是递归的,而 dylib 产物无 hash**。`-C metadata` 由 cargo 对**每个 unit** 计算,输入递归混入其**依赖单元**的指纹(feature 集、依赖图形状、profile、workspace flavor),而非只看该 crate 自身的 feature。同时共享 target 下 dylib 的落盘名是 `deps/libmolrs_ffi.dylib`——**无 hash**。两者相乘的后果是:任何一条分叉的构建图都会把自己的 molrs 单元写进**同一个文件名**,覆写并毒化先前的构建。实测报错文本为 `error: multiple different versions of crate molrs`(cxxapi→capi 10 errors;molpack/python→cxxapi 48 errors)。因此"并集 feature + profile 对齐"是**必要而不充分**条件——见 Design 的"单元同一性"节。

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

### probe 结论(2026-08-23,沙盒复刻:pyo3 0.29 双 cdylib 扩展 + staticlib + 共享 target;命令见下)

- **(a) = A1**。共享 target 下 dylib 的 install_name 是**无 hash 的绝对路径**(`<shared-target>/release/deps/libhost.dylib`),两个扩展引用同一路径;外部启动的 python(无 cargo 环境)`import ext, ext2` 均成功且第二个扩展构建 0.10s(完全复用)。build.rs 只需注入 **libstd 的 rpath**(`rustc --print target-libdir`);dylib 本身无需 rpath。命令:`RUSTFLAGS="-C prefer-dynamic -C link-args=-Wl,-rpath,$(rustc --print target-libdir)" cargo build --release` + 外部 `python -c "import ext, ext2"`。注:裸 cargo 下 pyo3 extension-module 还需 `-Wl,-undefined,dynamic_lookup`,真实链路由 maturin 注入,非本 spec 关注点。**推论**:install_name 无 hash 意味着 feature/profile 失配的第二次构建会**覆写同一文件**——并集收敛与 profile 对齐从"优化"升级为"正确性前提"(与 Design 的 profile 对齐节一致)。
- **(b) = B1**。staticlib 在 config 默认 `prefer-dynamic` 下构建通过(rustc 对自包含输出自动取 rlib 路径)。cxxapi/capi 的 `.a` 无需任何豁免前缀。
- **(c) = C2**。`lto = "thin"` + 图中含 Rust dylib 的 cdylib 输出被 rustc 拒绝:`only 'staticlib', 'bin', and 'cdylib' outputs are supported with LTO`。**回退落法**:五个 root 的 `[profile.release]` 删除 `lto = "thin"`(注释写明本错误);CI/publish 的静态路径以 env **`CARGO_PROFILE_RELEASE_LTO: thin`** 恢复(与 `RUSTFLAGS: -C prefer-dynamic=no` 并列)——已实测:该组合产出完全自包含的静态产物(otool 无 libhost/libstd 引用)。发布产物的 LTO 不变;仅本地动态形态无 LTO(dev 模式,可接受)。ac-011/ac-012 的判定条件相应含 `CARGO_PROFILE_RELEASE_LTO`。

### profile 必须逐字对齐(否则共享 dylib 会互相顶掉)

~~五个~~ **七个** workspace root(molrs 根、`molrs-ffi`、`molrs-python`、`molrs-capi`、`molrs-cxxapi`(轴 4 后独立)、molpack 根、`molpack/python`)各自声明 `[profile.release]`;profile 参与指纹,不一致会让 `target/release/deps/libmolrs_ffi-<hash>.dylib` 出现两份,而被 uplift 到 `target/release/libmolrs_ffi.dylib` 的那份取决于谁最后构建——`@rpath` 在运行时解析的正是这个无 hash 名字,于是"同一份 dylib"的不变量会随机破裂。因此把 `lto` / `codegen-units` / `opt-level` / `strip` 四项在七处对齐(以 `molrs-python` 现有值为准:`thin` / `1` / `3` / `symbols`,C2 分支下去掉 `lto`),`molpack/python/Cargo.toml` 需**新增** `[profile.release]`。

**原"五个"是错的,实测代价见收敛循环的 release 侧一节**:`molrs-ffi`(dylib 宿主本身)与 `molrs-cxxapi` 都是独立 workspace root 却没有 profile 块,裸 release 构建于是退回 cargo 默认(`codegen-units = 16`、不 strip),写出的 `libmolrs_ffi.dylib` 与 wheel 的完全不同。

### 单元同一性(unit identity)——第二条硬前置,原 Design 的假设已被推翻

原 Design 假设 **feature 并集 + profile 逐字对齐**足以让六个 workspace root 产出同一份 `libmolrs_ffi.dylib`。**该假设为假**,已由受控实验推翻(见 Domain basis 8)。干净树普查的结果是:**6 个 root → 6 个互不相同的 molrs 单元**。四条分叉轴被逐一隔离,每条配一个修法:

**轴 1 — lockfile 版本漂移(主因,54/54 个公共 crate 全部分叉)。** 六个 workspace root 各自持有一份**未跟踪**的 `Cargo.lock`,分别在不同时刻解析(实测:ffi root 的 `log 0.4.34` vs python root 的 `log 0.4.33`)。删掉六份 lock 后一次性重建,ffi 与 python 立刻收敛到**同一个** molrs 单元。
**修法 = 同步,而非跟踪**:新增 `scripts/sync-dylib-locks.sh`,删除并在**同一个 registry 时刻**一次性重建七份 lock(molrs 根、`molrs-ffi`、`molrs-python`、`molrs-capi`、`molrs-cxxapi`(轴 4 后独立)、molpack 根、`molpack/python`;molpack 侧沿用 `verify-shared-dylib.sh` 的 `MOLPACK_ROOT` 解析)。**lock 继续不跟踪**——这个不变量是**本机局部**的(动态圈只存在于本地;CI 是静态、逐仓的),把 lock 提交进仓会用一个发布层面的谎言去修一个本地问题。`verify-shared-dylib.sh` 在身份判定失败时**按名字**指向该脚本。`molrs-wasm` 的 lock **刻意不进 sweep**:它是既有的浏览器豁免,其构建目标是 wasm32(cfg 豁免动态化),把它拉进同步只会引入一个按设计就应当分叉的单元。

**轴 2 — `default` 这个字面 feature 标签。** 未写 `default-features = false` 的消费者解析出的 molrs feature 集**带 `default` 这个标签本身**;而 `molrs-ffi` / `molrs-python`(为 wasm/Pyodide 保留 `default-features = false`)解析出的集合不带——集合不同 → 单元不同。
**修法 = 标签收敛**:`molrs-ffi` 已作为探针落地 `molrs-default = ["molrs/default"]` 并列入 `default`;实测其后 molrs 自身的 `--cfg feature` 列表在 ffi 与 capi 两条图上**逐字相同**。`molrs-python` 补同一形状:其 molrs 依赖保留 `default-features = false, features = ["full","stream"]`(Pyodide 豁免不变),另加 default-on 的 `molrs-default = ["molrs/default"]` 前向——Pyodide 的 `maturin --no-default-features` 会连同 `fs`/`rayon` 一起把它丢掉。

**轴 3 — 兄弟依赖诱发的第三方 feature 加宽。** feature 统一是**逐 invocation-graph** 的,所以即使 molrs 的 feature 完全一致,单元仍可能不同:`molrs-capi` 直挂的 `serde_json = "1"` 会加宽 serde / serde_json / syn / once_cell / zarrs-cluster / tungstenite / rand-cluster / zerocopy / memchr / half / getrandom / ppv_lite86 / derive_more / auto_impl(实测 28 个 crate);molpack 侧的图(numpy→ndarray、molpack 根的 rand/log)加宽的是另一批。
**修法 = 把 molrs-ffi 当作统一锚点**:`molrs-ffi` 增加一个 default-on 的 `unify` feature,挂**可选、钉版本**的第三方依赖(`unify = ["dep:serde_json", …]`),镜像任一兄弟能诱发的**最宽** feature 集(具体清单由收敛循环实测得出,起点 `serde_json`,按需扩)。理由:`molrs-ffi` 定义上在**每一条**动态图里,它的 pin 因此把所有图的第三方统一推到同一个 superset——即 cargo-hakari 的 workspace-hack 模式,但**不新增 crate**。wasm 侧靠既有的 `default-features = false` 天然排除这些 pin。

**轴 4 — workspace-member 风味(flavor)。** `molrs-cxxapi` 是 molrs 根 workspace 的成员,构建它会铸出一个 **member 风味**的 molrs 单元(workspace 成员在 dev 下按增量编译,path 依赖不是);而它又依赖 `molrs-ffi`,于是它会用这个风味**覆写**共享 dylib。
**修法 = 把 molrs-cxxapi 逐出根 workspace**:根 `members = ["molrs"]`;`molrs-cxxapi` 自带 `[workspace]` 与逐字对齐的 `[profile.release]`(与"binder crate 一律独立 workspace"的既有教义一致)。此后根 workspace 的任何命令图里**不再含 molrs-ffi**,也就永远不会写那个 dylib。连带:CI 与 pre-commit 里的 cxxapi clippy 行必须从 `-p molcrafts-molrs-cxxapi` 改为 `--manifest-path molrs-cxxapi/Cargo.toml`——三处同一拼写:`.pre-commit-config.yaml`、`.github/workflows/ci-rust.yml`、`CLAUDE.md` 的 `build.check`。

#### 收敛循环(机械步骤,不是判断题)

四条轴落地后按此迭代,直到普查为 1:

1. 清掉共享 `target/debug`;
2. 依次构建五个原生 root:`molrs-ffi`、`molrs-python`、`molrs-capi`、`molrs-cxxapi`(独立 workspace)、`molpack/python`;
3. 普查 `ls <shared-target>/debug/deps/libmolrs-*.rlib | wc -l`;
4. 若 > 1:对分叉的 crate **word-diff 其 rustc 调用**(诊断阶段用的就是这个方法),把缺的 feature/依赖 pin 补进 `molrs-ffi` 的 `unify`,回到 1。

普查 == 1 之后再断言**真正的不变量**:release 构建 molrs wheel → `sha256` 该 dylib → 构建 molpack wheel → `sha256` **不变**。(molpack **根**的 test 图不在这五条里,它由 `abi_line` 测试单独覆盖;其 lock 已在 sweep 内。)

#### unify pin 清单(收敛循环实测,2026-08-23)

四条轴落地后普查 = **3**(`molrs-ffi` / `molrs-python` / `molpack/python` 已合流为一个单元;`molrs-capi` 一个;`molrs-cxxapi` 一个),再补下面四条 pin 后普查 = **1**。每条都标注**是哪个 root 诱发的**:

| pin(`molrs-ffi/Cargo.toml`) | 空间 | 诱发者 | 不 pin 时的连锁 |
|---|---|---|---|
| `serde_json = { version = "1", optional = true, default-features = true }` | normal | `molrs-capi` 自带的 `serde_json = "1"` | serde / serde_core / serde_json 的 `default` 标签与 indexmap 闭包 |
| `foldhash = { version = "0.2", optional = true, features = ["std"] }` | normal | `molrs-cxxapi` 的 `cxx`(要 `foldhash/std`) | foldhash → hashbrown → lru → zarrs → molrs |
| `syn = { version = "2", optional = true, features = ["full","extra-traits","visit","visit-mut","fold"] }` | **build** | `molrs-capi` 的 `cbindgen`(要 syn 2 的 `fold`) | syn2 → auto_impl / derive_more_impl / zerocopy_derive → derive_more + zerocopy → ppv-lite86 → rand → tungstenite,以及 zarrs 全子树 |
| `proc-macro2 = { version = "1", optional = true, features = ["span-locations"] }` | **build** | `molrs-cxxapi` 的 `cxx-build` | proc-macro2 → quote/syn → **所有**过程宏(serde_derive / thiserror_impl / futures_macro / tokio_macros)→ serde / tokio / molrs 闭包大半 |

两条必须落在 `[build-dependencies]`:cargo v2 resolver 给 host 代码(build script + 过程宏)单独一个 feature 空间,而加宽正是发生在那里(cbindgen、cxx-build 都是 build script)。写成普通依赖会加宽**另一个**空间,多铸一个单元而不是合并一个。`molrs-ffi` **不需要** build.rs——cargo 无论是否有 build script 都会解析 build-dependencies,实测 pin 后 `cargo tree -i syn@2` 在 ffi 图上即出现 `fold`。

**不需要的 pin(实测,不要凭猜补上)**:`ndarray` / `half`(molpack/python 的 numpy 0.29 并未加宽它们)、`rand`(molpack 根不在这五条图里)。轴 1 的 lock 同步 + 轴 2 的标签收敛之后,`molpack/python` 无需任何 pin 就与 `molrs-ffi` 同单元。

#### release 侧实测:第五个指纹输入是 rustflags,且 dylib 宿主自己也有"风味"

普查 == 1 后按不变量做 release 断言,实测出三件必须写进 spec 的事:

1. **不变量成立**:molrs wheel 构建后 `sha256(target/release/deps/libmolrs_ffi.dylib)` = `9ba2c3da…`,再构建 molpack wheel 后**仍是** `9ba2c3da…`(两次独立复现)。
2. **裸 `cargo build --release` 与 maturin 是两套 rustflags 制度,不可混用**。maturin 注入 `CARGO_ENCODED_RUSTFLAGS="-C prefer-dynamic -C link-arg=-undefined -C link-arg=dynamic_lookup"`(它把仓内 config 的 `prefer-dynamic` 合并了进去)。rustflags 是指纹输入,所以两制度必然是两个单元,而它们抢的是同一个**无 hash** 的 `deps/libmolrs_ffi.dylib`:后写者赢,先写者的消费者**硬失败** `error[E0463]: can't find crate for molrs_ffi`(不是静默降级——这点反而是好消息),且 **cargo 不会自愈**(它认为自己那个单元是 fresh 的),恢复手段是 `touch molrs-ffi/src/lib.rs` 后用需要的制度重建。**推论**:release 侧的 sha 断言只能夹在**两次 maturin 构建之间**(`verify-shared-dylib.sh` 正是这么做的),中间不得插入裸 `cargo build --release`;收敛循环的普查用 debug + 单一制度,也正确。
3. **轴 4 的"风味"对 dylib 宿主自身同样成立**:即使 rustflags 与 profile 完全对齐,把 `molrs-ffi` 当**工作区根**构建出的 dylib(`503d0508…`)与把它当**路径依赖**构建出的(`9ba2c3da…`)字节不同——`-C metadata` 相同(消费者能接受任一份,不报 E0463),但内容不同,于是 sha 断言会红。要么只用 maturin 侧的两次构建做断言,要么在断言前不碰裸构建。
4. **`molrs-ffi` 原本没有 `[profile.release]`**(本 spec 的 profile 对齐只点了五个 root,漏掉了 dylib 宿主自己),裸 release 构建因此退回 cargo 默认 profile(`codegen-units = 16`、不 strip),写出的 dylib 与 wheel 的完全不同。已补齐——**七个 root 都要有**这四个键(molrs 根、molrs-ffi、molrs-python、molrs-capi、molrs-cxxapi、molpack 根、molpack/python;molrs-wasm 因 wasm32 豁免不计)。

**比 `-v` 日志更便宜的诊断法**(等价于 word-diff rustc 调用,但不用重新编译):对每个 root 跑
`cargo build --manifest-path <root>/Cargo.toml --message-format json`,收集 `compiler-artifact` 的产物名并按 `<crate>-<16位hash>` 解析,得到"该 root 用了哪些单元"的表;两表求差即得**分叉的 crate 清单**(一次实测:capi 差 28 个 crate、cxxapi 差 49 个)。再用 `target/debug/.fingerprint/<crate>-<hash>/lib-*.json` 的 `features` 字段读出两个单元的 feature 差,和 `cargo tree -i <crate> --format "{p}|{f}"` 找出诱发者。

#### 门的加固:原第 8 步的路径比较是空转

`verify-shared-dylib.sh` 第 8 步"两个扩展的 `libmolrs_ffi` 路径相等"作为**同一性证明是空的**——install_name 无 hash,两者**永远**指向同一个绝对路径,即使文件已被第二次构建覆写。真正有牙齿的是第 9 步(同解释器 import + capsule 往返)。因此:在两次 wheel 构建**之间**插入 sha256 稳定性断言(覆写即红),路径比较降级为一行廉价 sanity print 保留。这是本仓 `feedback_tests_that_cannot_fail` 的同一条禁令——"结构上无法暴露缺陷的断言"必须被替换,而不是被信任。

#### molpack 根 dev-dep 的"去惰性化"

实测:molpack 根那条 path-only 的 `[dev-dependencies] molrs-ffi` 是**惰性的**——`molpack/{src,tests,benches,examples}` 里没有任何源码引用 `molrs_ffi`,rustc 因此从不加载这个 `--extern`,test/bench 二进制里**没有** `libmolrs_ffi` 条目。原 spec"`cargo test`/`cargo bench`/`--examples` 全部入圈"的论断**为假**。
**修法**:新增 `molpack/tests/abi_line.rs`——一条真实的单断言测试:`assert_eq!(molrs_ffi::abi::abi_line(), molrs::VERSION.rsplit_once('.').map(|(mm, _)| mm).unwrap())`。它一石二鸟:把 `molrs_ffi` 变成被真实引用的 extern(molpack 的测试二进制**真的**入圈),同时把"两侧 ABI 线配对"从口头约定变成可执行断言。这不是配置自证——它比较的是两个**独立编译单元**各自烘进去的版本串。

#### Reuse decision(supersede 增补)

- `scripts/fetch-test-data.sh` / `verify-shared-dylib.sh` — **pattern**:`sync-dylib-locks.sh` 沿用其形制。
- `scripts/verify-shared-dylib.sh` — **generalize**:sha256 断言加进这个已有的门,不新建第二个校验脚本。
- `molrs_ffi::abi::abi_line` 与 `molrs::VERSION` — **reuse**:`abi_line.rs` 直接调用,不重算、不硬编码 `"0.14"`。
- cargo-hakari / 新建 workspace-hack crate — **new(否决)**:`molrs-ffi` 已是"每条图都在"的 crate,pin 挂它身上等价 hack crate 且不新增跨仓发布依赖。

#### 被本节推翻或作废的旧论断

- "对齐 profile 即可保证唯一 dylib"——不成立,profile 只是第五个指纹输入。
- "molpack 的 test/bench/examples 全部入圈"——为假,见去惰性化。
- "`.cargo/config.toml` 当前 UNTRACKED"——已作废(def162c 已跟踪),不再需要 git add 或措辞修正。
- `molrs-cxxapi` 的成员身份从"编译成本项"升级为**正确性缺陷**(轴 4)。
- 保留不变:molpack CLI 二进制不入圈(molcrafts-molrs-ffi 未上 crates.io),Out of scope 的发布火车条目原样有效。


### molpack 根依赖:一个 librarian 未覆盖的硬约束(iron law 上报)

原始需求是"molpack 根加 molrs-ffi 依赖,使 CLI/测试入圈"。核查发现:**`molcrafts-molrs-ffi` 从未发布到 crates.io**(`.github/workflows/publish.yml` 只发 `molcrafts-molrs`;`.claude/notes/release.md` 的发布表同样只有一行 crates.io)。而 molpack 根是本链条里**唯一**会 `cargo publish` 的 crate(`publish-crate.yml`)。若加成普通依赖,下一次 molpack 打 tag 会直接 publish 失败——一个只在发布日才炸的静默雷。

决策:molpack 根以 **path-only `[dev-dependencies]`**(带 `path`、**不带** `version`)引入 molrs-ffi。cargo 在打包时会剥离 path-only dev-dependency,molpack 的可发布性零风险;`cargo test` / `cargo bench` / `--examples` 全部入圈(molpack pre-push 的 `cargo test --features io --lib --tests --examples` 与 `cargo build --benches` 即刻覆盖)。**代价必须写明**:`cargo build --features cli` 产出的 `molpack` CLI 二进制**不入圈**,仍静态嵌 molrs,直到 `molcrafts-molrs-ffi` 进入 crates.io 发布火车(已在 Out of scope,届时一行改成普通依赖即可)。

### 影响面(逐条,不许沉默)

- **`stream` 首次进入默认门**:今天 `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem` 与 `cargo test --lib --features full,filesystem` 都**没有**覆盖 `molrs/src/stream/**`(`stream` 不在 `full` 里)。并集后它进入 clippy(`-D warnings`)与单测门,实测该目录有 **20 个 `#[test]`/`#[tokio::test]`**。首跑若冒出 clippy 警告,按 iron law **就地修**,不得放宽门。
- **原生 wheel 转并行**:见 Domain basis 6。归约次序改变意味着浮点末位可能变化,`molrs-python` 572 条 pytest 与 molpack 132 条须逐条绿;任何"因并行而失败"的断言是真缺陷,不得放宽容差(先查该断言是否本就依赖顺序)。
- **编译面变宽**:所有原生消费者现在都编译 `stream`(tokio / tungstenite)与 `filesystem`(zarrs)。`molrs-cxxapi` 的 clippy(轴 4 之后是 `cargo clippy --manifest-path molrs-cxxapi/Cargo.toml --all-targets`,不再是 `-p`)**编译**时间随之上升(它只 lint cxxapi 自身代码,不 lint 依赖,故新增 lint 面主要来自 molrs 自身的 `stream`)。这是编译成本,不是 break。
- **`-p molcrafts-molrs-cxxapi` 在根 workspace 里从此不可用**(轴 4 的连带)。仓内仍有两处历史文档写着这条命令:`.claude/specs/release-0-12-05-cxxapi-panic-free.md:49` 与 `regressions/release-0-12-05-cxxapi-panic-free.md:12`(均为已关闭 spec 的记录,其"验证命令"现会直接失败)。本 spec 不改 `regressions/`,已上报给调用方走文档修正。
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
- `.cargo/config.toml` — 追加 rustflags(已落地;def162c 已跟踪,原 git-add 条目作废)
- `scripts/sync-dylib-locks.sh` (new) — 七份 Cargo.lock 的一次性同步重建器(轴 1)
- `Cargo.toml`(根)— `members = ["molrs"]`(cxxapi 逐出,轴 4)
- `molrs-cxxapi/Cargo.toml` — 新增 `[workspace]` + 逐字对齐 `[profile.release]`
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
- `CLAUDE.md` — "Build cache" 节补本地动态 / CI 静态开关一行;`build.check` 的 cxxapi clippy 改 `--manifest-path`
- **不改**:`.github/workflows/ci-wasm.yml`(wasm32 被 cfg 排除,无需 env)

### molpack 仓(`/Users/roykid/work/molcrafts/molpack`)

- `tests/abi_line.rs` (new) — 单断言 ABI 线配对测试,使 path-only dev-dep 真正被链接
- `Cargo.toml` — 更正 MEASURED LIMIT 注释(test 二进制经 abi_line 已入圈,仅 cli 二进制未入)

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

- [x] Probe cdylib load / staticlib build / release-LTO under `-C prefer-dynamic` and record branches A1|A2, B1|B2, C1|C2 in this spec's Design
- [x] Write failing `scripts/verify-shared-dylib.sh` (RED today: the two extensions share no `libmolrs_ffi`; refuse-and-explain on missing `MOLPACK_ROOT`, print both resolved dylib paths on failure)
- [x] Convert `molrs-ffi` into the dylib host in `molrs-ffi/Cargo.toml` + `molrs-ffi/src/abi.rs` (`crate-type`, `default = ["ff"]`, unconditional layout test + `compile_error!` guard) and drop the now-redundant `--features ff` from `.github/workflows/ci-rust.yml`
- [x] Widen `molcrafts-molrs` default to `full,stream,filesystem,rayon` and converge every consumer graph (`molrs-capi`, `molrs-cxxapi`, `molrs-python` rayon forward feature, `molpack/Cargo.toml` + path-only molrs-ffi dev-dep, `molpack/python/Cargo.toml`; document the molrs-wasm exemption)
- [x] Land the static/dynamic switch: append prefer-dynamic rustflags to both `.cargo/config.toml`, align `[profile.release]` across the five roots, and set `RUSTFLAGS: -C prefer-dynamic=no` in every cargo-running workflow of both repos
- [x] Emit the runtime rpath from build scripts: extend `molrs-capi/build.rs`, add `molrs-python/build.rs`, `molpack/build.rs`, `molpack/python/build.rs` (panic with the fix command on failure, per the `molrs-cxxapi/build.rs` two-tier policy)
- [x] Write `scripts/sync-dylib-locks.sh` deleting and regenerating the untracked Cargo.locks of every native root in one registry sweep (molrs root, molrs-ffi, molrs-python, molrs-capi, molrs-cxxapi, molpack root, molpack/python; molpack resolved via `MOLPACK_ROOT` as in `scripts/verify-shared-dylib.sh`; molrs-wasm deliberately excluded with a comment)
- [x] Strengthen `scripts/verify-shared-dylib.sh`: sha256 the release `libmolrs_ffi` before and after the molpack wheel build and fail on any change, demote the step-8 path compare to a sanity print, and name `scripts/sync-dylib-locks.sh` in every identity-failure message
- [x] Converge the feature labels: add the default-on `molrs-default = ["molrs/default"]` forward to `molrs-python/Cargo.toml` (matching the probe already applied to `molrs-ffi/Cargo.toml`) and add the default-on `unify` anchor feature with optional pinned third-party deps (start at `serde_json`) to `molrs-ffi/Cargo.toml`
- [x] Evict `molrs-cxxapi` from the root workspace: root `members = ["molrs"]`, own `[workspace]` + aligned `[profile.release]` in `molrs-cxxapi/Cargo.toml`, and retarget its clippy line to `--manifest-path molrs-cxxapi/Cargo.toml` in `.pre-commit-config.yaml`, `.github/workflows/ci-rust.yml` and `CLAUDE.md`'s `build.check`
- [x] Run the unit-identity convergence loop until the `libmolrs-*.rlib` census in the shared target is exactly 1 (clean `target/debug`, build the five native roots sequentially, word-diff diverged rustc invocations, extend `molrs-ffi`'s `unify` pins, repeat) and record the final pin list in the Design
- [x] Write `molpack/tests/abi_line.rs` asserting `molrs_ffi::abi::abi_line()` equals the major.minor of `molrs::VERSION`, and correct the stale "MEASURED LIMIT" comment in `molpack/Cargo.toml`
- [x] Convert `molrs-capi/tests/cpp/CMakeLists.txt` to a SHARED import with `BUILD_RPATH`, and delete the dead `MOLRS_F64` switch, the `-lc/-lm` interface list, and the stale f64/i64/u64 comment in `molrs-capi/build.rs`
- [x] Wire `verify-shared-dylib` into `.pre-commit-config.yaml` as a pre-push local hook, rewrite `docs/interop.md:167-170`, and add the local-dynamic / CI-static switch line to `CLAUDE.md`'s "Build cache" section
- [x] Add regression example `regressions/ffi-shared-dylib.py` (public API only; hard-coded goldens, no third-party runtime)
- [x] Run full check + test suite in both repos and confirm `molrs-ffi/src/layout.snapshot` is byte-identical
- [x] Hygiene: simplify pass ran clean (no debug residue/TODO in new files; shellcheck clean; four-fold build.rs duplication is the spec's deliberate Reuse decision; docs Mode A skipped — no new public symbols in diff)

## Testing strategy

**本 spec 不新增 Rust 单元测试,且这是刻意的**:它不新增任何函数/模块——变更全部落在 Cargo manifest、cargo 配置、build script 与 CMake 上,而"两个扩展加载同一份 dylib"是一个**链接期/进程期**属性,只有双 wheel 共存的进程外 harness 能观测。写一个 `assert!(cfg!(feature = "ff"))` 之类的"配置自证"单测属于本仓明令禁止的"结构上无法暴露缺陷的测试",不写。已有的 `layout_matches_committed_snapshot` 从 `ff`-gated 变为无条件,是本 spec 唯一的测试**门**变更。

- **Happy path(链接层)**:`bash scripts/verify-shared-dylib.sh` — 在一个临时 venv 里以 **release** profile 构建并安装 molrs 与 molpack 两个 wheel,用 `importlib.util.find_spec("molrs._lib").origin` / `find_spec("molpack.molpack").origin` 定位两个扩展,`otool -L`(macOS)/ `ldd`(Linux)取出各自的 `libmolrs_ffi` 条目并 `realpath`,断言**两者相等且非空**;失败时打印两条解析出的路径与原始 `otool` 输出。最后一步调用回归例子。
- **Happy path(运行期)**:`regressions/ffi-shared-dylib.py` — 同一解释器内 `import molrs` + `import molpack`(后者在扩展初始化时经 `interop::check_abi` 调 `molrs._ffi_abi_token()` 完成握手),然后走真正的零拷贝 capsule:`molrs.Block()` 插入 `element` / `x` / `y` / `z` 三原子 → `frame["atoms"] = block` → `molpack.Target(frame, 1)`,断言 `natoms == 3`、`elements == ["C","C","O"]`、`count == 1`,并断言 `molrs._ffi_abi_token()` 的 line 段 == `"0.14"`、capsule 名 == `"molrs.FrameRef/0.14"`(**硬编码 golden**,与 `molrs-python/tests/test_ffi_abi.py` 同形)。全程只用两个包的公开 API,零第三方运行期依赖。
- **Edge cases**:(1) `MOLPACK_ROOT` 未设且 `../molpack` 不存在 → 脚本非零退出并打印覆盖用法(refuse-and-explain,非静默 skip);(2) `MOLPACK_ROOT` 指向非 molpack 目录 → 同样非零退出;(3) 静态形态等价性:`RUSTFLAGS='-C prefer-dynamic=no'` 下重跑构建,断言两个扩展的 `otool -L` **不含** `libmolrs_ffi`;(4) wasm/Pyodide 不受影响:`cargo check --manifest-path molrs-wasm/Cargo.toml --target wasm32-unknown-unknown` 仅出既知警告,`wasm-pack build --release --target bundler` 绿。
- **单元同一性(链接前)**:`bash scripts/sync-dylib-locks.sh && rm -rf <shared-target>/debug` 后依次构建五个原生 root,`ls <shared-target>/debug/deps/libmolrs-*.rlib | wc -l` == **1**。> 1 即红,红的诊断动作是 word-diff 分叉 crate 的 rustc 调用(见 Design 收敛循环),不是放宽普查。
- **单元同一性(链接后)**:`scripts/verify-shared-dylib.sh` 内,molrs wheel 构建后与 molpack wheel 构建后的 release `libmolrs_ffi` sha256 **逐字节相同**。这条取代原第 8 步路径比较作为同一性证据——后者因 install_name 无 hash 而恒真。
- **dev-dep 入圈**:molpack `cargo test --features io --lib --tests` 后,`otool -L`/`ldd` 编译出的 `abi_line-*` 测试二进制含 `libmolrs_ffi` 条目;该测试本身断言两侧 ABI 线相等(运行期取值,不硬编码)。
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
