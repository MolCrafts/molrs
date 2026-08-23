---
title: 静态链接成为零参数默认,动态形态降为命令行 --config opt-in
slug: link-mode-static-default-01-invert
status: code-complete
created: 2026-08-23
last-updated: 2026-08-23
grilled: true
---

# 静态链接成为零参数默认,动态形态降为命令行 --config opt-in

## Summary

把**静态**链接变成零参数默认:两个仓库的 `.cargo/config.toml` 不再注入任何
`rustflags`,`lto = "thin"` 回到七个原生 root 的 `[profile.release]`,十个
workflow 的 `RUSTFLAGS` / `CARGO_PROFILE_RELEASE_LTO` env 块全部删除。此后
本地、CI、发布三条路径产出**逐字相同的构建参数**(没有参数),动态形态
(共享 `libmolrs_ffi.dylib`)退化为纯命令行的 opt-in,不以任何形式提交为开关
——实测下它按传递方式分两半:rustflags 走 `--config`(裸 cargo)或内联
`RUSTFLAGS`(maturin),profile 键始终走 `--config`。跨仓库的
`scripts/verify-shared-dylib.sh` 两者自带,因此动态形态在 pre-push 上依旧被
证明活着,不会烂掉。同时把兼容契约写死成
两行:静态 = patch 兼容;动态 = 任何改动(含 patch)都要求全部下游重编重发。

## Design

### 反转的技术理由(实测事实,不再重新推导)

1. **动态不是 patch 兼容的。** cargo 1.96,空 crate,源码不变,版本
   `0.1.0` → `0.1.1`,导出符号哈希从 `…17h7f515c415611beecE` 变为
   `…17h7f9d024041394482E`。因此"动态形态可以按 patch 线互通"是假命题。
2. **零 rustflags 才是真正的零参数静态。** 配置里没有任何 rustflags 时,
   `cargo build -v` 中 `prefer-dynamic` 出现 **0** 次。
3. **`--config` 的 rustflags 与配置文件数组是拼接,不是替换。** 保留
   "动态默认 + flag 覆盖回静态"会得到
   `["-C","prefer-dynamic","-C","prefer-dynamic=no"]`,只因 rustc 取最后一个
   `-C` 才碰巧正确 —— 依赖隐式行为。这是默认必须反转的技术原因,不是审美偏好。
4. **`--config <KEY=VALUE|PATH>` 是稳定的官方 cargo flag**(`cargo build --help`),
   maturin 1.13.3 同样暴露 `--config <KEY=VALUE>`,且**那是它唯一的相关入口**
   (`maturin build --help`:无 rustflags / extra-args 类 flag)。opt-in 不需要
   自造环境变量、不需要 cargo alias、不需要包装脚本、不需要提交式开关文件。
   **但 rustflags 走不通 maturin —— 实现期实测证伪(见下)。**

5. **运行期握手救不了错配的动态对。** dyld 在加载期就失败,自检代码根本没有
   机会运行。将来若真的发布动态形态,唯一正确的拒绝点是 pip 安装期的
   build-id local version(`0.14.1+abi.<hash>`,`==` 钉住)。本 spec **不实现**,
   但必须在 `docs/interop.md` 里记为显式未接线 TODO(不留隐性债)。

### 动态 opt-in 的确切形状 —— 两样东西,两条传递路(起草假设已被实测推翻)

起草时写的是"两个 `--config` 一起交给 cargo 或 maturin",**实测为假**:
门首跑 exit 1,两个扩展都是静态;`maturin build --release -v` 里
`molrs_python` 的 rustc 调用**没有** `-C prefer-dynamic`,而 rustc 同时被喂了
`libmolrs_ffi.dylib` 与 `.rlib` 两个候选 —— 没有该 flag 就取 rlib。机制:
**maturin 无条件设置 `CARGO_ENCODED_RUSTFLAGS`,env 层 rustflags 整体替换
config 层 rustflags**,包括 `--config` 供给的那份。翻转前能工作,只因 flag 当时
写在 `.cargo/config.toml` 里 —— 那个文件 maturin 会**读**并把自己的 link-arg
追加进去;`--config` 命令行覆盖进不了那套计算。反向探针决定性:
`RUSTFLAGS='-C prefer-dynamic' maturin build … -v` 下该调用**带上了**它。
`maturin build --help` 也确认它没有任何 rustflags / extra-args 入口,没有第三条路。

因此 opt-in 按传递方式拆成两半 —— 这是实测约束,不是设计偏好:

| 要送达 rustc 的东西 | 种类 | 裸 cargo | maturin |
|---|---|---|---|
| `-C prefer-dynamic` | rustflags | `--config "target.'cfg(not(target_arch = \"wasm32\"))'.rustflags=['-C','prefer-dynamic']"` | **只能内联 `RUSTFLAGS=`** |
| `lto = false` | profile 键 | `--config 'profile.release.lto=false'` | 同一个 `--config`,照常有效 |

两者缺一不可:后者是被前者逼出来的 —— 图里出现 Rust dylib 后 rustc 拒绝 LTO
(`only 'staticlib', 'bin', and 'cdylib' outputs are supported with LTO`),而
`lto` 现在写在七个 manifest 里,只有 `--config` 的 profile 键能就地关掉它。

**这些形式在仓库里只允许出现两处:`scripts/verify-shared-dylib.sh`(它*就是*
那道门,以 `DYN_RUSTFLAGS` + `DYN_CONFIG` 自带)和 `docs/interop.md`(唯一的
人类抄写来源)。** CI job 不重抄 —— job 调用脚本(02 号 spec)。`RUSTFLAGS` 只
内联在那两次 maturin 调用上,不 export、不进 workflow、不回写任何
`.cargo/config.toml`。`.cargo/config.toml` 只剩 `[build] target-dir`;所有产物
仍在同一个共享 `target/`,不新增 `target-dyn`。

操作者约束的实质完整成立:**静态默认这条路零参数、零 env**,没有任何提交式
开关,也没有任何本项目自造的环境变量 —— `RUSTFLAGS` 是 cargo 的标准变量,且
只活在那一个 opt-in 门脚本内部。

### 形态与内聚(law.md 硬约束下的解读)

本 spec 不引入任何新公开符号,也没有可单元测试的进程内单元:改动面是
构建配置、CI 配置与陈述性文字。唯一新增的"单元"是回归例
`regressions/link-mode-static-default.py`(单一职责:断言默认构建产出的扩展
**没有** `libmolrs_ffi` 动态链接条目,并跑一段最小公开 API),以及门脚本内
一对 `DYN_RUSTFLAGS` / `DYN_CONFIG` 常量(第二个真实调用点出现才提取 ——
脚本里有两次 `maturin build --release`,满足"第二次使用"门槛)。不新增 façade、不新增
context blob、不新增工厂函数。

### 门脚本的连带约束(非可选清理)

删掉 `.cargo/config.toml` 的 rustflags 之后,`scripts/verify-shared-dylib.sh`
会在 **line 302**(`FAIL — no libmolrs_ffi dynamic-link entry`)直接变红,因为
两个扩展都会静态链接。所以 flag 迁进脚本与配置删除**必须同一次落地**,
否则 pre-push 立刻红。脚本里三处 remediation 文案(176-179、311-318、358-361)
指向"两个 `.cargo/config.toml` 里的 `-C prefer-dynamic` rustflags",以及 159-162
的 release-profile 理由注释,一并改写。

### 行为保持不变的部分(明确不动)

- `molrs-ffi/Cargo.toml:21` `crate-type = ["dylib","rlib"]` **不改**:静态形态
  本来就走 rlib 分支。只改 13-20 的说明文字。
- 四个消费方 `build.rs` 的 `emit_runtime_rpath()`(`molrs-python`、`molrs-capi`、
  `molpack`、`molpack/python`)**保留**:libstd 的 `@rpath` 是动态 opt-in 的必要
  条件,静态下该 load command 无人解析,是惰性的。只改注释(见下)。
  `molrs-cxxapi/build.rs` 不发 rpath(staticlib),正确,不动。
- `molrs-wasm` 没有 `[profile.release]`(只有
  `[package.metadata.wasm-pack.profile.release]:61`),因此**不在七根之内、
  依旧不 LTO**;Pyodide 的 `--no-default-features` régime 不动。
- `CLAUDE.md` frontmatter 的 `build.check` / `build.test`(第 8-9 行)与链接形态
  无关 → 不动。`ci.local`(第 14 行)继续正确,因为动态门仍从同一个 pre-push
  钩子跑。
- `.pre-commit-config.yaml:77-83` 的 `verify-shared-dylib` 钩子不动:flag 进脚本
  后它自然继续正确;再加一个 pre-push 钩子等于把一道已经要跑两次 release
  wheel 构建的门复制一遍。
- 两个 tox `pass_env` 里保留 `RUSTFLAGS`(官方通用环境变量,例如本地
  `-C target-cpu=native`),**删除** `CARGO_PROFILE_RELEASE_LTO`(现在无人设置,
  且 `lto` 写在 manifest 里 tox 剥不掉),并改写那段已经变假的注释。
  全程不新增任何自造环境变量。

### 库存修正(librarian 报告漏掉、本 spec 实测补上)

`.claude/specs/ffi-shared-dylib.md` 已在关闭时删除(685bd24),悬空引用经 `git grep`
基线实测**共 17 处**(molrs 13 + molpack 4)——librarian 报 8 处、起草时修正为 13 处,
两者都低估。多出来的四处是 `molrs/Cargo.toml:7`(exclude 注释里的 axis 4)、
`molrs-cxxapi/Cargo.toml:22`、`molrs-ffi/Cargo.toml:79`(unify pins 的 Reuse decision)、
`CLAUDE.md:227`(Crate Structure 段的 cxxapi 那句),均在实现期补入。明细:
librarian 列出的 8 处(`molrs/Cargo.toml:42`、`molrs-ffi:37`、
`molrs-python/Cargo.toml:92`、`molrs-capi/Cargo.toml:23`、`molrs-cxxapi:35`、
`molpack/Cargo.toml:205`、`molpack/python/Cargo.toml:49`、
`verify-shared-dylib.sh:237`),外加 `molrs-python/build.rs:7`、
`molrs-capi/build.rs:100`、`molpack/build.rs:7`、`molpack/python/build.rs:7`、
`molrs-capi/tests/cpp/CMakeLists.txt:38`。这五处同时还断言了旧默认
("cfg-exempt from the prefer-dynamic default in .cargo/config.toml"、
"locally every native consumer resolves molrs through the one shared
libmolrs_ffi dylib"),落地后即为假,按 iron law 同一次改掉,不带走。
另:`molrs/Cargo.toml:43` 注释说 "These four keys are byte-identical" 而实际
只有三个键 —— 补回 `lto = "thin"` 后该句恰好重新成立。

### Reuse decision(逐条解决 librarian_report 的每个候选)

- `generalize` `scripts/verify-shared-dylib.sh:166,187` — 提取 `DYN_RUSTFLAGS`
  与 `DYN_CONFIG`,两次 `maturin build --release` 各自内联 `RUSTFLAGS=` 并带上
  profile 键(拆成两半的原因见上节实测)。**不写第二个动态构建脚本。**
- `generalize` `scripts/verify-shared-dylib.sh:159-162` — release-profile 理由
  注释改写为"manifest 已带 `lto = "thin"`,所以 `--release` 构建必须同时带
  `profile.release.lto=false`"。
- `reuse` `scripts/sync-dylib-locks.sh:62-70` `ROOTS=(...)` — 七个原生 root 的
  **唯一真源**,本 spec 的 `[profile.release]` 编辑面就是它,不另行罗列路径;
  只改写 line 56-61 里"cfg 把它们从 `-C prefer-dynamic` 里豁免"的说法
  (现在 wasm 的排除理由变成"它本来就没有 `[profile.release]`、且动态形态是
  命令行 opt-in")。
- `reuse unchanged` `regressions/ffi-shared-dylib.py` — 与链接形态无关(ABI
  token + capsule 穿越),继续作为动态门的 payload,由脚本 line 374 调用。
- `reuse unchanged` `.pre-commit-config.yaml:77-83` — flag 进脚本后自然正确。
- `generalize` `.github/workflows/ci-wasm.yml:3-4` — "No RUSTFLAGS override
  here" 在无人设置 RUSTFLAGS 之后变成无意义,改写为一句"零参数即静态"。
- `generalize` `molrs-python/pyproject.toml:73-90` +
  `molpack/python/pyproject.toml:143-158` — 删 `CARGO_PROFILE_RELEASE_LTO`,
  改注释,保留 `RUSTFLAGS`。
- `reuse unchanged` `molrs-ffi/Cargo.toml:21` `crate-type` — 不改类型,只改文字。
- `pattern`(本链不产生新 workflow,留给 02):`ci.yml:11-26` 纯编排、
  `ci-capi.yml:13-34` 模板、`ci-python.yml:39-53` uv+maturin、
  `molpack/ci.yml:70-84` 跨仓 checkout —— 记录在 02 号 spec。
- `new — 无既有等价物`:`regressions/link-mode-static-default.py`。现有
  `ffi-shared-dylib.py` 断言的是**动态**形态下的 capsule 穿越,无法证明
  "默认构建是静态"这一相反命题;两者互补,不是重复。

## Files to create or modify

- `molrs/.cargo/config.toml` — 删除 11-18 的注释与 `[target.'cfg(...)']` 段,只留 `[build] target-dir` 及其缓存注释
- `molpack/.cargo/config.toml` — 同上(9-16)
- `molrs/Cargo.toml` — `[profile.release]` 加 `lto = "thin"`;改写 39-46 注释(含悬空引用)
- `molrs/molrs-ffi/Cargo.toml` — 同上(34-38);改写 `[lib]` 13-20 说明
- `molrs/molrs-python/Cargo.toml` — 同上(86-93)
- `molrs/molrs-capi/Cargo.toml` — 同上(17-24)
- `molrs/molrs-cxxapi/Cargo.toml` — 同上(29-36)
- `molpack/Cargo.toml` — 同上(198-206)
- `molpack/python/Cargo.toml` — 同上(40-50)
- `molrs/scripts/verify-shared-dylib.sh` — `DYN_CONFIG` 数组 + 两次 maturin 调用;改写 159-162、176-179、237、311-318、358-361
- `molrs/scripts/sync-dylib-locks.sh` — 改写 56-61 注释(`ROOTS` 数组本身不动)
- `molrs/.github/workflows/ci-rust.yml` — 删 7-11 env 块及其注释
- `molrs/.github/workflows/ci-python.yml` — 删 7-11
- `molrs/.github/workflows/ci-capi.yml` — 删 7-11
- `molrs/.github/workflows/bench.yml` — 删 30-35 中两个键及注释
- `molrs/.github/workflows/nightly.yml` — 删 30-33
- `molrs/.github/workflows/publish.yml` — 删 14-17
- `molrs/.github/workflows/ci-wasm.yml` — 改写 3-4 注释
- `molpack/.github/workflows/ci.yml` — 13-21 共享 env 块**只删最后两个键**,保留 `MOLRS_GIT_REF` / sccache;改写 13-15 注释
- `molpack/.github/workflows/publish-crate.yml` — 删 14-20 中两个键
- `molpack/.github/workflows/publish-pypi.yml` — 删 15-21 中两个键
- `molpack/.github/workflows/bench.yml` — 删 23-30 中两个键
- `molrs/molrs-python/pyproject.toml` — tox `pass_env` 删 `CARGO_PROFILE_RELEASE_LTO`(80-83)
- `molpack/python/pyproject.toml` — 同上(150-153)
- `molrs/molrs-python/build.rs` — 注释改写 + 悬空引用(5-20)
- `molrs/molrs-capi/build.rs` — 同上(98-113、158-159)
- `molpack/build.rs` — 同上(5-20、65-66)
- `molpack/python/build.rs` — 同上
- `molrs/molrs-capi/tests/cpp/CMakeLists.txt` — 改写 35-39 注释 + 悬空引用
- `molrs/docs/interop.md` — 重写 § "Local link form"(173-216):三行表、理由、
  前置条件、豁免、maturin régime 段落;新增兼容契约两行 + pip build-id 未接线 TODO
- `molrs/CLAUDE.md` — 重写 150-164 "Link form is a switch" 段落(含 `touch molrs-ffi/src/lib.rs` 恢复说明)
- `molrs/.claude/specs/INDEX.md` — 修正第 37 行 "CI/发布 env 钉回静态" 的陈述
- `molrs/regressions/link-mode-static-default.py` (new) — 静态默认回归例

## Tasks

- [x] Delete the `[target.'cfg(...)'] rustflags` block from `molrs/.cargo/config.toml` and `molpack/.cargo/config.toml`, keeping only `[build] target-dir`
- [x] Restore `lto = "thin"` byte-identically to `[profile.release]` in the seven native roots enumerated by `scripts/sync-dylib-locks.sh` `ROOTS`, and rewrite their comment blocks
- [x] Move the two `--config` flags into `scripts/verify-shared-dylib.sh` as a `DYN_CONFIG` array used by both `maturin build --release` calls, and rewrite its four remediation/rationale texts
- [x] Delete `RUSTFLAGS` / `CARGO_PROFILE_RELEASE_LTO` from the 10 workflow env blocks in both repos and drop `CARGO_PROFILE_RELEASE_LTO` from both tox `pass_env` lists
- [x] Rewrite every prose site asserting the old default (`molrs-ffi/Cargo.toml` `[lib]`, four `build.rs`, `sync-dylib-locks.sh`, `ci-wasm.yml`, `molrs-capi/tests/cpp/CMakeLists.txt`, `CLAUDE.md`, `.claude/specs/INDEX.md:37`) and drop all 17 dangling `ffi-shared-dylib.md` citations
- [x] Rewrite `docs/interop.md` § "Local link form" with the two-row contract table, the measured patch-hash fact, and the pip build-id `==` mechanism marked explicitly un-wired
- [x] Add regression example `regressions/link-mode-static-default.py` (public API only; hard-coded goldens; no third-party runtime) and record its pre-flip RED run
- [x] Verify the dynamic opt-in by running `bash scripts/verify-shared-dylib.sh` green (both extensions carry the `libmolrs_ffi` entry; dylib sha256 identical across the two wheel builds)
- [x] Run full check + test suite in both repos

## Testing strategy

本变更没有进程内可单元测试的符号(改动面是构建配置 / CI 配置 / 陈述性文字),
molrs 也按 `CLAUDE.md` 明令**不存在** `molrs/tests/` 集成目录。因此验证一律
落在**观测产物属性**上,绝不断言配置字符串(禁止"不可能失败的测试")。

**回归例(本 spec 的 regression example):**
`regressions/link-mode-static-default.py`,仓库根 `regressions/` 下的最小公开
API 脚本,风格对齐既有 `regressions/ffi-shared-dylib.py`:

- 用 `importlib.util.find_spec("molrs._lib")` 定位扩展,按平台跑
  `otool -L`(Darwin)/ `ldd`(Linux),断言 `libmolrs_ffi` 条目数 **== 0**
  (硬编码 golden;这是"默认即静态"的产物级证据)。
- 跑一段最小公开 API:`molrs.Block()` 插入 `element/x/y/z/id` 三行 →
  `molrs.Frame()["atoms"] = block` → 读回并断言硬编码值
  (`ELEMENTS == ["C","C","O"]`,`x[1] == 1.5`)。
- 平台不支持(非 Darwin/Linux)→ **拒绝并解释**(打印观测值、期望值、编号
  修复步骤),绝不静默跳过 —— 与 `verify-shared-dylib.sh:35-37` 同一政策。
- 零第三方运行时依赖(numpy 是 molrs 自身运行时依赖,不是外部 oracle)。
- **该脚本在动态 opt-in 下必然红,这是正确的**:它断言的是发布形态。动态
  形态的 payload 是既有 `regressions/ffi-shared-dylib.py`,两者不得互换。

**产物级验证(happy path / edge / 反向门):**

1. 零参数默认:`cargo build -v` 输出中 `prefer-dynamic` 出现 **0** 次。
2. 动态 opt-in 真的还活着:`bash scripts/verify-shared-dylib.sh` 绿 —— 两个
   扩展**都有** `libmolrs_ffi` 动态链接条目,且 dylib sha256 在两次 wheel
   构建之间逐字相同(路径比较是空断言,已被脚本 step 4b 取代)。
3. env régime 清零:两仓 `.github/workflows/` 下 `RUSTFLAGS` /
   `CARGO_PROFILE_RELEASE_LTO` 命中 0;且全仓无新增自造环境变量。
4. 七根 `[profile.release]` 逐字同一且含 `lto = "thin"`;`molrs-wasm` 仍无该段。
5. 悬空引用清零:`rg -n 'ffi-shared-dylib\.md'` 两仓 0 命中。
6. 两仓默认套件绿:molrs `cargo test -p molcrafts-molrs --lib --features
   full,filesystem` + `cargo test --doc …`;molpack `cargo test --features io
   --lib --tests --examples`;两侧 `tox -e py`。
7. 已知副作用(需在 PR 描述里说明,不是失败):本地 `cargo bench` 现在也走
   LTO(此前只有 CI 通过 env 打开),因此本地 bench 变慢但与发布态一致。

## Out of scope

- 把动态形态用于**发布**(wheel / crate 仍自包含);
- pip 层 build-id local version(`0.14.1+abi.<hash>`)与 `==` 钉法 —— 本 spec
  只在 `docs/interop.md` 记为显式未接线 TODO,不实现;
- CI 双形态门(新 workflow / molpack job)—— 见 `link-mode-static-default-02-ci-gate`;
- C-ABI 路线(molpack 改走 molrs-capi):触发条件是 molrs core-sink 收口,
  规模实测为 molpack 跨 37 个文件用到 32 个不同 `molrs::` 条目;
- `/mol:map` blueprint 刷新(`architecture.md` 库存日期 2026-08-04 早于本表面);
- molrs-wasm / Pyodide 的 `--no-default-features` régime;
- **发现但不属本表面的 rot(按 iron law 记录并路由,不在此修)**:
  `molrs/.claude/notes/performance.md:115-118` 仍写着合并前的 crate 名
  `molcrafts-molrs-core` → 路由 `/mol:fix`。**janitor 复查补充:这不是一次纯
  改名** —— `molrs/benches/core/main.rs` 里只有 `core_benchmarks` /
  `compute_benchmarks` 两个 `[[bench]]` 目标,找不到 `potential`,所以那行
  `-- potential` 过滤器本身可能也已失效。盲目 s/molcrafts-molrs-core/
  molcrafts-molrs/ 会留下一条仍然跑不起来的命令,修时需要 perf 侧确认。
