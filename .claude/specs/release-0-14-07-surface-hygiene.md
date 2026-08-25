---
slug: release-0-14-07-surface-hygiene
title: release-0-14-07-surface-hygiene — README/examples 可执行、错误信息、wasm 对称性改期、过时蓝图
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-06-docs-migration
---

# release-0-14-07-surface-hygiene — 表面卫生

## Summary

把 0.14 发布前的表面腐坏一次清干净：修好 README / quickstart / examples 里已知会报错的路径并把 examples 接进 CI 冒烟；改善三处错误信息；把 wasm `NeighborQuery` 对称项**显式改期到 0.15 并写明证据**；修正两个 binder crate rustdoc 的 f32 谎言；补 gitignore；重写两仓 CLAUDE.md / architecture.md 的过时段落。

## Domain basis

N/A — 表面与工具链卫生，不含物理。唯一与科学面相关的是 rustdoc 的 f32 声称：molrs 的类型精度原则是 `F = f64` 恒定（CLAUDE.md § *Type Precision Principle*），binder rustdoc 写 f32 是**事实错误**，会让下游按错误精度接线——按铁律必须当场改，不能标注"先存"。

## Design

- **README / quickstart 可执行性。** 三处已知硬失败：`Conformer.generate` 返回元组（文档按单值用）、`nlist` `NameError` × 4（示例引用未定义变量）、`molrs-python/examples/embed_water.py` 的 `ImportError`。修法一律"改文档/示例去贴合已落地 API"，不为迁就文档改 API。
- **examples 进 CI 冒烟。** `molrs-python/examples/*.py` 三个脚本接进 `.github/workflows/ci-python.yml` 的冒烟步骤。示例一旦不可执行就红——这是唯一能防止它们再次腐坏的机制（"能被手写的清单会被手工缩短"，所以用 glob 跑目录里所有 `.py`，不写死三个文件名）。
- **错误信息改进（三处）：** `read_pdb` 失败时带上文件名；PyO3 参数错误带上参数名；`Block` 缺列时列出候选列名。每条都用一条断言消息内容的测试守住——错误信息是公开 API 的一部分。
- **rustdoc f32 谎言。** 两个 binder crate 的 rustdoc 声称 f32；molrs 的 `F = f64` 恒定。逐处改正，并加一条 grep 门防复发。
- **"Exposed as `molrs.X`" 路径谎言。** binder rustdoc 里若干条声称的 Python 路径与实际公开路径不符（例如声称 `molrs.Potentials` 而文档化路径是 `molrs.ff.Potentials`）。用**谓词**发现（扫描所有 `Exposed .* as \`molrs\.` 注释，与实际注册路径对拍），不用手写名单。
- **wasm `NeighborQuery`：0.15 改期，带证据（维护者裁定）。** `.claude/notes/notes.md` 的 binder-surface-symmetry 条目把它列为第一优先。核查结论是**删除不在选项内**：`NeighborQuery` 仍被 molrs 自己四处消费——`compute/hbond/detect.rs:147,149`（`from_columns` / `free_columns` 跨查询前门，承载 `QueryMode::CrossQuery`）、`compute/rdf/mod.rs:501`、`compute/dynamics/van_hove.rs:203`、`ff/potential/soft.rs:18`。因此 0.14 **不闭环、不删除**，而是把它记为**带日期的 0.15 改期**：条目里写明日期、目标版本、理由（wasm 侧尚无消费者，facade-first）与上述四处消费证据——证据必须在案，否则下一轮又会有人提"删掉算了"。给 wasm 补对称门本身属 0.15。
- **gitignore。** molrs 的 `target-aarch64/`（现 `.gitignore` 只忽略 `target`）；molpy 的 `benchmarks/md/`（内含整棵 LAMMPS 源码树）。
- **过时蓝图。** `.claude/notes/architecture.md` 形式上 `stale: false`，但日期停在 2026-08-04 / 0.12-prep 且**完全没提 `molrs/src/md`**；CLAUDE.md 与 molpy 侧同类段落也有 FieldSpec 幽灵层、legacy/op/embed/tool 幽灵包、`metadata`→`meta`、md/builder 缺行等错误。逐条改正。**注意排序**：13 号已把 `MolRec` 从这两份文档的命名行里清掉、14 号已改掉 CLAUDE.md 的类别行，本规范的重写不得把它们写回来。

### Reuse decision

- `reuse` `.pre-commit-config.yaml` / `ci-python.yml` 既有冒烟步骤形态，examples 冒烟作为一个新 step 接入，不新建工作流。
- `reuse` `.claude/notes/notes.md` binder-surface-symmetry 条目作为 wasm 改期的记录位置（就地更新状态，不另起笔记）。
- `reuse` 06 建立的 `molrs-python/tests/test_docs_gates.py` 承载新的 grep 门，不新建门文件。
- `generalize` `.claude/notes/architecture.md` 蓝图：从 0.12 单晶格快照扩到含 `md` 的当前模块图，服务 architect / librarian 两类调用方。
- `new` — none.

## Files to create or modify

- `README.md`
- `molrs-python/README.md`
- `molrs-python/examples/embed_water.py`
- `molrs-python/examples/full_pipeline.py`
- `molrs-python/examples/forcefield_ethane.py`
- `molrs-python/site-src/getting-started/quickstart-python.md`
- `.github/workflows/ci-python.yml`
- `molrs-python/src/io/mod.rs`
- `molrs-python/src/core/store/block.rs`
- `molrs-python/src/ff/mod.rs`
- `molrs-python/src/conformer/mod.rs`
- `molrs-python/src/core/spatial/region.rs`
- `molrs-wasm/src/lib.rs`
- `.gitignore`
- `.claude/notes/architecture.md`
- `.claude/notes/notes.md`
- `CLAUDE.md`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/.gitignore`
- `regressions/release-0-14-07-surface-hygiene.md` (new)

## Tasks

- [ ] Write failing tests for the three improved error messages (`read_pdb` names the file, PyO3 errors name the argument, `Block` lists column candidates) in `molrs-python/tests/test_io.py` and `molrs-python/tests/test_block.py`
- [ ] Write failing example-smoke test that globs `molrs-python/examples/*.py` and executes each one
- [ ] Fix the README / quickstart / examples breakages (`Conformer.generate` tuple, 4 × `nlist` NameError, `embed_water.py` ImportError)
- [ ] Wire the examples smoke step into `.github/workflows/ci-python.yml`
- [ ] Implement the three error-message improvements in `molrs-python/src/io/mod.rs`, `molrs-python/src/core/store/block.rs` and the PyO3 argument-parsing helpers
- [ ] Correct the binder rustdoc f32 claims and every mismatched "Exposed as `molrs.X`" path, found by predicate rather than a hand list
- [ ] Record the dated 0.15 deferral of the wasm `NeighborQuery` symmetry item in the `.claude/notes/notes.md` binder-surface-symmetry entry, naming the target version, the reason, and the four in-tree consumers that put deletion off the table
- [ ] Add `target-aarch64/` to `.gitignore` and `benchmarks/md/` to `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/.gitignore`
- [ ] Rewrite the stale paragraphs in `.claude/notes/architecture.md` and `CLAUDE.md` (FieldSpec ghost layer, legacy/op/embed/tool ghost packages, metadata→meta, missing md/builder rows) without reintroducing names 13 and 14 removed
- [ ] Add regression example `regressions/release-0-14-07-surface-hygiene.md` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

- 错误信息（每条一个断言）：`read_pdb("missing.pdb")` 的异常消息含 `"missing.pdb"`；某 PyO3 方法传错类型时消息含参数名；`Block.get_float("xx")` 消息含至少一个真实列名。
- examples 冒烟：glob `molrs-python/examples/*.py`，逐个 `subprocess` 执行，退出码必须为 0。**用 glob，不写死名单。**
- rustdoc 门：`cargo doc` 的既有 `-D warnings` 门不变；新增 grep 门断言 binder crate 源码中不含 `f32` 精度声称、且每条 `Exposed .* as \`molrs\.` 与注册路径一致。
- wasm 改期门：`.claude/notes/notes.md` 的 binder-surface-symmetry 条目含日期、目标版本 `0.15` 与四处消费者路径；同时断言 `NeighborQuery` 在 molrs 侧**仍然存在且被消费**（四处 import 命中 > 0）——改期的前提是它还活着，这条断言让"顺手删掉"当场变红。
- gitignore 门：`git check-ignore target-aarch64` 与 molpy 侧 `benchmarks/md` 均返回命中。
- 蓝图门：`.claude/notes/architecture.md` 含 `molrs/src/md` 模块行；不含 `FieldSpec` / `legacy` / `op` / `embed` / `tool` 幽灵包；不含 `metadata` 作为 Frame 字段名的说法；不含 13 / 14 已删除的名字（`MolRec`、`kspace` 类别）。
- 回归样例 `regressions/release-0-14-07-surface-hygiene.md`：checklist，逐条写死谓词与预期。
- 域验证：不适用。

## Open questions (maintainer ruling required)

- 无。

## Out of scope

- 新功能与 API 扩张
- wasm 侧对称 `NeighborQuery` 门的实现（0.15，已带日期改期）
- `KSpace` 表面删除与 PME 归位 pair（14）
- `MolRec` / zarr 公开命名（13）
- molpy 侧文档清扫（11）
- tag / 发布（08）
