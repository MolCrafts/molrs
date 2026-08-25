---
slug: release-0-14-12-joint-smoke
title: release-0-14-12-joint-smoke — 联合冒烟门与 molpy 0.14.0 收官
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-11-molpy-docs
---

# release-0-14-12-joint-smoke — 联合冒烟与收官

## Summary

在 molpy 新分支 × molrs 0.14 正式 wheel 的组合上跑联合冒烟门——molpy 全量 import + molnex 链 import——全绿后给 molpy 打 0.14.0 tag，并把两仓的 spec INDEX 收口。

## Domain basis

N/A — 集成门与发布收官。冒烟门只证明"装得上、导得进、跑得动"，不重复验证物理；物理断言的权威仍是 molrs 侧的 Rust NVE 单测与 `regressions/`（04）。

## Design

- **联合冒烟门（两段）：**
  1. **molpy 全量 import：** 对 `molpy` 包内每个子模块做 `importlib.import_module`（用 `pkgutil.walk_packages` **枚举**，不写死名单），在装有 molrs 0.14.0 正式 wheel 的环境里全部成功。这是唯一能抓住"某个角落还引用着 0.14 删掉的符号"的形态——单元测试只覆盖被测到的模块。
  2. **molnex 链 import：** molnex 消费链上的 import 冒烟，在 `.wheels-gh200` 已换成 0.14.0 件的 aarch64 venv 上跑。
- **`FutureWarning` 预期。** `import molpy` 不应因 md 的 experimental 警告而变吵：`molpy` 顶层保持惰性，`molpy.md` 首次访问才发警告。冒烟门显式断言这一点。
- **打 tag。** molpy `0.14.0` 落 master 并 tag。顺序上这必然晚于 molrs `v0.14.0`（08），铁律满足。
- **收口。** 两仓 `.claude/specs/INDEX.md` 记录 release-0-14 链完成状态；molrs `.claude/notes/notes.md` 的 md experimental 条目状态从 `provisional` 更新为发布后的实际状态。

### Reuse decision

- `reuse` `pkgutil.walk_packages` 枚举式全量 import——这是本仓既有"目录扫描优于手写名单"教训的直接应用。
- `reuse` 08 落地的 `.wheels-gh200` 0.14.0 wheel 与 aarch64 venv 作为冒烟环境，不另建环境。
- `reuse` `.claude/notes/release.md` 手工 checklist 作为 molpy tag 的流程依据。
- `new` — `<molpy>/tests/test_full_import.py`：目前没有任何门覆盖"每个子模块都导得进"，而这正是跨版本删除符号后最常见的漏网形态。

## Files to create or modify

- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/tests/test_full_import.py` (new)
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/.claude/specs/INDEX.md`
- `.claude/specs/INDEX.md`
- `.claude/notes/notes.md`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/regressions/release-0-14-12-joint-smoke.py` (new)

## Tasks

- [ ] Write failing full-import test enumerating every molpy submodule via `pkgutil.walk_packages` (`/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/tests/test_full_import.py`)
- [ ] Write failing warning-scope test: `import molpy` silent, first `molpy.md` access emits exactly one FutureWarning
- [ ] Run the joint smoke gate on the molpy 0.14 branch against the released molrs 0.14.0 wheel and fix every import failure it surfaces
- [ ] Run the molnex chain import smoke on the aarch64 venv built from `.wheels-gh200`
- [ ] Land molpy 0.14 on master and tag `0.14.0`
- [ ] Update `.claude/specs/INDEX.md` in both repos and refresh the md experimental entry in `.claude/notes/notes.md`
- [ ] Add regression example `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/regressions/release-0-14-12-joint-smoke.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

- 全量 import（happy path，枚举式）：`pkgutil.walk_packages(molpy.__path__)` 逐个 `import_module`，任何 `ImportError` / `AttributeError` 即红，失败信息带模块名。
- 警告作用域：`import molpy` 在 `catch_warnings(record=True)` 下零 `FutureWarning`；首次 `molpy.md` 访问恰好一条。
- molnex 链：在 aarch64 venv 上执行链上 import 序列，退出码 0。
- 边界：装着 **0.13.x** molrs 时全量 import 应失败（证明 pin 门确实有效，而不是碰巧能跑）——这条作为 bite-proof 手工执行一次并记录，不进常规 CI。
- 回归样例 `<molpy>/regressions/release-0-14-12-joint-smoke.py`：只 `import molpy`，跑一段最小端到端（建体系 → 分析 → 5 步 MD），断言写死黄金值；这是 0.14 对用户的"能用"证明。
- 域验证：不重复；引用 04 的 NVE 结论。

## Open questions (maintainer ruling required)

- 无。

## Out of scope

- 任何代码设计改动（若冒烟门暴露设计问题，回到相应规范或另开，不在收官步里改设计）
- 0.15 路线项（wasm/capi md 对称面、float32 积分器、LJCut 合并）
