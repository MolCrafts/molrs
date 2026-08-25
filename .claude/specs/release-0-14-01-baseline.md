---
slug: release-0-14-01-baseline
title: release-0-14-01-baseline — 0.14 发布基线：合并 master、统一版本串、补齐 release 记录
status: approved
grilled: true
created: 2026-08-25
depends_on: []
---

# release-0-14-01-baseline — 0.14 发布基线

## Summary

把 molrs `dev` 与 `origin/master` 合并成 0.14.0 的唯一发布基线，统一全部版本串到 `0.14.0`（删除 `.dev1` 预发布后缀），并把 0.13.0 / 0.13.1 / 0.13.2 / 0.14.0 四条发布记录补进 `.claude/notes/release.md`。本链后续每一条规范都落在这棵合并后的树上，所以它必须先绿。

## Domain basis

N/A — 本规范不改动任何物理量或数值路径（发布基线与清单卫生）。合并冲突的裁决依据是"已发布形状即契约"这一发布纪律，不是科学结论。

## Design

- **合并方向与冲突裁决。** 在 `dev` 上 `merge origin/master`。唯一实质冲突在 keys：`molrs/src/core/store/keys.rs` 与其 PyO3 注册面 `molrs-python/src/schema.rs`，**一律取 master 已发布的 `Key` 对象形状**——master 已公开发布，形状即对下游的契约；dev 侧的字符串化便利不进 0.14，改由 molpy 的 `_key_str` 兼容层在 09 收敛到同一形状。
- **版本串（6 + 2 处）。** 6 个 crate 清单：`Cargo.toml`（workspace `package.version` 与 `molcrafts-molrs` 路径依赖 pin）、`molrs-ffi/Cargo.toml`、`molrs-wasm/Cargo.toml`、`molrs-capi/Cargo.toml`、`molrs-cxxapi/Cargo.toml`、`molrs-python/Cargo.toml`（各自 `version` 与对 `molcrafts-molrs` / `molcrafts-molrs-ffi` 的 pin）。另 2 处：`molrs-python/pyproject.toml` 的 `0.13.2.dev1` → `0.14.0`（`.dev1` 删除），以及 workspace 依赖 pin 行。全部为 `0.14.0`，**不留任何预发布后缀**。
- **版本漂移永久化门。** 版本一致性不靠人肉 grep：新增 `molrs-python/tests/test_version_parity.py`，用 `tomllib` 读全部清单，断言它们与 `pyproject.toml` 同版本。这是"能被手写的清单就能被手工缩短"这条既有教训的直接应用——门用目录扫描/谓词，不用硬编码名单。
- **release.md 记录。** 按现有 `## v0.12.1 (YYYY-MM-DD)` 段落格式补 0.13.0 / 0.13.1 / 0.13.2 / 0.14.0 四段；0.14.0 段先写内容摘要，tag 与发布事实由 08 落定。
- **发布顺序铁律重申。** CLAUDE.md § *Release before molpy* 不变：molrs 先 tag。本链 09–12 的任何 molpy 改动都不得早于 08。

### Reuse decision

- `reuse` `.claude/notes/release.md` 既有 `v0.12.1` / `v0.12.2` 段落格式作为四条新记录的模板。
- `reuse` CLAUDE.md § *Release before molpy* 铁律，本规范只重申不改写。
- `new` — `molrs-python/tests/test_version_parity.py`：现有测试树里没有任何清单一致性门（唯一近邻 `test_units.py` 是物理单位，不是版本），而 "6+2 处" 正是靠人记忆维持的那类不变式。
- librarian 报告的 Area 1/2/3 候选不属于本规范，分别在 03 / 02 / 05 结算（见链级 reuse ledger）。

## Files to create or modify

- `Cargo.toml`
- `molrs-ffi/Cargo.toml`
- `molrs-wasm/Cargo.toml`
- `molrs-capi/Cargo.toml`
- `molrs-cxxapi/Cargo.toml`
- `molrs-python/Cargo.toml`
- `molrs-python/pyproject.toml`
- `molrs/src/core/store/keys.rs`
- `molrs-python/src/schema.rs`
- `.claude/notes/release.md`
- `molrs-python/tests/test_version_parity.py` (new)
- `regressions/release-0-14-01-baseline.md` (new)

## Tasks

- [ ] Write failing version-parity test over every manifest (`molrs-python/tests/test_version_parity.py`, `TestVersionParity`)
- [ ] Merge `origin/master` into `dev`, resolving `molrs/src/core/store/keys.rs` and `molrs-python/src/schema.rs` to the published master `Key` shape
- [ ] Bump the six crate manifests to `0.14.0` including every path-dependency pin
- [ ] Bump `molrs-python/pyproject.toml` to `0.14.0` and delete the `.dev1` suffix
- [ ] Backfill `.claude/notes/release.md` with 0.13.0 / 0.13.1 / 0.13.2 / 0.14.0 records
- [ ] Add regression example `regressions/release-0-14-01-baseline.md` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

molrs 的测试约定优先于通用镜像布局：Rust 单测就地 `#[cfg(test)]`（无 `molrs/tests/` 树），Python 单测平铺在 `molrs-python/tests/test_*.py`。

- 单测（happy path）：`TestVersionParity::test_all_manifests_agree` 解析 8 处版本串，断言全等于 `"0.14.0"`。
- 单测（edge）：断言没有任何清单带预发布后缀（`.dev` / `rc` / `a` / `b`）——这是本轮实际踩到的形态。
- 单测（edge）：keys 形状回归——合并后 `molrs.keys` 的公开常量集合与 master 发布形状一致（`molrs-python/tests/test_ecs_pybind.py` 既有断言必须原样通过，不得放宽）。
- 回归样例 `regressions/release-0-14-01-baseline.md`：checklist 形态（沿用 `regressions/release-0-12-01-harness-check.md` 先例），逐条写死 8 个版本串位置、四条 release 记录标题、以及 keys 冲突裁决结论。
- 域验证：不适用。

## Open questions (maintainer ruling required)

- 无。本规范只执行已裁决项。

## Out of scope

- md 子系统任何代码改动（02–04）
- compute 契约（05）
- 文档与卫生（06–07）
- tag / 发布动作（08）
- 任何 molpy 改动（09–12，且必须在 08 之后）
