---
slug: release-0-14-09-molpy-rebase
title: release-0-14-09-molpy-rebase — molpy 从 upstream/master 重开分支并收敛 keys 兼容层
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-08-ship-molrs
---

# release-0-14-09-molpy-rebase — molpy 重开基线

## Summary

molpy 从 `upstream/master`（v0.13.1）重开分支，把 dev 独有的 8 个提交（`7888e335..44adb594`）cherry-pick 过来，让 `_key_str` 兼容层收敛到 molrs master 已发布的 `Key` 对象形状，并把版本与 molrs pin 提到 0.14.0 / `>=0.14.0,<0.15`。本规范是 molpy 侧一切工作的基线，且**必须**在 molrs `v0.14.0` 已发布之后执行。

## Domain basis

N/A — 分支拓扑与依赖 pin。keys 形状裁决的依据是发布纪律（已发布形状即契约），不是科学结论；`_key_str` 的收敛不改变任何字段语义或数值。

## Design

- **重开分支。** 从 `upstream/master`（v0.13.1）新建 0.14 工作分支；**不**从现有 dev（0.7.0-based）继续——dev 与 upstream 已分叉到无法可靠合并的程度，重开比合并便宜且可审。
- **cherry-pick 8 个提交。** `7888e335..44adb594` 逐个 pick，每个 pick 后跑一次 molpy 测试门；冲突就地解决，**不**攒到最后。
- **`_key_str` 收敛。** `<molpy>/src/molpy/core/fields.py` 的 `_key_str` 是 dev 侧为适配旧 keys 形状而生的兼容层。molrs master 的 `Key` 对象形状已在 01 定为真值，本规范让 molpy 直接消费该形状；兼容层若在收敛后无任何调用方，**删除**（无人消费的兼容层就是债）。
- **版本与 pin。** `<molpy>/src/molpy/version.py` 提到 `0.14.0`；`<molpy>/pyproject.toml:47` 的 `molcrafts-molrs==0.7.0` 改为 `molcrafts-molrs>=0.14.0,<0.15`（共享 pin 规则：major.minor 必须匹配，patch 可不同）。当前 `==0.7.0` 的死 pin 本身就是腐坏，顺手改正。
- **顺序守卫。** 本规范开工前必须确认 molrs `v0.14.0` 已在 PyPI 可解析——把这条做成 molpy 侧的一个测试，而不是口头约定。

### Reuse decision

- `reuse` molrs 已发布的 `Key` 形状作为唯一真值——molpy 不再持有第二套 key 拼写规则。
- `reuse` `.claude/notes/release.md` 的共享 pin 规则（`>=X.Y.0,<X.(Y+1)`）。
- `reuse` upstream/master v0.13.1 作为分支基点（不重写历史，不 rebase dev）。
- `new` — `<molpy>/tests/test_molrs_pin.py`：目前没有任何门阻止 molpy 装着过期 molrs 跑测试，而 `==0.7.0` 这个死 pin 正是它缺席的证据。

## Files to create or modify

- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/pyproject.toml`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/version.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/core/fields.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/tests/test_molrs_pin.py` (new)
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/regressions/release-0-14-09-molpy-rebase.py` (new)

## Tasks

- [ ] Write failing pin-guard test asserting the installed molrs is >= 0.14.0 and the declared pin is `>=0.14.0,<0.15` (`/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/tests/test_molrs_pin.py`)
- [ ] Write failing key-shape test asserting molpy consumes molrs `Key` objects directly with no string-coercion layer
- [ ] Branch molpy from `upstream/master` (v0.13.1) and cherry-pick `7888e335..44adb594`, running the test gate after each pick
- [ ] Converge `_key_str` in `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/core/fields.py` onto the published `Key` shape and delete it if it loses every caller
- [ ] Bump `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/version.py` to 0.14.0 and the molrs pin in `pyproject.toml` to `>=0.14.0,<0.15`
- [ ] Add regression example `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/regressions/release-0-14-09-molpy-rebase.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

molpy 单测位于 `<molpy>/tests/`，平铺命名。

- 顺序守卫（happy path）：`importlib.metadata.version("molcrafts-molrs")` 解析为 `>= 0.14.0`；`pyproject.toml` 声明的 pin 字符串等于 `>=0.14.0,<0.15`（两条独立断言——装了什么与声明了什么是两件事）。
- keys 形状：用 molrs `Key` 对象直接读写一个 Frame 列，断言与旧 `_key_str` 路径结果**逐位相同**；若 `_key_str` 已删，断言符号不存在。
- cherry-pick 完整性：8 个提交的主题行全部出现在新分支 `git log` 中（用提交范围枚举，不手抄标题）。
- 回归样例 `<molpy>/regressions/release-0-14-09-molpy-rebase.py`：用公开 `molpy` API 建一个小体系、按 `Key` 读列、打印版本，断言写死的列值与版本串。
- 域验证：不适用。

## Open questions (maintainer ruling required)

1. **8 个 cherry-pick 中若有与 upstream 已含变更重复者，是跳过还是保留？** 逐个由维护者在 grill 时确认（当前假设：重复则跳过并在提交信息记录）。

## Out of scope

- 全表面下沉与 compute 契约改挂（10）
- 文档与拼写清扫（11）
- 联合冒烟与 molpy tag（12）
- 任何 molrs 侧改动（molrs 已 tag，不得再动）
