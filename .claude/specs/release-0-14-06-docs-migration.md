---
slug: release-0-14-06-docs-migration
title: release-0-14-06-docs-migration — 0.13→0.14 迁移指南、md 用户指南、时间单位唯一事实源
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-04-md-driver
  - release-0-14-05-compute-protocol
  - release-0-14-13-frame-store-naming
---

# release-0-14-06-docs-migration — 迁移与用户指南

## Summary

写出 0.13→0.14 迁移指南与一页 md 用户指南，并把"分析时间单位"的三方矛盾收敛到唯一事实源。所有用户可见的示例一律拼 `molpy`——用户侧永远只有 `molpy.xxx`，`molrs` 一词不出现在教程、示例与错误提示里。

## Domain basis

分析时间单位的唯一事实源是 `.claude/notes/science.md` 的单位表：**Time = fs**（LAMMPS `units real`，https://docs.lammps.org/units.html；LAMMPS DOI 10.1016/j.cpc.2021.108171）。任何声称分析 `dt`/lag 用 **ps** 的段落都是 0.12 之前的遗留，必须删除而不是并列保留——并列的双单位正是本次三方矛盾的成因。md 子系统的能量/时间约定与 02 落地的 `core::units` preset 一致。

## Design

- **迁移指南** `molrs-python/site-src/getting-started/migration-0-14.md`（new）：逐项写 0.13 → 0.14 的破坏性改动与替换写法——typifier 路径/类名、md 子系统（`Potential` 是唯一接口且为结构化契约、`dtype=`、`kb=`、单位不再隐式换算）、keys 形状、Neighbors API、`metadata` → `meta`。每条"旧写法 → 新写法"两行代码，不写散文式解释。单位一节明写新路径：`molpy.UnitPreset("real")`（引擎侧 `molrs.UnitPreset`，来自 `molrs::core::units`），旧的 `md.energy_to_md` / `preset_energy_to_md` / `kb_md` / `Potentials.set_energy_scale` 全部消失。
- **md 用户指南** `molrs-python/site-src/guides/md.md`（new）：一页，从 `UnitPreset` 取常数 → 建 `VerletSkin` → 建 `LJCut` → `VelocityVerlet` → `advance_n`。自定义力（NN/Torch 接缝）的示例**不继承任何基类**——只定义 `calc_energy_forces`，正文点明 `molpy.md.Potential` 是结构化契约（`runtime_checkable` Protocol），显式继承是可选的表意手段。所有代码块拼 `molpy.md`。
- **拼写纪律。** 站点与 README 的用户可见示例一律 `import molpy` / `molpy.md.*`；引擎内部、molrs 仓测试与 `regressions/` 保留 `import molrs`。这条纪律用 grep 门守（见 Testing strategy），不靠人肉复查。
- **时间单位收敛。** 以 `.claude/notes/science.md` 为唯一事实源，清掉 `molrs-python/site-src/guides/transport.md` 与 `molrs-python/python/molrs/compute/dielectric.py` docstring 中残留的 ps 说法；`science.md` 补一句"分析域无 ps 并列单位"。
- **导航。** 两个新页写进 `molrs-python/zensical.toml` 的 nav，否则页面存在但没人走得到。

### Reuse decision

- `reuse` `release-0-12-06-docs-surface` 的 grep 门做法（`regressions/release-0-12-06-docs-surface.md`）作为本规范回归清单格式。
- `reuse` `.claude/notes/science.md` 单位表作为时间单位唯一事实源——不新建第二份单位说明。
- `reuse` 02 的 `UnitPreset`、03 的 `md.Potential` Protocol、04 的 `dtype=` 作为指南里唯一被教的 API 形状。
- `new` — 两个新文档页；既有站点没有 md 指南也没有迁移页。

## Files to create or modify

- `molrs-python/site-src/getting-started/migration-0-14.md` (new)
- `molrs-python/site-src/guides/md.md` (new)
- `molrs-python/site-src/guides/transport.md`
- `molrs-python/site-src/index.md`
- `molrs-python/zensical.toml`
- `molrs-python/python/molrs/compute/dielectric.py`
- `.claude/notes/science.md`
- `molrs-python/tests/test_docs_gates.py` (new)
- `regressions/release-0-14-06-docs-migration.md` (new)

## Tasks

- [ ] Write failing doc grep gates: no `ps` analysis-time claim, no user-facing `import molrs` in site-src code blocks (`molrs-python/tests/test_docs_gates.py`)
- [ ] Write the 0.13→0.14 migration guide covering typifier, md, units, keys, Neighbors and metadata→meta in `molrs-python/site-src/getting-started/migration-0-14.md`
- [ ] Write the one-page md user guide in `molrs-python/site-src/guides/md.md` with a no-inheritance custom force and `molpy` spelling throughout
- [ ] Remove the residual ps analysis-time claims from `molrs-python/site-src/guides/transport.md` and `molrs-python/python/molrs/compute/dielectric.py`
- [ ] Update `.claude/notes/science.md` to state fs as the sole analysis time unit with no ps dual
- [ ] Register both new pages in `molrs-python/zensical.toml` nav and link them from `molrs-python/site-src/index.md`
- [ ] Add regression example `regressions/release-0-14-06-docs-migration.md` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

- 门测试在 `molrs-python/tests/test_docs_gates.py`，每条一个断言：
  - site-src 所有 ```python 代码块中 `import molrs` / `molrs.` 命中为零（**目录扫描**，不写死文件名单）。
  - site-src 与 `molrs-python/python/molrs/compute/**` 中，analysis / lag / dt 上下文里 `ps` 命中为零。
  - 两个新页在 `zensical.toml` nav 中出现。
  - 迁移指南与 md 指南中不出现任何已删符号（`energy_to_md`、`preset_energy_to_md`、`kb_md`、`set_energy_scale`、`prec=`、`resolve_prec`）。
- 文档内代码块的可执行性由 07 接线的 examples CI 冒烟覆盖；本规范只保证静态一致。
- 回归样例 `regressions/release-0-14-06-docs-migration.md`：checklist，逐条写死 grep 谓词与预期命中数（沿用 0-12-06 先例）。
- 域验证：`science.md` Time 行为 `fs`，全站无 ps 并列。

## Open questions (maintainer ruling required)

- 无（时间单位以 `science.md` 为准已是既定裁决；三方矛盾属执行债，不是待裁事项）。

## Out of scope

- README / quickstart / examples 的可执行性修复与 CI 冒烟（07）
- molpy 侧文档与 typifier 拼写清扫（11）
- 错误信息改进（07）
