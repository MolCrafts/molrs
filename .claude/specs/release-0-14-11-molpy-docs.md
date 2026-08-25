---
slug: release-0-14-11-molpy-docs
title: release-0-14-11-molpy-docs — typifier 拼写全站清扫、用户可见 molrs 词清零、molpy 迁移指南
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-10-molpy-mirror
---

# release-0-14-11-molpy-docs — 用户侧只有 molpy

## Summary

清扫 molpy 用户可见面的两类拼写债：typifier 类名/路径的 7 种错误拼写（`OplsTypifier` 等，全站 98 处 / 35 文件），以及 `molrs` 一词出现在用户可见处（`molpy/src` 253 处 / 54 文件 + 文档）。同时写出 molpy 侧的 0.13→0.14 迁移指南。

## Domain basis

N/A — 文档与拼写。分析时间单位沿用 06 号规范确立的唯一事实源（**fs**，`.claude/notes/science.md`）；molpy 文档中任何 ps 说法按同一裁决清除。

## Design

- **typifier 拼写清扫。** 7 种错误拼写（`OplsTypifier` 等）在 molpy 的 `docs/`、`docs/zh/`、`CHANGELOG.md`、`scripts/`、`.claude/notes/` 共 35 个文件、98 处。逐处改成实际存在的类名与导入路径。**用谓词发现**：以 molpy 运行时实际存在的 typifier 类名集合为真值，扫描文档中所有 `*Typifier` 拼写，任何不在集合内的即为错——而不是维护一张"7 种错误拼写"的名单（名单会漏第 8 种）。
- **`molrs` 词清零（用户可见面）。** 用户侧永远只有 `molpy.xxx`：文档、教程、docstring 示例、错误提示一律拼 `molpy`。范围是 `<molpy>/src/molpy/**` 的 **docstring 与错误消息**、`<molpy>/docs/**`、`<molpy>/README.md`。**不在范围内**：`import molrs` 的真实实现语句（引擎侧实现细节）、内部注释解释委托关系的地方。门必须能区分这两者——只扫 docstring / 字符串字面量 / 文档正文，不扫 import 语句。
- **molpy 迁移指南。** `<molpy>/docs/getting-started/migration-0-14.md`（new），与 06 的 molrs 侧指南互补：typifier 路径/类名、`molpy.md` 新表面、keys 形状、Neighbors API、`metadata` → `meta`、compute 契约改造对自定义子类的影响。中英双语站点各一份（`docs/` 与 `docs/zh/`），并写进 `<molpy>/zensical.toml` nav。
- **双语一致。** molpy 有 `docs/` 与 `docs/zh/` 两套；任何清扫必须两套同改，否则中文站会长期保留旧拼写。加一条两套页面集合一致性的门。

### Reuse decision

- `reuse` 06 号规范建立的 grep 门写法与 `.claude/notes/science.md` 单位事实源。
- `reuse` molpy 运行时的 typifier 类名集合作为拼写真值（谓词，不是名单）。
- `reuse` `<molpy>/zensical.toml` 现有 nav 结构与双语目录约定。
- `new` — 两份迁移指南页（中/英）；molpy 侧目前没有 0.14 迁移文档。

## Files to create or modify

- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/docs/getting-started/migration-0-14.md` (new)
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/docs/zh/getting-started/migration-0-14.md` (new)
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/docs/user-guide/06_typifier.md`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/docs/zh/user-guide/06_typifier.md`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/docs/getting-started/quickstart.md`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/docs/zh/getting-started/quickstart.md`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/README.md`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/CLAUDE.md`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/.claude/notes/architecture.md`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/zensical.toml`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/tests/test_docs_spelling.py` (new)
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/regressions/release-0-14-11-molpy-docs.md` (new)

## Tasks

- [ ] Write failing spelling gates: every `*Typifier` name in docs resolves to a real molpy class, and no user-visible `molrs` spelling survives in docstrings, error messages or docs (`/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/tests/test_docs_spelling.py`)
- [ ] Write failing bilingual-parity gate asserting `docs/` and `docs/zh/` expose the same page set
- [ ] Fix every mismatched typifier spelling and import path across molpy docs, README, CHANGELOG and notes
- [ ] Rewrite user-visible `molrs` spellings to `molpy` in docstrings, error messages and docs, leaving engine-side import statements untouched
- [ ] Write the molpy 0.13→0.14 migration guide in `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/docs/getting-started/migration-0-14.md` and its `docs/zh/` counterpart
- [ ] Register both migration pages in `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/zensical.toml` nav
- [ ] Update the stale paragraphs in molpy `CLAUDE.md` and `.claude/notes/architecture.md` (metadata→meta, compute conforming to the molrs Compute Protocol, md surface)
- [ ] Add regression example `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/regressions/release-0-14-11-molpy-docs.md` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

- 拼写门（谓词而非名单）：以 `dir(molpy.ff)`（或 typifier 所在模块）中实际的 `*Typifier` 类名为真值集合，扫描 `docs/**`、`README.md`、`CHANGELOG.md` 中所有 `\w+Typifier` 拼写，不在集合内即失败——失败信息列出错误拼写与最近的合法拼写。
- `molrs` 词门：扫描 `<molpy>/src/molpy/**` 的 docstring 与字符串字面量（用 `ast` 解析，**不**用行级 grep，这样 `import molrs` 天然不被误伤）+ `docs/**` 正文；命中即失败。
- 双语一致门：`docs/` 与 `docs/zh/` 的相对路径集合相等。
- 边界：迁移指南两份都在 nav 中；两份的小节标题一一对应。
- 回归样例 `<molpy>/regressions/release-0-14-11-molpy-docs.md`：checklist，写死三个门的谓词与预期命中数（0）。
- 域验证：文档中分析时间单位一律 fs，无 ps 并列。

## Open questions (maintainer ruling required)

- 无（拼写纪律与单位事实源均已裁决；本规范是执行）。

## Out of scope

- 代码行为改动（10 已完成对象层融合）
- 联合冒烟与 molpy tag（12）
- 任何 molrs 侧改动
