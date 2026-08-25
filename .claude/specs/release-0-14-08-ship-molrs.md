---
slug: release-0-14-08-ship-molrs
title: release-0-14-08-ship-molrs — molrs 先 tag：v0.14.0 落 master、发布、下游 wheel 顶掉 .dev1
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-07-surface-hygiene
---

# release-0-14-08-ship-molrs — molrs 先发

## Summary

按发布铁律先发 molrs：把 0.14 分支落到 `master`、打 `v0.14.0`、等 Publish 工作流把 crates.io / npm / PyPI 三个 registry 发完，然后用正式 0.14.0 wheel 顶掉 molnex `.wheels-gh200` 里的 `.dev1` 预发布件，并重建 aarch64 venv。molpy 的任何改动都不得早于本规范绿。

## Domain basis

N/A — 发布操作。唯一的"硬约束"是流程性的：APIs molpy 将 pin 的必须先在 `master` 上带 `vX.Y.Z` tag 并发布（CLAUDE.md § *Release before molpy*、`.claude/notes/release.md`）。本地 `maturin develop` 不算发布。

## Design

- **落 master。** 0.14 工作分支合并进 `master`，`master` 上跑完整门（`prek run --all-files --hook-stage pre-push` + rust 单测 + doctest + molrs-python tox）。
- **打 tag。** `v0.14.0`，触发 `.github/workflows/publish.yml` 的五个 job（`publish-molrs` → crates.io、`publish-wasm` → npm、`build-python` / `build-python-pyodide` → wheel artifact、`publish-python` → PyPI）。任一 job 红即停，不做"先发一半"。
- **下游 wheel。** molnex 的 `.wheels-gh200/` 目录里当前躺着 `0.13.2.dev1` 的预发布 wheel。发布后放入正式 `molcrafts_molrs-0.14.0-*.whl` 并**删除** `.dev1` 件——同时存在两份是下游最容易装错的形态。
- **aarch64 venv。** 用 `gh200_bootstrap.sbatch` 在 tag 之后重建 venv，确保 GH200 侧装到的是正式件。
- **补记 release.md。** 01 写下的 0.14.0 段在这里补上实际 tag 日期与发布结论。

*说明：* molnex 仓与 sbatch 脚本不在本仓路径下，故不列入 **Files to create or modify**（该节只列本仓已核实路径）；它们由 Tasks 与 acceptance 的 `pass_when` 用产物名指定。

### Reuse decision

- `reuse` `.github/workflows/publish.yml` 现有五 job 流水线——不新建发布脚本（`.claude/notes/release.md`：仓内只有 `scripts/fetch-test-data.sh`，无发布辅助脚本，这是刻意的）。
- `reuse` `.claude/notes/release.md` 的手工 checklist 作为发布流程唯一依据（无 pin-parity 自动化）。
- `reuse` 01 已写入的 0.14.0 release.md 段落，本规范只补事实。
- `new` — none.

## Files to create or modify

- `.claude/notes/release.md`
- `.claude/specs/INDEX.md`
- `regressions/release-0-14-08-ship-molrs.md` (new)

## Tasks

- [ ] Verify the full local gate on the merge candidate (`prek run --all-files --hook-stage pre-push`, rust lib + doc tests, molrs-python tox)
- [ ] Merge the 0.14 work into `master` and tag `v0.14.0`
- [ ] Verify all five `publish.yml` jobs are green and the artifacts are live on crates.io, npm and PyPI
- [ ] Replace the `.dev1` wheel in molnex `.wheels-gh200/` with the released `molcrafts_molrs-0.14.0-*.whl` and delete the prerelease file
- [ ] Rebuild the aarch64 venv via `gh200_bootstrap.sbatch` against the released wheel
- [ ] Record the tag date and publish outcome in `.claude/notes/release.md` and mark the chain state in `.claude/specs/INDEX.md`
- [ ] Add regression example `regressions/release-0-14-08-ship-molrs.md` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

- 门验证（happy path）：本地完整门在合并候选上全绿，`cargo test --doc` 一并绿（doctest 不在 `--lib` 覆盖内，是公开 API 的编译证明）。
- 发布验证：三个 registry 各自可解析 `0.14.0`——`pip download molcrafts-molrs==0.14.0`（纯下载，不执行第三方科学软件）、`cargo add molcrafts-molrs@0.14.0 --dry-run`、npm view。
- 下游验证：`.wheels-gh200/` 目录里 `0.14.0` wheel 存在且 `.dev1` 不存在；aarch64 venv 里 `import molrs` 成功。
- 回归样例 `regressions/release-0-14-08-ship-molrs.md`：checklist，写死 tag 名、五个 job 名、wheel 文件名模式、以及"molpy 不得早于此步"的顺序约束。
- 域验证：不适用。

## Open questions (maintainer ruling required)

- 无。发布顺序与流程为既有铁律。

## Out of scope

- 任何代码改动（02–07 已完成；此处发现问题应停下并回到相应规范，不在发布步里改代码）
- molpy 侧全部工作（09–12）
