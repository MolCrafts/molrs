# molrs — Spec Index

Live specs only.

## release-0-12 chain (molrs first) — done

| Slug | Status |
|---|---|
| [release-0-12-01-harness](release-0-12-01-harness.md) | done |
| [release-0-12-02-legacy-delete](release-0-12-02-legacy-delete.md) | done |
| [release-0-12-03-science-vacf-msd](release-0-12-03-science-vacf-msd.md) | done |
| [release-0-12-04-science-dielectric-zarr](release-0-12-04-science-dielectric-zarr.md) | done |
| [release-0-12-05-cxxapi-panic-free](release-0-12-05-cxxapi-panic-free.md) | done |
| [release-0-12-06-docs-surface](release-0-12-06-docs-surface.md) | done |

## Other live specs (not 0.12 ship gate)

| slug | Status |
|---|---|
| [cell-grid-api](cell-grid-api.md) | done |
| [chem-perceive-15-final-acceptance](chem-perceive-15-final-acceptance.md) | done|
| [core-generate-saw-path](core-generate-saw-path.md) | done |
| [net-streaming](net-streaming.md) | done |

## smiles-emit chain (molrs first; graph → SMILES/SMARTS write) — **done** 2026-08-05

All four stages verified (cargo lib tests + molrs-python `test_smiles_emit`). Public Python name: `molrs.io.write_smiles(mol, **flags)` / `SmilesIR.from_atomistic` / `write_smarts`. Specs deleted after close.

Downstream: molpy `smiles-emit-01-io-surface` after molrs tag ≥ emit surface.

## Live — angular distribution defaults

- [distribution-angular-default-range](distribution-angular-default-range.md) — PyO3 angular distributions default to a degrees range on a radians kernel; delegate to `over_natural_range` [approved]

## Live — link-mode-static-default chain (molrs first, molpack same landing)

- [link-mode-static-default-01-invert](link-mode-static-default-01-invert.md) — 静态成为零参数默认(两仓 config.toml 去 rustflags、七根 profile 补 lto、10 处 workflow env 删除、门脚本自带 flag、17 处悬空引用清理) [approved]
- [link-mode-static-default-02-ci-gate](link-mode-static-default-02-ci-gate.md) — CI 双形态门 `ci-link-form.yml`(static + dynamic 两 job)+ molpack `link-dynamic` job + MOLRS_GIT_REF 落地顺序前置条件 [approved]

## ffi-shared-dylib — **done** 2026-08-23(spec deleted after close)

molrs-ffi 成为共享 dylib 宿主:全原生消费者动态链接同一 libmolrs_ffi(sha256 跨 wheel 构建稳定)。当时的"本地动态默认 + CI/发布 env 钉静态"已被 link-mode-static-default-01 取代:静态成为零参数默认,动态形态降为命令行 `--config` opt-in(见 docs/interop.md)。单元同一性四轴(锁漂移/default 标签/兄弟加宽/成员风味)+ 第五轴(RUSTFLAGS régime)全部实测定案;unify 锚 pin:serde_json、foldhash、build-空间 syn+proc-macro2。门:verify-shared-dylib.sh(pre-push)+ abi_line 配对测试。遗留路由:molpy 0.14 对齐后 molpack tox 腿复跑(ac-009 偏差);CLI 二进制入圈等 molcrafts-molrs-ffi 上 crates.io;Atomiverse cdylib 翻转、wheel 内共享 .so 分发随发布火车。

## neighborlist chain — **done** 2026-08-10（4/4 closed & deleted，branch feat/neighborlist）

核心成果：`Neighbors` 表（`from_pairs`/`disp`/Option 列）→ `NeighborList` 引擎（`build`/`update`/`build_columns`/`for_each_pair`/`neighbors`，rayon 并行物化 N=100k 4.2×）→ compute 全量消费（require_* 统一缺列拒绝）→ Python/WASM 同名表面（默认 FULL；molvis 兼容别名保留）。公开面 Nb 缩写清零；重复自查询门删除（−303 行）。

沿途铁律修复：`filter_sann` SANN 判据（van Meel 2012，原式恒 no-op）、RDF release 全零 g(r)、Steinhardt parity 空转锚点 + 解析 golden、order kernel CrossQuery guard、3 处文档假属性。

遗留路由：linkcell cell-walk 双循环归一 + `check_points` 参数化 + `AabbQuery::cutoff` 惰性字段 → `/mol:refactor`；rustdoc 第三门未接线（notes 声称 stable 与事实不符，33 个先存链接错误）+ binder 两 crate 无 fmt/clippy 门（~28 处静默 lint/fmt 债）+ zensical docstring_style 失配 → `/mol:note` + `/mol:ci-sync`；molvis `SpatialNeighborQuery` 迁移 → 另仓 follow-up。
