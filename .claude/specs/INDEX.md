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

## Live — shared dylib host

- [ffi-shared-dylib](ffi-shared-dylib.md) — molrs-ffi becomes the shared dylib host; one feature-union default across every native build graph (native wheel gains rayon) [approved]

## neighborlist chain — **done** 2026-08-10（4/4 closed & deleted，branch feat/neighborlist）

核心成果：`Neighbors` 表（`from_pairs`/`disp`/Option 列）→ `NeighborList` 引擎（`build`/`update`/`build_columns`/`for_each_pair`/`neighbors`，rayon 并行物化 N=100k 4.2×）→ compute 全量消费（require_* 统一缺列拒绝）→ Python/WASM 同名表面（默认 FULL；molvis 兼容别名保留）。公开面 Nb 缩写清零；重复自查询门删除（−303 行）。

沿途铁律修复：`filter_sann` SANN 判据（van Meel 2012，原式恒 no-op）、RDF release 全零 g(r)、Steinhardt parity 空转锚点 + 解析 golden、order kernel CrossQuery guard、3 处文档假属性。

遗留路由：linkcell cell-walk 双循环归一 + `check_points` 参数化 + `AabbQuery::cutoff` 惰性字段 → `/mol:refactor`；rustdoc 第三门未接线（notes 声称 stable 与事实不符，33 个先存链接错误）+ binder 两 crate 无 fmt/clippy 门（~28 处静默 lint/fmt 债）+ zensical docstring_style 失配 → `/mol:note` + `/mol:ci-sync`；molvis `SpatialNeighborQuery` 迁移 → 另仓 follow-up。
