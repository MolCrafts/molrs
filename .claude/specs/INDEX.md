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

## Live — neighborlist chain (core → compute → binders)

| Slug | Status |
|---|---|
| neighborlist-01-types | **done** 2026-08-10（closed & deleted）— `Neighbors` + `from_pairs` + `disp`；沿途修复 `filter_sann` SANN 判据（van Meel 2012） |
| neighborlist-02-engine | **done** 2026-08-10（closed & deleted）— 引擎落地，`build_columns` SoA 入口，Nb 缩写清零；串行物化与双门降级显式路由 03 任务 9/10。**分支在 04 落地前不可 merge**（binder workspaces 红，04 已认领） |
| neighborlist-03-compute | **done** 2026-08-10（closed & deleted）— require_* 帮手统一缺列拒绝；rayon 并行物化接回（N=100k 4.2×）；重复自查询门删除（−303 行）；铁律修复:RDF release 全零 g(r)、parity 空转锚点、CrossQuery guard。遗留路由:linkcell cell-walk 双循环归一 → /mol:refactor;rustdoc 门未接线 → /mol:note |
| [neighborlist-04-binders](neighborlist-04-binders.md) | approved — Python/WASM 对齐；默认 FULL 物化；cross 出路显式 |
