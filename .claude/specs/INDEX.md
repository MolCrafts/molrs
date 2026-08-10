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
| [neighborlist-01-types](neighborlist-01-types.md) | approved — `Neighbors` + `from_pairs` + optional `dist_sq`/`disp`；QueryMode 带计数 |
| [neighborlist-02-engine](neighborlist-02-engine.md) | approved — `NeighborList` 总包 `build`/`update`/`for_each_pair`/`neighbors`；删 `NbList`，禁 Nb 缩写 |
| [neighborlist-03-compute](neighborlist-03-compute.md) | approved — compute 消费 `Neighbors`；order 缺 disp → BadShape |
| [neighborlist-04-binders](neighborlist-04-binders.md) | approved — Python/WASM 对齐；默认 FULL 物化；cross 出路显式 |
