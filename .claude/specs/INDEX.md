# molrs — Spec Index

Live specs only.

## release-0-14 chain (joint molrs + molpy release; molrs first) — code on `dev` 2026-08-25

08 / 12 (tag, publish, master merge) are **not** executed — stay on `dev`, no tag.

执行顺序按 `depends_on`，不按编号：01 → 02 → 03 → **14** → 04 → 05 → **13** → 06 → 07 → 08 → 09 → 10 → 11 → 12。

- [release-0-14-01-baseline](release-0-14-01-baseline.md) — merge origin/master, unify 6+2 version strings to 0.14.0, backfill release.md [approved]
- [release-0-14-02-units-purge](release-0-14-02-units-purge.md) — unit presets promoted to core::units (UnitPreset, LAMMPS-free naming), three vocabularies unified, zero unit conversion inside MD, MaxwellBoltzmann::new(kbt, seed) [approved]
- [release-0-14-03-potential-protocol](release-0-14-03-potential-protocol.md) — Potential as runtime_checkable Protocol, duck-typed dispatch with concrete-arms-first bite-proof, coordinates-only Python contract, .pyi Unions collapsed [approved]
- [release-0-14-04-md-driver](release-0-14-04-md-driver.md) — MD(dtype=) numpy-style precision entry, single MDState, seam-only tests + driver NVE in regressions/, 7 pre-existing failures green [approved]
- [release-0-14-05-compute-protocol](release-0-14-05-compute-protocol.md) — Compute as runtime_checkable Protocol (compute() only), 45 kernels satisfy unmodified [approved]
- [release-0-14-06-docs-migration](release-0-14-06-docs-migration.md) — 0.13→0.14 migration guide, md user guide, fs as sole analysis time unit, molpy spelling in all user examples [approved]
- [release-0-14-07-surface-hygiene](release-0-14-07-surface-hygiene.md) — README/examples repair + CI smoke, error messages, wasm NeighborQuery deferred to 0.15 with consumption evidence, rustdoc f32 lies, gitignore, stale blueprint [approved]
- [release-0-14-08-ship-molrs](release-0-14-08-ship-molrs.md) — tag v0.14.0 on master, publish to three registries, replace molnex .dev1 wheel, rebuild aarch64 venv [approved]
- [release-0-14-09-molpy-rebase](release-0-14-09-molpy-rebase.md) — re-branch molpy from upstream/master v0.13.1, cherry-pick 8 dev commits, converge _key_str, pin >=0.14.0,<0.15 [approved]
- [release-0-14-10-molpy-mirror](release-0-14-10-molpy-mirror.md) — sink-to-molrs frozen with a capability criterion: seven duplicated formats + Box geometry sink in 0.14 with per-format bit-identical parity, the remainder renamed molpy-native extensions, permanent anti-duplication gate [approved]
- [release-0-14-11-molpy-docs](release-0-14-11-molpy-docs.md) — typifier spelling sweep (predicate-based), user-visible molrs spelling zeroed, bilingual molpy migration guide [approved]
- [release-0-14-12-joint-smoke](release-0-14-12-joint-smoke.md) — full-import + molnex chain smoke on the released wheel, then tag molpy 0.14.0 [approved]
- [release-0-14-13-frame-store-naming](release-0-14-13-frame-store-naming.md) — public APIs named by object, not backend: molrs.MolRec → molrs.Record, read_zarr/write_zarr → read/write, cxxapi write_frame / read_first_frame (Atomiverse consumer on record); record.rs untouched [approved]
- [release-0-14-14-pair-kernel-merge](release-0-14-14-pair-kernel-merge.md) — one LJ pair kernel (pure API unification, bit-identical), per-step pair dataset passed through instead of the set_pairs snapshot (no MIC in potentials, one dataset shared), kspace name off the ForceField surface with PME as pair/coul/long/pme and the module kept as the FFT compilation unit [approved]

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

## neighborlist chain — **done** 2026-08-10（4/4 closed & deleted，branch feat/neighborlist）

核心成果：`Neighbors` 表（`from_pairs`/`disp`/Option 列）→ `NeighborList` 引擎（`build`/`update`/`build_columns`/`for_each_pair`/`neighbors`，rayon 并行物化 N=100k 4.2×）→ compute 全量消费（require_* 统一缺列拒绝）→ Python/WASM 同名表面（默认 FULL；molvis 兼容别名保留）。公开面 Nb 缩写清零；重复自查询门删除（−303 行）。

沿途铁律修复：`filter_sann` SANN 判据（van Meel 2012，原式恒 no-op）、RDF release 全零 g(r)、Steinhardt parity 空转锚点 + 解析 golden、order kernel CrossQuery guard、3 处文档假属性。

遗留路由：linkcell cell-walk 双循环归一 + `check_points` 参数化 + `AabbQuery::cutoff` 惰性字段 → `/mol:refactor`；rustdoc 第三门未接线（notes 声称 stable 与事实不符，33 个先存链接错误）+ binder 两 crate 无 fmt/clippy 门（~28 处静默 lint/fmt 债）+ zensical docstring_style 失配 → `/mol:note` + `/mol:ci-sync`；molvis `SpatialNeighborQuery` 迁移 → 另仓 follow-up。
