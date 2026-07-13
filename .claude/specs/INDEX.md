# molrs — Spec Index

One row per spec produced by `/mol:spec`. Newest on top.

| Date | Slug | Status | Owner crate(s) | Summary |
|---|---|---|---|---|
| 2026-07-12 | chem-perceive-12-cxx-bridge | approved | molrs-cxxapi | cxx bridge 改成返回 `Result<Vec<f64>>`——现在它声明 `-> Vec<f64>` 且函数体 `.expect()`，于是缺 BCC 参数这类**用户化学错误直接 abort 进程**而非抛可捕获的 C++ 异常；同时去掉 normalize 参数、加 parameter-set 选择器打通 ABCG2（现全文 0 命中）。跨仓：需 Atomiverse 配套改。链 12/13。 |
| 2026-07-12 | chem-perceive-13-python-bind | approved | molrs-python, molcrafts-molrs | 把 `Perceive`/`AtdTypifier`/`BccModel`/`MullikenModel`/`GasteigerModel` 暴露到 molrs-python，迁 `molrs::chem`→`molrs::perceive` 并删掉 01 的 compat alias。**Python 首次可达原生 AM1-BCC**（今天 molpy 只有 `antechamber -c bcc`），这是与 antechamber 对账的前提。链 13/13。 |
| 2026-07-12 | chem-perceive-14-all-tables | approved | molcrafts-molrs | 收尾「所有参数表 .rs 化」：mmff94/mmff94s/oplsaa + gen3d 的两个 fragment 库也转成 typed Rust 表，删空 `molrs/data/` 与全部 `include_str!`。**存在理由本身是纠错**——早前「编译时间会爆炸」的排除理由已被实测推翻（15,474 行 = +1071 KB / 0.37 s；而这些数据本来就以原始文本形式躺在二进制里，共 3974 KB）。纯表示层变更，数值零改动。链 14/14。 |
| 2026-07-11 | graph-sink-01-extract | done | molcrafts-molrs | Induced subgraph + multi-source `extract_ball` / leaf `extract_subgraph` (O(ball) regenerate path). Engine primitives for molpy region extract. Chain graph-sink 1/4. |
| 2026-07-11 | graph-sink-02-copy-merge | done | molcrafts-molrs | Lock copy = handle-preserving Clone; `merge` returns old→new node map; no identity-merge. Chain graph-sink 2/4. |
| 2026-07-11 | graph-sink-03-python-bind | done | molrs-python | PyO3: extract result type, merge→dict[int,int], copy contract tests. Blocks molpy wire-up. Chain graph-sink 3/4. |
| 2026-07-11 | graph-sink-04-hydrogens-coords | done | molcrafts-molrs | `add_hydrogens` places X–H xyz when heavy has coords (port molpy capping geometry). Chain graph-sink 4/4. |
| 2026-07-06 | net-streaming | partial (networking deferred) | molcrafts-molrs | Serialization foundation SHIPPED 0.7.0 as the `serde` + `stream` features (Frame/Block/Column/SimBox serde + MessagePack/JSON `frame_to_bytes`, WASM-clean). WebSocket networking + bidirectional control (`net` feature: tokio runtime, `FrameServer`, `ControlCommand`, crossbeam bridge) DEFERRED to a later release. See net-streaming.md STATUS. |
| 2026-07-05 | region-support-01-graph-hash | in-flight | molcrafts-molrs, molrs-python | Isomorphism-invariant structural graph hash (WL/Morgan) + canonical node order + `is_isomorphic` on `MolGraph` (AA+CG), exposed to Python. The dedup key for molpy incremental-typification (`AffectedRegion` hashes by it → identical polymer junctions retype once). Greenfield; reuses topo/neighbor kernels. See molpy notes/incremental-typification-design.md. |
| 2026-07-05 | region-support-02-reaction-touched | in-flight | molcrafts-molrs, molrs-python | `Reaction.apply` returns the touched atom handles (bond-forming endpoints + added atoms + deleted-atom surviving neighbors + prop-set atoms) so molpy can extract the retype-safe region. Small, on reaction-smarts-02. Return type None→list[int]. |
| 2026-07-05 | reaction-smarts-01-python-matcher | in-flight | molcrafts-molrs, molrs-python | Expose the existing atom-map-aware SMARTS engine (`molrs::SmartsPattern`, `core/chem/smarts/`, already parses `[C:1]`) to Python as `SmartsPattern` with map-keyed matches; add `PyAtomistic` graph-edit conveniences (`remove_atom`/`remove_bond`/`set_bond_order`/`copy`). Pure binding, no algorithm change. Consumed by molpy `crosslink-*`. |
| 2026-07-05 | reaction-smarts-02-smirks-applier | in-flight | molcrafts-molrs, molrs-python | Daylight reaction-SMARTS (SMIRKS) engine: parse `LHS>>RHS`, compile the atom-map diff to a `Transform` (Daylight SMIRKS semantics: pairwise maps preserved, unmapped LHS deleted, unmapped RHS added, bond diff→form/break/order), apply to one occurrence via existing core edits + `generate_topology`. Expose `Reaction` to Python. Greenfield, on top of 01. Permissive reaction SMARTS (no strict mode). Depends on 01. |

<!--
Status values:
  draft      — spec written, not yet implemented
  in-flight  — /mol:impl started against this spec
  shipped    — merged to master
  superseded — replaced by a later spec (link it in Summary)
-->
