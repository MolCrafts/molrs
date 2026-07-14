# molrs — Spec Index

One row per spec produced by `/mol:spec`. Newest on top.

| Date | Slug | Status | Owner crate(s) | Summary |
|---|---|---|---|---|
| 2026-07-14 | chem-perceive-15-final-acceptance | approved | molcrafts-molrs, molrs-python | **整体验收**：这条链跑了 16 个 spec，每个只验证自己那一块，没有任何一个验证过整体。把五条"只此一份"的架构承诺（一个参数地点/一个感知层/一个插值 seam/一条 MMFF 路径/忽略 `tp` 的构造器不是 Style）变成自己不能豁免自己的门禁；端到端对着**外部** oracle（antechamber 37 分子 + RDKit MMFF 11 分子）跑通全链路；Python 与 Rust **逐位**一致。门禁全部是**反向**的（断言某物不存在）。铁律两条：**一道从没红过的门禁 = 没有门禁**；**验收里不许修任何东西**——它找到的每个缺陷都必须停下来另立 spec。 |
| 2026-07-12 | chem-perceive-14-all-tables | approved | molcrafts-molrs | 收尾参数表编译期化：mmff94/mmff94s/oplsaa 转成 typed Rust 表，删除全部 `include_str!`。从未被 conformer 使用的 Open Babel `gen3d` fragment 坐标库直接删除，不生成 Rust 表。链 14/14。 |
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
