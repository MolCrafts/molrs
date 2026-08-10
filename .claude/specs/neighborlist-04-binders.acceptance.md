---
spec: neighborlist-04-binders
created: 2026-08-10
criteria:
  - id: ac-001
    summary: Python 导出 NeighborList 总包与 Neighbors
    type: code
    pass_when: |
      molrs Python 包可 import NeighborList 与 Neighbors（或文档记载的等价公开路径）；
      NeighborList 具 build/update（或 Frame 友好包装）与 neighbors 物化入口。
    status: verified
    last_checked: 2026-08-10
    evidence: "test_neighborlist.py 37/37 green under tox (565 total): engine NeighborList(cutoff)/brute_force/build/update/neighbors; Neighbors table exported; hygiene test asserts both attrs"
  - id: ac-002
    summary: WASM 导出 NeighborList 与 Neighbors
    type: code
    pass_when: |
      @molcrafts/molrs（或 molrs-wasm pkg）导出 Wasm 绑定的 NeighborList 与 Neighbors；
      总包可 build 并 neighbors 得到结果对象。
    status: verified
    last_checked: 2026-08-10
    evidence: "pkg/molrs.d.ts exports both; wasm-pack test --node 28 passed; 30-check node smoke (build→neighbors→columns→Cluster) passed"
  - id: ac-003
    summary: binder 默认物化含 disp（FULL）
    type: runtime
    pass_when: |
      默认 neighbors()（无额外 lean 参数）得到的 Neighbors 在 n_pairs>0 时 disp 可读
      （非空列）；Steinhardt.compute 在默认路径上不因缺列失败。
    status: verified
    last_checked: 2026-08-10
    evidence: "py: default-columns tests + TestDefaultPathFeedsSteinhardt green; wasm: default FULL in engine + aliases, node smoke confirms columns readable, lean column reads undefined not zeros"
  - id: ac-004
    summary: 文档说明 half-shell 与 FULL=列
    type: docs
    pass_when: |
      Python guide 或 rustdoc/wasm 注释至少一处写明：self 半壳 i<j；
      FULL 指 dist_sq+disp 列，非双向存储；disp 为非单位 MIC 位移。
    status: verified
    verified_by: agent-auto
    last_checked: 2026-08-10
    evidence: "site-src/guides/neighbor-search.md: half-shell i<j (L67-68), FULL=columns not bidirectional (L65,73), disp unnormalized MIC (L92,169), freud name mapping section (L161-171); objective presence check"
  - id: ac-005
    summary: 无 NeighborSearch 导出，且无 Nb 缩写
    type: code
    pass_when: |
      Python/WASM 公开 API 无 NeighborSearch，亦无 NbList/NbListAlgo 缩写导出。
    status: verified
    last_checked: 2026-08-10
    evidence: "py TestNeighborApiHygiene (no LinkedCell/NeighborSearch/Nb*) green; wasm d.ts grep clean"
  - id: ac-006
    summary: cross-query 能力不静默消失
    type: code
    pass_when: |
      迁移前 binder 暴露的 cross-query 能力，迁移后仍有等价公开入口；
      或 guide/CHANGELOG 明确记录其移除与替代路径。
    status: verified
    last_checked: 2026-08-10
    evidence: "py NeighborQuery.query survives (TestNeighborQueryCross: is_self_query False, contains i>j pair); wasm LinkedCell.query remains the cross door; guide documents both"
out_of_scope:
  - molvis stage SpatialNeighborQuery 重构
  - skin
---

# Acceptance — neighborlist-04-binders

绑定层名称与默认 FULL 物化对齐核心；order 默认路径安全；cross 出路显式。
