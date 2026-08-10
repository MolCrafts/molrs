---
spec: neighborlist-03-compute
created: 2026-08-10
criteria:
  - id: ac-001
    summary: compute 公共 Args 使用 Neighbors 而非旧结果 NeighborList
    type: code
    pass_when: |
      molrs compute 模块中面向邻居表的 Args 类型为 Neighbors（或 &Neighbors / Vec<Neighbors>），
      无对已删除结果类型名 NeighborList 的依赖（总包 NeighborList 仅用于构建表时除外）。
    status: verified
    last_checked: 2026-08-10
    evidence: "All Compute::Args are &Vec<Neighbors>; NeighborList appears only as the engine in construction paths (test_support, rdf self paths); compile + grep"
  - id: ac-002
    summary: order 内核缺 disp 返回 BadShape
    type: runtime
    pass_when: |
      Steinhardt 与 Hexatic 在 INDICES_ONLY 的 Neighbors（n_pairs>0）上 compute
      返回 Err(ComputeError::BadShape{..})，不 panic。
    status: verified
    last_checked: 2026-08-10
    evidence: "steinhardt_indices_only_neighbors_is_bad_shape + hexatic_indices_only_neighbors_is_bad_shape green; require_* helpers, message names column + pair count"
  - id: ac-003
    summary: Steinhardt 半壳 + parity 行为回归
    type: scientific
    pass_when: |
      既有 steinhardt 单测（octahedron q6、parity 两粒子、PBC）在 FULL Neighbors 下仍通过。
    status: pending
  - id: ac-004
    summary: RDF Self 仍含 half-list 因子 2
    type: scientific
    pass_when: |
      RDF SelfQuery 归一化路径仍对直方图使用 factor 2（判据为
      matches!(mode, SelfQuery{..})）；CrossQuery 不使用。既有 rdf 单测通过。
    status: pending
  - id: ac-005
    summary: full,filesystem feature lib 测试通过
    type: runtime
    pass_when: |
      cargo test -p molcrafts-molrs --lib --features full,filesystem 退出码 0（与 CI gate 一致）。
    status: verified
    last_checked: 2026-08-10
    evidence: "1547 passed (also 1547 under --no-default-features serial fallback); 66 doctests; fmt/clippy clean both crates"
out_of_scope:
  - Python/WASM
  - skin
---

# Acceptance — neighborlist-03-compute

compute 全量消费 `Neighbors`；order 缺列可预测失败；科学回归保持。
