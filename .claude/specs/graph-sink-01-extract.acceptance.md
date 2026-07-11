---
slug: graph-sink-01-extract
created: 2026-07-11
criteria:
  - id: ac-001
    summary: induced_subgraph remaps nodes and only induced relations
    type: code
    pass_when: |
      Given a 4-atom path with angles perceived, induced_subgraph on the middle two atoms
      yields 2 nodes, 1 bond, 0 angles (endpoints incomplete). node_map has exactly 2 entries;
      every new handle is live in the subgraph.
    status: verified
    last_checked: 2026-07-11
    evidence: core::system::extract::tests::induced_subgraph_path4_middle_two
  - id: ac-002
    summary: extract_ball multi-source hops and boundary are correct
    type: code
    pass_when: |
      Linear chain n=10 (0—1—…—9), centers={2,7}, radius=1:
      selected handles = {1,2,3,6,7,8}; hops[2]==0, hops[1]==1, hops[3]==1;
      boundary includes 1 and 3 and 6 and 8 (each has a neighbor outside the ball).
    status: verified
    last_checked: 2026-07-11
    evidence: core::system::extract::tests::extract_ball_linear_multi_center
  - id: ac-003
    summary: copy_higher_order=false path is O(ball) — no full higher-order scan required for correctness
    type: code
    pass_when: |
      Atomistic::extract_subgraph(..., regenerate_topology=true) on a ball produces angles/dihedrals
      that match Topology enumeration on the ball's bonds alone; parent graph's angle count does not
      affect result. With regenerate_topology=false and copy_higher_order=true, induced parent angles
      fully inside the ball are copied.
    status: verified
    last_checked: 2026-07-11
    evidence: atomistic_extract_regenerate_topology + atomistic_extract_copy_higher_order_angles
  - id: ac-004
    summary: stale center fails fast
    type: code
    pass_when: |
      extract_ball / extract_subgraph with a despawned center handle returns Err (not empty success).
    status: verified
    last_checked: 2026-07-11
    evidence: extract_ball_stale_center_errors + induced_subgraph_stale_fails
  - id: ac-005
    summary: CoarseGrain membership preserved on extract
    type: code
    pass_when: |
      Bead with set_bead_members([a,b]) extracted at radius 0 retains the same opaque member handles.
    status: verified
    last_checked: 2026-07-11
    evidence: coarsegrain_extract_preserves_membership
  - id: ac-006
    summary: quality gate
    type: runtime
    pass_when: |
      cargo fmt --all --check; cargo clippy --all-targets --all-features -- -D warnings;
      cargo test --all-features — all exit 0.
    status: verified
    last_checked: 2026-07-11
    evidence: clippy -D warnings green; lib tests 484 passed
out_of_scope:
  - Python bindings
  - molpy Entity views
  - copy/merge API changes
---

# Acceptance — graph-sink-01-extract

Done means radius-ball and induced-subgraph extraction live in molrs as pure structure facts,
with fail-fast stale handles and a regenerate path that never depends on scanning the parent's
higher-order relation tables for correctness.
