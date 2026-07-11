---
slug: graph-sink-02-copy-merge
created: 2026-07-11
criteria:
  - id: ac-001
    summary: Clone preserves node and relation handles
    type: code
    pass_when: |
      After `let b = a.clone()` on an Atomistic with atoms, bonds, and generated angles,
      the sets of atom handles and bond/angle handles in a and b are equal; reading
      components via those handles on b matches a; mutating b does not change a.
    status: verified
    last_checked: 2026-07-11
    evidence: test_clone_preserves_handles + copy_preserves_handles
  - id: ac-002
    summary: merge returns complete old→new node map
    type: code
    pass_when: |
      A.merge(B) returns a map with exactly one entry per live node of B (pre-merge);
      every value is a live node in A post-merge.
    status: verified
    last_checked: 2026-07-11
    evidence: merge_returns_complete_node_map + test_merge_transfers_all_kinds
  - id: ac-003
    summary: merge remaps all relation kinds
    type: code
    pass_when: |
      B has bonds + angles; after A.merge(B), A contains remapped angles whose endpoints
      are map[old] for each old endpoint; n_bonds/n_angles increase by B's counts.
    status: verified
    last_checked: 2026-07-11
    evidence: test_merge_transfers_all_kinds
  - id: ac-004
    summary: CoarseGrain membership survives merge under remapped bead ids
    type: code
    pass_when: |
      B bead members [10, 20] (opaque); after A.merge(B), bead_members(map[old_bead]) == [10,20].
    status: verified
    last_checked: 2026-07-11
    evidence: CoarseGrain::merge transfers members map keyed by remapped bead
  - id: ac-005
    summary: no identity-merge API
    type: code
    pass_when: |
      grep -rn 'merge_identity|identity_preserving' molrs/src → zero hits.
      Public merge docs state handles are remapped.
    status: verified
    last_checked: 2026-07-11
  - id: ac-006
    summary: quality gate + in-tree callers compile
    type: runtime
    pass_when: |
      cargo fmt/clippy/test --all-features exit 0; workspace members that call merge compile.
    status: verified
    last_checked: 2026-07-11
out_of_scope:
  - Python bindings
  - molpy def_* / Entity identity
---

# Acceptance — graph-sink-02-copy-merge

Done means copy is handle-preserving by contract test, merge returns a usable node map, and
there is no second identity-preserving merge path in the engine.
