---
slug: graph-sink-03-python-bind
created: 2026-07-11
criteria:
  - id: ac-001
    summary: Atomistic.merge returns dict[int,int] and empties other
    type: runtime
    pass_when: |
      In Python: map = a.merge(b); isinstance(map, dict); all keys/values int;
      b.n_atoms == 0 (or n_nodes == 0); a.n_atoms == n_a + n_b_before.
    status: verified
    last_checked: 2026-07-11
    evidence: tests/test_graph_sink.py::test_merge_returns_map_and_empties_other
  - id: ac-002
    summary: copy preserves handles in Python
    type: runtime
    pass_when: |
      h = mol.entities()[0]; c = mol.copy(); c.has_entity(h) is True;
      c.get(h, key) equals mol.get(h, key) for a set component.
    status: verified
    last_checked: 2026-07-11
    evidence: tests/test_graph_sink.py::test_copy_preserves_handles
  - id: ac-003
    summary: extract_subgraph returns graph + maps with consistent sizes
    type: runtime
    pass_when: |
      result = mol.extract_subgraph([center], 1); result.graph is Atomistic;
      set(result.node_map) == set(result.hops); len(result.boundary) >= 0;
      every boundary handle is in result.hops.
    status: verified
    last_checked: 2026-07-11
    evidence: tests/test_graph_sink.py::test_extract_subgraph_maps_consistent
  - id: ac-004
    summary: regenerate_topology flag works from Python
    type: runtime
    pass_when: |
      On a chain with no precomputed angles, extract_subgraph(..., regenerate_topology=True)
      yields n_angles > 0 on the ball when the ball has a 2-edge path.
    status: verified
    last_checked: 2026-07-11
    evidence: tests/test_graph_sink.py::test_extract_regenerate_topology
  - id: ac-005
    summary: stubs compile / match
    type: code
    pass_when: |
      molrs.pyi declares merge, extract_subgraph, induced_subgraph, ExtractedSubgraph (or equivalent).
    status: verified
    last_checked: 2026-07-11
  - id: ac-006
    summary: molrs-python tests green
    type: runtime
    pass_when: |
      pytest molrs-python/tests/test_graph_sink.py exit 0.
    status: verified
    last_checked: 2026-07-11
    evidence: 5 passed
out_of_scope:
  - molpy integration
  - PyPI publish
---

# Acceptance — graph-sink-03-python-bind

Done means molpy can call extract/merge/copy as pure `molrs` Python APIs with handle maps,
without importing molpy types.
