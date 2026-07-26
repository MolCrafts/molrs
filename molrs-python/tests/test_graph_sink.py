"""graph-sink-03 — Python bindings for extract / merge / copy contracts."""

from __future__ import annotations

import molrs


def _chain(n: int) -> tuple[molrs.Atomistic, list[int]]:
    mol = molrs.Atomistic()
    ids: list[int] = []
    for i in range(n):
        ids.append(mol.add_atom("C", float(i), 0.0, 0.0))
    for a, b in zip(ids, ids[1:]):
        mol.add_bond(a, b)
    return mol, ids


def test_copy_preserves_handles():
    mol, ids = _chain(2)
    h = ids[0]
    c = mol.copy()
    assert c.has_entity(h)
    assert c.get(h, "element") == mol.get(h, "element")


def test_merge_returns_map_and_empties_other():
    a, _ = _chain(2)
    b, bids = _chain(2)
    n_a = a.n_atoms
    n_b = b.n_atoms
    m = a.merge(b)
    assert isinstance(m, dict)
    assert len(m) == n_b
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in m.items())
    assert a.n_atoms == n_a + n_b
    assert b.n_atoms == 0
    for old in bids:
        assert old in m
        assert a.has_entity(m[old])


def test_extract_subgraph_maps_consistent():
    mol, ids = _chain(10)
    center = ids[5]
    result = mol.extract_subgraph([center], 2)
    assert isinstance(result, molrs.ExtractedSubgraph)
    assert isinstance(result.graph, molrs.Atomistic)
    assert set(result.node_map.keys()) == set(result.hops.keys())
    assert result.graph.n_atoms == len(result.node_map)
    for b in result.boundary:
        assert b in result.hops


def test_extract_regenerate_topology():
    mol, ids = _chain(5)
    # no angles on parent
    result = mol.extract_subgraph([ids[2]], 2, regenerate_topology=True)
    # full chain in ball; angles should appear
    assert result.graph.n_atoms == 5
    # angles kind should have relations if generate_topology ran
    assert "angles" in result.graph.kinds()
    assert result.graph.n_relations("angles") > 0


def test_induced_subgraph():
    mol, ids = _chain(4)
    sub, node_map = mol.induced_subgraph([ids[1], ids[2]])
    assert sub.n_atoms == 2
    assert len(node_map) == 2
    assert sub.n_relations("bonds") == 1
