"""FFI smoke for ``molrs.schema``: the block-name constants and the relation
lookup cross the seam as plain strings."""

from __future__ import annotations

import molrs


def test_block_names_are_the_rust_constants() -> None:
    assert molrs.schema.ATOMS == "atoms"
    assert molrs.schema.TOPOLOGY == (
        molrs.schema.BONDS,
        molrs.schema.ANGLES,
        molrs.schema.DIHEDRALS,
        molrs.schema.IMPROPERS,
    )
    assert molrs.schema.block(molrs.schema.PAIRS) is not None


def test_groups_and_meta_keys_are_projected() -> None:
    assert [str(key) for key in molrs.keys.IMAGES] == ["ix", "iy", "iz"]
    assert [str(key) for key in molrs.keys.AXIS] == ["axis_x", "axis_y", "axis_z"]
    assert molrs.keys.ATOM_TYPE_LABELS == "atom_type_labels"
    assert molrs.keys.UNITS == "units"
    assert molrs.keys.ATOM_TYPE_LABELS not in {
        spec.key for spec in molrs.schema.columns
    }


def test_relation_endpoints_reads_the_vocabulary_or_the_columns() -> None:
    rel = molrs.schema.relation_endpoints
    assert rel("bonds", []) == [("atomi", "atoms"), ("atomj", "atoms")]
    assert rel("links", ["atomi", "atomj"]) == [("atomi", "atoms"), ("atomj", "atoms")]
    assert rel("cell", []) == []
    assert rel("members", ["ibead", "atom"]) == [("ibead", "atoms")]
    assert rel("members", ["ibead", "atom"], {"atom": "/frame/atoms"}) == [
        ("ibead", "atoms"),
        ("atom", "/frame/atoms"),
    ]
    assert molrs.schema.block("members").declared_endpoints == ["atom"]


def test_the_topology_conventions_are_canonical() -> None:
    dtypes = {spec.key: spec.dtype for spec in molrs.schema.columns}
    assert dtypes["formal_charge"] == "i64"
    assert dtypes["chain"] == "string" and "chain_id" not in dtypes
    assert dtypes["fx"] == "float"
    assert [str(key) for key in molrs.keys.FORCES] == ["fx", "fy", "fz"]
    for name in ["constraints", "drudes", "members", "virtual_sites"]:
        assert molrs.schema.block(name) is not None
