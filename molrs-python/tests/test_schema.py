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
    assert molrs.keys.ATOM_TYPE_LABELS not in {spec.key for spec in molrs.schema.columns}


def test_relation_endpoints_reads_the_vocabulary_or_the_columns() -> None:
    assert molrs.schema.relation_endpoints("bonds", []) == ("atoms", ["atomi", "atomj"])
    assert molrs.schema.relation_endpoints("links", ["atomi", "atomj"]) == (
        "atoms",
        ["atomi", "atomj"],
    )
    assert molrs.schema.relation_endpoints("cell", []) is None
