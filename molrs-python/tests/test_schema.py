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


def test_relation_endpoints_reads_the_vocabulary_or_the_columns() -> None:
    assert molrs.schema.relation_endpoints("bonds", []) == ("atoms", ["atomi", "atomj"])
    assert molrs.schema.relation_endpoints("links", ["atomi", "atomj"]) == (
        "atoms",
        ["atomi", "atomj"],
    )
    assert molrs.schema.relation_endpoints("cell", []) is None
