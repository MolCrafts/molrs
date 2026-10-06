"""The fresh process of ``test_ff_ir_persist.py``, and the helpers both
processes share.

Run as a script, it registers nothing: it reads every record it is handed,
prices it over its frame, and prints what it found as JSON — floats by
``float.hex``, so the comparison is bit for bit.
"""

from __future__ import annotations

import json
import sys

import molrs
import numpy as np

XYZ = np.array(
    [[0.0, 0.0, 0.0], [1.52, 0.1, 0.05], [2.1, 1.45, -0.1], [3.55, 1.6, 0.6]]
)
ENDPOINTS = ("atomi", "atomj", "atomk", "atoml", "atomm")


def frame(block: str, rows: list[list[int]], types: list[str]) -> molrs.Frame:
    """Four atoms of type ``A`` and the terms ``rows`` (typed ``types``) in
    ``block``."""
    atoms = molrs.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, XYZ[:, d].copy())
    atoms.insert("type", ["A"] * 4)
    terms = molrs.Block()
    for i, key in enumerate(ENDPOINTS[: len(rows[0])]):
        terms.insert(key, np.array([r[i] for r in rows], dtype=np.uint32))
    terms.insert("type", list(types))
    out = molrs.Frame()
    out["atoms"] = atoms
    out[block] = terms
    return out


def price(
    ff: molrs.ff.ForceField, block: str, rows: list[list[int]], types: list[str]
) -> dict:
    """Energy and forces of ``ff`` over the frame, exactly, or the refusal."""
    f = frame(block, rows, types)
    try:
        e, forces = molrs.ff.PotentialCompiler(ff).compile(f).calc_energy_forces(f)
    except ValueError as err:
        return {"error": str(err)}
    return {"e": float(e).hex(), "f": [float(x).hex() for x in np.ravel(forces)]}


def exact(section: molrs.io.mrec.ForceFieldSection) -> dict:
    """The section as plain data: the document, and every column's shape and
    values."""
    tables = {}
    for name, block in section.tables.items():
        columns = {}
        for key in block:
            values = np.asarray(block[key])
            if values.dtype.kind == "f":
                cells = [float(x).hex() for x in values.ravel()]
            else:
                cells = [str(x) for x in values.ravel()]
            columns[key] = [list(values.shape), cells]
        tables[name] = columns
    return {"document": section.document, "tables": tables}


def main(cases: dict) -> dict:
    out = {}
    for name, (path, block, rows, types) in cases.items():
        ff = molrs.ff.ForceField.from_section(molrs.io.read_mrec_forcefield(path))
        out[name] = {
            "price": price(ff, block, rows, types),
            "section": exact(ff.to_section()),
            "styles": [[s.category, s.name] for s in ff.styles],
        }
    return out


if __name__ == "__main__":
    print(json.dumps(main(json.loads(sys.argv[1]))))
