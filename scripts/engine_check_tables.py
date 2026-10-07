"""The tables every engine check reads: molrs's energy TSVs, LAMMPS's thermo
line, element-from-mass and the molecules of a bond list.

Shared by scripts/ff_engine_codec_check.py, scripts/ff_equivalence_check.py
and the LAMMPS steps of scripts/*_check.sh (which import it with
``PYTHONPATH=scripts``). The LAMMPS log is parsed by molrs's own reader
(``molrs.io.read_lammps_log``), and every engine constant comes from
``molrs.core.constants``. molrs is imported inside the functions that need it,
so AmberTools' python (no molrs) can still load this module for the sander
step of ff_equivalence_check.

As a command, ``python scripts/engine_check_tables.py thermo LOG`` prints the
thermo line of LOG's last run as ``key=value`` words.
"""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path


def read_energy_tsv(path: Path) -> dict[tuple[str, int, str, str], float]:
    """A ``case  config  engine  term  value`` table (``#`` lines are comments)
    as ``{(case, config, engine, term): value}``."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        case, k, engine, term, value = line.split("\t")
        out[(case, int(k), engine, term)] = float(value)
    return out


def lammps_thermo(path: Path | str) -> dict[str, float]:
    """The first thermo row of the last run of the LAMMPS log at ``path``
    (a ``run 0`` single point), as ``{column: value}``."""
    import molrs

    log = molrs.io.read_lammps_log(path)
    runs = [r.thermo for r in log.runs if r.thermo is not None and r.thermo.n_rows]
    if not runs:
        raise SystemExit(f"{path}: no thermo output")
    thermo = runs[-1]
    return {c: float(thermo[c][0]) for c in thermo.columns}


@cache
def _element_masses() -> list[tuple[str, float]]:
    from molrs.core import Element

    out, z = [], 1
    while True:
        try:
            e = Element(z)
        except (ValueError, KeyError):
            return out
        out.append((e.symbol, e.mass))
        z += 1


def element_of_mass(mass: float) -> str:
    """The element whose standard mass is nearest ``mass``, for an atom type
    that names no element."""
    return min(_element_masses(), key=lambda e: abs(e[1] - mass))[0]


def molecule_ids(n: int, bonds) -> list[int]:
    """Each of ``n`` atoms' molecule index (the connected components of
    ``bonds``, numbered in order of their first atom)."""
    root = list(range(n))

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    for i, j in bonds:
        a, b = find(i), find(j)
        root[max(a, b)] = min(a, b)
    ids, out = {}, []
    for a in range(n):
        out.append(ids.setdefault(find(a), len(ids)))
    return out


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[0] != "thermo":
        raise SystemExit("usage: engine_check_tables.py thermo LOG")
    print(" ".join(f"{k}={v!r}" for k, v in lammps_thermo(argv[1]).items()))


if __name__ == "__main__":
    main(sys.argv[1:])
