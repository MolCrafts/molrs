#!/usr/bin/env python3
"""The engines' side of molrs's engine-codec check
(molrs/src/ff/engine_codec_check.rs; driven by scripts/ff_engine_codec_check.sh).

``collect DIR [--pin FILE]``: reads LAMMPS's logs of the LAMMPS cases (the
shell script ran ``lmp``), prices the OpenMM cases' molrs-written XML on the
Reference platform (``NoCutoff``, one force group per term, residue templates
built here from ``system.json``), prints every engine number beside molrs's
(``DIR/molrs.tsv``) with the relative difference, and writes the table
(``DIR/engines.tsv``, or FILE) the Rust test pins. Each row is ``case config
engine term value``; a LAMMPS case's numbers are in its files' units, an
OpenMM case's in kcal/mol.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

KJ = 4.184
ELEMENTS = [("H", 1.008), ("C", 12.011), ("N", 14.007), ("O", 15.999), ("S", 32.06)]


def element_of_mass(m: float) -> str:
    return min(ELEMENTS, key=lambda e: abs(e[1] - m))[0]


def read_tsv(path: Path):
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        case, k, engine, term, value = line.split("\t")
        out[(case, int(k), engine, term)] = float(value)
    return out


def lammps(sub: Path, k: int):
    lines = (sub / f"log.{k}").read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if l.split()[:1] == ["Step"])
    v = dict(zip(lines[i].split(), map(float, lines[i + 1].split())))
    names = {"E_bond": "bond", "E_angle": "angle", "E_vdwl": "vdw", "PotEng": "total"}
    return {t: v[k] for k, t in names.items() if k in v}


def components(n, bonds):
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


def openmm(sub: Path, sysj):
    import openmm as mm
    import openmm.unit as u
    from openmm import app

    root = ET.parse(sub / "ff.xml").getroot()
    element = {}
    for t in root.iter("Type"):
        element[t.get("name")] = t.get("element") or element_of_mass(float(t.get("mass")))
        t.set("element", element[t.get("name")])
    types, bonds = sysj["types"], sysj["bonds"]
    n = len(types)
    mol = components(n, bonds)
    residues = ET.SubElement(root, "Residues")
    top = app.Topology()
    chain = top.addChain()
    made = [None] * n
    # Topology order: molecule by molecule (residue templates are whole
    # molecules), so a position goes to the atom made from it.
    order = []
    for r in sorted(set(mol)):
        tmpl = ET.SubElement(residues, "Residue", name=f"R{r}")
        res = top.addResidue(f"R{r}", chain)
        for a in (a for a in range(n) if mol[a] == r):
            ET.SubElement(tmpl, "Atom", name=f"A{a}", type=types[a], charge="0")
            made[a] = top.addAtom(f"A{a}", app.Element.getBySymbol(element[types[a]]), res)
            order.append(a)
        for i, j in bonds:
            if mol[i] == r:
                ET.SubElement(tmpl, "Bond", atomName1=f"A{i}", atomName2=f"A{j}")
    for i, j in bonds:
        top.addBond(made[i], made[j])
    path = sub / "ff+residues.xml"
    ET.ElementTree(root).write(path)
    system = app.ForceField(str(path)).createSystem(
        top, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    term = {
        "CustomBondForce": "bond",
        "CustomAngleForce": "angle",
        "CustomTorsionForce": "improper",
        "CustomCompoundBondForce": "urey_bradley",
        "CustomNonbondedForce": "vdw",
    }
    groups = {}
    for i, f in enumerate(system.getForces()):
        name = type(f).__name__
        if name == "CMMotionRemover":
            continue
        f.setForceGroup(i)
        groups[i] = term[name]
    ctx = mm.Context(system, mm.VerletIntegrator(1.0), mm.Platform.getPlatformByName("Reference"))
    out = []
    for x in sysj["configs"]:
        ctx.setPositions([mm.Vec3(x[3 * a], x[3 * a + 1], x[3 * a + 2]) * 0.1 for a in order])
        terms = {}
        for g, name in groups.items():
            e = ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy()
            terms[name] = terms.get(name, 0.0) + e.value_in_unit(u.kilojoule_per_mole) / KJ
        e = ctx.getState(getEnergy=True).getPotentialEnergy()
        terms["total"] = e.value_in_unit(u.kilojoule_per_mole) / KJ
        out.append(terms)
    return out


def collect(d: Path, pin: Path | None):
    molrs = read_tsv(d / "molrs.tsv")
    rows = []
    for case, k, engine, term in sorted({key for key in molrs}):
        rows.append((case, k, engine, term))
    engines = {}
    for case in sorted({r[0] for r in rows}):
        sub = d / case
        engine = next(r[2] for r in rows if r[0] == case)
        configs = sorted({r[1] for r in rows if r[0] == case})
        if engine == "lammps":
            for k in configs:
                for t, v in lammps(sub, k).items():
                    engines[(case, k, engine, t)] = v
        else:
            sysj = json.loads((sub / "system.json").read_text())
            for k, terms in enumerate(openmm(sub, sysj)):
                for t, v in terms.items():
                    engines[(case, k, engine, t)] = v
    worst = 0.0
    lines = ["# case\tconfig\tengine\tterm\tvalue (scripts/ff_engine_codec_check.sh --pin)"]
    # A term the engine prints and molrs has no style of is zero, or the
    # check fails here.
    extra = {k: v for k, v in engines.items() if k not in molrs and v != 0.0}
    if extra:
        raise SystemExit(f"engine terms molrs has no style for: {extra}")
    for key in sorted(molrs):
        m, e = molrs.get(key), engines.get(key)
        rel = abs(e - m) / max(abs(m), 1.0) if m is not None and e is not None else float("nan")
        worst = max(worst, rel) if rel == rel else worst
        print(f"{key[0]:8s} {key[1]} {key[2]:7s} {key[3]:13s} engine {e!r:>24} molrs {m!r:>24} rel {rel:.1e}")
        if e is not None:
            lines.append("\t".join([key[0], str(key[1]), key[2], key[3], repr(e)]))
    print(f"worst relative difference: {worst:.1e}")
    out = pin or d / "engines.tsv"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("dir", type=Path)
    c.add_argument("--pin", type=Path)
    a = ap.parse_args()
    collect(a.dir, a.pin)


if __name__ == "__main__":
    main()
