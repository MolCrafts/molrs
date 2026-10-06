#!/usr/bin/env python3
"""The engines' side of molrs's cross-engine equivalence check
(molrs/src/ff/equivalence_check.rs; driven by scripts/ff_equivalence_check.sh).

The Rust test writes, per source, the files each engine reads and a
``system.json`` (atoms, bonds, the IR's impropers, the configurations in Å
and the force probes); LAMMPS and GROMACS are run by the shell script. This
script

- ``sander DIR``: prices each prmtop source at its configurations with
  pysander (AmberTools; ``igb = 0``, ``ntb = 0``, ``cut = 999``) and writes
  ``DIR/sander.tsv``;
- ``collect DIR [--pin FILE]``: prices the OpenMM inputs (the molrs-written
  XML with residue templates built here, and an OpenMM-native source's own
  XML) on the Reference platform, ``NoCutoff``, one force group per term;
  reads LAMMPS's logs and force dumps and GROMACS's .edr and .trr; merges
  sander's; prints every engine beside molrs (``DIR/molrs.tsv``) and writes
  the table (``DIR/engines.tsv``, or FILE) the Rust test pins.

Every energy is kcal/mol, every force kcal/(mol·Å). Each row is
``source config engine term value``; the terms are bond, angle (with
Urey-Bradley), dihedral, improper, cmap, vdw, coul (each with its 1-4
pairs), total, and the force fingerprint fdotv = ΣF·v, fnorm2 = Σ|F|².
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

KJ = 4.184
TERMS = ["bond", "angle", "dihedral", "improper", "cmap", "vdw", "coul", "total"]
REPO = Path(__file__).resolve().parents[1]

# Standard masses, for an atom type that names no element.
ELEMENTS = [
    ("H", 1.008),
    ("C", 12.011),
    ("N", 14.007),
    ("O", 15.999),
    ("F", 18.998),
    ("Na", 22.990),
    ("Mg", 24.305),
    ("P", 30.974),
    ("S", 32.06),
    ("Cl", 35.45),
    ("K", 39.098),
    ("Ca", 40.078),
    ("Br", 79.904),
    ("I", 126.90),
]


def element_of_mass(m: float) -> str:
    return min(ELEMENTS, key=lambda e: abs(e[1] - m))[0]


def sources(d: Path):
    for sub in sorted(d.iterdir()):
        if (sub / "system.json").exists():
            yield sub, json.loads((sub / "system.json").read_text())


def fingerprint(f, v):
    return sum(a * b for a, b in zip(f, v)), sum(a * a for a in f)


def rows_of(name, k, engine, terms, forces, probe):
    out = [(name, k, engine, t, terms.get(t, 0.0)) for t in TERMS]
    dot, norm2 = fingerprint(forces, probe)
    out += [(name, k, engine, "fdotv", dot), (name, k, engine, "fnorm2", norm2)]
    return out


def write_tsv(path: Path, rows):
    with open(path, "w") as f:
        f.write(
            "# source\tconfig\tengine\tterm\tvalue (kcal/mol; fdotv, fnorm2: kcal/(mol·Å), "
            "(kcal/(mol·Å))²) — scripts/ff_equivalence_check.sh --pin\n"
        )
        f.writelines(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{float(r[4])!r}\n" for r in rows)


def read_tsv(path: Path):
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        s, k, e, t, v = line.split("\t")
        out[(s, int(k), e, t)] = float(v)
    return out


# ── sander ────────────────────────────────────────────────────────────────


def run_sander(d: Path):
    import numpy as np
    import sander

    rows = []
    for sub, sysj in sources(d):
        if sysj["native"] != "sander":
            continue
        parm = str(REPO / sysj["native_file"])
        inp = sander.gas_input(0)
        inp.cut = 999.0
        with sander.setup(parm, np.array(sysj["configs"][0]), None, inp):
            for k, x in enumerate(sysj["configs"]):
                sander.set_positions(np.array(x))
                e, f = sander.energy_forces()
                terms = {
                    "bond": e.bond,
                    "angle": e.angle + e.angle_ub,
                    "dihedral": e.dihedral,
                    "improper": e.imp,
                    "cmap": e.cmap,
                    "vdw": e.vdw + e.vdw_14,
                    "coul": e.elec + e.elec_14,
                    "total": e.tot,
                }
                rows += rows_of(
                    sysj["source"], k, "native", terms, list(f), sysj["probes"][k]
                )
    write_tsv(d / "sander.tsv", rows)


# ── OpenMM ────────────────────────────────────────────────────────────────


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


def openmm_written(sub: Path, sysj):
    """The molrs-written XML with one residue template per molecule (atoms
    named A<i>, charges per atom), and the topology it matches."""
    from openmm import app

    root = ET.parse(sub / "openmm" / "ff.xml").getroot()
    element = {}
    for t in root.iter("Type"):
        element[t.get("name")] = t.get("element") or element_of_mass(
            float(t.get("mass"))
        )
    # OpenMM orders an AMBER improper's outer atoms by element: every type
    # names its element.
    for t in root.iter("Type"):
        t.set("element", element[t.get("name")])
    n = len(sysj["types"])
    mol = components(n, sysj["bonds"])
    residues = ET.SubElement(root, "Residues")
    top = app.Topology()
    chain = top.addChain()
    made, res_of = [], {}
    for r in sorted(set(mol)):
        members = [a for a in range(n) if mol[a] == r]
        tmpl = ET.SubElement(residues, "Residue", name=f"R{r}")
        res = top.addResidue(f"R{r}", chain)
        for a in members:
            ET.SubElement(
                tmpl,
                "Atom",
                name=f"A{a}",
                type=sysj["types"][a],
                charge=repr(sysj["charges"][a]),
            )
            res_of[a] = res
        for i, j in sysj["bonds"]:
            if mol[i] == r:
                ET.SubElement(tmpl, "Bond", atomName1=f"A{i}", atomName2=f"A{j}")
    for a in range(n):
        made.append(
            top.addAtom(
                f"A{a}", app.Element.getBySymbol(element[sysj["types"][a]]), res_of[a]
            )
        )
    for i, j in sysj["bonds"]:
        top.addBond(made[i], made[j])
    path = sub / "openmm" / "ff+residues.xml"
    ET.ElementTree(root).write(path)
    return path, top, root.get("combining_rule") == "geometric"


def openmm_native(sysj):
    """An OpenMM-native source's own XML and the topology of its one residue
    template (atom order = the source's)."""
    from openmm import app

    path = REPO / sysj["native_file"]
    root = ET.parse(path).getroot()
    element = {t.get("name"): t.get("element") for t in root.iter("Type")}
    tmpl = root.find("Residues").find("Residue")
    top = app.Topology()
    res = top.addResidue(tmpl.get("name"), top.addChain())
    made = [
        top.addAtom(a.get("name"), app.Element.getBySymbol(element[a.get("type")]), res)
        for a in tmpl.findall("Atom")
    ]
    for i, j in sysj["bonds"]:
        top.addBond(made[i], made[j])
    return path, top, root.get("combining_rule") == "geometric"


def split_system(system, bonds, impropers, foyer_geometric):
    """Every term family in its own force group; the groups."""
    import openmm as mm

    forces = [copy.deepcopy(f) for f in system.getForces()]
    while system.getNumForces():
        system.removeForce(0)
    groups, out = {}, []

    def add(name, force):
        force.setForceGroup(len(out))
        groups[len(out)] = name
        out.append(force)

    for f in forces:
        if isinstance(f, mm.HarmonicBondForce):
            real, ub = mm.HarmonicBondForce(), mm.HarmonicBondForce()
            for i in range(f.getNumBonds()):
                a, b, r0, k = f.getBondParameters(i)
                (real if frozenset((a, b)) in bonds else ub).addBond(a, b, r0, k)
            add("bond", real)
            if ub.getNumBonds():
                add("angle", ub)
        elif isinstance(f, mm.HarmonicAngleForce):
            add("angle", f)
        elif isinstance(f, mm.PeriodicTorsionForce):
            prop, imp = mm.PeriodicTorsionForce(), mm.PeriodicTorsionForce()
            for i in range(f.getNumTorsions()):
                *t, n, ph, k = f.getTorsionParameters(i)
                (imp if frozenset(t) in impropers else prop).addTorsion(*t, n, ph, k)
            add("dihedral", prop)
            add("improper", imp)
        elif isinstance(f, mm.RBTorsionForce):
            add("dihedral", f)
        elif isinstance(f, mm.CustomTorsionForce):
            add("improper", f)
        elif isinstance(f, mm.CMAPTorsionForce):
            add("cmap", f)
        elif isinstance(f, mm.NonbondedForce):
            lj, coul = copy.deepcopy(f), copy.deepcopy(f)
            for i in range(f.getNumParticles()):
                q, s, e = f.getParticleParameters(i)
                lj.setParticleParameters(i, 0.0, s, 0.0 if foyer_geometric else e)
                coul.setParticleParameters(i, q, 1.0, 0.0)
            for i in range(f.getNumExceptions()):
                a, b, qq, s, e = f.getExceptionParameters(i)
                if foyer_geometric and e._value != 0.0:
                    # foyer: the 1-4 sigma mixes geometrically too.
                    _, si, _ = f.getParticleParameters(a)
                    _, sj, _ = f.getParticleParameters(b)
                    s = math.sqrt(si._value * sj._value)
                lj.setExceptionParameters(i, a, b, 0.0, s, e)
                coul.setExceptionParameters(i, a, b, qq, 1.0, 0.0)
            add("vdw", lj)
            add("coul", coul)
            if foyer_geometric:
                geo = mm.CustomNonbondedForce(
                    "4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)"
                )
                geo.addPerParticleParameter("sigma")
                geo.addPerParticleParameter("epsilon")
                for i in range(f.getNumParticles()):
                    _, s, e = f.getParticleParameters(i)
                    geo.addParticle([s, e])
                for i in range(f.getNumExceptions()):
                    a, b, *_ = f.getExceptionParameters(i)
                    geo.addExclusion(a, b)
                geo.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
                add("vdw", geo)
        elif isinstance(f, (mm.CustomNonbondedForce, mm.CustomBondForce)):
            add("vdw", f)
        elif isinstance(f, mm.CMMotionRemover):
            continue
        else:
            raise SystemExit(f"unexpected force {type(f).__name__}")
    for f in out:
        system.addForce(f)
    return groups


def openmm_price(path, top, foyer, sysj):
    import openmm as mm
    import openmm.unit as u
    from openmm import app

    ff = app.ForceField(str(path))
    system = ff.createSystem(
        top, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    bonds = {frozenset(b) for b in sysj["bonds"]}
    impropers = {frozenset(t) for t in sysj["impropers"]}
    groups = split_system(system, bonds, impropers, foyer)
    ctx = mm.Context(
        system, mm.VerletIntegrator(1.0), mm.Platform.getPlatformByName("Reference")
    )
    out = []
    for k, x in enumerate(sysj["configs"]):
        ctx.setPositions(
            [
                mm.Vec3(x[3 * i], x[3 * i + 1], x[3 * i + 2]) * 0.1
                for i in range(len(x) // 3)
            ]
        )
        terms = {}
        for g, name in groups.items():
            e = ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy()
            terms[name] = (
                terms.get(name, 0.0) + e.value_in_unit(u.kilojoule_per_mole) / KJ
            )
        state = ctx.getState(getEnergy=True, getForces=True)
        terms["total"] = (
            state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / KJ
        )
        f = state.getForces(asNumpy=False).value_in_unit(
            u.kilojoule_per_mole / u.angstrom
        )
        forces = [c / KJ for v in f for c in (v.x, v.y, v.z)]
        out.append((terms, forces))
    return out


# ── LAMMPS ────────────────────────────────────────────────────────────────


def lammps(sub: Path, k: int):
    lines = (sub / "lammps" / f"log.{k}").read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if l.split()[:2] == ["Step", "E_vdwl"])
    v = dict(zip(lines[i].split(), map(float, lines[i + 1].split())))
    terms = {
        "bond": v["E_bond"],
        "angle": v["E_angle"],
        "dihedral": v["E_dihed"],
        "improper": v["E_impro"],
        "cmap": v.get("f_cmap", 0.0),
        "vdw": v["E_vdwl"],
        "coul": v["E_coul"],
        "total": v["PotEng"],
    }
    dump = (sub / "lammps" / f"forces.{k}.dump").read_text().splitlines()
    start = dump.index("ITEM: ATOMS id fx fy fz") + 1
    forces = [float(c) for row in dump[start:] for c in row.split()[1:]]
    return terms, forces


# ── GROMACS ───────────────────────────────────────────────────────────────

GMX_TERMS = {
    "bond": ["Bond"],
    "angle": ["Angle", "U-B"],
    "dihedral": ["Proper Dih.", "Ryckaert-Bell.", "Fourier Dih."],
    "improper": ["Improper Dih.", "Per. Imp. Dih."],
    "cmap": ["CMAP Dih."],
    "vdw": ["LJ-14", "LJ (SR)"],
    "coul": ["Coulomb-14", "Coulomb (SR)"],
    "total": ["Potential"],
}


def trr_forces(path: Path):
    """The forces of a GROMACS .trr's first frame (XDR, single or double)."""
    b = path.read_bytes()
    pos = 0

    def ints(n):
        nonlocal pos
        v = struct.unpack(f">{n}i", b[pos : pos + 4 * n])
        pos += 4 * n
        return v

    magic, _ = ints(2)
    assert magic == 1993, path
    (slen,) = ints(1)
    pos += (slen + 3) // 4 * 4
    (_ir, _e, box, vir, pres, _top, _sym, x, v, f, natoms, _step, _nre) = ints(13)
    real = 8 if (box or x or f) == (9 if box else 3 * natoms) * 8 else 4
    pos += 2 * real
    pos += box + vir + pres + x + v
    fmt = ">d" if real == 8 else ">f"
    return [
        struct.unpack(fmt, b[pos + real * i : pos + real * (i + 1)])[0] / KJ / 10.0
        for i in range(3 * natoms)
    ]


def gromacs(run: Path):
    import pyedr

    edr = pyedr.edr_to_dict(str(run / "sp.edr"))
    terms = {
        t: sum(float(edr[n][0]) for n in names if n in edr) / KJ
        for t, names in GMX_TERMS.items()
    }
    return terms, trr_forces(run / "sp.trr")


# ── collect ───────────────────────────────────────────────────────────────


def collect(d: Path, pin: Path | None):
    rows = []
    for sub, sysj in sources(d):
        name, probes = sysj["source"], sysj["probes"]
        for k in range(len(sysj["configs"])):
            rows += rows_of(name, k, "lammps", *lammps(sub, k), probes[k])
            rows += rows_of(
                name, k, "gromacs", *gromacs(sub / "gromacs" / f"run_{k}"), probes[k]
            )
            if sysj["native"] == "gromacs":
                rows += rows_of(
                    name,
                    k,
                    "native",
                    *gromacs(sub / "gromacs" / f"native_{k}"),
                    probes[k],
                )
        for k, (terms, forces) in enumerate(
            openmm_price(*openmm_written(sub, sysj), sysj)
        ):
            rows += rows_of(name, k, "openmm", terms, forces, probes[k])
        if sysj["native"] == "openmm":
            for k, (terms, forces) in enumerate(
                openmm_price(*openmm_native(sysj), sysj)
            ):
                rows += rows_of(name, k, "native", terms, forces, probes[k])
    if (d / "sander.tsv").exists():
        rows += [
            (s, k, e, t, v) for (s, k, e, t), v in read_tsv(d / "sander.tsv").items()
        ]
    rows.sort(
        key=lambda r: (
            r[0],
            r[1],
            ["native", "lammps", "openmm", "gromacs"].index(r[2]),
            (TERMS + ["fdotv", "fnorm2"]).index(r[3]),
        )
    )
    write_tsv(pin or d / "engines.tsv", rows)

    molrs = read_tsv(d / "molrs.tsv")
    worst = {}
    print(
        f"{'source':9s} {'k':>1s} {'engine':8s} {'term':9s} {'engine value':>24s} {'molrs':>24s} {'rel':>8s}"
    )
    for s, k, e, t, v in rows:
        m = molrs[(s, k, e, t)]
        if t == "fdotv":
            scale = math.sqrt(molrs[(s, k, e, "fnorm2")]) * math.sqrt(
                sum(
                    p * p
                    for p in json.loads((d / s / "system.json").read_text())["probes"][
                        k
                    ]
                )
            )
        else:
            scale = max(abs(m), 1.0)
        rel = abs(v - m) / scale
        key = (s, e, t)
        worst[key] = max(worst.get(key, 0.0), rel)
        print(f"{s:9s} {k:1d} {e:8s} {t:9s} {v:24.16g} {m:24.16g} {rel:8.1e}")
    print(
        "\nworst error over the configurations, per source × engine × term (relative to the"
        " term, or to 1 kcal/mol below it; fdotv to |F||v|):"
    )
    names = sorted({s for s, _, _ in worst})
    for s in names:
        for e in ["native", "lammps", "openmm", "gromacs"]:
            cells = " ".join(
                f"{t}={worst[(s, e, t)]:.0e}"
                for t in TERMS + ["fdotv", "fnorm2"]
                if (s, e, t) in worst
            )
            print(f"  {s:9s} {e:8s} {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["sander", "collect"])
    ap.add_argument("dir", type=Path)
    ap.add_argument(
        "--pin",
        type=Path,
        help="write the engines' table here (the Rust test's fixture)",
    )
    args = ap.parse_args()
    if args.what == "sander":
        run_sander(args.dir)
    else:
        collect(args.dir, args.pin)


if __name__ == "__main__":
    os.environ.setdefault("OPENMM_CPU_THREADS", "1")
    sys.exit(main())
