#!/usr/bin/env python3
"""Build the AMBER prmtop fixtures of molrs's prmtop check, and sander's
single-point energies on them.

Run under AmberTools (tleap, antechamber, parmchk2, ParmEd, pysander), e.g.

    module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 AmberTools/26.1
    python3 scripts/prmtop_fixtures.py <out_dir>

It writes, per case, ``<case>.parm7`` and ``<case>.rst7`` (coordinates
perturbed off the built geometry by a seeded 0.08 Å Gaussian, so no term sits
at a symmetric point), and prints sander's energy decomposition at those
coordinates (``imin=0``-style single point, ``igb=0``, ``ntb=0``,
``cut=999``) as ``sander <case> <term> <value>`` lines at full precision.

Cases:

- ``ff14sb``: ACE-PHE-NME, ff14SB — multi-term torsions, impropers, a
  six-membered ring, uniform SCEE/SCNB 1.2/2.0.
- ``gaff2``: N6,N6-dimethyl... the modXNA ``DMA`` fragment retyped to GAFF2
  (antechamber ``-at gaff2``, charges kept), parmchk2 for missing terms.
- ``gaff2_multi``: ``gaff2`` with two multi-term impropers: a second row on
  one improper quartet (ParmEd) and a negative-PN chain on another's type
  (hand edit of ``DIHEDRAL_PERIODICITY``).
- ``glycam``: alpha-D-glucose (GLYCAM_06j ``ROH``-``0GA``, SCEE = SCNB = 1)
  beside ACE-ALA-NME (ff14SB, 1.2 / 2.0) in one prmtop: non-uniform SCEE/SCNB.
- ``chamber``: CHARMM36 alanine dipeptide (``ALAD``) and alpha-D-glucose
  (``AGLC``) through ParmEd's chamber: Urey-Bradley, CHARMM impropers, CMAP,
  ``LENNARD_JONES_14_*``, CHARMM charges (sqrt(332.0716)).
- ``ff19sb``: ACE-ALA-NME, ff19SB — AMBER's own ``CMAP_*`` sections.
"""

import os
import re
import subprocess
import sys

import numpy as np
import parmed as pmd
import sander
from parmed.amber import AmberParm
from parmed.amber._chamberparm import ConvertFromPSF
from parmed.charmm import CharmmParameterSet, CharmmPsfFile

AMBERHOME = os.environ["AMBERHOME"]
CHAMBER_DAT = os.path.join(AMBERHOME, "dat", "chamber")
SEED = 20261006
SIGMA = 0.08


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tleap(script, cwd):
    with open(os.path.join(cwd, "leap.in"), "w") as f:
        f.write(script + "\nquit\n")
    run("tleap -f leap.in > leap.log 2>&1", cwd)
    log = open(os.path.join(cwd, "leap.log")).read()
    m = re.search(r"Errors = (\d+)", log)
    if m is None or int(m.group(1)) != 0:
        sys.exit(f"tleap failed in {cwd}:\n{log}")


def perturbed(xyz, salt):
    rng = np.random.default_rng(SEED + salt)
    return np.asarray(xyz) + rng.normal(0.0, SIGMA, np.shape(xyz))


def save(parm, xyz, out, case):
    parm.coordinates = xyz
    parm.box = None
    parm.save(os.path.join(out, f"{case}.parm7"), overwrite=True)
    parm.save(os.path.join(out, f"{case}.rst7"), format="rst7", overwrite=True)


def sander_terms(out, case):
    inp = sander.gas_input(0)
    inp.cut = 999.0
    parm = os.path.join(out, f"{case}.parm7")
    rst = os.path.join(out, f"{case}.rst7")
    with sander.setup(parm, rst, None, inp):
        e, _ = sander.energy_forces()
    for term in ("bond", "angle", "angle_ub", "dihedral", "imp", "cmap",
                 "vdw_14", "elec_14", "vdw", "elec", "tot"):
        print(f"sander {case} {term} {getattr(e, term)!r}")


def ff14sb(out, work):
    d = os.path.join(work, "ff14sb")
    os.makedirs(d, exist_ok=True)
    tleap("source leaprc.protein.ff14SB\n"
          "m = sequence { ACE PHE NME }\n"
          "saveamberparm m m.parm7 m.rst7", d)
    parm = pmd.load_file(os.path.join(d, "m.parm7"), os.path.join(d, "m.rst7"))
    save(parm, perturbed(parm.coordinates, 1), out, "ff14sb")


def ff19sb(out, work):
    d = os.path.join(work, "ff19sb")
    os.makedirs(d, exist_ok=True)
    tleap("source leaprc.protein.ff19SB\n"
          "m = sequence { ACE ALA NME }\n"
          "saveamberparm m m.parm7 m.rst7", d)
    parm = pmd.load_file(os.path.join(d, "m.parm7"), os.path.join(d, "m.rst7"))
    save(parm, perturbed(parm.coordinates, 6), out, "ff19sb")


def gaff2(out, work):
    d = os.path.join(work, "gaff2")
    os.makedirs(d, exist_ok=True)
    src = os.path.join(AMBERHOME, "dat", "modXNA", "lib_base", "DMA.mol2")
    run(f"antechamber -i {src} -fi mol2 -o dma.mol2 -fo mol2 -at gaff2 "
        "-rn DMA -pf y", d)
    run("parmchk2 -i dma.mol2 -f mol2 -o dma.frcmod -s gaff2", d)
    tleap("source leaprc.gaff2\n"
          "loadamberparams dma.frcmod\n"
          "m = loadmol2 dma.mol2\n"
          "saveamberparm m m.parm7 m.rst7", d)
    parm = pmd.load_file(os.path.join(d, "m.parm7"), os.path.join(d, "m.rst7"))
    xyz = perturbed(parm.coordinates, 2)
    save(parm, xyz, out, "gaff2")
    gaff2_multi(parm, xyz, out)


def gaff2_multi(parm, xyz, out):
    """Two multi-term impropers on the GAFF2 molecule: (a) a second, n = 3
    row on one improper's quartet (ParmEd), and (b) a negative periodicity on
    the type of another, which sander continues into the next type's term
    (hand edit of ``DIHEDRAL_PERIODICITY``)."""
    from parmed.topologyobjects import Dihedral, DihedralType

    impropers = [d for d in parm.dihedrals if d.improper]
    first = impropers[0]
    extra = DihedralType(0.35, 3, 0.0, 1.2, 2.0, list=parm.dihedral_types)
    parm.dihedral_types.append(extra)
    parm.dihedrals.append(Dihedral(first.atom1, first.atom2, first.atom3, first.atom4,
                                   improper=True, ignore_end=True, type=extra))
    parm.remake_parm()
    path = os.path.join(out, "gaff2_multi.parm7")
    parm.coordinates = xyz
    parm.save(path, overwrite=True)
    parm.save(os.path.join(out, "gaff2_multi.rst7"), format="rst7", overwrite=True)

    # (b): a type every row of which is an improper, not the table's last.
    data = AmberParm(path).parm_data
    rows = data["DIHEDRALS_INC_HYDROGEN"] + data["DIHEDRALS_WITHOUT_HYDROGEN"]
    users = {}
    for r in range(0, len(rows), 5):
        users.setdefault(rows[r + 4] - 1, []).append(rows[r + 3] < 0)
    n_types = len(data["DIHEDRAL_PERIODICITY"])
    extra_tid = [t for t, imp in users.items() if all(imp) and t != first.type.idx]
    candidates = [t for t in sorted(extra_tid) if t + 1 < n_types and t != extra.idx]
    if not candidates:
        sys.exit("gaff2_multi: no improper-only type to chain")
    chained = candidates[0]
    per = list(data["DIHEDRAL_PERIODICITY"])
    per[chained] = -abs(per[chained])
    rewrite_floats(path, "DIHEDRAL_PERIODICITY", per)
    print(f"gaff2_multi: improper type {chained + 1} chains into type {chained + 2}; "
          f"type {extra.idx + 1} is a second row on improper "
          f"{[first.atom1.idx + 1, first.atom2.idx + 1, first.atom3.idx + 1, first.atom4.idx + 1]}")


def rewrite_floats(path, flag, values):
    """Replace the data of ``%FLAG flag`` (``5E16.8``) in place."""
    lines = open(path).read().split("\n")
    start = lines.index(f"%FLAG {flag}")
    if not lines[start + 1].startswith("%FORMAT(5E16.8)"):
        sys.exit(f"{flag}: unexpected format {lines[start + 1]}")
    end = start + 2
    while end < len(lines) and not lines[end].startswith("%"):
        end += 1
    body = ["".join(f"{v:16.8E}" for v in values[i:i + 5]) for i in range(0, len(values), 5)]
    lines[start + 2:end] = body
    open(path, "w").write("\n".join(lines))


def glycam(out, work):
    d = os.path.join(work, "glycam")
    os.makedirs(d, exist_ok=True)
    tleap("source leaprc.protein.ff14SB\n"
          "source leaprc.GLYCAM_06j-1\n"
          "g = sequence { ROH 0GA }\n"
          "p = sequence { ACE ALA NME }\n"
          "translate p { 12.0 0.0 0.0 }\n"
          "m = combine { g p }\n"
          "savepdb g g.pdb\n"
          "savepdb p p.pdb\n"
          "saveamberparm m m.parm7 m.rst7", d)
    parm = pmd.load_file(os.path.join(d, "m.parm7"), os.path.join(d, "m.rst7"))
    save(parm, perturbed(parm.coordinates, 3), out, "glycam")
    return d


# CHARMM names of ALAD / AGLC <- tleap PDB names of ACE-ALA-NME / ROH-0GA.
ALAD_FROM = {
    ("ACE", "H1"): "HL1", ("ACE", "CH3"): "CL", ("ACE", "H2"): "HL2",
    ("ACE", "H3"): "HL3", ("ACE", "C"): "CLP", ("ACE", "O"): "OL",
    ("ALA", "N"): "NL", ("ALA", "H"): "HL", ("ALA", "CA"): "CA", ("ALA", "HA"): "HA",
    ("ALA", "CB"): "CB", ("ALA", "HB1"): "HB1", ("ALA", "HB2"): "HB2",
    ("ALA", "HB3"): "HB3", ("ALA", "C"): "CRP", ("ALA", "O"): "OR",
    ("NME", "N"): "NR", ("NME", "H"): "HR", ("NME", "C"): "CR",
    ("NME", "H1"): "HR1", ("NME", "H2"): "HR2", ("NME", "H3"): "HR3",
}
AGLC_RENAME = {"H6O": "HO6", "H4O": "HO4", "H3O": "HO3", "H2O": "HO2"}


def rtf_block(path, resname):
    """The IMPR and CMAP atom-name tuples of residue ``resname``."""
    impr, cmap, inside = [], [], False
    for line in open(path):
        line = line.split("!")[0]
        words = line.split()
        if not words:
            continue
        key = words[0].upper()
        if key in ("RESI", "PRES"):
            inside = words[1].upper() == resname
            continue
        if not inside:
            continue
        if key.startswith("IMPR") or key.startswith("IMPH"):
            names = words[1:]
            impr.extend(tuple(names[i:i + 4]) for i in range(0, len(names), 4))
        elif key == "CMAP":
            names = words[1:]
            for i in range(0, len(names), 8):
                a = names[i:i + 8]
                if a[1:4] != a[4:7]:
                    sys.exit(f"{resname}: CMAP {a} is not two consecutive dihedrals")
                cmap.append((a[0], a[1], a[2], a[3], a[7]))
    return impr, cmap


def chamber(out, work, glycam_dir):
    from parmed import Structure
    from parmed.topologyobjects import Angle, Atom, Bond, Cmap, Dihedral, Improper

    d = os.path.join(work, "chamber")
    os.makedirs(d, exist_ok=True)
    files = [os.path.join(CHAMBER_DAT, f) for f in
             ("top_all36_prot.rtf", "par_all36_prot.prm",
              "top_all36_carb.rtf", "par_all36_carb.prm")]
    params = CharmmParameterSet(*files)

    peptide = pmd.load_file(os.path.join(glycam_dir, "p.pdb"))
    sugar = pmd.load_file(os.path.join(glycam_dir, "g.pdb"))
    xyz_of = {}
    for a in peptide.atoms:
        xyz_of[("ALAD", ALAD_FROM[(a.residue.name, a.name)])] = (a.xx, a.xy, a.xz)
    for a in sugar.atoms:
        # Far from the peptide (the two sit apart, as in ``glycam``).
        xyz_of[("AGLC", AGLC_RENAME.get(a.name, a.name))] = (a.xx - 12.0, a.xy, a.xz)

    s = Structure()
    xyz = []
    for resnum, (resname, rtf) in enumerate((("ALAD", files[0]), ("AGLC", files[2])), 1):
        tmpl = params.residues[resname]
        base = len(s.atoms)
        index = {}
        for a in tmpl.atoms:
            at = params.atom_types[a.type]
            atom = Atom(name=a.name, type=a.type, charge=a.charge, mass=at.mass,
                        atomic_number=at.atomic_number)
            s.add_atom(atom, resname, resnum)
            index[a.name] = base + len(index)
            xyz.append(xyz_of[(resname, a.name)])
        for b in tmpl.bonds:
            s.bonds.append(Bond(s.atoms[index[b.atom1.name]], s.atoms[index[b.atom2.name]]))
        impr, cmap = rtf_block(rtf, resname)
        for names in impr:
            s.impropers.append(Improper(*(s.atoms[index[n]] for n in names)))
        for names in cmap:
            s.cmaps.append(Cmap(*(s.atoms[index[n]] for n in names)))
    # Angles and proper dihedrals from the bond graph.
    for j in s.atoms:
        nbr = sorted(j.bond_partners, key=lambda a: a.idx)
        for x in range(len(nbr)):
            for y in range(x + 1, len(nbr)):
                s.angles.append(Angle(nbr[x], j, nbr[y]))
    for b in s.bonds:
        j, k = b.atom1, b.atom2
        for i in j.bond_partners:
            if i is k:
                continue
            for l in k.bond_partners:
                if l is j or l is i:
                    continue
                s.dihedrals.append(Dihedral(i, j, k, l))
    psf_path = os.path.join(d, "system.psf")
    CharmmPsfFile.from_structure(s).write_psf(psf_path)
    psf = CharmmPsfFile(psf_path)
    psf.load_parameters(params)
    psf.coordinates = perturbed(xyz, 4)
    parm = ConvertFromPSF(psf, params, title="ALAD AGLC")
    parm.save(os.path.join(out, "chamber.parm7"), overwrite=True)
    parm.save(os.path.join(out, "chamber.rst7"), format="rst7", overwrite=True)


def main():
    out = os.path.abspath(sys.argv[1])
    work = os.path.join(out, "_work")
    os.makedirs(work, exist_ok=True)
    ff14sb(out, work)
    gaff2(out, work)
    glycam_dir = glycam(out, work)
    chamber(out, work, glycam_dir)
    ff19sb(out, work)
    for case in ("ff14sb", "gaff2", "gaff2_multi", "glycam", "chamber", "ff19sb"):
        sander_terms(out, case)


if __name__ == "__main__":
    main()
