#!/usr/bin/env bash
# molrs's CMAP kernel against LAMMPS `fix cmap` (molrs/src/ff/potential/cmap/
# lammps_check.rs). The test writes the LAMMPS inputs — data file with its
# CMAP section, the fix cmap file, the include with the `fix` line — when
# MOLRS_LAMMPS_CMAP_DIR is set; this script runs `run 0` on them and prints
# LAMMPS's f_cmap energy and per-atom forces beside molrs's, the numbers the
# test pins. Run it where cargo may build (a compute node), with `lmp` (or
# $LMP) built with the MOLECULE package.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT

MOLRS_LAMMPS_CMAP_DIR="$dir" cargo mrs-test -- \
    ff::potential::cmap::lammps_check --exact \
    ff::potential::cmap::lammps_check::energy_and_forces_are_lammps_fix_cmap --nocapture |
    grep '^molrs'

cat >"$dir/in.lammps" <<'IN'
atom_style      full
boundary        p p p
include         system.ff
read_data       data.lmp fix cmap crossterm CMAP
pair_style      zero 8.0
pair_coeff      * *
thermo_style    custom step pe f_cmap
thermo_modify   format float %.17g
dump            d all custom 1 forces.dump id fx fy fz
dump_modify     d format float %.17g sort id
run             0
IN
(cd "$dir" && "${LMP:-lmp}" -in in.lammps -log log.lammps -screen none)

python3 - "$dir" <<'PY'
import sys
from pathlib import Path

d = Path(sys.argv[1])
lines = (d / "log.lammps").read_text().splitlines()
head = next(i for i, l in enumerate(lines) if l.split()[:3] == ["Step", "PotEng", "f_cmap"])
step, pe, ecmap = lines[head + 1].split()
print(f"lammps energy {float(ecmap):.17e} (pe {float(pe):.17e})")
dump = (d / "forces.dump").read_text().splitlines()
start = dump.index("ITEM: ATOMS id fx fy fz") + 1
for row in dump[start:]:
    _, fx, fy, fz = row.split()
    print(f"lammps force {float(fx):.17e} {float(fy):.17e} {float(fz):.17e}")
PY
