#!/usr/bin/env bash
# LAMMPS's energies for molrs's OpenMM-XML check (molrs/src/ff/openmm_check.rs).
# The test writes, for each case (charmm, amber, opls), the data file and the
# include molrs writes from the OpenMM-read force field when
# MOLRS_OPENMM_CHECK_DIR is set, and prints molrs's and OpenMM's per-term
# energies; this script runs LAMMPS `run 0` on the files and prints its
# `evdwl ecoul ebond eangle edihed eimp f_cmap pe`, the numbers the test pins.
# OpenMM's numbers come from scripts/openmm_xml_check.py (needs OpenMM). Run it
# where cargo may build (a compute node), with `lmp` (or $LMP) built with the
# MOLECULE package.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT

MOLRS_OPENMM_CHECK_DIR="$dir" cargo mrs-test -- ff::openmm_check::report --exact --nocapture |
    grep -E '^(charmm|amber|opls) '

# An MPI-built lmp started more than once inside one `srun` step would reuse
# the step's PMI socket, and the second MPI_Init dies of SIGPIPE.
for v in ${!PMI_@}; do unset "$v"; done

for case in charmm amber opls; do
    read_data="read_data       $case.data"
    cmap=""
    if [ -f "$dir/$case.cmap" ]; then
        read_data="$read_data fix cmap crossterm CMAP"
        cmap=" f_cmap"
    fi
    cat >"$dir/in.$case" <<IN
atom_style      full
boundary        f f f
include         $case.pre
$read_data
include         $case.ff
neighbor        2.0 nsq
thermo_style    custom step evdwl ecoul ebond eangle edihed eimp$cmap pe
thermo_modify   format float %.17g
run             0
IN
    (cd "$dir" && "${LMP:-lmp}" -in "in.$case" -log "log.$case" -screen none)
    python3 - "$dir/log.$case" "$case" <<'PY'
import sys

lines = open(sys.argv[1]).read().splitlines()
head = next(i for i, l in enumerate(lines) if l.split()[:2] == ["Step", "E_vdwl"])
for key, value in zip(lines[head].split()[1:], lines[head + 1].split()[1:]):
    print(f"{sys.argv[2]} lammps {key:8s} {float(value):.17e}")
PY
done
