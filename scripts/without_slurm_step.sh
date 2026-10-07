# shellcheck shell=bash
# Sourced by the engine-check scripts (after they cd to the repository root):
#
#   source scripts/without_slurm_step.sh
#   without_slurm_step "$LMP" -in in.lammps ...
#
# `without_slurm_step CMD ARGS...` runs CMD with every Slurm, PMI, PMIx and
# Open MPI variable removed from its environment. Inside one Slurm step a
# second MPI singleton start (an MPI-built `lmp` or `gmx` launched by a
# script that itself runs under `srun`) takes the step's PMI socket and dies
# of SIGPIPE; without those variables it starts as a plain singleton.
without_slurm_step() {
    # shellcheck disable=SC2046 # one `-u NAME` word pair per variable
    env $(env | grep -o '^\(PMI\|PMIX\|SLURM\|OMPI\)[A-Za-z0-9_]*' | sed 's/^/-u /') "$@"
}
