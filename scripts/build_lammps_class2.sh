#!/usr/bin/env bash
# Build the LAMMPS that scripts/ff_ir_extension_lammps_check.sh runs the
# class2 `bond_angle` cases against (MOLRS_LMP_CLASS2): the `lmp` the other
# cases use, plus the CLASS2 package.
#
#   scripts/build_lammps_class2.sh PREFIX
#   MOLRS_LMP_CLASS2=PREFIX/bin/lmp scripts/ff_ir_extension_lammps_check.sh
#
# The source is PREFIX/src, or LAMMPS_SRC; when it holds no LAMMPS, LAMMPS
# develop at LAMMPS_REF (7f680de2, patch_30Mar2026-1074, the version the
# pinned tables were taken with) is fetched there. CMake, a C++17 compiler
# with OpenMP, MPI and FFTW3 must be on PATH (on the NAISS cluster:
# `module load buildenv-gcc/2026.03-mpich`). Build on a compute node; JOBS
# sets the parallelism (default: nproc).
set -euo pipefail
prefix=$(realpath -m "${1:?usage: $0 PREFIX}")
src=${LAMMPS_SRC:-$prefix/src}
ref=${LAMMPS_REF:-7f680de2968ee8b9dc33aae2061791bbf50ca82e}

if [[ ! -f $src/cmake/CMakeLists.txt ]]; then
    git init -q "$src"
    git -C "$src" fetch -q --depth 1 https://github.com/lammps/lammps.git "$ref"
    git -C "$src" checkout -q FETCH_HEAD
fi

cmake -S "$src/cmake" -B "$prefix/build" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="$prefix" \
    -D BUILD_MPI=on \
    -D BUILD_OMP=on \
    -D FFT=FFTW3 \
    -D PKG_CLASS2=on \
    -D PKG_MOLECULE=on \
    -D PKG_EXTRA-MOLECULE=on \
    -D PKG_EXTRA-PAIR=on \
    -D PKG_EXTRA-COMPUTE=on \
    -D PKG_EXTRA-DUMP=on \
    -D PKG_EXTRA-FIX=on \
    -D PKG_KSPACE=on \
    -D PKG_MANYBODY=on \
    -D PKG_MISC=on \
    -D PKG_OPENMP=on \
    -D PKG_QEQ=on \
    -D PKG_REPLICA=on \
    -D PKG_RIGID=on
cmake --build "$prefix/build" -j "${JOBS:-$(nproc)}"
cmake --install "$prefix/build" >/dev/null
[[ $("$prefix/bin/lmp" -h) == *CLASS2* ]] || { echo "built lmp lacks CLASS2" >&2; exit 1; }
echo "$prefix/bin/lmp"
