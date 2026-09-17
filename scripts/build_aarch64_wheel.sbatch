#!/bin/bash
# Build the molrs Python extension for aarch64 (GH200) as a *wheel*.
#
# Why a wheel and not `maturin develop`: molrs is installed editable, so the
# extension lands at molrs-python/python/molrs/_lib.abi3.so — one path shared
# by every interpreter. A develop build on the x86-64 login node and one on a
# GH200 node overwrite each other, and the loser fails at import with a wrong-
# ELF error that looks nothing like an architecture problem. Building to
# .wheels-aarch64/ keeps the two architectures apart; each interpreter installs
# the wheel it needs.
#
#SBATCH --job-name=molrs-aarch64
#SBATCH --account=naiss2026-4-715-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=scripts/molrs-aarch64-%j.out
#SBATCH --error=scripts/molrs-aarch64-%j.err

set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0

MOLCRAFTS="${MOLCRAFTS:-/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts}"
MOLRS="${MOLCRAFTS}/molrs"
PY="${PY:-/home/jicli594/work/.aarch64/bin/python}"
export RUSTUP_HOME="${MOLRS}/.rustup-aarch64"
export CARGO_HOME="${MOLRS}/.cargo-aarch64"
export CARGO_TARGET_DIR="${MOLRS}/molrs-python/target-aarch64"
export PATH="${CARGO_HOME}/bin:${PATH}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "=== bootstrapping rustup into ${RUSTUP_HOME} ==="
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable
fi
echo "=== $(cargo --version) / $(rustc --version) ==="

OUT="${MOLRS}/.wheels-aarch64"
mkdir -p "${OUT}"
cd "${MOLRS}/molrs-python"
"${PY}" -m maturin build --release --out "${OUT}"

echo "=== built ==="
ls -la "${OUT}"/*.whl | tail -3
echo "Install on an aarch64 interpreter with:"
echo "  \${PY} -m pip install --force-reinstall --no-deps \$(ls -1t ${OUT}/*.whl | head -1)"
echo "=== DONE $(date -Iseconds) ==="
