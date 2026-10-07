"""molrs — Rust-backed molecular simulation primitives.

**The top level is the subsystems, and nothing else** — exactly as the Rust
crate's root is. Every symbol lives in the module named after its Rust owner,
so the Python path and the Rust path are the same words
(``molrs.core.Frame`` is ``molrs::core::Frame``):

* :mod:`molrs.core` — the core data model: the column store and the frame
  (``Block``, ``Frame``, ``Trajectory``), space (``Box``, neighbour search,
  regions, meshes, point paths), the molecular graph (``MolGraph``,
  ``Atomistic``, ``CoarseGrain`` and their live views), elements and the
  unit engine; the vocabularies :mod:`molrs.core.keys`,
  :mod:`molrs.core.schema` and :mod:`molrs.core.constants`.
* :mod:`molrs.op` — pure numeric base: weighted superposition, centroids.
* :mod:`molrs.perceive` — chemical perception: rings, aromaticity,
  hydrogens, stereochemistry, SMARTS matching and reactions, coarse-grained
  bead-pattern matching.
* :mod:`molrs.io` — every file reader and writer: structure, trajectory and
  force-field files, SMILES and CGsmiles text, ``*.mrec`` records, frame
  bytes.
* :mod:`molrs.ff` — force fields, one submodule per Rust owner
  (``forcefield``, ``potential``, ``typifier``, ``charge``, ``ir``,
  ``params``, ``scale_lj``).
* :mod:`molrs.optimize` — geometry optimizers.
* :mod:`molrs.md` — in-process molecular dynamics: the integrators and the
  ``MD`` driver; it integrates potentials, it defines none.
* :mod:`molrs.conformer` — 3D conformer generation.
* :mod:`molrs.builder` — structure builders, site-graph assembly,
  coarse-graining.
* :mod:`molrs.compute` — trajectory analysis.
* :mod:`molrs.signal` — FFT autocorrelation, windows, frequency grids.
* :mod:`molrs.stream` — live Frame streaming (the transport).

Each name has exactly one spelling — ``molrs.io.smiles.SmilesIR`` and nothing
else — so there is one thing to learn, document, and grep for.
"""

from . import (
    builder,
    compute,
    conformer,
    core,
    ff,
    io,
    md,
    op,
    optimize,
    perceive,
    signal,
    stream,
)
from ._lib import __version__ as __version__

# FFI ABI handshake, read by name by downstream handle-bridge extensions
# (molpack) at their import time.
from ._lib import _ffi_abi_token  # noqa: F401

__all__ = [
    "builder",
    "compute",
    "conformer",
    "core",
    "ff",
    "io",
    "md",
    "op",
    "optimize",
    "perceive",
    "signal",
    "stream",
]
