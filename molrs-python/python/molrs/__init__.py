"""molrs — Rust-backed molecular simulation primitives.

**The top level is ``molrs::core``, and nothing else.** Storage (``Frame``,
``Block``, ``Trajectory``), the simulation cell and neighbor search, geometric
regions, the molecular-graph hierarchy and its live views, units, and the column
vocabulary — those are the primitives every other layer is written in terms of,
so they answer to ``molrs.<Name>`` directly.

Everything above core lives in the subpackage named after its Rust module, so
the Python path and the Rust path are the same word:

* :mod:`molrs.io` — file formats (PDB, XYZ, LAMMPS, GRO, DCD, TRR, XTC, CHGCAR,
  cube, SMILES). Field-canonicalizing; ``molrs.io.raw`` is the format-native
  binding.
* :mod:`molrs.perceive` — chemical perception: rings, aromaticity, hydrogens,
  stereochemistry, SMARTS matching, coarse-grained bead-pattern matching
  (``SubgraphMatcher``).
* :mod:`molrs.ff` — force fields, typifiers, charge models, potentials
  (:mod:`molrs.ff.potential`: the ``Potential`` protocol and one kernel class
  per style, e.g. ``LJCut``).
* :mod:`molrs.optimize` — geometry optimizers.
* :mod:`molrs.conformer` — 3D conformer generation.
* :mod:`molrs.md` — in-process molecular dynamics: velocity-Verlet/Langevin
  integrators and the ``MD`` driver; it integrates potentials, it defines none. Loaded lazily so a compiled ``_lib`` without ``md`` still
  imports.
* :mod:`molrs.op` — pure numeric base: weighted superposition, centroids.
* :mod:`molrs.builder` — structure builders (graphene, nanotubes).
* :mod:`molrs.compute` — analysis, one subpackage per ``molrs::compute`` domain.
* :mod:`molrs.signal` — FFT autocorrelation, windows, frequency grids.
* :mod:`molrs.stream` — live Frame streaming over WebSocket.

Each of those names has exactly one spelling — ``molrs.io.SmilesIR`` and
nothing else — so there is one thing to learn, document, and grep for.
"""

# `collections.abc.Mapping` is aliased so that it is not exported as `molrs.Mapping`.
from collections.abc import Mapping as _AbcMapping
from collections.abc import MutableMapping

from . import keys, schema
from ._lib import (
    Angle,
    Atom,
    Atomistic,
    Bead,
    Block,
    BlockDtypeError,
    Bond,
    Box,
    CGBond,
    CoarseGrain,
    Cuboid,
    Cylinder,
    Dihedral,
    DrudeParticle,
    Element,
    Ellipsoid,
    ExtractedSubgraph,
    Frame,
    FrameMeta,
    Graph,
    HalfSpace,
    Improper,
    MasslessSite,
    MetaDocument,
    MetaValue,
    NeighborList,
    NeighborQuery,
    Neighbors,
    NodeRef,
    Parallelepiped,
    Polyhedron,
    Port,
    Quantity,
    Reaction,
    Refs,
    Region,
    RelationBuckets,
    RelationRef,
    ScalarObservable,
    Sphere,
    SphereUnion,
    Topology,
    Trace,
    Trajectory,
    TriMesh,
    Unit,
    UnitPreset,
    UnitRegistry,
    UnitsError,
    VectorObservable,
    VerletSkin,
    VirtualSite,
    # FFI ABI handshake, read by name by downstream handle-bridge extensions
    # (molpack) at their import time.
    _ffi_abi_token,  # noqa: F401
)

# `frame.meta` implements the full mapping protocol in Rust; this makes
# `isinstance(frame.meta, MutableMapping)` say so too.
MutableMapping.register(FrameMeta)
# Registration supplies isinstance only — MetaDocument implements its own
# surface. Callers that branch on Mapping rather than dict:
# molvis/python/src/molvis/wire.py:389,530
# molrec/tests/molrs_adapter.py:110-113
_AbcMapping.register(MetaDocument)

from . import (
    builder,
    compute,  # analysis subpackage — one module per molrs::compute domain
    conformer,
    ff,
    io,
    op,
    optimize,
    perceive,
    signal,
    stream,
)


def __getattr__(name: str):
    """PEP 562 lazy loader for :mod:`molrs.md`.

    Loading it lazily keeps plain ``import molrs`` working against a compiled
    ``_lib`` that predates the ``md`` submodule.
    """
    if name == "md":
        import importlib

        return importlib.import_module(".md", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"md"})


__all__ = [
    "Angle",
    "Atom",
    "Atomistic",
    "Bead",
    "Block",
    # ---- molrs::core ----
    "BlockDtypeError",
    "Bond",
    "Box",
    "CGBond",
    "CoarseGrain",
    "Cuboid",
    "Cylinder",
    "Dihedral",
    "DrudeParticle",
    "Element",
    "Ellipsoid",
    "ExtractedSubgraph",
    "Frame",
    "FrameMeta",
    "Graph",
    "HalfSpace",
    "Improper",
    "MasslessSite",
    "MetaDocument",
    "MetaValue",
    "NeighborList",
    "NeighborQuery",
    "Neighbors",
    "NodeRef",
    "Parallelepiped",
    "Polyhedron",
    "Port",
    "Quantity",
    "Reaction",
    "Refs",
    "Region",
    "RelationBuckets",
    "RelationRef",
    "ScalarObservable",
    "Sphere",
    "SphereUnion",
    "Topology",
    "Trace",
    "Trajectory",
    "TriMesh",
    "Unit",
    "UnitPreset",
    "UnitRegistry",
    "UnitsError",
    "VectorObservable",
    "VerletSkin",
    "VirtualSite",
    "builder",
    # Subpackages — everything above molrs::core is reached through one of these.
    "compute",
    "conformer",
    "ff",
    "io",
    "keys",
    "md",
    "op",
    "optimize",
    "perceive",
    "schema",
    "signal",
    "stream",
]
