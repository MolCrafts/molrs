"""molrs — Rust-backed molecular simulation primitives.

**The top level is ``molrs::core``, and nothing else.** Storage (``Frame``,
``Block``, ``Trajectory``), the simulation cell and neighbor search, geometric
regions, the molecular-graph hierarchy and its views, units, and the column
vocabulary — those are the primitives every other layer is written in terms of,
so they answer to ``molrs.<Name>`` directly.

Everything above core lives in the subpackage named after its Rust module, so
the Python path and the Rust path are the same word:

* :mod:`molrs.io` — file formats (PDB, XYZ, LAMMPS, GRO, DCD, TRR, XTC, CHGCAR,
  cube, SMILES). Field-canonicalizing; ``molrs.io.raw`` is the format-native
  binding.
* :mod:`molrs.perceive` — chemical perception: rings, aromaticity, hydrogens,
  stereochemistry, SMARTS matching.
* :mod:`molrs.ff` — force fields, typifiers, charge models, potentials.
* :mod:`molrs.optimize` — geometry optimizers.
* :mod:`molrs.conformer` — 3D conformer generation.
* :mod:`molrs.builder` — structure builders (graphene, nanotubes, SARW paths).
* :mod:`molrs.compute` — analysis, one subpackage per ``molrs::compute`` domain.
* :mod:`molrs.signal` — FFT autocorrelation, windows, frequency grids.
* :mod:`molrs.stream` — live Frame streaming over WebSocket.

Each of those names has exactly one spelling — ``molrs.io.SmilesIR`` and
nothing else — so there is one thing to learn, document, and grep for.
"""

from ._lib import (
    # Public exceptions
    BlockDtypeError,
    UnitsError,
    # SimBox + neighbors
    Box,
    NeighborList,
    Neighbors,
    NeighborQuery,
    VerletSkin,
    # Block + Frame
    Block,
    MetaValue,
    Frame,
    FRAME_SCHEMA_VERSION,
    # FFI ABI handshake (consumed by downstream handle-bridge extensions,
    # e.g. molpack, at their import time)
    _ffi_abi_token,
    Unit,
    Quantity,
    UnitRegistry,
    Trajectory,
    ScalarObservable,
    VectorObservable,
    MolRec,
    Observables,
    # Regions
    Sphere,
    HollowSphere,
    Cuboid,
    Parallelepiped,
    Region,
    # Molecular graph hierarchy
    Element,
    Graph,
    Atomistic,
    CoarseGrain,
    ExtractedSubgraph,
    Reaction,
    # Systems (module-level free functions over a graph)
    translate,
    rotate,
    scale,
    align_direction,
    # Field-name convention + the Frame vocabulary, both projected from the
    # committed Rust tables.
    keys,
    schema,
)

# Rich Python Frame/Block layer (pandas-style API; CSV engine in Rust on the
# core Block). These subclass the bare PyO3 cores and SHADOW the top-level
# ``molrs.Block`` / ``molrs.Frame`` as the canonical types — every public API
# (io readers, etc.) yields these. The shadow is safe now that molpy re-exports
# them instead of subclassing the bare core (chain spec 04). Internal modules
# that need the raw cores import them from ``._lib`` directly.
from . import frame  # noqa: F401
from .frame import Block, Frame

from . import compute  # analysis subpackage — one module per molrs::compute domain
from . import conformer
from . import ff
from . import builder
from . import io
from . import md
from . import stream
from . import optimize
from . import perceive
from . import signal
from .views import (
    Angle,
    Atom,
    Atomistic,
    Bead,
    Bond,
    CGBond,
    CoarseGrain,
    Dihedral,
    DrudeParticle,
    GraphViews,
    Improper,
    MasslessSite,
    NodeRef,
    Refs,
    RelationRef,
    VirtualSite,
)

__all__ = [
    # Subpackages — everything above molrs::core is reached through one of these.
    "compute",
    "conformer",
    "ff",
    "builder",
    "io",
    "md",
    "stream",
    "optimize",
    "perceive",
    "signal",
    # ---- molrs::core ----
    "BlockDtypeError",
    "UnitsError",
    "Box",
    "NeighborList",
    "Neighbors",
    "NeighborQuery",
    "VerletSkin",
    "Block",
    "MetaValue",
    "Frame",
    "FRAME_SCHEMA_VERSION",
    "Unit",
    "Quantity",
    "UnitRegistry",
    "Trajectory",
    "ScalarObservable",
    "VectorObservable",
    "MolRec",
    "Observables",
    "Sphere",
    "HollowSphere",
    "Cuboid",
    "Parallelepiped",
    "Region",
    "Element",
    "Graph",
    "Atomistic",
    "CoarseGrain",
    "ExtractedSubgraph",
    "Reaction",
    "NodeRef",
    "RelationRef",
    "Refs",
    "GraphViews",
    "Atom",
    "VirtualSite",
    "DrudeParticle",
    "MasslessSite",
    "Bond",
    "Angle",
    "Dihedral",
    "Improper",
    "Bead",
    "CGBond",
    "translate",
    "rotate",
    "scale",
    "align_direction",
    "keys",
    "schema",
]
