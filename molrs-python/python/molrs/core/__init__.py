"""The core data model — ``molrs::core``.

Every name is flat on :mod:`molrs.core`, as in Rust:

* The column store and the frame: :class:`Block`, :class:`Frame` and its
  metadata (:class:`FrameMeta`, :class:`MetaValue` typed scalars,
  :class:`MetaDocument` nested documents), :class:`Trajectory` and its
  :class:`ScalarObservable` / :class:`VectorObservable` records, and
  :exc:`BlockDtypeError` (a column value the store cannot hold; a
  ``TypeError``).
* Space: :class:`Box` (the periodic / triclinic simulation cell; Rust's
  ``SimBox``), neighbour search (:class:`NeighborList`, :class:`Neighbors`,
  :class:`NeighborQuery`, :class:`VerletSkin`), the regions
  (:class:`Sphere`, :class:`Cuboid`, :class:`Parallelepiped`,
  :class:`HalfSpace`, :class:`Cylinder`, :class:`Ellipsoid`,
  :class:`Polyhedron`, :class:`SphereUnion` and their Boolean composition
  :class:`Region`), :class:`TriMesh` and :class:`Trace`.
* The molecular graph: :class:`MolGraph` and its leaves :class:`Atomistic` /
  :class:`CoarseGrain`, their live views (:class:`NodeRef`, :class:`Atom`,
  :class:`VirtualSite`, :class:`DrudeParticle`, :class:`MasslessSite`,
  :class:`Bead`, :class:`RelationRef`, :class:`Bond`, :class:`Angle`,
  :class:`Dihedral`, :class:`Improper`, :class:`CGBond`, :class:`Port`,
  :class:`Refs`, :class:`RelationBuckets`), :class:`ExtractedSubgraph`,
  :class:`Element` and the index-only :class:`Topology`.
* Units: :class:`Unit`, :class:`Quantity`, :class:`UnitRegistry`,
  :class:`UnitPreset`, and :exc:`UnitsError` (a ``ValueError``).

Three vocabularies are submodules: :mod:`molrs.core.keys` (column, block
and frame-meta keys), :mod:`molrs.core.schema` (column and block
specifications) and :mod:`molrs.core.constants` (physical and engine
constants).
"""

# `collections.abc.Mapping` is aliased so that it is not exported here.
from collections.abc import Mapping as _AbcMapping
from collections.abc import MutableMapping as _AbcMutableMapping

from .._lib import (
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
    HalfSpace,
    Improper,
    MasslessSite,
    MetaDocument,
    MetaValue,
    MolGraph,
    NeighborList,
    NeighborQuery,
    Neighbors,
    NodeRef,
    Parallelepiped,
    Polyhedron,
    Port,
    Quantity,
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
)
from . import constants, keys, schema

# `frame.meta` implements the full mapping protocol in Rust; this makes
# `isinstance(frame.meta, MutableMapping)` say so too.
_AbcMutableMapping.register(FrameMeta)
# Registration supplies isinstance only — MetaDocument implements its own
# surface. Callers that branch on Mapping rather than dict:
# molvis/python/src/molvis/wire.py:389,530
# molrec/tests/molrs_adapter.py:110-113
_AbcMapping.register(MetaDocument)

__all__ = [
    "Angle",
    "Atom",
    "Atomistic",
    "Bead",
    "Block",
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
    "HalfSpace",
    "Improper",
    "MasslessSite",
    "MetaDocument",
    "MetaValue",
    "MolGraph",
    "NeighborList",
    "NeighborQuery",
    "Neighbors",
    "NodeRef",
    "Parallelepiped",
    "Polyhedron",
    "Port",
    "Quantity",
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
    "constants",
    "keys",
    "schema",
]
