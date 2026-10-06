"""Space — ``molrs::spatial``.

* :class:`Box` — the periodic / triclinic simulation cell.
* Neighbour search: :class:`NeighborList` (the engine), :class:`Neighbors`
  (a materialized pair table), :class:`NeighborQuery` (cross-queries against
  a reference point set) and :class:`VerletSkin` (the skin-buffered rebuild
  policy an integrator owns).
* Regions — solids with a signed distance: :class:`Sphere`,
  :class:`Cuboid`, :class:`Parallelepiped`, :class:`HalfSpace`,
  :class:`Cylinder`, :class:`Ellipsoid`, :class:`Polyhedron`,
  :class:`SphereUnion`, and their Boolean composition :class:`Region`.
* :class:`TriMesh` — a triangle surface (what an STL reads into).
* :class:`Trace` — an ordered path of 3D points with no chemistry.
"""

from ._lib import (
    Box,
    Cuboid,
    Cylinder,
    Ellipsoid,
    HalfSpace,
    NeighborList,
    NeighborQuery,
    Neighbors,
    Parallelepiped,
    Polyhedron,
    Region,
    Sphere,
    SphereUnion,
    Trace,
    TriMesh,
    VerletSkin,
)

__all__ = [
    "Box",
    "Cuboid",
    "Cylinder",
    "Ellipsoid",
    "HalfSpace",
    "NeighborList",
    "NeighborQuery",
    "Neighbors",
    "Parallelepiped",
    "Polyhedron",
    "Region",
    "Sphere",
    "SphereUnion",
    "Trace",
    "TriMesh",
    "VerletSkin",
]
