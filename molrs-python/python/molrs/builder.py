"""Structure builders — ``molrs::builder``.

Graphene sheets (:class:`GrapheneBuilder`) and single-wall carbon nanotubes
(:class:`CarbonTubeBuilder`), each building a fresh :class:`molrs.core.Frame`.

Site-graph assembly: ``Assembler(library, SitePlacer(),
AxisOrienter()).assemble(sites)`` places one template copy per site of a
:class:`molrs.core.CoarseGrain`, joins bonded sites through their ports
(any topology), and returns the world as the graph class the caller names
(``assemble(sites, molrs.core.Atomistic)``; a bare
:class:`molrs.core.MolGraph` by default);
``Assembler(library, GrowthPlacer()).assemble(sites)`` grows a site graph
without positions (e.g. ``CgSmilesIr(...).to_coarsegrain()``).

Coarse-graining: :class:`Coarsener` maps disjoint node groups of a held
``CoarseGrain`` or ``Atomistic`` onto the sites of a new ``CoarseGrain``, each
at its group's centre of mass with an axis from the group's first member.
"""

from ._native import (
    Assembler,
    AxisOrienter,
    CarbonTubeBuilder,
    Coarsener,
    GrapheneBuilder,
    GrowthPlacer,
    SitePlacer,
)

__all__ = [
    "Assembler",
    "AxisOrienter",
    "CarbonTubeBuilder",
    "Coarsener",
    "GrapheneBuilder",
    "GrowthPlacer",
    "SitePlacer",
]
