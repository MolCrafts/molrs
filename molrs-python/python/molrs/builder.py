"""Structure builders — ``molrs::builder``.

Graphene sheets (:class:`GrapheneBuilder`) and single-wall carbon nanotubes
(:class:`CarbonTubeBuilder`), each building a fresh :class:`molrs.store.Frame`.

Site-graph assembly: ``Assembler(library, SitePlacer(),
AxisOrienter()).assemble(sites)`` places one template copy per site of a
:class:`molrs.system.CoarseGrain`, joins bonded sites through their ports
(any topology), and returns the world as the graph class the caller names
(``assemble(sites, molrs.system.Atomistic)``; a bare
:class:`molrs.system.Graph` by default);
``Assembler(library, GrowthPlacer()).assemble(sites)`` grows a site graph
without positions (e.g. ``CGSmilesIR(...).to_coarsegrain()``).

Coarse-graining: :class:`Coarsener` maps disjoint node groups of a held
``CoarseGrain`` or ``Atomistic`` onto the sites of a new ``CoarseGrain``, each
at its group's centre of mass with an axis from the group's first member.
"""

from ._lib import (
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
