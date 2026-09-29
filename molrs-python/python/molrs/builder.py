"""Structure builders — ``molrs::builder``.

Graphene sheets (:class:`GrapheneBuilder`) and single-wall carbon nanotubes
(:class:`CarbonTubeBuilder`), each building a fresh :class:`molrs.Frame`.

Site-graph assembly: ``Assembler(library, SitePlacer(),
AxisOrienter()).assemble(sites)`` places one template copy per site of a
:class:`molrs.CoarseGrain`, joins bonded sites through their ports (any
topology), and returns the world as the graph class the caller names
(``assemble(sites, molrs.Atomistic)``; a bare :class:`molrs.Graph` by default);
``Assembler(library, GrowthPlacer()).assemble(sites)`` grows a site graph
without positions (e.g. ``CGSmilesIR(...).to_coarsegrain()``).
"""

from ._lib import (
    Assembler,
    AxisOrienter,
    CarbonTubeBuilder,
    GrapheneBuilder,
    GrowthPlacer,
    SitePlacer,
)

__all__ = [
    "Assembler",
    "AxisOrienter",
    "CarbonTubeBuilder",
    "GrapheneBuilder",
    "GrowthPlacer",
    "SitePlacer",
]
