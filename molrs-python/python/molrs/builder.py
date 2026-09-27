"""Structure builders — ``molrs::builder``.

Graphene sheets (:class:`GrapheneBuilder`) and single-wall carbon nanotubes
(:class:`CarbonTubeBuilder`), each building a fresh :class:`molrs.Frame`.

Trace assembly: ``Assembler(library, TracePlacer()).assemble(traces, names)``
places one template copy per trace point, joins each trace's units ``>`` to
``<``, and returns the world as one :class:`molrs.Fragment`.
"""

from ._lib import Assembler as Assembler
from ._lib import CarbonTubeBuilder as CarbonTubeBuilder
from ._lib import GrapheneBuilder as GrapheneBuilder
from ._lib import TracePlacer as TracePlacer

__all__ = ["Assembler", "CarbonTubeBuilder", "GrapheneBuilder", "TracePlacer"]
