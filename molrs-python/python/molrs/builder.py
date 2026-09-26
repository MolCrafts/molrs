"""Structure builders — ``molrs::builder``.

Graphene sheets, carbon nanotubes, and fragment assembly: a
:class:`FragLibrary` of templates, a :class:`Placer` (:class:`TracePlacer`),
an :class:`Orienter` (:class:`NullOrienter`, :class:`RandomOrienter`,
:class:`HintOrienter`), a :class:`Reacter` (:class:`PortReacter`), the
:class:`Assembler` that composes them, and the :class:`Finalizer` step.
``Placer``, ``Orienter`` and ``Reacter`` are subclassable in Python. The core
types they consume — :class:`molrs.Trace`, :class:`molrs.FragGraph`,
:class:`molrs.Mapping` — live at the top level.
"""

from ._lib import Assembler as Assembler
from ._lib import CarbonTubeBuilder as CarbonTubeBuilder
from ._lib import Finalizer as Finalizer
from ._lib import FragLibrary as FragLibrary
from ._lib import GrapheneBuilder as GrapheneBuilder
from ._lib import HintOrienter as HintOrienter
from ._lib import NullOrienter as NullOrienter
from ._lib import Orienter as Orienter
from ._lib import Placer as Placer
from ._lib import PortReacter as PortReacter
from ._lib import RandomOrienter as RandomOrienter
from ._lib import Reacter as Reacter
from ._lib import TracePlacer as TracePlacer

__all__ = [
    "Assembler",
    "CarbonTubeBuilder",
    "Finalizer",
    "FragLibrary",
    "GrapheneBuilder",
    "HintOrienter",
    "NullOrienter",
    "Orienter",
    "Placer",
    "PortReacter",
    "RandomOrienter",
    "Reacter",
    "TracePlacer",
]
