"""The force-field IR as a protocol (``molrs::ff::ir``): register a new style
or a new category from Python, with nothing in molrs rebuilt.

The force-field IR adopts the LAMMPS standard; its *shape* is data. A
**category** names how many atoms a term has and which Frame block its terms
live in; a **style** names its ordered parameters, each with a dimension, and
its energy — as an expression, a vectorised Python kernel, or both. Anything
of that form registers into the process-wide registry every
:class:`~molrs.ff.potential.PotentialCompiler` reads, and then prices at both compile
doors and in MD exactly like a built-in. What does not conform is refused
with an :class:`IrError` subclass naming the offending item.

A new bond style, LAMMPS ``bond_style fene``, by its expression::

    from molrs.ff import ir

    class Fene(ir.StyleSpec):
        category = "bond"
        name = "fene"
        params = [ir.Param("k", "E/L^2"), ir.Param("r0", "L"),
                  ir.Param("epsilon", "E"), ir.Param("sigma", "L")]
        expression = ("-0.5*k*r0^2*log(1-(r/r0)^2)"
                      "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)")

    ff.def_style("bond", "fene").def_type("B-B", b, b, k=30.0, r0=1.5,
                                          epsilon=1.0, sigma=1.0)

The same style priced by numpy instead (one call per style per evaluation;
``q`` is the bond length of every term, each parameter an array beside it)::

    def fene(r, k, r0, epsilon, sigma):
        ...
        return e, de_dr

    ir.register_style("bond", "fene/np", params=Fene.params, kernel=fene)

Calling conventions (every tier): ``q`` is the category's coordinate — ``r``
(length), ``theta`` in [0, π] or ``phi`` in (−π, π] (radians); parameters
arrive **as stored** (angle values in degrees, the expression converts with
``0.017453292519943295``); a kernel returns the **unweighted** energy per term
and its derivative (or ``∂E/∂x`` for a compound kernel, not the force). The
pair weight, the cutoff and the chain rule onto Cartesian forces are the
form kernels'.

Refusals are :class:`IrError` (a ``ValueError``) subclasses of the same names
as the Rust variants: :class:`Sealed`, :class:`Conflict`,
:class:`UnboundVariable`, :class:`UnknownFunction`, :class:`KernelShape`,
:class:`NoKernel`, :class:`MissingParam`, :class:`BadValue`,
:class:`Derivative`, … A Python
kernel that raises during an evaluation, a compile or a registration surfaces
as :class:`KernelShape` with the original exception as ``__cause__``.
"""

from .._lib import ir as _ir
from ._style_spec import StyleSpec

IrError = _ir.IrError
UnknownCategory = _ir.UnknownCategory
BadName = _ir.BadName
Arity = _ir.Arity
BlockName = _ir.BlockName
ReservedParam = _ir.ReservedParam
DuplicateParam = _ir.DuplicateParam
Dim = _ir.Dim
Parse = _ir.Parse
UnboundVariable = _ir.UnboundVariable
UnknownFunction = _ir.UnknownFunction
FunctionArity = _ir.FunctionArity
Point = _ir.Point
CoordinateMismatch = _ir.CoordinateMismatch
Derivative = _ir.Derivative
Disagree = _ir.Disagree
Asymmetric = _ir.Asymmetric
Sealed = _ir.Sealed
Conflict = _ir.Conflict
NoKernel = _ir.NoKernel
NoMixing = _ir.NoMixing
MissingParam = _ir.MissingParam
BadValue = _ir.BadValue
KernelShape = _ir.KernelShape
NoEngineForm = _ir.NoEngineForm
FormConflict = _ir.FormConflict
NoForm = _ir.NoForm
OutOfImage = _ir.OutOfImage
Malformed = _ir.Malformed

Param = _ir.Param
StyleInfo = _ir.StyleInfo
CategoryInfo = _ir.CategoryInfo
register_category = _ir.register_category
register_style = _ir.register_style
register_engine_form = _ir.register_engine_form
unregister = _ir.unregister
styles = _ir.styles
categories = _ir.categories
evaluate = _ir.evaluate

StyleSpec.__module__ = __name__

__all__ = [
    "Arity",
    "Asymmetric",
    "BadName",
    "BlockName",
    "CategoryInfo",
    "Conflict",
    "CoordinateMismatch",
    "Derivative",
    "Dim",
    "Disagree",
    "DuplicateParam",
    "FormConflict",
    "FunctionArity",
    "IrError",
    "KernelShape",
    "Malformed",
    "MissingParam",
    "BadValue",
    "NoEngineForm",
    "NoForm",
    "NoKernel",
    "NoMixing",
    "OutOfImage",
    "Param",
    "Parse",
    "Point",
    "ReservedParam",
    "Sealed",
    "StyleInfo",
    "StyleSpec",
    "UnboundVariable",
    "UnknownCategory",
    "UnknownFunction",
    "categories",
    "evaluate",
    "register_category",
    "register_engine_form",
    "register_style",
    "styles",
    "unregister",
]
