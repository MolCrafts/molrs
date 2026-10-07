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

    class Fene(ir.StyleDeclaration):
        category = "bond"
        name = "fene"
        params = [ir.ParamSpec("k", "E/L^2"), ir.ParamSpec("r0", "L"),
                  ir.ParamSpec("epsilon", "E"), ir.ParamSpec("sigma", "L")]
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

Refusals are :class:`IrError` (a ``ValueError``) subclasses named after the
Rust variants with an ``Error`` suffix (``IrError::Dimension`` is
:class:`DimensionError`): :class:`SealedError`, :class:`ConflictError`,
:class:`UnboundVariableError`, :class:`UnknownFunctionError`, :class:`KernelShapeError`,
:class:`NoKernelError`, :class:`MissingParamError`, :class:`BadValueError`,
:class:`DerivativeError`, … A Python
kernel that raises during an evaluation, a compile or a registration surfaces
as :class:`KernelShapeError` with the original exception as ``__cause__``.
"""

from .._native import ir as _ir
from ._style_declaration import StyleDeclaration

IrError = _ir.IrError
UnknownCategoryError = _ir.UnknownCategoryError
BadNameError = _ir.BadNameError
ArityError = _ir.ArityError
BlockNameError = _ir.BlockNameError
ReservedParamError = _ir.ReservedParamError
DuplicateParamError = _ir.DuplicateParamError
DimensionError = _ir.DimensionError
ParseError = _ir.ParseError
UnboundVariableError = _ir.UnboundVariableError
UnknownFunctionError = _ir.UnknownFunctionError
FunctionArityError = _ir.FunctionArityError
PointError = _ir.PointError
CoordinateMismatchError = _ir.CoordinateMismatchError
DerivativeError = _ir.DerivativeError
DisagreeError = _ir.DisagreeError
AsymmetricError = _ir.AsymmetricError
SealedError = _ir.SealedError
ConflictError = _ir.ConflictError
NoKernelError = _ir.NoKernelError
NoMixingError = _ir.NoMixingError
MissingParamError = _ir.MissingParamError
BadValueError = _ir.BadValueError
KernelShapeError = _ir.KernelShapeError
NoEngineFormError = _ir.NoEngineFormError
FormConflictError = _ir.FormConflictError
NoFormError = _ir.NoFormError
OutOfImageError = _ir.OutOfImageError
MalformedError = _ir.MalformedError

ParamSpec = _ir.ParamSpec
StyleSpec = _ir.StyleSpec
CategorySpec = _ir.CategorySpec
register_category = _ir.register_category
register_style = _ir.register_style
register_engine_form = _ir.register_engine_form
unregister_style = _ir.unregister_style
styles = _ir.styles
categories = _ir.categories
evaluate = _ir.evaluate

StyleDeclaration.__module__ = __name__

__all__ = [
    "ArityError",
    "AsymmetricError",
    "BadNameError",
    "BadValueError",
    "BlockNameError",
    "CategorySpec",
    "ConflictError",
    "CoordinateMismatchError",
    "DerivativeError",
    "DimensionError",
    "DisagreeError",
    "DuplicateParamError",
    "FormConflictError",
    "FunctionArityError",
    "IrError",
    "KernelShapeError",
    "MalformedError",
    "MissingParamError",
    "NoEngineFormError",
    "NoFormError",
    "NoKernelError",
    "NoMixingError",
    "OutOfImageError",
    "ParamSpec",
    "ParseError",
    "PointError",
    "ReservedParamError",
    "SealedError",
    "StyleDeclaration",
    "StyleSpec",
    "UnboundVariableError",
    "UnknownCategoryError",
    "UnknownFunctionError",
    "categories",
    "evaluate",
    "register_category",
    "register_engine_form",
    "register_style",
    "styles",
    "unregister_style",
]
