"""The force-field IR as a protocol (``molrs::ff::ir``): register a new style
or a new category from Python, with nothing in molrs rebuilt.

The force-field IR adopts the LAMMPS standard; its *shape* is data. A
**category** names how many atoms a term has and which Frame block its terms
live in; a **style** names its ordered parameters, each with a dimension, and
its energy — as an expression, a vectorised Python kernel, or both. Anything
of that form registers into the process-wide registry every
:class:`~molrs.ff.PotentialCompiler` reads, and then prices at both compile
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
generic kernels'.

Refusals are :class:`IrError` (a ``ValueError``) subclasses of the same names
as the Rust variants: :class:`Sealed`, :class:`Conflict`,
:class:`UnboundVariable`, :class:`UnknownFunction`, :class:`KernelShape`,
:class:`NoKernel`, :class:`MissingParam`, :class:`BadValue`,
:class:`Derivative`, … A Python
kernel that raises during an evaluation, a compile or a registration surfaces
as :class:`KernelShape` with the original exception as ``__cause__``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from .._lib import ir as _ir

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


class StyleSpec:
    """Declare a force-field IR style as a class; defining the subclass
    registers it (:func:`register_style`).

    Attributes
    ----------
    category : str
        The category (built-in, or from :func:`register_category`).
    name : str
        The style name.
    params : list of Param, or dict of {name: dim}
        The per-type parameters, in order.
    style_params : list of Param, or dict of {name: dim}, optional
        Style-level parameters (``cutoff``, ``mixing``, ``special`` keep their
        reserved meanings).
    expression : str, optional
        The energy as an expression.
    special : {"lj", "coul"}, optional
        A pair style's special-bonds class.
    samples : list of dict, optional
        Registration samples, ``{"q": (lo, hi), <param>: value}``.
    compound : bool
        ``kernel`` takes positions ``x`` (``(n, arity, 3)``), not ``q``.
    replace : bool
        Replace a style of the name registered at run time.
    lammps : {"positional", "positional:<name>"}, optional
        The style's LAMMPS form (:func:`register_style`'s ``lammps``).

    A ``kernel(self, q, **params) -> (e, de_dq)`` method (or
    ``kernel(self, x, **params) -> (e, grad)`` for a compound style) prices
    the style by Python; with an ``expression`` beside it the two must agree.
    Pass ``register=False`` in the class statement for an intermediate base
    that declares no style.

    Examples
    --------
    >>> import numpy as np
    >>> from molrs.ff import ir
    >>> class Quartic(ir.StyleSpec):
    ...     category = "bond"
    ...     name = "quartic/doc"
    ...     params = {"k": "E/L^4", "r0": "L"}
    ...     expression = "k*(r-r0)^4"
    ...     def kernel(self, r, k, r0):
    ...         d = r - r0
    ...         return k * d**4, 4 * k * d**3
    >>> e, de = Quartic.evaluate([1.5], k=2.0, r0=1.0)
    >>> float(e[0]), float(de[0])
    (0.125, 1.0)
    >>> Quartic.unregister()
    """

    category: ClassVar[str]
    name: ClassVar[str]
    params: ClassVar[Sequence[Any] | Mapping[str, str]] = ()
    style_params: ClassVar[Sequence[Any] | Mapping[str, str]] = ()
    expression: ClassVar[str | None] = None
    special: ClassVar[str | None] = None
    samples: ClassVar[Sequence[Mapping[str, Any]] | None] = None
    compound: ClassVar[bool] = False
    replace: ClassVar[bool] = False
    lammps: ClassVar[str | None] = None

    def __init_subclass__(cls, *, register: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not register:
            return
        missing = [
            a
            for a in ("category", "name")
            if not isinstance(getattr(cls, a, None), str)
        ]
        if missing:
            raise TypeError(
                f"{cls.__qualname__}: a StyleSpec declares {' and '.join(missing)} (a str)"
            )
        kernel: Callable[..., Any] | None = None
        if callable(getattr(cls, "kernel", None)):
            kernel = cls().kernel  # type: ignore[attr-defined]
        register_style(
            cls.category,
            cls.name,
            params=cls.params,
            style_params=cls.style_params,
            expression=cls.expression,
            kernel=kernel,
            compound=cls.compound,
            special=cls.special,
            samples=cls.samples,
            replace=cls.replace,
            lammps=cls.lammps,
        )

    @classmethod
    def evaluate(
        cls, q: Any = None, *, x: Any = None, **params: Any
    ) -> tuple[Any, Any]:
        """:func:`evaluate` of this style."""
        return evaluate(cls.category, cls.name, q, x=x, **params)

    @classmethod
    def unregister(cls) -> None:
        """:func:`unregister` this style."""
        unregister(cls.category, cls.name)


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
