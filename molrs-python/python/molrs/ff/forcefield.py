"""Ergonomic force-field layer over the Rust ``molrs.ForceField``.

``Style`` and ``Type`` (and their per-category subclasses) are **handle views**:
each holds the owning :class:`ForceField` plus the identifiers needed to address
one style or type, and every read/write routes through the single Rust-side
state. There is no parallel Python storage — mirroring how :mod:`molrs.frame`'s
``Frame``/``Block`` view the Rust column store, and how
:class:`molpy.core.entity.Entity` views a molrs world.

A force field is built through three primitives, spelled as in Rust:
``ForceField.def_style(category, name, params=None)`` returns the category's
style handle, and ``Style.def_type(name, params=None)`` (endpoints parsed from
the name, ``"CT-CT"``) or ``Style.def_type_at(name, endpoints, params=None)``
(endpoints given, for names outside that grammar) define a type and return the
same handle for chaining. A type's name is the key a Frame's ``type`` column
carries. ``params`` is a dict: numbers (``k``, ``r0``, the numeric type ``id``)
go to the float bag, strings (``element``, ``mixing``, …) to the string params.
Both round-trip through ``params``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._lib import ForceField as _RsForceField
from .._lib import read_forcefield_xml as _rs_read_forcefield_xml
from .._lib import read_forcefield_xml_str as _rs_read_forcefield_xml_str
from .._lib import read_opls_xml as _rs_read_opls_xml
from .._lib import read_opls_xml_str as _rs_read_opls_xml_str
from .._lib import read_lammps_forcefield as _rs_read_lammps_forcefield
from .._lib import read_lammps_forcefield_str as _rs_read_lammps_forcefield_str
from .._lib import read_lammps_data_coeffs as _rs_read_lammps_data_coeffs
from .._lib import write_lammps_forcefield as _rs_write_lammps_forcefield
from .._lib import write_lammps_forcefield_str as _rs_write_lammps_forcefield_str
from .._lib import write_lammps_data_coeffs as _rs_write_lammps_data_coeffs
from .._lib import read_amber_prmtop_ff as _rs_read_amber_prmtop_ff
from .._lib import read_amber_prmtop_ff_str as _rs_read_amber_prmtop_ff_str
from .._lib import read_gromacs_top_ff as _rs_read_gromacs_top_ff
from .._lib import read_gromacs_top_ff_str as _rs_read_gromacs_top_ff_str
from .._lib import write_gromacs_top_ff as _rs_write_gromacs_top_ff
from .._lib import write_gromacs_top_ff_str as _rs_write_gromacs_top_ff_str
from .._lib import write_forcefield_xml as _rs_write_forcefield_xml
from .._lib import write_forcefield_xml_str as _rs_write_forcefield_xml_str

if TYPE_CHECKING:
    from ..frame import Frame


def _name_of(x: Any) -> str:
    """The atom-type name of ``x`` (a :class:`Type`/ref or a bare string)."""
    return x.name if hasattr(x, "name") else str(x)


# ===================================================================
#                          Parameters
# ===================================================================


class Parameters:
    """The parameter view of a :class:`Type` — keyword access plus the
    ``.kwargs`` mapping consumers read. The model is keyword-only, so ``.args``
    is always empty.
    """

    def __init__(self, mapping: dict[str, Any]) -> None:
        self._d = mapping

    @property
    def kwargs(self) -> dict[str, Any]:
        return self._d

    @property
    def args(self) -> list[Any]:
        return []

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def __iter__(self) -> Any:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def keys(self) -> Any:
        return self._d.keys()

    def values(self) -> Any:
        return self._d.values()

    def items(self) -> Any:
        return self._d.items()

    def __repr__(self) -> str:
        return f"Parameters(kwargs={self._d}, args=[])"


# ===================================================================
#                          Type handle views
# ===================================================================


class Type:
    """Handle view of one force-field type over a :class:`ForceField`."""

    _category: str = ""

    def __init__(self, ff: "ForceField", style: str, name: str) -> None:
        self._ff = ff
        self._style = style
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def params(self) -> Parameters:
        for nm, p in self._ff.types(self._category, self._style):
            if nm == self._name:
                return Parameters(p)
        return Parameters({})

    def __getitem__(self, key: str) -> Any:
        return self.params.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.params

    def __setitem__(self, key: str, value: float) -> None:
        self._ff.set_type_param(
            self._category, self._style, self._name, key, float(value)
        )

    def keys(self) -> Any:
        return self.params.keys()

    def items(self) -> Any:
        return self.params.items()

    @property
    def endpoints(self) -> tuple["AtomType", ...]:
        eps = self._ff.type_endpoints(self._category, self._style, self._name) or []
        return tuple(AtomType(self._ff, None, n) for n in eps)

    def __hash__(self) -> int:
        return hash((self._category, self._name))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Type)
            and self._category == other._category
            and self._name == other._name
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self._name}>"


class AtomType(Type):
    _category = "atom"

    @property
    def params(self) -> Parameters:
        # An endpoint ref (``_style is None``) carries only its name.
        if self._style is None:
            return Parameters({})
        return super().params


class BondType(Type):
    _category = "bond"

    @property
    def itom(self) -> AtomType:
        return self.endpoints[0]

    @property
    def jtom(self) -> AtomType:
        return self.endpoints[1]

    def matches(self, at1: Any, at2: Any) -> bool:
        i, j = (e.name for e in self.endpoints)
        a, b = _name_of(at1), _name_of(at2)
        return (i == a and j == b) or (i == b and j == a)


class AngleType(Type):
    _category = "angle"

    @property
    def itom(self) -> AtomType:
        return self.endpoints[0]

    @property
    def jtom(self) -> AtomType:
        return self.endpoints[1]

    @property
    def ktom(self) -> AtomType:
        return self.endpoints[2]

    def matches(self, at1: Any, at2: Any, at3: Any) -> bool:
        i, j, k = (e.name for e in self.endpoints)
        a, b, c = _name_of(at1), _name_of(at2), _name_of(at3)
        # central atom fixed; endpoints may reverse
        return j == b and ((i == a and k == c) or (i == c and k == a))


class DihedralType(Type):
    _category = "dihedral"

    @property
    def itom(self) -> AtomType:
        return self.endpoints[0]

    @property
    def jtom(self) -> AtomType:
        return self.endpoints[1]

    @property
    def ktom(self) -> AtomType:
        return self.endpoints[2]

    @property
    def ltom(self) -> AtomType:
        return self.endpoints[3]

    def matches(self, at1: Any, at2: Any, at3: Any, at4: Any) -> bool:
        i, j, k, length = (e.name for e in self.endpoints)
        a, b, c, d = _name_of(at1), _name_of(at2), _name_of(at3), _name_of(at4)
        fwd = i == a and j == b and k == c and length == d
        rev = i == d and j == c and k == b and length == a
        return fwd or rev


class ImproperType(Type):
    _category = "improper"

    @property
    def itom(self) -> AtomType:
        return self.endpoints[0]

    @property
    def jtom(self) -> AtomType:
        return self.endpoints[1]

    @property
    def ktom(self) -> AtomType:
        return self.endpoints[2]

    @property
    def ltom(self) -> AtomType:
        return self.endpoints[3]

    def matches(self, at1: Any, at2: Any, at3: Any, at4: Any) -> bool:
        i, j, k, length = (e.name for e in self.endpoints)
        return (
            i == _name_of(at1)
            and j == _name_of(at2)
            and k == _name_of(at3)
            and length == _name_of(at4)
        )


class PairType(Type):
    _category = "pair"

    @property
    def itom(self) -> AtomType:
        return self.endpoints[0]

    @property
    def jtom(self) -> AtomType:
        return self.endpoints[1]

    def matches(self, at1: Any, at2: Any = None) -> bool:
        i, j = (e.name for e in self.endpoints)
        a = _name_of(at1)
        b = a if at2 is None else _name_of(at2)
        return (i == a and j == b) or (i == b and j == a)


# ===================================================================
#                          Style handle views
# ===================================================================


class Style:
    """Handle view of one style over a :class:`ForceField`."""

    _category: str = ""
    _type_cls: type[Type] = Type

    def __init__(self, ff: "ForceField", name: str) -> None:
        self._ff = ff
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def types(self) -> list[Type]:
        return [
            self._type_cls(self._ff, self._name, nm)
            for nm, _ in self._ff.types(self._category, self._name)
        ]

    def get_types(self, type_cls: type[Type] = Type) -> list[Type]:
        return [t for t in self.types if isinstance(t, type_cls)]

    def get_type_by_name(self, name: str, type_cls: type[Type] = Type) -> Type | None:
        for t in self.types:
            if t.name == name and isinstance(t, type_cls):
                return t
        return None

    def def_type(self, name: str, params: dict[str, Any] | None = None) -> Style:
        """Define a type whose endpoints are parsed from ``name`` (``"CT-OH"``;
        ``"A::B"`` when a label contains ``-``) and return this style for
        chaining. A malformed name raises ``ValueError``."""
        self._ff._def_type(self._category, self._name, name, params)
        return self

    def def_type_at(
        self,
        name: str,
        endpoints: list[str],
        params: dict[str, Any] | None = None,
    ) -> Style:
        """Define a type named ``name`` with the given ``endpoints`` (for names
        outside the endpoint grammar, such as MMFF's ``0_1_5``) and return this
        style for chaining. An endpoint count that does not match the category
        raises ``ValueError``."""
        self._ff._def_type_at(self._category, self._name, name, list(endpoints), params)
        return self

    def __hash__(self) -> int:
        return hash((self._category, self._name))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Style)
            and self._category == other._category
            and self._name == other._name
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self._name}>"


class AtomStyle(Style):
    _category = "atom"
    _type_cls = AtomType


class BondStyle(Style):
    _category = "bond"
    _type_cls = BondType


class AngleStyle(Style):
    _category = "angle"
    _type_cls = AngleType


class DihedralStyle(Style):
    _category = "dihedral"
    _type_cls = DihedralType


class ImproperStyle(Style):
    _category = "improper"
    _type_cls = ImproperType


class PairStyle(Style):
    _category = "pair"
    _type_cls = PairType


# ===================================================================
#                          ForceField
# ===================================================================

# Style handle class per molrs category, returned by ``ForceField.def_style``.
_STYLE_CLASSES: dict[str, type[Style]] = {
    "atom": AtomStyle,
    "bond": BondStyle,
    "angle": AngleStyle,
    "dihedral": DihedralStyle,
    "improper": ImproperStyle,
    "pair": PairStyle,
}
_TYPE_CLASSES: dict[str, type[Type]] = {
    "atom": AtomType,
    "bond": BondType,
    "angle": AngleType,
    "dihedral": DihedralType,
    "improper": ImproperType,
    "pair": PairType,
}


class ForceField(_RsForceField):
    """A molrs force field with the chainable, object-style builder layer.

    Subclasses the Rust :class:`molrs.ForceField` (inheriting ``types`` /
    ``style_names`` / …); :meth:`def_style` returns a chainable
    :class:`Style` handle whose ``def_type`` / ``def_type_at`` define types.
    Style/type query helpers sit on top.
    """

    # ---- raw <-> Python conversion (so all FF-returning APIs yield this type) ----
    @classmethod
    def _from_raw(cls, raw: _RsForceField) -> "ForceField":
        """Re-wrap a bare Rust force field (what the readers return) as a
        :class:`ForceField`: an exact copy, through the Rust ``merge``.

        The new force field declares nothing, so it adopts ``raw``'s declared
        ``units`` and ``special_bonds``; every style keeps its full params
        (``cutoff``, ``mixing``, …) and every type its endpoints and params.
        """
        return cls(raw.name).merge(raw)

    # ---- the style primitive (types are defined on the returned handle) ----
    def def_style(
        self, category: str, name: str, params: dict[str, Any] | None = None
    ) -> Style:
        """Define the ``category`` style ``name`` (or keep the existing one) and
        return its handle (:class:`AtomStyle` … :class:`PairStyle`). ``params``
        (numbers and strings, e.g. ``{"cutoff": 10.0, "mixing": "geometric"}``)
        apply when the style is new. An unknown category raises ``ValueError``.
        """
        super().def_style(category, name, params)
        return _STYLE_CLASSES[category](self, name)

    # ---- style / type queries ----
    def _styles(self) -> list[Style]:
        out: list[Style] = []
        for cat_name in self.style_names():
            category, name = cat_name.split(":", 1)
            cls = _STYLE_CLASSES.get(category)
            if cls is not None:
                out.append(cls(self, name))
        return out

    @property
    def styles(self) -> list[Style]:
        return self._styles()

    def get_style(self, category: str, name: str) -> Style | None:
        cls = _STYLE_CLASSES.get(category)
        if cls is None:
            return None
        for cat_name in self.style_names():
            cat, nm = cat_name.split(":", 1)
            if cat == category and nm == name:
                return cls(self, name)
        return None

    def get_styles(self, category_or_cls: Any) -> list[Style]:
        """Styles of a category (str) or by :class:`Style` subclass."""
        if isinstance(category_or_cls, str):
            return [s for s in self._styles() if s.category == category_or_cls]
        return [s for s in self._styles() if isinstance(s, category_or_cls)]

    def get_types(self, category_or_cls: Any) -> list[Type]:
        """Types in a category.

        Pass a category string (``"angle"``), a :class:`Type` subclass
        (``AngleType``), or a :class:`Style` subclass (``AngleStyle``). A
        style class selects that category's types — not an empty list.
        """
        if isinstance(category_or_cls, str):
            cats = {category_or_cls}
            type_cls: type[Type] = Type
        elif isinstance(category_or_cls, type) and issubclass(category_or_cls, Style):
            cats = {
                c
                for c, sc in _STYLE_CLASSES.items()
                if issubclass(category_or_cls, sc) or issubclass(sc, category_or_cls)
            }
            type_cls = (
                _TYPE_CLASSES[next(iter(cats))] if len(cats) == 1 else Type
            )
        else:
            type_cls = category_or_cls
            cats = {c for c, tc in _TYPE_CLASSES.items() if issubclass(tc, type_cls)}
        out: list[Type] = []
        for s in self._styles():
            if s.category in cats:
                out.extend(t for t in s.types if isinstance(t, type_cls))
        return out

    # ---- rename / remove (molpy signatures: by Style subclass) ----
    def rename_type(self, style_cls: Any, old: str, new: str) -> int:
        """Rename type ``old`` -> ``new`` across all styles of ``style_cls``'s
        category (molpy signature)."""
        category = style_cls._category
        n = 0
        for s in self.get_styles(category):
            n += _RsForceField.rename_type(self, category, s.name, old, new)
        return n

    def remove_type(self, style_cls: Any, name: str) -> int:
        category = style_cls._category
        n = 0
        for s in self.get_styles(category):
            n += _RsForceField.remove_type(self, category, s.name, name)
        return n

    def remove_style(self, style_cls: Any, name: str) -> bool:
        return _RsForceField.remove_style(self, style_cls._category, name)


# ---- XML readers re-wrapped to yield the Python ForceField ----


def read_forcefield_xml(path: str) -> ForceField:
    return ForceField._from_raw(_rs_read_forcefield_xml(path))


def read_forcefield_xml_str(xml: str) -> ForceField:
    return ForceField._from_raw(_rs_read_forcefield_xml_str(xml))


def read_opls_xml(path: str) -> ForceField:
    return ForceField._from_raw(_rs_read_opls_xml(path))


def read_opls_xml_str(xml: str) -> ForceField:
    return ForceField._from_raw(_rs_read_opls_xml_str(xml))


def read_lammps_forcefield(path: str) -> ForceField:
    return ForceField._from_raw(_rs_read_lammps_forcefield(path))


def read_lammps_forcefield_str(text: str) -> ForceField:
    return ForceField._from_raw(_rs_read_lammps_forcefield_str(text))


def read_amber_prmtop_ff(path: str) -> ForceField:
    """Read AMBER prmtop force-field tables into a :class:`ForceField`.

    Structure/connectivity is :func:`molrs.io.read_amber_prmtop`. Harmonic
    form map (``k = 2·K``), Fourier dihedrals, and LJ A/B → σ/ε run in native
    Rust. Result is pure molrs store units; the reader declares ``units``
    ``"real"``.
    """
    return ForceField._from_raw(_rs_read_amber_prmtop_ff(path))


def read_amber_prmtop_ff_str(text: str) -> ForceField:
    """Parse AMBER prmtop force-field tables from a string."""
    return ForceField._from_raw(_rs_read_amber_prmtop_ff_str(text))


def read_gromacs_top_ff(path: str, *, include: bool = False) -> ForceField:
    """Read a GROMACS ``.top`` / ``.itp`` into a :class:`ForceField`.

    Bonded parameters (when present) are converted from GROMACS units to molrs
    store units at this boundary. ``include`` controls ``#include`` expansion.
    """
    return ForceField._from_raw(_rs_read_gromacs_top_ff(path, include=include))


def read_gromacs_top_ff_str(text: str, *, include: bool = False) -> ForceField:
    """Parse GROMACS topology force-field tables from a string."""
    return ForceField._from_raw(_rs_read_gromacs_top_ff_str(text, include=include))


def write_gromacs_top_ff(
    path: str, forcefield: ForceField, *, precision: int = 6
) -> None:
    """Write a ForceField to GROMACS ``.top``/``.itp`` tables (inverse unit map)."""
    _rs_write_gromacs_top_ff(path, forcefield, precision=precision)


def write_gromacs_top_ff_str(forcefield: ForceField, *, precision: int = 6) -> str:
    """Serialize a ForceField to a GROMACS topology force-field string."""
    return _rs_write_gromacs_top_ff_str(forcefield, precision=precision)


def write_forcefield_xml(
    path: str, forcefield: ForceField, *, precision: int = 6
) -> None:
    """Write a ForceField to OpenMM-style XML."""
    _rs_write_forcefield_xml(path, forcefield, precision=precision)


def write_forcefield_xml_str(forcefield: ForceField, *, precision: int = 6) -> str:
    """Serialize a ForceField to OpenMM-style XML string."""
    return _rs_write_forcefield_xml_str(forcefield, precision=precision)


def read_lammps_data_coeffs(
    coeffs_text: str,
    *,
    units: str = "real",
    atom_labels: dict[int, str] | None = None,
    bond_labels: dict[int, str] | None = None,
    angle_labels: dict[int, str] | None = None,
    dihedral_labels: dict[int, str] | None = None,
    improper_labels: dict[int, str] | None = None,
) -> ForceField:
    """Parse data-file ``* Coeffs`` sections into a :class:`ForceField`.

    Form map and units conversion run in native Rust. Style defaults are
    harmonic / ``lj/cut`` when the fragment has no style lines.
    """
    return ForceField._from_raw(
        _rs_read_lammps_data_coeffs(
            coeffs_text,
            units=units,
            atom_labels=atom_labels,
            bond_labels=bond_labels,
            angle_labels=angle_labels,
            dihedral_labels=dihedral_labels,
            improper_labels=improper_labels,
        )
    )


def write_lammps_forcefield(
    path: str,
    forcefield: ForceField,
    frame: Frame,
    *,
    precision: int = 6,
    skip_pair_style: bool = False,
    skip_units: bool = False,
    units: str = "real",
) -> None:
    """Write the coefficients ``frame`` uses to a LAMMPS ``*.ff`` include.

    Coefficient writing is keyed by the system's type labels: every
    ``atoms`` / ``bonds`` / ``angles`` / ``dihedrals`` / ``impropers`` label of
    ``frame`` is looked up in ``forcefield`` (bond, angle and dihedral labels in
    either orientation, impropers exactly) and written in label id order;
    force-field types no label uses are not written.

    Inverse of :func:`read_lammps_forcefield`: molrs store → LAMMPS file units
    (``K = k/2``, angles in degrees). Energy/length for ``metal``/``lj`` go
    through the lj reduced hub in native Rust; this is a thin façade.

    Args:
        path: Destination path for the include.
        forcefield: Force field in molrs store units.
        frame: The system whose type labels select the coefficients.
        precision: Decimal places for floating coefficients.
        skip_pair_style: Omit ``pair_style`` and ``special_bonds``.
        skip_units: Omit the ``units`` line.
        units: LAMMPS ``units`` style for the written file (``real``, ``metal``,
            ``lj``). Default ``real``.

    Raises:
        ValueError: A label of ``frame`` has no type in ``forcefield`` (the
            message names the block and the label), or an unsupported style
            holds a used type.
    """
    _rs_write_lammps_forcefield(
        path,
        forcefield,
        frame,
        precision=precision,
        skip_pair_style=skip_pair_style,
        skip_units=skip_units,
        units=units,
    )


def write_lammps_forcefield_str(
    forcefield: ForceField,
    frame: Frame,
    *,
    precision: int = 6,
    skip_pair_style: bool = False,
    skip_units: bool = False,
    units: str = "real",
) -> str:
    """Serialize the coefficients ``frame`` uses to a LAMMPS ``*.ff`` string.

    Same labels, format and errors as :func:`write_lammps_forcefield`.
    """
    return _rs_write_lammps_forcefield_str(
        forcefield,
        frame,
        precision=precision,
        skip_pair_style=skip_pair_style,
        skip_units=skip_units,
        units=units,
    )


def write_lammps_data_coeffs(
    forcefield: ForceField,
    frame: Frame,
    *,
    precision: int = 6,
    units: str = "real",
) -> str:
    """Serialize the coefficients ``frame`` uses to data-file ``* Coeffs`` text.

    Same labels, form map and units conversion as
    :func:`write_lammps_forcefield`, but emits ``Pair Coeffs`` / ``Bond Coeffs``
    / … whose integer ids are ``frame``'s type-label ids. ``Pair Coeffs`` holds
    self pairs only; a used explicit cross pair raises ``ValueError``.
    """
    return _rs_write_lammps_data_coeffs(
        forcefield,
        frame,
        precision=precision,
        units=units,
    )
