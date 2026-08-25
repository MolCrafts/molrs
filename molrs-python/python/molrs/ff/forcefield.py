"""Ergonomic force-field layer over the Rust ``molrs.ForceField``.

``Style`` and ``Type`` (and their per-category subclasses) are **handle views**:
each holds the owning :class:`ForceField` plus the identifiers needed to address
one style or type, and every read/write routes through the single Rust-side
state. There is no parallel Python storage — mirroring how :mod:`molrs.frame`'s
``Frame``/``Block`` view the Rust column store, and how
:class:`molpy.core.entity.Entity` views a molrs world.

A type's matching label (the key a Frame's ``type`` column carries) is the
``name`` keyword when given, else the endpoint-derived composite (``"CT-CT"``) —
following the molpy convention where the typifier writes ``type.name`` into the
frame. Parameters flow as keyword args by convention: numeric ones (``k``,
``r0``, the numeric type ``id``) live in the float bag; string metadata
(``element``, …) is carried as string params. Both round-trip through ``params``.
"""

from __future__ import annotations

from typing import Any, TypeVar

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

# def_style returns a bound handle of the SAME Style subclass it was given, so
# `ff.def_style(AtomStyle(...)).def_type(...)` keeps the subclass-specific def_type.
_StyleT = TypeVar("_StyleT", bound="Style")


def _name_of(x: Any) -> str:
    """The atom-type name of ``x`` (a :class:`Type`/ref or a bare string)."""
    return x.name if hasattr(x, "name") else str(x)


def _numeric(params: dict[str, Any]) -> dict[str, float]:
    """Keep only the float-representable params the molrs ``Params`` bag holds."""
    out: dict[str, float] = {}
    for k, v in params.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


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

    def __init__(self, ff: "ForceField | None" = None, name: str = "") -> None:
        # ``ff is None`` is an *unbound* style marker (e.g. ``BondHarmonicStyle()``)
        # passed to :meth:`ForceField.def_style`, which binds and registers it.
        self._ff = ff
        self._name = name or self._name_default()

    def _name_default(self) -> str:
        return ""

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

    def _finish_type(
        self, default_label: str, name: str, params: dict[str, Any]
    ) -> Type:
        """Apply the optional ``name`` keyword as the type's matching label and
        persist any string params (e.g. ``element``).

        The matching label (the key a Frame's ``type`` column carries) is the
        ``name`` keyword when given, else the endpoint-derived ``default_label``.
        ``name`` is the molpy convention — the typifier writes ``type.name`` into
        the frame; numeric ``id`` flows through as an ordinary numeric param, and
        string metadata (``element``, …) is carried as string params.
        """
        label = name or default_label
        if label != default_label:
            _RsForceField.rename_type(
                self._ff, self._category, self._name, default_label, label
            )
        for k, v in params.items():
            if isinstance(v, str):
                self._ff.set_type_str_param(self._category, self._name, label, k, v)
        return self._type_cls(self._ff, self._name, label)

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

    def def_type(self, name: str, **params: Any) -> AtomType:
        self._ff.def_atomtype(self._name, name, _numeric(params))
        for k, v in params.items():
            if isinstance(v, str):
                self._ff.set_type_str_param(self._category, self._name, name, k, v)
        return AtomType(self._ff, self._name, name)


class BondStyle(Style):
    _category = "bond"
    _type_cls = BondType

    def def_type(self, itom: Any, jtom: Any, name: str = "", **params: Any) -> BondType:
        i, j = _name_of(itom), _name_of(jtom)
        self._ff.def_bondtype(self._name, i, j, _numeric(params))
        return self._finish_type(f"{i}-{j}", name, params)


class AngleStyle(Style):
    _category = "angle"
    _type_cls = AngleType

    def def_type(
        self, itom: Any, jtom: Any, ktom: Any, name: str = "", **params: Any
    ) -> AngleType:
        i, j, k = _name_of(itom), _name_of(jtom), _name_of(ktom)
        self._ff.def_angletype(self._name, i, j, k, _numeric(params))
        return self._finish_type(f"{i}-{j}-{k}", name, params)


class DihedralStyle(Style):
    _category = "dihedral"
    _type_cls = DihedralType

    def def_type(
        self, itom: Any, jtom: Any, ktom: Any, ltom: Any, name: str = "", **params: Any
    ) -> DihedralType:
        i, j, k, length = (
            _name_of(itom),
            _name_of(jtom),
            _name_of(ktom),
            _name_of(ltom),
        )
        self._ff.def_dihedraltype(self._name, i, j, k, length, _numeric(params))
        return self._finish_type(f"{i}-{j}-{k}-{length}", name, params)


class ImproperStyle(Style):
    _category = "improper"
    _type_cls = ImproperType

    def def_type(
        self, itom: Any, jtom: Any, ktom: Any, ltom: Any, name: str = "", **params: Any
    ) -> ImproperType:
        i, j, k, length = (
            _name_of(itom),
            _name_of(jtom),
            _name_of(ktom),
            _name_of(ltom),
        )
        self._ff.def_impropertype(self._name, i, j, k, length, _numeric(params))
        return self._finish_type(f"{i}-{j}-{k}-{length}", name, params)


class PairStyle(Style):
    _category = "pair"
    _type_cls = PairType

    def def_type(
        self, itom: Any, jtom: Any = None, name: str = "", **params: Any
    ) -> PairType:
        i = _name_of(itom)
        j = i if jtom is None else _name_of(jtom)
        self._ff.def_pairtype(self._name, i, j, _numeric(params))
        return self._finish_type(f"{i}-{j}" if i != j else i, name, params)


# ===================================================================
#        Named specialized styles/types (fixed kernel name)
# ===================================================================
# Thin subclasses that pin the kernel/style name, so a reader can write
# ``ff.def_style(BondHarmonicStyle())`` instead of ``ff.def_bondstyle("harmonic")``.
# The energy math lives in the molrs kernels keyed by these names; the combined
# lj/coul pair styles use their LAMMPS names for I/O round-trips (molrs evaluates
# the separable lj/cut + coul kernels).


class BondHarmonicStyle(BondStyle):
    def _name_default(self) -> str:
        return "harmonic"


class AngleHarmonicStyle(AngleStyle):
    def _name_default(self) -> str:
        return "harmonic"


class DihedralOPLSStyle(DihedralStyle):
    def _name_default(self) -> str:
        return "opls"


class PairCoulLongStyle(PairStyle):
    def _name_default(self) -> str:
        return "coul/long"


# ===================================================================
#                          ForceField
# ===================================================================

# Style subclass per molrs category + the chainable ``def_*style`` builder name.
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

    Subclasses the Rust :class:`molrs.ForceField` (inheriting ``def_type`` /
    ``types`` / ``to_potentials``) and adds ``def_*style`` factories that return
    chainable :class:`Style` handles plus style/type query helpers.
    """

    def __init__(self, name: str = "forcefield", units: str = "real") -> None:
        # The Rust ``__new__`` already built the inner force field from ``name``.
        self.units = units

    # ---- raw <-> Python conversion (so all FF-returning APIs yield this type) ----
    @classmethod
    def _from_raw(cls, raw: _RsForceField) -> "ForceField":
        """Re-wrap a bare Rust force field as a :class:`ForceField` by replaying
        its styles and types (used by readers / ``subset`` which return the core
        type).

        Numeric and string params are both preserved (``_replay_type``) so OPLS
        ``class_`` / ``type_`` / ``element`` metadata survives the wrap.
        """
        ff = cls(name=raw.name)
        for cat_name in raw.style_names():
            category, sname = cat_name.split(":", 1)
            if category == "pair":
                _RsForceField.def_pairstyle(
                    ff, sname, raw.style_params(category, sname)
                )
            else:
                ff._ensure_style(category, sname)
            for tname, params in raw.types(category, sname):
                endpoints = raw.type_endpoints(category, sname, tname)
                ff._replay_type(
                    category, sname, tname, params, endpoints=endpoints
                )
        return ff

    def subset(self, frame: Any) -> "ForceField":
        return ForceField._from_raw(_RsForceField.subset(self, frame))

    def map_type(self, frame: Any) -> Any:
        """Stamp ``type_id`` on Frame blocks from existing ``type`` labels.

        Frame is the source of truth: this method does **not** invent types.
        For each of ``atoms`` / ``bonds`` / ``angles`` / ``dihedrals`` /
        ``impropers``:

        * if ``type`` is present → write ``type_id`` (1-based; pure-integer
          labels use identity ``id = int(label)``, else dense ids in numeric-
          aware sort order of unique labels);
        * if only ``type_id`` is present → leave as-is;
        * if neither is present on ``atoms`` → ``ValueError`` (incomplete
          Frame). Connectivity blocks without either column are skipped.

        Returns:
            The same ``frame`` (mutated in place).
        """
        import numpy as np

        def _is_int_token(value: object) -> bool:
            s = str(value).strip()
            if not s:
                return False
            if s[0] in "+-":
                s = s[1:]
            return s.isdigit()

        def _sorted_names(names: list[str]) -> list[str]:
            if names and all(_is_int_token(n) for n in names):
                return sorted(names, key=lambda n: int(n))
            return sorted(names)

        def _name_to_id(names: list[str]) -> dict[str, int]:
            unique = list(dict.fromkeys(names))
            if unique and all(_is_int_token(n) for n in unique):
                return {n: int(n) for n in unique}
            ordered = _sorted_names(unique)
            return {n: i + 1 for i, n in enumerate(ordered)}

        atoms_ok = False
        for block_name in ("atoms", "bonds", "angles", "dihedrals", "impropers"):
            if block_name not in frame:
                continue
            block = frame[block_name]
            if getattr(block, "nrows", 0) == 0:
                continue
            has_type = "type" in block
            has_tid = "type_id" in block
            if not has_type and not has_tid:
                if block_name == "atoms":
                    raise ValueError(
                        "frame['atoms'] has neither 'type' nor 'type_id'; "
                        "assign types (e.g. typify) before ForceField.map_type"
                    )
                continue
            if has_type:
                types = [str(t) for t in list(block["type"])]
                mapping = _name_to_id(types)
                block["type_id"] = np.asarray(
                    [mapping[t] for t in types], dtype=np.uint32
                )
            if block_name == "atoms":
                atoms_ok = True
        if "atoms" in frame and frame["atoms"].nrows > 0 and not atoms_ok:
            # atoms present but skipped somehow
            if "type" not in frame["atoms"] and "type_id" not in frame["atoms"]:
                raise ValueError(
                    "frame['atoms'] has neither 'type' nor 'type_id'; "
                    "assign types before ForceField.map_type"
                )
        return frame

    # ---- chainable style factories (ensure-exists, return a handle) ----
    def def_atomstyle(self, name: str) -> AtomStyle:
        super().def_atomstyle(name)
        return AtomStyle(self, name)

    def def_bondstyle(self, name: str) -> BondStyle:
        super().def_bondstyle(name)
        return BondStyle(self, name)

    def def_anglestyle(self, name: str) -> AngleStyle:
        super().def_anglestyle(name)
        return AngleStyle(self, name)

    def def_dihedralstyle(self, name: str) -> DihedralStyle:
        super().def_dihedralstyle(name)
        return DihedralStyle(self, name)

    def def_improperstyle(self, name: str) -> ImproperStyle:
        super().def_improperstyle(name)
        return ImproperStyle(self, name)

    def def_pairstyle(
        self, name: str, params: dict[str, Any] | None = None, **kwparams: Any
    ) -> PairStyle:
        merged = dict(params or {})
        merged.update(kwparams)
        super().def_pairstyle(name, _numeric(merged))
        return PairStyle(self, name)

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
        """Types of a category (str) or by :class:`Type` subclass, across styles."""
        if isinstance(category_or_cls, str):
            cats = {category_or_cls}
            type_cls: type[Type] = Type
        else:
            type_cls = category_or_cls
            cats = {c for c, tc in _TYPE_CLASSES.items() if issubclass(tc, type_cls)}
        out: list[Type] = []
        for s in self._styles():
            if s.category in cats:
                out.extend(t for t in s.types if isinstance(t, type_cls))
        return out

    # ---- def_style(instance): register an (unbound) Style, return bound ----
    def _ensure_style(self, category: str, name: str) -> None:
        if category == "pair":
            _RsForceField.def_pairstyle(self, name, {})
        else:
            getattr(_RsForceField, f"def_{category}style")(self, name)

    def def_style(self, style: _StyleT) -> _StyleT:
        """Register ``style`` (an unbound :class:`Style`, e.g.
        ``BondHarmonicStyle()``) and return a bound handle of the same class."""
        self._ensure_style(style.category, style.name)
        bound = object.__new__(type(style))
        bound._ff = self
        bound._name = style.name
        return bound

    # ---- merge / rename / remove (molpy signatures: by Style subclass) ----
    def _replay_type(
        self,
        category: str,
        style: str,
        name: str,
        params: dict[str, Any],
        endpoints: list[str] | None = None,
    ) -> None:
        """Write one type, replacing any prior definition of the same name.

        Overlay merges (e.g. CL&P onto OPLS) otherwise accumulate duplicate
        names: ``def_type`` always appends, while ``set_type_str_param`` updates
        the *first* match — so a class-wildcard ``NA`` would swallow the
        overlay's string metadata and leave a second, charge-only ``NA``.
        Removing first keeps one coherent type with numeric + string params.

        Pair (and other multi-endpoint) types whose labels contain ``-`` must
        be rebuilt from *endpoints*, not by dash-splitting *name* — otherwise
        ``tip3p-O`` becomes a spurious tip3p×O cross-pair.
        """
        floats = {
            k: v
            for k, v in params.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        _RsForceField.remove_type(self, category, style, name)
        if category == "pair" and endpoints:
            if len(endpoints) >= 2 and endpoints[0] == endpoints[1]:
                _RsForceField.def_pairtype(self, style, endpoints[0], None, floats)
            elif len(endpoints) >= 2:
                _RsForceField.def_pairtype(
                    self, style, endpoints[0], endpoints[1], floats
                )
            else:
                _RsForceField.def_pairtype(self, style, endpoints[0], None, floats)
        elif category == "atom":
            _RsForceField.def_atomtype(self, style, name, floats)
        else:
            _RsForceField.def_type(self, category, style, name, floats)
        for k, v in params.items():
            if isinstance(v, str):
                self.set_type_str_param(category, style, name, k, v)

    def merge(self, other: "ForceField") -> "ForceField":
        """Merge ``other``'s styles and types into this force field (in place)."""
        for cat_name in other.style_names():
            category, sname = cat_name.split(":", 1)
            self._ensure_style(category, sname)
            for tname, params in other.types(category, sname):
                endpoints = other.type_endpoints(category, sname, tname)
                self._replay_type(
                    category, sname, tname, params, endpoints=endpoints
                )
        return self

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
    Rust. Result is pure molrs store units; ``units`` is set to ``\"real\"``.
    """
    ff = ForceField._from_raw(_rs_read_amber_prmtop_ff(path))
    ff.units = "real"
    return ff


def read_amber_prmtop_ff_str(text: str) -> ForceField:
    """Parse AMBER prmtop force-field tables from a string."""
    ff = ForceField._from_raw(_rs_read_amber_prmtop_ff_str(text))
    ff.units = "real"
    return ff


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
    *,
    precision: int = 6,
    skip_pair_style: bool = False,
    units: str = "real",
    atom_types: set[str] | None = None,
    bond_types: set[str] | None = None,
    angle_types: set[str] | None = None,
    dihedral_types: set[str] | None = None,
    improper_types: set[str] | None = None,
    type_ids: dict[str, int] | None = None,
) -> None:
    """Write a force field to a LAMMPS ``*.ff`` include (AMBER/GAFF flavour).

    Inverse of :func:`read_lammps_forcefield`: molrs store → LAMMPS file units
    (``K = k/2``, angles in degrees). Energy/length for ``metal``/``lj`` go
    through the lj reduced hub in native Rust; this is a thin façade.

    Args:
        units: LAMMPS ``units`` style for the written file (``real``, ``metal``,
            ``lj``). Default ``real``.
        type_ids: Optional name→id map (unused for ``*.ff`` command form).
    """
    _rs_write_lammps_forcefield(
        path,
        forcefield,
        precision=precision,
        skip_pair_style=skip_pair_style,
        units=units,
        atom_types=atom_types,
        bond_types=bond_types,
        angle_types=angle_types,
        dihedral_types=dihedral_types,
        improper_types=improper_types,
        type_ids=type_ids,
    )


def write_lammps_forcefield_str(
    forcefield: ForceField,
    *,
    precision: int = 6,
    skip_pair_style: bool = False,
    units: str = "real",
    atom_types: set[str] | None = None,
    bond_types: set[str] | None = None,
    angle_types: set[str] | None = None,
    dihedral_types: set[str] | None = None,
    improper_types: set[str] | None = None,
    type_ids: dict[str, int] | None = None,
) -> str:
    """Serialize a force field to a LAMMPS ``*.ff`` include string."""
    return _rs_write_lammps_forcefield_str(
        forcefield,
        precision=precision,
        skip_pair_style=skip_pair_style,
        units=units,
        atom_types=atom_types,
        bond_types=bond_types,
        angle_types=angle_types,
        dihedral_types=dihedral_types,
        improper_types=improper_types,
        type_ids=type_ids,
    )


def write_lammps_data_coeffs(
    forcefield: ForceField,
    *,
    precision: int = 6,
    units: str = "real",
    atom_types: set[str] | None = None,
    bond_types: set[str] | None = None,
    angle_types: set[str] | None = None,
    dihedral_types: set[str] | None = None,
    improper_types: set[str] | None = None,
    type_ids: dict[str, int] | None = None,
) -> str:
    """Serialize a force field to LAMMPS data-file ``* Coeffs`` section text.

    Same form map / units conversion as :func:`write_lammps_forcefield`, but
    emits ``Pair Coeffs`` / ``Bond Coeffs`` / … with integer type ids. Pass
    ``type_ids`` from the Frame (via :meth:`ForceField.map_type` + inventory)
    when type names are non-integer labels.
    """
    return _rs_write_lammps_data_coeffs(
        forcefield,
        precision=precision,
        units=units,
        atom_types=atom_types,
        bond_types=bond_types,
        angle_types=angle_types,
        dihedral_types=dihedral_types,
        improper_types=improper_types,
        type_ids=type_ids,
    )
