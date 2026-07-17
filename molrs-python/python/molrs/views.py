"""Live object views over a :mod:`molrs` graph.

There is exactly one object state: every :class:`NodeRef` and
:class:`RelationRef` is bound to a live graph handle from construction.  Nodes
and relations are created by a graph factory, never as detached Python objects.
The graph remains the sole owner of all data.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any, Protocol, overload
from weakref import WeakValueDictionary

import numpy as np

from . import fields
from .fields import FieldSpec
from .molrs import Atomistic as _RsAtomistic
from .molrs import CoarseGrain as _RsCoarseGrain


class RefLike(Protocol):
    data: MutableMapping[str, Any]
    handle: int
    world: Any

    def __getitem__(self, key: str | FieldSpec) -> Any: ...
    def __setitem__(self, key: str | FieldSpec, value: Any) -> None: ...
    def get(self, key: str | FieldSpec, default: Any = None) -> Any: ...


class _DictView:
    __slots__ = ()

    data: MutableMapping[str, Any]

    @staticmethod
    def _key(key: Any) -> Any:
        return key.key if isinstance(key, FieldSpec) else key

    def __getitem__(self, key: Any) -> Any:
        return self.data[self._key(key)]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[self._key(key)] = value

    def __delitem__(self, key: Any) -> None:
        del self.data[self._key(key)]

    def __contains__(self, key: object) -> bool:
        return self._key(key) in self.data

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def keys(self) -> Any:
        return self.data.keys()

    def values(self) -> Any:
        return self.data.values()

    def items(self) -> Any:
        return self.data.items()

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self.data[self._key(key)]
        except KeyError:
            return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.data.update(*args, **kwargs)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self.data:
            self.data[key] = default
        return self.data[key]

    def pop(self, key: str, *default: Any) -> Any:
        return self.data.pop(key, *default)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


_XYZ_KEYS = (fields.POS_X.key, fields.POS_Y.key, fields.POS_Z.key)
_XYZ = fields.XYZ.key


class _NodeData(MutableMapping[str, Any]):
    __slots__ = ("world", "handle")

    def __init__(self, world: Any, handle: int) -> None:
        self.world = world
        self.handle = handle

    def _keys(self) -> list[str]:
        return list(self.world.node_keys(self.handle))

    def _has(self, key: str) -> bool:
        return self.world.has(self.handle, key)

    def _get(self, key: str) -> Any:
        return self.world.get(self.handle, key)

    def _set(self, key: str, value: Any) -> None:
        self.world.set(self.handle, key, value)

    def _delete(self, key: str) -> None:
        self.world.delete(self.handle, key)

    def __getitem__(self, key: str) -> Any:
        if key == _XYZ:
            return [self._get(k) for k in _XYZ_KEYS]
        if self._has(key):
            return self._get(key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == _XYZ:
            if len(value) != 3:
                raise ValueError(f"xyz must be a 3-vector, got {value!r}")
            for component, number in zip(_XYZ_KEYS, value):
                self._set(component, float(number))
            return
        if value is None:
            if self._has(key):
                self._delete(key)
            return
        self._set(key, value)

    def __delitem__(self, key: str) -> None:
        if not self._has(key):
            raise KeyError(key)
        self._delete(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys())

    def __len__(self) -> int:
        return len(self._keys())

    def __contains__(self, key: object) -> bool:
        if key == _XYZ:
            return all(self._has(k) for k in _XYZ_KEYS)
        return isinstance(key, str) and self._has(key)

    def __repr__(self) -> str:
        return repr(dict(self))

    def copy(self) -> dict[str, Any]:
        return dict(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        from copy import deepcopy

        return {deepcopy(k, memo): deepcopy(v, memo) for k, v in self.items()}


class _RelationData(_NodeData):
    __slots__ = ("kind",)

    def __init__(self, world: Any, kind: str, handle: int) -> None:
        super().__init__(world, handle)
        self.kind = kind

    def _keys(self) -> list[str]:
        return list(self.world.relation_keys(self.kind, self.handle))

    def _has(self, key: str) -> bool:
        return self.world.get_relation_prop(self.kind, self.handle, key) is not None

    def _get(self, key: str) -> Any:
        return self.world.get_relation_prop(self.kind, self.handle, key)

    def _set(self, key: str, value: Any) -> None:
        self.world.set_relation_prop(self.kind, self.handle, key, value)

    def _delete(self, key: str) -> None:
        self.world.delete_relation_prop(self.kind, self.handle, key)


class NodeRef(_DictView):
    """A live ``(world, handle)`` node reference.

    Direct construction only wraps an existing handle.  Use the owning graph's
    ``def_*`` factory to create a new node.
    """

    __slots__ = ("world", "handle", "data", "__weakref__")

    def __init__(self, world: Any, handle: int) -> None:
        if not world.has_entity(handle):
            raise ValueError(f"cannot bind node view to stale handle {handle}")
        self.world = world
        self.handle = handle
        self.data = _NodeData(world, handle)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.handle}: {dict(self.data)}>"


class RelationRef[T: NodeRef](_DictView):
    """A live relation reference with ordered, interned endpoint views."""

    __slots__ = ("world", "kind", "handle", "endpoints", "data", "__weakref__")
    _kind = "bonds"
    _arity: int | None = None

    def __init__(
        self,
        world: Any,
        kind: str,
        handle: int,
        endpoints: tuple[T, ...],
    ) -> None:
        actual = tuple(world.relation_nodes(kind, handle))
        supplied = tuple(endpoint.handle for endpoint in endpoints)
        if actual != supplied or any(
            endpoint.world is not world for endpoint in endpoints
        ):
            raise ValueError(
                f"relation {kind}/{handle} endpoints are {actual}, got {supplied}"
            )
        self.world = world
        self.kind = kind
        self.handle = handle
        self.endpoints = endpoints
        self.data = _RelationData(world, kind, handle)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.handle}: {self.endpoints}>"


class Refs[R: RefLike](list[R]):
    """List of live refs with vector-style property lookup by field name."""

    @overload
    def __getitem__(self, key: int) -> R: ...  # type: ignore[override]

    @overload
    def __getitem__(self, key: slice) -> list[R]: ...  # type: ignore[override]

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | slice | str) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            values = [ref.get(key) for ref in self]
            try:
                return np.array(values)
            except (ValueError, TypeError):
                return np.array(values, dtype=object)
        return super().__getitem__(key)


class GraphViews:
    """Mixin adding live refs and factories to a native molrs graph leaf."""

    _node_cls: type[NodeRef] = NodeRef
    _relation_classes: dict[str, type[RelationRef]] = {}

    def __init__(self, **props: Any) -> None:
        self._props: dict[str, Any] = dict(props)
        # Registrations belong to one world.  Mutating the class dictionary would
        # make a custom relation type in one graph leak into every graph of that
        # Python class.
        self._relation_classes = dict(type(self)._relation_classes)
        self._node_refs: WeakValueDictionary[int, NodeRef] = WeakValueDictionary()
        self._relation_refs: dict[str, WeakValueDictionary[int, RelationRef]] = {}

    def __getitem__(self, key: str) -> Any:
        return self._props[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._props[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._props

    def get(self, key: str, default: Any = None) -> Any:
        return self._props.get(key, default)

    def _intern_node(self, handle: int, cls: type[NodeRef] | None = None) -> NodeRef:
        ref = self._node_refs.get(handle)
        if ref is None:
            ref = (cls or self._node_cls)(self, handle)
            self._node_refs[handle] = ref
        return ref

    def _intern_relation(
        self,
        kind: str,
        handle: int,
        cls: type[RelationRef] | None = None,
    ) -> RelationRef:
        table = self._relation_refs.setdefault(kind, WeakValueDictionary())
        ref = table.get(handle)
        if ref is None:
            endpoints = tuple(
                self._intern_node(node) for node in self.relation_nodes(kind, handle)
            )
            relation_cls = cls or self._relation_classes.get(kind, RelationRef)
            ref = relation_cls(self, kind, handle, endpoints)
            table[handle] = ref
        return ref

    def _create_node(
        self,
        mapping: Any = None,
        /,
        *,
        cls: type[NodeRef] | None = None,
        **attrs: Any,
    ) -> NodeRef:
        data = {} if mapping is None else dict(mapping)
        data.update(attrs)
        handle = self.spawn()
        try:
            ref = self._intern_node(handle, cls)
            ref.update(data)
        except Exception:
            self._node_refs.pop(handle, None)
            self.despawn(handle)
            raise
        return ref

    def _create_relation(
        self,
        kind: str,
        endpoints: tuple[NodeRef, ...],
        /,
        *,
        cls: type[RelationRef] | None = None,
        **attrs: Any,
    ) -> RelationRef:
        if any(endpoint.world is not self for endpoint in endpoints):
            raise ValueError("relation endpoints must belong to this graph")
        handle = self.add_relation(kind, [endpoint.handle for endpoint in endpoints])
        try:
            ref = self._intern_relation(kind, handle, cls)
            ref.update(attrs)
        except Exception:
            self._relation_refs.get(kind, {}).pop(handle, None)
            self.remove_relation(kind, handle)
            raise
        return ref

    def _node_handles(self) -> list[int]:
        return self.entities()

    def _node_views(self) -> Refs[NodeRef]:
        return Refs(self._intern_node(handle) for handle in self._node_handles())

    def _relation_views(self, kind: str) -> Refs[RelationRef]:
        if kind not in self.kinds():
            return Refs()
        return Refs(
            self._intern_relation(kind, handle) for handle in self.relation_ids(kind)
        )

    def _all_relation_views(self) -> Refs[RelationRef]:
        refs: Refs[RelationRef] = Refs()
        # The native graph owns the open relation registry.  Python class
        # registrations select a richer view; they must never select which native
        # relations exist or silently hide an unrecognised kind.
        for kind in self.kinds():
            refs.extend(self._relation_views(kind))
        return refs

    @property
    def nodes(self) -> Refs[NodeRef]:
        return self._node_views()

    @property
    def links(self) -> "RelationBuckets":
        return RelationBuckets(self)

    def _remove_node(self, ref: NodeRef) -> None:
        if ref.world is not self:
            raise ValueError("node belongs to another graph")
        for kind in self.kinds():
            for handle in list(self.relation_ids(kind)):
                if ref.handle in self.relation_nodes(kind, handle):
                    self._remove_relation(self._intern_relation(kind, handle))
        self.despawn(ref.handle)
        self._node_refs.pop(ref.handle, None)

    def _remove_relation(self, ref: RelationRef) -> None:
        if ref.world is not self:
            raise ValueError("relation belongs to another graph")
        self.remove_relation(ref.kind, ref.handle)
        self._relation_refs.get(ref.kind, {}).pop(ref.handle, None)


class RelationBuckets:
    """Kind-filtered view over a graph's live relations."""

    def __init__(self, world: GraphViews) -> None:
        self.world = world

    def all(self) -> Refs[RelationRef]:
        return self.world._all_relation_views()

    def bucket(self, cls: type) -> Refs[RelationRef]:
        return Refs(ref for ref in self.all() if isinstance(ref, cls))

    def exact_bucket(self, cls: type) -> Refs[RelationRef]:
        return Refs(ref for ref in self.all() if type(ref) is cls)

    def classes(self) -> Iterator[type]:
        return iter(dict.fromkeys(type(ref) for ref in self.all()))

    def remove(self, *refs: RelationRef) -> None:
        for ref in refs:
            self.world._remove_relation(ref)

    def register_type(self, cls: type) -> None:
        if not isinstance(cls, type) or not issubclass(cls, RelationRef):
            raise TypeError("relation view type must inherit RelationRef")
        kind = getattr(cls, "_kind", None)
        if not isinstance(kind, str) or not kind:
            raise TypeError("relation view type must declare a non-empty _kind")
        table = self.world._relation_refs.get(kind)
        if table is not None and len(table) != 0:
            raise RuntimeError(
                f"cannot register {cls.__name__} for {kind!r} after views were interned"
            )
        if kind not in self.world.kinds():
            arity = getattr(cls, "_arity", None)
            if not isinstance(arity, int) or arity <= 0:
                raise ValueError(
                    f"cannot register native kind {kind!r}: {cls.__name__} must declare _arity"
                )
            self.world.register_kind(kind, arity)
        self.world._relation_classes[kind] = cls

    def __getitem__(self, cls: type) -> Refs[RelationRef]:
        return self.bucket(cls)

    def __len__(self) -> int:
        return len(self.all())

    def __iter__(self) -> Iterator[RelationRef]:
        return iter(self.all())


class Atom(NodeRef):
    __slots__ = ()

    @property
    def is_virtual(self) -> bool:
        return self.get("vsite") is not None

    def __repr__(self) -> str:
        ident = self.get(fields.ELEMENT) or self.get(fields.TYPE)
        return f"<Atom {self.handle}: {ident or '?'}>"


class VirtualSite(Atom):
    __slots__ = ()
    _vsite_kind = "virtual"


class DrudeParticle(VirtualSite):
    __slots__ = ()
    _vsite_kind = "drude"


class MasslessSite(VirtualSite):
    __slots__ = ()
    _vsite_kind = "massless"


class Bond(RelationRef[Atom]):
    __slots__ = ()
    _kind = "bonds"
    _arity = 2

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]


class Angle(RelationRef[Atom]):
    __slots__ = ()
    _kind = "angles"
    _arity = 3

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        return self.endpoints[2]


class Dihedral(RelationRef[Atom]):
    __slots__ = ()
    _kind = "dihedrals"
    _arity = 4

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        return self.endpoints[2]

    @property
    def ltom(self) -> Atom:
        return self.endpoints[3]


class Improper(Dihedral):
    __slots__ = ()
    _kind = "impropers"


class Bead(NodeRef):
    __slots__ = ()

    def __repr__(self) -> str:
        ident = self.get("type") or self.get("name")
        return f"<Bead {self.handle}: {ident or '?'}>"

    def __getitem__(self, key: Any) -> Any:
        if key == "atoms":
            return self.world._resolve_bead_atoms(self.handle)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if key == "atoms":
            atoms = self.world._resolve_bead_atoms(self.handle)
            return atoms if atoms else default
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if key == "atoms":
            return bool(self.world._resolve_bead_atoms(self.handle))
        return super().__contains__(key)


class CGBond(RelationRef[Bead]):
    __slots__ = ()
    _kind = "bonds"
    _arity = 2

    @property
    def ibead(self) -> Bead:
        return self.endpoints[0]

    @property
    def jbead(self) -> Bead:
        return self.endpoints[1]


class Atomistic(_RsAtomistic, GraphViews):
    """Public all-atom graph with live node and relation views.

    The native PyO3 leaf remains the storage owner and first base.  This class
    contributes factories and handle views only; native algorithms continue to
    accept it directly because it is an ``_RsAtomistic`` subclass.
    """

    _node_cls = Atom
    _relation_classes = {
        "bonds": Bond,
        "angles": Angle,
        "dihedrals": Dihedral,
        "impropers": Improper,
    }

    def __init__(self, **props: Any) -> None:
        GraphViews.__init__(self, **props)

    @property
    def atoms(self) -> Refs[Atom]:
        return self._node_views()  # type: ignore[return-value]

    @property
    def bonds(self) -> Refs[Bond]:
        return self._relation_views("bonds")  # type: ignore[return-value]

    @property
    def angles(self) -> Refs[Angle]:
        return self._relation_views("angles")  # type: ignore[return-value]

    @property
    def dihedrals(self) -> Refs[Dihedral]:
        return self._relation_views("dihedrals")  # type: ignore[return-value]

    @property
    def impropers(self) -> Refs[Improper]:
        return self._relation_views("impropers")  # type: ignore[return-value]

    def def_atom(self, mapping: Any = None, /, **attrs: Any) -> Atom:
        return self._create_node(mapping, cls=Atom, **attrs)  # type: ignore[return-value]

    def def_virtual_site(
        self,
        mapping: Any = None,
        /,
        *,
        kind: type[VirtualSite] = VirtualSite,
        **attrs: Any,
    ) -> VirtualSite:
        attrs.setdefault("vsite", kind._vsite_kind)
        return self._create_node(mapping, cls=kind, **attrs)  # type: ignore[return-value]

    def def_bond(self, a: Atom, b: Atom, /, **attrs: Any) -> Bond:
        return self._create_relation(  # type: ignore[return-value]
            "bonds", (a, b), cls=Bond, **attrs
        )

    def def_angle(self, a: Atom, b: Atom, c: Atom, /, **attrs: Any) -> Angle:
        return self._create_relation(  # type: ignore[return-value]
            "angles", (a, b, c), cls=Angle, **attrs
        )

    def def_dihedral(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any
    ) -> Dihedral:
        return self._create_relation(  # type: ignore[return-value]
            "dihedrals", (a, b, c, d), cls=Dihedral, **attrs
        )

    def def_improper(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any
    ) -> Improper:
        return self._create_relation(  # type: ignore[return-value]
            "impropers", (a, b, c, d), cls=Improper, **attrs
        )

    def del_atom(self, *atoms: Atom) -> None:
        for atom in atoms:
            self._remove_node(atom)

    def remove_link(self, *links: RelationRef) -> None:
        for link in links:
            self._remove_relation(link)


class CoarseGrain(_RsCoarseGrain, GraphViews):
    """Public coarse-grained graph with live bead and bond views."""

    _node_cls = Bead
    _relation_classes = {"bonds": CGBond}

    def __init__(self, **props: Any) -> None:
        GraphViews.__init__(self, **props)
        self._member_world: Any = None

    @property
    def beads(self) -> Refs[Bead]:
        return self._node_views()  # type: ignore[return-value]

    @property
    def cgbonds(self) -> Refs[CGBond]:
        return self._relation_views("bonds")  # type: ignore[return-value]

    def def_bead(self, mapping: Any = None, /, **attrs: Any) -> Bead:
        atoms = attrs.pop("atoms", None)
        if mapping is not None and "atoms" in mapping:
            mapping = dict(mapping)
            atoms = mapping.pop("atoms")
        bead = self._create_node(mapping, cls=Bead, **attrs)
        if atoms is not None:
            self._set_bead_atoms(bead, tuple(atoms))
        return bead  # type: ignore[return-value]

    def _set_bead_atoms(self, bead: Bead, atoms: tuple[NodeRef, ...]) -> None:
        handles: list[int] = []
        for atom in atoms:
            if self._member_world is None:
                self._member_world = atom.world
            elif atom.world is not self._member_world:
                raise ValueError(
                    "bead membership atoms must all come from the same source world"
                )
            handles.append(atom.handle)
        self.set_bead_members(bead.handle, handles)

    def _resolve_bead_atoms(self, bead_handle: int) -> tuple[NodeRef, ...]:
        handles = self.bead_members(bead_handle)
        if not handles or self._member_world is None:
            return ()
        return tuple(self._member_world._intern_node(handle) for handle in handles)

    def def_cgbond(self, a: Bead, b: Bead, /, **attrs: Any) -> CGBond:
        return self._create_relation(  # type: ignore[return-value]
            "bonds", (a, b), cls=CGBond, **attrs
        )

    def del_bead(self, *beads: Bead) -> None:
        for bead in beads:
            self._remove_node(bead)

    def remove_link(self, *links: RelationRef) -> None:
        for link in links:
            self._remove_relation(link)


# Compatibility names are aliases, not alternate implementations.
Entity = NodeRef
Link = RelationRef
Entities = Refs
_GraphViews = GraphViews


__all__ = [
    "Angle",
    "Atom",
    "Atomistic",
    "Bead",
    "Bond",
    "CGBond",
    "CoarseGrain",
    "Dihedral",
    "DrudeParticle",
    "Entities",
    "Entity",
    "GraphViews",
    "Improper",
    "Link",
    "MasslessSite",
    "NodeRef",
    "Refs",
    "RelationRef",
    "VirtualSite",
]
