"""FFI seam of the assembly surface (assembly-07 §1–2).

``Trace`` / ``FragGraph`` / ``Mapping`` at the top level; ``FragLibrary``, the
placers, orienters, reacters, ``Finalizer`` and ``Assembler`` under
``molrs.builder``. These prove construction, the tuple/array shapes at the
boundary, error mapping, and the explicit Python-subclass crossing: one call
per batch (per template group for a placer, per ``place_many`` for an
orienter, per ``assemble`` for a reacter), a world moved into and back out of
a Python reacter, and exceptions surfacing as ``ValueError``.

No numerics are re-derived here: placement, orientation, linking and mapping
are proven by the unit tests in ``molrs/src/builder/``. Fixtures are the
1–3-atom template U of ``builder::assemble::tests`` (C at the origin, a
capping H at x = ∓1 carrying port ``<`` (ordinal 0) and ``>`` (ordinal 1));
a three-unit path joins ``>`` of unit i to ``<`` of unit i+1, so two links
leave 3 C + 2 end H = 5 atoms and 2 unpaired ports (hand-derived there).
"""

from __future__ import annotations

import numpy as np
import pytest

import molrs
from molrs import _lib
from molrs import builder

C_MASS = 12.011
H_MASS = 1.008

# One point per unit, one unit per carbon of the U chain.
THREE_POINTS = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [3.08, 0.0, 0.0]])


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _template(label: str) -> molrs.Fragment:
    """Template U (or a renamed copy): one bead `label` of C + two H handles."""
    frag = molrs.Fragment()
    stamp = {"bead": 0, "bead_type": label}
    c = frag.def_atom(element="C", x=0.0, y=0.0, z=0.0, mass=C_MASS, **stamp)
    left = frag.def_atom(element="H", x=-1.0, y=0.0, z=0.0, mass=H_MASS, **stamp)
    right = frag.def_atom(element="H", x=1.0, y=0.0, z=0.0, mass=H_MASS, **stamp)
    frag.def_bond(c, left)
    frag.def_bond(c, right)
    frag.def_port(c, left, "<")
    frag.def_port(c, right, ">")
    return frag


def _library(*names: str) -> builder.FragLibrary:
    library = builder.FragLibrary()
    for name in names:
        library.insert(name, _template(name))
    return library


def _u_path(n: int = 3) -> molrs.FragGraph:
    return molrs.FragGraph.path(["U"] * n, (1, 0))


def _trace_placer(seq: list[str] | None = None) -> builder.TracePlacer:
    return builder.TracePlacer(molrs.Trace(THREE_POINTS), seq or ["U", "U", "U"])


def _identity_motions(units: np.ndarray, spacing: float = 1.54):
    """Identity rotations and translations `spacing · unit` along x."""
    n = len(units)
    rotations = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    translations = np.zeros((n, 3))
    translations[:, 0] = spacing * np.asarray(units, dtype=np.float64)
    return rotations, translations


class RecordingPlacer(builder.Placer):
    """Records every `place_many` call; places unit u at x = 1.54 u."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, list[int], np.dtype, int]] = []

    def place_many(self, units, name, template):
        self.calls.append((name, list(units), units.dtype, template.n_atoms))
        return _identity_motions(units)


class RecordingOrienter(builder.Orienter):
    """Records every call; completes each fit with the fit's own motion."""

    def __init__(self) -> None:
        super().__init__()
        self.orient_many_calls: list[tuple[list[int], list, object]] = []
        self.orient_calls = 0

    def orient(self, unit, template, fit, hint):
        self.orient_calls += 1
        return fit.rotation, fit.translation

    def orient_many(self, units, template, fits, hints):
        self.orient_many_calls.append((list(units), list(fits), hints))
        return (
            np.stack([f.rotation for f in fits]),
            np.stack([f.translation for f in fits]),
        )


class RecordingReacter(builder.Reacter):
    """Keeps every `world` it is handed and links through the native PortReacter."""

    def __init__(self) -> None:
        super().__init__()
        self.worlds: list[molrs.Fragment] = []
        self.pairs: list[list[tuple[int, int]]] = []
        self._native = builder.PortReacter()

    def link_many(self, world, pairs):
        self.worlds.append(world)
        self.pairs.append(list(pairs))
        return self._native.link_many(world, pairs)


class NoLinkReacter(builder.Reacter):
    """Keeps every `world` it is handed; links nothing (for edge-free graphs)."""

    def __init__(self) -> None:
        super().__init__()
        self.worlds: list[molrs.Fragment] = []

    def link_many(self, world, pairs):
        self.worlds.append(world)
        return []


# --------------------------------------------------------------------------- #
# Registered surface                                                          #
# --------------------------------------------------------------------------- #


BUILDER_NAMES = (
    "FragLibrary",
    "Placer",
    "TracePlacer",
    "Orienter",
    "NullOrienter",
    "RandomOrienter",
    "HintOrienter",
    "Reacter",
    "PortReacter",
    "Finalizer",
    "Assembler",
)


@pytest.mark.parametrize("name", BUILDER_NAMES)
def test_builder_reexports_the_native_class(name):
    assert getattr(builder, name) is getattr(_lib, name)


@pytest.mark.parametrize("name", ["Trace", "FragGraph", "Mapping"])
def test_core_assembly_types_live_at_the_top_level(name):
    assert getattr(molrs, name) is getattr(_lib, name)


def test_native_components_subclass_their_base():
    assert issubclass(builder.TracePlacer, builder.Placer)
    for orienter in (builder.NullOrienter, builder.RandomOrienter, builder.HintOrienter):
        assert issubclass(orienter, builder.Orienter)
    assert issubclass(builder.PortReacter, builder.Reacter)


# --------------------------------------------------------------------------- #
# Trace                                                                       #
# --------------------------------------------------------------------------- #


class TestTrace:
    def test_one_point_per_unit_by_default(self):
        trace = molrs.Trace(THREE_POINTS)
        assert trace.n_units == 3
        unit = trace.unit(1)
        assert unit.dtype == np.float64
        assert unit.shape == (1, 3)
        assert trace.hint(0) is None

    def test_offsets_split_ragged_units_and_hints_are_normalised(self):
        points = np.arange(15, dtype=np.float64).reshape(5, 3)
        trace = molrs.Trace(points, offsets=[0, 2, 5], hints=[[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
        assert trace.n_units == 2
        assert trace.unit(1).shape == (3, 3)
        np.testing.assert_array_equal(trace.unit(0), points[:2])
        np.testing.assert_allclose(trace.hint(0), [1.0, 0.0, 0.0], rtol=0, atol=1e-12)

    def test_bad_offsets_are_a_value_error(self):
        with pytest.raises(ValueError):
            molrs.Trace(THREE_POINTS, offsets=[0, 2])


# --------------------------------------------------------------------------- #
# FragGraph / Mapping / FragLibrary                                           #
# --------------------------------------------------------------------------- #


class TestFragGraph:
    def test_path_edges_cross_as_four_tuples(self):
        graph = _u_path()
        assert graph.nodes == ["U", "U", "U"]
        assert graph.edges == [(0, 1, 1, 0), (1, 2, 1, 0)]

    def test_constructor_cycle_and_star(self):
        assert molrs.FragGraph(["A", "B"], [(0, 1, 0, 0)]).edges == [(0, 1, 0, 0)]
        assert len(molrs.FragGraph.cycle(["U", "U", "U"], (1, 0)).edges) == 3
        star = molrs.FragGraph.star("C", [0, 1], "U", 0)
        assert star.nodes == ["C", "U", "U"]

    def test_a_self_edge_is_a_value_error(self):
        with pytest.raises(ValueError):
            molrs.FragGraph(["A"], [(0, 0, 0, 1)])


def _one_bead_cg(bead_type: str = "4"):
    cg = molrs.CoarseGrain()
    bead = cg.def_bead(bead_type=bead_type, x=1.0, y=2.0, z=3.0)
    return cg, bead


def _one_bead_template(label: str = "A") -> molrs.Fragment:
    frag = molrs.Fragment()
    frag.def_atom(element="C", x=0.0, y=0.0, z=0.0, mass=C_MASS, bead=0, bead_type=label)
    return frag


class TestFragLibrary:
    def test_insert_get_names(self):
        library = _library("U", "V")
        assert sorted(library.names()) == ["U", "V"]
        got = library.get("U")
        assert isinstance(got, molrs.Fragment)
        assert got.n_atoms == 3
        assert library.get("missing") is None

    def test_insert_refuses_an_atom_without_a_bead(self):
        frag = molrs.Fragment()
        frag.def_atom(element="C", x=0.0, y=0.0, z=0.0, bead_type="A")
        with pytest.raises(ValueError):
            builder.FragLibrary().insert("A", frag)

    def test_map_with_a_pair_rule_returns_a_one_unit_mapping(self):
        cg, _ = _one_bead_cg("4")
        library = builder.FragLibrary()
        library.insert("A1", _one_bead_template("A"))

        mapping = library.map(cg, [("4", "A")])

        assert isinstance(mapping, molrs.Mapping)
        assert mapping.rules == [("4", "A")]
        assert mapping.labels(0) == ["A"]
        assert mapping.trace(cg).n_units == 1

    def test_map_with_a_non_pair_rule_is_a_type_error(self):
        cg, _ = _one_bead_cg("4")
        library = builder.FragLibrary()
        library.insert("A1", _one_bead_template("A"))
        with pytest.raises(TypeError):
            library.map(cg, ["4"])

    def test_map_failure_is_a_value_error(self):
        cg, _ = _one_bead_cg("4")
        library = builder.FragLibrary()
        library.insert("A1", _one_bead_template("A"))
        with pytest.raises(ValueError):
            library.map(cg, [])


class TestMapping:
    def test_hand_built_mapping_traces_its_source_beads(self):
        cg, bead = _one_bead_cg("4")
        mapping = molrs.Mapping(molrs.FragGraph(["A1"], []), [[bead]], [["A"]], [("4", "A")])

        trace = mapping.trace(cg)
        assert trace.n_units == 1
        np.testing.assert_array_equal(trace.unit(0), [[1.0, 2.0, 3.0]])

    def test_a_label_no_rule_licenses_is_a_value_error(self):
        _, bead = _one_bead_cg("4")
        with pytest.raises(ValueError):
            molrs.Mapping(molrs.FragGraph(["A1"], []), [[bead]], [["B"]], [("4", "A")])


# --------------------------------------------------------------------------- #
# Native components                                                           #
# --------------------------------------------------------------------------- #


class TestNativeComponents:
    def test_trace_placer_refuses_a_seq_of_another_length(self):
        with pytest.raises(ValueError):
            builder.TracePlacer(molrs.Trace(THREE_POINTS), ["U", "U"])

    def test_orienters_construct(self):
        builder.NullOrienter()
        builder.RandomOrienter(7)
        builder.HintOrienter(axis="principal")
        builder.HintOrienter(axis="dipole")

    def test_hint_orienter_refuses_an_unknown_axis(self):
        with pytest.raises(ValueError):
            builder.HintOrienter(axis="tangent")

    def test_with_orienter_returns_a_placer(self):
        placer = _trace_placer().with_orienter(builder.RandomOrienter(7))
        assert isinstance(placer, builder.Placer)

    def test_finalizer_completes_angles_and_dihedrals(self):
        # Butane skeleton C–C–C–C: 2 angles, 1 dihedral, no improper.
        mol = molrs.Atomistic()
        carbons = [mol.def_atom(element="C", x=1.5 * k, y=0.0, z=0.0) for k in range(4)]
        for a, b in zip(carbons, carbons[1:]):
            mol.def_bond(a, b)

        counts = builder.Finalizer(impropers=False).finalize(mol)

        assert counts == (2, 1, 0)
        assert mol.n_relations("angles") == 2
        assert mol.n_relations("dihedrals") == 1


class TestAssemblerNative:
    def test_path_library_trace_placer_and_port_reacter_assemble_a_fragment(self):
        assembler = builder.Assembler(_library("U"), _trace_placer(), builder.PortReacter())

        world = assembler.assemble(_u_path())

        assert type(world) is molrs.Fragment
        assert world.n_atoms == 5
        assert world.n_ports == 2
        assert {world.frag_id(a) for a in world.entities()} == {0, 1, 2}

    def test_a_trace_placer_whose_seq_names_another_template_names_unit_and_both_names(self):
        # The graph's units are "U"; the placer's seq says "V" for every unit.
        library = _library("U", "V")
        assembler = builder.Assembler(library, _trace_placer(["V", "V", "V"]), builder.PortReacter())

        with pytest.raises(ValueError) as caught:
            assembler.assemble(_u_path())

        message = str(caught.value)
        assert "unit 0" in message
        assert "'V'" in message
        assert "'U'" in message

    def test_an_unknown_template_is_a_value_error(self):
        assembler = builder.Assembler(_library("V"), _trace_placer(), builder.PortReacter())
        with pytest.raises(ValueError, match="'U'"):
            assembler.assemble(_u_path())

    def test_an_atom_without_mass_is_named_by_its_python_handle(self):
        # `FragLibrary.insert` does not check mass; the TracePlacer does.
        template = molrs.Fragment()
        massless = template.def_atom(element="C", x=0.0, y=0.0, z=0.0, bead=0, bead_type="U")
        library = builder.FragLibrary()
        library.insert("U", template)
        placer = builder.TracePlacer(molrs.Trace(THREE_POINTS[:1]), ["U"])
        assembler = builder.Assembler(library, placer, builder.PortReacter())

        with pytest.raises(ValueError) as caught:
            assembler.assemble(molrs.FragGraph(["U"], []))

        message = str(caught.value)
        assert f"atom {massless.handle} " in message
        assert "NodeId(" not in message


# --------------------------------------------------------------------------- #
# Python subclasses                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "call",
    [
        lambda: builder.Placer().place(0, "U", _template("U")),
        lambda: builder.Orienter().orient(
            0, _template("U"), molrs.op.superpose(THREE_POINTS[:1], THREE_POINTS[:1]), None
        ),
        lambda: builder.Reacter().link(_template("U"), 0, 1),
    ],
    ids=["Placer.place", "Orienter.orient", "Reacter.link"],
)
def test_a_bare_base_method_is_not_implemented(call):
    with pytest.raises(NotImplementedError):
        call()


def test_a_placer_subclass_may_take_its_own_constructor_arguments():
    class P(builder.Placer):
        def __init__(self, cg):
            super().__init__()
            self.cg = cg

    cg = object()
    placer = P(cg)
    assert placer.cg is cg
    assert isinstance(placer, builder.Placer)


class TestPythonPlacer:
    def test_place_many_is_called_once_per_template_group(self):
        # Units 0 and 2 are "U", unit 1 is "V": two groups, two calls.
        placer = RecordingPlacer()
        graph = molrs.FragGraph.path(["U", "V", "U"], (1, 0))
        assembler = builder.Assembler(_library("U", "V"), placer, builder.PortReacter())

        world = assembler.assemble(graph)

        assert type(world) is molrs.Fragment
        assert world.n_atoms == 5
        assert sorted((name, units) for name, units, _, _ in placer.calls) == [
            ("U", [0, 2]),
            ("V", [1]),
        ]
        for _, _, dtype, n_template_atoms in placer.calls:
            assert dtype == np.int64
            assert n_template_atoms == 3

    def test_mutating_the_handed_template_does_not_reach_the_library(self):
        class Vandal(builder.Placer):
            def place_many(self, units, name, template):
                template.def_atom(element="Xe", x=0.0, y=0.0, z=0.0)
                return _identity_motions(units)

        library = _library("U")
        builder.Assembler(library, Vandal(), builder.PortReacter()).assemble(_u_path())
        assert library.get("U").n_atoms == 3

    @pytest.mark.parametrize(
        "motions",
        [
            lambda n: (np.zeros((n, 3)), np.zeros((n, 3))),
            lambda n: (np.broadcast_to(np.eye(3), (n, 3, 3)).copy(), np.zeros((n, 2))),
            lambda n: (np.broadcast_to(np.eye(3), (n + 1, 3, 3)).copy(), np.zeros((n + 1, 3))),
        ],
        ids=["rotations-Nx3", "translations-Nx2", "one-motion-too-many"],
    )
    def test_a_wrong_shape_place_many_return_is_a_value_error(self, motions):
        class Misshapen(builder.Placer):
            def place_many(self, units, name, template):
                return motions(len(units))

        assembler = builder.Assembler(_library("U"), Misshapen(), builder.PortReacter())
        with pytest.raises(ValueError):
            assembler.assemble(_u_path())


    def test_a_place_only_subclass_is_reached_once_per_unit_by_the_base_place_many(self):
        # Only `place` is overridden: the base `place_many` loops over it. Unit u
        # goes to (10 u, 5, -2) with the identity rotation, so the carbon at the
        # template origin lands exactly there.
        class PlaceOnly(builder.Placer):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[tuple[int, str, int]] = []

            def place(self, unit, name, template):
                self.calls.append((unit, name, template.n_atoms))
                return np.eye(3), np.array([10.0 * unit, 5.0, -2.0])

        placer = PlaceOnly()
        assembler = builder.Assembler(_library("U"), placer, builder.PortReacter())

        world = assembler.assemble(_u_path())

        assert sorted(placer.calls) == [(0, "U", 3), (1, "U", 3), (2, "U", 3)]
        assert all(type(unit) is int for unit, _, _ in placer.calls)
        carbons = {
            world.frag_id(a): (world.get(a, "x"), world.get(a, "y"), world.get(a, "z"))
            for a in world.entities()
            if world.get(a, "element") == "C"
        }
        assert sorted(carbons) == [0, 1, 2]
        for unit, xyz in carbons.items():
            np.testing.assert_allclose(xyz, [10.0 * unit, 5.0, -2.0], rtol=0, atol=1e-12)


class TestPythonOrienter:
    def test_a_native_trace_placer_calls_orient_many_once_per_place_many(self):
        # k = 1 bead per unit: every fit is "free", so all three units of the
        # one "U" group go to the orienter in one batch.
        orienter = RecordingOrienter()
        placer = _trace_placer().with_orienter(orienter)
        assembler = builder.Assembler(_library("U"), placer, builder.PortReacter())

        world = assembler.assemble(_u_path())

        assert world.n_atoms == 5
        assert len(orienter.orient_many_calls) == 1
        assert orienter.orient_calls == 0
        units, fits, hints = orienter.orient_many_calls[0]
        assert units == [0, 1, 2]
        assert all(isinstance(f, molrs.op.Fit) and f.freedom == "free" for f in fits)
        assert hints is None

    def test_hints_cross_as_an_n_by_3_float64_array(self):
        orienter = RecordingOrienter()
        trace = molrs.Trace(THREE_POINTS, hints=np.tile([0.0, 0.0, 1.0], (3, 1)))
        placer = builder.TracePlacer(trace, ["U", "U", "U"]).with_orienter(orienter)

        builder.Assembler(_library("U"), placer, builder.PortReacter()).assemble(_u_path())

        (_, _, hints), = orienter.orient_many_calls
        assert hints.dtype == np.float64
        assert hints.shape == (3, 3)


    def test_an_orient_only_subclass_is_reached_once_per_unit_with_a_free_fit(self):
        # Only `orient` is overridden: the base `orient_many` loops over it.
        # k = 1 bead per unit, so every fit is "free".
        class OrientOnly(builder.Orienter):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[tuple[int, object, object]] = []

            def orient(self, unit, template, fit, hint):
                self.calls.append((unit, fit, hint))
                return fit.rotation, fit.translation

        orienter = OrientOnly()
        placer = _trace_placer().with_orienter(orienter)
        assembler = builder.Assembler(_library("U"), placer, builder.PortReacter())

        world = assembler.assemble(_u_path())

        assert world.n_atoms == 5
        assert sorted(unit for unit, _, _ in orienter.calls) == [0, 1, 2]
        for _, fit, hint in orienter.calls:
            assert isinstance(fit, molrs.op.Fit)
            assert fit.freedom == "free"
            assert hint is None


class TestPythonReacter:
    def test_link_many_is_called_once_with_every_edge_as_an_int_pair(self):
        reacter = RecordingReacter()
        assembler = builder.Assembler(_library("U"), _trace_placer(), reacter)

        world = assembler.assemble(_u_path())

        assert world.n_atoms == 5
        assert len(reacter.pairs) == 1
        (pairs,) = reacter.pairs
        assert len(pairs) == 2
        assert all(isinstance(a, int) and isinstance(b, int) for a, b in pairs)

    def test_a_retained_world_is_empty_after_assemble_returns(self):
        # One unit, no edge: the world holds the three template atoms.
        reacter = NoLinkReacter()
        placer = builder.TracePlacer(molrs.Trace(THREE_POINTS[:1]), ["U"])
        assembler = builder.Assembler(_library("U"), placer, reacter)

        world = assembler.assemble(molrs.FragGraph(["U"], []))

        assert world.n_atoms == 3
        (retained,) = reacter.worlds
        assert isinstance(retained, molrs.Fragment)
        assert retained is not world
        assert retained.n_atoms == 0

    def test_a_retained_world_of_a_linked_chain_is_empty_too(self):
        reacter = RecordingReacter()
        world = builder.Assembler(_library("U"), _trace_placer(), reacter).assemble(_u_path())

        assert world.n_atoms == 5
        (retained,) = reacter.worlds
        assert retained.n_atoms == 0


    def test_a_retained_world_is_empty_even_when_link_many_raises(self):
        class KeepsThenFails(builder.Reacter):
            def __init__(self) -> None:
                super().__init__()
                self.worlds: list[molrs.Fragment] = []

            def link_many(self, world, pairs):
                self.worlds.append(world)
                raise RuntimeError("kept-then-failed")

        reacter = KeepsThenFails()
        assembler = builder.Assembler(_library("U"), _trace_placer(), reacter)

        with pytest.raises(ValueError, match="kept-then-failed"):
            assembler.assemble(_u_path())

        (retained,) = reacter.worlds
        assert isinstance(retained, molrs.Fragment)
        assert retained.n_atoms == 0


class TestSubclassExceptions:
    def test_a_placer_exception_surfaces_as_value_error_with_its_message(self):
        class Failing(builder.Placer):
            def place_many(self, units, name, template):
                raise RuntimeError("placer-boom-1729")

        assembler = builder.Assembler(_library("U"), Failing(), builder.PortReacter())
        with pytest.raises(ValueError, match="placer-boom-1729"):
            assembler.assemble(_u_path())

    def test_an_orienter_exception_surfaces_as_value_error_with_its_message(self):
        class Failing(builder.Orienter):
            def orient_many(self, units, template, fits, hints):
                raise RuntimeError("orienter-boom-1729")

        placer = _trace_placer().with_orienter(Failing())
        assembler = builder.Assembler(_library("U"), placer, builder.PortReacter())
        with pytest.raises(ValueError, match="orienter-boom-1729"):
            assembler.assemble(_u_path())

    def test_a_reacter_exception_surfaces_as_value_error_with_its_message(self):
        class Failing(builder.Reacter):
            def link_many(self, world, pairs):
                raise RuntimeError("reacter-boom-1729")

        assembler = builder.Assembler(_library("U"), _trace_placer(), Failing())
        with pytest.raises(ValueError, match="reacter-boom-1729"):
            assembler.assemble(_u_path())

    def test_a_link_only_subclass_failing_on_its_second_call_names_edge_1(self):
        # Only `link` is overridden: the base `link_many` walks the two path
        # edges in order, so the second call is edge 1.
        class SecondFails(builder.Reacter):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0
                self._native = builder.PortReacter()

            def link(self, world, a, b):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("nope")
                return self._native.link(world, a, b)

        assembler = builder.Assembler(_library("U"), _trace_placer(), SecondFails())
        with pytest.raises(ValueError) as caught:
            assembler.assemble(_u_path())

        assert "edge 1" in str(caught.value)

    def test_a_link_many_exception_without_a_pair_tag_names_an_edge(self):
        class Untagged(builder.Reacter):
            def link_many(self, world, pairs):
                raise RuntimeError("x")

        assembler = builder.Assembler(_library("U"), _trace_placer(), Untagged())
        with pytest.raises(ValueError) as caught:
            assembler.assemble(_u_path())

        assert "an edge" in str(caught.value)


def _assembler_raising(exc: BaseException, component: str) -> builder.Assembler:
    """An assembler whose Python `component` raises `exc` from its batched method."""

    class RaisingPlacer(builder.Placer):
        def place_many(self, units, name, template):
            raise exc

    class RaisingOrienter(builder.Orienter):
        def orient_many(self, units, template, fits, hints):
            raise exc

    class RaisingReacter(builder.Reacter):
        def link_many(self, world, pairs):
            raise exc

    if component == "placer":
        return builder.Assembler(_library("U"), RaisingPlacer(), builder.PortReacter())
    if component == "orienter":
        placer = _trace_placer().with_orienter(RaisingOrienter())
        return builder.Assembler(_library("U"), placer, builder.PortReacter())
    return builder.Assembler(_library("U"), _trace_placer(), RaisingReacter())


COMPONENTS = ("placer", "orienter", "reacter")


class TestSubclassExceptionCause:
    @pytest.mark.parametrize("component", COMPONENTS)
    def test_a_subclass_exception_is_the_value_errors_cause(self, component):
        raised = KeyError("k")
        with pytest.raises(ValueError) as caught:
            _assembler_raising(raised, component).assemble(_u_path())

        assert caught.value.__cause__ is raised

    @pytest.mark.parametrize("component", COMPONENTS)
    def test_a_keyboard_interrupt_propagates_unchanged(self, component):
        with pytest.raises(KeyboardInterrupt):
            _assembler_raising(KeyboardInterrupt(), component).assemble(_u_path())
