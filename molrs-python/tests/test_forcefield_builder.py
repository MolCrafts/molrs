"""Seam tests for the ``molrs.ff.ForceField`` construction primitives.

A force field is built through three doors: ``ForceField.def_style(category,
name, params=None)``, ``Style.def_type(name, params=None)`` and
``Style.def_type_at(name, endpoints, params=None)``. The Rust builder semantics
(style identity, name grammar, arity rules) are unit-tested in
``molrs/src/ff/forcefield/mod.rs``. These tests prove only the binding seam:
chaining, the str/float split of ``params``, and error mapping.
"""

import numpy as np
import pytest

import molrs


def test_empty_forcefield_constructs():
    ff = molrs.ff.ForceField("scratch")
    assert ff.name == "scratch"
    assert ff.style_names() == []


def test_def_style_then_def_type_chains():
    ff = molrs.ff.ForceField("chain")
    (
        ff.def_style("bond", "harmonic", {"scale": 1.0})
        .def_type("CT-OH", {"k": 300.0, "r0": 1.4})
        .def_type("CT-CT", {"k": 310.0, "r0": 1.5})
    )
    assert ff.style_names() == ["bond:harmonic"]
    assert ff.types("bond", "harmonic") == [
        ("CT-OH", {"k": 300.0, "r0": 1.4}),
        ("CT-CT", {"k": 310.0, "r0": 1.5}),
    ]


def test_def_style_returns_the_category_handle():
    ff = molrs.ff.ForceField("handle")
    assert isinstance(ff.def_style("pair", "lj/cut"), molrs.ff.PairStyle)


def test_def_type_at_stores_the_given_endpoints():
    ff = molrs.ff.ForceField("mmff")
    ff.def_style("bond", "mmff").def_type_at("0_1_5", ["1", "5"], {"kb": 4.258})
    assert list(ff.type_endpoints("bond", "mmff", "0_1_5")) == ["1", "5"]


def test_str_param_round_trips_through_types():
    ff = molrs.ff.ForceField("strings")
    ff.def_style("atom", "full").def_type("CT", {"mass": 12.011, "element": "C"})
    assert ff.types("atom", "full") == [("CT", {"mass": 12.011, "element": "C"})]


@pytest.mark.parametrize(
    "category,name",
    [("bond", "CT"), ("bond", "A-B-C"), ("angle", "A-B"), ("dihedral", "A-B-C")],
)
def test_malformed_type_name_raises_value_error(category, name):
    ff = molrs.ff.ForceField("guard")
    style = ff.def_style(category, "s")
    with pytest.raises(ValueError):
        style.def_type(name, {"k": 1.0})


def test_style_params_exposes_mixing():
    ff = molrs.ff.ForceField("lj")
    ff.def_style("pair", "lj/cut", {"cutoff": 10.0, "mixing": "geometric"})
    assert ff.style_params("pair", "lj/cut") == {"cutoff": 10.0, "mixing": "geometric"}


def test_forcefield_has_no_def_bondstyle():
    assert not hasattr(molrs.ff.ForceField, "def_bondstyle")


def test_kspace_is_not_a_category():
    from molrs._lib import ForceField as RsForceField

    assert not hasattr(RsForceField, "def_kspacestyle")
    ff = molrs.ff.ForceField("guard")
    with pytest.raises(ValueError, match="unknown"):
        ff.def_style("kspace", "pme")


def test_types_on_missing_style_raises():
    ff = molrs.ff.ForceField("empty")
    with pytest.raises(ValueError):
        ff.types("bond", "nope")


def test_conflicting_def_type_raises_value_error():
    ff = molrs.ff.ForceField("conflict")
    style = ff.def_style("bond", "harmonic")
    style.def_type("CT-OH", {"k": 300.0, "r0": 1.4})
    with pytest.raises(ValueError):
        style.def_type("CT-OH", {"k": 310.0, "r0": 1.4})


def test_conflicting_def_style_raises_value_error():
    ff = molrs.ff.ForceField("conflict")
    ff.def_style("pair", "lj/cut", {"cutoff": 10.0})
    with pytest.raises(ValueError):
        ff.def_style("pair", "lj/cut", {"cutoff": 12.0})


# ---- merge: one Rust implementation behind the Python seam ----


def test_merge_returns_self():
    ff = molrs.ff.ForceField("target")
    other = molrs.ff.ForceField("source")
    other.def_style("bond", "harmonic").def_type("CT-OH", {"k": 300.0, "r0": 1.4})
    assert ff.merge(other) is ff
    assert ff.types("bond", "harmonic") == [("CT-OH", {"k": 300.0, "r0": 1.4})]


def test_conflicting_merge_raises_value_error():
    ff = molrs.ff.ForceField("target")
    ff.def_style("bond", "harmonic").def_type("CT-OH", {"k": 300.0, "r0": 1.4})
    other = molrs.ff.ForceField("source")
    other.def_style("bond", "harmonic").def_type("CT-OH", {"k": 310.0, "r0": 1.4})
    with pytest.raises(ValueError):
        ff.merge(other)


# ---- declared units ----


def test_declared_units_is_none_on_an_undeclared_forcefield():
    ff = molrs.ff.ForceField("plain")
    assert ff.declared_units() is None
    assert ff.units == "real"


def test_units_ctor_arg_declares_units():
    ff = molrs.ff.ForceField("reduced", units="lj")
    assert ff.declared_units() == "lj"
    assert ff.units == "lj"


# ---- readers keep full style params through _from_raw ----

_OPLS_GEOMETRIC = """<ForceField name="OPLS-AA" combining_rule="geometric">
  <AtomTypes>
    <Type name="opls_001" class="opls_001" element="C" mass="12.011"/>
  </AtomTypes>
  <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
    <Atom type="opls_001" charge="0.0" sigma="0.375" epsilon="0.43932"/>
  </NonbondedForce>
</ForceField>"""


def test_read_opls_xml_str_keeps_the_combining_rule_as_mixing():
    ff = molrs.ff.read_opls_xml_str(_OPLS_GEOMETRIC)
    assert ff.style_params("pair", "lj/cut")["mixing"] == "geometric"


# ---- PotentialCompiler: the one door from a ForceField to kernels ----
#
# The compile semantics (skipping absent blocks, the 1.5 kcal/mol closed form,
# refusals) are unit-tested in ``molrs/src/ff/potential/compile.rs``. These
# tests prove only the binding seam: construction, the three doors, their
# return types, error mapping and the single public path.


def _bond_ff() -> molrs.ff.ForceField:
    ff = molrs.ff.ForceField("compile")
    ff.def_style("bond", "harmonic").def_type("CT-CT", {"k": 300.0, "r0": 1.5})
    return ff


def _bonded_pair(label: str = "CT-CT") -> molrs.Frame:
    atoms = molrs.Block()
    for key, values in (("x", [0.0, 1.6]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0])):
        atoms.insert(key, np.array(values))
    bonds = molrs.Block()
    bonds.insert("atomi", np.array([0], dtype=np.uint32))
    bonds.insert("atomj", np.array([1], dtype=np.uint32))
    bonds.insert("type", [label])
    frame = molrs.Frame()
    frame["atoms"] = atoms
    frame["bonds"] = bonds
    return frame


def test_potential_compiler_constructs_from_a_forcefield():
    compiler = molrs.ff.PotentialCompiler(_bond_ff())
    assert isinstance(compiler, molrs.ff.PotentialCompiler)


def test_compile_returns_potentials():
    pots = molrs.ff.PotentialCompiler(_bond_ff()).compile(_bonded_pair())
    assert isinstance(pots, molrs.ff.Potentials)
    assert len(pots) == 1


def test_compile_none_raises_type_error():
    compiler = molrs.ff.PotentialCompiler(_bond_ff())
    with pytest.raises(TypeError):
        compiler.compile(None)  # type: ignore[arg-type]


def test_defer_returns_empty_potentials_that_bind_on_evaluation():
    compiler = molrs.ff.PotentialCompiler(_bond_ff())
    deferred = compiler.defer()
    assert isinstance(deferred, molrs.ff.Potentials)
    assert len(deferred) == 0
    frame = _bonded_pair()
    assert deferred.calc_energy(frame) == compiler.compile(frame).calc_energy(frame)


def test_compile_typed_returns_typed_potentials():
    from molrs._lib import TypedPotentials

    typed = molrs.ff.PotentialCompiler(_bond_ff()).compile_typed(_bonded_pair())
    assert isinstance(typed, TypedPotentials)
    assert len(typed) == 1


def test_compile_unknown_type_label_raises_value_error():
    compiler = molrs.ff.PotentialCompiler(_bond_ff())
    with pytest.raises(ValueError):
        compiler.compile(_bonded_pair("XX-XX"))


def test_potential_compiler_has_one_public_path():
    assert hasattr(molrs.ff, "PotentialCompiler")
    assert not hasattr(molrs.ff.potential, "PotentialCompiler")


@pytest.mark.parametrize("method", ["to_potentials", "to_typed_potentials"])
def test_forcefield_has_no_compile_method(method):
    from molrs._lib import ForceField as RsForceField

    assert not hasattr(molrs.ff.ForceField, method)
    assert not hasattr(RsForceField, method)
