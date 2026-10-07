"""One public path per symbol, mirroring the Rust owner.

``molrs`` holds the subsystems and nothing else (as the Rust crate root
does); every class, function and constant is reachable at exactly one public
path, and a class's or function's ``__module__`` and ``__name__`` are that
path, so ``repr``, pickle and the docs all name it the way users import it.

The walk is over ``dir()``, not only ``__all__``: a public module exports
exactly its ``__all__`` and nothing else, so a ``typing`` or stdlib name
imported for an annotation cannot leak into it as a second public spelling.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import molrs
import pytest

SUBSYSTEMS = {
    "builder",
    "compute",
    "conformer",
    "ff",
    "io",
    "md",
    "op",
    "optimize",
    "perceive",
    "signal",
    "spatial",
    "store",
    "stream",
    "system",
    "units",
}


def _public_names(module: ModuleType) -> set[str]:
    return {name for name in dir(module) if not name.startswith("_")}


def _modules() -> dict[str, ModuleType]:
    """Every public module, by its path."""
    out: dict[str, ModuleType] = {}

    def walk(module: ModuleType, path: str) -> None:
        out[path] = module
        for name in module.__all__:
            value = getattr(module, name)
            if inspect.ismodule(value):
                walk(value, f"{path}.{name}")

    walk(molrs, "molrs")
    return out


def _objects() -> dict[str, object]:
    """Every public class, function and constant, by its public path."""
    return {
        f"{path}.{name}": getattr(module, name)
        for path, module in _modules().items()
        for name in module.__all__
        if not inspect.ismodule(getattr(module, name))
    }


def _named(value: object) -> bool:
    """A class or a function: an object that states its own path."""
    return inspect.isclass(value) or inspect.isroutine(value)


def test_the_top_level_is_the_subsystems():
    assert set(molrs.__all__) == SUBSYSTEMS
    assert _public_names(molrs) == SUBSYSTEMS


@pytest.mark.parametrize("path", sorted(_modules()))
def test_a_module_exports_exactly_its_all(path):
    module = _modules()[path]
    assert module.__name__ == path
    assert len(module.__all__) == len(set(module.__all__)), path
    assert _public_names(module) == set(module.__all__)


def test_every_class_and_function_has_one_public_path():
    paths: dict[int, list[str]] = {}
    for path, value in _objects().items():
        if _named(value):
            paths.setdefault(id(value), []).append(path)
    assert not [p for p in paths.values() if len(p) > 1]


def test_a_class_or_function_names_its_public_path():
    wrong = [
        (path, f"{value.__module__}.{value.__name__}")
        for path, value in _objects().items()
        if _named(value) and f"{value.__module__}.{value.__name__}" != path
    ]
    assert not wrong


def test_a_constant_is_a_value_not_a_second_door():
    """A public non-class, non-function name is plain data: it re-exports no
    class or function under a second name."""
    callables = [
        path
        for path, value in _objects().items()
        if not _named(value) and callable(value)
    ]
    assert not callables


@pytest.mark.parametrize(
    "gone",
    [
        "molrs.fields",
        "molrs.io.raw",
        "molrs.keys",
        "molrs.schema",
        "molrs.md.driver",
        "molrs.compute.protocol",
        "molrs.compute.density",
        "molrs.ff.potential.protocol",
        "molrs.io._trajectory",
    ],
)
def test_retired_modules_do_not_import(gone):
    with pytest.raises(ModuleNotFoundError):
        __import__(gone)


@pytest.mark.parametrize(
    "gone",
    [
        # 0.16: the top level is the subsystems only.
        "molrs.Block",
        "molrs.Box",
        "molrs.Frame",
        "molrs.Atomistic",
        "molrs.Element",
        "molrs.Trajectory",
        "molrs.fields",
        # MD integrates potentials; it defines none.
        "molrs.md.LJCut",
        "molrs.md.Potential",
        "molrs.md.Potentials",
        "molrs.md.kernel",
        # molrs.ff holds only its submodules.
        "molrs.ff.ForceField",
        "molrs.ff.Potential",
        # io: no raw layer, no aliases.
        "molrs.io.raw",
        "molrs.io.write_smiles",
        "molrs.io.parse_lammps_log_text",
        # Every file reader / writer is a function at the top of molrs.io:
        # the mrec doors are flat there ...
        "molrs.io.mrec.read",
        "molrs.io.mrec.write",
        "molrs.io.mrec.read_system",
        "molrs.io.mrec.write_system",
        "molrs.io.mrec.read_trajectory",
        "molrs.io.mrec.write_trajectory",
        "molrs.io.mrec.read_forcefield",
        "molrs.io.mrec.write_forcefield",
        "molrs.io.mrec.read_meta",
        "molrs.io.mrec_sections",
        # ... and its store reader / writer are named as in Rust.
        "molrs.io.mrec.FrameSequence",
        "molrs.io.mrec.FrameSequenceWriter",
        "molrs.io.mrec.TrajectoryReader",
        "molrs.io.mrec.TrajectoryWriter",
        # The frame-bytes codec is io's, not stream's.
        "molrs.stream.read_frame_bytes",
        "molrs.stream.write_frame_bytes",
        # Force-field files are io's; ff.forcefield is the data model only.
        "molrs.ff.forcefield.read_lammps_forcefield",
        "molrs.ff.forcefield.read_lammps_data_coeffs",
        "molrs.ff.forcefield.read_lammps_cmap",
        "molrs.ff.forcefield.read_gromacs_top_ff",
        "molrs.ff.forcefield.read_gromacs_system",
        "molrs.ff.forcefield.read_amber_prmtop_ff",
        "molrs.ff.forcefield.read_amber_prmtop_system",
        "molrs.ff.forcefield.read_forcefield_xml",
        "molrs.ff.forcefield.read_opls_xml",
        "molrs.ff.forcefield.write_lammps_forcefield",
        "molrs.ff.forcefield.write_lammps_forcefield_str",
        "molrs.ff.forcefield.write_lammps_data_coeffs",
        "molrs.ff.forcefield.write_lammps_cmap",
        "molrs.ff.forcefield.write_gromacs_top_ff",
        "molrs.ff.forcefield.write_gromacs_system",
        "molrs.ff.forcefield.write_amber_frcmod",
        "molrs.ff.forcefield.write_forcefield_xml",
        # A class of one format is that format's submodule's.
        "molrs.io.TrajectoryReader",
        "molrs.io.SmilesIR",
        "molrs.io.SmilesError",
        "molrs.io.CGSmilesIR",
        "molrs.io.CGGraph",
        "molrs.io.CGNode",
        "molrs.io.CGEdge",
        "molrs.io.CGFragmentDef",
        "molrs.io.ResolvedPair",
        "molrs.io.PairEnd",
        "molrs.io.BondingDescriptor",
        "molrs.io.BondReactTemplate",
        "molrs.io.LammpsLog",
        "molrs.io.LammpsLogHeader",
        "molrs.io.LammpsRun",
        "molrs.io.LammpsThermo",
        "molrs.io.LammpsWarning",
        "molrs.io.LammpsPerformance",
        "molrs.io.LammpsTimingBreakdown",
        "molrs.io.LammpsTimingRow",
        "molrs.io.LammpsCpuUse",
        "molrs.io.LammpsLoadBalance",
        "molrs.io.LammpsLoopTime",
        "molrs.io.LammpsMemoryUsage",
        "molrs.io.LammpsNeighborStatistics",
        # Second doors on a class.
        "molrs.store.Trajectory.from_frames",
        "molrs.store.Trajectory.count_frames",
        "molrs.perceive.SmartsMatch.as_list",
        "molrs.perceive.SmartsMatch.as_dict",
    ],
)
def test_retired_names_are_absent(gone):
    owner_path, _, name = gone.rpartition(".")
    owner: object = molrs
    for part in owner_path.split(".")[1:]:
        owner = getattr(owner, part)
    assert not hasattr(owner, name), gone


# --- One shape per file-format factory --------------------------------------
#
# A factory that reads or writes a file format is either a function at the
# top of molrs.io (``read_<fmt>[_<what>]`` / ``write_<fmt>[_<what>]``) or a
# class of the format's own submodule (``molrs.io.<fmt>.<Fmt>Reader`` /
# ``<Fmt>Writer``). Nothing else may carry those names.


def _factory_name(name: str) -> bool:
    return name.startswith(("read_", "write_"))


def test_read_and_write_functions_live_at_the_top_of_io():
    stray = sorted(
        path
        for path, value in _objects().items()
        if callable(value)
        and _factory_name(path.rpartition(".")[2])
        and path.rpartition(".")[0] != "molrs.io"
    )
    assert not stray


def test_no_class_hides_a_read_or_write_factory():
    """A ``read_*`` / ``write_*`` static or class method is a second door
    onto a format: the door is a function of molrs.io."""
    doors = []
    for path, value in _objects().items():
        if not inspect.isclass(value):
            continue
        for name, attr in vars(value).items():
            if _factory_name(name) and isinstance(attr, (staticmethod, classmethod)):
                doors.append(f"{path}.{name}")
    assert not doors


def test_readers_and_writers_are_classes_of_a_format_submodule():
    stray = []
    for path, value in _objects().items():
        if not inspect.isclass(value) or not value.__name__.endswith(("Reader", "Writer")):
            continue
        owner = path.rpartition(".")[0]
        if not (owner.startswith("molrs.io.") and owner.count(".") == 2):
            stray.append(path)
    assert not stray


def test_the_top_of_io_is_functions_and_format_submodules():
    """A class belongs to one format, so it lives in that format's
    submodule; the top of molrs.io holds the read / write functions."""
    classes = [
        name for name in molrs.io.__all__ if inspect.isclass(getattr(molrs.io, name))
    ]
    assert not classes
    functions = [
        name for name in molrs.io.__all__ if inspect.isroutine(getattr(molrs.io, name))
    ]
    assert all(_factory_name(name) for name in functions), functions


def test_forcefield_is_the_data_model_and_stream_the_transport():
    assert not [n for n in molrs.ff.forcefield.__all__ if _factory_name(n)]
    assert set(molrs.stream.__all__) <= {"ControlCommand", "Publisher"}
    assert set(molrs.io.mrec.__all__) == {
        "ForceFieldSection",
        "MrecReader",
        "MrecWriter",
        "SequenceSchema",
        "pack",
        "schema",
        "section_names",
    }


@pytest.mark.parametrize(
    "path",
    [
        "molrs.io.read_smiles",
        "molrs.io.read_mrec",
        "molrs.io.write_mrec_trajectory",
        "molrs.io.read_frame_bytes",
        "molrs.io.write_frame_bytes",
        "molrs.io.read_lammps_forcefield",
        "molrs.io.write_forcefield_xml",
        "molrs.io.read_lammps_log_str",
        "molrs.io.trajectory.TrajectoryReader",
        "molrs.io.mrec.MrecReader",
        "molrs.io.mrec.MrecWriter",
        "molrs.io.smiles.SmilesIR",
        "molrs.io.smiles.CGSmilesIR",
        "molrs.io.log.LammpsLog",
        "molrs.io.lammps_bond_react.BondReactTemplate",
    ],
)
def test_the_one_path_exists(path):
    assert path in _objects()


def test_find_matches_has_no_mapped_shortcut():
    """``SmartsMatch.mapping`` is the one door onto the atom-map projection."""
    assert "mapped" not in molrs.perceive.SmartsPattern.find_matches.__text_signature__
