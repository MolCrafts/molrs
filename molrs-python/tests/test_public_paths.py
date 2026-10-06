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
        "molrs.io.read_frame_bytes",
        "molrs.io.write_frame_bytes",
        # Every *.mrec door is molrs.io.mrec's.
        "molrs.io.read_mrec",
        "molrs.io.write_mrec",
        "molrs.io.read_mrec_system",
        "molrs.io.write_mrec_system",
        "molrs.io.read_mrec_trajectory",
        "molrs.io.write_mrec_trajectory",
        "molrs.io.read_mrec_forcefield",
        "molrs.io.write_mrec_forcefield",
        "molrs.io.read_mrec_meta",
        "molrs.io.mrec_sections",
        # One TrajectoryReader: the mrec cursor is FrameSequence.
        "molrs.io.mrec.TrajectoryReader",
        "molrs.io.mrec.TrajectoryWriter",
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


def test_find_matches_has_no_mapped_shortcut():
    """``SmartsMatch.mapping`` is the one door onto the atom-map projection."""
    assert "mapped" not in molrs.perceive.SmartsPattern.find_matches.__text_signature__
