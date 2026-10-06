"""One public path per symbol, mirroring the Rust owner.

``molrs`` holds the subsystems and nothing else (as the Rust crate root
does); every class and function is reachable at exactly one public path, and
a class's ``__module__`` is that path, so ``repr``, pickle and the docs all
name it the way users import it.
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


def _walk(module: ModuleType, path: str, out: dict[int, list[str]]) -> None:
    for name in module.__all__:
        value = getattr(module, name)
        if inspect.ismodule(value):
            _walk(value, f"{path}.{name}", out)
        elif inspect.isclass(value) or callable(value):
            out.setdefault(id(value), []).append(f"{path}.{name}")


def _public() -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    _walk(molrs, "molrs", out)
    return out


def test_the_top_level_is_the_subsystems():
    assert set(molrs.__all__) == SUBSYSTEMS
    public = {n for n in dir(molrs) if not n.startswith("_")}
    assert public == SUBSYSTEMS


def test_every_symbol_has_one_public_path():
    duplicated = [paths for paths in _public().values() if len(paths) > 1]
    assert not duplicated


def test_a_class_names_its_public_module():
    for paths in _public().values():
        (path,) = paths
        module, _, name = path.rpartition(".")
        value = getattr(__import__(module, fromlist=[name]), name)
        if inspect.isclass(value):
            assert value.__module__ == module, (path, value.__module__)


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
