"""`_lib.pyi` declares every compiled export, with the same parameter names.

The stub is hand-maintained and is what static tools and the docs build read
instead of the compiled module. This is the freshness guard the contributing
guide points at: it fails when a ``#[pyclass]`` / ``#[pyfunction]`` lands
without its declaration, and when a parameter is renamed on one side only.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import ModuleType

import molrs
from molrs import _lib

STUB = Path(inspect.getfile(molrs)).parent / "_lib.pyi"

# Dunders whose parameter names PyO3 generates ("value", "key", …); the stub
# spells them the way the data model does. Nothing calls them by keyword.
_GENERATED_DUNDERS = frozenset(
    {
        "__and__",
        "__or__",
        "__add__",
        "__sub__",
        "__mul__",
        "__rmul__",
        "__truediv__",
        "__getitem__",
        "__setitem__",
        "__eq__",
        "__ror__",
        "__ior__",
    }
)


def _public(name: str) -> str:
    """A leading underscore marks a parameter unused *in Rust*, not a rename."""
    return name.lstrip("_")


def _stub_tree() -> ast.Module:
    return ast.parse(STUB.read_text())


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


class TestStubDeclaresEveryExport:
    def test_every_lib_export_is_declared(self) -> None:
        declared = _top_level_names(_stub_tree())
        exported = {
            name
            for name in dir(_lib)
            if not name.startswith("__")
            and not isinstance(getattr(_lib, name), ModuleType)
        }
        assert not sorted(exported - declared)


class TestStubParameterNames:
    def test_declared_parameters_match_the_compiled_signature(self) -> None:
        mismatches = _collect_mismatches()
        assert not mismatches


def _runtime_params(obj: object) -> tuple[list[str], list[str]] | None:
    try:
        sig = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    positional: list[str] = []
    keyword: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            positional.append(name)
        elif param.kind == param.KEYWORD_ONLY:
            keyword.append(name)
        else:  # *args / **kwargs — the signature carries no names to compare
            return None
    return positional, keyword


def _stub_params(node: ast.FunctionDef) -> tuple[list[str], list[str]]:
    args = node.args
    if args.vararg or args.kwarg:
        return [], []
    positional = [a.arg for a in args.posonlyargs + args.args]
    return positional, [a.arg for a in args.kwonlyargs]


def _is_property_or_overload(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id in {"property", "overload"}:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "setter":
            return True
    return False


def _collect_mismatches(
    body: list[ast.stmt] | None = None,
    scope: object = _lib,
    prefix: str = "",
) -> list[str]:
    if body is None:
        body = _stub_tree().body
    out: list[str] = []
    for node in body:
        if isinstance(node, ast.ClassDef):
            nested = getattr(scope, node.name, None)
            if nested is not None:
                out += _collect_mismatches(node.body, nested, f"{prefix}{node.name}.")
        elif isinstance(node, ast.FunctionDef):
            if _is_property_or_overload(node) or node.name in _GENERATED_DUNDERS:
                continue
            runtime = (
                scope if node.name == "__init__" else getattr(scope, node.name, None)
            )
            if runtime is None:
                continue
            params = _runtime_params(runtime)
            if params is None:
                continue
            stub_pos, stub_kw = _stub_params(node)
            if not stub_pos and not stub_kw:
                continue
            stub_pos = [p for p in stub_pos if p not in {"self", "cls"}]
            rt_pos = [_public(p) for p in params[0] if p not in {"self", "cls"}]
            rt_kw = [_public(p) for p in params[1]]
            if stub_pos != rt_pos or sorted(stub_kw) != sorted(rt_kw):
                out.append(
                    f"{prefix}{node.name}: stub {(stub_pos, stub_kw)} != "
                    f"compiled {(rt_pos, rt_kw)}"
                )
    return out
