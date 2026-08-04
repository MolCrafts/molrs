"""Every public typifier shares the nominal molrs Typifier base."""

from __future__ import annotations

import molrs


def test_native_typifiers_inherit_the_public_base():
    for cls in (
        molrs.ff.typifier.OPLSAATypifier,
        molrs.ff.typifier.MMFF94Typifier,
        molrs.ff.typifier.MMFF94STypifier,
        molrs.ff.typifier.AtdTypifier,
    ):
        assert issubclass(cls, molrs.ff.typifier.Typifier)
        instance = (
            cls()
            if cls is not molrs.ff.typifier.AtdTypifier
            else cls(parameter_set="gaff")
        )
        assert isinstance(instance, molrs.ff.typifier.Typifier)


def test_python_typifier_can_inherit_the_native_base():
    class IdentityTypifier(molrs.ff.typifier.Typifier):
        def typify(self, graph):
            return graph.copy()

    graph = molrs.Atomistic()
    assert isinstance(IdentityTypifier(), molrs.ff.typifier.Typifier)
    assert isinstance(IdentityTypifier().typify(graph), molrs.Atomistic)


def test_typifier_module_and_root_export_the_same_base():
    assert molrs.ff.typifier.Typifier is molrs.ff.typifier.Typifier
