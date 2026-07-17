"""Every public typifier shares the nominal molrs Typifier base."""

from __future__ import annotations

import molrs


def test_native_typifiers_inherit_the_public_base():
    for cls in (
        molrs.OPLSAATypifier,
        molrs.MMFF94Typifier,
        molrs.MMFF94STypifier,
        molrs.AtdTypifier,
    ):
        assert issubclass(cls, molrs.Typifier)
        instance = cls() if cls is not molrs.AtdTypifier else cls(parameter_set="gaff")
        assert isinstance(instance, molrs.Typifier)


def test_python_typifier_can_inherit_the_native_base():
    class IdentityTypifier(molrs.Typifier):
        def typify(self, graph):
            return graph.copy()

    graph = molrs.Atomistic()
    assert isinstance(IdentityTypifier(), molrs.Typifier)
    assert isinstance(IdentityTypifier().typify(graph), molrs.Atomistic)


def test_typifier_module_and_root_export_the_same_base():
    assert molrs.typifier.Typifier is molrs.Typifier
