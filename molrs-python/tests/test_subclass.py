"""Verify that Python subclasses of molrs PyO3 classes are allowed.

This is a contract test for downstream packages (e.g. molcrafts-molpy) that
inherit from molrs primitives instead of wrapping them. If a class loses its
``subclass`` flag in a future molrs release, these tests fail loudly.
"""

import numpy as np
import pytest

import molrs


class TestBoxSubclass:
    """``class Sub(molrs.Box)`` must instantiate and inherit base methods."""

    def test_subclass_can_be_defined(self):
        class Sub(molrs.Box):
            pass

        assert issubclass(Sub, molrs.Box)

    def test_subclass_instance_is_a_box(self):
        class Sub(molrs.Box):
            pass

        h = np.eye(3) * 10.0
        instance = Sub(h)
        assert isinstance(instance, molrs.Box)
        assert isinstance(instance, Sub)

    def test_subclass_inherits_methods(self):
        class Sub(molrs.Box):
            pass

        instance = Sub(np.eye(3) * 10.0)
        # Inherited from molrs.Box.
        assert instance.volume() == pytest.approx(1000.0)

    def test_subclass_can_add_python_attributes(self):
        # PyO3 `#[new]` is the binding constructor, so a subclass that wants
        # additional kwargs must override __new__ to strip them before
        # delegating. Subsequent Python attribute assignment is unrestricted.
        class Sub(molrs.Box):
            def __new__(cls, h, *, label):
                instance = super().__new__(cls, h)
                instance.label = label
                return instance

        instance = Sub(np.eye(3) * 5.0, label="cube-5")
        assert instance.label == "cube-5"
        assert instance.volume() == pytest.approx(125.0)

    def test_subclass_can_override_repr(self):
        class Sub(molrs.Box):
            def __repr__(self):
                return "<Sub>"

        instance = Sub(np.eye(3) * 2.0)
        assert repr(instance) == "<Sub>"


class TestFrameSubclass:
    """``class Sub(molrs.Frame)`` must instantiate and inherit base methods."""

    def test_subclass_can_be_defined(self):
        class Sub(molrs.Frame):
            pass

        assert issubclass(Sub, molrs.Frame)

    def test_subclass_instance_is_a_frame(self):
        class Sub(molrs.Frame):
            pass

        instance = Sub()
        assert isinstance(instance, molrs.Frame)
        assert isinstance(instance, Sub)

    def test_subclass_inherits_methods(self):
        class Sub(molrs.Frame):
            pass

        instance = Sub()
        instance["atoms"] = molrs.Block()
        assert "atoms" in instance
        assert list(instance.keys()) == ["atoms"]

    def test_subclass_can_add_python_attributes(self):
        class Sub(molrs.Frame):
            def __new__(cls, *, label):
                instance = super().__new__(cls)
                instance.label = label
                return instance

            def __init__(self, *, label):
                super().__init__()

        instance = Sub(label="frame-A")
        assert instance.label == "frame-A"
        assert isinstance(instance, molrs.Frame)

    def test_subclass_accepted_by_molrs_api(self, tmp_path):
        """Subclass instance must extract through PyO3 FromPyObject without TypeError.

        Exercised via ``molrs.io.raw.write_xyz`` (signature takes ``Frame``); if the
        downcast rejects the subclass we get ``TypeError`` at the call site.
        """

        class Sub(molrs.Frame):
            pass

        instance = Sub()
        atoms = molrs.Block()
        atoms.insert("symbol", ["H"])
        atoms.insert("x", np.array([0.0], dtype=np.float32))
        atoms.insert("y", np.array([0.0], dtype=np.float32))
        atoms.insert("z", np.array([0.0], dtype=np.float32))
        instance["atoms"] = atoms

        out = tmp_path / "sub.xyz"
        molrs.io.raw.write_xyz(str(out), instance)
        assert out.exists()


class TestBlockSubclass:
    """``class Sub(molrs.Block)`` must instantiate and inherit base methods."""

    def test_subclass_can_be_defined(self):
        class Sub(molrs.Block):
            pass

        assert issubclass(Sub, molrs.Block)

    def test_subclass_instance_is_a_block(self):
        class Sub(molrs.Block):
            pass

        instance = Sub()
        assert isinstance(instance, molrs.Block)
        assert isinstance(instance, Sub)

    def test_subclass_inherits_methods(self):
        class Sub(molrs.Block):
            pass

        instance = Sub()
        instance.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert instance.nrows == 3
        assert "x" in instance

    def test_subclass_accepted_by_molrs_api(self):
        """Subclass instance must extract through PyO3 FromPyObject without TypeError.

        Exercised via ``Frame.__setitem__`` (signature takes ``Block``); if the
        downcast rejects the subclass we get ``TypeError`` at assignment.
        """

        class Sub(molrs.Block):
            pass

        sub = Sub()
        sub.insert("x", np.array([1.0, 2.0], dtype=np.float32))

        frame = molrs.Frame()
        frame["atoms"] = sub  # would raise TypeError if Sub were rejected
        assert "atoms" in frame
        assert frame["atoms"].nrows == 2
