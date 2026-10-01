"""Verify that Python subclasses of ``molrs.Box`` are allowed.

molpy's ``Box`` subclasses the native one. ``Frame`` and ``Block`` are not
subclassable: there is one class each (see test_frame.py / test_block.py).
"""

import molrs
import numpy as np
import pytest


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
