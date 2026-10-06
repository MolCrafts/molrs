"""Verify that Python subclasses of ``molrs.Box`` are allowed.

molpy's ``Box`` subclasses the native one. The other core data classes
(``Frame``, ``Block``, ``Atomistic``, ``ForceField``, …) are subclassable too
(see test_frame.py, test_block.py, test_views.py, test_forcefield_builder.py).
"""

import pickle

import molrs
import numpy as np
import pytest


class _PickledBox(molrs.Box):
    """Module level, so pickle can find it."""


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

    def test_subclass_pickles_as_itself_with_its_attributes(self):
        instance = _PickledBox(np.eye(3) * 3.0)
        instance.label = "cube-3"
        back = pickle.loads(pickle.dumps(instance))
        assert type(back) is _PickledBox
        assert back.label == "cube-3"
        assert back.volume() == pytest.approx(27.0)
