"""``molrs.optimize``: the L-BFGS minimizer and its one report type.

The minimizer's numerics are proven by the Rust unit tests in
``molrs/src/optimize/``; these check the Python face: the Rust names, the
defaults read from ``LbfgsSettings::DEFAULT``, and the report fields.
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest
from molrs.ff.compile import compile_explicit_terms


def _lj_dimer():
    # One LJ pair at 1.5 sigma: the minimum is at 2^(1/6) sigma.
    return compile_explicit_terms("pair", "lj/cut", [[0, 1]], epsilon=1.0, sigma=1.0)


def test_names_are_the_rust_names():
    assert molrs.optimize.__all__ == ["Lbfgs", "OptimizationReport"]


def test_minimize_relaxes_a_dimer_to_the_lj_minimum():
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    out, report = molrs.optimize.Lbfgs(_lj_dimer(), fmax=1e-6).minimize(coords)
    assert isinstance(report, molrs.optimize.OptimizationReport)
    assert report.converged
    assert report.final_fmax < 1e-6
    assert report.final_grad_rms <= report.final_fmax
    assert np.linalg.norm(out[1] - out[0]) == pytest.approx(2 ** (1 / 6), abs=1e-5)


def test_defaults_are_the_rust_defaults():
    assert repr(molrs.optimize.Lbfgs(_lj_dimer())) == (
        "Lbfgs(fmax=0.05, max_steps=500, max_step=0.2, memory=8)"
    )
