"""Soft (penetrable) packing potential — the parameter interface for
``molrs::ff::potential::soft::SoftPotential``.

The energy kernel and neighbour-list handling live in the Rust core; this class
carries the soft-potential **parameters** (the functional form's knobs) so a
consumer — e.g. ``molpack.LBFGSMinimizer`` — can build the actual potential over
a system's topology. ``soft`` names the potential form only; it is not a
neighbour list and knows nothing about periodicity.
"""
from __future__ import annotations


class SoftPotential:
    """Parameters of the soft packing potential.

    E = k_bond (r-r0)^2 [1-2] + k_ang (r-a0)^2 [1-3]
      + A (sigma-r)^2 for r<sigma            (soft repulsion)
      - B (r-sigma)(rcut-r) for sigma<r<rcut (optional soft attraction)
    """

    __slots__ = ("sigma", "a_rep", "b_attract", "rcut", "k_bond", "k_ang")

    def __init__(self, sigma=2.6, a_rep=8.0, b_attract=0.0, rcut=5.0,
                 k_bond=50.0, k_ang=8.0):
        self.sigma = float(sigma)
        self.a_rep = float(a_rep)
        self.b_attract = float(b_attract)
        self.rcut = float(rcut)
        self.k_bond = float(k_bond)
        self.k_ang = float(k_ang)

    def __repr__(self):
        return (f"SoftPotential(sigma={self.sigma}, a_rep={self.a_rep}, "
                f"b_attract={self.b_attract}, rcut={self.rcut})")
