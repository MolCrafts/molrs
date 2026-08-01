"""Analysis routines, organized by domain.

One subpackage per ``molrs::compute`` module, so the Python path and the Rust
path are the same word: ``molrs.compute.transport`` is ``compute::transport``.
This is the single public path for analysis. The flat ``molrs.<Class>``
spellings still import — PyO3 declares ``module = "molrs"`` — but they are
plumbing, not the documented surface.

Two Rust modules deliberately have no subpackage here. ``compute::rdf`` and
``compute::shape`` are internal groupings whose types surface under the
freud-parity names callers expect: ``RDF`` under :mod:`~molrs.compute.density`,
and the gyration / inertia / radius-of-gyration tensors under
:mod:`~molrs.compute.cluster`. Mirroring them as their own subpackages would
break the freud port path this layout exists to preserve.
"""

from . import (
    check,
    cluster,
    density,
    dielectric,
    diffraction,
    distribution,
    dynamics,
    environment,
    fit,
    hbond,
    ml,
    msd,
    order,
    pmft,
    spectroscopy,
    transport,
    voronoi,
)

__all__ = [
    "check",
    "cluster",
    "density",
    "dielectric",
    "diffraction",
    "distribution",
    "dynamics",
    "environment",
    "fit",
    "hbond",
    "ml",
    "msd",
    "order",
    "pmft",
    "spectroscopy",
    "transport",
    "voronoi",
]
