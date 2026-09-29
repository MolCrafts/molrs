"""Density analyzers — :class:`RDF`, :class:`GaussianDensity`, :class:`LocalDensity`."""

from molrs._lib import (
    RDF as RDF,
)
from molrs._lib import (
    GaussianDensity as GaussianDensity,
)
from molrs._lib import (
    LocalDensity as LocalDensity,
)
from molrs._lib import (
    RDFResult as RDFResult,
)
from molrs._lib import (
    SpatialDistribution as SpatialDistribution,
)
from molrs._lib import (
    SpatialDistributionResult as SpatialDistributionResult,
)

__all__ = [
    "RDF",
    "GaussianDensity",
    "LocalDensity",
    "RDFResult",
    "SpatialDistribution",
    "SpatialDistributionResult",
]

# SpatialDistribution is a 3-D density field, optionally oriented via the frame's `orientations` block.
