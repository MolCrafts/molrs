"""Density analyzers — :class:`RDF`, :class:`GaussianDensity`, :class:`LocalDensity`."""

from molrs._lib import (
    RDF as RDF,
    RDFResult as RDFResult,
    GaussianDensity as GaussianDensity,
    LocalDensity as LocalDensity,
    SpatialDistribution as SpatialDistribution,
    SpatialDistributionResult as SpatialDistributionResult,
)

__all__ = [
    "SpatialDistribution",
    "SpatialDistributionResult",
    "RDF",
    "RDFResult",
    "GaussianDensity",
    "LocalDensity",
]

# SpatialDistribution is a 3-D density field, optionally oriented via the frame's `orientations` block.
