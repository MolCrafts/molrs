# Neighbor Search

Neighbor search turns coordinates into pairs. Many molecular analyses are
defined over distances within a cutoff: radial distribution functions count
pairs by radius, cluster analysis connects nearby particles, and contact
queries compare one set of points against another. molrs uses a PBC-aware cell partition (`CellGrid`) under LinkCell neighbor
search for the common case where the cutoff is small relative to the system
size.

The key inputs are a coordinate array, a cutoff, and a boundary model. With a
periodic `Box`, distances use minimum-image behavior where requested. Without a
box, molrs can treat the data as a free-boundary system and construct a
non-periodic bounding volume around the points.

## Neighbor queries are either self or cross

A self query searches one set of points against itself and reports unique
pairs. This is the usual path for RDF and cluster analysis. A cross query
compares two different point sets, which is useful for solute-solvent contacts
or any analysis where query points and reference points have different roles.

The neighbor list records query indices, point indices, distances, and squared
distances. Keeping both index arrays makes the result independent of whether
the query was self or cross. Downstream analyses can consume the same shape of
data without special-casing how it was produced.

## Worked Example: Periodic Self Query

This example places four points in a cubic periodic box. Two points are close
in ordinary Cartesian space; two are close only through periodic wrapping.

```python
import numpy as np
import molrs

box = molrs.Box.cube(10.0)
points = np.array(
    [
        [0.1, 0.0, 0.0],
        [9.9, 0.0, 0.0],
        [4.0, 4.0, 4.0],
        [4.8, 4.0, 4.0],
    ],
    dtype=np.float64,
)

nq = molrs.NeighborQuery(box, points, cutoff=1.0)
nlist = nq.query_self()

print("pairs:", nlist.n_pairs)
print(nlist.pairs())
print(nlist.distances)
```

The pair `(0, 1)` is found because the box is periodic: the minimum-image
distance across the boundary is `0.2`, not `9.8`. The pair `(2, 3)` is found
because the Cartesian distance is `0.8`.

## Cross Query

A cross query compares reference points with a separate query set. This is the
right shape for solute-solvent contacts or "which atoms are near this probe?"
questions.

```python
query_points = np.array(
    [
        [0.2, 0.0, 0.0],
        [8.0, 8.0, 8.0],
    ],
    dtype=np.float64,
)

cross = nq.query(query_points)
print("query indices:", cross.query_point_indices)
print("point indices:", cross.point_indices)
print("distances:", cross.distances)
```

For each row in the neighbor list, `query_point_indices[k]` indexes the query
array and `point_indices[k]` indexes the original reference points used to
construct `NeighborQuery`.

## Feeding RDF

RDF consumes the explicit neighbor list. This keeps the cutoff, periodicity,
and self-vs-cross decision outside the RDF object.

```python
frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("x", points[:, 0])
atoms.insert("y", points[:, 1])
atoms.insert("z", points[:, 2])
atoms.insert("element", ["C", "C", "C", "C"])
frame["atoms"] = atoms
frame.simbox = box

from molrs.compute.density import RDF
rdf = RDF(n_bins=20, r_max=1.0)
result = rdf.compute(frame, nlist)
print(result.bin_centers[:3])
print(result.rdf[:3])
```

If RDF results are surprising, check the neighbor list first. A wrong cutoff
or missing box will usually be visible in the pair count before it shows up as
a confusing distribution.
