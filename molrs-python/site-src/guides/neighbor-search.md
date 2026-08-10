# Neighbor Search

Neighbor search turns coordinates into pairs. Many molecular analyses are
defined over distances within a cutoff: radial distribution functions count
pairs by radius, cluster analysis connects nearby particles, and contact
queries compare one set of points against another. molrs uses a PBC-aware cell
partition under the cell-list neighbor search for the common case where the
cutoff is small relative to the system size.

The key inputs are a coordinate array, a cutoff, and a boundary model. With a
periodic `Box`, distances use minimum-image behavior where requested. Without a
box, molrs can treat the data as a free-boundary system and construct a
non-periodic bounding volume around the points.

## Two types: the engine and the table

`NeighborList` is the **engine**. It owns the cutoff and the backend that
indexes space, and it keeps indexing separate from enumeration: `build` and
`update` place the coordinates in space and produce no pairs at all, while
`neighbors()` materializes the pairs into a table.

`Neighbors` is that **table**: read-only columns, one row per pair.

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

nl = molrs.NeighborList(1.0)      # O(N) cell list — the production backend
nl.build(points, box)             # index only; no pair table yet
neigh = nl.neighbors()            # materialize

print("pairs:", neigh.n_pairs)
print(neigh.query_point_indices(), neigh.point_indices())
print(np.sqrt(neigh.dist_sq()))   # distances in Å
```

The pair `(0, 1)` is found because the box is periodic: the minimum-image
distance across the boundary is `0.2`, not `9.8`. The pair `(2, 3)` is found
because the Cartesian distance is `0.8`.

`NeighborList.brute_force(cutoff)` builds the same engine over the O(N²)
all-pairs backend. It finds exactly the same pairs, which is what makes it
useful as a reference; prefer it only for very small systems or to check the
cell list.

When the positions move but the box does not, `nl.update(new_points)`
re-indexes in the box captured by the last `build`. There is no skin: `update`
re-indexes every time rather than deciding for you that the previous index was
still good enough. If the *box* changed (a barostat), call `build` again —
`update` would fold minimum images against a stale cell. Calling `update`
before any `build` raises `ValueError`, because the box is then unknown and
guessing one silently changes the answer.

## Half-shell, and what "full" means

A **self** search is half-shell: each unordered pair appears exactly once, with
`i < j`, and never as `i == j`. So `n_pairs` is the number of unordered pairs,
not twice it, and an analysis that wants both orderings supplies the factor of
two itself.

`neighbors()` keeps both physical columns by default. That default names the
**columns** a table carries — it is *not* a bidirectional (full-shell) pair
list. Column policy and pair direction are independent choices:

```python
neigh = nl.neighbors()                            # dist_sq + disp (the default)
lean = nl.neighbors(dist_sq=True, disp=False)     # indices + d² only
assert lean.disp() is None                        # not stored, so not readable
```

A column that was not stored reads back as `None`, never as a column of zeros:
a zero displacement is a physically meaningful value (two coincident
particles), so fabricating one would turn a missing column into a wrong answer.
A dropped column cannot be added afterwards — rerun the search with it enabled.

The two physical columns are

| accessor | quantity | unit |
|---|---|---|
| `dist_sq()` | squared minimum-image distance | Å² |
| `disp()` | minimum-image displacement `r_j - r_i`, **unnormalized** | Å |

`disp` is not divided by the distance: its length *is* the pair distance, and
it points from `i` to `j`, so swapping the indices flips its sign. Both columns
come from the same minimum-image evaluation, so `dist_sq == |disp|²` always.
The square root is left to the call site (`np.sqrt(neigh.dist_sq())`) rather
than hidden inside an accessor.

## Cross query

A cross query compares reference points with a separate query set. This is the
right shape for solute-solvent contacts or "which atoms are near this probe?"
questions, and it is what `NeighborQuery` is for. It is directed: every query
point reports all of its reference neighbors, with no `i < j` rule.

```python
nq = molrs.NeighborQuery(box, points, cutoff=1.0)

query_points = np.array(
    [
        [0.2, 0.0, 0.0],
        [8.0, 8.0, 8.0],
    ],
    dtype=np.float64,
)

cross = nq.query(query_points)
print("query indices:", cross.query_point_indices())
print("point indices:", cross.point_indices())
print("is_self_query:", cross.is_self_query)   # False
```

For each row `k`, `query_point_indices()[k]` indexes the query array and
`point_indices()[k]` indexes the original reference points used to construct
`NeighborQuery`. `NeighborQuery.query_self()` answers the half-shell self
question over the reference set, and `NeighborQuery.free(points, cutoff)`
derives a non-periodic bounding box for isolated coordinates.

## Feeding an analysis

Analyses consume the materialized table. That keeps the cutoff, the
periodicity, and the self-vs-cross decision outside the analysis object.

```python
frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("x", points[:, 0])
atoms.insert("y", points[:, 1])
atoms.insert("z", points[:, 2])
atoms.insert("element", ["C", "C", "C", "C"])
frame["atoms"] = atoms
frame.box = box

from molrs.compute.density import RDF
rdf = RDF(n_bins=20, r_max=1.0)
result = rdf.compute(frame, neigh)
print(result.bin_centers[:3])
print(result.rdf[:3])
```

Order parameters such as `Steinhardt` need the *direction* to each neighbor,
so they read the `disp` column. The default `neighbors()` supplies it; a table
materialized with `disp=False` is rejected by name rather than silently
producing zeros.

If results are surprising, check the neighbor table first. A wrong cutoff or a
missing box is usually visible in the pair count before it shows up as a
confusing distribution.

## Name mapping from freud

| freud | molrs |
|---|---|
| `freud.locality.LinkCell` / `AABBQuery` | `NeighborList` (cutoff search) |
| `NeighborList.query_point_indices` | `Neighbors.query_point_indices()` |
| `NeighborList.point_indices` | `Neighbors.point_indices()` |
| `NeighborList.distances` (r) | `Neighbors.dist_sq()` (r²) — take `np.sqrt` yourself |
| `NeighborList.vectors` | `Neighbors.disp()` — same unnormalized MIC vector |

Note the vocabulary difference: freud's `NeighborList` is the *result*, while
in molrs `NeighborList` is the *search* and `Neighbors` is the result.

## Removed names

`molrs.LinkedCell` is gone: the backend is a constructor on the engine
(`NeighborList(cutoff)` for the cell list, `NeighborList.brute_force(cutoff)`
for the O(N²) reference), not a separate class. The old
`NeighborList.distances` and `NeighborList.pairs()` accessors are gone too —
read `np.sqrt(dist_sq())` and the two index columns instead.
