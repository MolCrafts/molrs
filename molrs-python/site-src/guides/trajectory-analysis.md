# Trajectory Analysis

Trajectory analysis reuses the frame model over time. A trajectory is a
sequence of frames plus optional time or step arrays. Analyses such as RDF,
MSD, clustering, centers of mass, inertia tensors, gyration tensors, PCA, and
k-means all consume frames and return typed result objects.

The common pattern is to compute neighbor lists first, then pass those lists
alongside frames to analyses that need distance connectivity. That keeps the
cutoff and boundary model explicit. It also lets several analyses share the
same neighbor search result instead of recomputing pair lists independently.

## One call handles one frame or many

Python wrappers accept either one frame or a sequence of frames for several
analyses. A single frame returns a single result where that is meaningful; a
sequence returns results aligned with the input sequence. This mirrors how
users work in notebooks while preserving the Rust-side distinction between a
frame and a trajectory.

For periodic systems, attach the simulation box before computing neighbor lists
or volume-normalized quantities. For isolated molecules or finite clusters,
use the free-boundary behavior intentionally and document that choice near the
analysis.

## Worked Example: MSD Over Three Frames

This example constructs three frames with the same two particles moving along
the x-axis. MSD uses frame 0 as the reference.

```python
import numpy as np
import molrs

def make_frame(offset: float) -> molrs.Frame:
    atoms = molrs.Block()
    atoms.insert("element", ["Ar", "Ar"])
    atoms.insert("x", np.array([0.0 + offset, 1.0 + offset], dtype=np.float64))
    atoms.insert("y", np.array([0.0, 0.0], dtype=np.float64))
    atoms.insert("z", np.array([0.0, 0.0], dtype=np.float64))

    frame = molrs.Frame()
    frame["atoms"] = atoms
    frame.simbox = molrs.Box.cube(20.0)
    return frame

frames = [make_frame(0.0), make_frame(0.1), make_frame(0.2)]

from molrs.compute.msd import MSD
msd = MSD()
series = msd.compute(frames)

print("points:", len(series))
print("mean:", series.mean)
print("first frame mean:", series[0].mean)
```

The result is a time series aligned with the input frames. If you pass a single
frame, the wrapper still follows the same analysis semantics, but most
trajectory metrics are more meaningful on sequences.

## Worked Example: Cluster Analysis

Clustering consumes a neighbor list. In this toy system, two close pairs form
two clusters when the cutoff connects only local neighbors.

```python
points = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.5, 0.0, 0.0],
    ],
    dtype=np.float64,
)

atoms = molrs.Block()
atoms.insert("element", ["C", "C", "C", "C"])
atoms.insert("x", points[:, 0])
atoms.insert("y", points[:, 1])
atoms.insert("z", points[:, 2])

frame = molrs.Frame()
frame["atoms"] = atoms
frame.simbox = molrs.Box.cube(20.0)

nq = molrs.NeighborQuery(frame.simbox, points, cutoff=1.0)
nlist = nq.query_self()

from molrs.compute.cluster import Cluster, ClusterCenters
clusters = Cluster(min_cluster_size=1).compute(frame, nlist)
centers = ClusterCenters().compute(frame, clusters)

print("clusters:", clusters.num_clusters)
print("sizes:", clusters.cluster_sizes)
print("centers:", centers.centers.reshape(clusters.num_clusters, 3))
```

The neighbor list defines connectivity. Change the cutoff and you change the
graph that clustering sees.

## Transport & Electrolyte Analysis

Three further kernels target electrolyte transport properties — Onsager
coefficients (`transport.Onsager`), Green–Kubo conductivity from the current ACF
(composed from `GreenKuboConductivity` + `RunningIntegral`), and pair persistence
/ residence times (`transport.Persist`). Like the dielectric kernels they are array-based free
functions: the caller assembles the per-frame collective quantities and the Rust
layer does the windowed correlation. See the dedicated
[Transport Kernels](transport.md) reference for signatures, units, equations, and
examples.

## Choosing Batch Shape

Use one frame when you are checking a single snapshot. Use a sequence of frames
when the quantity is explicitly temporal or when you want one result per
snapshot. Keep arrays aligned: if `frames` has length `N`, a parallel list of
neighbor lists should also have length `N`.

For reproducible notebooks, print the pair counts or cluster sizes before
plotting. Small diagnostic prints make it obvious when a missing box or cutoff
change altered the analysis.
