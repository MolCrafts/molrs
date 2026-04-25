# molcrafts-molrs-compute

Trajectory analysis for molrs: RDF, MSD, clustering, gyration / inertia tensors, PCA, k-means — built around a single unified `Compute` trait and a lightweight typed DAG.

## Core ideas

```text
         ┌────────────┐
frames ──┤            │
         │            ├──▶ Cluster ──┬──▶ COM ──┬──▶ Rg
nlists ──┤   Graph    │              │         │
         │            │              │         └──▶ Inertia
masses ──┤            │              │
         │            │              └──▶ GyrationTensor
         └────────────┘
```

1. **`Compute`** — every analysis implements this trait:

   ```rust
   pub trait Compute {
       type Args<'a>;
       type Output: ComputeResult + Clone + Send + Sync + 'static;
       fn compute<'a, FA: FrameAccess + 'a>(
           &self,
           frames: &[&'a FA],
           args: Self::Args<'a>,
       ) -> Result<Self::Output, ComputeError>;
   }
   ```

   A single frame is a length-1 slice. A trajectory is a longer slice. A dataset is still just a slice. Each impl decides how to interpret it.

2. **`Graph` / `Slot` / `Store`** — compose `Compute` nodes into a typed DAG. Each node runs exactly once per `run`, even if many downstream consumers share it (Cluster feeds Rg, Inertia, and Gyration? Cluster runs once).

3. **`ComputeResult`** — every output implements it; accumulating outputs override `finalize` to normalize. `Graph::run` calls `finalize` on each node's output once before inserting it into the `Store`.

4. **`DescriptorRow`** — flatten any Compute output into `&[F]` rows so PCA and k-means can consume arbitrary upstream per-frame outputs as a matrix.

No `Accumulator`, no `Reducer`. No hidden state in any `Compute`. No free functions pretending not to be Compute nodes.

## Usage

### Single-frame

```rust
use molrs::Frame;
use molrs::neighbors::{NeighborList, NeighborQuery};
use molrs_compute::{Graph, Inputs, RDF};

let mut g = Graph::<Frame>::new();
let nlists = g.input::<Vec<NeighborList>>();
let rdf_slot = g.add(RDF::new(40, 5.0, 0.0)?, move |s| s.get(nlists));

let frame: Frame = /* ... */;
let nlist: NeighborList = /* ... */;
let store = g.run(&[&frame], Inputs::new().with(nlists, vec![nlist]))?;
let rdf = store.get(rdf_slot);
println!("{:?}", rdf.rdf);
```

### Diamond reuse — shared intermediates run once

```rust
use molrs_compute::{
    Cluster, CenterOfMass, ClusterCenters,
    GyrationTensor, InertiaTensor, RadiusOfGyration,
    Graph, Inputs,
};

let mut g = Graph::<Frame>::new();
let nlists = g.input::<Vec<NeighborList>>();

let clusters = g.add(Cluster::new(1),        move |s| s.get(nlists));
let com      = g.add(CenterOfMass::new(),    move |s| s.get(clusters));
let centers  = g.add(ClusterCenters::new(),  move |s| s.get(clusters));

let rg       = g.add(RadiusOfGyration::new(), move |s| (s.get(clusters), s.get(com)));
let inertia  = g.add(InertiaTensor::new(),    move |s| (s.get(clusters), s.get(com)));
let gyration = g.add(GyrationTensor::new(),   move |s| (s.get(clusters), s.get(centers)));

// One run. Cluster runs once, COM runs once, ClusterCenters runs once.
let store = g.run(&frames, Inputs::new().with(nlists, per_frame_nlists))?;
```

### PCA + k-means on a dataset

```rust
use molrs_compute::{Graph, Inputs, RadiusOfGyration, Cluster, CenterOfMass, Pca2, KMeans};

let mut g = Graph::<Frame>::new();
let nlists = g.input::<Vec<NeighborList>>();
let clusters = g.add(Cluster::new(1),     move |s| s.get(nlists));
let com      = g.add(CenterOfMass::new(), move |s| s.get(clusters));
let rg       = g.add(RadiusOfGyration::new(), move |s| (s.get(clusters), s.get(com)));

// PCA consumes per-frame descriptor rows. `RgResult` impls `DescriptorRow`.
let pca    = g.add(Pca2::<molrs_compute::RgResult>::new(), move |s| s.get(rg));
let labels = g.add(KMeans::new(3, 100, 42)?, move |s| s.get(pca));
```

## Available analyses

| Module | Args | Output |
|--------|------|--------|
| `rdf` | `&Vec<NeighborList>` | `RDFResult` |
| `msd` | `()` | `MSDTimeSeries` |
| `cluster` | `&Vec<NeighborList>` | `Vec<ClusterResult>` |
| `cluster_centers` | `&Vec<ClusterResult>` | `Vec<ClusterCentersResult>` |
| `center_of_mass` | `&Vec<ClusterResult>` | `Vec<COMResult>` |
| `radius_of_gyration` | `(&Vec<ClusterResult>, &Vec<COMResult>)` | `Vec<RgResult>` |
| `inertia_tensor` | `(&Vec<ClusterResult>, &Vec<COMResult>)` | `Vec<InertiaTensorResult>` |
| `gyration_tensor` | `(&Vec<ClusterResult>, &Vec<ClusterCentersResult>)` | `Vec<GyrationTensorResult>` |
| `pca` | `&Vec<T: DescriptorRow>` | `PcaResult` |
| `kmeans` | `&PcaResult` | `KMeansResult` |

## Testing

```
cargo test -p molcrafts-molrs-compute
cargo bench -p molcrafts-molrs-compute
```

Integration tests in `tests/graph_tests.rs` verify:

- Diamond reuse: Cluster + COM + ClusterCenters each run exactly once when three leaves (Rg, Inertia, Gyration) depend on them.
- Node errors are wrapped in `ComputeError::Node { node_id, source }`.
- Missing inputs return `ComputeError::MissingInput`.

## Benchmarks

Criterion benches are organized one file per compute kernel. Each kernel file
holds two sweep functions pulling their axes from `helpers.rs`:

- **`<kernel>/size_sweep/{N}`** — `N` ∈ `{100, 500, 2 000, 10 000, 50 000}` at
  constant liquid density (ρ = 0.03 Å⁻³), 4 frames per point. Exposes
  single-threaded per-particle scaling.
- **`<kernel>/frame_sweep/{n_frames}`** — `N = 5 000`, `n_frames` ∈
  `{1, 2, 4, 8, 16, 32, 64}`. Exposes rayon parallel scaling across frames.

Layout:

```
benches/
├── benchmarks.rs            # criterion_main!
├── helpers.rs               # shared sweep axes + fixture builders
├── rdf.rs                   # rdf/size_sweep, rdf/frame_sweep
├── cluster.rs               # cluster/…
├── cluster_centers.rs
├── center_of_mass.rs
├── gyration_tensor.rs
├── inertia_tensor.rs
├── radius_of_gyration.rs
├── msd.rs
└── graph.rs                 # graph/diamond + graph/single_node_rdf gates
```

Narrow to a single kernel or axis:

```
cargo bench -p molcrafts-molrs-compute -- rdf
cargo bench -p molcrafts-molrs-compute -- cluster/frame_sweep
cargo bench -p molcrafts-molrs-compute -- gyration_tensor/size_sweep/50000
```

`graph.rs` keeps two separate perf gates (not part of the per-kernel sweep):

- Graph diamond vs. manual cascade ≤ 0.7× (sharing actually pays off).
- Graph single-node vs. direct call ≤ 1.05× (Graph overhead bound).
