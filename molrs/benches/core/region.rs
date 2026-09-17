use criterion::{Criterion, criterion_group};
use molrs::spatial::TriMesh;
use molrs::spatial::region::{Polyhedron, Region, SphereUnion};
use molrs::spatial::simbox::SimBox;
use molrs::types::F;
use ndarray::{Array2, array};

use crate::helpers;

/// Deterministic uniform numbers — no rand dependency in a bench.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> F {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as F / (1u64 << 53) as F
    }
}

/// 40 000 beads in an 83.89 Å cube: the PE-in-solvent scene a packer masks
/// its lattice against, ~10⁶ queries per run.
fn bench_sphere_union(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/region/sphere_union/distance");
    helpers::configure(&mut group);

    let l = 83.89 as F;
    let n = 40_000;
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let mut centers = Array2::zeros((n, 3));
    for i in 0..n {
        for d in 0..3 {
            centers[[i, d]] = rng.next() * l;
        }
    }
    let radii = vec![3.09 as F; n];
    let probes: Vec<[F; 3]> = (0..1000)
        .map(|_| [rng.next() * l, rng.next() * l, rng.next() * l])
        .collect();

    let periodic = SimBox::cube(l, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap();
    let union = SphereUnion::new(centers.view(), &radii, &periodic).unwrap();
    group.bench_function("periodic", |bencher| {
        let mut i = 0;
        bencher.iter(|| {
            i = (i + 1) % probes.len();
            std::hint::black_box(union.distance(&probes[i]))
        });
    });

    let free = SphereUnion::free(centers.view(), &radii).unwrap();
    group.bench_function("free", |bencher| {
        let mut i = 0;
        bencher.iter(|| {
            i = (i + 1) % probes.len();
            std::hint::black_box(free.distance(&probes[i]))
        });
    });

    group.finish();
}

/// A cube subdivided into ~10⁴ faces: the closest-point and parity walks at
/// the face count of a mesh from a mesher.
fn bench_polyhedron(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/region/polyhedron/distance");
    helpers::configure(&mut group);

    let n = 40; // 40×40 quads per face × 2 tris × 6 faces = 19 200 triangles
    let edge = 100.0 as F;
    let h = edge / n as F;
    let mut tris: Vec<[[F; 3]; 3]> = Vec::with_capacity(6 * 2 * n * n);
    // Each face: (fixed axis, value, outward sign) with the two in-plane axes.
    for (axis, value, sign) in [
        (0, 0.0, -1.0),
        (0, edge, 1.0),
        (1, 0.0, -1.0),
        (1, edge, 1.0),
        (2, 0.0, -1.0),
        (2, edge, 1.0),
    ] {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        for i in 0..n {
            for j in 0..n {
                let p = |a: usize, b: usize| {
                    let mut q = [0.0 as F; 3];
                    q[axis] = value;
                    q[u] = a as F * h;
                    q[v] = b as F * h;
                    q
                };
                let (a, b, c2, d) = (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
                if sign > 0.0 {
                    tris.push([a, b, c2]);
                    tris.push([a, c2, d]);
                } else {
                    tris.push([a, c2, b]);
                    tris.push([a, d, c2]);
                }
            }
        }
    }
    let region = Polyhedron::new(TriMesh::from_triangles(&tris)).expect("closed cube");
    let mut rng = Lcg(0x2545_F491_4F6C_DD1D);
    let probes: Vec<[F; 3]> = (0..1000)
        .map(|_| {
            [
                rng.next() * 1.4 * edge - 0.2 * edge,
                rng.next() * 1.4 * edge - 0.2 * edge,
                rng.next() * 1.4 * edge - 0.2 * edge,
            ]
        })
        .collect();

    group.bench_function("subdivided_cube", |bencher| {
        let mut i = 0;
        bencher.iter(|| {
            i = (i + 1) % probes.len();
            std::hint::black_box(region.distance(&probes[i]))
        });
    });
    group.bench_function("subdivided_cube_contains", |bencher| {
        let mut i = 0;
        bencher.iter(|| {
            i = (i + 1) % probes.len();
            std::hint::black_box(region.contains_point(&probes[i]))
        });
    });

    group.finish();
}

criterion_group!(benches, bench_sphere_union, bench_polyhedron);
