//! Smoke test / demo for the self-avoiding random-walk path generator.
//!
//! Run with:
//! ```text
//! cargo run -p molcrafts-molrs --example saw_demo
//! ```
//!
//! Generates a few melts (FCC and off-lattice, periodic and reflective) and
//! prints per-run statistics: chain shape, bond-length spread, minimum pairwise
//! separation, box volume, and whether every point landed inside the box.

use molrs::builder::{FccLattice, OffLattice, SelfAvoidingWalk, WalkOutput};
use molrs::spatial::simbox::SimBox;
use molrs::types::{F, F3};

fn pt(v: &F3) -> [F; 3] {
    [v[0], v[1], v[2]]
}

fn min_image_dist(sb: &SimBox, a: &F3, b: &F3) -> F {
    let d = sb.shortest_vector_impl(pt(a), pt(b));
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn report(label: &str, bond_length: F, out: &WalkOutput) {
    let edge = out.simbox.lengths()[0];
    let n_chains = out.paths.len();
    let chain_len = out.paths.first().map_or(0, |c| c.len());

    // bond-length spread (consecutive, min-image)
    let mut bmin = F::INFINITY;
    let mut bmax = 0.0_f64;
    for chain in &out.paths {
        for w in chain.windows(2) {
            let d = min_image_dist(&out.simbox, &w[0], &w[1]);
            bmin = bmin.min(d);
            bmax = bmax.max(d);
        }
    }

    // minimum pairwise separation across all monomers (min-image)
    let all: Vec<&F3> = out.paths.iter().flatten().collect();
    let mut sep = F::INFINITY;
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            sep = sep.min(min_image_dist(&out.simbox, all[i], all[j]));
        }
    }

    // containment
    let inside = out
        .paths
        .iter()
        .flatten()
        .all(|p| (0..3).all(|k| p[k] >= 0.0 && p[k] < edge));

    println!("── {label}");
    println!(
        "   chains × length : {n_chains} × {chain_len}  ({} monomers)",
        all.len()
    );
    println!(
        "   box edge / vol  : {edge:.3} / {:.1}",
        out.simbox.volume()
    );
    println!("   bond length     : [{bmin:.6}, {bmax:.6}]  (target {bond_length})");
    println!("   min separation  : {sep:.4}");
    println!("   all in-box      : {inside}");
}

fn main() {
    let b = 1.53;

    let fcc = SelfAvoidingWalk {
        n_chains: 5,
        chain_length: 40,
        bond_length: b,
        target_density: 0.05,
        pbc: [true, true, true],
        seed: 9062,
        strategy: FccLattice,
    };
    report("FCC, periodic", b, &fcc.generate().expect("fcc periodic"));

    let fcc_reflect = SelfAvoidingWalk {
        n_chains: 5,
        chain_length: 40,
        bond_length: b,
        target_density: 0.05,
        pbc: [false, false, false],
        seed: 9062,
        strategy: FccLattice,
    };
    report(
        "FCC, reflective walls",
        b,
        &fcc_reflect.generate().expect("fcc reflective"),
    );

    let off = SelfAvoidingWalk {
        n_chains: 5,
        chain_length: 40,
        bond_length: b,
        target_density: 0.05,
        pbc: [true, true, true],
        seed: 9062,
        strategy: OffLattice {
            excluded_radius: 1.0,
        },
    };
    report(
        "OffLattice, periodic",
        b,
        &off.generate().expect("off periodic"),
    );

    let off_reflect = SelfAvoidingWalk {
        pbc: [false, false, false],
        strategy: OffLattice {
            excluded_radius: 1.0,
        },
        ..SelfAvoidingWalk {
            n_chains: 5,
            chain_length: 40,
            bond_length: b,
            target_density: 0.05,
            pbc: [true, true, true],
            seed: 9062,
            strategy: OffLattice {
                excluded_radius: 1.0,
            },
        }
    };
    report(
        "OffLattice, reflective walls",
        b,
        &off_reflect.generate().expect("off reflective"),
    );

    // determinism check
    let a1 = off.generate().unwrap();
    let a2 = off.generate().unwrap();
    let identical = a1
        .paths
        .iter()
        .flatten()
        .zip(a2.paths.iter().flatten())
        .all(|(p, q)| pt(p) == pt(q));
    println!("── determinism (same seed) : {identical}");
}
