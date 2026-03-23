//! Fold a polyethylene chain (with explicit H) into a sphere using pure MC
//! torsion rotation.
//!
//! No packer, no constraints — just a Metropolis MC loop with a spherical
//! penalty as the objective. Writes an XYZ trajectory so you can watch the
//! chain collapse.
//!
//! ```sh
//! cargo run -p molrs-pack --example mc_fold_chain --release
//! ```

use std::io::Write;
use std::path::PathBuf;

use std::collections::HashSet;

use molrs::hydrogens::add_hydrogens;
use molrs::molgraph::{Atom, AtomId, MolGraph};
use molrs::types::F;
use molrs_pack::{Hook, TorsionMcHook, compute_excluded_pairs, self_avoidance_penalty};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Random unit vector via Marsaglia method.
fn random_direction(rng: &mut impl Rng) -> [F; 3] {
    loop {
        let u: F = rng.random::<F>() * 2.0 - 1.0;
        let v: F = rng.random::<F>() * 2.0 - 1.0;
        let s = u * u + v * v;
        if s < 1.0 {
            let factor = 2.0 * (1.0 - s).sqrt();
            return [u * factor, v * factor, 1.0 - 2.0 * s];
        }
    }
}

fn place_hydrogen(
    parent: &[F; 3],
    c_neighbors: &[[F; 3]],
    h_index: usize,
    bond_len: F,
    rng: &mut impl Rng,
) -> [F; 3] {
    let tetrahedral_angle: F = 109.47_f32.to_radians() as F;
    let half_angle = tetrahedral_angle / 2.0;

    match c_neighbors.len() {
        0 => {
            let dir = random_direction(rng);
            [
                parent[0] + dir[0] * bond_len,
                parent[1] + dir[1] * bond_len,
                parent[2] + dir[2] * bond_len,
            ]
        }
        1 => {
            let axis = normalize(sub(parent, &c_neighbors[0]));
            let perp = arbitrary_perpendicular(&axis);
            let angle_offset = (h_index as F) * 2.0 * std::f32::consts::PI as F / 3.0;
            let rotated = rotate_around_axis(&perp, &axis, angle_offset);
            let dir = add_scaled(&axis, half_angle.cos(), &rotated, half_angle.sin());
            let d = normalize(dir);
            [
                parent[0] + d[0] * bond_len,
                parent[1] + d[1] * bond_len,
                parent[2] + d[2] * bond_len,
            ]
        }
        _ => {
            let v1 = normalize(sub(&c_neighbors[0], parent));
            let v2 = normalize(sub(&c_neighbors[1], parent));
            let bisector = normalize(add_scaled(&v1, 1.0, &v2, 1.0));
            let anti = [-bisector[0], -bisector[1], -bisector[2]];
            let normal = normalize(cross(&v1, &v2));
            let tilt: F = (half_angle - 0.5 * angle_between(&v1, &v2)).max(0.1);
            let sign: F = if h_index == 0 { 1.0 } else { -1.0 };
            let dir = normalize(add_scaled(&anti, tilt.cos(), &normal, sign * tilt.sin()));
            [
                parent[0] + dir[0] * bond_len,
                parent[1] + dir[1] * bond_len,
                parent[2] + dir[2] * bond_len,
            ]
        }
    }
}

// ── Vector utilities ─────────────────────────────────────────────────────

fn sub(a: &[F; 3], b: &[F; 3]) -> [F; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &[F; 3], b: &[F; 3]) -> F {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[F; 3], b: &[F; 3]) -> [F; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [F; 3]) -> [F; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        return [1.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn add_scaled(a: &[F; 3], sa: F, b: &[F; 3], sb: F) -> [F; 3] {
    [
        a[0] * sa + b[0] * sb,
        a[1] * sa + b[1] * sb,
        a[2] * sa + b[2] * sb,
    ]
}

fn angle_between(a: &[F; 3], b: &[F; 3]) -> F {
    dot(a, b).clamp(-1.0, 1.0).acos()
}

fn arbitrary_perpendicular(v: &[F; 3]) -> [F; 3] {
    let candidate = if v[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize(cross(v, &candidate))
}

fn rotate_around_axis(v: &[F; 3], axis: &[F; 3], angle: F) -> [F; 3] {
    let c = angle.cos();
    let s = angle.sin();
    let d = dot(v, axis);
    let cr = cross(axis, v);
    [
        v[0] * c + cr[0] * s + axis[0] * d * (1.0 - c),
        v[1] * c + cr[1] * s + axis[1] * d * (1.0 - c),
        v[2] * c + cr[2] * s + axis[2] * d * (1.0 - c),
    ]
}

/// Minimum distance among non-excluded atom pairs.
fn min_nonbonded_distance(
    coords: &[[F; 3]],
    excluded: &HashSet<(usize, usize)>,
) -> (F, usize, usize) {
    let n = coords.len();
    let mut min_d: F = F::INFINITY;
    let mut min_i = 0;
    let mut min_j = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let key = (i.min(j), i.max(j));
            if excluded.contains(&key) {
                continue;
            }
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d < min_d {
                min_d = d;
                min_i = i;
                min_j = j;
            }
        }
    }
    (min_d, min_i, min_j)
}

/// Spherical penalty: sum of max(0, r_i - R)^2 for each atom.
fn sphere_penalty(coords: &[[F; 3]], radius: F) -> F {
    coords
        .iter()
        .map(|p| {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let excess = (r - radius).max(0.0);
            excess * excess
        })
        .sum()
}

/// Write one extended XYZ frame with element symbols.
fn write_xyz(w: &mut impl Write, coords: &[[F; 3]], elements: &[String], comment: &str) {
    let n = coords.len();
    let _ = writeln!(w, "{n}");
    let _ = writeln!(w, "{comment}");
    for (i, p) in coords.iter().enumerate() {
        let elem = elements.get(i).map(|s| s.as_str()).unwrap_or("X");
        let _ = writeln!(w, "{elem}  {:.6}  {:.6}  {:.6}", p[0], p[1], p[2]);
    }
}

/// Build a zigzag (all-trans) PE chain with explicit H.
/// This starts non-self-intersecting by construction.
fn zigzag_pe_chain(
    n_carbons: usize,
    cc_bond: F,
    rng: &mut impl Rng,
) -> (MolGraph, Vec<[F; 3]>, Vec<String>) {
    let ch_bond: F = 1.09;
    let theta = (109.5_f64 * std::f64::consts::PI / 180.0) as F;

    // Build C-only backbone.
    let mut backbone = MolGraph::new();
    let mut c_ids: Vec<AtomId> = Vec::with_capacity(n_carbons);
    for _ in 0..n_carbons {
        let mut a = Atom::new();
        a.set("symbol", "C");
        c_ids.push(backbone.add_atom(a));
    }
    for i in 0..n_carbons - 1 {
        backbone
            .add_bond(c_ids[i], c_ids[i + 1])
            .expect("add backbone bond");
    }

    // Zigzag backbone in the xz-plane (all-trans).
    // Half-angle decomposition: α = (π − θ)/2 gives the correct bond projections.
    let alpha = (std::f64::consts::PI as F - theta) / 2.0;
    let dx = cc_bond * alpha.cos();
    let dz = cc_bond * alpha.sin();
    let mut c_coords: Vec<[F; 3]> = Vec::with_capacity(n_carbons);
    c_coords.push([0.0, 0.0, 0.0]);
    for i in 1..n_carbons {
        let prev = c_coords[i - 1];
        let sign: F = if i % 2 == 0 { 1.0 } else { -1.0 };
        c_coords.push([prev[0] + dx, 0.0, prev[2] + sign * dz]);
    }

    // Add hydrogens (topology).
    let full_graph = add_hydrogens(&backbone);
    let n_total = full_graph.n_atoms();
    let all_ids: Vec<AtomId> = full_graph.atoms().map(|(id, _)| id).collect();
    let id_to_idx: std::collections::HashMap<AtomId, usize> =
        all_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let mut coords: Vec<[F; 3]> = vec![[0.0; 3]; n_total];
    let mut elements: Vec<String> = vec![String::new(); n_total];

    for (c_idx, &c_id) in c_ids.iter().enumerate() {
        let idx = id_to_idx[&c_id];
        coords[idx] = c_coords[c_idx];
        elements[idx] = "C".to_string();
    }

    for &atom_id in &all_ids {
        let sym = full_graph
            .get_atom(atom_id)
            .ok()
            .and_then(|a| a.get_str("symbol"))
            .unwrap_or("X");
        if sym != "H" {
            continue;
        }
        let global_idx = id_to_idx[&atom_id];
        elements[global_idx] = "H".to_string();
        let parent_id = full_graph.neighbors(atom_id).next().unwrap();
        let parent_pos = coords[id_to_idx[&parent_id]];
        let c_neighbors: Vec<[F; 3]> = full_graph
            .neighbors(parent_id)
            .filter(|&nid| {
                nid != atom_id
                    && full_graph
                        .get_atom(nid)
                        .ok()
                        .and_then(|a| a.get_str("symbol"))
                        .map(|s| s != "H")
                        .unwrap_or(false)
            })
            .map(|nid| coords[id_to_idx[&nid]])
            .collect();
        let h_index: usize = full_graph
            .neighbors(parent_id)
            .filter(|&nid| {
                nid != atom_id
                    && full_graph
                        .get_atom(nid)
                        .ok()
                        .and_then(|a| a.get_str("symbol"))
                        .map(|s| s == "H")
                        .unwrap_or(false)
                    && id_to_idx[&nid] < global_idx
            })
            .count();
        coords[global_idx] = place_hydrogen(&parent_pos, &c_neighbors, h_index, ch_bond, rng);
    }

    (full_graph, coords, elements)
}

fn main() {
    let n_carbons: usize = 40;
    let cc_bond: F = 1.54;
    let radius: F = 5.0;
    let max_iters: usize = 5000;

    let mut rng = SmallRng::seed_from_u64(42);

    // Use zigzag (non-self-intersecting) initial configuration
    let (graph, coords, elements) = zigzag_pe_chain(n_carbons, cc_bond, &mut rng);
    let n_atoms = coords.len();

    let excluded = compute_excluded_pairs(&graph);

    let sa_radius: F = 0.75;

    let f0 = sphere_penalty(&coords, radius);
    let sa0 = self_avoidance_penalty(&coords, sa_radius, &excluded);
    let (min_d0, mi, mj) = min_nonbonded_distance(&coords, &excluded);
    eprintln!(
        "Initial: sphere={f0:.4}  sa_penalty={sa0:.4}  min_nb_dist={min_d0:.4} ({mi}={},{mj}={})  ({n_atoms} atoms, target r={radius}, excluded={})",
        elements[mi],
        elements[mj],
        excluded.len()
    );

    let hook = TorsionMcHook::new(&graph)
        .with_self_avoidance(0.75)
        .with_temperature(0.1)
        .with_steps(1)
        .with_max_delta(std::f64::consts::PI as F / 8.0);

    let mut runner = hook.build(&coords);
    let mut current = coords;
    let mut f_current = f0;

    // Open trajectory file.
    let out_path = PathBuf::from(file!())
        .parent()
        .unwrap()
        .join("trajectory.xyz");
    let mut file = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&out_path)
            .expect("cannot open trajectory file"),
    );

    write_xyz(
        &mut file,
        &current,
        &elements,
        &format!("iter=0  f={f0:.4e}"),
    );

    for iter in 1..=max_iters {
        if f_current < 1e-6 {
            eprintln!("Converged at iter {iter}");
            break;
        }

        let result = runner.on_iter(
            &current,
            f_current,
            &mut |trial| sphere_penalty(trial, radius),
            &mut rng,
        );

        if let Some(new_coords) = result {
            f_current = sphere_penalty(&new_coords, radius);
            current = new_coords;
        }

        write_xyz(
            &mut file,
            &current,
            &elements,
            &format!("iter={iter}  f={f_current:.4e}"),
        );

        if iter % 500 == 0 {
            let max_r = current
                .iter()
                .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
                .fold(0.0 as F, F::max);
            let sa = self_avoidance_penalty(&current, sa_radius, &excluded);
            let (min_d, mi, mj) = min_nonbonded_distance(&current, &excluded);
            // Count pairs within cutoff
            let cutoff = 2.0 * sa_radius;
            let n_overlap: usize = (0..n_atoms)
                .flat_map(|i| (i + 1..n_atoms).map(move |j| (i, j)))
                .filter(|&(i, j)| {
                    !excluded.contains(&(i.min(j), i.max(j))) && {
                        let dx = current[i][0] - current[j][0];
                        let dy = current[i][1] - current[j][1];
                        let dz = current[i][2] - current[j][2];
                        (dx * dx + dy * dy + dz * dz).sqrt() < cutoff
                    }
                })
                .count();
            eprintln!(
                "  iter {iter:>4}  sphere={f_current:.4e}  sa={sa:.2e}  min_d={min_d:.3}({mi}={},{mj}={})  max_r={max_r:.2}  overlap_pairs={n_overlap}",
                elements[mi], elements[mj]
            );
        }
    }

    file.flush().unwrap();

    let max_r = current
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
        .fold(0.0 as F, F::max);
    let sa_final = self_avoidance_penalty(&current, sa_radius, &excluded);
    let (min_d_final, mi, mj) = min_nonbonded_distance(&current, &excluded);
    eprintln!(
        "Done — sphere={f_current:.4e}  sa={sa_final:.4e}  min_d={min_d_final:.3} ({mi}={},{mj}={})  max_r={max_r:.2}  trajectory: {}",
        elements[mi],
        elements[mj],
        out_path.display()
    );
}
