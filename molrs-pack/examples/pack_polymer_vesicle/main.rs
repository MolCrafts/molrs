//! Pack multiple polyethylene chains (with explicit H) into a spherical vesicle.
//!
//! Each PE chain is generated as a 3D random-walk backbone, then hydrogenated
//! with `add_hydrogens`. During packing, MC torsion rotations fold the chains
//! to compact them into the inner sphere.
//!
//! Structure:
//! - 5 PE chains (20 carbons each, with explicit H) inside sphere r=10
//! - 90 lipids forming the inner leaflet
//! - 300 lipids forming the outer leaflet
//! - 2000 water molecules in outer shell
//!
//! Run with:
//! ```sh
//! cargo run -p molrs-pack --example pack_polymer_vesicle --release
//! ```

use std::path::PathBuf;

use molrs::element::Element;
use molrs::hydrogens::add_hydrogens;
use molrs::io::pdb::read_pdb_frame;
use molrs::molgraph::{Atom, AtomId, MolGraph};
use molrs::types::F;
use molrs_pack::{
    InsideBoxConstraint, InsideSphereConstraint, Molpack, OutsideSphereConstraint, ProgressHandler,
    RegionConstraint, Target, TorsionMcHook, XYZHandler,
};
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

/// Generate a polyethylene chain with explicit hydrogen atoms.
///
/// Returns `(graph, coords, radii, elements)` where:
/// - `graph`: MolGraph with C backbone + H atoms and all bonds
/// - `coords`: 3D positions for all atoms (C first, then H)
/// - `radii`: VdW radii per atom
/// - `elements`: element symbol strings per atom
fn polyethylene_chain(
    n_carbons: usize,
    cc_bond: F,
    rng: &mut impl Rng,
) -> (MolGraph, Vec<[F; 3]>, Vec<F>, Vec<String>) {
    let ch_bond: F = 1.09;

    // ── 1. Build C-only backbone topology ────────────────────────────────
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

    // ── 2. Random-walk backbone coordinates ──────────────────────────────
    let mut c_coords: Vec<[F; 3]> = Vec::with_capacity(n_carbons);
    c_coords.push([0.0, 0.0, 0.0]);
    for _ in 1..n_carbons {
        let dir = random_direction(rng);
        let prev = c_coords.last().unwrap();
        c_coords.push([
            prev[0] + dir[0] * cc_bond,
            prev[1] + dir[1] * cc_bond,
            prev[2] + dir[2] * cc_bond,
        ]);
    }

    // ── 3. Add hydrogens (topology only) ─────────────────────────────────
    let full_graph = add_hydrogens(&backbone);
    let n_total = full_graph.n_atoms();

    // Collect atom IDs in iteration order (C atoms first, then H atoms).
    let all_ids: Vec<AtomId> = full_graph.atoms().map(|(id, _)| id).collect();

    // Build a map from AtomId → index for coordinate lookup.
    let id_to_idx: std::collections::HashMap<AtomId, usize> =
        all_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // ── 4. Generate H coordinates ────────────────────────────────────────
    // C atoms keep their backbone coords; H atoms are placed at tetrahedral
    // positions around their parent C.
    let mut coords: Vec<[F; 3]> = vec![[0.0; 3]; n_total];
    let mut radii: Vec<F> = vec![0.0; n_total];
    let mut elements: Vec<String> = vec![String::new(); n_total];

    // First pass: place C atoms and record their element info.
    for (c_idx, &c_id) in c_ids.iter().enumerate() {
        let global_idx = id_to_idx[&c_id];
        coords[global_idx] = c_coords[c_idx];
        radii[global_idx] = Element::by_symbol("C").unwrap().vdw_radius() as F;
        elements[global_idx] = "C".to_string();
    }

    // Second pass: place H atoms around their parent C.
    for &atom_id in &all_ids {
        let atom = full_graph.get_atom(atom_id).expect("atom exists");
        let sym = atom.get_str("symbol").unwrap_or("X");
        if sym != "H" {
            continue;
        }

        let global_idx = id_to_idx[&atom_id];
        elements[global_idx] = "H".to_string();
        radii[global_idx] = Element::by_symbol("H").unwrap().vdw_radius() as F;

        // Find parent C (the only neighbor of this H).
        let parent_id = full_graph.neighbors(atom_id).next().unwrap();
        let parent_idx = id_to_idx[&parent_id];
        let parent_pos = coords[parent_idx];

        // Collect the C-neighbors of the parent (excluding this H and other H atoms).
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

        // Count how many H atoms on this parent have already been placed
        // (have a lower index). This determines the rotation angle offset.
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

        let h_pos = place_hydrogen(&parent_pos, &c_neighbors, h_index, ch_bond, rng);
        coords[global_idx] = h_pos;
    }

    (full_graph, coords, radii, elements)
}

/// Place a hydrogen atom around `parent` given its heavy-atom neighbors.
///
/// - Internal C (2 C-neighbors): bisector-based tetrahedral placement.
/// - Terminal C (1 C-neighbor): 3 H at ~109.5° intervals around anti-bond axis.
/// - Isolated C (0 C-neighbors): random direction.
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
            // Isolated atom — place H along a random direction.
            let dir = random_direction(rng);
            [
                parent[0] + dir[0] * bond_len,
                parent[1] + dir[1] * bond_len,
                parent[2] + dir[2] * bond_len,
            ]
        }
        1 => {
            // Terminal C: axis from neighbor → parent (anti-bond direction).
            let axis = normalize(sub(parent, &c_neighbors[0]));
            let perp = arbitrary_perpendicular(&axis);
            let angle_offset = (h_index as F) * 2.0 * std::f32::consts::PI as F / 3.0;
            let rotated = rotate_around_axis(&perp, &axis, angle_offset);
            // Tilt away from axis by tetrahedral half-angle.
            let dir = add_scaled(&axis, half_angle.cos(), &rotated, half_angle.sin());
            let d = normalize(dir);
            [
                parent[0] + d[0] * bond_len,
                parent[1] + d[1] * bond_len,
                parent[2] + d[2] * bond_len,
            ]
        }
        _ => {
            // Internal C (2 neighbors): anti-bisector direction.
            let v1 = normalize(sub(&c_neighbors[0], parent));
            let v2 = normalize(sub(&c_neighbors[1], parent));
            let bisector = normalize(add_scaled(&v1, 1.0, &v2, 1.0));
            let anti = [-bisector[0], -bisector[1], -bisector[2]];
            // Normal to the C-C-C plane.
            let normal = normalize(cross(&v1, &v2));
            // Two H: one above, one below the plane.
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

/// Find an arbitrary vector perpendicular to `v`.
fn arbitrary_perpendicular(v: &[F; 3]) -> [F; 3] {
    let candidate = if v[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize(cross(v, &candidate))
}

/// Rotate vector `v` around `axis` (unit) by `angle` radians (Rodrigues).
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reuse PDB structures from pack_spherical
    let spherical_dir = PathBuf::from(file!())
        .parent()
        .unwrap()
        .join("../pack_spherical");
    let water = read_pdb_frame(spherical_dir.join("water.pdb"))?;
    let lipid = read_pdb_frame(spherical_dir.join("palmitoil.pdb"))?;

    let origin = [0.0, 0.0, 0.0];
    let mut rng = SmallRng::seed_from_u64(42);

    // ── Polyethylene chains ──────────────────────────────────────────────
    let n_chains = 5;
    let n_carbons = 20;
    let mut targets = Vec::new();

    for i in 0..n_chains {
        let (graph, coords, radii, elements) = polyethylene_chain(n_carbons, 1.54, &mut rng);

        let hook = TorsionMcHook::new(&graph)
            .with_self_avoidance(0.75)
            .with_temperature(0.5)
            .with_steps(20)
            .with_max_delta(std::f64::consts::PI as F / 4.0);

        let mut target = Target::from_coords(&coords, &radii, 1)
            .with_name(format!("PE_{i}"))
            .with_constraint(InsideSphereConstraint::new(10.0, origin))
            .with_hook(hook);
        target.elements = elements.iter().map(|s| s.into()).collect();
        targets.push(target);
    }

    // ── Vesicle (from pack_spherical, scaled down) ──────────────────────

    // Inner lipid leaflet
    let lipid_inner = Target::new(lipid.clone(), 90)
        .with_constraint_for_atoms(&[37], InsideSphereConstraint::new(14.0, origin))
        .with_constraint_for_atoms(&[5], OutsideSphereConstraint::new(26.0, origin))
        .with_name("lipid_inner");

    // Outer lipid leaflet
    let lipid_outer = Target::new(lipid, 300)
        .with_constraint_for_atoms(&[5], InsideSphereConstraint::new(29.0, origin))
        .with_constraint_for_atoms(&[37], OutsideSphereConstraint::new(41.0, origin))
        .with_name("lipid_outer");

    // Outer water shell (reduced count for faster demo)
    let water_outer = Target::new(water, 2000)
        .with_constraint(
            InsideBoxConstraint::new([-47.5, -47.5, -47.5], [47.5, 47.5, 47.5])
                .and(OutsideSphereConstraint::new(43.0, origin)),
        )
        .with_name("water_outer");

    targets.push(lipid_inner);
    targets.push(lipid_outer);
    targets.push(water_outer);

    // ── Pack ────────────────────────────────────────────────────────────
    let out_dir = PathBuf::from(file!()).parent().unwrap().to_path_buf();

    let mut packer = Molpack::new();
    if std::env::var_os("MOLRS_PACK_EXAMPLE_PROGRESS").is_some() {
        packer = packer.add_handler(ProgressHandler::new());
    }
    if std::env::var_os("MOLRS_PACK_EXAMPLE_XYZ").is_some() {
        packer = packer.add_handler(XYZHandler::new(out_dir.join("polymer_vesicle.xyz"), 10));
    }

    let result = packer.pack(&targets, 500, Some(42))?;

    eprintln!(
        "Done — {} atoms, converged={}, fdist={:.2e}, frest={:.2e}",
        result.natoms(),
        result.converged,
        result.fdist,
        result.frest,
    );

    Ok(())
}
