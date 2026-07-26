//! Geometry *systems* — free functions that transform a [`MolGraph`]'s node
//! coordinates in place.
//!
//! Under the ECS model the graph is pure data; spatial transforms are systems
//! that operate over the world, so they live here as free functions rather than
//! as methods on the data structure. Coordinates are read and written through
//! the canonical [`crate::store::keys`] coordinate convention — no field-name literals.

use crate::store::keys;
use crate::system::molgraph::MolGraph;

/// Translate every node that has coordinates by `delta` (nodes without a full
/// coordinate set are left untouched).
///
/// Operates directly on each dense coordinate column rather than per-node handle
/// lookups, so cost is linear in the number of atoms with one pass per axis.
pub fn translate(mol: &mut MolGraph, delta: [f64; 3]) {
    let table = mol.node_table_mut();
    for (i, key) in keys::COORDS.iter().enumerate() {
        if let Ok((data, valid)) = table.column_f64_mut(key) {
            for (row, val) in data.iter_mut().enumerate() {
                if valid.get(row) {
                    *val += delta[i];
                }
            }
        }
    }
}

/// Scale every node that has coordinates by a per-axis `factor` about an
/// optional center (defaults to the origin). Pass `[s, s, s]` for a uniform
/// scale. Nodes missing any coordinate are left untouched.
///
/// Operates directly on each dense coordinate column (one pass per axis), so
/// cost is linear in the number of atoms.
pub fn scale(mol: &mut MolGraph, factor: [f64; 3], about: Option<[f64; 3]>) {
    let origin = about.unwrap_or([0.0, 0.0, 0.0]);
    let table = mol.node_table_mut();
    for (i, key) in keys::COORDS.iter().enumerate() {
        if let Ok((data, valid)) = table.column_f64_mut(key) {
            for (row, val) in data.iter_mut().enumerate() {
                if valid.get(row) {
                    *val = (*val - origin[i]) * factor[i] + origin[i];
                }
            }
        }
    }
}

/// Rotate every node that has coordinates around `axis` by `angle` radians,
/// optionally about a center point (defaults to the origin). Nodes missing any
/// coordinate are left untouched.
pub fn rotate(mol: &mut MolGraph, axis: [f64; 3], angle: f64, about: Option<[f64; 3]>) {
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if len < 1e-15 {
        return;
    }
    let k = [axis[0] / len, axis[1] / len, axis[2] / len];
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let origin = about.unwrap_or([0.0, 0.0, 0.0]);

    let table = mol.node_table_mut();

    // Read the three dense coordinate columns once, rotate only rows whose
    // x/y/z are all present, and write each column back in a single pass.
    // Rows missing any coordinate keep their original value (identity write).
    let (nx, ny, nz) = {
        let (x, vx) = match table.column_f64(keys::X) {
            Ok(t) => t,
            Err(_) => return,
        };
        let (y, vy) = match table.column_f64(keys::Y) {
            Ok(t) => t,
            Err(_) => return,
        };
        let (z, vz) = match table.column_f64(keys::Z) {
            Ok(t) => t,
            Err(_) => return,
        };
        let mut nx = x.to_vec();
        let mut ny = y.to_vec();
        let mut nz = z.to_vec();
        for row in 0..x.len() {
            if !(vx.get(row) && vy.get(row) && vz.get(row)) {
                continue;
            }
            let p = [x[row] - origin[0], y[row] - origin[1], z[row] - origin[2]];

            // Rodrigues' rotation formula.
            let kdotp = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
            let cross = [
                k[1] * p[2] - k[2] * p[1],
                k[2] * p[0] - k[0] * p[2],
                k[0] * p[1] - k[1] * p[0],
            ];
            nx[row] = p[0] * cos_a + cross[0] * sin_a + k[0] * kdotp * (1.0 - cos_a) + origin[0];
            ny[row] = p[1] * cos_a + cross[1] * sin_a + k[1] * kdotp * (1.0 - cos_a) + origin[1];
            nz[row] = p[2] * cos_a + cross[2] * sin_a + k[2] * kdotp * (1.0 - cos_a) + origin[2];
        }
        (nx, ny, nz)
    };

    // Safe: columns exist (checked above) and lengths are unchanged.
    table
        .column_f64_mut(keys::X)
        .unwrap()
        .0
        .copy_from_slice(&nx);
    table
        .column_f64_mut(keys::Y)
        .unwrap()
        .0
        .copy_from_slice(&ny);
    table
        .column_f64_mut(keys::Z)
        .unwrap()
        .0
        .copy_from_slice(&nz);
}

/// Align one direction with another about an anchor, then translate the anchor.
///
/// If both directions are supplied, the graph is rigidly rotated so `from_dir`
/// points along `to_dir` (or its negative when `flip` is true). The anchor
/// `from` remains fixed during rotation. Finally the whole graph is translated
/// so `from` lands on `to`.
pub fn align_direction(
    mol: &mut MolGraph,
    from: [f64; 3],
    to: [f64; 3],
    from_dir: Option<[f64; 3]>,
    to_dir: Option<[f64; 3]>,
    flip: bool,
) {
    if let (Some(from_dir), Some(mut to_dir)) = (from_dir, to_dir) {
        if flip {
            to_dir = [-to_dir[0], -to_dir[1], -to_dir[2]];
        }
        let normalize = |v: [f64; 3]| {
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            (norm > 1e-15).then(|| [v[0] / norm, v[1] / norm, v[2] / norm])
        };
        if let (Some(a), Some(b)) = (normalize(from_dir), normalize(to_dir)) {
            let cross = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            let cross_norm =
                (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
            if cross_norm > 1e-15 {
                rotate(mol, cross, cross_norm.atan2(dot), Some(from));
            } else if dot < 0.0 {
                let basis = if a[0].abs() <= a[1].abs() && a[0].abs() <= a[2].abs() {
                    [1.0, 0.0, 0.0]
                } else if a[1].abs() <= a[2].abs() {
                    [0.0, 1.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0]
                };
                let axis = [
                    a[1] * basis[2] - a[2] * basis[1],
                    a[2] * basis[0] - a[0] * basis[2],
                    a[0] * basis[1] - a[1] * basis[0],
                ];
                rotate(mol, axis, std::f64::consts::PI, Some(from));
            }
        }
    }
    translate(mol, [to[0] - from[0], to[1] - from[1], to[2] - from[2]]);
}
