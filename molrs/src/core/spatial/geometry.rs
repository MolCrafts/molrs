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
            [nx[row], ny[row], nz[row]] =
                rotate_point([x[row], y[row], z[row]], k, cos_a, sin_a, origin);
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

/// Rotate one point about the unit axis `k` through `origin` (Rodrigues'
/// formula), given the angle's cosine and sine.
pub(crate) fn rotate_point(
    point: [f64; 3],
    k: [f64; 3],
    cos_a: f64,
    sin_a: f64,
    origin: [f64; 3],
) -> [f64; 3] {
    let p = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    let kdotp = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
    let cross = [
        k[1] * p[2] - k[2] * p[1],
        k[2] * p[0] - k[0] * p[2],
        k[0] * p[1] - k[1] * p[0],
    ];
    [
        p[0] * cos_a + cross[0] * sin_a + k[0] * kdotp * (1.0 - cos_a) + origin[0],
        p[1] * cos_a + cross[1] * sin_a + k[1] * kdotp * (1.0 - cos_a) + origin[1],
        p[2] * cos_a + cross[2] * sin_a + k[2] * kdotp * (1.0 - cos_a) + origin[2],
    ]
}

/// Shortest length (Å) a vector must have to count as a direction.
///
/// Well below any bond length, well above the rounding noise of a coordinate
/// difference.
const MIN_DIRECTION_LENGTH: f64 = 1e-6;

/// `vector` scaled to unit length, or `None` when it is not a direction: a
/// component is non-finite, its squared length overflows, or it is shorter
/// than [`MIN_DIRECTION_LENGTH`].
pub(crate) fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let norm_sq = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    // NaN fails both tests, so a NaN component is rejected too.
    if !(norm_sq.is_finite() && norm_sq > MIN_DIRECTION_LENGTH * MIN_DIRECTION_LENGTH) {
        return None;
    }
    let norm = norm_sq.sqrt();
    Some([vector[0] / norm, vector[1] / norm, vector[2] / norm])
}

/// A unit vector perpendicular to `axis`, crossed with the basis vector least
/// aligned with it. `None` when `axis` is not a direction ([`normalize`]).
pub(crate) fn perpendicular(axis: [f64; 3]) -> Option<[f64; 3]> {
    let axis = normalize(axis)?;
    let basis = if axis[0].abs() <= axis[1].abs() && axis[0].abs() <= axis[2].abs() {
        [1.0, 0.0, 0.0]
    } else if axis[1].abs() <= axis[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    normalize([
        axis[1] * basis[2] - axis[2] * basis[1],
        axis[2] * basis[0] - axis[0] * basis[2],
        axis[0] * basis[1] - axis[1] * basis[0],
    ])
}

/// The rotation, as a unit axis and an angle in radians, that turns `from_dir`
/// onto `to_dir`. `None` when the two already point the same way or either one
/// is not a direction ([`normalize`]).
pub(crate) fn alignment(from_dir: [f64; 3], to_dir: [f64; 3]) -> Option<([f64; 3], f64)> {
    let (a, b) = (normalize(from_dir)?, normalize(to_dir)?);
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let cross_norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
    // Near-antiparallel: `cross` is too short to carry a reliable axis, so turn
    // half a revolution about any perpendicular instead.
    if dot < 0.0 && cross_norm < 1e-8 {
        return Some((perpendicular(a)?, std::f64::consts::PI));
    }
    (cross_norm > 1e-15).then(|| {
        (
            [
                cross[0] / cross_norm,
                cross[1] / cross_norm,
                cross[2] / cross_norm,
            ],
            cross_norm.atan2(dot),
        )
    })
}

/// Rotate about `anchor` so `from_dir` points along `to_dir`.
///
/// The anchor stays fixed. This is only the facing half of a rigid motion;
/// moving the anchor onto a trace point is [`crate::builder::TracePlacer`].
pub fn orient(
    mol: &mut MolGraph,
    anchor: [f64; 3],
    from_dir: [f64; 3],
    mut to_dir: [f64; 3],
    flip: bool,
) {
    if flip {
        to_dir = [-to_dir[0], -to_dir[1], -to_dir[2]];
    }
    if let Some((axis, angle)) = alignment(from_dir, to_dir) {
        rotate(mol, axis, angle, Some(anchor));
    }
}
