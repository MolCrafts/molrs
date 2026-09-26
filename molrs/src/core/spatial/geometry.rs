//! Geometry *systems* — free functions that transform a [`MolGraph`]'s node
//! coordinates in place.
//!
//! Under the ECS model the graph is pure data; spatial transforms are systems
//! that operate over the world, so they live here as free functions rather than
//! as methods on the data structure. Coordinates are read and written through
//! the canonical [`crate::store::keys`] coordinate convention — no field-name literals.

use crate::error::MolRsError;
use crate::op::rigid::{self, apply, axis_angle};
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
///
/// Only the direction of `axis` matters; its length does not.
///
/// # Errors
///
/// [`MolRsError::Validation`] when `axis` has no direction (every component
/// zero, a component non-finite, or a squared length that overflows) or
/// `angle` is not finite. Nothing is written then.
pub fn rotate(
    mol: &mut MolGraph,
    axis: [f64; 3],
    angle: f64,
    about: Option<[f64; 3]>,
) -> Result<(), MolRsError> {
    // `axis_angle` is the one rule for "has a direction": any finite nonzero
    // axis, whatever its length.
    let rotation = axis_angle(axis, angle).ok_or_else(|| MolRsError::Validation {
        message: format!("rotation axis {axis:?} has no direction"),
    })?;
    if !angle.is_finite() {
        return Err(MolRsError::Validation {
            message: format!("rotation angle {angle} is not finite"),
        });
    }
    let motion = rigid::about(rotation, about.unwrap_or([0.0, 0.0, 0.0]));

    let table = mol.node_table_mut();

    // Read the three dense coordinate columns once, rotate only rows whose
    // x/y/z are all present, and write each column back in a single pass.
    // Rows missing any coordinate keep their original value (identity write).
    let (nx, ny, nz) = {
        // A graph without a coordinate column has nothing to rotate.
        let (x, vx) = match table.column_f64(keys::X) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };
        let (y, vy) = match table.column_f64(keys::Y) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };
        let (z, vz) = match table.column_f64(keys::Z) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };
        let mut nx = x.to_vec();
        let mut ny = y.to_vec();
        let mut nz = z.to_vec();
        for row in 0..x.len() {
            if !(vx.get(row) && vy.get(row) && vz.get(row)) {
                continue;
            }
            [nx[row], ny[row], nz[row]] = apply(&motion, [x[row], y[row], z[row]]);
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::atomistic::Atomistic;

    #[test]
    fn rotating_about_an_axis_with_no_direction_is_an_error() {
        for axis in [
            [0.0, 0.0, 0.0],
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0],
        ] {
            let mut mol = Atomistic::new();
            let atom = mol.add_atom_xyz("C", 1.0, 2.0, 3.0);
            let result = rotate(mol.as_molgraph_mut(), axis, 1.0, None);
            assert!(result.is_err(), "axis {axis:?} was accepted");
            let node = mol.as_molgraph().get_node(atom).unwrap();
            let position = [keys::X, keys::Y, keys::Z].map(|key| node.get_f64(key).unwrap());
            assert_eq!(position, [1.0, 2.0, 3.0], "axis {axis:?} moved the atom");
        }
    }
}
