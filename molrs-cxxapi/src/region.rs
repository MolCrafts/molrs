//! Geometric regions for the C++ engine — the CXX face of
//! `molrs::core::Region` (through `molrs_ffi::RegionRef`).

/// Bridge handle over a shared region.
///
/// The region is `Arc<dyn Region>` inside; the trait object never crosses the
/// bridge. C++ holds a `Box<RegionRef>` — the same `molrs_ffi::RegionRef` the
/// Python capsule and the C API carry, so a region built on any of those
/// surfaces answers identically here.
pub struct RegionRef(pub molrs_ffi::RegionRef);

impl RegionRef {
    fn wrap(region: std::sync::Arc<dyn molrs::core::Region + Send + Sync>) -> Box<Self> {
        Box::new(RegionRef(molrs_ffi::RegionRef::new(region)))
    }
}

fn triple(v: &[f64]) -> [f64; 3] {
    [v[0], v[1], v[2]]
}

/// Ball of `radius` about `center` (3 values). Empty `center` yields the
/// origin, so a malformed call cannot silently read past the slice.
pub(crate) fn region_sphere(center: &[f64], radius: f64) -> Box<RegionRef> {
    let c = if center.len() == 3 {
        triple(center)
    } else {
        [0.0; 3]
    };
    RegionRef::wrap(std::sync::Arc::new(molrs::core::Sphere::new(
        molrs::op::types::F3::from_vec(c.to_vec()),
        radius,
    )))
}

/// Axis-aligned box with a corner at `origin` and edge `lengths`.
pub(crate) fn region_cuboid(origin: &[f64], lengths: &[f64]) -> Box<RegionRef> {
    let o = if origin.len() == 3 {
        triple(origin)
    } else {
        [0.0; 3]
    };
    let l = if lengths.len() == 3 {
        triple(lengths)
    } else {
        [0.0; 3]
    };
    RegionRef::wrap(std::sync::Arc::new(molrs::core::Cuboid::new(
        molrs::op::types::F3::from_vec(o.to_vec()),
        molrs::op::types::F3::from_vec(l.to_vec()),
    )))
}

/// Everything on the `normal` side of the plane through `point`.
/// A degenerate normal yields an error, reported as an empty handle upstream.
pub(crate) fn region_half_space(normal: &[f64], point: &[f64]) -> Result<Box<RegionRef>, String> {
    if normal.len() != 3 || point.len() != 3 {
        return Err("region_half_space: normal and point must each have 3 elements".into());
    }
    molrs::core::HalfSpace::new(triple(normal), triple(point))
        .map(|r| RegionRef::wrap(std::sync::Arc::new(r)))
        .map_err(|e| e.to_string())
}

/// Finite cylinder of `radius` and `length` from `base` along `axis`.
pub(crate) fn region_cylinder(
    base: &[f64],
    axis: &[f64],
    radius: f64,
    length: f64,
) -> Result<Box<RegionRef>, String> {
    if base.len() != 3 || axis.len() != 3 {
        return Err("region_cylinder: base and axis must each have 3 elements".into());
    }
    molrs::core::Cylinder::new(triple(base), triple(axis), radius, length)
        .map(|r| RegionRef::wrap(std::sync::Arc::new(r)))
        .map_err(|e| e.to_string())
}

/// Ellipsoid about `center` with the given `semi_axes`.
pub(crate) fn region_ellipsoid(
    center: &[f64],
    semi_axes: &[f64],
) -> Result<Box<RegionRef>, String> {
    if center.len() != 3 || semi_axes.len() != 3 {
        return Err("region_ellipsoid: centre and semi-axes must each have 3 elements".into());
    }
    molrs::core::Ellipsoid::new(triple(center), triple(semi_axes))
        .map(|r| RegionRef::wrap(std::sync::Arc::new(r)))
        .map_err(|e| e.to_string())
}

/// Intersection of `a` and `b`. Both stay usable.
pub(crate) fn region_and(a: &RegionRef, b: &RegionRef) -> Box<RegionRef> {
    RegionRef::wrap(std::sync::Arc::new(molrs::core::AndRegion::new(
        a.0.region(),
        b.0.region(),
    )))
}

/// Union of `a` and `b`.
pub(crate) fn region_or(a: &RegionRef, b: &RegionRef) -> Box<RegionRef> {
    RegionRef::wrap(std::sync::Arc::new(molrs::core::OrRegion::new(
        a.0.region(),
        b.0.region(),
    )))
}

/// Everything `a` is not. A shell is `and(outer, not(inner))`.
pub(crate) fn region_not(a: &RegionRef) -> Box<RegionRef> {
    RegionRef::wrap(std::sync::Arc::new(molrs::core::NotRegion::new(
        a.0.region(),
    )))
}

/// Signed distance of each point to the boundary: negative inside, positive
/// outside. `points` is flat `[x, y, z, …]`; a ragged length yields empty.
pub(crate) fn region_distance(rref: &RegionRef, points: &[f64]) -> Vec<f64> {
    if !points.len().is_multiple_of(3) {
        return Vec::new();
    }
    rref.0.with_region(|r| {
        points
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| r.distance(p))
            .collect()
    })
}

/// `1` for each point inside the solid, `0` outside. cxx has no `Vec<bool>`.
pub(crate) fn region_contains(rref: &RegionRef, points: &[f64]) -> Vec<u8> {
    if !points.len().is_multiple_of(3) {
        return Vec::new();
    }
    rref.0.with_region(|r| {
        points
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| u8::from(r.contains_point(p)))
            .collect()
    })
}

/// Axis-aligned bounds as `[xmin, xmax, ymin, ymax, zmin, zmax]`.
pub(crate) fn region_bounds(rref: &RegionRef) -> Vec<f64> {
    rref.0.with_region(|r| r.bounds().iter().copied().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shell_is_and_of_outer_and_not_inner() {
        let outer = region_sphere(&[0.0, 0.0, 0.0], 3.0);
        let inner = region_sphere(&[0.0, 0.0, 0.0], 2.0);
        let shell = region_and(&outer, &region_not(&inner));
        // 2.5 is in the shell, 1.0 is in the hole, 4.0 is outside both.
        let pts = [2.5, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        assert_eq!(region_contains(&shell, &pts), vec![1, 0, 0]);
        let d = region_distance(&outer, &pts);
        assert!((d[0] + 0.5).abs() < 1e-12, "{d:?}");
        assert_eq!(region_bounds(&outer).len(), 6);
    }

    #[test]
    fn a_ragged_point_slice_yields_nothing_rather_than_reading_past_it() {
        let ball = region_sphere(&[0.0, 0.0, 0.0], 1.0);
        assert!(region_distance(&ball, &[0.0, 0.0]).is_empty());
        assert!(region_contains(&ball, &[0.0, 0.0]).is_empty());
    }
}
