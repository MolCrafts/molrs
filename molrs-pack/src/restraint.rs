//! Restraint trait and concrete soft-penalty types for molecular packing.
//!
//! Each `*Restraint` struct is a **concrete, independent type** holding its own
//! geometric parameters — no `Builtin*` wrapper, no tagged-union `{kind, params[9]}`
//! blob, no builder pattern. User extensions `impl Restraint` identically and sit
//! beside the 14 Packmol-originals in type space.
//!
//! Numerical equivalence to the Fortran `comprest.f90` (value) and `gwalls.f90`
//! (gradient) is preserved branch-for-branch; see `docs/packmol_alignment.md`.
//!
//! **Gradient convention**: `Restraint::fg` accumulates INTO `g` with `+=`.
//! Do not overwrite; many restraints may contribute to the same atom.
//!
//! **Two-scale contract** (Packmol convention): linear penalties
//! (box / cube / plane, kinds 2/3/6/7/10/11) use `scale`; quadratic penalties
//! (sphere / ellipsoid / cylinder / gaussian, kinds 4/5/8/9/12/13/14/15) use
//! `scale2`. Each `impl Restraint` decides internally which to consume.
//!
//! Direction-3 rule (see spec §0 bullet 9): all molrs-pack extension points
//! follow `pub trait X` + N concrete pub structs that `impl X`; user-defined
//! structs `impl X` the same way. No `Builtin*` prefix, no wrapper, no builder.

use molrs::types::F;

// ============================================================================
// Trait
// ============================================================================

/// Soft-penalty restraint evaluated per atom during packing.
///
/// - `f` — value only (line-search interpolation)
/// - `fg` — fused value + gradient; gradient accumulates INTO `g` with `+=`
/// - `is_parallel_safe` — if `false`, scheduler serializes this restraint
///   (Python-backed restraints MUST return `false`)
/// - `name` — human-readable identifier (default: `std::any::type_name::<Self>()`)
///
/// `Debug` is required on concrete impls so `Target` / `Molpack` remain
/// printable for diagnostics. All built-in restraints derive it; user types
/// should do the same.
pub trait Restraint: Send + Sync + std::fmt::Debug {
    fn f(&self, x: &[F; 3], scale: F, scale2: F) -> F;
    fn fg(&self, x: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F;
    fn is_parallel_safe(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Blanket impl so `Box<dyn Restraint>` itself implements the trait.
impl Restraint for Box<dyn Restraint> {
    #[inline]
    fn f(&self, x: &[F; 3], scale: F, scale2: F) -> F {
        (**self).f(x, scale, scale2)
    }
    #[inline]
    fn fg(&self, x: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        (**self).fg(x, scale, scale2, g)
    }
    #[inline]
    fn is_parallel_safe(&self) -> bool {
        (**self).is_parallel_safe()
    }
    #[inline]
    fn name(&self) -> &'static str {
        (**self).name()
    }
}

// ============================================================================
// Kind 2: InsideCubeRestraint
// ============================================================================

/// Packmol kind 2 — quadratic penalty forcing atom inside axis-aligned cube.
#[derive(Debug, Clone, Copy)]
pub struct InsideCubeRestraint {
    pub origin: [F; 3],
    pub side: F,
}

impl InsideCubeRestraint {
    pub fn new(origin: [F; 3], side: F) -> Self {
        Self { origin, side }
    }
}

impl Restraint for InsideCubeRestraint {
    fn f(&self, pos: &[F; 3], scale: F, _scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let (xmin, ymin, zmin) = (self.origin[0], self.origin[1], self.origin[2]);
        let (xmax, ymax, zmax) = (xmin + self.side, ymin + self.side, zmin + self.side);
        let a1 = (x - xmin).min(0.0);
        let a2 = (y - ymin).min(0.0);
        let a3 = (z - zmin).min(0.0);
        let mut f = scale * (a1 * a1 + a2 * a2 + a3 * a3);
        let a1 = (x - xmax).max(0.0);
        let a2 = (y - ymax).max(0.0);
        let a3 = (z - zmax).max(0.0);
        f += scale * (a1 * a1 + a2 * a2 + a3 * a3);
        f
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let (xmin, ymin, zmin) = (self.origin[0], self.origin[1], self.origin[2]);
        let (xmax, ymax, zmax) = (xmin + self.side, ymin + self.side, zmin + self.side);
        let a1 = x - xmin;
        let a2 = y - ymin;
        let a3 = z - zmin;
        if a1 < 0.0 {
            g[0] += scale * 2.0 * a1;
        }
        if a2 < 0.0 {
            g[1] += scale * 2.0 * a2;
        }
        if a3 < 0.0 {
            g[2] += scale * 2.0 * a3;
        }
        let a1 = x - xmax;
        let a2 = y - ymax;
        let a3 = z - zmax;
        if a1 > 0.0 {
            g[0] += scale * 2.0 * a1;
        }
        if a2 > 0.0 {
            g[1] += scale * 2.0 * a2;
        }
        if a3 > 0.0 {
            g[2] += scale * 2.0 * a3;
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 3: InsideBoxRestraint
// ============================================================================

/// Packmol kind 3 — quadratic penalty forcing atom inside axis-aligned box.
#[derive(Debug, Clone, Copy)]
pub struct InsideBoxRestraint {
    pub min: [F; 3],
    pub max: [F; 3],
}

impl InsideBoxRestraint {
    pub fn new(min: [F; 3], max: [F; 3]) -> Self {
        Self { min, max }
    }

    /// Construct an axis-aligned cubic box from an origin and isotropic side length.
    pub fn cube_from_origin(origin: [F; 3], side: F) -> Self {
        Self {
            min: origin,
            max: [origin[0] + side, origin[1] + side, origin[2] + side],
        }
    }

    /// Construct an axis-aligned box from a molrs-core `SimBox`.
    ///
    /// The restraint bounds are `origin` → `origin + lengths`. Off-axis
    /// cells (triclinic tilts) are not representable as an axis-aligned
    /// restraint — write a custom `impl Restraint` for those.
    pub fn from_simbox(simbox: &molrs::region::SimBox) -> Self {
        let origin = simbox.origin_view();
        let lengths = simbox.lengths();
        let o = [origin[0], origin[1], origin[2]];
        Self {
            min: o,
            max: [o[0] + lengths[0], o[1] + lengths[1], o[2] + lengths[2]],
        }
    }
}

impl Restraint for InsideBoxRestraint {
    fn f(&self, pos: &[F; 3], scale: F, _scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let a1 = (x - self.min[0]).min(0.0);
        let a2 = (y - self.min[1]).min(0.0);
        let a3 = (z - self.min[2]).min(0.0);
        let mut f = scale * (a1 * a1 + a2 * a2 + a3 * a3);
        let a1 = (x - self.max[0]).max(0.0);
        let a2 = (y - self.max[1]).max(0.0);
        let a3 = (z - self.max[2]).max(0.0);
        f += scale * (a1 * a1 + a2 * a2 + a3 * a3);
        f
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let a1 = x - self.min[0];
        let a2 = y - self.min[1];
        let a3 = z - self.min[2];
        if a1 < 0.0 {
            g[0] += scale * 2.0 * a1;
        }
        if a2 < 0.0 {
            g[1] += scale * 2.0 * a2;
        }
        if a3 < 0.0 {
            g[2] += scale * 2.0 * a3;
        }
        let a1 = x - self.max[0];
        let a2 = y - self.max[1];
        let a3 = z - self.max[2];
        if a1 > 0.0 {
            g[0] += scale * 2.0 * a1;
        }
        if a2 > 0.0 {
            g[1] += scale * 2.0 * a2;
        }
        if a3 > 0.0 {
            g[2] += scale * 2.0 * a3;
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 4: InsideSphereRestraint
// ============================================================================

/// Packmol kind 4 — quadratic penalty forcing atom inside sphere.
#[derive(Debug, Clone, Copy)]
pub struct InsideSphereRestraint {
    pub center: [F; 3],
    pub radius: F,
}

impl InsideSphereRestraint {
    pub fn new(center: [F; 3], radius: F) -> Self {
        Self { center, radius }
    }
}

impl Restraint for InsideSphereRestraint {
    fn f(&self, pos: &[F; 3], _scale: F, scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let w = (x - c[0]).powi(2) + (y - c[1]).powi(2) + (z - c[2]).powi(2) - self.radius.powi(2);
        let a1 = w.max(0.0);
        scale2 * a1 * a1
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let d = (x - c[0]).powi(2) + (y - c[1]).powi(2) + (z - c[2]).powi(2) - self.radius.powi(2);
        if d > 0.0 {
            g[0] += 4.0 * scale2 * (x - c[0]) * d;
            g[1] += 4.0 * scale2 * (y - c[1]) * d;
            g[2] += 4.0 * scale2 * (z - c[2]) * d;
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 5: InsideEllipsoidRestraint
// ============================================================================

/// Packmol kind 5 — quadratic penalty forcing atom inside ellipsoid.
#[derive(Debug, Clone, Copy)]
pub struct InsideEllipsoidRestraint {
    pub center: [F; 3],
    pub axes: [F; 3],
    pub exponent: F,
}

impl InsideEllipsoidRestraint {
    pub fn new(center: [F; 3], axes: [F; 3], exponent: F) -> Self {
        Self {
            center,
            axes,
            exponent,
        }
    }
}

impl Restraint for InsideEllipsoidRestraint {
    fn f(&self, pos: &[F; 3], _scale: F, scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let ax = self.axes;
        let a1 = (x - c[0]).powi(2) / ax[0].powi(2);
        let a2 = (y - c[1]).powi(2) / ax[1].powi(2);
        let a3 = (z - c[2]).powi(2) / ax[2].powi(2);
        let w = a1 + a2 + a3 - self.exponent.powi(2);
        let v = w.max(0.0);
        scale2 * v * v
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let a1 = x - c[0];
        let b1 = y - c[1];
        let c1 = z - c[2];
        let a2 = self.axes[0].powi(2);
        let b2 = self.axes[1].powi(2);
        let c2 = self.axes[2].powi(2);
        let d = a1.powi(2) / a2 + b1.powi(2) / b2 + c1.powi(2) / c2 - self.exponent.powi(2);
        if d > 0.0 {
            g[0] += scale2 * 4.0 * d * a1 / a2;
            g[1] += scale2 * 4.0 * d * b1 / b2;
            g[2] += scale2 * 4.0 * d * c1 / c2;
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 6: OutsideCubeRestraint
// ============================================================================

/// Packmol kind 6 — linear penalty forcing atom outside axis-aligned cube.
#[derive(Debug, Clone, Copy)]
pub struct OutsideCubeRestraint {
    pub origin: [F; 3],
    pub side: F,
}

impl OutsideCubeRestraint {
    pub fn new(origin: [F; 3], side: F) -> Self {
        Self { origin, side }
    }
}

impl Restraint for OutsideCubeRestraint {
    fn f(&self, pos: &[F; 3], scale: F, _scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let (xmin, ymin, zmin) = (self.origin[0], self.origin[1], self.origin[2]);
        let (xmax, ymax, zmax) = (xmin + self.side, ymin + self.side, zmin + self.side);
        if x > xmin && x < xmax && y > ymin && y < ymax && z > zmin && z < zmax {
            let xmed = (xmax - xmin) / 2.0;
            let ymed = (ymax - ymin) / 2.0;
            let zmed = (zmax - zmin) / 2.0;
            let a1 = if x <= xmin + xmed { x - xmin } else { xmax - x };
            let a2 = if y <= ymin + ymed { y - ymin } else { ymax - y };
            let a3 = if z <= zmin + zmed { z - zmin } else { zmax - z };
            scale * (a1 + a2 + a3)
        } else {
            0.0
        }
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let (xmin, ymin, zmin) = (self.origin[0], self.origin[1], self.origin[2]);
        let (xmax, ymax, zmax) = (xmin + self.side, ymin + self.side, zmin + self.side);
        if x > xmin && x < xmax && y > ymin && y < ymax && z > zmin && z < zmax {
            let xmed = (xmax - xmin) / 2.0;
            let ymed = (ymax - ymin) / 2.0;
            let zmed = (zmax - zmin) / 2.0;
            let (a1, a4) = if x <= xmin + xmed {
                (1.0, 0.0)
            } else {
                (0.0, -1.0)
            };
            let (a2, a5) = if y <= ymin + ymed {
                (1.0, 0.0)
            } else {
                (0.0, -1.0)
            };
            let (a3, a6) = if z <= zmin + zmed {
                (1.0, 0.0)
            } else {
                (0.0, -1.0)
            };
            g[0] += scale * (a1 + a4);
            g[1] += scale * (a2 + a5);
            g[2] += scale * (a3 + a6);
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 7: OutsideBoxRestraint
// ============================================================================

/// Packmol kind 7 — linear penalty forcing atom outside axis-aligned box.
#[derive(Debug, Clone, Copy)]
pub struct OutsideBoxRestraint {
    pub min: [F; 3],
    pub max: [F; 3],
}

impl OutsideBoxRestraint {
    pub fn new(min: [F; 3], max: [F; 3]) -> Self {
        Self { min, max }
    }
}

impl Restraint for OutsideBoxRestraint {
    fn f(&self, pos: &[F; 3], scale: F, _scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let (xmin, ymin, zmin) = (self.min[0], self.min[1], self.min[2]);
        let (xmax, ymax, zmax) = (self.max[0], self.max[1], self.max[2]);
        if x > xmin && x < xmax && y > ymin && y < ymax && z > zmin && z < zmax {
            let xmed = (xmax - xmin) / 2.0;
            let ymed = (ymax - ymin) / 2.0;
            let zmed = (zmax - zmin) / 2.0;
            let a1 = if x <= xmin + xmed { x - xmin } else { xmax - x };
            let a2 = if y <= ymin + ymed { y - ymin } else { ymax - y };
            let a3 = if z <= zmin + zmed { z - zmin } else { zmax - z };
            scale * (a1 + a2 + a3)
        } else {
            0.0
        }
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let (xmin, ymin, zmin) = (self.min[0], self.min[1], self.min[2]);
        let (xmax, ymax, zmax) = (self.max[0], self.max[1], self.max[2]);
        if x > xmin && x < xmax && y > ymin && y < ymax && z > zmin && z < zmax {
            let xmed = (xmax - xmin) / 2.0;
            let ymed = (ymax - ymin) / 2.0;
            let zmed = (zmax - zmin) / 2.0;
            let (a1, a4) = if x <= xmin + xmed {
                (1.0, 0.0)
            } else {
                (0.0, -1.0)
            };
            let (a2, a5) = if y <= ymin + ymed {
                (1.0, 0.0)
            } else {
                (0.0, -1.0)
            };
            let (a3, a6) = if z <= zmin + zmed {
                (1.0, 0.0)
            } else {
                (0.0, -1.0)
            };
            g[0] += scale * (a1 + a4);
            g[1] += scale * (a2 + a5);
            g[2] += scale * (a3 + a6);
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 8: OutsideSphereRestraint
// ============================================================================

/// Packmol kind 8 — quadratic penalty forcing atom outside sphere.
#[derive(Debug, Clone, Copy)]
pub struct OutsideSphereRestraint {
    pub center: [F; 3],
    pub radius: F,
}

impl OutsideSphereRestraint {
    pub fn new(center: [F; 3], radius: F) -> Self {
        Self { center, radius }
    }
}

impl Restraint for OutsideSphereRestraint {
    fn f(&self, pos: &[F; 3], _scale: F, scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let w = (x - c[0]).powi(2) + (y - c[1]).powi(2) + (z - c[2]).powi(2) - self.radius.powi(2);
        let a1 = w.min(0.0);
        scale2 * a1 * a1
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let d = (x - c[0]).powi(2) + (y - c[1]).powi(2) + (z - c[2]).powi(2) - self.radius.powi(2);
        if d < 0.0 {
            g[0] += 4.0 * scale2 * (x - c[0]) * d;
            g[1] += 4.0 * scale2 * (y - c[1]) * d;
            g[2] += 4.0 * scale2 * (z - c[2]) * d;
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 9: OutsideEllipsoidRestraint
// ============================================================================

/// Packmol kind 9 — quadratic penalty forcing atom outside ellipsoid.
/// Note: Fortran source does NOT multiply by `scale2` for kind 9.
#[derive(Debug, Clone, Copy)]
pub struct OutsideEllipsoidRestraint {
    pub center: [F; 3],
    pub axes: [F; 3],
    pub exponent: F,
}

impl OutsideEllipsoidRestraint {
    pub fn new(center: [F; 3], axes: [F; 3], exponent: F) -> Self {
        Self {
            center,
            axes,
            exponent,
        }
    }
}

impl Restraint for OutsideEllipsoidRestraint {
    fn f(&self, pos: &[F; 3], _scale: F, _scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let ax = self.axes;
        let a1 = (x - c[0]).powi(2) / ax[0].powi(2);
        let a2 = (y - c[1]).powi(2) / ax[1].powi(2);
        let a3 = (z - c[2]).powi(2) / ax[2].powi(2);
        let w = a1 + a2 + a3 - self.exponent.powi(2);
        let v = w.min(0.0);
        v * v
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let a1 = x - c[0];
        let b1 = y - c[1];
        let c1 = z - c[2];
        let a2 = self.axes[0].powi(2);
        let b2 = self.axes[1].powi(2);
        let c2 = self.axes[2].powi(2);
        let d = a1.powi(2) / a2 + b1.powi(2) / b2 + c1.powi(2) / c2 - self.exponent.powi(2);
        if d < 0.0 {
            let ds = scale2 * d;
            g[0] += 4.0 * ds * a1 / a2;
            g[1] += 4.0 * ds * b1 / b2;
            g[2] += 4.0 * ds * c1 / c2;
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 10: AbovePlaneRestraint
// ============================================================================

/// Packmol kind 10 — quadratic penalty forcing atom above plane `n·x >= d`.
#[derive(Debug, Clone, Copy)]
pub struct AbovePlaneRestraint {
    pub normal: [F; 3],
    pub distance: F,
}

impl AbovePlaneRestraint {
    pub fn new(normal: [F; 3], distance: F) -> Self {
        Self { normal, distance }
    }
}

impl Restraint for AbovePlaneRestraint {
    fn f(&self, pos: &[F; 3], scale: F, _scale2: F) -> F {
        let n = self.normal;
        let w = n[0] * pos[0] + n[1] * pos[1] + n[2] * pos[2] - self.distance;
        let a1 = w.min(0.0);
        scale * a1 * a1
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let n = self.normal;
        let d = n[0] * pos[0] + n[1] * pos[1] + n[2] * pos[2] - self.distance;
        if d < 0.0 {
            let ds = scale * d;
            g[0] += 2.0 * n[0] * ds;
            g[1] += 2.0 * n[1] * ds;
            g[2] += 2.0 * n[2] * ds;
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 11: BelowPlaneRestraint
// ============================================================================

/// Packmol kind 11 — quadratic penalty forcing atom below plane `n·x <= d`.
#[derive(Debug, Clone, Copy)]
pub struct BelowPlaneRestraint {
    pub normal: [F; 3],
    pub distance: F,
}

impl BelowPlaneRestraint {
    pub fn new(normal: [F; 3], distance: F) -> Self {
        Self { normal, distance }
    }
}

impl Restraint for BelowPlaneRestraint {
    fn f(&self, pos: &[F; 3], scale: F, _scale2: F) -> F {
        let n = self.normal;
        let w = n[0] * pos[0] + n[1] * pos[1] + n[2] * pos[2] - self.distance;
        let a1 = w.max(0.0);
        scale * a1 * a1
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let n = self.normal;
        let d = n[0] * pos[0] + n[1] * pos[1] + n[2] * pos[2] - self.distance;
        if d > 0.0 {
            let ds = scale * d;
            g[0] += 2.0 * n[0] * ds;
            g[1] += 2.0 * n[1] * ds;
            g[2] += 2.0 * n[2] * ds;
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 12: InsideCylinderRestraint
// ============================================================================

/// Packmol kind 12 — quadratic penalty forcing atom inside finite cylinder.
#[derive(Debug, Clone, Copy)]
pub struct InsideCylinderRestraint {
    pub center: [F; 3],
    pub axis: [F; 3],
    pub radius: F,
    pub length: F,
}

impl InsideCylinderRestraint {
    pub fn new(center: [F; 3], axis: [F; 3], radius: F, length: F) -> Self {
        Self {
            center,
            axis,
            radius,
            length,
        }
    }
}

impl Restraint for InsideCylinderRestraint {
    fn f(&self, pos: &[F; 3], _scale: F, scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let ax = self.axis;
        let (a1, a2, a3) = (x - c[0], y - c[1], z - c[2]);
        let vnorm = (ax[0].powi(2) + ax[1].powi(2) + ax[2].powi(2)).sqrt();
        let (vv1, vv2, vv3) = (ax[0] / vnorm, ax[1] / vnorm, ax[2] / vnorm);
        let w = vv1 * a1 + vv2 * a2 + vv3 * a3;
        let d = (a1 - vv1 * w).powi(2) + (a2 - vv2 * w).powi(2) + (a3 - vv3 * w).powi(2);
        scale2
            * ((-w).max(0.0).powi(2)
                + (w - self.length).max(0.0).powi(2)
                + (d - self.radius.powi(2)).max(0.0).powi(2))
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let ax = self.axis;
        let (a1, a2, a3) = (x - c[0], y - c[1], z - c[2]);
        let vnorm = (ax[0].powi(2) + ax[1].powi(2) + ax[2].powi(2)).sqrt();
        let (vv1, vv2, vv3) = (ax[0] / vnorm, ax[1] / vnorm, ax[2] / vnorm);
        let w = vv1 * a1 + vv2 * a2 + vv3 * a3;
        let d = (a1 - vv1 * w).powi(2) + (a2 - vv2 * w).powi(2) + (a3 - vv3 * w).powi(2);
        let len = self.length;
        let r2 = self.radius.powi(2);
        let rg0 = scale2
            * (-2.0 * (-w).max(0.0) * vv1
                + 2.0 * (w - len).max(0.0) * vv1
                + 2.0
                    * (d - r2).max(0.0)
                    * (2.0 * (a1 - vv1 * w) * (1.0 - vv1.powi(2))
                        + 2.0 * (a2 - vv2 * w) * (-vv2 * vv1)
                        + 2.0 * (a3 - vv3 * w) * (-vv3 * vv1)));
        let rg1 = scale2
            * (-2.0 * (-w).max(0.0) * vv2
                + 2.0 * (w - len).max(0.0) * vv2
                + 2.0
                    * (d - r2).max(0.0)
                    * (2.0 * (a1 - vv1 * w) * (-vv1 * vv2)
                        + 2.0 * (a2 - vv2 * w) * (1.0 - vv2.powi(2))
                        + 2.0 * (a3 - vv3 * w) * (-vv3 * vv2)));
        let rg2 = scale2
            * (-2.0 * (-w).max(0.0) * vv3
                + 2.0 * (w - len).max(0.0) * vv3
                + 2.0
                    * (d - r2).max(0.0)
                    * (2.0 * (a1 - vv1 * w) * (-vv1 * vv3)
                        + 2.0 * (a2 - vv2 * w) * (-vv2 * vv3)
                        + 2.0 * (a3 - vv3 * w) * (1.0 - vv3.powi(2))));
        g[0] += rg0;
        g[1] += rg1;
        g[2] += rg2;
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 13: OutsideCylinderRestraint
// ============================================================================

/// Packmol kind 13 — quadratic penalty forcing atom outside finite cylinder.
#[derive(Debug, Clone, Copy)]
pub struct OutsideCylinderRestraint {
    pub center: [F; 3],
    pub axis: [F; 3],
    pub radius: F,
    pub length: F,
}

impl OutsideCylinderRestraint {
    pub fn new(center: [F; 3], axis: [F; 3], radius: F, length: F) -> Self {
        Self {
            center,
            axis,
            radius,
            length,
        }
    }
}

impl Restraint for OutsideCylinderRestraint {
    fn f(&self, pos: &[F; 3], _scale: F, scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let ax = self.axis;
        let (a1, a2, a3) = (x - c[0], y - c[1], z - c[2]);
        let vnorm = (ax[0].powi(2) + ax[1].powi(2) + ax[2].powi(2)).sqrt();
        let (vv1, vv2, vv3) = (ax[0] / vnorm, ax[1] / vnorm, ax[2] / vnorm);
        let w = vv1 * a1 + vv2 * a2 + vv3 * a3;
        let d = (a1 - vv1 * w).powi(2) + (a2 - vv2 * w).powi(2) + (a3 - vv3 * w).powi(2);
        scale2
            * ((-w).min(0.0).powi(2)
                * (w - self.length).min(0.0).powi(2)
                * (d - self.radius.powi(2)).min(0.0).powi(2))
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let c = self.center;
        let ax = self.axis;
        let (a1, a2, a3) = (x - c[0], y - c[1], z - c[2]);
        let vnorm = (ax[0].powi(2) + ax[1].powi(2) + ax[2].powi(2)).sqrt();
        let (vv1, vv2, vv3) = (ax[0] / vnorm, ax[1] / vnorm, ax[2] / vnorm);
        let w = vv1 * a1 + vv2 * a2 + vv3 * a3;
        let d = (a1 - vv1 * w).powi(2) + (a2 - vv2 * w).powi(2) + (a3 - vv3 * w).powi(2);
        let len = self.length;
        let r2 = self.radius.powi(2);
        let fra = (-w).min(0.0).powi(2);
        let frb = (w - len).min(0.0).powi(2);
        let frc = (d - r2).min(0.0).powi(2);
        let frab = fra * frb;
        let frac = fra * frc;
        let frbc = frb * frc;
        let dfra0 = -2.0 * (-w).min(0.0) * vv1;
        let dfrb0 = 2.0 * (w - len).min(0.0) * vv1;
        let dfrc0 = 2.0
            * (d - r2).min(0.0)
            * (2.0 * (a1 - vv1 * w) * (1.0 - vv1.powi(2))
                + 2.0 * (a2 - vv2 * w) * (-vv2 * vv1)
                + 2.0 * (a3 - vv3 * w) * (-vv3 * vv1));
        let dfra1 = -2.0 * (-w).min(0.0) * vv2;
        let dfrb1 = 2.0 * (w - len).min(0.0) * vv2;
        let dfrc1 = 2.0
            * (d - r2).min(0.0)
            * (2.0 * (a1 - vv1 * w) * (-vv1 * vv2)
                + 2.0 * (a2 - vv2 * w) * (1.0 - vv2.powi(2))
                + 2.0 * (a3 - vv3 * w) * (-vv3 * vv2));
        let dfra2 = -2.0 * (-w).min(0.0) * vv3;
        let dfrb2 = 2.0 * (w - len).min(0.0) * vv3;
        let dfrc2 = 2.0
            * (d - r2).min(0.0)
            * (2.0 * (a1 - vv1 * w) * (-vv1 * vv3)
                + 2.0 * (a2 - vv2 * w) * (-vv2 * vv3)
                + 2.0 * (a3 - vv3 * w) * (1.0 - vv3.powi(2)));
        g[0] += scale2 * (dfra0 * frbc + dfrb0 * frac + dfrc0 * frab);
        g[1] += scale2 * (dfra1 * frbc + dfrb1 * frac + dfrc1 * frab);
        g[2] += scale2 * (dfra2 * frbc + dfrb2 * frac + dfrc2 * frab);
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 14: AboveGaussianRestraint
// ============================================================================

/// Packmol kind 14 — quadratic penalty above Gaussian surface.
#[derive(Debug, Clone, Copy)]
pub struct AboveGaussianRestraint {
    pub cx: F,
    pub cy: F,
    pub sx: F,
    pub sy: F,
    pub z0: F,
    pub height: F,
}

impl AboveGaussianRestraint {
    pub fn new(cx: F, cy: F, sx: F, sy: F, z0: F, height: F) -> Self {
        Self {
            cx,
            cy,
            sx,
            sy,
            z0,
            height,
        }
    }
}

impl Restraint for AboveGaussianRestraint {
    fn f(&self, pos: &[F; 3], scale: F, _scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let e1 = -(x - self.cx).powi(2) / (2.0 * self.sx.powi(2));
        let e2 = -(y - self.cy).powi(2) / (2.0 * self.sy.powi(2));
        let w = if e1 + e2 <= -50.0 {
            -(z - self.z0)
        } else {
            self.height * (e1 + e2).exp() - (z - self.z0)
        };
        let a1 = w.max(0.0);
        scale * a1 * a1
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let e1 = -(x - self.cx).powi(2) / (2.0 * self.sx.powi(2));
        let e2 = -(y - self.cy).powi(2) / (2.0 * self.sy.powi(2));
        let d_raw = if e1 + e2 <= -50.0 {
            -(z - self.z0)
        } else {
            self.height * (e1 + e2).exp() - (z - self.z0)
        };
        if d_raw > 0.0 {
            let d = scale * d_raw;
            g[0] += -2.0 * d * (x - self.cx) * (d + (z - self.z0)) / self.sx.powi(2);
            g[1] += -2.0 * d * (y - self.cy) * (d + (z - self.z0)) / self.sy.powi(2);
            g[2] += -2.0 * d;
        }
        self.f(pos, scale, scale2)
    }
}

// ============================================================================
// Kind 15: BelowGaussianRestraint
// ============================================================================

/// Packmol kind 15 — quadratic penalty below Gaussian surface.
#[derive(Debug, Clone, Copy)]
pub struct BelowGaussianRestraint {
    pub cx: F,
    pub cy: F,
    pub sx: F,
    pub sy: F,
    pub z0: F,
    pub height: F,
}

impl BelowGaussianRestraint {
    pub fn new(cx: F, cy: F, sx: F, sy: F, z0: F, height: F) -> Self {
        Self {
            cx,
            cy,
            sx,
            sy,
            z0,
            height,
        }
    }
}

impl Restraint for BelowGaussianRestraint {
    fn f(&self, pos: &[F; 3], scale: F, _scale2: F) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let e1 = -(x - self.cx).powi(2) / (2.0 * self.sx.powi(2));
        let e2 = -(y - self.cy).powi(2) / (2.0 * self.sy.powi(2));
        let w = if e1 + e2 <= -50.0 {
            -(z - self.z0)
        } else {
            self.height * (e1 + e2).exp() - (z - self.z0)
        };
        let a1 = w.min(0.0);
        scale * a1 * a1
    }

    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let e1 = -(x - self.cx).powi(2) / (2.0 * self.sx.powi(2));
        let e2 = -(y - self.cy).powi(2) / (2.0 * self.sy.powi(2));
        let d_raw = if e1 + e2 <= -50.0 {
            -(z - self.z0)
        } else {
            self.height * (e1 + e2).exp() - (z - self.z0)
        };
        if d_raw < 0.0 {
            let d = scale * d_raw;
            g[0] += -2.0 * d * (x - self.cx) * (d + (z - self.z0)) / self.sx.powi(2);
            g[1] += -2.0 * d * (y - self.cy) * (d + (z - self.z0)) / self.sy.powi(2);
            g[2] += -2.0 * d;
        }
        self.f(pos, scale, scale2)
    }
}
