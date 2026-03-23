//! Constraint types for molecular packing.
//! Exact port of `comprest.f90` and `gwalls.f90` from Packmol.
//!
//! Each `Restraint` stores all parameters needed to compute the penalty
//! function value and gradient for a single atom.

use molrs::types::F;
/// A single restraint on an atom position.
/// Parameters are stored in the same layout as Packmol's `restpars(irest, 1..9)`.
#[derive(Debug, Clone)]
pub struct Restraint {
    pub kind: u8, // type number 2-15 as in comprest.f90
    pub params: [F; 9],
}

impl Restraint {
    // ---- Constructors matching Packmol constraint types ----

    /// Type 2: inside cube (origin + side length).
    pub fn inside_cube(origin: [F; 3], side: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = origin[0];
        params[1] = origin[1];
        params[2] = origin[2];
        params[3] = side;
        Self { kind: 2, params }
    }

    /// Type 3: inside box (min corner, max corner).
    pub fn inside_box(min: [F; 3], max: [F; 3]) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = min[0];
        params[1] = min[1];
        params[2] = min[2];
        params[3] = max[0];
        params[4] = max[1];
        params[5] = max[2];
        Self { kind: 3, params }
    }

    /// Type 4: inside sphere (center, radius).
    pub fn inside_sphere(center: [F; 3], radius: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = center[0];
        params[1] = center[1];
        params[2] = center[2];
        params[3] = radius;
        Self { kind: 4, params }
    }

    /// Type 5: inside ellipsoid (center, semi-axes a1,a2,a3, exponent).
    pub fn inside_ellipsoid(center: [F; 3], axes: [F; 3], exponent: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = center[0];
        params[1] = center[1];
        params[2] = center[2];
        params[3] = axes[0];
        params[4] = axes[1];
        params[5] = axes[2];
        params[6] = exponent;
        Self { kind: 5, params }
    }

    /// Type 6: outside cube.
    pub fn outside_cube(origin: [F; 3], side: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = origin[0];
        params[1] = origin[1];
        params[2] = origin[2];
        params[3] = side;
        Self { kind: 6, params }
    }

    /// Type 7: outside box.
    pub fn outside_box(min: [F; 3], max: [F; 3]) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = min[0];
        params[1] = min[1];
        params[2] = min[2];
        params[3] = max[0];
        params[4] = max[1];
        params[5] = max[2];
        Self { kind: 7, params }
    }

    /// Type 8: outside sphere.
    pub fn outside_sphere(center: [F; 3], radius: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = center[0];
        params[1] = center[1];
        params[2] = center[2];
        params[3] = radius;
        Self { kind: 8, params }
    }

    /// Type 9: outside ellipsoid.
    pub fn outside_ellipsoid(center: [F; 3], axes: [F; 3], exponent: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = center[0];
        params[1] = center[1];
        params[2] = center[2];
        params[3] = axes[0];
        params[4] = axes[1];
        params[5] = axes[2];
        params[6] = exponent;
        Self { kind: 9, params }
    }

    /// Type 10: above plane (n·x >= d).
    pub fn above_plane(normal: [F; 3], distance: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = normal[0];
        params[1] = normal[1];
        params[2] = normal[2];
        params[3] = distance;
        Self { kind: 10, params }
    }

    /// Type 11: below plane (n·x <= d).
    pub fn below_plane(normal: [F; 3], distance: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = normal[0];
        params[1] = normal[1];
        params[2] = normal[2];
        params[3] = distance;
        Self { kind: 11, params }
    }

    /// Type 12: inside cylinder (center, axis, radius, length).
    pub fn inside_cylinder(center: [F; 3], axis: [F; 3], radius: F, length: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = center[0];
        params[1] = center[1];
        params[2] = center[2];
        params[3] = axis[0];
        params[4] = axis[1];
        params[5] = axis[2];
        params[6] = radius;
        params[8] = length;
        Self { kind: 12, params }
    }

    /// Type 13: outside cylinder (center, axis, radius, length).
    pub fn outside_cylinder(center: [F; 3], axis: [F; 3], radius: F, length: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = center[0];
        params[1] = center[1];
        params[2] = center[2];
        params[3] = axis[0];
        params[4] = axis[1];
        params[5] = axis[2];
        params[6] = radius;
        params[8] = length;
        Self { kind: 13, params }
    }

    /// Type 14: above gaussian surface.
    pub fn above_gaussian(cx: F, cy: F, sx: F, sy: F, z0: F, h: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = cx;
        params[1] = cy;
        params[2] = sx;
        params[3] = sy;
        params[4] = z0;
        params[5] = h;
        Self { kind: 14, params }
    }

    /// Type 15: below gaussian surface.
    pub fn below_gaussian(cx: F, cy: F, sx: F, sy: F, z0: F, h: F) -> Self {
        let mut params = [0.0 as F; 9];
        params[0] = cx;
        params[1] = cy;
        params[2] = sx;
        params[3] = sy;
        params[4] = z0;
        params[5] = h;
        Self { kind: 15, params }
    }

    // ---- comprest: penalty function value ----

    /// Compute restraint penalty value for atom at position `pos`.
    /// Exact port of `comprest.f90`.
    pub fn value(&self, pos: &[F; 3], scale: F, scale2: F) -> F {
        let p = &self.params;
        let (x, y, z) = (pos[0], pos[1], pos[2]);

        match self.kind {
            2 => {
                // inside cube
                let clength = p[3];
                let (xmin, ymin, zmin) = (p[0], p[1], p[2]);
                let (xmax, ymax, zmax) = (p[0] + clength, p[1] + clength, p[2] + clength);
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
            3 => {
                // inside box
                let (xmin, ymin, zmin) = (p[0], p[1], p[2]);
                let (xmax, ymax, zmax) = (p[3], p[4], p[5]);
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
            4 => {
                // inside sphere
                let w = (x - p[0]).powi(2) + (y - p[1]).powi(2) + (z - p[2]).powi(2) - p[3].powi(2);
                let a1 = w.max(0.0);
                scale2 * a1 * a1
            }
            5 => {
                // inside ellipsoid
                let a1 = (x - p[0]).powi(2) / p[3].powi(2);
                let a2 = (y - p[1]).powi(2) / p[4].powi(2);
                let a3 = (z - p[2]).powi(2) / p[5].powi(2);
                let w = a1 + a2 + a3 - p[6].powi(2);
                let v = w.max(0.0);
                scale2 * v * v
            }
            6 => {
                // outside cube
                let clength = p[3];
                let (xmin, ymin, zmin) = (p[0], p[1], p[2]);
                let (xmax, ymax, zmax) = (p[0] + clength, p[1] + clength, p[2] + clength);
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
            7 => {
                // outside box
                let (xmin, ymin, zmin) = (p[0], p[1], p[2]);
                let (xmax, ymax, zmax) = (p[3], p[4], p[5]);
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
            8 => {
                // outside sphere
                let w = (x - p[0]).powi(2) + (y - p[1]).powi(2) + (z - p[2]).powi(2) - p[3].powi(2);
                let a1 = w.min(0.0);
                scale2 * a1 * a1
            }
            9 => {
                // outside ellipsoid
                let a1 = (x - p[0]).powi(2) / p[3].powi(2);
                let a2 = (y - p[1]).powi(2) / p[4].powi(2);
                let a3 = (z - p[2]).powi(2) / p[5].powi(2);
                let w = a1 + a2 + a3 - p[6].powi(2);
                let v = w.min(0.0);
                v * v // no scale2 in Fortran for type 9
            }
            10 => {
                // above plane: n·x >= d
                let w = p[0] * x + p[1] * y + p[2] * z - p[3];
                let a1 = w.min(0.0);
                scale * a1 * a1
            }
            11 => {
                // below plane: n·x <= d
                let w = p[0] * x + p[1] * y + p[2] * z - p[3];
                let a1 = w.max(0.0);
                scale * a1 * a1
            }
            12 => {
                // inside cylinder
                let (a1, a2, a3) = (x - p[0], y - p[1], z - p[2]);
                let vnorm = (p[3].powi(2) + p[4].powi(2) + p[5].powi(2)).sqrt();
                let (vv1, vv2, vv3) = (p[3] / vnorm, p[4] / vnorm, p[5] / vnorm);
                let w = vv1 * a1 + vv2 * a2 + vv3 * a3;
                let d = (a1 - vv1 * w).powi(2) + (a2 - vv2 * w).powi(2) + (a3 - vv3 * w).powi(2);
                scale2
                    * ((-w).max(0.0).powi(2)
                        + (w - p[8]).max(0.0).powi(2)
                        + (d - p[6].powi(2)).max(0.0).powi(2))
            }
            13 => {
                // outside cylinder
                let (a1, a2, a3) = (x - p[0], y - p[1], z - p[2]);
                let vnorm = (p[3].powi(2) + p[4].powi(2) + p[5].powi(2)).sqrt();
                let (vv1, vv2, vv3) = (p[3] / vnorm, p[4] / vnorm, p[5] / vnorm);
                let w = vv1 * a1 + vv2 * a2 + vv3 * a3;
                let d = (a1 - vv1 * w).powi(2) + (a2 - vv2 * w).powi(2) + (a3 - vv3 * w).powi(2);
                scale2
                    * ((-w).min(0.0).powi(2)
                        * (w - p[8]).min(0.0).powi(2)
                        * (d - p[6].powi(2)).min(0.0).powi(2))
            }
            14 => {
                // above gaussian surface
                let e1 = -(x - p[0]).powi(2) / (2.0 * p[2].powi(2));
                let e2 = -(y - p[1]).powi(2) / (2.0 * p[3].powi(2));
                let w = if e1 + e2 <= -50.0 {
                    -(z - p[4])
                } else {
                    p[5] * (e1 + e2).exp() - (z - p[4])
                };
                let a1 = w.max(0.0);
                scale * a1 * a1
            }
            15 => {
                // below gaussian surface
                let e1 = -(x - p[0]).powi(2) / (2.0 * p[2].powi(2));
                let e2 = -(y - p[1]).powi(2) / (2.0 * p[3].powi(2));
                let w = if e1 + e2 <= -50.0 {
                    -(z - p[4])
                } else {
                    p[5] * (e1 + e2).exp() - (z - p[4])
                };
                let a1 = w.min(0.0);
                scale * a1 * a1
            }
            _ => 0.0,
        }
    }

    // ---- gwalls: gradient accumulation ----

    /// Accumulate constraint gradient into `gxcar`.
    /// Exact port of `gwalls.f90`.
    pub fn gradient(&self, pos: &[F; 3], scale: F, scale2: F, gxcar: &mut [F; 3]) {
        let p = &self.params;
        let (x, y, z) = (pos[0], pos[1], pos[2]);

        match self.kind {
            2 => {
                let clength = p[3];
                let (xmin, ymin, zmin) = (p[0], p[1], p[2]);
                let (xmax, ymax, zmax) = (p[0] + clength, p[1] + clength, p[2] + clength);
                let a1 = x - xmin;
                let a2 = y - ymin;
                let a3 = z - zmin;
                if a1 < 0.0 {
                    gxcar[0] += scale * 2.0 * a1;
                }
                if a2 < 0.0 {
                    gxcar[1] += scale * 2.0 * a2;
                }
                if a3 < 0.0 {
                    gxcar[2] += scale * 2.0 * a3;
                }
                let a1 = x - xmax;
                let a2 = y - ymax;
                let a3 = z - zmax;
                if a1 > 0.0 {
                    gxcar[0] += scale * 2.0 * a1;
                }
                if a2 > 0.0 {
                    gxcar[1] += scale * 2.0 * a2;
                }
                if a3 > 0.0 {
                    gxcar[2] += scale * 2.0 * a3;
                }
            }
            3 => {
                let (xmin, ymin, zmin) = (p[0], p[1], p[2]);
                let (xmax, ymax, zmax) = (p[3], p[4], p[5]);
                let a1 = x - xmin;
                let a2 = y - ymin;
                let a3 = z - zmin;
                if a1 < 0.0 {
                    gxcar[0] += scale * 2.0 * a1;
                }
                if a2 < 0.0 {
                    gxcar[1] += scale * 2.0 * a2;
                }
                if a3 < 0.0 {
                    gxcar[2] += scale * 2.0 * a3;
                }
                let a1 = x - xmax;
                let a2 = y - ymax;
                let a3 = z - zmax;
                if a1 > 0.0 {
                    gxcar[0] += scale * 2.0 * a1;
                }
                if a2 > 0.0 {
                    gxcar[1] += scale * 2.0 * a2;
                }
                if a3 > 0.0 {
                    gxcar[2] += scale * 2.0 * a3;
                }
            }
            4 => {
                let d = (x - p[0]).powi(2) + (y - p[1]).powi(2) + (z - p[2]).powi(2) - p[3].powi(2);
                if d > 0.0 {
                    gxcar[0] += 4.0 * scale2 * (x - p[0]) * d;
                    gxcar[1] += 4.0 * scale2 * (y - p[1]) * d;
                    gxcar[2] += 4.0 * scale2 * (z - p[2]) * d;
                }
            }
            5 => {
                let a1 = x - p[0];
                let b1 = y - p[1];
                let c1 = z - p[2];
                let a2 = p[3].powi(2);
                let b2 = p[4].powi(2);
                let c2 = p[5].powi(2);
                let d = a1.powi(2) / a2 + b1.powi(2) / b2 + c1.powi(2) / c2 - p[6].powi(2);
                if d > 0.0 {
                    gxcar[0] += scale2 * 4.0 * d * a1 / a2;
                    gxcar[1] += scale2 * 4.0 * d * b1 / b2;
                    gxcar[2] += scale2 * 4.0 * d * c1 / c2;
                }
            }
            6 => {
                let clength = p[3];
                let (xmin, ymin, zmin) = (p[0], p[1], p[2]);
                let (xmax, ymax, zmax) = (p[0] + clength, p[1] + clength, p[2] + clength);
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
                    gxcar[0] += scale * (a1 + a4);
                    gxcar[1] += scale * (a2 + a5);
                    gxcar[2] += scale * (a3 + a6);
                }
            }
            7 => {
                let (xmin, ymin, zmin) = (p[0], p[1], p[2]);
                let (xmax, ymax, zmax) = (p[3], p[4], p[5]);
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
                    gxcar[0] += scale * (a1 + a4);
                    gxcar[1] += scale * (a2 + a5);
                    gxcar[2] += scale * (a3 + a6);
                }
            }
            8 => {
                let d = (x - p[0]).powi(2) + (y - p[1]).powi(2) + (z - p[2]).powi(2) - p[3].powi(2);
                if d < 0.0 {
                    gxcar[0] += 4.0 * scale2 * (x - p[0]) * d;
                    gxcar[1] += 4.0 * scale2 * (y - p[1]) * d;
                    gxcar[2] += 4.0 * scale2 * (z - p[2]) * d;
                }
            }
            9 => {
                let a1 = x - p[0];
                let b1 = y - p[1];
                let c1 = z - p[2];
                let a2 = p[3].powi(2);
                let b2 = p[4].powi(2);
                let c2 = p[5].powi(2);
                let d = a1.powi(2) / a2 + b1.powi(2) / b2 + c1.powi(2) / c2 - p[6].powi(2);
                if d < 0.0 {
                    let ds = scale2 * d;
                    gxcar[0] += 4.0 * ds * a1 / a2;
                    gxcar[1] += 4.0 * ds * b1 / b2;
                    gxcar[2] += 4.0 * ds * c1 / c2;
                }
            }
            10 => {
                let d = p[0] * x + p[1] * y + p[2] * z - p[3];
                if d < 0.0 {
                    let ds = scale * d;
                    gxcar[0] += 2.0 * p[0] * ds;
                    gxcar[1] += 2.0 * p[1] * ds;
                    gxcar[2] += 2.0 * p[2] * ds;
                }
            }
            11 => {
                let d = p[0] * x + p[1] * y + p[2] * z - p[3];
                if d > 0.0 {
                    let ds = scale * d;
                    gxcar[0] += 2.0 * p[0] * ds;
                    gxcar[1] += 2.0 * p[1] * ds;
                    gxcar[2] += 2.0 * p[2] * ds;
                }
            }
            12 => {
                let (a1, a2, a3) = (x - p[0], y - p[1], z - p[2]);
                let vnorm = (p[3].powi(2) + p[4].powi(2) + p[5].powi(2)).sqrt();
                let (vv1, vv2, vv3) = (p[3] / vnorm, p[4] / vnorm, p[5] / vnorm);
                let w = vv1 * a1 + vv2 * a2 + vv3 * a3;
                let d = (a1 - vv1 * w).powi(2) + (a2 - vv2 * w).powi(2) + (a3 - vv3 * w).powi(2);
                let rg0 = scale2
                    * (-2.0 * (-w).max(0.0) * vv1
                        + 2.0 * (w - p[8]).max(0.0) * vv1
                        + 2.0
                            * (d - p[6].powi(2)).max(0.0)
                            * (2.0 * (a1 - vv1 * w) * (1.0 - vv1.powi(2))
                                + 2.0 * (a2 - vv2 * w) * (-vv2 * vv1)
                                + 2.0 * (a3 - vv3 * w) * (-vv3 * vv1)));
                let rg1 = scale2
                    * (-2.0 * (-w).max(0.0) * vv2
                        + 2.0 * (w - p[8]).max(0.0) * vv2
                        + 2.0
                            * (d - p[6].powi(2)).max(0.0)
                            * (2.0 * (a1 - vv1 * w) * (-vv1 * vv2)
                                + 2.0 * (a2 - vv2 * w) * (1.0 - vv2.powi(2))
                                + 2.0 * (a3 - vv3 * w) * (-vv3 * vv2)));
                let rg2 = scale2
                    * (-2.0 * (-w).max(0.0) * vv3
                        + 2.0 * (w - p[8]).max(0.0) * vv3
                        + 2.0
                            * (d - p[6].powi(2)).max(0.0)
                            * (2.0 * (a1 - vv1 * w) * (-vv1 * vv3)
                                + 2.0 * (a2 - vv2 * w) * (-vv2 * vv3)
                                + 2.0 * (a3 - vv3 * w) * (1.0 - vv3.powi(2))));
                gxcar[0] += rg0;
                gxcar[1] += rg1;
                gxcar[2] += rg2;
            }
            13 => {
                let (a1, a2, a3) = (x - p[0], y - p[1], z - p[2]);
                let vnorm = (p[3].powi(2) + p[4].powi(2) + p[5].powi(2)).sqrt();
                let (vv1, vv2, vv3) = (p[3] / vnorm, p[4] / vnorm, p[5] / vnorm);
                let w = vv1 * a1 + vv2 * a2 + vv3 * a3;
                let d = (a1 - vv1 * w).powi(2) + (a2 - vv2 * w).powi(2) + (a3 - vv3 * w).powi(2);
                let fra = (-w).min(0.0).powi(2);
                let frb = (w - p[8]).min(0.0).powi(2);
                let frc = (d - p[6].powi(2)).min(0.0).powi(2);
                let frab = fra * frb;
                let frac = fra * frc;
                let frbc = frb * frc;
                let dfra0 = -2.0 * (-w).min(0.0) * vv1;
                let dfrb0 = 2.0 * (w - p[8]).min(0.0) * vv1;
                let dfrc0 = 2.0
                    * (d - p[6].powi(2)).min(0.0)
                    * (2.0 * (a1 - vv1 * w) * (1.0 - vv1.powi(2))
                        + 2.0 * (a2 - vv2 * w) * (-vv2 * vv1)
                        + 2.0 * (a3 - vv3 * w) * (-vv3 * vv1));
                let dfra1 = -2.0 * (-w).min(0.0) * vv2;
                let dfrb1 = 2.0 * (w - p[8]).min(0.0) * vv2;
                let dfrc1 = 2.0
                    * (d - p[6].powi(2)).min(0.0)
                    * (2.0 * (a1 - vv1 * w) * (-vv1 * vv2)
                        + 2.0 * (a2 - vv2 * w) * (1.0 - vv2.powi(2))
                        + 2.0 * (a3 - vv3 * w) * (-vv3 * vv2));
                let dfra2 = -2.0 * (-w).min(0.0) * vv3;
                let dfrb2 = 2.0 * (w - p[8]).min(0.0) * vv3;
                let dfrc2 = 2.0
                    * (d - p[6].powi(2)).min(0.0)
                    * (2.0 * (a1 - vv1 * w) * (-vv1 * vv3)
                        + 2.0 * (a2 - vv2 * w) * (-vv2 * vv3)
                        + 2.0 * (a3 - vv3 * w) * (1.0 - vv3.powi(2)));
                gxcar[0] += scale2 * (dfra0 * frbc + dfrb0 * frac + dfrc0 * frab);
                gxcar[1] += scale2 * (dfra1 * frbc + dfrb1 * frac + dfrc1 * frab);
                gxcar[2] += scale2 * (dfra2 * frbc + dfrb2 * frac + dfrc2 * frab);
            }
            14 => {
                let e1 = -(x - p[0]).powi(2) / (2.0 * p[2].powi(2));
                let e2 = -(y - p[1]).powi(2) / (2.0 * p[3].powi(2));
                let d_raw = if e1 + e2 <= -50.0 {
                    -(z - p[4])
                } else {
                    p[5] * (e1 + e2).exp() - (z - p[4])
                };
                if d_raw > 0.0 {
                    let d = scale * d_raw;
                    gxcar[0] += -2.0 * d * (x - p[0]) * (d + (z - p[4])) / p[2].powi(2);
                    gxcar[1] += -2.0 * d * (y - p[1]) * (d + (z - p[4])) / p[3].powi(2);
                    gxcar[2] += -2.0 * d;
                }
            }
            15 => {
                let e1 = -(x - p[0]).powi(2) / (2.0 * p[2].powi(2));
                let e2 = -(y - p[1]).powi(2) / (2.0 * p[3].powi(2));
                let d_raw = if e1 + e2 <= -50.0 {
                    -(z - p[4])
                } else {
                    p[5] * (e1 + e2).exp() - (z - p[4])
                };
                if d_raw < 0.0 {
                    let d = scale * d_raw;
                    gxcar[0] += -2.0 * d * (x - p[0]) * (d + (z - p[4])) / p[2].powi(2);
                    gxcar[1] += -2.0 * d * (y - p[1]) * (d + (z - p[4])) / p[3].powi(2);
                    gxcar[2] += -2.0 * d;
                }
            }
            _ => {}
        }
    }
}

/// High-level constraint builder used in `Target`.
/// Wraps one or more `Restraint`s that are applied to all atoms of a molecule.
#[derive(Debug, Clone, Default)]
pub struct MoleculeConstraint {
    pub restraints: Vec<Restraint>,
}

impl MoleculeConstraint {
    pub fn new() -> Self {
        Self {
            restraints: Vec::new(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, r: Restraint) -> Self {
        self.restraints.push(r);
        self
    }

    pub fn and(mut self, other: impl Into<MoleculeConstraint>) -> Self {
        let other_mc = other.into();
        self.restraints.extend(other_mc.restraints);
        self
    }
}

// ============================================================================
// RegionConstraint trait — enables fluent .and() chaining on constraint types
// ============================================================================

/// Trait implemented by all high-level constraint types.
/// Import this trait to use the `.and()` method for composing constraints.
pub trait RegionConstraint: Into<MoleculeConstraint> + Clone + Sized {
    /// Compose this constraint with another, returning a combined `MoleculeConstraint`.
    fn and<C: Into<MoleculeConstraint>>(self, other: C) -> MoleculeConstraint {
        let mut mc: MoleculeConstraint = self.into();
        let other_mc: MoleculeConstraint = other.into();
        mc.restraints.extend(other_mc.restraints);
        mc
    }
}

impl RegionConstraint for MoleculeConstraint {}

// ============================================================================
// Convenience constraint types — thin wrappers around MoleculeConstraint
// ============================================================================

macro_rules! region_constraint_newtype {
    ($name:ident) => {
        #[derive(Debug, Clone)]
        pub struct $name(MoleculeConstraint);

        impl From<$name> for MoleculeConstraint {
            fn from(c: $name) -> Self {
                c.0
            }
        }

        impl RegionConstraint for $name {}
    };
}

region_constraint_newtype!(InsideBoxConstraint);
region_constraint_newtype!(InsideSphereConstraint);
region_constraint_newtype!(OutsideSphereConstraint);
region_constraint_newtype!(AbovePlaneConstraint);
region_constraint_newtype!(BelowPlaneConstraint);

impl InsideBoxConstraint {
    /// Box constraint from min and max corners.
    pub fn new(min: [F; 3], max: [F; 3]) -> Self {
        Self(MoleculeConstraint::new().add(Restraint::inside_box(min, max)))
    }

    /// Cube constraint: origin + side in all axes.
    pub fn cube_from_origin(side: F, origin: [F; 3]) -> Self {
        let max = [origin[0] + side, origin[1] + side, origin[2] + side];
        Self(MoleculeConstraint::new().add(Restraint::inside_box(origin, max)))
    }
}

impl InsideSphereConstraint {
    /// Sphere constraint: radius and center.
    pub fn new(radius: F, center: [F; 3]) -> Self {
        Self(MoleculeConstraint::new().add(Restraint::inside_sphere(center, radius)))
    }
}

impl OutsideSphereConstraint {
    /// Outside-sphere constraint: radius and center.
    pub fn new(radius: F, center: [F; 3]) -> Self {
        Self(MoleculeConstraint::new().add(Restraint::outside_sphere(center, radius)))
    }
}

impl AbovePlaneConstraint {
    /// Above-plane constraint: n·x >= d.
    pub fn new(normal: [F; 3], distance: F) -> Self {
        Self(MoleculeConstraint::new().add(Restraint::above_plane(normal, distance)))
    }
}

impl BelowPlaneConstraint {
    /// Below-plane constraint: n·x <= d.
    pub fn new(normal: [F; 3], distance: F) -> Self {
        Self(MoleculeConstraint::new().add(Restraint::below_plane(normal, distance)))
    }
}

/// Per-atom constraints: applied only to the selected atom indices.
#[derive(Debug, Clone)]
pub struct AtomConstraint {
    /// Atom indices (0-based within the molecule) these restraints apply to.
    pub atom_indices: Vec<usize>,
    pub restraints: Vec<Restraint>,
}

impl AtomConstraint {
    pub fn new(atom_indices: impl IntoIterator<Item = usize>, r: Restraint) -> Self {
        Self {
            atom_indices: atom_indices.into_iter().collect(),
            restraints: vec![r],
        }
    }
}
