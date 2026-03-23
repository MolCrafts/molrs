//! Small 3D geometry helpers used by the Gen3D pipeline.

use rand::Rng;

#[inline]
pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n < 1e-12 {
        [1.0, 0.0, 0.0]
    } else {
        scale(a, 1.0 / n)
    }
}

pub(crate) fn arbitrary_perpendicular(axis: [f64; 3]) -> [f64; 3] {
    let a = normalize(axis);
    let ref_axis = if a[0].abs() < 0.8 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize(cross(a, ref_axis))
}

pub(crate) fn rotate_about_axis(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let k = normalize(axis);
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let kdotv = dot(k, v);
    let kxv = cross(k, v);
    [
        v[0] * cos_a + kxv[0] * sin_a + k[0] * kdotv * (1.0 - cos_a),
        v[1] * cos_a + kxv[1] * sin_a + k[1] * kdotv * (1.0 - cos_a),
        v[2] * cos_a + kxv[2] * sin_a + k[2] * kdotv * (1.0 - cos_a),
    ]
}

pub(crate) fn random_unit(rng: &mut impl Rng) -> [f64; 3] {
    loop {
        let x = rng.random_range(-1.0..1.0);
        let y = rng.random_range(-1.0..1.0);
        let z = rng.random_range(-1.0..1.0);
        let v = [x, y, z];
        let n2 = dot(v, v);
        if (1e-8..=1.0).contains(&n2) {
            return scale(v, 1.0 / n2.sqrt());
        }
    }
}
