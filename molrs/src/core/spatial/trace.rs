//! A trace is a trajectory of points. It has no chemistry and no facing.
//!
//! [`SelfAvoidingWalk`](crate::builder::SelfAvoidingWalk) is one generator of a
//! trace. A straight segment is another. Neither generator is the trace.

use crate::types::{F, F3};

/// Ordered sample of a path. Point `i` is where a fragment anchor may sit;
/// [`tangent`](Trace::tangent) is the path direction at that sample.
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    points: Vec<F3>,
}

impl Trace {
    /// A trace of the given points, in order. An empty list is a valid trace.
    pub fn from_points(points: Vec<F3>) -> Self {
        Self { points }
    }

    /// A trace from fixed-size samples.
    pub fn from_arrays(points: Vec<[F; 3]>) -> Self {
        use ndarray::Array1;
        Self::from_points(
            points
                .into_iter()
                .map(|p| Array1::from_vec(p.to_vec()))
                .collect(),
        )
    }

    /// The samples, in order.
    pub fn points(&self) -> &[F3] {
        &self.points
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// No samples.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Sample `index` as a fixed array.
    ///
    /// # Errors
    ///
    /// Returns `None` when `index` is past the end.
    pub fn point(&self, index: usize) -> Option<[F; 3]> {
        self.points.get(index).map(|p| [p[0], p[1], p[2]])
    }

    /// Unit direction of the path at `index`.
    ///
    /// Interior and leading samples use the outgoing segment
    /// `point[i + 1] - point[i]`. The last sample repeats the incoming
    /// segment. A trace shorter than two points has no direction.
    pub fn tangent(&self, index: usize) -> Option<[F; 3]> {
        let n = self.points.len();
        if index >= n || n < 2 {
            return None;
        }
        let (a, b) = if index + 1 < n {
            (index, index + 1)
        } else {
            (index - 1, index)
        };
        let d = [
            self.points[b][0] - self.points[a][0],
            self.points[b][1] - self.points[a][1],
            self.points[b][2] - self.points[a][2],
        ];
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        (norm > 1e-15).then_some([d[0] / norm, d[1] / norm, d[2] / norm])
    }
}
