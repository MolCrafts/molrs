//! A trace is an ordered path of 3D points grouped into consecutive units. It
//! has no chemistry: it says *where* each unit of a structure should go (for
//! example, the bead positions of a coarse-grained polymer, one group of
//! points per monomer), not *what* goes there. Points are in the caller's
//! length unit, Å throughout molrs.
//!
//! `builder::SelfAvoidingWalk` is one generator of a
//! trace (one point per unit); [`Mapping::trace`](crate::system::mapping::Mapping::trace)
//! is another (one point per source bead, grouped by unit). Neither generator
//! is the trace.

use crate::error::MolRsError;
use crate::op::types::Vec3;
use crate::op::vec3::normalize;

/// Ordered points, split into consecutive non-empty units, with an optional
/// direction hint per unit.
///
/// Unit `i` is `points[offsets[i]..offsets[i + 1]]`. A hint is a caller-computed
/// direction the placement may use to fix a free orientation; the trace never
/// derives one itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    points: Vec<Vec3>,
    offsets: Vec<usize>,
    hints: Option<Vec<Vec3>>,
}

impl Trace {
    /// One point per unit, in order. An empty list is a trace of no units.
    pub fn from_points(points: Vec<Vec3>) -> Self {
        let offsets = (0..=points.len()).collect();
        Self {
            points,
            offsets,
            hints: None,
        }
    }

    /// `points` split into units at `offsets`.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] unless `offsets` has at least one entry
    /// (`n_units + 1` of them), starts at 0, ends at `points.len()`, and
    /// strictly increases (so every unit is non-empty).
    pub fn ragged(points: Vec<Vec3>, offsets: Vec<usize>) -> Result<Self, MolRsError> {
        match (offsets.first(), offsets.last()) {
            (Some(&0), Some(&end)) if end == points.len() => {}
            _ => {
                return Err(MolRsError::validation(format!(
                    "trace offsets {offsets:?} must start at 0 and end at the point count {}",
                    points.len()
                )));
            }
        }
        if let Some(u) = offsets.windows(2).position(|w| w[0] >= w[1]) {
            return Err(MolRsError::validation(format!(
                "trace unit {u} is empty or reversed: offsets {} then {}",
                offsets[u],
                offsets[u + 1]
            )));
        }
        Ok(Self {
            points,
            offsets,
            hints: None,
        })
    }

    /// This trace with one direction hint per unit, each stored normalised to
    /// unit length (so a hint is dimensionless; only its direction matters).
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when the hint count differs from
    /// [`n_units`](Self::n_units), or a hint is not a direction: a component
    /// is non-finite or its length is not above
    /// [`MIN_DIRECTION_LENGTH`](crate::op::vec3::MIN_DIRECTION_LENGTH).
    pub fn with_hints(self, hints: Vec<Vec3>) -> Result<Self, MolRsError> {
        if hints.len() != self.n_units() {
            return Err(MolRsError::validation(format!(
                "{} hints for a trace of {} units",
                hints.len(),
                self.n_units()
            )));
        }
        let hints = hints
            .into_iter()
            .enumerate()
            .map(|(u, h)| {
                normalize(h).ok_or_else(|| {
                    MolRsError::validation(format!("hint {h:?} of unit {u} is not a direction"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            hints: Some(hints),
            ..self
        })
    }

    /// Number of units.
    pub fn n_units(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Unit `i`'s points, or `None` when `i` is not a unit index.
    pub fn unit(&self, i: usize) -> Option<&[Vec3]> {
        let (&start, &end) = (self.offsets.get(i)?, self.offsets.get(i + 1)?);
        Some(&self.points[start..end])
    }

    /// Unit `i`'s unit-length hint, or `None` when the trace has no hints or
    /// `i` is not a unit index.
    pub fn hint(&self, i: usize) -> Option<Vec3> {
        self.hints.as_ref()?.get(i).copied()
    }

    /// Every point, in order, across all units.
    pub fn points(&self) -> &[Vec3] {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    use super::Trace;
    use crate::op::types::Vec3;
    use crate::types::F;

    fn five_points() -> Vec<Vec3> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    }

    /// Units of 2 and 3 points.
    fn two_three() -> Trace {
        Trace::ragged(five_points(), vec![0, 2, 5]).expect("offsets 0,2,5 are valid")
    }

    #[test]
    fn from_points_gives_one_point_per_unit() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]];
        let t = Trace::from_points(pts.clone());
        assert_eq!(t.n_units(), 3);
        for (i, p) in pts.iter().enumerate() {
            assert_eq!(t.unit(i), Some(std::slice::from_ref(p)));
        }
        assert_eq!(t.unit(3), None);
        assert_eq!(t.points(), pts.as_slice());
    }

    #[test]
    fn ragged_splits_points_at_offsets() {
        let t = two_three();
        let pts = five_points();
        assert_eq!(t.n_units(), 2);
        assert_eq!(t.unit(0), Some(&pts[0..2]));
        assert_eq!(t.unit(1), Some(&pts[2..5]));
        assert_eq!(t.unit(2), None);
        assert_eq!(t.points(), pts.as_slice());
    }

    #[test]
    fn ragged_refuses_offsets_not_starting_at_zero() {
        assert!(Trace::ragged(five_points(), vec![1, 2, 5]).is_err());
    }

    #[test]
    fn ragged_refuses_offsets_not_ending_at_points_len() {
        assert!(Trace::ragged(five_points(), vec![0, 2, 4]).is_err());
        assert!(Trace::ragged(five_points(), vec![0, 2, 6]).is_err());
    }

    #[test]
    fn ragged_refuses_decreasing_offsets() {
        assert!(Trace::ragged(five_points(), vec![0, 3, 2, 5]).is_err());
    }

    #[test]
    fn ragged_refuses_an_empty_unit() {
        assert!(Trace::ragged(five_points(), vec![0, 2, 2, 5]).is_err());
    }

    #[test]
    fn ragged_refuses_a_wrong_offset_count() {
        // Fewer than two entries cannot delimit n_units + 1 boundaries ending at 5.
        assert!(Trace::ragged(five_points(), vec![]).is_err());
        assert!(Trace::ragged(five_points(), vec![0]).is_err());
    }

    #[test]
    fn with_hints_refuses_a_count_mismatch() {
        assert!(two_three().with_hints(vec![[1.0, 0.0, 0.0]]).is_err());
        assert!(
            two_three()
                .with_hints(vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                .is_err()
        );
    }

    #[test]
    fn with_hints_refuses_a_zero_hint() {
        assert!(
            two_three()
                .with_hints(vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                .is_err()
        );
    }

    #[test]
    fn with_hints_refuses_a_nan_hint() {
        assert!(
            two_three()
                .with_hints(vec![[F::NAN, 1.0, 0.0], [0.0, 1.0, 0.0]])
                .is_err()
        );
    }

    #[test]
    fn with_hints_keeps_one_hint_per_unit() {
        let t = two_three()
            .with_hints(vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            .expect("two finite non-zero hints for two units");
        assert_eq!(t.hint(0), Some([1.0, 0.0, 0.0]));
        assert_eq!(t.hint(1), Some([0.0, 1.0, 0.0]));
        assert_eq!(t.hint(2), None);
        assert_eq!(t.n_units(), 2);
    }

    #[test]
    fn hint_is_none_without_hints() {
        let t = two_three();
        assert_eq!(t.hint(0), None);
        assert_eq!(t.hint(1), None);
    }
}
