//! Rigid placement of template copies: the [`Placer`] trait and the
//! translation-only [`TracePlacer`].
//!
//! A placer answers one question for the
//! [`Assembler`](crate::builder::Assembler): given a template and the points
//! its copies must sit on, which rigid motion ([`Rigid`], `p' = R p + t`)
//! moves the template onto each point? It never sees the other units, the
//! ports or the world.
//!
//! The trait has one implementor. It is kept by operator mandate
//! ("placement is the assembler's placer's job", operator, 2026-09-27);
//! the next builder spec decides whether a second placer exists or the
//! trait collapses into [`TracePlacer`].

use std::fmt;

use crate::op::rigid::Rigid;
use crate::op::types::Vec3;
use crate::spatial::geometry::CenterError;
use crate::system::fragment::Fragment;

/// Turns one template and a list of points into one rigid motion per point.
///
/// Implementors are `Send + Sync` so an
/// [`Assembler`](crate::builder::Assembler) holding one can cross threads.
pub trait Placer: Send + Sync {
    /// One [`Rigid`] per point of `points`, in order: the motion that places
    /// a copy of `template` on that point. Points are in Å.
    ///
    /// # Errors
    ///
    /// A [`PlaceError`] naming why the template or a point cannot be placed.
    fn place_many(&self, template: &Fragment, points: &[Vec3]) -> Result<Vec<Rigid>, PlaceError>;
}

/// The translation-only placer: each copy keeps the template's orientation
/// and its centre of mass lands on its point.
///
/// With the template's centre of mass `R_c = Σ mⱼ rⱼ / Σ mⱼ`
/// ([`Fragment::center`], Å) and point `p_k` (Å), copy `k` moves by `R = I`
/// and `t_k = p_k − R_c`, so its centre of mass lands on `p_k` to rounding.
/// This is the translation-only form of geometric backmapping (Wassenaar et
/// al., *J. Chem. Theory Comput.* **10**, 676–690 (2014),
/// doi:10.1021/ct400617g); orientation and overlap are left to a later
/// relaxation.
///
/// # Examples
///
/// ```
/// use molrs::builder::{Placer, TracePlacer};
/// use molrs::store::keys;
/// use molrs::system::fragment::Fragment;
///
/// // Two carbons 1.5 Å apart: the centre of mass is (0.75, 0, 0).
/// let mut template = Fragment::new();
/// for x in [0.0, 1.5] {
///     let c = template.add_atom_xyz("C", x, 0.0, 0.0);
///     template.set_node(c, keys::MASS, 12.0).unwrap();
/// }
///
/// let rigids = TracePlacer::new()
///     .place_many(&template, &[[10.0, 0.0, 0.0]])
///     .unwrap();
/// assert_eq!(rigids[0].translation, [9.25, 0.0, 0.0]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct TracePlacer;

impl TracePlacer {
    /// The placer. It takes no configuration.
    pub fn new() -> Self {
        Self
    }
}

impl Placer for TracePlacer {
    /// `Rigid { rotation: I, translation: p − R_c }` per point.
    ///
    /// No points give `Ok(vec![])` without reading the template. Otherwise
    /// the template's centre is computed once.
    ///
    /// # Errors
    ///
    /// [`PlaceError::Template`] when the template has no centre of mass
    /// ([`Fragment::center`]); [`PlaceError::NonFinitePoint`] for the first
    /// point with a NaN or infinite coordinate.
    fn place_many(&self, template: &Fragment, points: &[Vec3]) -> Result<Vec<Rigid>, PlaceError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let center = template.center().map_err(PlaceError::Template)?;
        points
            .iter()
            .enumerate()
            .map(|(index, p)| {
                if !p.iter().all(|c| c.is_finite()) {
                    return Err(PlaceError::NonFinitePoint { index });
                }
                Ok(Rigid {
                    rotation: Rigid::IDENTITY.rotation,
                    translation: [p[0] - center[0], p[1] - center[1], p[2] - center[2]],
                })
            })
            .collect()
    }
}

/// Why a [`Placer`] could not place a template.
#[derive(Debug, Clone, PartialEq)]
pub enum PlaceError {
    /// The template has no centre of mass.
    Template(CenterError),
    /// Point `index` has a NaN or infinite coordinate.
    NonFinitePoint {
        /// Index of the offending point in the slice passed.
        index: usize,
    },
}

impl fmt::Display for PlaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Template(e) => write!(f, "the template has no centre of mass: {e}"),
            Self::NonFinitePoint { index } => {
                write!(f, "point {index} has a non-finite coordinate")
            }
        }
    }
}

impl std::error::Error for PlaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Template(e) => Some(e),
            Self::NonFinitePoint { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PlaceError, Placer, TracePlacer};
    use crate::op::rigid::Rigid;
    use crate::spatial::geometry::CenterError;
    use crate::store::keys;
    use crate::system::fragment::Fragment;

    /// Monomer M: C0 (0,0,0), C1 (1.5,0,0), H0 (−1,0,0), H1 (2.5,0,0); masses
    /// 12, 12, 1, 1. Centre of mass x = (0·12 + 1.5·12 − 1·1 + 2.5·1) / 26
    /// = 19.5 / 26 = 0.75, y = z = 0. Bonds and ports play no part here.
    fn monomer(with_mass: bool) -> Fragment {
        let mut m = Fragment::new();
        for (symbol, x, mass) in [
            ("C", 0.0, 12.0),
            ("C", 1.5, 12.0),
            ("H", -1.0, 1.0),
            ("H", 2.5, 1.0),
        ] {
            let id = m.add_atom_xyz(symbol, x, 0.0, 0.0);
            if with_mass {
                m.set_node(id, keys::MASS, mass).expect("stamp mass");
            }
        }
        m
    }

    #[test]
    fn trace_placer_moves_the_centre_of_mass_onto_the_point() {
        let rigids = TracePlacer::new()
            .place_many(&monomer(true), &[[10.0, 0.0, 0.0]])
            .expect("M is placeable");

        // t = p − R_c = (10, 0, 0) − (0.75, 0, 0).
        assert_eq!(rigids.len(), 1);
        assert_eq!(rigids[0].translation, [9.25, 0.0, 0.0]);
        assert_eq!(rigids[0].rotation, Rigid::IDENTITY.rotation);
    }

    #[test]
    fn trace_placer_of_no_points_is_empty_without_reading_the_template() {
        // The massless template has no centre; no points means it is not read.
        let rigids = TracePlacer
            .place_many(&monomer(false), &[])
            .expect("no points place nothing");

        assert!(rigids.is_empty());
    }

    #[test]
    fn trace_placer_refuses_a_template_without_mass() {
        let err = TracePlacer::new()
            .place_many(&monomer(false), &[[0.0, 0.0, 0.0]])
            .expect_err("a massless template has no centre");

        assert!(
            matches!(err, PlaceError::Template(CenterError::BadMass { .. })),
            "{err:?}"
        );
    }

    #[test]
    fn trace_placer_names_the_first_non_finite_point() {
        let err = TracePlacer::new()
            .place_many(
                &monomer(true),
                &[
                    [0.0, 0.0, 0.0],
                    [f64::NAN, 0.0, 0.0],
                    [0.0, f64::INFINITY, 0.0],
                ],
            )
            .expect_err("a NaN point cannot be placed");

        assert_eq!(err, PlaceError::NonFinitePoint { index: 1 });
    }
}
