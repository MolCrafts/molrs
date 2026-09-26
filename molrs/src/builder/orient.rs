//! Orienters: complete the rotation a superposition fit leaves undetermined.
//!
//! A superposition [`Fit`] (see [`crate::op::superpose`]) can leave the
//! rotation partly open: two beads fix a line but not the twist about it, and
//! one bead fixes no rotation at all. A [`Fit`] whose [`Freedom`] is not
//! `Unique` is therefore optimal along a whole family of rotations about
//! [`Fit::center`]: a spin `Rot(axis, θ)` (the rotation by angle `θ` about
//! the unit `axis`) for [`Freedom::Spin`], any rotation for
//! [`Freedom::Free`]. An [`Orienter`] picks one member `Q` of that family and
//! returns `compose(about(Q, fit.center), fit.rigid)` (first the fit's own
//! motion, then `Q` about the centre), so the fit's RMSD (root-mean-square
//! deviation) is kept.
//!
//! | Orienter | Choice of `Q` |
//! |----------|---------------|
//! | [`NullOrienter`] | none: `fit.rigid` itself |
//! | [`RandomOrienter`] | a uniform spin angle, or a Haar-uniform rotation (every orientation equally likely), a pure function of `(seed, unit)` |
//! | [`HintOrienter`] | turns a body axis ([`BodyAxis`]) onto the unit's direction hint |
//!
//! # References
//!
//! - Uniform spin and rotation: Shoemake, "Uniform random rotations",
//!   *Graphics Gems III* (1992), doi:10.1016/B978-0-08-050755-2.50036-1, via
//!   [`crate::op::so3`].
//! - Hint turn about a unit axis `u`: with `a⊥ = a − (a·u)u` and
//!   `h⊥ = h − (h·u)u`, `θ = atan2(u·(a⊥ × h⊥), a⊥·h⊥)`.

use crate::op::linalg::eigh_sym_3x3;
use crate::op::rigid::{Rigid, about, alignment, apply, axis_angle, compose};
use crate::op::so3::{random_angles, random_rotations};
use crate::op::superpose::{Fit, Freedom, centroid};
use crate::op::types::{F, Mat3, Vec3};
use crate::op::vec3::{MIN_DIRECTION_LENGTH, cross, dot, norm, normalize, scale, sub};
use crate::store::keys;
use crate::system::fragment::Fragment;

/// Relative gap `(λ₁ − λ₂)/λ₁` of the gyration tensor below which the top
/// eigenvalue is degenerate and the fragment has no long axis.
const PRINCIPAL_GAP_TOL: F = 1e-12;

/// Completes a non-`Unique` [`Fit`] with one member of its optimal family.
///
/// Called only for a fit whose [`Freedom`] is `Spin` or `Free`. An
/// implementor returns `compose(about(Q, fit.center), fit.rigid)` with `Q`
/// a rotation from the fit's family.
pub trait Orienter: Send + Sync {
    /// Orient unit `unit`, an instance of `fragment`, given its `fit` and its
    /// optional direction `hint`.
    ///
    /// # Errors
    ///
    /// An [`OrientError`] naming `unit` when this orienter cannot choose.
    fn orient(
        &self,
        unit: usize,
        fragment: &Fragment,
        fit: &Fit,
        hint: Option<Vec3>,
    ) -> Result<Rigid, OrientError>;

    /// [`orient`](Self::orient) for each `units[n]` with `fits[n]` and
    /// `hints[n]`, in order. The default loops over `orient`; on success an
    /// override must return values bitwise equal to per-unit `orient`.
    ///
    /// # Errors
    ///
    /// [`OrientError::Other`] when `units`, `fits` and `hints` differ in
    /// length; otherwise the first [`OrientError`] of the batch.
    fn orient_many(
        &self,
        units: &[usize],
        fragment: &Fragment,
        fits: &[Fit],
        hints: &[Option<Vec3>],
    ) -> Result<Vec<Rigid>, OrientError> {
        OrientError::check_batch(units, fits, hints)?;
        units
            .iter()
            .zip(fits)
            .zip(hints)
            .map(|((&u, f), &h)| self.orient(u, fragment, f, h))
            .collect()
    }
}

/// Why an [`Orienter`] could not complete a fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrientError {
    /// The orienter needs a direction hint and unit `unit` has none, or has
    /// a hint that is not a direction (too short to normalise).
    MissingHint {
        /// The unit being oriented.
        unit: usize,
    },
    /// The fragment has no body axis for unit `unit`.
    ///
    /// Orienters assume a fragment the placer has already validated (every
    /// atom with a `mass` and x/y/z, every bead of positive finite mass), so
    /// the causes are the axis's own: a zero dipole, an atom without
    /// `charge` under [`BodyAxis::Dipole`], or a degenerate top gyration
    /// eigenvalue under [`BodyAxis::Principal`]. An unvalidated fragment
    /// whose atoms lack `mass` or coordinates, or whose total mass is not
    /// positive finite, is reported here too.
    Unorientable {
        /// The unit being oriented.
        unit: usize,
    },
    /// A failure reported by an implementor outside this crate.
    Other(String),
}

impl std::fmt::Display for OrientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingHint { unit } => write!(f, "unit {unit} has no direction hint"),
            Self::Unorientable { unit } => {
                write!(f, "unit {unit}: the fragment has no body axis to orient")
            }
            Self::Other(msg) => write!(f, "orienter failed: {msg}"),
        }
    }
}

impl std::error::Error for OrientError {}

impl OrientError {
    /// `Other` when a batch's `units`, `fits` and `hints` differ in length.
    fn check_batch(units: &[usize], fits: &[Fit], hints: &[Option<Vec3>]) -> Result<(), Self> {
        if units.len() == fits.len() && units.len() == hints.len() {
            return Ok(());
        }
        Err(Self::Other(format!(
            "batch of {} units has {} fits and {} hints",
            units.len(),
            fits.len(),
            hints.len()
        )))
    }
}

/// Leaves every fit as `superpose` returned it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NullOrienter;

impl Orienter for NullOrienter {
    fn orient(
        &self,
        _unit: usize,
        _fragment: &Fragment,
        fit: &Fit,
        _hint: Option<Vec3>,
    ) -> Result<Rigid, OrientError> {
        Ok(fit.rigid)
    }
}

/// A uniform member of the fit's family, drawn from stream `seed` at index
/// `unit`: a uniform angle in `[0, 2π)` about the spin axis, or a Haar-random
/// rotation (uniform over all orientations; Shoemake 1992,
/// doi:10.1016/B978-0-08-050755-2.50036-1) for a free fit. A `Unique` fit is
/// returned unchanged.
///
/// The result is a pure function of `(seed, unit)`, independent of batch
/// order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomOrienter {
    seed: u64,
}

impl RandomOrienter {
    /// An orienter drawing from the stream `seed`.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl Orienter for RandomOrienter {
    fn orient(
        &self,
        unit: usize,
        _fragment: &Fragment,
        fit: &Fit,
        _hint: Option<Vec3>,
    ) -> Result<Rigid, OrientError> {
        let index = [unit as u64];
        let turn = match fit.freedom {
            Freedom::Unique => return Ok(fit.rigid),
            Freedom::Spin { axis } => axis_angle(axis, random_angles(self.seed, &index)[0])
                .ok_or(OrientError::Unorientable { unit })?,
            Freedom::Free => random_rotations(self.seed, &index)[0],
        };
        Ok(compose(&about(turn, fit.center), &fit.rigid))
    }
}

/// The body axis a [`HintOrienter`] turns onto the hint. Both are computed
/// about the mass-weighted centroid `c` of the fragment's atoms (positions
/// `rᵢ` in Å, masses `mᵢ` in g/mol, charges `qᵢ` in e); only the axis
/// direction is used, so its units drop out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyAxis {
    /// The long axis: top eigenvector of the gyration tensor
    /// `G = Σ mᵢ (rᵢ−c)(rᵢ−c)ᵀ / Σ mᵢ` (Å², the mass-weighted spread of the
    /// atoms along each direction), its largest-magnitude component made
    /// positive; among components of equal magnitude the lowest index (x,
    /// then y, then z) decides, so the sign does not depend on the
    /// eigensolver's.
    Principal,
    /// The dipole `μ = Σ qᵢ (rᵢ − c)`, normalised.
    Dipole,
}

impl BodyAxis {
    /// `v` with its largest-magnitude component made positive; among
    /// components of equal magnitude the lowest index decides.
    fn signed(v: Vec3) -> Vec3 {
        // Only a strictly larger magnitude displaces the incumbent, so a tie
        // keeps the lower index.
        let largest = v[1..]
            .iter()
            .fold(v[0], |best, &x| if x.abs() > best.abs() { x } else { best });
        if largest < 0.0 { scale(v, -1.0) } else { v }
    }
}

/// Turns the fragment's [`BodyAxis`] onto the unit's direction hint.
///
/// For a `Spin` fit about unit axis `u`, the spin angle is
/// `θ = atan2(u·(a⊥ × h⊥), a⊥·h⊥)` with `a` the body axis in the fitted
/// frame and `⊥` the projection off `u`; `θ = 0` when either projection is
/// shorter than [`MIN_DIRECTION_LENGTH`]. For a `Free` fit the body axis is
/// turned onto the hint by [`alignment`] about `fit.center` (no turn when they
/// are already parallel). A `Unique` fit is returned unchanged.
///
/// Refuses with [`OrientError::MissingHint`] when a non-`Unique` unit has no
/// hint (or one too short to normalise), and with
/// [`OrientError::Unorientable`] when the fragment has no body axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HintOrienter {
    axis: BodyAxis,
}

impl HintOrienter {
    /// An orienter aligning `axis` with each unit's hint.
    pub fn new(axis: BodyAxis) -> Self {
        Self { axis }
    }

    /// The unit body axis of `fragment` in its own frame, or `None` when it
    /// has none.
    fn body_axis(&self, fragment: &Fragment) -> Option<Vec3> {
        let mut points: Vec<Vec3> = Vec::with_capacity(fragment.n_atoms());
        let mut masses: Vec<F> = Vec::with_capacity(fragment.n_atoms());
        let mut charges: Vec<Option<F>> = Vec::with_capacity(fragment.n_atoms());
        for (_, atom) in fragment.nodes() {
            points.push(atom.position()?);
            masses.push(atom.get_f64(keys::MASS)?);
            charges.push(atom.get_f64(keys::CHARGE));
        }
        let c = centroid(&points, &masses)?;
        match self.axis {
            BodyAxis::Principal => {
                let total: F = masses.iter().sum();
                let mut g: Mat3 = [[0.0; 3]; 3];
                for (&p, &m) in points.iter().zip(&masses) {
                    let d = sub(p, c);
                    for (row, &da) in g.iter_mut().zip(&d) {
                        for (x, &db) in row.iter_mut().zip(&d) {
                            *x += m * da * db / total;
                        }
                    }
                }
                let (lambda, vecs) = eigh_sym_3x3(&g);
                if !(lambda[0] > 0.0 && (lambda[0] - lambda[1]) / lambda[0] >= PRINCIPAL_GAP_TOL) {
                    return None;
                }
                normalize(BodyAxis::signed([vecs[0][0], vecs[1][0], vecs[2][0]]))
            }
            BodyAxis::Dipole => {
                let mut mu = [0.0; 3];
                for (&p, q) in points.iter().zip(charges) {
                    let d = scale(sub(p, c), q?);
                    for (m, di) in mu.iter_mut().zip(d) {
                        *m += di;
                    }
                }
                normalize(mu)
            }
        }
    }
}

impl HintOrienter {
    /// [`Orienter::orient`] with the fragment's body axis `body` already
    /// computed (`None` when it has none).
    fn orient_with(
        unit: usize,
        body: Option<Vec3>,
        fit: &Fit,
        hint: Option<Vec3>,
    ) -> Result<Rigid, OrientError> {
        let spin_axis = match fit.freedom {
            Freedom::Unique => return Ok(fit.rigid),
            Freedom::Spin { axis } => Some(axis),
            Freedom::Free => None,
        };
        let h = hint
            .and_then(normalize)
            .ok_or(OrientError::MissingHint { unit })?;
        let body = body.ok_or(OrientError::Unorientable { unit })?;
        // The body axis in the fitted frame: rotated by the fit, not moved.
        let a = apply(
            &Rigid {
                rotation: fit.rigid.rotation,
                translation: [0.0; 3],
            },
            body,
        );
        let turn = match spin_axis {
            Some(u) => {
                let a_perp = sub(a, scale(u, dot(a, u)));
                let h_perp = sub(h, scale(u, dot(h, u)));
                if norm(a_perp) < MIN_DIRECTION_LENGTH || norm(h_perp) < MIN_DIRECTION_LENGTH {
                    return Ok(fit.rigid);
                }
                let theta = dot(u, cross(a_perp, h_perp)).atan2(dot(a_perp, h_perp));
                axis_angle(u, theta).ok_or(OrientError::Unorientable { unit })?
            }
            None => match alignment(a, h) {
                Some((axis, angle)) => {
                    axis_angle(axis, angle).ok_or(OrientError::Unorientable { unit })?
                }
                // Already parallel: no turn.
                None => return Ok(fit.rigid),
            },
        };
        Ok(compose(&about(turn, fit.center), &fit.rigid))
    }
}

impl Orienter for HintOrienter {
    fn orient(
        &self,
        unit: usize,
        fragment: &Fragment,
        fit: &Fit,
        hint: Option<Vec3>,
    ) -> Result<Rigid, OrientError> {
        Self::orient_with(unit, self.body_axis(fragment), fit, hint)
    }

    /// Computes the fragment's body axis once for the whole batch; each
    /// result is bitwise the per-unit [`orient`](Orienter::orient).
    fn orient_many(
        &self,
        units: &[usize],
        fragment: &Fragment,
        fits: &[Fit],
        hints: &[Option<Vec3>],
    ) -> Result<Vec<Rigid>, OrientError> {
        OrientError::check_batch(units, fits, hints)?;
        let body = self.body_axis(fragment);
        units
            .iter()
            .zip(fits)
            .zip(hints)
            .map(|((&u, f), &h)| Self::orient_with(u, body, f, h))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{BodyAxis, HintOrienter, NullOrienter, OrientError, Orienter, RandomOrienter};
    use crate::op::rigid::{Rigid, apply, axis_angle};
    use crate::op::superpose::{Fit, Freedom};
    use crate::op::types::{F, Vec3};
    use crate::op::vec3::{dot, sub};
    use crate::store::keys;
    use crate::system::fragment::Fragment;

    // Every expected value below is hand-derived from the fixtures following
    // `.claude/specs/assembly-05-place.md` § Domain basis and § Design 4, and
    // acceptance ac-008. No external program produced any value here. Every
    // `Fit` is hand-built, not produced by `superpose`.

    /// Exact-arithmetic position tolerance (spec: goldens at 1e-12).
    const TOL: F = 1e-12;
    const SEED: u64 = 20260926;

    const X: Vec3 = [1.0, 0.0, 0.0];
    const Y: Vec3 = [0.0, 1.0, 0.0];
    const Z: Vec3 = [0.0, 0.0, 1.0];

    // -- fixtures -------------------------------------------------------------

    /// Atoms of mass 1 at `points`, bead 0, no charge.
    fn body(points: &[Vec3]) -> Fragment {
        let mut frag = Fragment::new();
        for &[x, y, z] in points {
            let id = frag.add_atom_xyz("C", x, y, z);
            frag.set_node(id, keys::BEAD, 0_i32).expect("stamp bead");
            frag.set_node(id, keys::MASS, 1.0).expect("stamp mass");
        }
        frag
    }

    /// Atoms of mass 1 at `atoms[i].0` carrying charge `atoms[i].1`.
    fn charged(atoms: &[(Vec3, F)]) -> Fragment {
        let mut frag = Fragment::new();
        for &([x, y, z], q) in atoms {
            let id = frag.add_atom_xyz("C", x, y, z);
            frag.set_node(id, keys::BEAD, 0_i32).expect("stamp bead");
            frag.set_node(id, keys::MASS, 1.0).expect("stamp mass");
            frag.set_node(id, keys::CHARGE, q).expect("stamp charge");
        }
        frag
    }

    fn fit(rigid: Rigid, rmsd: F, center: Vec3, freedom: Freedom) -> Fit {
        Fit {
            rigid,
            rmsd,
            rho: 0.0,
            center,
            freedom,
        }
    }

    /// The identity fit of a spin about ẑ through the origin.
    fn spin_z() -> Fit {
        fit(Rigid::IDENTITY, 0.0, [0.0; 3], Freedom::Spin { axis: Z })
    }

    /// A spin fit whose rigid is Rz(90°) then +(5,0,0). It maps the
    /// references (∓1,0,0) to (5,∓1,0), onto the targets (5,∓2,0) with
    /// RMSD √((1 + 1)/2) = 1; the spin axis is ŷ through the target centroid
    /// (5,0,0), the line both images lie on.
    fn spin_y_through_5() -> (Fit, [Vec3; 2], [Vec3; 2]) {
        let rigid = Rigid {
            rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [5.0, 0.0, 0.0],
        };
        let refs = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let targets = [[5.0, -2.0, 0.0], [5.0, 2.0, 0.0]];
        (
            fit(rigid, 1.0, [5.0, 0.0, 0.0], Freedom::Spin { axis: Y }),
            refs,
            targets,
        )
    }

    /// Unweighted RMSD of `rigid` applied to `refs` against `targets`.
    fn rmsd(rigid: &Rigid, refs: &[Vec3], targets: &[Vec3]) -> F {
        let sum: F = refs
            .iter()
            .zip(targets)
            .map(|(&r, &y)| {
                let d = sub(apply(rigid, r), y);
                dot(d, d)
            })
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let n = refs.len() as F;
        (sum / n).sqrt()
    }

    fn assert_vec_close(got: Vec3, want: Vec3, tol: F, what: &str) {
        for k in 0..3 {
            assert!(
                (got[k] - want[k]).abs() <= tol,
                "{what}[{k}]: got {got:?}, want {want:?} (tol {tol})"
            );
        }
    }

    fn assert_rigid_close(got: &Rigid, want: &Rigid, tol: F) {
        for r in 0..3 {
            assert_vec_close(got.rotation[r], want.rotation[r], tol, "rotation row");
        }
        assert_vec_close(got.translation, want.translation, tol, "translation");
    }

    // -- NullOrienter ---------------------------------------------------------

    #[test]
    fn null_orienter_returns_the_fit_rigid() {
        let (f, refs, _) = spin_y_through_5();
        let frag = body(&refs);

        let got = NullOrienter
            .orient(0, &frag, &f, None)
            .expect("null orients");

        assert_eq!(got, f.rigid);
    }

    // -- RandomOrienter -------------------------------------------------------

    #[test]
    fn random_orienter_is_a_pure_function_of_seed_and_unit() {
        let (f, refs, _) = spin_y_through_5();
        let frag = body(&refs);

        let a = RandomOrienter::new(SEED).orient(5, &frag, &f, None);
        let b = RandomOrienter::new(SEED).orient(5, &frag, &f, None);
        let other = RandomOrienter::new(SEED).orient(6, &frag, &f, None);

        let a = a.expect("unit 5 orients");
        assert_eq!(a, b.expect("unit 5 orients again"));
        assert_ne!(a, other.expect("unit 6 orients"), "units 5 and 6 coincide");
    }

    /// orient(5) alone equals the unit-5 entry of orient_many([0, 5]).
    #[test]
    fn random_orienter_is_independent_of_batch_order() {
        let (f, refs, _) = spin_y_through_5();
        let frag = body(&refs);
        let orienter = RandomOrienter::new(SEED);

        let alone = orienter.orient(5, &frag, &f, None).expect("unit 5 orients");
        let batch = orienter
            .orient_many(&[0, 5], &frag, &[f, f], &[None, None])
            .expect("batch orients");

        assert_eq!(batch.len(), 2);
        assert_eq!(batch[1], alone);
    }

    /// A random turn about the spin axis through `fit.center` keeps the
    /// points the fit put on that axis where they were, and so leaves the
    /// fit's RMSD (1) unchanged.
    #[test]
    fn random_orienter_spins_about_the_fit_axis() {
        let (f, refs, targets) = spin_y_through_5();
        let frag = body(&refs);

        let got = RandomOrienter::new(SEED)
            .orient(3, &frag, &f, None)
            .expect("spin orients");

        for &r in &refs {
            assert_vec_close(apply(&got, r), apply(&f.rigid, r), TOL, "on-axis image");
        }
        let got_rmsd = rmsd(&got, &refs, &targets);
        assert!(
            (got_rmsd - f.rmsd).abs() <= TOL,
            "rmsd {got_rmsd} != fit rmsd {}",
            f.rmsd
        );
    }

    /// A `Free` fit (R = I, t = (1,2,3), center (1,2,3)) is turned about its
    /// center: the body point the fit sends to the center stays there.
    #[test]
    fn random_orienter_turns_a_free_fit_about_its_center() {
        let frag = body(&[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let f = fit(
            Rigid {
                rotation: Rigid::IDENTITY.rotation,
                translation: [1.0, 2.0, 3.0],
            },
            0.0,
            [1.0, 2.0, 3.0],
            Freedom::Free,
        );

        let got = RandomOrienter::new(SEED)
            .orient(2, &frag, &f, None)
            .expect("free orients");

        assert_vec_close(apply(&got, [0.0; 3]), [1.0, 2.0, 3.0], TOL, "center");
    }

    // -- HintOrienter ---------------------------------------------------------

    /// Domain golden: atoms (±1,0,0), spin about ẑ, hint ŷ → θ = +π/2 and
    /// (1,0,0) → (0,1,0).
    #[test]
    fn hint_orienter_turns_the_long_axis_onto_the_hint_by_a_quarter_turn() {
        let frag = body(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);

        let got = HintOrienter::new(BodyAxis::Principal)
            .orient(0, &frag, &spin_z(), Some(Y))
            .expect("principal orients");

        assert_vec_close(
            apply(&got, [1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            TOL,
            "(1,0,0)",
        );
        assert_vec_close(
            apply(&got, [-1.0, 0.0, 0.0]),
            [0.0, -1.0, 0.0],
            TOL,
            "(-1,0,0)",
        );
    }

    /// Charges −1 at (1,0,0) and +1 at (−1,0,0): μ = (−2,0,0), so a = −x̂ and
    /// θ = atan2(ẑ·(−x̂ × ŷ), 0) = −π/2; (1,0,0) → (0,−1,0). The principal
    /// axis of the same atoms would give +π/2 — the dipole keeps its sign.
    #[test]
    fn hint_orienter_turns_the_dipole_onto_the_hint() {
        let frag = charged(&[([1.0, 0.0, 0.0], -1.0), ([-1.0, 0.0, 0.0], 1.0)]);

        let got = HintOrienter::new(BodyAxis::Dipole)
            .orient(0, &frag, &spin_z(), Some(Y))
            .expect("dipole orients");

        assert_vec_close(
            apply(&got, [1.0, 0.0, 0.0]),
            [0.0, -1.0, 0.0],
            TOL,
            "(1,0,0)",
        );
    }

    /// Atoms along ẑ with a spin about ẑ: the projected body axis is shorter
    /// than `MIN_DIRECTION_LENGTH`, so θ = 0 and the fit is returned.
    #[test]
    fn hint_orienter_leaves_the_fit_when_the_axis_lies_on_the_spin_axis() {
        let frag = body(&[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]);
        let f = spin_z();

        let got = HintOrienter::new(BodyAxis::Principal)
            .orient(0, &frag, &f, Some(Y))
            .expect("on-axis orients");

        assert_rigid_close(&got, &f.rigid, TOL);
    }

    /// A body elongated along x (G = diag(2, 0.125, 0)) under a `Free` fit
    /// (R = I, t = (1,2,3), center (1,2,3)) with hint ẑ: the long axis is
    /// turned onto ẑ about the center, so (±2,0,0) → (1,2,3) ± 2ẑ.
    #[test]
    fn hint_orienter_aligns_the_principal_axis_on_a_free_fit() {
        let frag = body(&[
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0],
        ]);
        let f = fit(
            Rigid {
                rotation: Rigid::IDENTITY.rotation,
                translation: [1.0, 2.0, 3.0],
            },
            0.0,
            [1.0, 2.0, 3.0],
            Freedom::Free,
        );

        let got = HintOrienter::new(BodyAxis::Principal)
            .orient(0, &frag, &f, Some(Z))
            .expect("free principal orients");

        assert_vec_close(
            apply(&got, [2.0, 0.0, 0.0]),
            [1.0, 2.0, 5.0],
            TOL,
            "(2,0,0)",
        );
        assert_vec_close(
            apply(&got, [-2.0, 0.0, 0.0]),
            [1.0, 2.0, 1.0],
            TOL,
            "(-2,0,0)",
        );
    }

    #[test]
    fn hint_orienter_refuses_a_missing_hint() {
        let frag = body(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);

        let err = HintOrienter::new(BodyAxis::Principal)
            .orient(3, &frag, &spin_z(), None)
            .err();

        assert!(
            matches!(err, Some(OrientError::MissingHint { unit: 3 })),
            "got {err:?}"
        );
    }

    /// Equal charges at (±1,0,0): μ = 0.
    #[test]
    fn hint_orienter_refuses_a_zero_dipole() {
        let frag = charged(&[([1.0, 0.0, 0.0], 1.0), ([-1.0, 0.0, 0.0], 1.0)]);

        let err = HintOrienter::new(BodyAxis::Dipole)
            .orient(4, &frag, &spin_z(), Some(Y))
            .err();

        assert!(
            matches!(err, Some(OrientError::Unorientable { unit: 4 })),
            "got {err:?}"
        );
    }

    #[test]
    fn hint_orienter_refuses_a_dipole_without_charges() {
        let frag = body(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);

        let err = HintOrienter::new(BodyAxis::Dipole)
            .orient(4, &frag, &spin_z(), Some(Y))
            .err();

        assert!(
            matches!(err, Some(OrientError::Unorientable { unit: 4 })),
            "got {err:?}"
        );
    }

    /// Atoms at (±1,0,0), (0,±1,0): G = diag(½, ½, 0), a degenerate top
    /// eigenvalue, so no long axis.
    #[test]
    fn hint_orienter_refuses_a_degenerate_principal_axis() {
        let frag = body(&[
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]);

        let err = HintOrienter::new(BodyAxis::Principal)
            .orient(1, &frag, &spin_z(), Some(Y))
            .err();

        assert!(
            matches!(err, Some(OrientError::Unorientable { unit: 1 })),
            "got {err:?}"
        );
    }

    /// (−½, ½, 0): x and y tie in magnitude, the lowest index (x = −½)
    /// decides, so the vector is negated to (½, −½, 0). A last-wins rule
    /// would keep it.
    #[test]
    fn principal_sign_tie_goes_to_the_lowest_index() {
        assert_eq!(BodyAxis::signed([-0.5, 0.5, 0.0]), [0.5, -0.5, 0.0]);
    }

    // -- default orient_many --------------------------------------------------

    /// Implements only `orient`, so `orient_many` is the trait default:
    /// turn by `unit` radians about ẑ, then shift by `unit` along x.
    struct UnitTurn;

    impl Orienter for UnitTurn {
        fn orient(
            &self,
            unit: usize,
            _fragment: &Fragment,
            _fit: &Fit,
            _hint: Option<Vec3>,
        ) -> Result<Rigid, OrientError> {
            #[allow(clippy::cast_precision_loss)]
            let u = unit as F;
            Ok(Rigid {
                rotation: axis_angle(Z, u).expect("ẑ is a direction"),
                translation: [u, 0.0, 0.0],
            })
        }
    }

    #[test]
    fn default_orient_many_equals_per_unit_orient() {
        let frag = body(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        let f = spin_z();

        let batch = UnitTurn
            .orient_many(&[3, 1, 2], &frag, &[f, f, f], &[None, None, None])
            .expect("batch orients");
        let single: Vec<Rigid> = [3, 1, 2]
            .iter()
            .map(|&u| UnitTurn.orient(u, &frag, &f, None).expect("unit orients"))
            .collect();

        assert_eq!(batch, single);
    }

    /// Three units but two fits: the default batch is refused, not
    /// truncated to the shorter slice.
    #[test]
    fn default_orient_many_refuses_mismatched_lengths() {
        let frag = body(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        let f = spin_z();

        let err = UnitTurn
            .orient_many(&[0, 1, 2], &frag, &[f, f], &[None, None, None])
            .err();

        assert!(matches!(err, Some(OrientError::Other(_))), "got {err:?}");
    }

    /// Three units but two hints: the `HintOrienter` batch is refused too.
    #[test]
    fn hint_orienter_orient_many_refuses_mismatched_lengths() {
        let frag = body(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        let f = spin_z();

        let err = HintOrienter::new(BodyAxis::Principal)
            .orient_many(&[0, 1, 2], &frag, &[f, f, f], &[Some(Y), Some(Y)])
            .err();

        assert!(matches!(err, Some(OrientError::Other(_))), "got {err:?}");
    }

    /// Per-unit hints are paired with their own unit in the batch.
    #[test]
    fn hint_orienter_orient_many_equals_per_unit_orient() {
        let frag = body(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        let f = spin_z();
        let hints = [Some(Y), Some([0.0, -1.0, 0.0]), Some(X)];
        let orienter = HintOrienter::new(BodyAxis::Principal);

        let batch = orienter
            .orient_many(&[0, 1, 2], &frag, &[f, f, f], &hints)
            .expect("batch orients");
        let single: Vec<Rigid> = (0..3)
            .map(|u| {
                orienter
                    .orient(u, &frag, &f, hints[u])
                    .expect("unit orients")
            })
            .collect();

        assert_eq!(batch, single);
    }
}
