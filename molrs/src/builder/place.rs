//! Per-unit rigid placement: [`Placer`] and the trace-driven [`TracePlacer`].
//!
//! A [`Placer`] answers one question: where does unit `i`, an instance of a
//! given fragment, go? The answer is a rigid motion ([`Rigid`]: rotate, then
//! translate) applied to the template's own coordinates. It never sees the
//! whole graph (only the [`Assembler`](crate::builder::Assembler) does) and
//! never depends on neighbours already placed.
//!
//! [`TracePlacer`] works from a [`Trace`]: an ordered list of 3D points (Å)
//! grouped into units, typically the bead positions of a coarse-grained model
//! with one group per unit. It superposes the fragment's bead reference
//! points onto the unit's trace points. Bead `b`'s reference point is its
//! mass-weighted centroid `c_b = Σ mᵢ rᵢ / Σ mᵢ` over the bead's atoms `i`
//! (positions `rᵢ` in Å, masses `mᵢ` in g/mol from each atom's `mass`), with
//! superposition weight `M_b = Σ mᵢ` (g/mol); trace point `j` of a unit
//! targets template bead `j`. The fit is Horn's weighted superposition
//! ([`superpose`]; Horn 1987, doi:10.1364/JOSAA.4.000629; Coutsias 2004,
//! doi:10.1002/jcc.20110) at [`DEFAULT_GAP_TOL`]; a fit the points leave
//! under-determined (for example one or two beads, or beads on a line) is
//! completed by an [`Orienter`].

use crate::builder::orient::{NullOrienter, OrientError, Orienter};
use crate::op::rigid::Rigid;
use crate::op::superpose::{
    DEFAULT_GAP_TOL, Fit, Freedom, SuperposeError, centroid, superpose, superpose_many,
};
use crate::op::types::{F, Vec3};
use crate::spatial::Trace;
use crate::store::keys;
use crate::system::fragment::{BeadError, Fragment};
use crate::system::molgraph::NodeId;

/// Places one unit of a coarse-grained sequence as a rigid motion of its
/// fragment.
pub trait Placer: Send + Sync {
    /// The motion that places unit `unit`, an instance of the template whose
    /// library key is `name`, given as `fragment`.
    ///
    /// # Errors
    ///
    /// A [`PlaceError`] naming why the unit cannot be placed.
    fn place(&self, unit: usize, name: &str, fragment: &Fragment) -> Result<Rigid, PlaceError>;

    /// [`place`](Self::place) for each of `units`, in order, all instances of
    /// the same template. The default loops over `place`; on success an
    /// override must return values bitwise equal to per-unit `place`.
    ///
    /// # Errors
    ///
    /// A [`PlaceError`] of some unit of the batch; which one, when several
    /// units fail, is unspecified (the default returns the first in `units`
    /// order, an override may check the batch in another order).
    fn place_many(
        &self,
        units: &[usize],
        name: &str,
        fragment: &Fragment,
    ) -> Result<Vec<Rigid>, PlaceError> {
        units
            .iter()
            .map(|&u| self.place(u, name, fragment))
            .collect()
    }
}

/// Why a [`Placer`] could not place a unit.
#[derive(Debug, Clone, PartialEq)]
pub enum PlaceError {
    /// `unit` is not below the trace's unit count.
    UnitOutOfRange {
        /// The requested unit.
        unit: usize,
        /// Units in the trace.
        n_units: usize,
    },
    /// The sequence and the trace disagree on the unit count.
    SeqLength {
        /// Entries in the sequence.
        seq: usize,
        /// Units in the trace.
        units: usize,
    },
    /// The template handed in is not the one the sequence names for `unit`.
    FragmentMismatch {
        /// The unit being placed.
        unit: usize,
        /// The sequence's template name for `unit`.
        expected: String,
        /// The template name handed in.
        got: String,
    },
    /// Unit `unit` has a different number of trace points than the template
    /// has beads.
    BeadCount {
        /// The unit being placed.
        unit: usize,
        /// Trace points of the unit.
        points: usize,
        /// Beads of the template.
        beads: usize,
    },
    /// Atom `node` carries no non-negative integer `bead`.
    MissingBead {
        /// The offending atom.
        node: NodeId,
    },
    /// Bead index `bead` lies below the template's bead count but no atom
    /// carries it.
    EmptyBead {
        /// The missing bead index.
        bead: usize,
    },
    /// Bead `bead`'s total mass `M_b` is not a positive finite number, so it
    /// has neither a centroid nor a superposition weight.
    BadBeadMass {
        /// The template bead index.
        bead: usize,
    },
    /// Atom `node` carries no `mass` (g/mol).
    MissingMass {
        /// The offending atom.
        node: NodeId,
    },
    /// Atom `node` lacks one of `x`, `y`, `z`.
    MissingCoordinates {
        /// The offending atom.
        node: NodeId,
    },
    /// The superposition refused its input.
    Superpose(SuperposeError),
    /// The orienter could not complete an under-determined fit.
    Orient(OrientError),
    /// A failure reported by an implementor outside this crate. Displays as
    /// the message alone; `AssembleError::Place` adds the one
    /// "placement failed" prefix.
    Other(String),
}

impl std::fmt::Display for PlaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnitOutOfRange { unit, n_units } => {
                write!(
                    f,
                    "unit {unit} is out of range for a trace of {n_units} units"
                )
            }
            Self::SeqLength { seq, units } => write!(
                f,
                "sequence has {seq} entries but the trace has {units} units"
            ),
            Self::FragmentMismatch {
                unit,
                expected,
                got,
            } => write!(
                f,
                "unit {unit} is template '{expected}' but template '{got}' was handed in"
            ),
            Self::BeadCount {
                unit,
                points,
                beads,
            } => write!(
                f,
                "unit {unit} has {points} trace points but the template has {beads} beads"
            ),
            Self::MissingBead { node } => {
                write!(f, "atom {node:?} carries no non-negative '{}'", keys::BEAD)
            }
            Self::EmptyBead { bead } => write!(f, "no atom carries bead {bead}"),
            Self::BadBeadMass { bead } => {
                write!(
                    f,
                    "bead {bead}'s total mass is not a positive finite number"
                )
            }
            Self::MissingMass { node } => write!(f, "atom {node:?} carries no '{}'", keys::MASS),
            Self::MissingCoordinates { node } => {
                write!(f, "atom {node:?} lacks x/y/z coordinates")
            }
            Self::Superpose(e) => write!(f, "superposition failed: {e}"),
            Self::Orient(e) => write!(f, "orientation failed: {e}"),
            Self::Other(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for PlaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Superpose(e) => Some(e),
            Self::Orient(e) => Some(e),
            _ => None,
        }
    }
}

impl From<SuperposeError> for PlaceError {
    fn from(e: SuperposeError) -> Self {
        Self::Superpose(e)
    }
}

impl From<OrientError> for PlaceError {
    fn from(e: OrientError) -> Self {
        Self::Orient(e)
    }
}

/// Places unit `i` by superposing the fragment's bead reference points onto
/// the trace's unit `i` points (module docs), completing a non-`Unique` fit
/// with its [`Orienter`] (default [`NullOrienter`]).
///
/// `seq[i]` is the template name of unit `i`; a unit is only placed with that
/// template. Every atom of the template must carry `bead`, `mass` (g/mol) and
/// `x`/`y`/`z` (Å); otherwise placing refuses with the matching
/// [`PlaceError`] variant.
pub struct TracePlacer {
    trace: Trace,
    seq: Vec<String>,
    orienter: Box<dyn Orienter>,
}

impl TracePlacer {
    /// A placer over `trace` whose unit `i` is an instance of template
    /// `seq[i]`.
    ///
    /// # Errors
    ///
    /// [`PlaceError::SeqLength`] when `seq.len() != trace.n_units()`.
    pub fn new(trace: Trace, seq: Vec<String>) -> Result<Self, PlaceError> {
        if seq.len() != trace.n_units() {
            return Err(PlaceError::SeqLength {
                seq: seq.len(),
                units: trace.n_units(),
            });
        }
        Ok(Self {
            trace,
            seq,
            orienter: Box::new(NullOrienter),
        })
    }

    /// Complete under-determined fits with `orienter` instead.
    pub fn with_orienter(self, orienter: Box<dyn Orienter>) -> Self {
        Self { orienter, ..self }
    }

    /// The trace the units are placed on.
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// The template name of each unit.
    pub fn seq(&self) -> &[String] {
        &self.seq
    }

    /// Unit `unit`'s trace points, once `unit` is in range and `seq` names
    /// `name` for it.
    fn unit_points(&self, unit: usize, name: &str) -> Result<&[Vec3], PlaceError> {
        let points = self.trace.unit(unit).ok_or(PlaceError::UnitOutOfRange {
            unit,
            n_units: self.trace.n_units(),
        })?;
        let expected = &self.seq[unit];
        if expected != name {
            return Err(PlaceError::FragmentMismatch {
                unit,
                expected: expected.clone(),
                got: name.to_owned(),
            });
        }
        Ok(points)
    }

    /// The bead reference points `c_b` and weights `M_b` of `fragment`, in
    /// bead order over [`Fragment::beads`].
    fn references(fragment: &Fragment) -> Result<(Vec<Vec3>, Vec<F>), PlaceError> {
        let beads = fragment.beads().map_err(|e| match e {
            BeadError::Missing { atom } | BeadError::Negative { atom, .. } => {
                PlaceError::MissingBead { node: atom }
            }
            BeadError::Empty { bead } => PlaceError::EmptyBead { bead },
        })?;
        let mut refs = Vec::with_capacity(beads.len());
        let mut weights = Vec::with_capacity(beads.len());
        for (bead, atoms) in beads.iter().enumerate() {
            let mut points = Vec::with_capacity(atoms.len());
            let mut masses = Vec::with_capacity(atoms.len());
            for &node in atoms {
                let atom = fragment
                    .get_node(node)
                    .map_err(|e| PlaceError::Other(e.to_string()))?;
                masses.push(
                    atom.get_f64(keys::MASS)
                        .ok_or(PlaceError::MissingMass { node })?,
                );
                points.push(
                    atom.position()
                        .ok_or(PlaceError::MissingCoordinates { node })?,
                );
            }
            refs.push(centroid(&points, &masses).ok_or(PlaceError::BadBeadMass { bead })?);
            weights.push(masses.iter().sum());
        }
        Ok((refs, weights))
    }
}

impl Placer for TracePlacer {
    fn place(&self, unit: usize, name: &str, fragment: &Fragment) -> Result<Rigid, PlaceError> {
        let points = self.unit_points(unit, name)?;
        let (refs, weights) = Self::references(fragment)?;
        if points.len() != refs.len() {
            return Err(PlaceError::BeadCount {
                unit,
                points: points.len(),
                beads: refs.len(),
            });
        }
        let fit = superpose(&refs, points, &weights, DEFAULT_GAP_TOL)?;
        match fit.freedom {
            Freedom::Unique => Ok(fit.rigid),
            Freedom::Spin { .. } | Freedom::Free => {
                Ok(self
                    .orienter
                    .orient(unit, fragment, &fit, self.trace.hint(unit))?)
            }
        }
    }

    /// Checks every unit's range and template, builds the reference points
    /// once, checks every unit's point count, fits all units with one
    /// [`superpose_many`] and completes all non-`Unique` fits with one
    /// [`Orienter::orient_many`]. Each result is bitwise the per-unit
    /// [`place`](Placer::place).
    fn place_many(
        &self,
        units: &[usize],
        name: &str,
        fragment: &Fragment,
    ) -> Result<Vec<Rigid>, PlaceError> {
        if units.is_empty() {
            return Ok(Vec::new());
        }
        let unit_points = units
            .iter()
            .map(|&u| self.unit_points(u, name))
            .collect::<Result<Vec<_>, _>>()?;
        let (refs, weights) = Self::references(fragment)?;
        let mut targets = Vec::with_capacity(units.len() * refs.len());
        for (&unit, points) in units.iter().zip(&unit_points) {
            if points.len() != refs.len() {
                return Err(PlaceError::BeadCount {
                    unit,
                    points: points.len(),
                    beads: refs.len(),
                });
            }
            targets.extend_from_slice(points);
        }
        let fits = superpose_many(&refs, &targets, &weights, DEFAULT_GAP_TOL)?;

        let mut placed: Vec<Rigid> = fits.iter().map(|f| f.rigid).collect();
        let open: Vec<usize> = (0..fits.len())
            .filter(|&n| fits[n].freedom != Freedom::Unique)
            .collect();
        if open.is_empty() {
            return Ok(placed);
        }
        let open_units: Vec<usize> = open.iter().map(|&n| units[n]).collect();
        let open_fits: Vec<Fit> = open.iter().map(|&n| fits[n]).collect();
        let open_hints: Vec<Option<Vec3>> =
            open_units.iter().map(|&u| self.trace.hint(u)).collect();
        let oriented = self
            .orienter
            .orient_many(&open_units, fragment, &open_fits, &open_hints)?;
        if oriented.len() != open.len() {
            return Err(PlaceError::Other(format!(
                "orienter returned {} motions for {} units",
                oriented.len(),
                open.len()
            )));
        }
        for (n, rigid) in open.into_iter().zip(oriented) {
            placed[n] = rigid;
        }
        Ok(placed)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{PlaceError, Placer, TracePlacer};
    use crate::builder::orient::{OrientError, Orienter};
    use crate::op::linalg::det3;
    use crate::op::rigid::{Rigid, about, apply, axis_angle, compose};
    use crate::op::superpose::Fit;
    use crate::op::types::{F, Mat3, Vec3};
    use crate::spatial::Trace;
    use crate::store::keys;
    use crate::system::fragment::Fragment;
    use crate::system::molgraph::NodeId;

    // Every expected value below is hand-derived from the fixtures following
    // `.claude/specs/assembly-05-place.md` § Domain basis and § Design 3, and
    // acceptance ac-004..ac-007 and ac-011. No external program produced any
    // value here.

    /// Exact-arithmetic position tolerance (spec: goldens at 1e-12).
    const TOL: F = 1e-12;

    // -- fixtures -------------------------------------------------------------

    /// One fixture atom: optional template bead, optional mass, optional
    /// coordinates. `None` leaves that property unset on the atom.
    struct Spec {
        bead: Option<i32>,
        mass: Option<F>,
        xyz: Option<Vec3>,
    }

    /// A complete atom: bead `bead`, mass `mass`, at `xyz`.
    fn atom(bead: i32, mass: F, xyz: Vec3) -> Spec {
        Spec {
            bead: Some(bead),
            mass: Some(mass),
            xyz: Some(xyz),
        }
    }

    /// Build a fragment from `specs`, returning it with the atom ids in order.
    fn fragment(specs: &[Spec]) -> (Fragment, Vec<NodeId>) {
        let mut frag = Fragment::new();
        let ids = specs
            .iter()
            .map(|s| {
                let id = match s.xyz {
                    Some([x, y, z]) => frag.add_atom_xyz("C", x, y, z),
                    None => frag.add_atom_bare("C"),
                };
                if let Some(b) = s.bead {
                    frag.set_node(id, keys::BEAD, b).expect("stamp bead");
                }
                if let Some(m) = s.mass {
                    frag.set_node(id, keys::MASS, m).expect("stamp mass");
                }
                id
            })
            .collect();
        (frag, ids)
    }

    /// One atom of mass 1 per bead, bead `j` at `points[j]`.
    fn one_atom_beads(points: &[Vec3]) -> Fragment {
        let specs: Vec<Spec> = points
            .iter()
            .enumerate()
            .map(|(j, &p)| atom(i32::try_from(j).expect("small bead index"), 1.0, p))
            .collect();
        fragment(&specs).0
    }

    fn seq(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    /// A placer over a trace of one unit holding `points`, `seq = [name]`.
    fn one_unit_placer(points: Vec<Vec3>, name: &str) -> TracePlacer {
        let n = points.len();
        let trace = Trace::ragged(points, vec![0, n]).expect("one-unit trace");
        TracePlacer::new(trace, seq(&[name])).expect("seq matches the trace")
    }

    /// Three k = 2 units pointing along distinct directions, all of length 1
    /// (the template segment below), so each fit is a `Spin`.
    fn three_segment_trace() -> Trace {
        Trace::ragged(
            vec![
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.0, 0.0, 1.0],
                [0.0, 5.0, 0.0],
                [0.6, 5.8, 0.0],
            ],
            vec![0, 2, 4, 6],
        )
        .expect("three-unit ragged trace")
    }

    fn assert_vec_close(got: Vec3, want: Vec3, tol: F, what: &str) {
        for k in 0..3 {
            assert!(
                (got[k] - want[k]).abs() <= tol,
                "{what}[{k}]: got {got:?}, want {want:?} (tol {tol})"
            );
        }
    }

    fn assert_mat_close(got: &Mat3, want: &Mat3, tol: F, what: &str) {
        for r in 0..3 {
            assert_vec_close(got[r], want[r], tol, &format!("{what} row {r}"));
        }
    }

    // -- regression example (ac-004, ac-011) ------------------------------------

    /// One-atom beads at (0,0,0), (1,0,0), (0,1,0) placed on the same triangle
    /// turned by Rz(90°): (0,0,0), (0,1,0), (−1,0,0). The fit is unique, so
    /// the placer returns R = Rz(90°) = [[0,−1,0],[1,0,0],[0,0,1]] and t = 0.
    #[test]
    fn trace_placer_recovers_a_quarter_turn_from_three_beads() {
        let frag = one_atom_beads(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let placer = one_unit_placer(
            vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            "U",
        );

        let rigid = placer.place(0, "U", &frag).expect("quarter turn places");

        assert_mat_close(
            &rigid.rotation,
            &[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            TOL,
            "rotation",
        );
        assert_vec_close(rigid.translation, [0.0, 0.0, 0.0], TOL, "translation");
    }

    // -- mass-weighted reference points (ac-005) ------------------------------

    /// A single bead of masses 1 at x = 0 and 3 at x = 4 has reference point
    /// c = (3,0,0). With k = 1 the fit is `Free` (R = I), so placing it on
    /// (10,0,0) translates by (10,0,0) − (3,0,0) = (7,0,0).
    #[test]
    fn reference_point_is_the_mass_weighted_bead_centroid() {
        let (frag, _) = fragment(&[atom(0, 1.0, [0.0, 0.0, 0.0]), atom(0, 3.0, [4.0, 0.0, 0.0])]);
        let placer = one_unit_placer(vec![[10.0, 0.0, 0.0]], "U");

        let rigid = placer.place(0, "U", &frag).expect("one bead places");

        assert_mat_close(&rigid.rotation, &Rigid::IDENTITY.rotation, TOL, "rotation");
        assert_vec_close(rigid.translation, [7.0, 0.0, 0.0], TOL, "translation");
    }

    /// Bead 0 = the bead above (c₀ = (3,0,0), M₀ = 4), bead 1 = mass 1 at
    /// (5,0,0) (c₁ = (5,0,0), M₁ = 1). Targets (0,0,0) and (4,0,0) are not
    /// congruent, so the weights matter: the fit maps the weighted reference
    /// centroid (4·3 + 5)/5 = 3.4 onto the weighted target centroid
    /// (4·0 + 4)/5 = 0.8 along x. Every optimal R fixes the x axis (the
    /// spin family is about x̂), so t = (0.8 − 3.4, 0, 0) = (−2.6, 0, 0) for
    /// every member. Unit weights would give t = (2 − 4, 0, 0) = (−2, 0, 0).
    #[test]
    fn bead_weight_is_the_bead_mass() {
        let (frag, _) = fragment(&[
            atom(0, 1.0, [0.0, 0.0, 0.0]),
            atom(0, 3.0, [4.0, 0.0, 0.0]),
            atom(1, 1.0, [5.0, 0.0, 0.0]),
        ]);
        let placer = one_unit_placer(vec![[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], "U");

        let rigid = placer.place(0, "U", &frag).expect("two beads place");

        assert_vec_close(rigid.translation, [-2.6, 0.0, 0.0], TOL, "translation");
        // x̂ stays x̂: R's first column is (1,0,0).
        assert_vec_close(
            [
                rigid.rotation[0][0],
                rigid.rotation[1][0],
                rigid.rotation[2][0],
            ],
            [1.0, 0.0, 0.0],
            TOL,
            "R x̂",
        );
    }

    // -- k = 2 with the default NullOrienter ----------------------------------

    /// Two one-atom beads (mass 1) at (0,0,0), (1,0,0) on targets (1,1,1),
    /// (1,2,1). Both segments have length 1, so the k = 2 fit is exact (a
    /// `Spin`, completed by the default `NullOrienter`): whichever member of
    /// the spin family is returned, it maps bead 0 onto (1,1,1) and bead 1
    /// onto (1,2,1), and its rotation is proper (RᵀR = I, det R = +1).
    #[test]
    fn k2_unit_with_null_orienter_maps_each_bead_onto_its_trace_point() {
        let refs = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let targets = [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]];
        let frag = one_atom_beads(&refs);
        let placer = one_unit_placer(targets.to_vec(), "U");

        let rigid = placer.place(0, "U", &frag).expect("k = 2 places");

        for (j, (&r, &y)) in refs.iter().zip(&targets).enumerate() {
            assert_vec_close(apply(&rigid, r), y, TOL, &format!("bead {j} image"));
        }
        let m = &rigid.rotation;
        let mut rtr = [[0.0; 3]; 3];
        for (a, row) in rtr.iter_mut().enumerate() {
            for (b, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| m[k][a] * m[k][b]).sum();
            }
        }
        assert_mat_close(&rtr, &Rigid::IDENTITY.rotation, TOL, "RᵀR");
        assert!((det3(m) - 1.0).abs() <= TOL, "det R = {}, want +1", det3(m));
    }

    // -- refusals (ac-006) ----------------------------------------------------

    #[test]
    fn new_refuses_a_seq_shorter_than_the_trace() {
        let trace = Trace::from_points(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);

        let err = TracePlacer::new(trace, seq(&["U"])).err();

        assert!(
            matches!(err, Some(PlaceError::SeqLength { seq: 1, units: 2 })),
            "got {err:?}"
        );
    }

    #[test]
    fn place_refuses_a_unit_out_of_range() {
        let frag = one_atom_beads(&[[0.0, 0.0, 0.0]]);
        let placer = one_unit_placer(vec![[0.0, 0.0, 0.0]], "U");

        let err = placer.place(3, "U", &frag).err();

        assert!(
            matches!(
                err,
                Some(PlaceError::UnitOutOfRange {
                    unit: 3,
                    n_units: 1
                })
            ),
            "got {err:?}"
        );
    }

    /// `seq[0] = "A"` but the caller hands in template "B".
    #[test]
    fn place_refuses_a_template_other_than_seq() {
        let frag = one_atom_beads(&[[0.0, 0.0, 0.0]]);
        let placer = one_unit_placer(vec![[0.0, 0.0, 0.0]], "A");

        let err = placer.place(0, "B", &frag).err();

        assert!(
            matches!(
                &err,
                Some(PlaceError::FragmentMismatch { unit: 0, expected, got })
                    if expected == "A" && got == "B"
            ),
            "got {err:?}"
        );
    }

    /// 3 trace points against a 2-bead template.
    #[test]
    fn place_refuses_a_point_count_other_than_the_bead_count() {
        let frag = one_atom_beads(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let placer = one_unit_placer(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], "U");

        let err = placer.place(0, "U", &frag).err();

        assert!(
            matches!(
                err,
                Some(PlaceError::BeadCount {
                    unit: 0,
                    points: 3,
                    beads: 2
                })
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn place_refuses_an_atom_without_a_bead() {
        let (frag, ids) = fragment(&[
            atom(0, 1.0, [0.0, 0.0, 0.0]),
            Spec {
                bead: None,
                mass: Some(1.0),
                xyz: Some([1.0, 0.0, 0.0]),
            },
        ]);
        let placer = one_unit_placer(vec![[0.0, 0.0, 0.0]], "U");

        let err = placer.place(0, "U", &frag).err();

        assert!(
            matches!(err, Some(PlaceError::MissingBead { node }) if node == ids[1]),
            "got {err:?}"
        );
    }

    /// Beads {0, 2}: the count is max + 1 = 3 (matching the 3 points), and
    /// bead 1 is carried by no atom.
    #[test]
    fn place_refuses_a_gap_in_the_bead_indices() {
        let (frag, _) = fragment(&[atom(0, 1.0, [0.0, 0.0, 0.0]), atom(2, 1.0, [1.0, 0.0, 0.0])]);
        let placer = one_unit_placer(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], "U");

        let err = placer.place(0, "U", &frag).err();

        assert!(
            matches!(err, Some(PlaceError::EmptyBead { bead: 1 })),
            "got {err:?}"
        );
    }

    #[test]
    fn place_refuses_an_atom_without_mass() {
        let (frag, ids) = fragment(&[
            atom(0, 1.0, [0.0, 0.0, 0.0]),
            Spec {
                bead: Some(1),
                mass: None,
                xyz: Some([1.0, 0.0, 0.0]),
            },
        ]);
        let placer = one_unit_placer(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "U");

        let err = placer.place(0, "U", &frag).err();

        assert!(
            matches!(err, Some(PlaceError::MissingMass { node }) if node == ids[1]),
            "got {err:?}"
        );
    }

    #[test]
    fn place_refuses_an_atom_without_coordinates() {
        let (frag, ids) = fragment(&[
            atom(0, 1.0, [0.0, 0.0, 0.0]),
            Spec {
                bead: Some(1),
                mass: Some(1.0),
                xyz: None,
            },
        ]);
        let placer = one_unit_placer(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "U");

        let err = placer.place(0, "U", &frag).err();

        assert!(
            matches!(err, Some(PlaceError::MissingCoordinates { node }) if node == ids[1]),
            "got {err:?}"
        );
    }

    /// Bead 0 is one atom of mass 0: it has no centroid and no weight, and
    /// the placer says so itself rather than as a superposition error.
    #[test]
    fn place_refuses_a_bead_whose_mass_is_not_positive() {
        let (frag, _) = fragment(&[atom(0, 0.0, [0.0, 0.0, 0.0]), atom(1, 1.0, [1.0, 0.0, 0.0])]);
        let placer = one_unit_placer(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "U");

        let err = placer.place(0, "U", &frag).err();

        assert!(
            matches!(err, Some(PlaceError::BadBeadMass { bead: 0 })),
            "got {err:?}"
        );
    }

    // -- batched placement (ac-007) ---------------------------------------------

    /// Deterministic local completion: after `fit.rigid`, turn by
    /// (unit + 1)/4 rad about ẑ through `fit.center`. The angle depends on the
    /// unit, so a batch that mixed up units or fits would differ from the
    /// per-unit results.
    struct UnitSpin;

    impl Orienter for UnitSpin {
        fn orient(
            &self,
            unit: usize,
            _fragment: &Fragment,
            fit: &Fit,
            _hint: Option<Vec3>,
        ) -> Result<Rigid, OrientError> {
            #[allow(clippy::cast_precision_loss)]
            let angle = (unit as F + 1.0) * 0.25;
            let turn = axis_angle([0.0, 0.0, 1.0], angle).expect("ẑ is a direction");
            Ok(compose(&about(turn, fit.center), &fit.rigid))
        }
    }

    /// Three k = 2 units, each a `Spin` fit completed by the local `UnitSpin`
    /// orienter: the batch in order [2, 0, 1] is bitwise the per-unit results.
    #[test]
    fn place_many_equals_per_unit_place_bitwise() {
        let frag = one_atom_beads(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let placer = TracePlacer::new(three_segment_trace(), seq(&["U", "U", "U"]))
            .expect("seq matches the trace")
            .with_orienter(Box::new(UnitSpin));

        let batch = placer
            .place_many(&[2, 0, 1], "U", &frag)
            .expect("batch places");
        let single: Vec<Rigid> = [2, 0, 1]
            .iter()
            .map(|&u| placer.place(u, "U", &frag).expect("unit places"))
            .collect();

        assert_eq!(batch, single);
    }

    /// Counts calls, then delegates `orient_many` to the per-unit `orient`.
    struct CountingOrienter {
        orient_calls: Arc<AtomicUsize>,
        many_calls: Arc<AtomicUsize>,
    }

    impl Orienter for CountingOrienter {
        fn orient(
            &self,
            _unit: usize,
            _fragment: &Fragment,
            fit: &Fit,
            _hint: Option<Vec3>,
        ) -> Result<Rigid, OrientError> {
            self.orient_calls.fetch_add(1, Ordering::SeqCst);
            Ok(fit.rigid)
        }

        fn orient_many(
            &self,
            units: &[usize],
            fragment: &Fragment,
            fits: &[Fit],
            hints: &[Option<Vec3>],
        ) -> Result<Vec<Rigid>, OrientError> {
            self.many_calls.fetch_add(1, Ordering::SeqCst);
            units
                .iter()
                .zip(fits)
                .zip(hints)
                .map(|((&u, f), &h)| self.orient(u, fragment, f, h))
                .collect()
        }
    }

    /// Three `Spin` units in one `place_many`: one `orient_many` call, whose
    /// delegation accounts for all three `orient` calls (none made directly).
    #[test]
    fn place_many_orients_its_non_unique_units_in_one_batch() {
        let frag = one_atom_beads(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let orient_calls = Arc::new(AtomicUsize::new(0));
        let many_calls = Arc::new(AtomicUsize::new(0));
        let placer = TracePlacer::new(three_segment_trace(), seq(&["U", "U", "U"]))
            .expect("seq matches the trace")
            .with_orienter(Box::new(CountingOrienter {
                orient_calls: Arc::clone(&orient_calls),
                many_calls: Arc::clone(&many_calls),
            }));

        let placed = placer
            .place_many(&[0, 1, 2], "U", &frag)
            .expect("batch places");

        assert_eq!(placed.len(), 3);
        assert_eq!(many_calls.load(Ordering::SeqCst), 1, "orient_many calls");
        assert_eq!(orient_calls.load(Ordering::SeqCst), 3, "orient calls");
    }
}
