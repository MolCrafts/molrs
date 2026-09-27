//! [`Assembler`]: one placed, linked world [`Fragment`] from a list of traces
//! and their unit names.
//!
//! The assembler holds a library (name → one template [`Fragment`], with or
//! without ports) and a [`Placer`]. Its one verb,
//! [`assemble`](Assembler::assemble), builds every molecule of the input in a
//! single O(N) call: one placed template copy per trace point, consecutive
//! units of a trace joined `>` to `<`, each atom stamped with its unit
//! (`frag_id`) and its molecule (`mol_id`).
//!
//! `assemble` is a composed operation by operator ruling (notes.md
//! 2026-09-27): it supersedes the 2026-09-26 "primitives only" ruling for
//! this one concern.

use std::collections::HashMap;
use std::fmt;

use crate::builder::place::{PlaceError, Placer};
use crate::error::MolRsError;
use crate::op::types::Vec3;
use crate::spatial::Trace;
use crate::store::keys;
use crate::system::atomistic::AtomId;
use crate::system::fragment::{Fragment, PortId, PortKind};
use crate::system::link::LinkManyError;
use crate::types::I;

/// Why [`Assembler::assemble`] refused its input.
///
/// Every variant but [`Replicate`](Self::Replicate) and [`Link`](Self::Link)
/// is found before the first copy is placed. `trace` is a 0-based index into
/// the traces passed and `unit` a 0-based position within that trace.
#[derive(Debug)]
pub enum AssembleError {
    /// The traces and the name sequences differ in count.
    LengthMismatch {
        /// Traces passed.
        traces: usize,
        /// Name sequences passed.
        sequences: usize,
    },
    /// Trace `trace` and its name sequence differ in length.
    SequenceLength {
        /// The trace.
        trace: usize,
        /// Points of the trace.
        points: usize,
        /// Names of its sequence.
        names: usize,
    },
    /// The unit count, or the trace count when it is larger, exceeds
    /// [`i32::MAX`], so `frag_id` or `mol_id` would not fit its `i32` node
    /// column.
    TooManyUnits {
        /// The larger of the unit count and the trace count.
        units: usize,
    },
    /// Unit `unit` of trace `trace` names a template the library lacks.
    UnknownName {
        /// The trace.
        trace: usize,
        /// The unit within the trace.
        unit: usize,
        /// The name it carries.
        name: String,
    },
    /// Unit `unit` of trace `trace` must join a neighbour, but its template
    /// has no port of kind `kind`.
    MissingPort {
        /// The trace.
        trace: usize,
        /// The unit within the trace.
        unit: usize,
        /// The unit's template name.
        name: String,
        /// The port kind it lacks: [`PortKind::Right`] to join the next unit,
        /// [`PortKind::Left`] to join the previous one.
        kind: PortKind,
    },
    /// Template `name` has `count` ports of the kind a join needs, so the
    /// join is ambiguous.
    AmbiguousPort {
        /// The template name.
        name: String,
        /// The port kind.
        kind: PortKind,
        /// How many ports of that kind the template has (at least 2).
        count: usize,
    },
    /// A port of template `name` does not read back.
    Template {
        /// The template name.
        name: String,
        /// Why the port does not read back.
        source: MolRsError,
    },
    /// The placer refused the copies of template `name`; `(trace, unit)` is
    /// the unit it names, or the group's first unit when the error names no
    /// point.
    Place {
        /// The template name.
        name: String,
        /// The trace.
        trace: usize,
        /// The unit within the trace.
        unit: usize,
        /// The placer's refusal.
        source: PlaceError,
    },
    /// The world refused the copies of template `name` (for example a column
    /// type that contradicts another template's), or a copy lost a port.
    Replicate {
        /// The template name.
        name: String,
        /// The world's refusal.
        source: MolRsError,
    },
    /// The batch join refused a link; `(trace, unit)` is the left unit of the
    /// offending link (for a two-pair refusal, of its `first` pair).
    Link {
        /// The trace.
        trace: usize,
        /// The left unit within the trace.
        unit: usize,
        /// The batch join's refusal; its pair indices count links in
        /// trace-major order.
        source: LinkManyError,
    },
}

impl fmt::Display for AssembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { traces, sequences } => write!(
                f,
                "{traces} traces but {sequences} name sequences; they pair one to one"
            ),
            Self::SequenceLength {
                trace,
                points,
                names,
            } => write!(
                f,
                "trace {trace} has {points} points but its sequence has {names} names"
            ),
            Self::TooManyUnits { units } => write!(
                f,
                "{units} units or traces exceed {}, the widest id a node column stores",
                I::MAX
            ),
            Self::UnknownName { trace, unit, name } => write!(
                f,
                "unit {unit} of trace {trace} names '{name}', which the library lacks"
            ),
            Self::MissingPort {
                trace,
                unit,
                name,
                kind,
            } => write!(
                f,
                "unit {unit} of trace {trace} ('{name}') has no '{}' port to join its neighbour",
                kind.as_str()
            ),
            Self::AmbiguousPort { name, kind, count } => write!(
                f,
                "template '{name}' has {count} '{}' ports; a join needs exactly one",
                kind.as_str()
            ),
            Self::Template { name, source } => {
                write!(
                    f,
                    "a port of template '{name}' does not read back: {source}"
                )
            }
            Self::Place {
                name,
                trace,
                unit,
                source,
            } => write!(
                f,
                "unit {unit} of trace {trace} ('{name}') cannot be placed: {source}"
            ),
            Self::Replicate { name, source } => {
                write!(f, "the copies of template '{name}' were refused: {source}")
            }
            Self::Link {
                trace,
                unit,
                source,
            } => write!(
                f,
                "unit {unit} of trace {trace} cannot join unit {}: {source}",
                unit + 1
            ),
        }
    }
}

impl std::error::Error for AssembleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Template { source, .. } | Self::Replicate { source, .. } => Some(source),
            Self::Place { source, .. } => Some(source),
            Self::Link { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// `(anchor row, handle row)` of a template's `>` and `<` ports.
struct TemplatePorts {
    right: Vec<(usize, usize)>,
    left: Vec<(usize, usize)>,
}

/// The units that use one template: its copies, in trace-major order.
struct Group<'a> {
    name: &'a str,
    template: &'a Fragment,
    /// Global trace-major unit ordinals, one per copy.
    units: Vec<usize>,
    /// Read on the first unit that needs a join; `None` until then.
    ports: Option<TemplatePorts>,
}

/// Builds every molecule of a list of traces as one placed, linked world
/// [`Fragment`].
///
/// # Placement
///
/// Unit `k` of a trace sits on the trace's point `p_k` (Å). The placer turns
/// the unit's template into a rigid motion; with [`TracePlacer`] the copy
/// moves by `R = I` and `t = p_k − R_c`, `R_c` the template's centre of mass
/// (Å), so the copy's centre of mass lands on `p_k` and its orientation is
/// the template's own. Relax the result before use.
///
/// # Link rule
///
/// In a trace of `n ≥ 2` units, unit `i`'s one `>` port joins unit `i + 1`'s
/// one `<` port through the port rule of
/// [`Fragment::link`] (equal label and order; the leaving groups are removed
/// and their charge folds onto the anchors). All links of all traces run in
/// one batch. Only `>` / `<` ports are joined: `$` and `!` ports are never
/// linked and stay on the world, as do the chain-end ports and every port of
/// a single-point trace.
///
/// # Ids
///
/// - `frag_id` = the unit's global ordinal, counted trace-major over all
///   traces (0-based).
/// - `mol_id` = the trace's ordinal + 1: one trace is one molecule, and
///   [`to_frame`](Fragment::to_frame) emits it as the schema's `mol_id`
///   column.
///
/// # World layout
///
/// Atoms are grouped by template name, in the order names first appear, and
/// copy-major within a name, until the batch join swap-removes the leaving
/// groups and refills their rows from the end. Read a unit's atoms by
/// `frag_id`, never by row.
///
/// **Partial columns.** Templates may differ in their optional columns (one
/// carries `formal_charge`, another does not); the world then holds those
/// columns for some atoms only, and [`to_frame`](Fragment::to_frame) writes
/// 0.0 in the rows without the prop (routed `/mol:fix`, notes.md
/// 2026-09-27).
///
/// # Known limits
///
/// No orientation, overlap removal, relaxation or wrapping; only linear
/// chains (no `$` / `!` joins, no branches or rings).
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use molrs::builder::{Assembler, TracePlacer};
/// use molrs::spatial::Trace;
/// use molrs::store::keys;
/// use molrs::system::bond::BondNumber;
/// use molrs::system::fragment::{Fragment, PortKind};
///
/// // A carbon with a `<` hydrogen and a `>` hydrogen.
/// let mut unit = Fragment::new();
/// let c = unit.add_atom_xyz("C", 0.0, 0.0, 0.0);
/// let hl = unit.add_atom_xyz("H", -1.0, 0.0, 0.0);
/// let hr = unit.add_atom_xyz("H", 1.0, 0.0, 0.0);
/// for (atom, mass) in [(c, 12.0), (hl, 1.0), (hr, 1.0)] {
///     unit.set_node(atom, keys::MASS, mass).unwrap();
/// }
/// unit.add_bond(c, hl).unwrap();
/// unit.add_bond(c, hr).unwrap();
/// unit.add_port(c, hl, PortKind::Left, "", BondNumber::Single).unwrap();
/// unit.add_port(c, hr, PortKind::Right, "", BondNumber::Single).unwrap();
///
/// let assembler = Assembler::new(
///     HashMap::from([("U".to_owned(), unit)]),
///     Box::new(TracePlacer::new()),
/// );
/// let trace = Trace::from_points(vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]);
/// let world = assembler
///     .assemble(&[trace], &[vec!["U".to_owned(), "U".to_owned()]])
///     .unwrap();
///
/// // Two copies of 3 atoms, one link removes 2 hydrogens.
/// assert_eq!(world.n_atoms(), 4);
/// assert_eq!(world.n_ports(), 2);
/// ```
///
/// [`TracePlacer`]: crate::builder::TracePlacer
pub struct Assembler {
    library: HashMap<String, Fragment>,
    placer: Box<dyn Placer>,
}

impl Assembler {
    /// An assembler over `library` (name → template), placing copies with
    /// `placer`. A portless molecule is a template with no ports.
    pub fn new(library: HashMap<String, Fragment>, placer: Box<dyn Placer>) -> Self {
        Self { library, placer }
    }

    /// Place one copy of `names[t][k]`'s template on each point `k` of
    /// `traces[t]`, join each trace's consecutive units, and return the
    /// world. See the type docs for the rules.
    ///
    /// `traces = []` returns an empty [`Fragment`].
    ///
    /// # Errors
    ///
    /// Checked over the whole input before the first copy is placed, the
    /// first offender in trace-major order:
    /// [`LengthMismatch`](AssembleError::LengthMismatch),
    /// [`SequenceLength`](AssembleError::SequenceLength),
    /// [`TooManyUnits`](AssembleError::TooManyUnits),
    /// [`UnknownName`](AssembleError::UnknownName), then the port needs of
    /// every unit of an `n ≥ 2` trace:
    /// [`MissingPort`](AssembleError::MissingPort),
    /// [`AmbiguousPort`](AssembleError::AmbiguousPort) and
    /// [`Template`](AssembleError::Template). While building:
    /// [`Place`](AssembleError::Place), [`Replicate`](AssembleError::Replicate)
    /// and [`Link`](AssembleError::Link).
    pub fn assemble(
        &self,
        traces: &[Trace],
        names: &[Vec<String>],
    ) -> Result<Fragment, AssembleError> {
        // ---- checks: nothing is placed until every one passes ----
        if traces.len() != names.len() {
            return Err(AssembleError::LengthMismatch {
                traces: traces.len(),
                sequences: names.len(),
            });
        }
        for (trace, (points, seq)) in traces.iter().zip(names).enumerate() {
            if points.points().len() != seq.len() {
                return Err(AssembleError::SequenceLength {
                    trace,
                    points: points.points().len(),
                    names: seq.len(),
                });
            }
        }
        let n_units: usize = names.iter().map(Vec::len).sum();
        let widest = n_units.max(traces.len());
        if I::try_from(widest).is_err() {
            return Err(AssembleError::TooManyUnits { units: widest });
        }

        let mut group_of: HashMap<&str, usize> = HashMap::new();
        let mut groups: Vec<Group<'_>> = Vec::new();
        // By global unit ordinal: (trace, unit within trace) and (group, copy).
        let mut location: Vec<(usize, usize)> = Vec::with_capacity(n_units);
        let mut copy_of: Vec<(usize, usize)> = Vec::with_capacity(n_units);
        for (trace, seq) in names.iter().enumerate() {
            let n = seq.len();
            for (unit, name) in seq.iter().enumerate() {
                let g = match group_of.get(name.as_str()) {
                    Some(&g) => g,
                    None => {
                        let (key, template) =
                            self.library.get_key_value(name).ok_or_else(|| {
                                AssembleError::UnknownName {
                                    trace,
                                    unit,
                                    name: name.clone(),
                                }
                            })?;
                        group_of.insert(key, groups.len());
                        groups.push(Group {
                            name: key,
                            template,
                            units: Vec::new(),
                            ports: None,
                        });
                        groups.len() - 1
                    }
                };
                let group = &mut groups[g];
                for (needed, kind) in [(unit + 1 < n, PortKind::Right), (unit > 0, PortKind::Left)]
                {
                    if !needed {
                        continue;
                    }
                    if group.ports.is_none() {
                        group.ports = Some(Self::template_ports(group.name, group.template)?);
                    }
                    let ports = group.ports.as_ref().expect("read just above");
                    let count = match kind {
                        PortKind::Right => ports.right.len(),
                        _ => ports.left.len(),
                    };
                    match count {
                        1 => {}
                        0 => {
                            return Err(AssembleError::MissingPort {
                                trace,
                                unit,
                                name: name.clone(),
                                kind,
                            });
                        }
                        count => {
                            return Err(AssembleError::AmbiguousPort {
                                name: name.clone(),
                                kind,
                                count,
                            });
                        }
                    }
                }
                copy_of.push((g, group.units.len()));
                group.units.push(location.len());
                location.push((trace, unit));
            }
        }

        // ---- place, replicate and stamp mol_id: one pass per name ----
        let mut world = Fragment::new();
        let mut copies: Vec<Vec<AtomId>> = Vec::with_capacity(groups.len());
        for group in &groups {
            let points: Vec<Vec3> = group
                .units
                .iter()
                .map(|&u| {
                    let (trace, unit) = location[u];
                    traces[trace].points()[unit]
                })
                .collect();
            let rigids = self
                .placer
                .place_many(group.template, &points)
                .map_err(|source| {
                    let at = match source {
                        PlaceError::NonFinitePoint { index } => index,
                        PlaceError::Template(_) => 0,
                    };
                    let (trace, unit) =
                        location[group.units.get(at).copied().unwrap_or(group.units[0])];
                    AssembleError::Place {
                        name: group.name.to_owned(),
                        trace,
                        unit,
                        source,
                    }
                })?;
            let replicate_err = |source| AssembleError::Replicate {
                name: group.name.to_owned(),
                source,
            };
            let frag_ids: Vec<I> = group
                .units
                .iter()
                .map(|&u| I::try_from(u).expect("TooManyUnits bounds every unit ordinal"))
                .collect();
            let atoms = world
                .replicate(group.template, &rigids, &frag_ids)
                .map_err(replicate_err)?;
            let n = group.template.n_atoms();
            for (c, &u) in group.units.iter().enumerate() {
                let mol_id = I::try_from(location[u].0 + 1)
                    .expect("TooManyUnits bounds every trace ordinal");
                for &atom in &atoms[c * n..(c + 1) * n] {
                    world
                        .set_node(atom, keys::MOL_ID, mol_id)
                        .map_err(replicate_err)?;
                }
            }
            copies.push(atoms);
        }

        // ---- one scan of the world's ports ----
        let mut world_ports: HashMap<(AtomId, AtomId), PortId> =
            HashMap::with_capacity(world.n_ports());
        for id in world.ports() {
            // A port that does not read back belongs to a unit no join needs
            // (every needed template port was read at the checks); it stays.
            if let Ok(port) = world.port(id) {
                world_ports.insert((port.anchor, port.handle), id);
            }
        }
        let world_port = |u: usize, kind: PortKind| -> Result<PortId, AssembleError> {
            let (g, c) = copy_of[u];
            let group = &groups[g];
            let ports = group
                .ports
                .as_ref()
                .expect("a joined unit's ports were read");
            let (anchor, handle) = match kind {
                PortKind::Right => ports.right[0],
                _ => ports.left[0],
            };
            let base = c * group.template.n_atoms();
            let key = (copies[g][base + anchor], copies[g][base + handle]);
            world_ports
                .get(&key)
                .copied()
                .ok_or_else(|| AssembleError::Replicate {
                    name: group.name.to_owned(),
                    source: MolRsError::validation(format!(
                        "copy {c} lost its '{}' port in the world",
                        kind.as_str()
                    )),
                })
        };

        // ---- one batch join: unit i `>` to unit i + 1 `<` ----
        let mut pairs: Vec<(PortId, PortId)> = Vec::new();
        let mut left_unit: Vec<(usize, usize)> = Vec::new();
        for u in 1..location.len() {
            if location[u].0 == location[u - 1].0 {
                pairs.push((
                    world_port(u - 1, PortKind::Right)?,
                    world_port(u, PortKind::Left)?,
                ));
                left_unit.push(location[u - 1]);
            }
        }
        world.link_many(&pairs).map_err(|source| {
            let pair = match &source {
                LinkManyError::Pair { pair, .. } => *pair,
                LinkManyError::PortReused { first, .. }
                | LinkManyError::DuplicateBond { first, .. }
                | LinkManyError::BranchesOverlap { first, .. } => *first,
            };
            let (trace, unit) = left_unit[pair];
            AssembleError::Link {
                trace,
                unit,
                source,
            }
        })?;

        Ok(world)
    }

    /// `(anchor row, handle row)` of each `>` and `<` port of `template`.
    fn template_ports(name: &str, template: &Fragment) -> Result<TemplatePorts, AssembleError> {
        let refuse = |source| AssembleError::Template {
            name: name.to_owned(),
            source,
        };
        let row = |atom: AtomId| {
            template.node_table().row(atom).ok_or_else(|| {
                refuse(MolRsError::validation(format!(
                    "a port names {atom:?}, which is no live atom"
                )))
            })
        };
        let mut ports = TemplatePorts {
            right: Vec::new(),
            left: Vec::new(),
        };
        for id in template.ports() {
            let port = template.port(id).map_err(refuse)?;
            let rows = (row(port.anchor)?, row(port.handle)?);
            match port.kind {
                PortKind::Right => ports.right.push(rows),
                PortKind::Left => ports.left.push(rows),
                PortKind::Symmetric | PortKind::Shared => {}
            }
        }
        Ok(ports)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{AssembleError, Assembler};
    use crate::builder::place::{PlaceError, Placer, TracePlacer};
    use crate::op::rigid::Rigid;
    use crate::op::types::Vec3;
    use crate::spatial::Trace;
    use crate::store::keys;
    use crate::system::atomistic::AtomId;
    use crate::system::bond::BondNumber;
    use crate::system::fragment::{Fragment, PortKind};
    use crate::system::link::{LinkError, LinkManyError};

    const TOL: f64 = 1e-12;

    // ---- fixtures ----------------------------------------------------------
    //
    // Monomer M: C0 (0,0,0, m 12), C1 (1.5,0,0, m 12), H0 (−1,0,0, m 1),
    // H1 (2.5,0,0, m 1); bonds C0–C1, C0–H0, C1–H1; ports (C0, H0, `<`) and
    // (C1, H1, `>`), unlabelled, Single. Hand CoM x = 19.5 / 26 = 0.75, so a
    // copy placed on x = p has C0 at p − 0.75 and C1 at p + 0.75.
    //
    // A link removes the two handles (−2 atoms, −2 handle bonds, −2 ports)
    // and adds one C1–C0 bond. An n-unit M chain therefore has 4n − 2(n−1)
    // atoms, 3n − 2(n−1) + (n−1) bonds and 2 ports.

    /// Add one atom with a mass.
    fn atom(f: &mut Fragment, symbol: &str, x: f64, mass: f64) -> AtomId {
        let id = f.add_atom_xyz(symbol, x, 0.0, 0.0);
        f.set_node(id, keys::MASS, mass).expect("stamp mass");
        id
    }

    /// Monomer M with its `<` port labelled `left` and `>` labelled `right`.
    fn monomer_labelled(left: &str, right: &str) -> Fragment {
        let mut m = Fragment::new();
        let c0 = atom(&mut m, "C", 0.0, 12.0);
        let c1 = atom(&mut m, "C", 1.5, 12.0);
        let h0 = atom(&mut m, "H", -1.0, 1.0);
        let h1 = atom(&mut m, "H", 2.5, 1.0);
        m.add_bond(c0, c1).expect("C0–C1");
        m.add_bond(c0, h0).expect("C0–H0");
        m.add_bond(c1, h1).expect("C1–H1");
        m.add_port(c0, h0, PortKind::Left, left, BondNumber::Single)
            .expect("port <");
        m.add_port(c1, h1, PortKind::Right, right, BondNumber::Single)
            .expect("port >");
        m
    }

    fn monomer() -> Fragment {
        monomer_labelled("", "")
    }

    /// Li: one atom, mass 6.94, no port.
    fn lithium() -> Fragment {
        let mut li = Fragment::new();
        atom(&mut li, "Li", 0.0, 6.94);
        li
    }

    /// Only a `<` port: C with one H handle.
    fn left_only() -> Fragment {
        let mut f = Fragment::new();
        let c = atom(&mut f, "C", 0.0, 12.0);
        let h = atom(&mut f, "H", -1.0, 1.0);
        f.add_bond(c, h).expect("C–H");
        f.add_port(c, h, PortKind::Left, "", BondNumber::Single)
            .expect("port <");
        f
    }

    /// Two `>` ports on one C, one per H handle.
    fn two_right() -> Fragment {
        let mut f = Fragment::new();
        let c = atom(&mut f, "C", 0.0, 12.0);
        for x in [-1.0, 1.0] {
            let h = atom(&mut f, "H", x, 1.0);
            f.add_bond(c, h).expect("C–H");
            f.add_port(c, h, PortKind::Right, "", BondNumber::Single)
                .expect("port >");
        }
        f
    }

    /// A – X – C with X the handle of both ports: (A, X, `<`) and
    /// (C, X, `>`). Each link alone is fine, but in a 3-unit chain the middle
    /// unit's two leaving groups {X, C} and {X, A} share X.
    fn shared_handle() -> Fragment {
        let mut f = Fragment::new();
        let a = atom(&mut f, "C", 0.0, 12.0);
        let x = atom(&mut f, "O", 1.4, 16.0);
        let c = atom(&mut f, "C", 2.8, 12.0);
        f.add_bond(a, x).expect("A–X");
        f.add_bond(x, c).expect("X–C");
        f.add_port(a, x, PortKind::Left, "", BondNumber::Single)
            .expect("port <");
        f.add_port(c, x, PortKind::Right, "", BondNumber::Single)
            .expect("port >");
        f
    }

    fn library() -> HashMap<String, Fragment> {
        HashMap::from([
            ("M".to_owned(), monomer()),
            ("Li".to_owned(), lithium()),
            ("L".to_owned(), left_only()),
            ("D".to_owned(), two_right()),
            ("P".to_owned(), monomer_labelled("x", "y")),
            ("T".to_owned(), shared_handle()),
        ])
    }

    fn assembler() -> Assembler {
        Assembler::new(library(), Box::new(TracePlacer::new()))
    }

    /// A trace with one point per x, on the x axis.
    fn trace(xs: &[f64]) -> Trace {
        Trace::from_points(xs.iter().map(|&x| [x, 0.0, 0.0]).collect())
    }

    fn seq(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_owned()).collect()
    }

    /// The atom of `world` carrying `frag_id` whose x is `x` (to `TOL`).
    fn atom_at(world: &Fragment, frag_id: u32, x: f64) -> AtomId {
        let found: Vec<AtomId> = world
            .nodes()
            .filter(|(id, a)| {
                world.frag_id(*id) == Some(frag_id)
                    && a.get_f64(keys::X).is_some_and(|ax| (ax - x).abs() < TOL)
            })
            .map(|(id, _)| id)
            .collect();
        assert_eq!(found.len(), 1, "one atom of unit {frag_id} at x = {x}");
        found[0]
    }

    /// `mol_id` of every atom, read from the emitted frame's atoms block.
    fn mol_ids(world: &Fragment) -> Vec<u64> {
        let frame = world.to_frame().expect("the world emits a frame");
        let atoms = frame.get("atoms").expect("an atoms block");
        atoms
            .get_uint(keys::MOL_ID)
            .expect("a uint mol_id column")
            .iter()
            .copied()
            .collect()
    }

    // ---- builds --------------------------------------------------------------

    #[test]
    fn assemble_links_a_three_unit_chain_into_one_molecule() {
        let world = assembler()
            .assemble(&[trace(&[0.0, 5.0, 10.0])], &[seq(&["M", "M", "M"])])
            .expect("a 3-unit M chain assembles");

        // 4·3 − 2·2 atoms; 3·3 − 2·2 + 2 bonds; the two end ports.
        assert_eq!(world.n_atoms(), 8);
        assert_eq!(world.n_bonds(), 7);
        assert_eq!(world.n_ports(), 2);
        let mut frag_ids: Vec<u32> = world
            .node_ids()
            .map(|id| world.frag_id(id).expect("every atom has a frag_id"))
            .collect();
        frag_ids.sort_unstable();
        frag_ids.dedup();
        assert_eq!(frag_ids, vec![0, 1, 2]);
        assert!(mol_ids(&world).iter().all(|&m| m == 1), "one trace, mol 1");
        // Unit k sits on x = 5k: C0 at 5k − 0.75, C1 at 5k + 0.75.
        for k in 0..2u32 {
            let c1 = atom_at(&world, k, 5.0 * f64::from(k) + 0.75);
            let next_c0 = atom_at(&world, k + 1, 5.0 * f64::from(k + 1) - 0.75);
            assert!(
                world.is_bonded(c1, next_c0),
                "unit {k} C1 – unit {} C0",
                k + 1
            );
        }
        let c0 = atom_at(&world, 1, 4.25);
        let x = world.get_node(c0).unwrap().get_f64(keys::X).unwrap();
        assert!((x - 4.25).abs() < TOL, "unit 1 C0 at x = 4.25, got {x}");
    }

    #[test]
    fn assemble_gives_each_trace_its_own_mol_id() {
        let world = assembler()
            .assemble(
                &[trace(&[0.0, 5.0]), trace(&[20.0])],
                &[seq(&["M", "M"]), seq(&["Li"])],
            )
            .expect("a 2-unit M chain and a Li assemble");

        // 4·2 − 2 + 1 atoms.
        assert_eq!(world.n_atoms(), 7);
        let li = atom_at(&world, 2, 20.0);
        assert_eq!(world.frag_id(li), Some(2));
        assert_eq!(world.neighbors(li).count(), 0, "Li is bonded to nothing");
        let frame = world.to_frame().expect("frame");
        let atoms = frame.get("atoms").expect("atoms");
        let elements = atoms.get_string(keys::ELEMENT).expect("element column");
        for (element, mol) in elements.iter().zip(mol_ids(&world)) {
            let expected = if element == "Li" { 2 } else { 1 };
            assert_eq!(mol, expected, "{element} is in molecule {expected}");
        }
    }

    #[test]
    fn assemble_keeps_every_port_of_a_single_point_unit() {
        let world = assembler()
            .assemble(&[trace(&[3.0])], &[seq(&["M"])])
            .expect("one M assembles");

        assert_eq!(world.n_atoms(), 4);
        assert_eq!(world.n_ports(), 2);
    }

    #[test]
    fn assemble_of_no_traces_is_an_empty_fragment() {
        let world = assembler().assemble(&[], &[]).expect("nothing to build");

        assert_eq!(world.n_atoms(), 0);
        assert_eq!(world.n_ports(), 0);
    }

    /// Counts `place_many` calls, then delegates to `TracePlacer`.
    struct CountingPlacer(Arc<AtomicUsize>);

    impl Placer for CountingPlacer {
        fn place_many(
            &self,
            template: &Fragment,
            points: &[Vec3],
        ) -> Result<Vec<Rigid>, PlaceError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            TracePlacer::new().place_many(template, points)
        }
    }

    #[test]
    fn assemble_places_once_per_distinct_name() {
        let calls = Arc::new(AtomicUsize::new(0));
        let assembler = Assembler::new(library(), Box::new(CountingPlacer(calls.clone())));

        assembler
            .assemble(
                &[trace(&[0.0, 5.0]), trace(&[20.0]), trace(&[30.0, 35.0])],
                &[seq(&["M", "M"]), seq(&["Li"]), seq(&["M", "M"])],
            )
            .expect("M and Li assemble");

        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "one call for M, one for Li"
        );
    }

    // ---- refusals ------------------------------------------------------------

    #[test]
    fn assemble_refuses_a_trace_count_other_than_the_sequence_count() {
        let err = assembler()
            .assemble(&[trace(&[0.0])], &[seq(&["M"]), seq(&["M"])])
            .expect_err("1 trace, 2 sequences");

        assert!(
            matches!(
                err,
                AssembleError::LengthMismatch {
                    traces: 1,
                    sequences: 2
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_refuses_a_sequence_shorter_than_its_trace() {
        let err = assembler()
            .assemble(&[trace(&[0.0, 5.0])], &[seq(&["M"])])
            .expect_err("2 points, 1 name");

        assert!(
            matches!(
                err,
                AssembleError::SequenceLength {
                    trace: 0,
                    points: 2,
                    names: 1
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_names_an_unknown_template_by_trace_and_unit() {
        let err = assembler()
            .assemble(&[trace(&[0.0]), trace(&[5.0])], &[seq(&["M"]), seq(&["Q"])])
            .expect_err("Q is not in the library");

        assert!(
            matches!(
                &err,
                AssembleError::UnknownName { trace: 1, unit: 0, name } if name == "Q"
            ),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_refuses_a_unit_without_the_port_its_neighbour_needs() {
        let err = assembler()
            .assemble(&[trace(&[0.0, 5.0])], &[seq(&["L", "M"])])
            .expect_err("L has no `>` port");

        assert!(
            matches!(
                &err,
                AssembleError::MissingPort {
                    trace: 0,
                    unit: 0,
                    name,
                    kind: PortKind::Right,
                } if name == "L"
            ),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_refuses_a_template_with_two_ports_of_the_needed_kind() {
        let err = assembler()
            .assemble(&[trace(&[0.0, 5.0])], &[seq(&["D", "M"])])
            .expect_err("D has two `>` ports");

        assert!(
            matches!(
                &err,
                AssembleError::AmbiguousPort {
                    name,
                    kind: PortKind::Right,
                    count: 2,
                } if name == "D"
            ),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_maps_a_non_finite_point_to_its_unit() {
        let err = assembler()
            .assemble(
                &[trace(&[0.0]), trace(&[5.0, f64::NAN])],
                &[seq(&["M"]), seq(&["M", "M"])],
            )
            .expect_err("a NaN point");

        // The M group is units (0,0), (1,0), (1,1); the NaN is its point 2.
        assert!(
            matches!(
                &err,
                AssembleError::Place {
                    name,
                    trace: 1,
                    unit: 1,
                    source: PlaceError::NonFinitePoint { index: 2 },
                } if name == "M"
            ),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_names_the_left_unit_of_a_refused_link() {
        // P's `>` is labelled y and its `<` x: P–P cannot join.
        let err = assembler()
            .assemble(&[trace(&[0.0, 5.0])], &[seq(&["P", "P"])])
            .expect_err("mismatched labels");

        assert!(
            matches!(
                err,
                AssembleError::Link {
                    trace: 0,
                    unit: 0,
                    source: LinkManyError::Pair {
                        pair: 0,
                        source: LinkError::Incompatible { .. },
                    },
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_maps_a_two_pair_link_error_to_its_first_pair() {
        // Pair 0 joins the M chain; pairs 1 and 2 are the T chain's, whose
        // middle unit's leaving groups share its X.
        let err = assembler()
            .assemble(
                &[trace(&[0.0, 5.0]), trace(&[20.0, 25.0, 30.0])],
                &[seq(&["M", "M"]), seq(&["T", "T", "T"])],
            )
            .expect_err("overlapping leaving groups");

        assert!(
            matches!(
                err,
                AssembleError::Link {
                    trace: 1,
                    unit: 0,
                    source: LinkManyError::BranchesOverlap {
                        first: 1,
                        second: 2
                    },
                }
            ),
            "{err:?}"
        );
    }
}
