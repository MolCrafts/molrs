//! [`FragLibrary`]: named fragment templates, and `map`, which covers a coarse-grained graph with them under a (coarse type, template label) rule relation.
//!
//! A **coarse-grained** (CG) model replaces groups of atoms by single
//! particles called **beads**; a CG graph ([`CoarseGrain`]) has one node per
//! bead, each tagged with a `bead_type` string, and one edge per CG bond. A
//! **template** is an all-atom [`Fragment`] (a molecule piece with open
//! attachment points, *ports*) whose atoms are grouped into beads: each atom
//! carries `bead` (its bead index inside the template) and `bead_type` (that
//! bead's *label*). Mapping a CG graph means partitioning its beads into
//! template occurrences, so every CG bead is assigned one template bead.
//!
//! A template keeps its own bead labels (chemistry-level names). The only
//! bridge from a coarse `bead_type` to a template label is the caller's
//! many-to-many relation `rules`: a coarse bead of type `t` may stand for a
//! template bead labelled `L` iff `(t, L)` is a rule. The two strings are
//! never compared with each other.

use std::collections::btree_map::Entry as Slot;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use crate::store::keys;
use crate::system::bond::BondNumber;
use crate::system::coarsegrain::CoarseGrain;
use crate::system::frag_graph::{FragEdge, FragGraph};
use crate::system::fragment::{Fragment, Port};
use crate::system::graph_match::{Labelling, MatchGraph, SubgraphMatcher};
use crate::system::mapping::Mapping;
use crate::system::molgraph::NodeId;

/// Why a template was refused or a coarse graph could not be mapped.
///
/// Every `bead`, and `a` / `b` of [`Unmatchable`](Self::Unmatchable), is a
/// coarse bead's [`NodeId`] — the handle a [`Mapping`] records its sources
/// by. `a` and `b` of [`AmbiguousRules`](Self::AmbiguousRules) are template
/// names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragLibraryError {
    /// The template lacks a well-formed, connected bead partition (or the
    /// name is already taken).
    InvalidTemplate {
        /// The name the template was to be stored under.
        name: String,
        /// Why it was refused.
        reason: String,
    },
    /// `map` was given no rule.
    NoRules,
    /// No template has every label licensed by some rule.
    NoCandidates,
    /// Two candidates are isomorphic under a bijection that preserves the set
    /// of coarse types each bead label stands for.
    AmbiguousRules {
        /// One template name.
        a: String,
        /// The other template name.
        b: String,
    },
    /// A selected unit admits two assignments of its template beads that
    /// differ in what `bead` is: two different labels over one bead set, or
    /// two port signatures that both carry the unit's inter-unit bonds.
    AmbiguousAssignment {
        /// The lowest-row coarse bead on which the assignments differ.
        bead: NodeId,
    },
    /// A bead is covered by no remaining occurrence.
    Unmapped {
        /// The uncovered coarse bead.
        bead: NodeId,
    },
    /// Every uncovered bead has two or more occurrences left.
    ///
    /// This says unit propagation (repeatedly fixing any bead that has
    /// exactly one occurrence left) found no forced choice, not that the
    /// cover is proven non-unique: a search could still find exactly one
    /// cover.
    Ambiguous {
        /// The lowest-row uncovered coarse bead.
        bead: NodeId,
    },
    /// The coarse bond `a`–`b`, from unit `template_a` to unit `template_b`
    /// (lower unit first), has no pair of free, accepting ports.
    ///
    /// Ports are paired greedily, bond by bond, first free pair first, as
    /// CGsmiles pairs them; a different pairing could succeed where the
    /// greedy one refuses.
    Unmatchable {
        /// The bond's coarse bead in the lower-numbered unit.
        a: NodeId,
        /// The bond's coarse bead in the higher-numbered unit.
        b: NodeId,
        /// The template of `a`'s unit.
        template_a: String,
        /// The template of `b`'s unit.
        template_b: String,
    },
}

impl fmt::Display for FragLibraryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTemplate { name, reason } => {
                write!(f, "template '{name}' is invalid: {reason}")
            }
            Self::NoRules => write!(f, "no (coarse type, template label) rule was given"),
            Self::NoCandidates => write!(f, "no template has every bead label licensed by a rule"),
            Self::AmbiguousRules { a, b } => write!(
                f,
                "templates '{a}' and '{b}' are indistinguishable under the rules"
            ),
            Self::AmbiguousAssignment { bead } => write!(
                f,
                "coarse bead {bead:?} admits two template-bead assignments in its unit"
            ),
            Self::Unmapped { bead } => {
                write!(
                    f,
                    "coarse bead {bead:?} is covered by no template occurrence"
                )
            }
            Self::Ambiguous { bead } => write!(
                f,
                "coarse bead {bead:?} and every other uncovered bead have several occurrences"
            ),
            Self::Unmatchable {
                a,
                b,
                template_a,
                template_b,
            } => write!(
                f,
                "coarse bond {a:?}-{b:?} between units '{template_a}' and '{template_b}' has no \
                 free pair of accepting ports"
            ),
        }
    }
}

impl std::error::Error for FragLibraryError {}

/// One stored template with what `map` reads off it.
#[derive(Debug, Clone)]
struct Entry {
    template: Fragment,
    /// Matcher over the bead pattern: one bead per template bead (row = bead
    /// index, `bead_type` = its label), bonded wherever two beads' atoms are
    /// bonded, snapshotted in the [`Labelling::BeadType`] vocabulary.
    matcher: SubgraphMatcher,
    /// Label per template bead.
    labels: Vec<String>,
    /// Per template bead, `(ordinal, port)` of the ports anchored on it, in
    /// ascending ordinal.
    ports: Vec<Vec<(usize, Port)>>,
    /// Per template bead, its port class: two beads share a class iff their
    /// ports carry one multiset of (kind, label, order).
    port_class: Vec<usize>,
}

impl Entry {
    /// Read the bead partition, bead pattern and per-bead ports of
    /// `template`, or say why it has none.
    fn new(template: Fragment) -> Result<Self, String> {
        let beads = template.beads().map_err(|e| e.to_string())?;
        let n_beads = beads.len();
        if n_beads == 0 {
            return Err("template has no atoms".to_owned());
        }
        let mut bead_of: HashMap<NodeId, usize> = HashMap::new();
        let mut labels: Vec<String> = Vec::with_capacity(n_beads);
        for (bead, atoms) in beads.iter().enumerate() {
            let mut bead_label: Option<String> = None;
            for &id in atoms {
                let atom = template.get_node(id).map_err(|e| e.to_string())?;
                let label = atom
                    .get_str(keys::BEAD_TYPE)
                    .ok_or_else(|| format!("atom {id:?} carries no '{}'", keys::BEAD_TYPE))?;
                match &bead_label {
                    Some(seen) if seen != label => {
                        return Err(format!("bead {bead} carries labels '{seen}' and '{label}'"));
                    }
                    Some(_) => {}
                    None => bead_label = Some(label.to_owned()),
                }
                bead_of.insert(id, bead);
            }
            labels.push(bead_label.ok_or_else(|| format!("bead {bead} has no atom"))?);
        }

        let mut bead_bonds: BTreeSet<(usize, usize)> = BTreeSet::new();
        for (_, bond) in template.bonds() {
            let (bx, by) = (bead_of[&bond.nodes[0]], bead_of[&bond.nodes[1]]);
            if bx != by {
                bead_bonds.insert((bx.min(by), bx.max(by)));
            }
        }
        // A disconnected pattern matches per component, O(N^k) occurrences.
        let mut reached = vec![false; n_beads];
        let mut stack = vec![0];
        reached[0] = true;
        while let Some(bead) = stack.pop() {
            for &(x, y) in &bead_bonds {
                let other = match bead {
                    b if b == x => y,
                    b if b == y => x,
                    _ => continue,
                };
                if !std::mem::replace(&mut reached[other], true) {
                    stack.push(other);
                }
            }
        }
        if let Some(apart) = reached.iter().position(|&r| !r) {
            return Err(format!(
                "the bead pattern is disconnected: bead {apart} is not bonded to bead 0's component"
            ));
        }
        let mut pattern = CoarseGrain::new();
        let beads: Vec<NodeId> = labels.iter().map(|l| pattern.add_bead_bare(l)).collect();
        for (bx, by) in bead_bonds {
            pattern
                .add_bond(beads[bx], beads[by])
                .map_err(|e| format!("bead pattern bond {bx}-{by}: {e}"))?;
        }
        let matcher =
            SubgraphMatcher::new(MatchGraph::new(pattern.as_molgraph(), Labelling::BeadType));

        let mut ports: Vec<Vec<(usize, Port)>> = vec![Vec::new(); n_beads];
        let ordered = template.ordered_ports().map_err(|e| e.to_string())?;
        for (ordinal, id) in ordered.into_iter().enumerate() {
            let port = template.port(id).map_err(|e| format!("port {id:?}: {e}"))?;
            let bead = *bead_of
                .get(&port.anchor)
                .ok_or_else(|| format!("port {ordinal} is anchored on no live atom"))?;
            ports[bead].push((ordinal, port));
        }

        let mut classes: BTreeMap<Vec<(&'static str, &str, BondNumber)>, usize> = BTreeMap::new();
        let port_class = ports
            .iter()
            .map(|on_bead| {
                let mut key: Vec<(&'static str, &str, BondNumber)> = on_bead
                    .iter()
                    .map(|(_, p)| (p.kind.as_str(), p.label.as_str(), p.order))
                    .collect();
                key.sort_unstable();
                let next = classes.len();
                *classes.entry(key).or_insert(next)
            })
            .collect();

        Ok(Self {
            template,
            matcher,
            labels,
            ports,
            port_class,
        })
    }

    /// `(coarse row, label)` of each template bead under the map `rows`
    /// (template bead `i` → coarse row `rows[i]`), ascending by row.
    fn row_labels(&self, rows: &[usize]) -> Vec<(usize, &str)> {
        let mut out: Vec<(usize, &str)> = rows
            .iter()
            .zip(&self.labels)
            .map(|(&row, label)| (row, label.as_str()))
            .collect();
        out.sort_unstable();
        out
    }

    /// The port signature of the map `rows`: `(coarse row, port class)` of
    /// each template bead, ascending by row.
    fn signature(&self, rows: &[usize]) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = rows
            .iter()
            .copied()
            .zip(self.port_class.iter().copied())
            .collect();
        out.sort_unstable();
        out
    }

    /// The lowest coarse row of the map `rows` whose template bead has fewer
    /// ports than the row has inter-unit bonds (`inter`), if any.
    fn overloaded(&self, rows: &[usize], inter: &[usize]) -> Option<usize> {
        rows.iter()
            .zip(&self.ports)
            .filter(|&(&row, ports)| ports.len() < inter[row])
            .map(|(&row, _)| row)
            .min()
    }
}

/// One occurrence of a template in the coarse graph: a bead set with the
/// assignments of template beads to it.
struct Occurrence<'a> {
    name: &'a str,
    entry: &'a Entry,
    /// The coarse rows covered, ascending.
    set: Vec<usize>,
    /// One map per port-signature class, in the order first met (so the
    /// first is the lexicographically smallest map): the coarse row of each
    /// template bead.
    alternatives: Vec<Vec<usize>>,
    /// The lowest row two maps over `set` label differently, when any do.
    conflict: Option<usize>,
}

impl Occurrence<'_> {
    /// Fold one more map over this bead set in: a different labelling marks
    /// the set conflicting; an equal labelling with a new port signature is
    /// a new alternative; anything else is equivalent to a kept map.
    fn absorb(&mut self, rows: Vec<usize>) {
        let kept = self.entry.row_labels(&self.alternatives[0]);
        let differing = kept
            .iter()
            .zip(self.entry.row_labels(&rows))
            .find(|(k, r)| k.1 != r.1)
            .map(|(k, _)| k.0);
        if let Some(row) = differing {
            self.conflict = Some(self.conflict.map_or(row, |c| c.min(row)));
            return;
        }
        let signature = self.entry.signature(&rows);
        if self
            .alternatives
            .iter()
            .all(|alt| self.entry.signature(alt) != signature)
        {
            self.alternatives.push(rows);
        }
    }
}

/// Named fragment templates, each keeping its own bead labels.
///
/// A template's atoms carry `bead` (the template-local bead index, beads
/// numbered `0..k`) and `bead_type` (that bead's label). Ports are addressed
/// by ordinal, their index in [`Fragment::ordered_ports`].
#[derive(Debug, Clone, Default)]
pub struct FragLibrary {
    entries: BTreeMap<String, Entry>,
}

impl FragLibrary {
    /// An empty library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `template` under `name`.
    ///
    /// # Errors
    ///
    /// [`FragLibraryError::InvalidTemplate`] when `name` is already taken,
    /// when the template has no atom, when an atom lacks a non-negative
    /// integer `bead` or a string `bead_type`, when the beads are not
    /// numbered `0..k` contiguously, when one bead carries two labels, when
    /// the bead pattern is disconnected (it would match per component), or
    /// when a port cannot be read back.
    pub fn insert(&mut self, name: &str, template: Fragment) -> Result<(), FragLibraryError> {
        let invalid = |reason: String| FragLibraryError::InvalidTemplate {
            name: name.to_owned(),
            reason,
        };
        if self.entries.contains_key(name) {
            return Err(invalid(
                "a template of this name is already stored".to_owned(),
            ));
        }
        let entry = Entry::new(template).map_err(invalid)?;
        self.entries.insert(name.to_owned(), entry);
        Ok(())
    }

    /// The template stored under `name`.
    pub fn get(&self, name: &str) -> Option<&Fragment> {
        self.entries.get(name).map(|e| &e.template)
    }

    /// Template names, in lexicographic order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Partition `graph` into template occurrences under `rules`, a
    /// (coarse type, template label) relation, and pair their ports.
    ///
    /// Coarse beads are labelled by `bead_type` alone and coarse bonds carry
    /// no label, whatever `element`, `bond_type` or `bond_number` columns the
    /// graph also holds.
    ///
    /// 1. A template is a candidate iff every label is the template label of
    ///    some rule; others are skipped.
    /// 2. Two candidates isomorphic under a bijection preserving the set of
    ///    coarse types each label stands for are refused.
    /// 3. Occurrences are the induced matches of each candidate's bead
    ///    pattern, grouped by bead set. Label-identical maps are one
    ///    occurrence, kept as one alternative per port signature (per coarse
    ///    bead, the multiset of ports on its template bead; the
    ///    lexicographically smallest map of a signature is kept). Maps that
    ///    label one bead set differently mark it conflicting.
    /// 4. An exact cover is forced by unit propagation: the lowest-row bead
    ///    with exactly one occurrence fixes it, and every overlapping
    ///    occurrence is dropped. Units are numbered by lowest source row.
    /// 5. A selected conflicting set is refused. A unit with several
    ///    alternatives keeps the one whose every bead has at least as many
    ///    ports as inter-unit bonds.
    /// 6. Inter-unit bonds, in (min unit, max unit, bond row) order, take the
    ///    first free port pair on their two beads that [`Port::accepts`]
    ///    (greedy).
    ///
    /// `sources` of the result are defined up to a port-preserving label
    /// automorphism of each template.
    ///
    /// # Errors
    ///
    /// [`NoRules`](FragLibraryError::NoRules),
    /// [`NoCandidates`](FragLibraryError::NoCandidates),
    /// [`AmbiguousRules`](FragLibraryError::AmbiguousRules),
    /// [`Unmapped`](FragLibraryError::Unmapped),
    /// [`Ambiguous`](FragLibraryError::Ambiguous),
    /// [`AmbiguousAssignment`](FragLibraryError::AmbiguousAssignment) and
    /// [`Unmatchable`](FragLibraryError::Unmatchable), each at the step above
    /// that detects it (`Unmatchable` also at step 5, when no alternative of
    /// a unit has ports enough for its bonds).
    pub fn map(
        &self,
        graph: &CoarseGrain,
        rules: &[(&str, &str)],
    ) -> Result<Mapping, FragLibraryError> {
        if rules.is_empty() {
            return Err(FragLibraryError::NoRules);
        }
        let licensed: HashSet<&str> = rules.iter().map(|&(_, label)| label).collect();
        let candidates: Vec<(&str, &Entry)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.labels.iter().all(|l| licensed.contains(l.as_str())))
            .map(|(name, e)| (name.as_str(), e))
            .collect();
        if candidates.is_empty() {
            return Err(FragLibraryError::NoCandidates);
        }

        // Step 2: types(L) = {t : (t, L) ∈ rules}.
        let mut types: HashMap<&str, BTreeSet<&str>> = HashMap::new();
        for &(t, label) in rules {
            types.entry(label).or_default().insert(t);
        }
        for (i, &(name_a, a)) in candidates.iter().enumerate() {
            for &(name_b, b) in &candidates[i + 1..] {
                if a.labels.len() != b.labels.len() {
                    continue;
                }
                // Equal node counts make an induced injective match a bijection.
                let same = a
                    .matcher
                    .find_all(b.matcher.pattern(), |la, lb| types.get(la) == types.get(lb));
                if !same.is_empty() {
                    return Err(FragLibraryError::AmbiguousRules {
                        a: name_a.to_owned(),
                        b: name_b.to_owned(),
                    });
                }
            }
        }

        // Step 3: occurrences, grouped by bead set.
        let target = MatchGraph::new(graph.as_molgraph(), Labelling::BeadType);
        let ids: Vec<NodeId> = graph.node_ids().collect();
        let row_of: HashMap<NodeId, usize> =
            ids.iter().enumerate().map(|(r, &id)| (id, r)).collect();
        let rule_set: HashSet<(&str, &str)> = rules.iter().copied().collect();
        let mut occurrences: Vec<Occurrence<'_>> = Vec::new();
        for &(name, entry) in &candidates {
            // `find_all` sorts its maps, so the first of a bead set is the
            // lexicographically smallest.
            let maps = entry
                .matcher
                .find_all(&target, |label, t| rule_set.contains(&(t, label)));
            let mut groups: BTreeMap<Vec<usize>, Occurrence<'_>> = BTreeMap::new();
            for m in maps {
                let rows: Vec<usize> = m.iter().map(|id| row_of[id]).collect();
                let mut set = rows.clone();
                set.sort_unstable();
                match groups.entry(set) {
                    Slot::Vacant(slot) => {
                        let set = slot.key().clone();
                        slot.insert(Occurrence {
                            name,
                            entry,
                            set,
                            alternatives: vec![rows],
                            conflict: None,
                        });
                    }
                    Slot::Occupied(mut slot) => slot.get_mut().absorb(rows),
                }
            }
            occurrences.extend(groups.into_values());
        }

        // Step 4: exact cover by unit propagation.
        let n = ids.len();
        let mut cover_of: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (o, occ) in occurrences.iter().enumerate() {
            for &row in &occ.set {
                cover_of[row].push(o);
            }
        }
        let mut count: Vec<usize> = cover_of.iter().map(Vec::len).collect();
        if let Some(row) = count.iter().position(|&c| c == 0) {
            return Err(FragLibraryError::Unmapped { bead: ids[row] });
        }
        let mut alive = vec![true; occurrences.len()];
        let mut covered = vec![false; n];
        let mut forced: BTreeSet<usize> = (0..n).filter(|&b| count[b] == 1).collect();
        let mut remaining = n;
        let mut chosen: Vec<usize> = Vec::new();
        while remaining > 0 {
            let Some(row) = forced.pop_first() else {
                let row = covered
                    .iter()
                    .position(|&c| !c)
                    .expect("a bead remains uncovered while `remaining` > 0");
                return Err(FragLibraryError::Ambiguous { bead: ids[row] });
            };
            let Some(&fixed) = cover_of[row].iter().find(|&&o| alive[o]) else {
                return Err(FragLibraryError::Unmapped { bead: ids[row] });
            };
            chosen.push(fixed);
            for &r in &occurrences[fixed].set {
                covered[r] = true;
                remaining -= 1;
                forced.remove(&r);
            }
            let mut emptied: Vec<usize> = Vec::new();
            for &r in &occurrences[fixed].set {
                for &o in &cover_of[r] {
                    if !std::mem::replace(&mut alive[o], false) {
                        continue;
                    }
                    for &other in &occurrences[o].set {
                        if covered[other] {
                            continue;
                        }
                        count[other] -= 1;
                        match count[other] {
                            0 => {
                                forced.remove(&other);
                                emptied.push(other);
                            }
                            1 => {
                                forced.insert(other);
                            }
                            _ => {}
                        }
                    }
                }
            }
            if let Some(row) = emptied.into_iter().min() {
                return Err(FragLibraryError::Unmapped { bead: ids[row] });
            }
        }
        chosen.sort_by_key(|&o| occurrences[o].set[0]);

        // Step 5: one assignment per unit.
        let mut unit_of = vec![0; n];
        for (u, &o) in chosen.iter().enumerate() {
            for &row in &occurrences[o].set {
                unit_of[row] = u;
            }
        }
        let mut links: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
        for (bond_row, (_, bond)) in graph.bonds().enumerate() {
            let (mut x, mut y) = (row_of[&bond.nodes[0]], row_of[&bond.nodes[1]]);
            if unit_of[x] == unit_of[y] {
                continue;
            }
            if unit_of[x] > unit_of[y] {
                std::mem::swap(&mut x, &mut y);
            }
            links.push((unit_of[x], unit_of[y], bond_row, x, y));
        }
        links.sort_unstable();
        let mut inter = vec![0usize; n];
        for &(_, _, _, x, y) in &links {
            inter[x] += 1;
            inter[y] += 1;
        }
        let unmatchable = |x: usize, y: usize| FragLibraryError::Unmatchable {
            a: ids[x],
            b: ids[y],
            template_a: occurrences[chosen[unit_of[x]]].name.to_owned(),
            template_b: occurrences[chosen[unit_of[y]]].name.to_owned(),
        };
        let mut assigned: Vec<&[usize]> = Vec::with_capacity(chosen.len());
        for &o in &chosen {
            let occ = &occurrences[o];
            if let Some(row) = occ.conflict {
                return Err(FragLibraryError::AmbiguousAssignment { bead: ids[row] });
            }
            if let [only] = occ.alternatives.as_slice() {
                assigned.push(only);
                continue;
            }
            let feasible: Vec<&Vec<usize>> = occ
                .alternatives
                .iter()
                .filter(|alt| occ.entry.overloaded(alt, &inter).is_none())
                .collect();
            match feasible.as_slice() {
                [one] => assigned.push(one),
                [] => {
                    let row = occ
                        .entry
                        .overloaded(&occ.alternatives[0], &inter)
                        .expect("an infeasible alternative has an overloaded bead");
                    let &(_, _, _, x, y) = links
                        .iter()
                        .find(|l| l.3 == row || l.4 == row)
                        .expect("an overloaded bead has an inter-unit bond");
                    return Err(unmatchable(x, y));
                }
                [first, second, ..] => {
                    let row = occ
                        .entry
                        .signature(first)
                        .into_iter()
                        .zip(occ.entry.signature(second))
                        .find(|(f, s)| f != s)
                        .map(|(f, _)| f.0)
                        .expect("two port-signature classes differ on some bead");
                    return Err(FragLibraryError::AmbiguousAssignment { bead: ids[row] });
                }
            }
        }

        // Step 6: ports.
        let mut bead_in_unit = vec![0; n];
        for rows in &assigned {
            for (i, &row) in rows.iter().enumerate() {
                bead_in_unit[row] = i;
            }
        }
        let mut used: Vec<HashSet<usize>> = vec![HashSet::new(); chosen.len()];
        let mut edges: Vec<FragEdge> = Vec::with_capacity(links.len());
        for &(a, b, _, x, y) in &links {
            let ports_a = &occurrences[chosen[a]].entry.ports[bead_in_unit[x]];
            let ports_b = &occurrences[chosen[b]].entry.ports[bead_in_unit[y]];
            let pair = ports_a
                .iter()
                .filter(|(o, _)| !used[a].contains(o))
                .find_map(|(oa, pa)| {
                    ports_b
                        .iter()
                        .find(|(ob, pb)| !used[b].contains(ob) && pa.accepts(pb))
                        .map(|(ob, _)| (*oa, *ob))
                });
            let Some((port_a, port_b)) = pair else {
                return Err(unmatchable(x, y));
            };
            used[a].insert(port_a);
            used[b].insert(port_b);
            edges.push(FragEdge {
                a,
                b,
                port_a,
                port_b,
            });
        }

        // Step 7: the mapping.
        let nodes: Vec<String> = chosen
            .iter()
            .map(|&o| occurrences[o].name.to_owned())
            .collect();
        let sources: Vec<Vec<NodeId>> = assigned
            .iter()
            .map(|rows| rows.iter().map(|&r| ids[r]).collect())
            .collect();
        let labels: Vec<Vec<String>> = chosen
            .iter()
            .map(|&o| occurrences[o].entry.labels.clone())
            .collect();
        let rules: Vec<(String, String)> = rules
            .iter()
            .map(|&(t, l)| (t.to_owned(), l.to_owned()))
            .collect();
        // Every refusal of the two constructors is excluded above: units
        // a < b in range, no port used twice, a disjoint, non-empty cover,
        // and labels drawn from candidates whose every label some rule
        // licenses.
        let graph = FragGraph::new(nodes, edges)
            .expect("inter-unit edges are in range, not self-edges, and use each port once");
        Ok(Mapping::new(graph, sources, labels, rules)
            .expect("a disjoint cover by rule-licensed candidates is a valid mapping"))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{FragLibrary, FragLibraryError};
    use crate::store::keys;
    use crate::system::atomistic::AtomId;
    use crate::system::bond::BondNumber;
    use crate::system::coarsegrain::CoarseGrain;
    use crate::system::frag_graph::FragEdge;
    use crate::system::fragment::{Fragment, PortKind};
    use crate::system::mapping::Mapping;
    use crate::system::molgraph::NodeId;

    // Every expected value below is hand-derived from the fixtures, following
    // `.claude/specs/assembly-04-fraggraph.md` § Design 5 and acceptance
    // ac-006..ac-010, ac-012. No external program produced any value here.
    //
    // Conventions:
    // * A template has one heavy atom per bead (row = bead index), then its
    //   capping hydrogens, appended bead by bead in the order the port kinds
    //   are listed. A port's ordinal is its handle's row rank, so the
    //   ordinals follow that listing order.
    // * Every port is unlabelled (`""`) and single-order, so `Port::accepts`
    //   pairs exactly `<` with `>`.
    // * Sources are asserted as coarse bead ROWS (index in `node_ids()`).
    // * Units are numbered by their lowest source row.

    // -- fixtures -------------------------------------------------------------

    /// Add one atom stamped with its template bead index and label.
    fn stamped(frag: &mut Fragment, element: &str, bead: i32, label: &str) -> AtomId {
        let id = frag.add_atom_bare(element);
        frag.set_node(id, keys::BEAD, bead).expect("stamp bead");
        frag.set_node(id, keys::BEAD_TYPE, label)
            .expect("stamp bead_type");
        id
    }

    /// A template with one heavy atom per `beads[i] = (label, port kinds)`,
    /// bead index `i`, bonded along `bonds`; each listed port kind becomes a
    /// capping hydrogen (stamped with its anchor's bead) carrying that port.
    fn template(beads: &[(&str, &[PortKind])], bonds: &[(usize, usize)]) -> Fragment {
        let mut frag = Fragment::new();
        let heavy: Vec<AtomId> = beads
            .iter()
            .enumerate()
            .map(|(i, &(label, _))| stamped(&mut frag, "C", bead_index(i), label))
            .collect();
        for &(a, b) in bonds {
            frag.add_bond(heavy[a], heavy[b]).expect("template bond");
        }
        for (i, &(label, kinds)) in beads.iter().enumerate() {
            for &kind in kinds {
                let h = stamped(&mut frag, "H", bead_index(i), label);
                frag.add_bond(heavy[i], h).expect("handle bond");
                frag.add_port(heavy[i], h, kind, "", BondNumber::Single)
                    .expect("template port");
            }
        }
        frag
    }

    fn bead_index(i: usize) -> i32 {
        i32::try_from(i).expect("fixture bead index fits i32")
    }

    const LR: &[PortKind] = &[PortKind::Left, PortKind::Right];
    const NONE: &[PortKind] = &[];

    /// PMA: beads B(0), S1(1), S2(2), A(3) on the path B–S1–S2–A; two
    /// handles on B, `<` (ordinal 0) then `>` (ordinal 1).
    fn pma() -> Fragment {
        template(
            &[("B", LR), ("S1", NONE), ("S2", NONE), ("A", NONE)],
            &[(0, 1), (1, 2), (2, 3)],
        )
    }

    /// A one-bead template labelled `label`, with no ports.
    fn single(label: &str) -> Fragment {
        template(&[(label, NONE)], &[])
    }

    /// AA: two bonded beads, both labelled `A`, each carrying `<` then `>`
    /// (ordinals 0,1 on bead 0 and 2,3 on bead 1).
    fn aa() -> Fragment {
        template(&[("A", LR), ("A", LR)], &[(0, 1)])
    }

    fn library(entries: Vec<(&str, Fragment)>) -> FragLibrary {
        let mut lib = FragLibrary::new();
        for (name, frag) in entries {
            lib.insert(name, frag)
                .unwrap_or_else(|e| panic!("insert({name:?}) failed: {e:?}"));
        }
        lib
    }

    /// A coarse graph with one bead per entry of `types` (row = index) and one
    /// CG bond per `(row, row)` pair.
    fn cg(types: &[&str], bonds: &[(usize, usize)]) -> CoarseGrain {
        let mut g = CoarseGrain::new();
        let ids: Vec<NodeId> = types.iter().map(|t| g.add_bead(t, 0.0, 0.0, 0.0)).collect();
        for &(a, b) in bonds {
            g.add_bond(ids[a], ids[b]).expect("fixture bond");
        }
        g
    }

    /// Monomer `m` of the PMA target: rows A 4m, S2 4m+1, B 4m+2, S1 4m+3.
    const PMA_TYPES: [&str; 4] = ["4", "1", "1", "1"];

    /// Intra-monomer bonds of monomer `m`: B–S1, S1–S2, S2–A.
    fn pma_bonds(m: usize) -> [(usize, usize); 3] {
        let o = 4 * m;
        [(o + 2, o + 3), (o + 3, o + 1), (o + 1, o)]
    }

    const PMA_RULES: [(&str, &str); 4] = [("1", "B"), ("1", "S1"), ("1", "S2"), ("4", "A")];

    /// Three PMA monomers with scrambled rows, joined B–B along the backbone
    /// by the bonds (2,6) and (6,10).
    fn pma_chain() -> CoarseGrain {
        let types: Vec<&str> = (0..3).flat_map(|_| PMA_TYPES).collect();
        let mut bonds: Vec<(usize, usize)> = (0..3).flat_map(pma_bonds).collect();
        bonds.extend([(2, 6), (6, 10)]);
        cg(&types, &bonds)
    }

    /// The hand-derived PMA golden (spec § Design 5): three units whose
    /// sources are B, S1, S2, A of each monomer, joined `<` (0) to `>` (1).
    fn assert_pma_golden(graph: &CoarseGrain, m: &Mapping) {
        assert_eq!(nodes(m), vec!["PMA", "PMA", "PMA"]);
        assert_eq!(
            source_rows(graph, m),
            vec![vec![2, 3, 1, 0], vec![6, 7, 5, 4], vec![10, 11, 9, 8]]
        );
        for u in 0..3 {
            assert_eq!(
                m.labels(u),
                Some(&["B", "S1", "S2", "A"].map(String::from)[..]),
                "unit {u}"
            );
        }
        assert_eq!(m.graph().edges(), &[edge(0, 1, 0, 1), edge(1, 2, 0, 1)]);
    }

    /// Each unit's sources as coarse bead rows.
    fn source_rows(graph: &CoarseGrain, m: &Mapping) -> Vec<Vec<usize>> {
        let order: Vec<NodeId> = graph.node_ids().collect();
        (0..m.n_units())
            .map(|u| {
                m.sources(u)
                    .expect("unit in range")
                    .iter()
                    .map(|id| {
                        order
                            .iter()
                            .position(|n| n == id)
                            .expect("a source is a live coarse bead")
                    })
                    .collect()
            })
            .collect()
    }

    fn nodes(m: &Mapping) -> Vec<&str> {
        m.graph().nodes().iter().map(String::as_str).collect()
    }

    fn edge(a: usize, b: usize, port_a: usize, port_b: usize) -> FragEdge {
        FragEdge {
            a,
            b,
            port_a,
            port_b,
        }
    }

    // -- insert ---------------------------------------------------------------

    #[test]
    fn insert_accepts_a_template_without_a_site_column() {
        let mut lib = FragLibrary::new();
        lib.insert("PMA", pma())
            .expect("bead + bead_type suffice for a template");
        assert_eq!(lib.get("PMA").map(Fragment::n_atoms), Some(6));
        assert!(lib.get("PC").is_none());
        assert_eq!(lib.names().collect::<Vec<_>>(), vec!["PMA"]);
    }

    #[test]
    fn insert_refuses_an_atom_lacking_bead() {
        let mut frag = template(&[("A", NONE)], &[]);
        let bare = frag.add_atom_bare("C");
        frag.set_node(bare, keys::BEAD_TYPE, "A")
            .expect("stamp bead_type");
        let err = FragLibrary::new()
            .insert("Bad", frag)
            .expect_err("an atom without `bead` is refused");
        assert!(
            matches!(err, FragLibraryError::InvalidTemplate { ref name, .. } if name == "Bad"),
            "{err:?}"
        );
    }

    #[test]
    fn insert_refuses_non_contiguous_beads() {
        let mut frag = Fragment::new();
        let a = stamped(&mut frag, "C", 0, "A");
        let b = stamped(&mut frag, "C", 2, "B");
        frag.add_bond(a, b).expect("bond");
        let err = FragLibrary::new()
            .insert("Gap", frag)
            .expect_err("beads {0, 2} skip bead 1");
        assert!(
            matches!(err, FragLibraryError::InvalidTemplate { ref name, .. } if name == "Gap"),
            "{err:?}"
        );
    }

    #[test]
    fn insert_refuses_one_bead_with_two_labels() {
        let mut frag = Fragment::new();
        let a = stamped(&mut frag, "C", 0, "A");
        let b = stamped(&mut frag, "C", 0, "B");
        frag.add_bond(a, b).expect("bond");
        let err = FragLibrary::new()
            .insert("Split", frag)
            .expect_err("bead 0 carries labels A and B");
        assert!(
            matches!(err, FragLibraryError::InvalidTemplate { ref name, .. } if name == "Split"),
            "{err:?}"
        );
    }

    #[test]
    fn insert_refuses_a_disconnected_bead_pattern() {
        // Beads A(0) and B(1) share no bond: the bead pattern has two
        // components.
        let frag = template(&[("A", NONE), ("B", NONE)], &[]);
        let err = FragLibrary::new()
            .insert("Apart", frag)
            .expect_err("a disconnected bead pattern is refused");
        assert!(
            matches!(err, FragLibraryError::InvalidTemplate { ref name, .. } if name == "Apart"),
            "{err:?}"
        );
    }

    // -- map: regression golden (ac-007, ac-012) --------------------------------

    /// Hard-coded golden (hand-derived, spec § Design 5). Three PMA monomers
    /// with scrambled rows; three of each monomer's four beads share coarse
    /// type "1", so only connectivity tells B, S1 and S2 apart: A is the one
    /// type-"4" bead, S2 its only neighbour, S1 the next, B the last.
    /// Ports: each backbone bond joins two B beads; unit a's `<` (ordinal 0)
    /// pairs unit b's `>` (ordinal 1); the second bond finds unit 1's `>`
    /// used, so it takes `<` (0) and unit 2's `>` (1).
    #[test]
    fn map_resolves_pma_type1_beads_by_connectivity() {
        let lib = library(vec![("PMA", pma())]);
        let graph = pma_chain();

        let m = lib.map(&graph, &PMA_RULES).expect("PMA maps uniquely");

        assert_pma_golden(&graph, &m);
        let rules: Vec<(String, String)> = PMA_RULES
            .iter()
            .map(|&(t, l)| (t.to_owned(), l.to_owned()))
            .collect();
        assert_eq!(m.rules(), rules.as_slice());
    }

    /// Coarse beads are labelled by `bead_type` only: an `element` column on
    /// a frame-derived coarse graph must not replace the type the rules
    /// read. Every bead also carries element "C"; the result is the golden.
    #[test]
    fn map_labels_coarse_beads_by_bead_type_even_with_an_element_column() {
        let lib = library(vec![("PMA", pma())]);
        let mut graph = pma_chain();
        let ids: Vec<NodeId> = graph.node_ids().collect();
        for id in ids {
            graph
                .set_node(id, keys::ELEMENT, "C")
                .expect("stamp element");
        }

        let m = lib
            .map(&graph, &PMA_RULES)
            .expect("the element column is ignored");

        assert_pma_golden(&graph, &m);
    }

    /// CG bonds carry no edge label: `bond_type` / `bond_number` on the
    /// coarse bonds (as a frame-derived graph may hold) must not stop the
    /// template's unlabelled bead bonds from matching them.
    #[test]
    fn map_ignores_bond_type_on_coarse_bonds() {
        let lib = library(vec![("PMA", pma())]);
        let mut graph = pma_chain();
        let kind = graph.kind_id("bonds").expect("a coarse graph has bonds");
        let bonds: Vec<_> = graph.bonds().map(|(id, _)| id).collect();
        for id in bonds {
            graph
                .set_relation_prop(kind, id, keys::BOND_TYPE, 1)
                .expect("stamp bond_type");
            graph
                .set_relation_prop(kind, id, keys::BOND_NUMBER, 1)
                .expect("stamp bond_number");
        }

        let m = lib
            .map(&graph, &PMA_RULES)
            .expect("bond_type / bond_number are ignored");

        assert_pma_golden(&graph, &m);
    }

    /// BSS (B–S1–S2) fits every type-"1" 3-path of the PMA chain forwards and
    /// backwards, so each of its bead sets is conflicting. Propagation forces
    /// the three PMA units first (rows 0, 4, 8 each have one candidate), and
    /// they overlap every BSS set, which is dropped rather than refused.
    #[test]
    fn map_ignores_a_conflicting_bead_set_the_cover_drops() {
        let lib = library(vec![
            ("PMA", pma()),
            (
                "BSS",
                template(
                    &[("B", NONE), ("S1", NONE), ("S2", NONE)],
                    &[(0, 1), (1, 2)],
                ),
            ),
        ]);
        let graph = pma_chain();

        let m = lib
            .map(&graph, &PMA_RULES)
            .expect("the cover never selects a conflicting BSS set");

        assert_pma_golden(&graph, &m);
    }

    // -- map: candidates (ac-008) ---------------------------------------------

    #[test]
    fn map_skips_templates_with_an_unmapped_label() {
        let lib = library(vec![
            ("PMA", pma()),
            ("PC", single("PC")),
            ("Li", single("Li")),
            ("X", single("Z")),
        ]);
        let mut types: Vec<&str> = PMA_TYPES.to_vec();
        types.extend(["3", "2"]);
        let graph = cg(&types, &pma_bonds(0));
        let mut rules: Vec<(&str, &str)> = PMA_RULES.to_vec();
        rules.extend([("3", "PC"), ("2", "Li")]);

        let m = lib
            .map(&graph, &rules)
            .expect("X (label Z) is skipped, not an error");

        assert_eq!(nodes(&m), vec!["PMA", "PC", "Li"]);
        assert_eq!(
            source_rows(&graph, &m),
            vec![vec![2, 3, 1, 0], vec![4], vec![5]]
        );
        assert!(m.graph().edges().is_empty());
    }

    #[test]
    fn map_refuses_empty_rules() {
        let lib = library(vec![("PMA", pma())]);
        let graph = cg(&PMA_TYPES, &pma_bonds(0));
        let err = lib.map(&graph, &[]).expect_err("no rules");
        assert!(matches!(err, FragLibraryError::NoRules), "{err:?}");
    }

    #[test]
    fn map_refuses_rules_that_leave_no_candidate() {
        let lib = library(vec![("PMA", pma())]);
        let graph = cg(&PMA_TYPES, &pma_bonds(0));
        let err = lib
            .map(&graph, &[("9", "Q")])
            .expect_err("every PMA label is unmapped");
        assert!(matches!(err, FragLibraryError::NoCandidates), "{err:?}");
    }

    // -- map: ambiguity refusals (ac-009) -------------------------------------

    #[test]
    fn map_refuses_candidates_indistinguishable_under_the_rules() {
        let lib = library(vec![("PC", single("PC")), ("PCb", single("PCb"))]);
        let graph = cg(&["3"], &[]);
        let err = lib
            .map(&graph, &[("3", "PC"), ("3", "PCb")])
            .expect_err("PC and PCb both stand for type 3 only");
        let FragLibraryError::AmbiguousRules { a, b } = err else {
            panic!("expected AmbiguousRules, got {err:?}");
        };
        let pair: HashSet<String> = [a, b].into_iter().collect();
        let expected: HashSet<String> = ["PC", "PCb"].map(String::from).into_iter().collect();
        assert_eq!(pair, expected);
    }

    #[test]
    fn map_refuses_one_bead_set_with_two_label_assignments() {
        // B–S1–S2 on 1–1–1 fits forwards and backwards: row 0 is B or S2.
        let lib = library(vec![(
            "BSS",
            template(
                &[("B", NONE), ("S1", NONE), ("S2", NONE)],
                &[(0, 1), (1, 2)],
            ),
        )]);
        let graph = cg(&["1", "1", "1"], &[(0, 1), (1, 2)]);
        let err = lib
            .map(&graph, &[("1", "B"), ("1", "S1"), ("1", "S2")])
            .expect_err("two label assignments over one bead set");
        assert!(
            matches!(err, FragLibraryError::AmbiguousAssignment { .. }),
            "{err:?}"
        );
    }

    /// AA occurs on {0,1} and {0,2} only, so the star has no exact cover.
    /// Propagation takes the lowest-row bead with one candidate (bead 1),
    /// fixes {0,1}, drops the overlapping {0,2}, and leaves bead 2 with none.
    #[test]
    fn map_refuses_a_star_that_has_no_exact_cover() {
        let lib = library(vec![("AA", aa())]);
        let graph = cg(&["1", "1", "1"], &[(0, 1), (0, 2)]);
        let ids: Vec<NodeId> = graph.node_ids().collect();
        let err = lib.map(&graph, &[("1", "A")]).expect_err("star");
        assert_eq!(err, FragLibraryError::Unmapped { bead: ids[2] });
    }

    #[test]
    fn map_refuses_an_ambiguous_cover_of_a_ring() {
        // AA occurs on {0,1}, {1,2}, {2,3}, {0,3}: every bead has two.
        let lib = library(vec![("AA", aa())]);
        let graph = cg(&["1", "1", "1", "1"], &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let err = lib.map(&graph, &[("1", "A")]).expect_err("ring");
        assert!(matches!(err, FragLibraryError::Ambiguous { .. }), "{err:?}");
    }

    /// Chain rows 1-0-2-3: AA occurs on {0,1}, {0,2}, {2,3}. Beads 1 and 3
    /// each have one candidate, which forces {0,1} and {2,3}. AA is
    /// symmetric, so each bead set has two label-identical maps; the
    /// lexicographically smallest (template bead 0 → lower row) is kept.
    #[test]
    fn map_forces_the_cover_by_unit_propagation() {
        let lib = library(vec![("AA", aa())]);
        let graph = cg(&["1", "1", "1", "1"], &[(1, 0), (0, 2), (2, 3)]);
        let m = lib
            .map(&graph, &[("1", "A")])
            .expect("the chain has one exact cover");
        assert_eq!(nodes(&m), vec!["AA", "AA"]);
        assert_eq!(source_rows(&graph, &m), vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn map_refuses_a_bead_no_rule_covers() {
        let lib = library(vec![("PMA", pma())]);
        let mut types: Vec<&str> = PMA_TYPES.to_vec();
        types.push("5");
        let graph = cg(&types, &pma_bonds(0));
        let ids: Vec<NodeId> = graph.node_ids().collect();
        let err = lib
            .map(&graph, &PMA_RULES)
            .expect_err("row 4 (type 5) has no candidate");
        assert_eq!(err, FragLibraryError::Unmapped { bead: ids[4] });
    }

    #[test]
    fn map_refuses_bonded_units_whose_ports_cannot_pair() {
        // Both units offer only `<`; `<`/`<` is not a pair.
        let lib = library(vec![("L", template(&[("L", &[PortKind::Left])], &[]))]);
        let graph = cg(&["1", "1"], &[(0, 1)]);
        let ids: Vec<NodeId> = graph.node_ids().collect();
        let err = lib
            .map(&graph, &[("1", "L")])
            .expect_err("no port pair accepts");
        // Units 0 (row 0) and 1 (row 1): `a` is the bond's bead in the lower
        // unit, `b` the one in the higher, each with its unit's template.
        assert_eq!(
            err,
            FragLibraryError::Unmatchable {
                a: ids[0],
                b: ids[1],
                template_a: "L".to_owned(),
                template_b: "L".to_owned(),
            }
        );
    }

    // -- map: assignment choice by port signature (amended step 5) -----------

    /// AA': two bonded beads both labelled `A`; only bead 0 carries ports,
    /// `<` (ordinal 0) then `>` (ordinal 1).
    fn aa_one_sided() -> Fragment {
        template(&[("A", LR), ("A", NONE)], &[(0, 1)])
    }

    /// Chain rows 0-1-2-3, all type "1". AA' occurs on {0,1}, {1,2}, {2,3};
    /// rows 0 and 3 force {0,1} (unit 0) and {2,3} (unit 1). The inter-unit
    /// bond 1-2 sits on unit 0's HIGHER row, so of unit 0's two port
    /// signatures only template bead 0 → row 1 gives that bead a port:
    /// sources [row 1, row 0]. Unit 1's bond sits on its lower row: [row 2,
    /// row 3]. Ports: row 1 offers `<`(0), `>`(1), row 2 the same; the first
    /// free accepting pair is `<`(0) with `>`(1).
    #[test]
    fn map_picks_the_assignment_whose_ports_carry_the_bonds() {
        let lib = library(vec![("AA1", aa_one_sided())]);
        let graph = cg(&["1", "1", "1", "1"], &[(0, 1), (2, 3), (1, 2)]);
        let ids: Vec<NodeId> = graph.node_ids().collect();

        let m = lib
            .map(&graph, &[("1", "A")])
            .expect("one port signature per unit carries its bond");

        assert_eq!(nodes(&m), vec!["AA1", "AA1"]);
        assert_eq!(m.sources(0), Some(&[ids[1], ids[0]][..]));
        assert_eq!(m.sources(1), Some(&[ids[2], ids[3]][..]));
        assert_eq!(m.graph().edges(), &[edge(0, 1, 0, 1)]);
    }

    /// AB: two bonded beads both labelled `A`, bead 0 carrying `<` (ordinal
    /// 0) and bead 1 carrying `>` (ordinal 1). C: one bead labelled `C`
    /// carrying `<` (0) and `>` (1).
    ///
    /// Coarse chain rows 0-1-2-3 with types "2","1","1","2". Every
    /// occurrence is forced: C on {0} and {3}, AB on {1,2}. AB's two
    /// label-identical maps over {1,2} have different port signatures — row
    /// 1 gets `<` and row 2 `>`, or the reverse — and each gives both rows
    /// one port for their one inter-unit bond, so both classes are feasible
    /// (and C's `<`/`>` would pair with either). Two feasible classes are a
    /// genuine ambiguity. The spec does not pin which bead is reported, so
    /// the test accepts either bead of the ambiguous unit.
    #[test]
    fn map_refuses_two_feasible_port_signatures() {
        let lib = library(vec![
            (
                "AB",
                template(
                    &[("A", &[PortKind::Left]), ("A", &[PortKind::Right])],
                    &[(0, 1)],
                ),
            ),
            ("C", template(&[("C", LR)], &[])),
        ]);
        let graph = cg(&["2", "1", "1", "2"], &[(0, 1), (1, 2), (2, 3)]);
        let ids: Vec<NodeId> = graph.node_ids().collect();

        let err = lib
            .map(&graph, &[("1", "A"), ("2", "C")])
            .expect_err("both port signatures of AB carry its bonds");

        assert!(
            matches!(
                err,
                FragLibraryError::AmbiguousAssignment { bead } if bead == ids[1] || bead == ids[2]
            ),
            "{err:?}"
        );
    }
}
