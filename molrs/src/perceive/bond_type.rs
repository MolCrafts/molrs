//! BCC bond-type perception — the bond typing AM1-BCC is keyed on.
//!
//! AM1-BCC does not correct charges per *bond order*; it corrects them per **BCC
//! bond type**, a richer alphabet that distinguishes an aromatic single bond from
//! an aliphatic one and, crucially, marks *delocalized* bonds (a carboxylate's two
//! C–O bonds are neither "single" nor "double" — they are the same bond). Both the
//! `ATOMTYPE_BCC.DEF` atom-type rules (which count `sb`/`db`/`ab`/`DL` bonds) and
//! the `BCCPARM.DAT` correction table are keyed on it, so getting it wrong
//! silently corrupts every downstream charge.
//!
//! # The alphabet
//!
//! | Type | Meaning | Produced by |
//! |---|---|---|
//! | 1 | single | bond order 1 |
//! | 2 | double | bond order 2 |
//! | 3 | triple | bond order 3 |
//! | 6 | N–O/S on a **non-delocalized** N carrying a second terminal chalcogen (nitrite) | part 3 |
//! | 7 | aromatic single | part 1 — aromatic promotion |
//! | 8 | aromatic double | part 1 — aromatic promotion |
//! | 9 | delocalized (carboxylate, nitro, sulfonate, phosphate) | parts 2, 4 and 5 |
//!
//! The part numbering is antechamber's own (`bondtype.c::finalize()`), kept so the
//! rules here can be read against the source they came from.
//!
//! Two further values exist in the tables but are **never emitted**:
//!
//! * **10** is not a peer of 7/8 — it is the *unresolved* aromatic precursor (the
//!   SYBYL `ar` input token). It is an **input**, resolved here into 7 or 8, and
//!   never survives into the output.
//! * **11** occupies 26 same-type diagonal rows of `BCCPARM.DAT`, all with a
//!   correction of exactly `0.0000`, and no rule reaches it. Verified against
//!   AmberTools25 over a 33-molecule probe sweep: the emitted alphabet is
//!   `{1,2,3,6,7,8,9}` — never 10, never 11.
//!
//! # Aromatic promotion is not "is the bond aromatic?"
//!
//! A bond is promoted to 7/8 only when **both** endpoints are aromatic *and* they
//! share a ring of size **5 or 6 in which every ring atom is aromatic**. So
//! biphenyl's inter-ring bond and every bond of a 7-membered aromatic ring stay
//! 1/2 (part 1 of the perception).
//!
//! # Kekulé structures are re-perceived, not read
//!
//! An aromatic input carries no Kekulé structure (order 1.5), and the one an input
//! file *does* carry is not authoritative — AmberTools ignores it and re-derives
//! its own (verified: flipping a benzene SDF's Kekulé phase does not move its
//! output). So this module re-derives one too, by minimising the same
//! valence-state penalty (`APS.DAT`) over the aromatic subsystem.
//!
//! Which of 7/8 a given ring bond ends up with is *charge-invariant* — `BCCPARM`
//! stores identical corrections for types 7, 8 and 10 — but the **atom types** are
//! not: a heteroaromatic ring with two degenerate Kekulé structures (imidazolium)
//! puts the N–C double bond on a different nitrogen in each, and those two
//! nitrogens type differently. The tie-break is therefore load-bearing, and is
//! pinned to AmberTools by the antechamber oracle.
//!
//! # Provenance
//!
//! A reimplementation of the perception in AmberTools' `antechamber/bondtype.c`
//! (`finalize()` and the `conjatom[]` flags), written by reading that source with
//! the AmberTools developers' permission; see `.claude/notes/notes.md`
//! (2026-07-12) for the licensing posture. It is **not** a line-for-line port: two
//! defects are deliberately repaired, see below.
//!
//! # Type 6 — the order-dependence fix
//!
//! `bondtype.c`'s type-6 rule (`/*part3*/`) has two defects that make its output
//! depend on the order the bonds appear in the input file:
//!
//! 1. the neighbour scan's `break` is unbraced, so it stops after the *first*
//!    non-partner neighbour — the answer depends on `con[]` order;
//! 2. the mirrored branch (partner stored first) assigns `type = 6`
//!    *unconditionally*, above a loop whose result it then ignores.
//!
//! Measured consequences, against AmberTools25: **nitrate**'s two topologically
//! identical single-bonded O⁻ receive *different* types (6 and 9), splitting their
//! final charges by 0.28 e; and **pyridine-N-oxide**'s N–O bond types as 6 or 9
//! purely according to whether the file wrote that bond as `O-N` or `N-O`.
//!
//! This module repairs both: the neighbour scan is **exhaustive**, and the rule is
//! **symmetric in the bond's endpoints**. Everything else follows antechamber. The
//! well-behaved cases are unmoved (nitrite stays 6/6, nitromethane stays 9/9,
//! nitrobenzene 9/9, TMAO 9); the only outputs that move are the ones antechamber
//! itself cannot reproduce under a permutation of its own input.
//!
//! # The perceived type is a *perceived fact*, and lives in its own key
//!
//! The type is written to [`BCC_BOND_TYPE`] — never to the bond's [`keys::TYPE`],
//! which belongs to the **caller**: it is where a bond's force-field type *name*
//! (`c3-c3`) or a reader's LAMMPS bond-type id lives, and it is what `to_frame` puts
//! in the `bonds` block's `type` column for every bonded kernel to resolve its
//! parameters by. Two facts, two keys — a
//! component column is typed by its first write and molrs (deliberately) refuses to
//! coerce it, so an `i32` BCC code sitting in `type` makes the molecule unusable for
//! the force field that must later put a `String` name there.
//!
//! This is the same rule the charge models keep (`ff::charge`, ac-004): perception
//! neither reads nor writes `keys::TYPE`, so a molecule's own labels — GAFF names,
//! hostile LAMMPS ids, nothing at all — survive perception **byte-identical** and
//! cannot steer its answer.

use std::collections::HashMap;

use crate::perceive::aromaticity::perceive_aromaticity;
use crate::perceive::rings::find_rings;
use crate::store::keys;
use crate::system::atomistic::{AtomId, Atomistic, BondId};
use crate::system::bond::{BondNumber, BondType};
use crate::system::molgraph::PropValue;
use molrs::Element;

/// Bond prop holding the perceived BCC bond type, as an `i32` in
/// `{1, 2, 3, 6, 7, 8, 9}`.
///
/// Written by [`find_bond_types`] and read by everything keyed on the antechamber
/// alphabet — the `ATOMTYPE_*.DEF` rule engine
/// ([`AtdTypifier`](crate::ff::typifier::AtdTypifier)) and the `BCCPARM.DAT`
/// corrector ([`BCCCorrector`](crate::ff::typifier::am1bcc::BCCCorrector)).
///
/// Deliberately **not** [`keys::TYPE`]: that key is the caller's, and holds the
/// force-field type *name*. See the [module docs](self).
pub const BCC_BOND_TYPE: &str = "bcc_bond_type";

/// BCC bond type: a plain single bond.
const SINGLE: i32 = 1;
/// BCC bond type: a plain double bond.
const DOUBLE: i32 = 2;
/// BCC bond type: a triple bond.
const TRIPLE: i32 = 3;
/// BCC bond type: N–O/S on a non-delocalized nitrogen bearing a second terminal
/// chalcogen (nitrite, and only its kin).
const N_CHALCOGEN: i32 = 6;
/// BCC bond type: aromatic single.
const AROMATIC_SINGLE: i32 = 7;
/// BCC bond type: aromatic double.
const AROMATIC_DOUBLE: i32 = 8;
/// BCC bond type: delocalized (carboxylate, nitro, sulfonate, phosphate …).
const DELOCALIZED: i32 = 9;
/// BCC bond type: the *unresolved* aromatic precursor (SYBYL `ar`). Accepted as an
/// input marking, never emitted.
const AROMATIC_UNRESOLVED: i32 = 10;

/// A valence-state penalty that is never worth paying: used for the valences an
/// element's `APS.DAT` row leaves blank (`*`).
const FORBIDDEN: u32 = 1000;

/// Perceive the BCC bond type of every bond.
///
/// Graph in / graph out and **non-mutating**: `mol` is cloned, the clone's bonds
/// receive a [`BCC_BOND_TYPE`] prop holding the perceived type, and the clone is
/// returned. Bond `order` is *not* rewritten — the perceived Kekulé structure is
/// consumed internally and does not leak into the graph. Neither is the bond's
/// [`keys::TYPE`], which is the caller's (see the [module docs](self)).
///
/// The type is always (re)derived from structure: a [`BCC_BOND_TYPE`] already on
/// the input is read only as an aromaticity *hint* (7, 8 and 10 mark an aromatic
/// bond), never trusted as an answer. That is what resolves the unresolved
/// aromatic precursor, type 10, into 7 or 8. To bypass perception entirely and
/// supply your own types, drive [`crate::ff::typifier::am1bcc::BCCCorrector`]
/// directly.
///
/// Aromaticity is taken from the graph when it carries any aromatic marking
/// (a truthy `is_aromatic` bond prop, an `order` of 1.5, or a [`BCC_BOND_TYPE`] of
/// 7/8/10); when it carries none, [`perceive_aromaticity`] is run on the clone to
/// supply it.
///
/// # Arguments
///
/// * `mol` — the molecule to perceive; left untouched.
///
/// # Returns
///
/// A clone of `mol` whose every bond carries a [`BCC_BOND_TYPE`] prop in
/// `{1, 2, 3, 6, 7, 8, 9}`.
///
/// # Determinism
///
/// The result is invariant under any permutation of the input's bonds and under
/// swapping any bond's endpoints. It is keyed on atom order alone.
///
/// # Examples
///
/// ```
/// use molrs::Atomistic;
/// use molrs::perceive::Perceive;
/// use molrs::perceive::bond_type::BCC_BOND_TYPE;
/// use molrs::store::keys;
/// use molrs::system::bond::BondType;
///
/// // Acetate: both C–O bonds are delocalized, so both oxygens must correct
/// // identically.
/// let mut mol = Atomistic::new();
/// let c = mol.add_atom_xyz("C", 0.86, 0.12, 0.13);
/// let o1 = mol.add_atom_xyz("O", 1.18, 1.25, 0.59);
/// let o2 = mol.add_atom_xyz("O", 1.59, -0.86, -0.19);
/// let me = mol.add_atom_xyz("C", -0.63, -0.09, -0.09);
/// let b1 = mol.add_bond(c, o1).unwrap();
/// mol.set_bond_type(b1, BondType::Double).unwrap();
/// let b2 = mol.add_bond(c, o2).unwrap();
/// mol.add_bond(c, me).unwrap();
///
/// let typed = Perceive::new().find_bond_types(&mol);
/// let bond_type = |b| typed.get_bond(b).unwrap().props.get(BCC_BOND_TYPE).unwrap().as_f64();
/// assert_eq!(bond_type(b1), Some(9.0)); // was a double
/// assert_eq!(bond_type(b2), Some(9.0)); // was a single
/// ```
///
/// The caller's own bond labels are untouched — perception neither reads nor
/// writes [`keys::TYPE`], so a molecule already carrying force-field bond-type
/// *names* (a `String` column an `i32` could never share) survives it unchanged
/// and is still usable to build a force field:
///
/// ```
/// use molrs::Atomistic;
/// use molrs::perceive::Perceive;
/// use molrs::perceive::bond_type::BCC_BOND_TYPE;
/// use molrs::store::keys;
/// use molrs::system::bond::BondType;
/// use molrs::system::molgraph::PropValue;
///
/// let mut mol = Atomistic::new();
/// let c = mol.add_atom_xyz("C", 0.86, 0.12, 0.13);
/// let o1 = mol.add_atom_xyz("O", 1.18, 1.25, 0.59);
/// let o2 = mol.add_atom_xyz("O", 1.59, -0.86, -0.19);
/// let me = mol.add_atom_xyz("C", -0.63, -0.09, -0.09);
/// let b1 = mol.add_bond(c, o1).unwrap();
/// mol.set_bond_type(b1, BondType::Double).unwrap();
/// mol.set_bond_prop(b1, keys::TYPE, "c-o").unwrap(); // the caller's FF bond type NAME
/// let b2 = mol.add_bond(c, o2).unwrap();
/// mol.set_bond_prop(b2, keys::TYPE, "c-o").unwrap();
/// let b3 = mol.add_bond(c, me).unwrap();
/// mol.set_bond_prop(b3, keys::TYPE, "c-c3").unwrap();
///
/// let typed = Perceive::new().find_bond_types(&mol);
/// let props = |b| typed.get_bond(b).unwrap().props;
/// // The perceived fact went to its own key …
/// assert_eq!(props(b1).get(BCC_BOND_TYPE).unwrap().as_f64(), Some(9.0));
/// assert_eq!(props(b2).get(BCC_BOND_TYPE).unwrap().as_f64(), Some(9.0));
/// // … and the caller's name came through byte-identical.
/// let name = PropValue::Str("c-o".to_owned());
/// assert_eq!(props(b1).get(keys::TYPE), Some(&name));
/// assert_eq!(props(b2).get(keys::TYPE), Some(&name));
/// ```
pub fn find_bond_types(mol: &Atomistic) -> Atomistic {
    let mut out = mol.clone();
    if out.n_bonds() == 0 {
        return out;
    }
    if !has_aromatic_marking(&out) {
        let _ = perceive_aromaticity(&mut out);
    }

    let graph = BondGraph::new(&out);
    let types = graph.perceive();

    for (bid, ty) in graph.bond_ids.iter().zip(types) {
        let _ = out.set_bond_prop(*bid, BCC_BOND_TYPE, ty);
    }
    out
}

/// Assign a legal localized [`BondNumber`] to every aromatic bond, graph in /
/// graph out.
///
/// The non-mutating face of [`assign_kekule_numbers`], which carries the full
/// contract. It **only** kekulizes: it does not perceive aromaticity, so a
/// molecule whose aromatic bonds are not yet marked comes back unchanged. That
/// is the division of labour — perception decides *which* bonds are aromatic,
/// this decides *what phase* they take, and neither calls the other.
///
/// # Arguments
///
/// * `mol` — the molecule to kekulize; left untouched.
///
/// # Returns
///
/// A clone of `mol` whose aromatic bonds carry a legal localized number, or an
/// unchanged clone when no legal assignment exists.
pub fn find_kekule_orders(mol: &Atomistic) -> Atomistic {
    let mut out = mol.clone();
    assign_kekule_numbers(&mut out);
    out
}

/// Assign a legal localized [`BondNumber`] to every aromatic bond.
///
/// This is the Kekulé standardization the standard defines: **ensure every
/// aromatic subgraph carries a legal localized integer**. It is *in place* and
/// deliberately narrow — see [`Perceive::find_kekule_orders`] for the
/// graph-in/graph-out face.
///
/// The guarantees, and where each comes from:
///
/// * **Local** — only bonds whose class is `Aromatic` are written. A
///   non-aromatic bond is not read for a decision and not touched.
/// * **Conservative** — an aromatic bond that already states a number keeps it
///   when the whole system is already consistent; nothing is renumbered for its
///   own sake.
/// * **Complete** — on success every aromatic bond has `Single` or `Double`.
/// * **Deterministic** — degenerate assignments are real (benzene has two), so
///   the tie-break is part of the answer. It is keyed on **atom** order alone,
///   so permuting bonds or swapping endpoints cannot move it.
/// * **Transactional** — a system with no legal assignment leaves *no* bond
///   changed, rather than a ring half-assigned. Returns `false` in that case.
///
/// Returns whether every aromatic bond came out with a legal number.
pub fn assign_kekule_numbers(mol: &mut Atomistic) -> bool {
    if mol.n_bonds() == 0 {
        return true;
    }
    let graph = BondGraph::new(mol);
    if !graph.aromatic.iter().any(|a| *a) {
        return true;
    }

    // Conservative: a legal assignment already on the graph is *the* answer.
    // Re-deriving one would rewrite the phase an input stated for itself, so a
    // file that round-trips would come back with its bonds renumbered.
    let stated: Vec<bool> = graph
        .bond_ids
        .iter()
        .enumerate()
        .map(|(k, bid)| graph.aromatic[k] && mol.bond_number(*bid) == BondNumber::Double)
        .collect();
    let all_stated = graph
        .bond_ids
        .iter()
        .enumerate()
        .filter(|(k, _)| graph.aromatic[*k])
        .all(|(_, bid)| mol.bond_number(*bid) != BondNumber::Unknown);
    if all_stated && valences_are_satisfiable(&graph, &stated) {
        return true;
    }

    let doubles = kekulize(&graph);

    // Transactionality: decide first, write second. An aromatic atom that ends
    // up in no double bond and had no lone pair to give is the signal that no
    // legal assignment exists — the search returns its least-bad answer, and
    // only a completeness check can tell the two apart.
    let mut assignment: Vec<(BondId, BondNumber)> = Vec::new();
    for (k, bid) in graph.bond_ids.iter().enumerate() {
        if !graph.aromatic[k] {
            continue;
        }
        assignment.push((
            *bid,
            if doubles[k] {
                BondNumber::Double
            } else {
                BondNumber::Single
            },
        ));
    }
    if !valences_are_satisfiable(&graph, &doubles) {
        return false;
    }

    for (bid, number) in assignment {
        let _ = mol.set_bond_prop(bid, keys::BOND_NUMBER, number);
    }
    true
}

/// Does every aromatic atom end this assignment with a satisfiable valence?
///
/// The kekulizer minimises a valence-state penalty rather than failing, so a
/// ring with no legal assignment still comes back with *an* answer. This is the
/// check that tells "the best assignment" from "a legal one": an aromatic atom
/// whose element demands a π bond, that got none, and whose penalty says the
/// resulting valence is forbidden.
fn valences_are_satisfiable(graph: &BondGraph, doubles: &[bool]) -> bool {
    // Same totals the search scored against: implicit hydrogens count toward
    // both, or this guard passes a structure the penalty would have rejected.
    let mut valence: Vec<i32> = graph.implicit_h.iter().map(|h| *h as i32).collect();
    let mut degree: Vec<usize> = graph.implicit_h.iter().map(|h| *h as usize).collect();
    for (k, (i, j)) in graph.ends.iter().copied().enumerate() {
        let w = if graph.aromatic[k] {
            if doubles[k] { 2 } else { 1 }
        } else {
            (graph.order[k].round() as i32).clamp(SINGLE, TRIPLE)
        };
        valence[i] += w;
        valence[j] += w;
        degree[i] += 1;
        degree[j] += 1;
    }
    graph
        .z
        .iter()
        .enumerate()
        .filter(|(i, _)| graph.aromatic_atom[*i])
        .all(|(i, _)| valence_penalty(graph.scored_as_z[i], degree[i], valence[i]) < FORBIDDEN)
}

/// Does any bond already claim to be aromatic?
///
/// Perception is only run when the answer is no; a graph that already carries
/// aromatic markings keeps them, so a caller's own aromaticity model wins over
/// ours.
fn has_aromatic_marking(mol: &Atomistic) -> bool {
    mol.bonds().any(|(_, bond)| aromatic_marking(&bond.props))
}

/// Is this bond marked aromatic by any of the three markings we accept: an
/// `is_aromatic` prop, an `order` of 1.5, or a [`BCC_BOND_TYPE`] of 7/8/10 (the
/// aromatic single, the aromatic double, and the unresolved `ar` precursor)?
///
/// The last of the three is what makes perception idempotent, and what resolves a
/// caller-supplied type 10 into 7 or 8. It reads **our own** key: the bond's
/// `keys::TYPE` is the caller's, and a LAMMPS bond-type id that happened to be 7
/// must not make a bond aromatic.
fn aromatic_marking(props: &HashMap<String, PropValue>) -> bool {
    BondType::from_prop(props.get(keys::BOND_TYPE)).is_aromatic()
        || props
            .get(BCC_BOND_TYPE)
            .and_then(PropValue::as_f64)
            .is_some_and(|t| {
                let t = t.round() as i32;
                matches!(t, AROMATIC_SINGLE | AROMATIC_DOUBLE | AROMATIC_UNRESOLVED)
            })
}

/// The molecule flattened into the dense index space the rules are written in:
/// atom index `i` is the `i`-th atom of [`Atomistic::atoms`], bond index `k` the
/// `k`-th bond of [`Atomistic::bonds`].
///
/// antechamber's rules read `atomicnum` and `connum` (the *degree*, hydrogens
/// included) off exactly this shape, so building it once keeps the rule bodies a
/// faithful transcription rather than a translation.
struct BondGraph {
    /// Per bond: its handle, in bond index order.
    bond_ids: Vec<BondId>,
    /// Per bond: its two endpoint atom indices.
    ends: Vec<(usize, usize)>,
    /// Per bond: the input bond order.
    order: Vec<f64>,
    /// Per bond: whether the input marks it aromatic.
    aromatic: Vec<bool>,
    /// Per atom: atomic number (0 when the element is unknown).
    z: Vec<u8>,
    /// Per atom: hydrogens implied by valence but not drawn in the graph.
    implicit_h: Vec<u32>,
    /// Per atom: the element the π-demand penalty scores it as — the real
    /// element for a neutral atom, its isoelectronic neighbour for a charged
    /// one. Read *only* by the aromatic assignment's penalty.
    scored_as_z: Vec<u8>,
    /// Per atom: neighbour atom indices.
    adj: Vec<Vec<usize>>,
    /// Per atom: whether it carries an aromatic bond.
    aromatic_atom: Vec<bool>,
    /// The SSSR rings, as atom indices.
    rings: Vec<Vec<usize>>,
}

impl BondGraph {
    /// Flatten a molecule. Atoms whose element is unknown get `z = 0` and so match
    /// no rule; they fall through to their plain bond order.
    fn new(mol: &Atomistic) -> Self {
        let atom_ids: Vec<AtomId> = mol.atoms().map(|(aid, _)| aid).collect();
        let index: HashMap<AtomId, usize> = atom_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(i, aid)| (aid, i))
            .collect();

        let z: Vec<u8> = atom_ids
            .iter()
            .map(|aid| {
                mol.get_atom(*aid)
                    .ok()
                    .and_then(|atom| atom.get_str(keys::ELEMENT).and_then(Element::by_symbol))
                    .map_or(0, |el| el.z())
            })
            .collect();

        // The element `APS.DAT` is *scored* as, for the π-demand search only.
        //
        // A formal charge changes how many bonds an atom wants, and the penalty
        // table is keyed on the element alone, so a charged atom is scored as
        // its isoelectronic neighbour: a carbocation like boron, a carbanion
        // like nitrogen. Without it tropylium's C+ is handed a double bond and
        // reaches valence 4, which the table accepts.
        //
        // This is a **scoring proxy inside the aromatic assignment**, nothing
        // more. It never leaves this struct, `z` above stays the real atomic
        // number, and no general valence or element semantics change. It should
        // become an explicit charged-aromatic-atom rule; scoring by proxy is
        // the shortest correct step, not the final shape.
        let implicit_h: Vec<u32> = atom_ids
            .iter()
            .map(|aid| crate::perceive::hydrogens::implicit_h_count(mol, *aid).unwrap_or(0))
            .collect();

        let scored_as_z: Vec<u8> = atom_ids
            .iter()
            .zip(&z)
            .map(|(aid, &zi)| {
                let charge = mol
                    .get_atom(*aid)
                    .ok()
                    .and_then(|atom| atom.get("formal_charge").and_then(PropValue::as_f64))
                    .map_or(0i32, |c| c.round() as i32);
                let shifted = i32::from(zi) - charge;
                if zi == 0 || shifted <= 0 {
                    zi
                } else {
                    shifted.min(i32::from(u8::MAX)) as u8
                }
            })
            .collect();

        let n = atom_ids.len();
        let mut bond_ids = Vec::new();
        let mut ends = Vec::new();
        let mut order = Vec::new();
        let mut aromatic = Vec::new();
        let mut adj = vec![Vec::new(); n];
        let mut aromatic_atom = vec![false; n];

        for (bid, bond) in mol.bonds() {
            let (Some(&i), Some(&j)) = (index.get(&bond.nodes[0]), index.get(&bond.nodes[1]))
            else {
                continue;
            };
            let is_aromatic = aromatic_marking(&bond.props);
            // An aromatic bond may not have a number yet — that is what this
            // module is about to decide — so it enters the valence sum as the
            // one bond every bond is at least.
            let o = BondNumber::from_prop(bond.props.get(keys::BOND_NUMBER))
                .count()
                .max(1) as f64;

            adj[i].push(j);
            adj[j].push(i);
            if is_aromatic {
                aromatic_atom[i] = true;
                aromatic_atom[j] = true;
            }
            bond_ids.push(bid);
            ends.push((i, j));
            order.push(o);
            aromatic.push(is_aromatic);
        }

        let ring_info = find_rings(mol);
        let rings = ring_info
            .rings()
            .iter()
            .map(|ring| {
                ring.iter()
                    .filter_map(|aid| index.get(aid).copied())
                    .collect()
            })
            .collect();

        Self {
            bond_ids,
            ends,
            order,
            aromatic,
            z,
            implicit_h,
            scored_as_z,
            adj,
            aromatic_atom,
            rings,
        }
    }

    /// Number of bonds on an atom — antechamber's `connum`. Hydrogens count.
    /// Drawn neighbours — antechamber's `connum`.
    ///
    /// Deliberately *not* including implicit hydrogens: the `ATOMTYPE_*.DEF`
    /// rules are transcribed against this count, and an acetate written without
    /// its formal charge relies on the unfilled valence to read as a terminal
    /// oxygen rather than a hydroxyl.
    fn degree(&self, atom: usize) -> usize {
        self.adj[atom].len()
    }

    /// Drawn neighbours **plus** the hydrogens the graph implies but does not
    /// draw — the count `valence_penalty` must see.
    ///
    /// The penalty is keyed on `(element, degree, valence)`, so a hydrogen the
    /// graph never drew is a bond the table never sees. Without it a
    /// pyrrole-type N and a pyridine-type N are the same atom to it — degree 2,
    /// valence 2 — imidazole's two candidate Kekulé structures tie, and the
    /// tie-break puts the double on the N already carrying a hydrogen.
    fn total_degree(&self, atom: usize) -> usize {
        self.adj[atom].len() + self.implicit_h[atom] as usize
    }

    /// A terminal (one-bonded) oxygen or sulfur — the chalcogen every
    /// delocalization rule is written around.
    fn is_terminal_chalcogen(&self, atom: usize) -> bool {
        self.degree(atom) == 1 && matches!(self.z[atom], 8 | 16)
    }

    /// How many terminal O/S an atom carries.
    fn terminal_chalcogens(&self, atom: usize) -> usize {
        self.adj[atom]
            .iter()
            .filter(|nb| self.is_terminal_chalcogen(**nb))
            .count()
    }

    /// Run the whole perception, returning one BCC type per bond in bond order.
    fn perceive(&self) -> Vec<i32> {
        let mut types = self.seed_types();
        let conjugated = self.conjugated_atoms();
        self.promote_aromatic(&mut types);

        for k in 0..self.ends.len() {
            if self.delocalize_double(k, &mut types, &conjugated) {
                continue;
            }
            if self.type_n_chalcogen(k, &mut types, &conjugated) {
                continue;
            }
            self.delocalize_single(k, &mut types);
        }
        self.delocalize_hypervalent(&mut types);
        types
    }

    /// Seed every bond from its (Kekulé) bond order: 1/2/3 → 1/2/3.
    ///
    /// Aromatic bonds have no Kekulé order of their own, so they take the one
    /// [`kekulize`] derives. A non-integral, non-aromatic order (which no chemistry
    /// produces) rounds to its nearest neighbour rather than failing the whole
    /// molecule.
    fn seed_types(&self) -> Vec<i32> {
        let doubles = kekulize(self);
        self.ends
            .iter()
            .enumerate()
            .map(|(k, _)| {
                if self.aromatic[k] {
                    if doubles[k] { DOUBLE } else { SINGLE }
                } else {
                    (self.order[k].round() as i32).clamp(SINGLE, TRIPLE)
                }
            })
            .collect()
    }

    /// The `conjatom[]` flag: atoms whose bonds to terminal chalcogens are
    /// resonance-averaged rather than localized.
    ///
    /// A carboxylate carbon, a phosphate phosphorus, a sulfonate sulfur and a nitro
    /// nitrogen — each identified purely by degree and by how many terminal O/S it
    /// carries. (The thresholds are antechamber's, including its comment that
    /// `S` needs *three* terminal chalcogens so that SO₂'s S=O bonds are **not**
    /// delocalized.)
    fn conjugated_atoms(&self) -> Vec<bool> {
        (0..self.z.len())
            .map(|a| {
                let chalcogens = self.terminal_chalcogens(a);
                match (self.z[a], self.degree(a)) {
                    (6, 3) => chalcogens >= 2,  // carboxylate / thiocarboxylate C
                    (15, 4) => chalcogens >= 2, // phosphate P
                    (16, 4) => chalcogens >= 3, // sulfonate S — not SO2
                    (7, 3) => chalcogens >= 2,  // nitro / nitrate N
                    _ => false,
                }
            })
            .collect()
    }

    /// **Part 1 — aromatic promotion.** 1 → 7 and 2 → 8, but only inside a genuine
    /// aromatic ring.
    ///
    /// The boundary is deliberately narrow: both endpoints must be aromatic **and**
    /// share a ring of size 5 or 6 **every** atom of which is aromatic. Biphenyl's
    /// inter-ring bond joins two aromatic atoms but lies in no ring, and a
    /// 7-membered aromatic ring is out of range — neither is promoted.
    fn promote_aromatic(&self, types: &mut [i32]) {
        let aromatic_rings: Vec<&Vec<usize>> = self
            .rings
            .iter()
            .filter(|ring| {
                matches!(ring.len(), 5 | 6) && ring.iter().all(|a| self.aromatic_atom[*a])
            })
            .collect();

        for (k, (i, j)) in self.ends.iter().copied().enumerate() {
            if !(self.aromatic_atom[i] && self.aromatic_atom[j]) {
                continue;
            }
            let shared = aromatic_rings
                .iter()
                .any(|ring| ring.contains(&i) && ring.contains(&j));
            if !shared {
                continue;
            }
            if types[k] == SINGLE {
                types[k] = AROMATIC_SINGLE;
            } else if types[k] == DOUBLE {
                types[k] = AROMATIC_DOUBLE;
            }
        }
    }

    /// **Part 2 — delocalized double.** A double bond from a conjugated centre to a
    /// terminal O/S is delocalized: the carboxylate C=O, the nitro N=O.
    ///
    /// Returns whether the bond was claimed.
    fn delocalize_double(&self, k: usize, types: &mut [i32], conjugated: &[bool]) -> bool {
        if types[k] != DOUBLE {
            return false;
        }
        let (i, j) = self.ends[k];
        let claimed = (conjugated[i] && self.is_terminal_chalcogen(j))
            || (conjugated[j] && self.is_terminal_chalcogen(i));
        if claimed {
            types[k] = DELOCALIZED;
        }
        claimed
    }

    /// **Part 3 — type 6.** A bond from a nitrogen of degree 2 or 3 to a terminal
    /// O/S, where that nitrogen carries a *second* terminal chalcogen: nitrite, and
    /// only its kin.
    ///
    /// Two deliberate repairs to `bondtype.c` (see the [module docs](self)):
    ///
    /// * the second-chalcogen scan is **exhaustive**, not "first non-partner
    ///   neighbour only" — antechamber's `break` is outside its `if`, which makes
    ///   the answer depend on the order the bonds were listed;
    /// * the rule is **symmetric in the endpoints** — antechamber's mirrored branch
    ///   assigns 6 unconditionally, so the same molecule types differently
    ///   depending on whether the file wrote `O-N` or `N-O`.
    ///
    /// The conjugated gate is what keeps the repair from over-firing: a *nitro*
    /// nitrogen also carries a second terminal chalcogen, but its bonds are
    /// delocalized (type 9), which is exactly what `conjatom[]` says and what parts
    /// 2 and 4 deliver. Without the gate, nitromethane's N–O would become 6 — a
    /// 0.28 e error on an oracle molecule.
    ///
    /// Returns whether the bond was claimed.
    fn type_n_chalcogen(&self, k: usize, types: &mut [i32], conjugated: &[bool]) -> bool {
        let (i, j) = self.ends[k];
        let Some((n, chalcogen)) = self.nitrogen_chalcogen_ends(i, j) else {
            return false;
        };
        if conjugated[n] {
            return false;
        }
        let second = self.adj[n]
            .iter()
            .any(|nb| *nb != chalcogen && self.is_terminal_chalcogen(*nb));
        if second {
            types[k] = N_CHALCOGEN;
        }
        second
    }

    /// Orient a bond as `(nitrogen, terminal chalcogen)` if it is one, in whichever
    /// order its endpoints were stored.
    fn nitrogen_chalcogen_ends(&self, i: usize, j: usize) -> Option<(usize, usize)> {
        let is_n = |a: usize| self.z[a] == 7 && matches!(self.degree(a), 2 | 3);
        if is_n(i) && self.is_terminal_chalcogen(j) {
            Some((i, j))
        } else if is_n(j) && self.is_terminal_chalcogen(i) {
            Some((j, i))
        } else {
            None
        }
    }

    /// **Part 4 — delocalized single.** A single bond to a terminal O/S is
    /// delocalized: the carboxylate C–O⁻, the nitro N–O⁻.
    ///
    /// This is the partner of part 2, and together they are what make the type-9
    /// answer *resonance-invariant*: whichever of a carboxylate's two C–O bonds the
    /// Kekulé structure happened to call the double one, both come out 9.
    fn delocalize_single(&self, k: usize, types: &mut [i32]) {
        if types[k] != SINGLE {
            return;
        }
        let (i, j) = self.ends[k];
        if self.is_terminal_chalcogen(i) || self.is_terminal_chalcogen(j) {
            types[k] = DELOCALIZED;
        }
    }

    /// **Part 5 — hypervalent centres.** A sulfur of degree 3 with exactly two
    /// terminal chalcogens (a sulfone/sulfonamide S), and a phosphorus of degree 4
    /// with at least two (a phosphate P), delocalize *their* bonds to those
    /// chalcogens — whatever type the earlier parts gave them.
    fn delocalize_hypervalent(&self, types: &mut [i32]) {
        for a in 0..self.z.len() {
            let chalcogens = self.terminal_chalcogens(a);
            let hypervalent = match (self.z[a], self.degree(a)) {
                (16, 3) => chalcogens == 2,
                (15, 4) => chalcogens >= 2,
                _ => false,
            };
            if !hypervalent {
                continue;
            }
            for (k, (i, j)) in self.ends.iter().copied().enumerate() {
                let other = if i == a {
                    j
                } else if j == a {
                    i
                } else {
                    continue;
                };
                if self.is_terminal_chalcogen(other) {
                    types[k] = DELOCALIZED;
                }
            }
        }
    }
}

/// Derive a Kekulé structure for the aromatic subsystem: which aromatic bonds are
/// the double ones.
///
/// The aromatic bonds are the only unknowns — every other bond keeps its input
/// order — so this is a **matching** problem: each aromatic atom either takes one
/// double bond or none, and no two doubles may share an atom. Among the legal
/// matchings we minimise the total *valence-state penalty*, the same objective
/// AmberTools minimises (`APS.DAT`, transcribed in [`valence_penalty`]). That is
/// what makes pyridine's N take a double while pyrrole's N does not, and what keeps
/// thiophene's S out of the matching.
///
/// # Tie-break
///
/// Degenerate minima are real (benzene has two; imidazolium has two that type its
/// two nitrogens differently), so the tie-break is part of the answer, not an
/// implementation detail. Aromatic bonds are visited ordered by
/// `(min endpoint asc, max endpoint desc)`, doubles tried before singles, and the
/// first minimum-penalty solution wins. This is calibrated to reproduce
/// AmberTools25 on every aromatic molecule of the antechamber oracle
/// (benzene, toluene, phenol, aniline, fluorobenzene, pyridine, imidazole,
/// imidazolium, thiophene), and — unlike antechamber's own search — it depends on
/// **atom** order only, so permuting the input's bonds cannot move it.
fn kekulize(graph: &BondGraph) -> Vec<bool> {
    let n_bonds = graph.ends.len();
    let mut doubles = vec![false; n_bonds];

    let mut aromatic: Vec<usize> = (0..n_bonds).filter(|k| graph.aromatic[*k]).collect();
    if aromatic.is_empty() {
        return doubles;
    }
    aromatic.sort_by_key(|k| {
        let (i, j) = graph.ends[*k];
        (i.min(j), std::cmp::Reverse(i.max(j)))
    });

    // sigma[a]: the bond-order sum already committed at atom `a` — every
    // non-aromatic bond at its input order, plus one per aromatic bond (each is at
    // least single). A matched atom ends one higher.
    // Seeded with the implicit hydrogens: a bond the graph never drew is still
    // a bond the atom's valence has spent, and `valence_penalty` reads the
    // total.
    let mut sigma: Vec<i32> = graph.implicit_h.iter().map(|h| *h as i32).collect();
    for (k, (i, j)) in graph.ends.iter().copied().enumerate() {
        let w = if graph.aromatic[k] {
            1
        } else {
            (graph.order[k].round() as i32).clamp(SINGLE, TRIPLE)
        };
        sigma[i] += w;
        sigma[j] += w;
    }

    // Atoms touched by the search, and how many of their aromatic bonds are still
    // undecided — an atom's penalty is payable as soon as that count hits zero.
    let mut pending = vec![0usize; graph.z.len()];
    for k in &aromatic {
        let (i, j) = graph.ends[*k];
        pending[i] += 1;
        pending[j] += 1;
    }

    let mut search = Kekule {
        graph,
        aromatic: &aromatic,
        sigma,
        pending,
        matched: vec![false; graph.z.len()],
        best: None,
        best_doubles: Vec::new(),
        chosen: Vec::new(),
    };
    search.walk(0, 0);

    for k in search.best_doubles {
        doubles[k] = true;
    }
    doubles
}

/// The backtracking state of [`kekulize`].
struct Kekule<'a> {
    /// The molecule being kekulized.
    graph: &'a BondGraph,
    /// Aromatic bond indices, in visit order.
    aromatic: &'a [usize],
    /// Per atom: bond-order sum already committed (see [`kekulize`]).
    sigma: Vec<i32>,
    /// Per atom: aromatic bonds not yet decided.
    pending: Vec<usize>,
    /// Per atom: already carries its double bond.
    matched: Vec<bool>,
    /// Penalty of the best complete assignment so far.
    best: Option<u32>,
    /// The double bonds of that assignment.
    best_doubles: Vec<usize>,
    /// The double bonds of the assignment under construction.
    chosen: Vec<usize>,
}

impl Kekule<'_> {
    /// Visit aromatic bond `pos`, having already paid `penalty`.
    ///
    /// An atom's valence penalty is charged the moment its last aromatic bond is
    /// decided, which lets a branch be abandoned as soon as it can no longer beat
    /// the incumbent. Pruning on `>=` (not `>`) is what makes the first
    /// minimum-penalty solution the winner and keeps the tie-break deterministic.
    fn walk(&mut self, pos: usize, penalty: u32) {
        if self.best.is_some_and(|best| penalty >= best) {
            return;
        }
        let Some(&k) = self.aromatic.get(pos) else {
            self.best = Some(penalty);
            self.best_doubles = self.chosen.clone();
            return;
        };
        let (i, j) = self.graph.ends[k];

        for as_double in [true, false] {
            if as_double && (self.matched[i] || self.matched[j]) {
                continue;
            }
            if as_double {
                self.matched[i] = true;
                self.matched[j] = true;
                self.chosen.push(k);
            }
            self.pending[i] -= 1;
            self.pending[j] -= 1;
            let settled = self.settle(i) + self.settle(j);

            self.walk(pos + 1, penalty.saturating_add(settled));

            self.pending[i] += 1;
            self.pending[j] += 1;
            if as_double {
                self.matched[i] = false;
                self.matched[j] = false;
                self.chosen.pop();
            }
        }
    }

    /// The penalty an atom owes now that a bond of its has been decided: zero until
    /// its last aromatic bond is placed, then the cost of the valence it landed on.
    fn settle(&self, atom: usize) -> u32 {
        if self.pending[atom] > 0 {
            return 0;
        }
        let valence = self.sigma[atom] + i32::from(self.matched[atom]);
        // `total_degree`, not `degree`: `sigma` is seeded with the implicit
        // hydrogens, so the degree handed to the same penalty must count them
        // too. The acceptance guard already scores this way — scoring the
        // search differently lets it pick a structure the guard then rejects.
        valence_penalty(
            self.graph.scored_as_z[atom],
            self.graph.total_degree(atom),
            valence,
        )
    }
}

/// The penalty of an atom sitting at a given total valence — AmberTools' `APS.DAT`,
/// restricted to the elements that can carry an aromatic bond.
///
/// The numbers are the table's, not ours: they are what rank pyrrole's
/// three-valent N (0) above its four-valent alternative (1), and what forbid
/// thiophene's S from reaching valence 3 (64). `FORBIDDEN` stands in for the
/// table's `*` — a valence the element does not have.
///
/// Only the generic rows are transcribed. The context-specific rows (`APSNO2`,
/// `APSN3+`, …) re-rank valences for centres that are *already* decided by their
/// substituents, and cannot change which Kekulé structure wins for any ring.
fn valence_penalty(z: u8, degree: usize, valence: i32) -> u32 {
    let row: &[(i32, u32)] = match (z, degree) {
        (6, 1) => &[(3, 1), (4, 0), (5, 32)],
        (6, _) => &[(2, 64), (3, 32), (4, 0), (5, 32), (6, 64)],
        (7, 1) => &[(2, 3), (3, 0), (4, 32)],
        (7, 2) => &[(2, 4), (3, 0), (4, 2)],
        (7, 3) => &[(2, 32), (3, 0), (4, 1), (5, 2)],
        (7, _) => &[(3, 64), (4, 0), (5, 64)],
        (8, 1) => &[(1, 1), (2, 0), (3, 64)],
        (8, _) => &[(1, 32), (2, 0), (3, 64)],
        (15, 1) => &[(2, 2), (3, 0), (4, 32)],
        (15, 2) => &[(2, 4), (3, 0), (4, 2)],
        (15, 3) => &[(2, 32), (3, 0), (4, 1), (5, 2)],
        (15, _) => &[(3, 64), (4, 1), (5, 0), (6, 32)],
        (16, 1 | 2) => &[(1, 2), (2, 0), (3, 64)],
        (16, 3) => &[(3, 1), (4, 0), (5, 2), (6, 2)],
        (16, _) => &[(4, 4), (5, 2), (6, 0)],
        // Everything else (H, halogens, metals, unknown elements) has one valence
        // and nothing to choose; it never constrains a matching.
        _ => return 0,
    };
    row.iter()
        .find(|(v, _)| *v == valence)
        .map_or(FORBIDDEN, |(_, penalty)| *penalty)
}

#[cfg(all(test, feature = "smiles"))]
mod tests {
    use super::*;
    use crate::io::smiles::{parse_smiles, to_atomistic};

    /// Imidazole exactly as the antechamber oracle states it (case `imidazole`,
    /// SMILES `c1cnc[nH]1`): ring C0 C1 N2 C3 N4 with **implicit** hydrogens —
    /// N2 carries none (pyridine-type), N4 carries one (pyrrole-type).
    ///
    /// The hydrogens must stay implicit: drawn explicitly, `total_degree` and
    /// `degree` agree and the tie-break under test cannot be reached.
    fn imidazole() -> Atomistic {
        to_atomistic(&parse_smiles("c1cnc[nH]1").expect("parse")).expect("to_atomistic")
    }

    /// Ring bonds as `((i, j), is_double)`, endpoints low-high, sorted.
    fn ring_doubles(mol: &Atomistic) -> Vec<((usize, usize), bool)> {
        let index: HashMap<AtomId, usize> = mol
            .atoms()
            .enumerate()
            .map(|(position, (id, _))| (id, position))
            .collect();
        let mut out: Vec<((usize, usize), bool)> = mol
            .bonds()
            .map(|(id, _)| {
                let (a, b) = mol.bond_endpoints(id).expect("endpoints");
                let (a, b) = (index[&a], index[&b]);
                (
                    (a.min(b), a.max(b)),
                    mol.bond_number(id) == BondNumber::Double,
                )
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn the_fixture_keeps_its_hydrogens_implicit() {
        // Guards the two tests below: were `to_atomistic` to start drawing
        // hydrogens, `total_degree` would equal `degree` and they would pass
        // against either implementation without testing anything.
        let mol = imidazole();
        assert_eq!(mol.n_atoms(), 5, "hydrogens must not be drawn");
        let implicit: Vec<u32> = mol
            .atoms()
            .map(|(id, _)| crate::perceive::hydrogens::implicit_h_count(&mol, id).unwrap_or(0))
            .collect();
        assert_eq!(
            implicit,
            vec![1, 1, 0, 1, 1],
            "N2 pyridine-type, N4 pyrrole-type"
        );
    }

    #[test]
    fn the_search_and_the_acceptance_guard_count_degree_the_same_way() {
        // The guard in `kekule_is_legal` seeds its degree with the implicit
        // hydrogens ("implicit hydrogens count toward both, or this guard
        // passes a structure the penalty would have rejected"). `settle` scores
        // the search against the same penalty, so it must count them too —
        // that is what `total_degree` is, and wiring it is what keeps the two
        // sides of that comment true.
        //
        // Both feed `valence_penalty`, whose rows differ by degree: a
        // pyrrole-type N reads row (7, 2) on the drawn count and (7, 3) on the
        // total, which price valence 2 at 4 vs 32 and valence 5 at
        // FORBIDDEN vs 2.
        let mol = imidazole();
        let graph = BondGraph::new(&mol);
        let pyrrole_n = 4;
        assert_eq!(graph.degree(pyrrole_n), 2, "drawn neighbours only");
        assert_eq!(
            graph.total_degree(pyrrole_n),
            3,
            "drawn neighbours plus the implicit hydrogen"
        );
        assert_ne!(
            valence_penalty(7, graph.degree(pyrrole_n), 2),
            valence_penalty(7, graph.total_degree(pyrrole_n), 2),
            "the two counts must reach different penalty rows, or this is moot"
        );
    }

    #[test]
    fn imidazole_kekulizes_the_way_antechamber_does() {
        let mol = find_kekule_orders(&imidazole());
        // AmberTools25 oracle `bcc_bond_types`: (0,1) and (2,3) are type 8
        // (aromatic double); (1,2), (3,4), (4,0) are type 7 (aromatic single).
        assert_eq!(
            ring_doubles(&mol),
            vec![
                ((0, 1), true),
                ((0, 4), false),
                ((1, 2), false),
                ((2, 3), true),
                ((3, 4), false),
            ],
        );
    }

    #[test]
    fn the_pyrrole_nitrogen_takes_no_double() {
        let mol = find_kekule_orders(&imidazole());
        // N4 already spends a valence on an undrawn hydrogen, so a double here
        // makes it five-valent. This is what `BondGraph::total_degree` exists
        // for: with the drawn-neighbour `degree`, N4 and N2 both look like
        // (degree 2, valence 2) to `valence_penalty`, the two Kekule structures
        // tie, and the double lands on the nitrogen carrying the hydrogen.
        for ((i, j), is_double) in ring_doubles(&mol) {
            if i == 4 || j == 4 {
                assert!(!is_double, "double placed on the pyrrole N: ({i}, {j})");
            }
        }
    }
}
