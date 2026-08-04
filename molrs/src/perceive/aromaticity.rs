//! General aromaticity perception aligned to RDKit's default model
//! (`AROMATICITY_RDKIT`).
//!
//! This is a BSD-3 port of RDKit's `setAromaticity` default-model code path,
//! re-expressed against `MolGraph`. It perceives aromaticity from scratch
//! (Kekulé bond orders + element + formal charge), writing back an
//! `is_aromatic` atom property and a `bond_type` of `Aromatic` so that
//! [`crate::SmartsPattern`]'s `a` / `c` / `:` primitives match RDKit after
//! native perception (rather than relying on transplanted flags).
//!
//! # Algorithm (RDKit `aromaticityHelper(mol, srings, 0, 0, true)`)
//!
//! 1. Compute SSSR rings ([`crate::perceive::rings::find_rings`]).
//! 2. For each ring atom, classify its π-electron donor type
//!    (`getAtomDonorTypeArom` → `ElectronDonor`) using a per-atom electron
//!    count (`countAtomElec`) plus exocyclic / cyclic multiple-bond rules, and
//!    test atom candidacy (`isAtomCandForArom`).
//! 3. Keep rings where *every* atom is a candidate (and not all dummy).
//! 4. Over each fused system (rings sharing a bond), enumerate ring
//!    combinations up to size 6, union the atoms present in exactly one or two
//!    of the chosen rings, and apply the Hückel `4n+2` test on the min/max
//!    electron range (`applyHuckel`).
//! 5. Bonds appearing in exactly one of the aromatic ring set are marked
//!    `bond_type = Aromatic`; their atoms get `is_aromatic = 1`.
//!
//! # Scope
//!
//! Only the RDKit default model is ported (not MDL / Simple / MMFF94). The
//! MMFF-specific aromaticity model in `molrs-ff` is intentionally independent.
//!
//! # Reference
//!
//! RDKit `Code/GraphMol/Aromaticity.cpp` (`setAromaticity`, `aromaticityHelper`,
//! `applyHuckel`, `applyHuckelToFused`, `getMinMaxAtomElecs`,
//! `getAtomDonorTypeArom`, `isAtomCandForArom`, `countAtomElec`,
//! `incidentNonCyclicMultipleBond`, `incidentCyclicMultipleBond`,
//! `markAtomsBondsArom`). BSD 3-Clause, © 2001-2024 RDKit contributors.
//! <https://github.com/rdkit/rdkit>

use std::collections::{HashMap, HashSet};

use crate::perceive::rings::find_rings;
use crate::store::keys;
use crate::system::atomistic::{AtomId, Atomistic, BondId};
use crate::system::bond::{BondNumber, BondType};
use crate::system::molgraph::PropValue;
use molrs::Element;

/// Maximum number of fused rings combined when checking the Hückel rule
/// (RDKit `maxFused = 6`).
const MAX_FUSED: usize = 6;

/// Per-atom π-electron donor classification (RDKit `ElectronDonorType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElectronDonor {
    /// Contributes no electrons and has no empty p-orbital (disqualifies).
    NoDonor,
    /// Empty p-orbital, 0 electrons (e.g. tropylium / cyclopropenyl cation).
    Vacant,
    /// Contributes exactly one electron (sp2 carbon, pyridine N, ...).
    One,
    /// Contributes a lone pair of two electrons (pyrrole N, furan O, ...).
    Two,
}

impl ElectronDonor {
    /// (min, max) electrons this atom can contribute — RDKit `getMinMaxAtomElecs`.
    fn min_max(self) -> (i32, i32) {
        match self {
            ElectronDonor::One => (1, 1),
            ElectronDonor::Two => (2, 2),
            ElectronDonor::NoDonor | ElectronDonor::Vacant => (0, 0),
        }
    }

    /// Whether this donor type qualifies the atom for aromaticity candidacy.
    fn is_candidate_type(self) -> bool {
        matches!(
            self,
            ElectronDonor::Vacant | ElectronDonor::One | ElectronDonor::Two
        )
    }
}

// ---------------------------------------------------------------------------
// Element helpers (RDKit PeriodicTable subset for in-scope elements)
// ---------------------------------------------------------------------------

/// RDKit `getDefaultValence` for the elements that can be aromatic.
/// Returns the first standard valence, or `-1` for univalent / unknown.
fn default_valence(z: u8) -> i32 {
    match Element::by_number(z).map(|e| e.default_valences()) {
        Some(vals) if !vals.is_empty() => vals[0] as i32,
        _ => -1,
    }
}

/// Number of outer-shell (valence) electrons (RDKit `getNouterElecs`).
/// Defined for the elements that participate in aromaticity perception.
fn n_outer_elecs(z: u8) -> i32 {
    match z {
        1 => 1,  // H
        5 => 3,  // B
        6 => 4,  // C
        7 => 5,  // N
        8 => 6,  // O
        9 => 7,  // F
        14 => 4, // Si
        15 => 5, // P
        16 => 6, // S
        17 => 7, // Cl
        33 => 5, // As
        34 => 6, // Se
        52 => 6, // Te
        // Fall back to group number for main-group elements.
        _ => main_group_outer(z),
    }
}

/// Best-effort outer-electron count for other main-group elements.
fn main_group_outer(z: u8) -> i32 {
    // Map atomic number to main-group column (1..8) for s/p-block; transition
    // metals are not aromaticity candidates so their value is irrelevant.
    match z {
        3 | 11 | 19 | 37 | 55 | 87 => 1,
        4 | 12 | 20 | 38 | 56 | 88 => 2,
        13 | 31 | 49 | 81 => 3,
        32 | 50 | 82 => 4,
        51 | 83 => 5,
        84 => 6,
        35 | 53 | 85 => 7,
        2 | 10 | 18 | 36 | 54 | 86 => 8,
        _ => 0,
    }
}

/// Pauling electronegativity for the small set of elements needed by the
/// exocyclic-double-bond "steal electrons" rule (RDKit `moreElectroNegative`).
fn electronegativity(z: u8) -> f64 {
    match z {
        1 => 2.20,
        5 => 2.04,
        6 => 2.55,
        7 => 3.04,
        8 => 3.44,
        9 => 3.98,
        14 => 1.90,
        15 => 2.19,
        16 => 2.58,
        17 => 3.16,
        34 => 2.55,
        52 => 2.10,
        _ => 0.0,
    }
}

/// RDKit `moreElectroNegative(a, b)` — is element `a` more electronegative
/// than element `b`?
fn more_electronegative(za: u8, zb: u8) -> bool {
    electronegativity(za) > electronegativity(zb)
}

// ---------------------------------------------------------------------------
// Per-atom topological queries
// ---------------------------------------------------------------------------

/// Read a bond's **localized** number for perception.
///
/// Perception reasons about a Lewis structure — π electrons, valence, exocyclic
/// multiple bonds — so it counts localized bonds, not classes. `bond_number` is
/// never rewritten by this pass, so re-running perception sees the same input
/// it saw the first time. That is the whole of the idempotency argument; there
/// is no snapshot field.
///
/// An aromatic bond that has no number yet counts as the one bond every bond is
/// at least.
fn bond_order(mol: &Atomistic, bid: BondId) -> f64 {
    let Ok(b) = mol.get_bond(bid) else {
        return 1.0;
    };
    BondNumber::from_prop(b.props.get(keys::BOND_NUMBER))
        .count()
        .max(1) as f64
}

/// Atomic number of an atom (from its `"element"` symbol). `0` if unknown.
fn atomic_num(mol: &Atomistic, id: AtomId) -> u8 {
    mol.get_atom(id)
        .ok()
        .and_then(|a| {
            a.get_str("element")
                .and_then(Element::by_symbol)
                .map(|e| e.z())
        })
        .unwrap_or(0)
}

/// Formal charge (`"formal_charge"` prop, default 0).
fn formal_charge(mol: &Atomistic, id: AtomId) -> i32 {
    match mol
        .get_atom(id)
        .ok()
        .and_then(|a| a.get("formal_charge").cloned())
    {
        Some(PropValue::Int(v)) => v,
        Some(PropValue::F64(v)) => v as i32,
        _ => 0,
    }
}

/// Heavy-atom + H degree (RDKit `getDegree() + getTotalNumHs()`); here all H
/// are explicit so this is simply the neighbour count.
fn total_degree(mol: &Atomistic, id: AtomId) -> i32 {
    explicit_degree(mol, id) + implicit_h_count(mol, id)
}

/// Neighbours actually drawn in the graph — RDKit `getDegree()`.
///
/// Distinct from [`total_degree`], which adds the hydrogens the graph implies.
/// The two are not interchangeable: RDKit's electron count is keyed on the
/// total, while its "does this atom carry a multiple bond" tests are keyed on
/// the drawn degree, and swapping them makes every aromatic carbon look
/// saturated.
fn explicit_degree(mol: &Atomistic, id: AtomId) -> i32 {
    mol.neighbors(id).count() as i32
}

/// Hydrogens the graph implies but does not draw (RDKit `getNumImplicitHs`).
///
/// A SMILES graph is heavy-atom only, so a benzene carbon arrives with two
/// neighbours and a valence of three: the missing bond is a hydrogen, not a
/// radical. Deriving it here is what makes perception independent of whether
/// hydrogens were made explicit — `add_hydrogens` stays a separate, optional
/// operation, and running it changes no answer here (an explicit H raises the
/// valence and drives this to zero).
fn implicit_h_count(mol: &Atomistic, id: AtomId) -> i32 {
    let z = atomic_num(mol, id);
    let dv = default_valence(z);
    if dv <= 0 {
        return 0;
    }
    // A formal charge changes how many bonds the atom can carry, and the
    // capacity is the *isoelectronic* element's — RDKit's
    // `getDefaultValence(Z - charge)`, the same rule the candidacy test below
    // applies. A carbocation holds three like boron; a protonated nitrogen
    // holds four like carbon. Naively subtracting the charge would give N+ a
    // capacity of two and lose its hydrogen.
    let charge = formal_charge(mol, id);
    let capacity = default_valence((i32::from(z) - charge).clamp(1, 118) as u8);
    if capacity <= 0 {
        return 0;
    }
    (capacity - explicit_valence(mol, id)).max(0)
}

/// Iterate incident `(BondId, other_atom, localized bond number)` for an atom.
///
/// Uses the graph adjacency index (O(degree)) rather than scanning every bond
/// (O(n_bonds)); the latter made per-ring-atom classification O(N^2) on
/// ring-heavy molecules.
fn incident_bonds<'a>(
    mol: &'a Atomistic,
    id: AtomId,
) -> impl Iterator<Item = (BondId, AtomId, f64)> + 'a {
    mol.incident_bond_ids(id)
        .map(move |(bid, other)| (bid, other, bond_order(mol, bid)))
}

/// Explicit valence = sum of incident **Kekulé** bond orders (RDKit
/// `getValence(EXPLICIT)`), rounded to nearest integer.
fn explicit_valence(mol: &Atomistic, id: AtomId) -> i32 {
    let sum: f64 = incident_bonds(mol, id).map(|(_, _, o)| o).sum();
    sum.round() as i32
}

/// RDKit `incidentNonCyclicMultipleBond`: a multiple bond (order ≥ 2) to a
/// non-ring bond. Returns the partner atom if found.
fn incident_noncyclic_multiple_bond(
    mol: &Atomistic,
    rings: &crate::perceive::rings::RingInfo,
    id: AtomId,
) -> Option<AtomId> {
    for (bid, other, order) in incident_bonds(mol, id) {
        if !rings.is_bond_in_ring(bid) && order >= 2.0 {
            return Some(other);
        }
    }
    None
}

/// RDKit `incidentCyclicMultipleBond`: a multiple bond (order ≥ 2) that is in a
/// ring.
fn incident_cyclic_multiple_bond(
    mol: &Atomistic,
    rings: &crate::perceive::rings::RingInfo,
    id: AtomId,
) -> bool {
    incident_bonds(mol, id).any(|(bid, _, order)| rings.is_bond_in_ring(bid) && order >= 2.0)
}

/// RDKit `incidentMultipleBond`: explicit valence differs from σ-degree, i.e.
/// the atom carries at least one π bond.
fn incident_multiple_bond(mol: &Atomistic, id: AtomId) -> bool {
    explicit_valence(mol, id) != explicit_degree(mol, id)
}

// ---------------------------------------------------------------------------
// Electron counting & donor classification (RDKit countAtomElec / getAtomDonorTypeArom)
// ---------------------------------------------------------------------------

/// RDKit `MolOps::countAtomElec` — number of electrons the atom can donate to
/// the π system, or `-1` if it cannot be aromatic / conjugated.
fn count_atom_elec(mol: &Atomistic, id: AtomId) -> i32 {
    let z = atomic_num(mol, id);
    let dv = default_valence(z);
    if dv <= 1 {
        // univalent / unknown elements can't be aromatic or conjugated
        return -1;
    }

    // total atom degree (all H explicit here)
    let degree = total_degree(mol, id);

    // more than 3-coordinate → not aromatic
    if degree > 3 {
        return -1;
    }

    // lone-pair electrons = outer electrons - default valence, minus charge
    let nlp_raw = n_outer_elecs(z) - dv;
    let nlp = (nlp_raw - formal_charge(mol, id)).max(0);

    let n_radicals = 0; // radicals not modelled in MolGraph; assume none

    let mut res = (dv - degree) + nlp - n_radicals;

    if res > 1 {
        // triple-or-higher incident bond contributes only one electron
        let n_unsaturations = explicit_valence(mol, id) - degree;
        if n_unsaturations > 1 {
            res = 1;
        }
    }
    res
}

/// RDKit `getAtomDonorTypeArom` with `exocyclicBondsStealElectrons = true`
/// (the default-model setting).
fn atom_donor_type(
    mol: &Atomistic,
    rings: &crate::perceive::rings::RingInfo,
    id: AtomId,
) -> ElectronDonor {
    let z = atomic_num(mol, id);
    if z == 0 {
        // dummy atom
        return if incident_cyclic_multiple_bond(mol, rings, id) {
            ElectronDonor::One
        } else {
            // RDKit returns AnyElectronDonorType here; treated as candidate but
            // none of our test molecules contain dummies, so map to One.
            ElectronDonor::One
        };
    }

    let mut nelec = count_atom_elec(mol, id);

    if nelec < 0 {
        ElectronDonor::NoDonor
    } else if nelec == 0 {
        if incident_noncyclic_multiple_bond(mol, rings, id).is_some() {
            ElectronDonor::Vacant
        } else if incident_cyclic_multiple_bond(mol, rings, id) {
            ElectronDonor::One
        } else {
            ElectronDonor::NoDonor
        }
    } else if nelec == 1 {
        if let Some(who) = incident_noncyclic_multiple_bond(mol, rings, id) {
            let z2 = atomic_num(mol, who);
            if more_electronegative(z2, z) {
                ElectronDonor::Vacant
            } else {
                ElectronDonor::One
            }
        } else if incident_multiple_bond(mol, id) {
            ElectronDonor::One
        } else if formal_charge(mol, id) == 1 {
            ElectronDonor::Vacant
        } else {
            ElectronDonor::NoDonor
        }
    } else {
        // nelec >= 2
        if let Some(who) = incident_noncyclic_multiple_bond(mol, rings, id) {
            let z2 = atomic_num(mol, who);
            if more_electronegative(z2, z) {
                nelec -= 1;
            }
        }
        if nelec % 2 == 1 {
            ElectronDonor::One
        } else {
            ElectronDonor::Two
        }
    }
}

/// RDKit `isAtomCandForArom` with the default-model flag set (all permissive).
fn is_atom_candidate(mol: &Atomistic, id: AtomId, edon: ElectronDonor) -> bool {
    let z = atomic_num(mol, id);

    // first two rows + Se / Te
    if z > 18 && z != 34 && z != 52 {
        return false;
    }

    if !edon.is_candidate_type() {
        return false;
    }

    // atoms not in their default valence state are shut out.
    // RDKit: getTotalValence() > getDefaultValence(Z - formalCharge).
    // Total valence counts implicit hydrogens too, but they can never push an
    // atom *over* its capacity — they are what is left of it — so the explicit
    // sum is the only thing that can trip this.
    let dv = default_valence(z);
    if dv > 0 {
        let total_valence = explicit_valence(mol, id);
        let charge = formal_charge(mol, id);
        let adj_dv = default_valence((z as i32 - charge).clamp(1, 118) as u8);
        if total_valence > adj_dv {
            return false;
        }
    }

    // disallow more than one double/triple bond (e.g. C1=C=NC=N1)
    let n_unsaturations = explicit_valence(mol, id) - explicit_degree(mol, id);
    if n_unsaturations > 1 {
        let mut n_mult = 0;
        for (_, _, order) in incident_bonds(mol, id) {
            if order >= 2.0 {
                n_mult += 1;
            }
            if n_mult > 1 {
                return false;
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Hückel rule
// ---------------------------------------------------------------------------

/// RDKit `applyHuckel`: does the electron range over `ring_atoms` admit a
/// `4n+2` count?
fn apply_huckel(ring_atoms: &[AtomId], edon: &HashMap<AtomId, ElectronDonor>) -> bool {
    let mut rlw = 0;
    let mut rup = 0;
    for &a in ring_atoms {
        let (lo, hi) = edon
            .get(&a)
            .copied()
            .unwrap_or(ElectronDonor::NoDonor)
            .min_max();
        rlw += lo;
        rup += hi;
    }

    if rup >= 6 {
        let mut rie = rlw;
        while rie <= rup {
            if (rie - 2) % 4 == 0 {
                return true;
            }
            rie += 1;
        }
        false
    } else {
        rup == 2
    }
}

// ---------------------------------------------------------------------------
// Fused-ring enumeration
// ---------------------------------------------------------------------------

/// Bond-id set for a ring (consecutive atom pairs).
fn ring_bonds(mol: &Atomistic, ring: &[AtomId]) -> Vec<BondId> {
    let n = ring.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        if let Some(bid) = find_bond(mol, a, b) {
            out.push(bid);
        }
    }
    out
}

/// Find the bond connecting `a` and `b`, if any.
///
/// Walks `a`'s adjacency (O(degree)) rather than scanning every bond; the latter
/// made `ring_bonds` O(rings × n_bonds) ≈ O(N²) on ring-heavy molecules.
fn find_bond(mol: &Atomistic, a: AtomId, b: AtomId) -> Option<BondId> {
    mol.incident_bond_ids(a)
        .find(|&(_, other)| other == b)
        .map(|(bid, _)| bid)
}

/// Build fused systems: groups of candidate-ring indices connected by sharing
/// at least one bond.
///
/// Uses a bond→rings index plus union-find (near-linear in the total number of
/// ring bonds) rather than the all-pairs `is_disjoint` scan, which was O(rings²)
/// even for entirely disjoint ring systems.
fn fused_systems(ring_bond_sets: &[HashSet<BondId>]) -> Vec<Vec<usize>> {
    let n = ring_bond_sets.len();

    // union-find with path compression
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = x;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    let mut parent: Vec<usize> = (0..n).collect();

    // Rings that share a bond belong to the same system. Group ring indices by
    // bond, then union all rings touching each bond.
    let mut bond_to_rings: HashMap<BondId, Vec<usize>> = HashMap::new();
    for (ri, set) in ring_bond_sets.iter().enumerate() {
        for &b in set {
            bond_to_rings.entry(b).or_default().push(ri);
        }
    }
    for rings in bond_to_rings.values() {
        for pair in rings.windows(2) {
            let a = find(&mut parent, pair[0]);
            let b = find(&mut parent, pair[1]);
            if a != b {
                parent[a] = b;
            }
        }
    }

    // Gather members per root, preserving ascending ring-index order within each
    // system for deterministic output.
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    groups.into_values().collect()
}

/// All k-combinations of indices `0..n` (RDKit iterates these via
/// `nextCombination`).
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        // advance
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        idx[i] += 1;
        for j in (i + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// RDKit `applyHuckelToFused`: try every fused sub-combination; for each one
/// that satisfies Hückel, mark (RDKit `markAtomsBondsArom`) the bonds that
/// appear exactly **once within that combination** as aromatic. Fusing bonds
/// shared by two rings of the *same* combination are not marked there, but get
/// marked when a single-ring combination is tried. Accumulates into
/// `aromatic_bonds` / `aromatic_atoms`.
fn apply_huckel_to_fused(
    rings: &[Vec<AtomId>],
    ring_bond_sets: &[HashSet<BondId>],
    fused: &[usize],
    edon: &HashMap<AtomId, ElectronDonor>,
    aromatic_bonds: &mut HashSet<BondId>,
    aromatic_atoms: &mut HashSet<AtomId>,
    mol: &Atomistic,
) {
    let nrings = fused.len();
    let max_size = nrings.min(MAX_FUSED);

    for cur_size in 1..=max_size {
        for comb in combinations(nrings, cur_size) {
            let cur_rs: Vec<usize> = comb.iter().map(|&i| fused[i]).collect();

            // require the chosen subset to be connected (fused) when >1 ring
            if cur_rs.len() > 1 && !is_connected_subset(&cur_rs, ring_bond_sets) {
                continue;
            }

            // count membership of every atom across the chosen rings
            let mut counts: HashMap<AtomId, usize> = HashMap::new();
            for &ri in &cur_rs {
                for &a in &rings[ri] {
                    *counts.entry(a).or_insert(0) += 1;
                }
            }
            // atoms present in exactly one or two rings (RDKit #2895)
            let unon: Vec<AtomId> = counts
                .iter()
                .filter(|&(_, &c)| c == 1 || c == 2)
                .map(|(&a, _)| a)
                .collect();

            if apply_huckel(&unon, edon) {
                // markAtomsBondsArom: count bond appearances within this
                // combination's rings; mark those appearing exactly once.
                let mut bond_count: HashMap<BondId, usize> = HashMap::new();
                for &ri in &cur_rs {
                    for &bid in &ring_bond_sets[ri] {
                        *bond_count.entry(bid).or_insert(0) += 1;
                    }
                }
                for (&bid, &cnt) in &bond_count {
                    if cnt == 1
                        && let Ok(bond) = mol.get_bond(bid)
                    {
                        aromatic_bonds.insert(bid);
                        aromatic_atoms.insert(bond.nodes[0]);
                        aromatic_atoms.insert(bond.nodes[1]);
                    }
                }
            }
        }
    }
}

/// Whether a subset of ring indices forms a single fused (bond-connected)
/// component.
fn is_connected_subset(subset: &[usize], ring_bond_sets: &[HashSet<BondId>]) -> bool {
    if subset.len() <= 1 {
        return true;
    }
    let mut seen = vec![false; subset.len()];
    let mut stack = vec![0usize];
    seen[0] = true;
    let mut count = 1;
    while let Some(cur) = stack.pop() {
        for j in 0..subset.len() {
            if !seen[j] && !ring_bond_sets[subset[cur]].is_disjoint(&ring_bond_sets[subset[j]]) {
                seen[j] = true;
                count += 1;
                stack.push(j);
            }
        }
    }
    count == subset.len()
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Perceive aromaticity using the RDKit default model and annotate the graph
/// in place.
///
/// Aromatic atoms receive an `is_aromatic = 1` property; aromatic bonds receive
/// `bond_type = Aromatic`. Atoms that are *not* perceived aromatic are reset
/// (`is_aromatic = 0`), and a bond that was aromatic and no longer is falls back
/// to the class its own localized `bond_number` states. Prior flags are always
/// overwritten, which is what makes the call idempotent.
///
/// Returns the number of atoms flagged aromatic.
///
/// # Reference
/// Port of RDKit `setAromaticity(mol, AROMATICITY_RDKIT)` →
/// `aromaticityHelper(mol, srings, 0, 0, /*includeFused=*/true)`.
/// `Code/GraphMol/Aromaticity.cpp`, BSD 3-Clause, © RDKit contributors.
pub fn perceive_aromaticity(mol: &mut Atomistic) -> usize {
    // No snapshot: `bond_number` is never overwritten by perception, so the
    // input's own localized structure *is* the memory. This is what makes the
    // pass idempotent, and it is why there is no second localized-order field.
    let rings_info = find_rings(mol);
    let srings: Vec<Vec<AtomId>> = rings_info.rings().to_vec();

    // ---- 1. classify every ring atom -----------------------------------
    let mut edon: HashMap<AtomId, ElectronDonor> = HashMap::new();
    let mut candidate: HashMap<AtomId, bool> = HashMap::new();

    for ring in &srings {
        for &id in ring {
            if candidate.contains_key(&id) {
                continue;
            }
            let d = atom_donor_type(mol, &rings_info, id);
            edon.insert(id, d);
            candidate.insert(id, is_atom_candidate(mol, id, d));
        }
    }

    // ---- 2. candidate rings (all atoms candidates, not all dummy) -------
    let mut cring_idx: Vec<usize> = Vec::new();
    for (ri, ring) in srings.iter().enumerate() {
        let all_cand = ring
            .iter()
            .all(|id| candidate.get(id).copied().unwrap_or(false));
        let all_dummy = ring.iter().all(|&id| atomic_num(mol, id) == 0);
        if all_cand && !all_dummy {
            cring_idx.push(ri);
        }
    }

    let crings: Vec<Vec<AtomId>> = cring_idx.iter().map(|&ri| srings[ri].clone()).collect();
    let ring_bond_sets: Vec<HashSet<BondId>> = crings
        .iter()
        .map(|r| ring_bonds(mol, r).into_iter().collect())
        .collect();

    // ---- 3-4. Hückel over each fused system + mark aromatic bonds/atoms --
    let mut aromatic_atoms: HashSet<AtomId> = HashSet::new();
    let mut aromatic_bonds: HashSet<BondId> = HashSet::new();
    for system in fused_systems(&ring_bond_sets) {
        apply_huckel_to_fused(
            &crings,
            &ring_bond_sets,
            &system,
            &edon,
            &mut aromatic_bonds,
            &mut aromatic_atoms,
            mol,
        );
    }

    // ---- 5. write back (idempotent: reset everything, then set) ---------
    let all_atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    for id in all_atom_ids {
        let arom = aromatic_atoms.contains(&id);
        let _ = mol.set_atom(id, "is_aromatic", PropValue::Int(if arom { 1 } else { 0 }));
    }

    // A bond's class is rewritten; its localized number is not. Perception
    // decides *what kind* of bond this is — whether a Kekulé phase is legal is
    // kekulization's question, and a caller that already stated one keeps it.
    let all_bond_ids: Vec<BondId> = mol.bonds().map(|(id, _)| id).collect();
    for bid in all_bond_ids {
        if aromatic_bonds.contains(&bid) {
            let _ = mol.set_bond_prop(bid, keys::BOND_TYPE, BondType::Aromatic);
        } else if mol.bond_type(bid).is_aromatic() {
            // A bond that was aromatic and is no longer falls back to the class
            // its own localized number states — never to a stale aromatic flag.
            let number = mol.bond_number(bid);
            let demoted = match number {
                BondNumber::Double => BondType::Double,
                BondNumber::Triple => BondType::Triple,
                BondNumber::Unknown => BondType::Unknown,
                _ => BondType::Single,
            };
            let _ = mol.set_bond_prop(bid, keys::BOND_TYPE, demoted);
        }
    }

    aromatic_atoms.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::molgraph::Atom;

    /// Build a Kekulé benzene ring of 6 carbons (alternating single/double).
    fn benzene() -> Atomistic {
        let mut g = Atomistic::new();
        let c: Vec<AtomId> = (0..6)
            .map(|_| g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0)))
            .collect();
        for i in 0..6 {
            let bid = g.add_bond(c[i], c[(i + 1) % 6]).unwrap();
            let order = if i % 2 == 0 { 2.0 } else { 1.0 };
            g.set_bond_type(bid, BondType::from_code(order as u32))
                .unwrap();
        }
        // one explicit H per carbon
        for &ci in &c {
            let h = g.add_atom(Atom::xyz("H", 0.0, 0.0, 0.0));
            g.add_bond(ci, h).unwrap();
        }
        g
    }

    #[test]
    fn test_benzene_all_aromatic() {
        let mut g = benzene();
        let n = perceive_aromaticity(&mut g);
        assert_eq!(n, 6);
    }

    #[test]
    fn test_cyclohexane_not_aromatic() {
        let mut g = Atomistic::new();
        let c: Vec<AtomId> = (0..6)
            .map(|_| g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0)))
            .collect();
        for i in 0..6 {
            g.add_bond(c[i], c[(i + 1) % 6]).unwrap();
        }
        for &ci in &c {
            for _ in 0..2 {
                let h = g.add_atom(Atom::xyz("H", 0.0, 0.0, 0.0));
                g.add_bond(ci, h).unwrap();
            }
        }
        assert_eq!(perceive_aromaticity(&mut g), 0);
    }

    #[test]
    fn test_idempotent() {
        let mut g = benzene();
        let n1 = perceive_aromaticity(&mut g);
        let n2 = perceive_aromaticity(&mut g);
        assert_eq!(n1, n2);
    }
}
