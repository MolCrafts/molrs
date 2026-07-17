//! antechamber's second pass: 2-colouring the conjugated systems.
//!
//! An `ATOMTYPE_GFF*.DEF` row can only ever emit the **phase-1** name of a
//! conjugated system — `cc`, `ce`, `cg`, `nc`, `ne`, `pc`, `pe`. The names
//! antechamber answers for the other half — `cd`, `cf`, `ch`, `nd`, `nf`, `pd`,
//! `pf` — appear in no `.DEF` row at all, so no amount of rule matching can
//! produce them. antechamber assigns them in a pass the `.DEF` does not describe,
//! and this module is that pass:
//!
//! > 2-colour the conjugated subgraph. A **single** bond keeps the letter, a
//! > **multiple** bond flips it. Seed each component at its phase-1 name, in atom
//! > order. An atom in no conjugated system keeps its phase-1 name.
//!
//! The pass owns only the *colouring*. The names it colours with are data — the
//! rule's [`alternate`](AtdRule::alternate), which the generator reads out of
//! `PARMCHK.DAT`'s `equivalent_flag` column — so the engine stays table-generic:
//! it never spells a GAFF type, and a table whose rules carry no `alternate` (BCC,
//! ABCG2, GAS, AMBER, SYBYL) passes through untouched.
//!
//! # Two things this pass is NOT
//!
//! **Not positional.** It is tempting to alternate around the ring — every other
//! atom gets the partner — and that reproduces pyrrole, furan and thiophene. It is
//! still wrong: 2-pyridone is `cc cd cd cc`, because the bond between its two
//! middle carbons is *single* and the letter is kept across it. The colouring
//! follows BOND ORDER, never ring position.
//!
//! **Not gated on aromaticity.** The subgraph is the *conjugated* one, not the
//! aromatic one. 1,4-benzoquinone has no aromatic bond whatsoever and is still
//! `o c cc cd c o cc cd`; hexatriene is an open chain and is still `c2 ce ce cf cf
//! c2`. Conversely a pure aromatic ring never reaches the pass at all — naphthalene
//! is ten `ca`, biphenyl is `cp cp` — because `ca` and `cp` carry no alternate.

use std::collections::VecDeque;

use molrs::AtomId;

use super::facts::MolFacts;
use crate::ff::params::AtdRule;

/// The final atom type of every atom, in `atom_ids` order.
///
/// `assigned[i]` is the rule that matched `atom_ids[i]` — its phase-1 answer.
/// Atoms whose rule carries an [`alternate`](AtdRule::alternate) form the
/// conjugated subgraph; each of its connected components is 2-coloured, and every
/// atom on the far colour is renamed to its own rule's alternate.
///
/// Note "its **own** rule's": the two colours are not two names but two *phases*.
/// Vinylacetylene (`C=C-C#C`) is one component holding a `ce` and a `cg`, and
/// antechamber answers `ce cg` — same phase, different rules — where an
/// implementation that propagated a name instead of a colour would answer `ce cf`.
pub(super) fn resolve_types(
    atom_ids: &[AtomId],
    assigned: &[&'static AtdRule],
    facts: &MolFacts,
) -> Vec<&'static str> {
    let n = atom_ids.len();
    // `phase_2[i]` — is atom i on the far colour of its component?
    let mut phase_2 = vec![false; n];
    let mut coloured = vec![false; n];

    // Seeding in atom order is load-bearing, not an implementation detail: the
    // seed is what fixes which half of the component keeps the phase-1 name.
    // Seeding pyrrole at its third carbon instead of its first answers
    // `cd cd cc na cc` — the same 2-colouring, the wrong way round.
    for seed in 0..n {
        if coloured[seed] || assigned[seed].alternate.is_none() {
            continue;
        }
        coloured[seed] = true;
        let mut queue = VecDeque::from([seed]);
        while let Some(i) = queue.pop_front() {
            for (neighbor, bond_type, _) in &facts.neighbors[i] {
                let Ok(j) = facts.index_of(*neighbor) else {
                    continue;
                };
                // Only atoms that HAVE an alternate are in the conjugated
                // subgraph. The carbonyl `c` of 2-pyridone bridges two of its
                // carbons in the molecular graph but not in this one, which is
                // what splits benzoquinone into two components of two.
                if coloured[j] || assigned[j].alternate.is_none() {
                    continue;
                }
                coloured[j] = true;
                phase_2[j] = phase_2[i] ^ flips_colour(*bond_type);
                queue.push_back(j);
            }
        }
    }

    (0..n)
        .map(|i| match (phase_2[i], assigned[i].alternate) {
            (true, Some(alternate)) => alternate,
            _ => assigned[i].atom_type,
        })
        .collect()
}

/// Does walking across this bond flip the colour?
///
/// The two phases of a conjugated system alternate across a **multiple** bond —
/// the π bond is what joins them — and are kept across a single one. So:
///
/// * **flips** — a double bond (2), a triple bond (3), and an *aromatic* double
///   (8). Aromatic doubles are doubles: the pass walks the Kekulé structure
///   antechamber perceives, which is why pyrrole comes back `cc cc cd na cd`.
///   Triple bonds flip exactly like doubles — diphenylacetylene is `cg ch` across
///   its C≡C, and hexa-1,5-dien-3-yne is `c2 ce cg ch cf c2`.
/// * **keeps** — a single bond (1), an aromatic single (7), and a delocalized
///   bond (9), which `MolFacts` already counts as a single. Keeping the letter
///   across the single bond *inside* a component is what makes 2-pyridone
///   `cc cd cd cc` rather than a positional `cc cd cc cd`.
///
/// Type 10 (the unresolved aromatic precursor) keeps the letter too: it carries no
/// Kekulé phase to alternate on. It should never survive perception — bond typing
/// resolves every aromatic bond to 7 or 8 — and a 10 that did reach here is an
/// unkekulized ring, where flipping would be a guess rather than a fact.
fn flips_colour(bond_type: i32) -> bool {
    matches!(bond_type, 2 | 3 | 8)
}
