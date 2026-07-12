//! Testing one `ATD` row against one atom.
//!
//! A rule is a **conjunction**: every column it constrains must hold, and an
//! unconstrained column (`*` / `&` in the source) constrains nothing. Rules are
//! tried in file order and the first match wins, so the table's own ordering —
//! most specific first — is load-bearing and must never be sorted or dedup'd.

use molrs::AtomId;

use super::facts::MolFacts;
use crate::ff::params::{AtdRule, AtdTable};

/// The atom type of the first rule of `table` that matches `aid`.
///
/// `None` means the table declares no rule for this atom. The caller turns that
/// into an error rather than a fallback type: molrs assigns the type the table
/// names, or none at all.
pub(super) fn assign_atom_type(
    table: &AtdTable,
    aid: AtomId,
    facts: &MolFacts,
) -> Option<&'static str> {
    table
        .rules
        .iter()
        .find(|rule| rule_matches(rule, aid, facts))
        .map(|rule| rule.atom_type)
}

/// Test one pre-parsed `ATD` rule against an atom.
fn rule_matches(rule: &AtdRule, aid: AtomId, facts: &MolFacts) -> bool {
    let Ok(i) = facts.index_of(aid) else {
        return false;
    };
    if rule.residue != "*" && rule.residue != facts.residue[i].as_str() {
        return false;
    }
    if rule
        .atomic_number
        .is_some_and(|z| z != facts.atomic_number[i])
    {
        return false;
    }
    if rule.degree.is_some_and(|degree| degree != facts.degree[i]) {
        return false;
    }
    if rule
        .hydrogen_count
        .is_some_and(|h_count| h_count != facts.hydrogen_count[i])
    {
        return false;
    }
    if rule.ewd_count.is_some_and(|n| {
        facts
            .ewd_count_around_attachment(aid)
            .is_none_or(|actual| actual != n)
    }) {
        return false;
    }
    if let Some(prop) = rule.atom_property
        && !facts.atom_property_matches(aid, None, &prop)
    {
        return false;
    }
    if let Some(env) = rule.environment
        && !facts.environment_matches(aid, env, rule.environment_bonds)
    {
        return false;
    }
    true
}
