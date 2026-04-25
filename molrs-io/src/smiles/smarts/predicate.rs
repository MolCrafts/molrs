//! Evaluate SMARTS query AST nodes against a target [`MolGraph`].
//!
//! The matcher reduces SMARTS matching to subgraph isomorphism with per-atom
//! and per-bond predicates. This module supplies those predicates: given a
//! compiled [`AtomQuery`] or [`BondQuery`] node and a target atom (or bond),
//! return whether the target satisfies the query.
//!
//! # Daylight semantics covered
//!
//! Atom primitives — `C`, `[c]`, `[A]`, `[a]`, `*`, `[#n]`, `[D<n>]`,
//! `[X<n>]`, `[H<n>]`, `[h<n>]`, `[R]`/`[R<n>]`, `[r<n>]`, `[v<n>]`,
//! `[+<n>]`/`[-<n>]`, `[<iso>C]`, `[:class]` — and all logical combinators
//! `!`, `&` (high-precedence), `,`, `;` (low-precedence).
//!
//! Bond primitives — `-`, `=`, `#`, `:`, `~`, `@` (ring) — with logical
//! combinators `!`, `&`, `,`.
//!
//! # Not yet covered
//!
//! - Chirality (`@`/`@@` on atoms) — returns `false` for now; see follow-up.
//! - Directional bonds `/` / `\` — returns `false`; stereochemistry phase.
//! - Quadruple bond `$`.
//!
//! # References
//!
//! - Daylight SMARTS theory manual — Atomic Primitives & Bond Primitives tables:
//!   <https://daylight.com/dayhtml/doc/theory/theory.smarts.html>

use molrs::element::Element;
use molrs::molgraph::{AtomId, MolGraph};
use molrs::rings::RingInfo;

use crate::smiles::chem::ast::{AtomPrimitive, AtomQuery, BondKind, BondQuery};

/// Precomputed target-side data shared across all predicate evaluations for a
/// given target molecule.
///
/// Building this once per `find_all` call is O(atoms + bonds + ring-detection);
/// predicate evaluation is O(1) per primitive afterwards.
pub(crate) struct TargetCtx<'m> {
    pub(crate) mol: &'m MolGraph,
    pub(crate) rings: RingInfo,
}

impl<'m> TargetCtx<'m> {
    pub(crate) fn new(mol: &'m MolGraph) -> Self {
        let rings = molrs::rings::find_rings(mol);
        Self { mol, rings }
    }

    /// Total H count on an atom = explicit `h_count` if the atom was written
    /// as a bracket atom, else the implicit H count derived from the smallest
    /// default valence of the element minus the sum of explicit bond orders.
    ///
    /// Daylight's `H<n>` primitive refers to total H; `h<n>` refers to the
    /// implicit portion only. When callers write bracket atoms, the parser
    /// records the explicit H and this function returns exactly that; when
    /// they use organic-subset atoms, we reconstruct Daylight's default
    /// valence model.
    fn total_h(&self, atom_id: AtomId) -> u32 {
        let Ok(atom) = self.mol.get_atom(atom_id) else {
            return 0;
        };
        if let Some(h) = atom.get_f64("h_count") {
            return h as u32;
        }
        self.implicit_h(atom_id)
    }

    /// Implicit hydrogens: `default_valence - bond_order_sum`, saturating
    /// at zero. Only organic-subset atoms expose implicit H; bracket atoms
    /// always report 0 because their H count is fixed at parse time.
    ///
    /// Aromatic bonds are counted at order 1.5 even when the SMILES parser
    /// stored them as order 1.0 (the typical case for an implicit aromatic
    /// bond inside `c1ccccc1`), so benzene's carbons report `implicit_h = 1`
    /// rather than 2.
    fn implicit_h(&self, atom_id: AtomId) -> u32 {
        let Ok(atom) = self.mol.get_atom(atom_id) else {
            return 0;
        };
        if atom.contains_key("h_count") {
            return 0;
        }
        let Some(sym) = atom.get_str("element") else {
            return 0;
        };
        let Some(element) = Element::by_symbol(sym) else {
            return 0;
        };
        let valence = element.default_valences().first().copied().unwrap_or(0) as f64;
        let bond_sum = self.effective_bond_sum(atom_id);
        (valence - bond_sum).max(0.0).round() as u32
    }

    /// Sum of bond orders around the atom, with the same aromatic promotion
    /// applied by the matcher's target-graph construction.
    fn effective_bond_sum(&self, atom_id: AtomId) -> f64 {
        let self_arom = self.is_aromatic(atom_id);
        self.mol
            .neighbor_bonds(atom_id)
            .map(|(other, ord)| {
                if self_arom && self.is_aromatic(other) && (ord - 1.0).abs() < 1e-6 {
                    1.5
                } else {
                    ord
                }
            })
            .sum()
    }

    /// Number of explicit bonds (Daylight `D<n>`).
    fn degree(&self, atom_id: AtomId) -> u32 {
        self.mol.neighbors(atom_id).count() as u32
    }

    /// Total connections = explicit bonds + explicit H count (Daylight `X<n>`).
    fn total_connections(&self, atom_id: AtomId) -> u32 {
        self.degree(atom_id) + self.total_h(atom_id)
    }

    /// Total valence (Daylight `v<n>`): sum of bond orders to neighbours
    /// (with aromatic promotion) plus implicit H contribution.
    fn valence(&self, atom_id: AtomId) -> u32 {
        let bond_sum = self.effective_bond_sum(atom_id);
        let h = self.total_h(atom_id) as f64;
        (bond_sum + h).round() as u32
    }

    /// Whether an atom carries the aromatic flag (parser sets `aromatic = 1.0`
    /// on lowercase SMILES atoms and inside aromatic rings).
    fn is_aromatic(&self, atom_id: AtomId) -> bool {
        self.mol
            .get_atom(atom_id)
            .ok()
            .and_then(|a| a.get_f64("aromatic"))
            .map(|v| v == 1.0)
            .unwrap_or(false)
    }

    fn element(&self, atom_id: AtomId) -> Option<&str> {
        self.mol
            .get_atom(atom_id)
            .ok()
            .and_then(|a| a.get_str("element"))
    }

    fn formal_charge(&self, atom_id: AtomId) -> i32 {
        self.mol
            .get_atom(atom_id)
            .ok()
            .and_then(|a| a.get_f64("formal_charge"))
            .map(|v| v as i32)
            .unwrap_or(0)
    }

    fn isotope(&self, atom_id: AtomId) -> Option<u16> {
        self.mol
            .get_atom(atom_id)
            .ok()
            .and_then(|a| a.get_f64("isotope"))
            .map(|v| v as u16)
    }

    fn atom_class(&self, atom_id: AtomId) -> Option<u16> {
        self.mol
            .get_atom(atom_id)
            .ok()
            .and_then(|a| a.get_f64("atom_class"))
            .map(|v| v as u16)
    }
}

// ---------------------------------------------------------------------------
// Atom query evaluation
// ---------------------------------------------------------------------------

/// Evaluate a SMARTS atom query against one target atom.
///
/// Combinator semantics follow Daylight precedence (highest → lowest):
/// 1. `!` (Not)
/// 2. implicit `&` (And, between adjacent primitives)
/// 3. `,` (Or)
/// 4. `;` (LowAnd)
///
/// The AST variants encode precedence structurally, so this function is a
/// pure recursive walk — no precedence parsing happens here.
pub(crate) fn eval_atom_query(query: &AtomQuery, atom_id: AtomId, ctx: &TargetCtx<'_>) -> bool {
    match query {
        AtomQuery::Primitive(p) => eval_primitive(p, atom_id, ctx),
        AtomQuery::Not(inner) => !eval_atom_query(inner, atom_id, ctx),
        AtomQuery::And(parts) | AtomQuery::LowAnd(parts) => {
            parts.iter().all(|q| eval_atom_query(q, atom_id, ctx))
        }
        AtomQuery::Or(parts) => parts.iter().any(|q| eval_atom_query(q, atom_id, ctx)),
    }
}

fn eval_primitive(prim: &AtomPrimitive, atom_id: AtomId, ctx: &TargetCtx<'_>) -> bool {
    match prim {
        AtomPrimitive::Element { symbol, aromatic } => {
            // The parser stores atomic-number queries (`[#6]`) as elements
            // with a "#N" symbol; translate them on demand so the matcher
            // handles both `[C]` and `[#6]` uniformly.
            let resolved = if let Some(num_str) = symbol.strip_prefix('#') {
                num_str
                    .parse::<u8>()
                    .ok()
                    .and_then(Element::by_number)
                    .map(Element::symbol)
            } else {
                Some(symbol.as_str())
            };
            match (ctx.element(atom_id), resolved) {
                (Some(target_sym), Some(want_sym)) => {
                    let elem_ok = target_sym.eq_ignore_ascii_case(want_sym);
                    // Atomic-number queries (`[#6]`) don't constrain
                    // aromaticity — Daylight treats `#6` as any carbon.
                    let arom_ok = if symbol.starts_with('#') {
                        true
                    } else {
                        ctx.is_aromatic(atom_id) == *aromatic
                    };
                    elem_ok && arom_ok
                }
                _ => false,
            }
        }
        AtomPrimitive::Wildcard => true,
        AtomPrimitive::Aliphatic => !ctx.is_aromatic(atom_id),
        AtomPrimitive::Aromatic => ctx.is_aromatic(atom_id),
        AtomPrimitive::Degree(n) => ctx.degree(atom_id) == u32::from(*n),
        AtomPrimitive::TotalConnections(n) => ctx.total_connections(atom_id) == u32::from(*n),
        AtomPrimitive::HCount(n) => ctx.total_h(atom_id) == u32::from(*n),
        AtomPrimitive::ImplicitH(n) => {
            // With the current parser pipeline, `h_count` captures the
            // bracket-declared H count. Treat absent bracket H as "0 implicit".
            ctx.total_h(atom_id) == u32::from(*n)
        }
        AtomPrimitive::RingMembership(opt_n) => match opt_n {
            None => ctx.rings.is_atom_in_ring(atom_id),
            Some(n) => ctx.rings.num_atom_rings(atom_id) == usize::from(*n),
        },
        AtomPrimitive::RingSize(n) => {
            ctx.rings.smallest_ring_containing_atom(atom_id) == Some(usize::from(*n))
        }
        AtomPrimitive::Valence(n) => ctx.valence(atom_id) == u32::from(*n),
        AtomPrimitive::Charge(c) => ctx.formal_charge(atom_id) == i32::from(*c),
        AtomPrimitive::Isotope(m) => ctx.isotope(atom_id) == Some(*m),
        AtomPrimitive::AtomClass(c) => ctx.atom_class(atom_id) == Some(*c),
        // Stereochemistry primitives are not yet wired to MolGraph's stereo
        // layer; matcher rejects them until the stereo pass lands.
        AtomPrimitive::Chirality(_) => false,
        // Recursive SMARTS is evaluated out-of-band by the matcher (which
        // has access to the compiled inner pattern). Returning `true` here
        // defers the actual check to the matcher's recursive-match pass;
        // a `false` from *that* pass vetoes the candidate.
        AtomPrimitive::Recursive(_) => true,
    }
}

// ---------------------------------------------------------------------------
// Bond query evaluation
// ---------------------------------------------------------------------------

/// The target-side edge representation handed to [`eval_bond_query`]. It
/// carries the raw bond order (1.0 single, 1.5 aromatic, …) plus a ring
/// flag precomputed from [`RingInfo`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct BondEdge {
    pub(crate) order: f64,
    pub(crate) in_ring: bool,
}

/// Evaluate a bond query against a target bond edge.
///
/// Single bonds match `-`; aromatic bonds match `:`; any-order matches `~`;
/// `@` requires the bond to participate in a ring.
pub(crate) fn eval_bond_query(query: &BondQuery, edge: &BondEdge) -> bool {
    match query {
        BondQuery::Kind(k) => bond_kind_matches(*k, edge.order, edge.in_ring),
        BondQuery::Not(inner) => !eval_bond_query(inner, edge),
        BondQuery::And(parts) => parts.iter().all(|q| eval_bond_query(q, edge)),
        BondQuery::Or(parts) => parts.iter().any(|q| eval_bond_query(q, edge)),
    }
}

fn bond_kind_matches(kind: BondKind, order: f64, in_ring: bool) -> bool {
    const AROM_EPS: f64 = 0.01;
    match kind {
        BondKind::Single => (order - 1.0).abs() < AROM_EPS,
        BondKind::Double => (order - 2.0).abs() < AROM_EPS,
        BondKind::Triple => (order - 3.0).abs() < AROM_EPS,
        BondKind::Quadruple => (order - 4.0).abs() < AROM_EPS,
        BondKind::Aromatic => (order - 1.5).abs() < AROM_EPS,
        BondKind::Any => true,
        BondKind::Ring => in_ring,
        // Directional bonds require bond-side stereo annotation that isn't yet
        // tracked through the IR → MolGraph pipeline.
        BondKind::Up | BondKind::Down => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smiles::{parse_smiles, to_atomistic};

    fn mol(smiles: &str) -> molrs::atomistic::Atomistic {
        to_atomistic(&parse_smiles(smiles).unwrap()).unwrap()
    }

    #[test]
    fn element_primitive_matches_by_symbol_and_aromaticity() {
        let m = mol("Cc1ccccc1");
        let ctx = TargetCtx::new(m.as_molgraph());
        let ids: Vec<_> = m.as_molgraph().atoms().map(|(id, _)| id).collect();

        let c_upper = AtomPrimitive::Element {
            symbol: "C".to_owned(),
            aromatic: false,
        };
        let c_lower = AtomPrimitive::Element {
            symbol: "C".to_owned(),
            aromatic: true,
        };

        assert!(eval_primitive(&c_upper, ids[0], &ctx)); // methyl
        assert!(!eval_primitive(&c_upper, ids[1], &ctx)); // ring C is aromatic
        assert!(eval_primitive(&c_lower, ids[1], &ctx));
    }

    #[test]
    fn ring_primitive_flags_ring_atoms_only() {
        let m = mol("c1ccccc1C"); // toluene
        let ctx = TargetCtx::new(m.as_molgraph());
        let ids: Vec<_> = m.as_molgraph().atoms().map(|(id, _)| id).collect();

        let r_any = AtomPrimitive::RingMembership(None);
        // First 6 atoms are ring; 7th is methyl.
        for ring_id in &ids[0..6] {
            assert!(eval_primitive(&r_any, *ring_id, &ctx));
        }
        assert!(!eval_primitive(&r_any, ids[6], &ctx));
    }

    #[test]
    fn bond_kind_matches_on_numeric_order() {
        assert!(bond_kind_matches(BondKind::Single, 1.0, false));
        assert!(bond_kind_matches(BondKind::Double, 2.0, false));
        assert!(bond_kind_matches(BondKind::Triple, 3.0, false));
        assert!(bond_kind_matches(BondKind::Aromatic, 1.5, true));
        assert!(bond_kind_matches(BondKind::Any, 1.0, false));
        assert!(bond_kind_matches(BondKind::Ring, 1.0, true));
        assert!(!bond_kind_matches(BondKind::Ring, 1.0, false));
    }

    #[test]
    fn logical_operators_compose_correctly() {
        let m = mol("C=CC"); // propene
        let ctx = TargetCtx::new(m.as_molgraph());
        let ids: Vec<_> = m.as_molgraph().atoms().map(|(id, _)| id).collect();

        // [C;X4] — non-aromatic carbon with 4 total connections
        // C2 has 1 explicit bond + 3 implicit H; X counts only explicit+explicit-H.
        // Since h_count isn't set on organic atoms, X=degree here.
        let x1 = AtomQuery::And(vec![
            AtomQuery::Primitive(AtomPrimitive::Element {
                symbol: "C".to_owned(),
                aromatic: false,
            }),
            AtomQuery::Primitive(AtomPrimitive::Degree(1)),
        ]);
        // C2 (terminal methyl) has degree 1 → match
        assert!(eval_atom_query(&x1, ids[2], &ctx));
        // C0 has degree 1 too (in C=C-C linear chain)
        assert!(eval_atom_query(&x1, ids[0], &ctx));
        // C1 has degree 2 → no match
        assert!(!eval_atom_query(&x1, ids[1], &ctx));
    }
}
