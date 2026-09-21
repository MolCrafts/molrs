//! `CGsmiles` coarse-graph intermediate representation.
//!
//! One `CGsmiles` string describes a molecule at one or more *resolutions*:
//! the coarse level names beads (`{[#PEO][#PEO]}`), and later levels resolve
//! each bead into a fragment. These types are the parsed form of a coarse
//! level — named nodes, typed edges, and a byte [`Span`] back into the input
//! for every one of them.
//!
//! They live here, next to their only consumer, rather than in `chem/`: that
//! module's vocabulary is what SMILES and SMARTS *share*, and a coarse-grained
//! bead is neither. The one type this module borrows from there is
//! [`BondingDescriptor`], which the fragment dialect and the coarse graph
//! genuinely do share.
//!
//! Every type here is a value produced by
//! [`parse_cgsmiles`](crate::io::smiles::parse_cgsmiles): the fields are public
//! to read, there is no public constructor, and there is no supported
//! mutation — the same shape as
//! [`SmilesIR`](crate::io::smiles::SmilesIR).

use std::collections::BTreeMap;

use crate::core::types::F;
use crate::io::smiles::chem::ast::{BondingDescriptor, SmilesIR, Span};

/// A parsed `CGsmiles` string: the blocks that were written, and the levels
/// they denote.
///
/// A *resolution level* is one view of the same molecule: the coarse level
/// names beads, and a finer level names what each bead is made of. A
/// `CGsmiles` string carries one base block and any number of **fragment
/// blocks** after it, each mapping the names used one level up to the bodies
/// that replace them.
///
/// # Authority
///
/// `fragments[k]` is the **authority** for a fragment's shape: it is what the
/// string wrote. `levels[k + 1]` is the **expansion** that table denotes — one
/// disjoint copy of a body per node of `levels[k]` — a derived representation
/// the reader produces once. Editing one would not update the other, which is
/// why there is no public constructor and no supported mutation: a value of
/// this type is read, not built.
///
/// # Alignment invariant
///
/// A base-only string has no fragment table and one level:
/// `fragments.is_empty() && levels.len() == 1`. Otherwise `fragments.len()` is
/// the number of fragment blocks that were written and
/// `levels.len() == fragments.len()` — one fewer level than one might expect,
/// because the **last** block is atomistic and builds no coarse level. That
/// asymmetry is intentional: levels are graphs of beads, and atoms are not
/// [`CGNode`]s. The last table's bodies stay
/// [`FragmentBody::Smiles`](FragmentBody::Smiles) values with their bonding
/// descriptors intact.
///
/// # A value the reader returns is finished
///
/// Every `CGSmilesIR` returned by
/// [`parse_cgsmiles`](crate::io::smiles::parse_cgsmiles) is **fully
/// instantiated and validated**: every name used at a level has a definition,
/// every intermediate table has been expanded into the next level, and every
/// refusal this version makes (starting with the squash operator `[!]`) has
/// already been raised. The reader hands out no half-resolved state. The
/// fields are `pub`, so the type does not enforce this for a hand-built value
/// — the reader is the only supported source.
#[derive(Debug, Clone, PartialEq)]
pub struct CGSmilesIR {
    /// Resolution levels, coarsest first. Level 0 is the base block that was
    /// written; every later level is the expansion of the table above it.
    pub levels: Vec<CGGraph>,
    /// Fragment tables, coarsest first, each keyed by the fragment name
    /// written after `#`. `fragments[k]` resolves the names of `levels[k]`.
    ///
    /// [`BTreeMap`] rather than a hash map so iteration — and therefore any
    /// diagnostic or example that walks a table — is in name order, the way
    /// the crate already keys name tables.
    pub fragments: Vec<BTreeMap<String, CGFragmentDef>>,
    /// Byte range of the whole parsed string.
    pub span: Span,
}

/// One entry of a fragment block: `#PEO=[$]COC[$]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CGFragmentDef {
    /// The fragment name written after `#`, without the sigil.
    pub name: String,
    /// What the name stands for: a coarse graph of the next level's nodes, or
    /// an atomistic SMILES fragment body.
    pub body: FragmentBody,
    /// Byte range of the whole entry — name, `=` and body — in the parsed
    /// input.
    pub span: Span,
}

/// The two shapes a fragment body may take, told apart by the block's
/// **position** rather than by any syntax: the last block's bodies are
/// atomistic, every earlier block's are coarse.
///
/// The notation marks neither (the reference implementation passes its reader
/// a `last_all_atom` flag instead), so a string whose deepest resolution is
/// meant to stay coarse-grained cannot be written — the limitation that
/// positional dispatch buys.
#[derive(Debug, Clone, PartialEq)]
pub enum FragmentBody {
    /// A coarse graph over the *next* level's `[#X]` nodes, as written in an
    /// intermediate block: `#B1=[>][#PEO][#PEO][<]`.
    Graph(CGGraph),
    /// An atomistic OpenSMILES body with bonding descriptors, as written in
    /// the last block: `#PEO=[>]COC[<]`. It is kept as the parser returned
    /// it — never expanded into atoms here.
    Smiles(SmilesIR),
}

/// One resolution level: coarse-grained nodes and the edges between them.
///
/// Both vectors are in **parse order** — nodes in the order their brackets
/// were read, edges in the order the notation formed them, with a ring-closure
/// edge emitted where its closing marker was read. The `CGsmiles` reference
/// implementation (github.com/gruenewald-lab/CGsmiles) iterates its graph
/// library's adjacency order instead, so a ring-bearing string indexes edges
/// differently there.
#[derive(Debug, Clone, PartialEq)]
pub struct CGGraph {
    /// Nodes in parse order; a node is addressed by its index in this vector.
    pub nodes: Vec<CGNode>,
    /// Edges in parse order, each naming two indices into `nodes`.
    pub edges: Vec<CGEdge>,
}

/// One coarse-grained node: `[#PEO]`, `[#A;q=-0.5]`.
///
/// Two occurrences of `[#A]` are two distinct nodes that happen to share a
/// name; the name is a *fragment* name to be resolved later, not an identity.
#[derive(Debug, Clone, PartialEq)]
pub struct CGNode {
    /// The fragment name written after `#`, without the sigil: `PEO`, or `*`
    /// for the wildcard node.
    pub name: String,
    /// Partial charge in `e` (`CGsmiles` `q`, positional slot 2), `None` when
    /// the notation wrote none.
    ///
    /// The unit is the molrs convention — the notation states none — and the
    /// field is a **partial**, not a formal, charge: the formal charge of an
    /// atom is
    /// [`AtomSpec::Bracket`](crate::io::smiles::AtomSpec::Bracket)'s
    /// `charge: Option<i8>`, an integer count of elementary charges written
    /// `[NH4+]`. This one is a fractional force-field charge attached to a
    /// bead.
    pub charge: Option<F>,
    /// Annotations the dialect does not reserve, as written, in written order.
    ///
    /// `[#A;q=-0.5;kind=ether]` leaves `[("kind", "ether")]` here. The
    /// reserved keys never reach this list: `q` is bound to
    /// [`charge`](CGNode::charge); `w` — the mapping weight, dimensionless —
    /// is accepted only at its default `1` and then dropped, any other value
    /// being refused; and `x`, the chirality key, is refused outright. A field
    /// written with no `=` names no key and takes the next positional slot
    /// instead (slot 2 is `q`, slot 3 is `w`), so `[#A;-0.5]` sets the charge
    /// rather than landing here.
    pub annotations: Vec<(String, String)>,
    /// Bonding descriptors written beside this node, in written order.
    ///
    /// A *bonding descriptor* — `[$]`, `[<]`, `[>]` or `[!]`, optionally
    /// labelled as in `[$1]` — marks a site at which this bead may later be
    /// joined to another fragment; [`BondingDescriptor`] carries the glyph,
    /// the label and the bond order of the join.
    ///
    /// A descriptor is a pairing *class*, consumed one occurrence at a time —
    /// which is why the `|n` repeat operator clones the descriptors of the
    /// unit it copies into every copy, while it refuses to replay a ring
    /// marker (a one-shot identity).
    pub descriptors: Vec<BondingDescriptor>,
    /// Index into the *previous* level's `nodes` of the node this one was
    /// instantiated from.
    ///
    /// `None` in `levels[0]` — nothing above the base graph instantiated it —
    /// and `None` in every [`FragmentBody::Graph`] body, which is a template
    /// rather than an instance. It is **set by instantiation, never
    /// inherited**: the copies of one body used by two different nodes carry
    /// the two different parents, not the body's own value.
    pub parent: Option<usize>,
    /// Byte range of the node's own text within the parsed input.
    ///
    /// Spans in a [`CGGraph`] are neither disjoint nor monotonic: the copies
    /// the `|n` repeat operator makes are re-read from the template's bytes,
    /// so every copy **shares** the template's span.
    pub span: Span,
}

/// One coarse edge, joining `nodes[i]` and `nodes[j]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CGEdge {
    /// Index of the first endpoint in [`CGGraph::nodes`].
    pub i: usize,
    /// Index of the second endpoint in [`CGGraph::nodes`].
    pub j: usize,
    /// Multiplicity of the edge, from the bond symbol that formed it, or
    /// [`CGBondOrder::Single`] when the notation wrote no symbol.
    pub order: CGBondOrder,
    /// Byte range of the text at which the edge was formed.
    ///
    /// Which text that is depends on how the edge came about: for a bond
    /// between two adjacent nodes it is the bracket of the *second* node, for
    /// a ring closure it is the closing marker, and for the bond that chains
    /// one `|n` copy to the previous one it is the `|` itself. Like
    /// [`CGNode::span`], a replayed copy shares the template's span.
    pub span: Span,
}

/// Multiplicity of a coarse edge, as its bond symbol writes it: a
/// dimensionless count of 1..=4 bonds.
///
/// This is deliberately not
/// [`BondKind`](crate::io::smiles::BondKind): that enum also carries
/// `Aromatic`, `Up`, `Down`, `Any` and `Ring`, none of which means anything
/// between two beads, and reusing it would let a stereo bond typecheck its way
/// into a coarse edge. An edge here is a *multiplicity*; the chemistry of a
/// resolved bond is a `BondKind` and is named `kind` where it belongs.
///
/// Order 0 — `.`, the virtual edge the notation permits between virtual
/// particles — is not representable here on purpose:
/// [`parse_cgsmiles`](crate::io::smiles::parse_cgsmiles) refuses `.` with
/// [`SmilesErrorKind::CgInvalidBondOrder`](crate::io::smiles::SmilesErrorKind::CgInvalidBondOrder)
/// rather than admit an edge that is not an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CGBondOrder {
    /// `-`, and the default when no symbol is written.
    Single,
    /// `=`.
    Double,
    /// `#`.
    Triple,
    /// `$`.
    Quadruple,
}

impl CGBondOrder {
    /// The multiplicity as a number, 1..=4: how many bonds this edge stands
    /// for when the level is expanded into atoms. A pure count, dimensionless.
    pub fn multiplicity(&self) -> u8 {
        match self {
            CGBondOrder::Single => 1,
            CGBondOrder::Double => 2,
            CGBondOrder::Triple => 3,
            CGBondOrder::Quadruple => 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bond-symbol → multiplicity table: `-` 1, `=` 2, `#` 3, `$` 4. (Rule
    // R2.4 of `.claude/specs/cgsmiles-01b-graph.md` § Domain basis.)
    // Hand-written from the cited documentation; no external tool.

    #[test]
    fn test_single_bond_has_multiplicity_one() {
        assert_eq!(CGBondOrder::Single.multiplicity(), 1);
    }

    #[test]
    fn test_double_bond_has_multiplicity_two() {
        assert_eq!(CGBondOrder::Double.multiplicity(), 2);
    }

    #[test]
    fn test_triple_bond_has_multiplicity_three() {
        assert_eq!(CGBondOrder::Triple.multiplicity(), 3);
    }

    #[test]
    fn test_quadruple_bond_has_multiplicity_four() {
        assert_eq!(CGBondOrder::Quadruple.multiplicity(), 4);
    }
}
