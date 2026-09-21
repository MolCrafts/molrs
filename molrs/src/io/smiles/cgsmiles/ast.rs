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

use crate::core::types::F;
use crate::io::smiles::chem::ast::{BondingDescriptor, Span};

/// A parsed `CGsmiles` string: one graph per resolution level.
///
/// A *resolution level* is one view of the same molecule: the coarse level
/// names beads, and a finer level names what each bead is made of. A
/// `CGsmiles` string may carry several, written as several `{…}` blocks.
///
/// # Invariant (this version)
///
/// `levels.len() == 1`. One block is one resolution level, and
/// [`parse_cgsmiles`](crate::io::smiles::parse_cgsmiles) currently accepts
/// exactly one block; the vector shape is already here because resolving
/// fragments will append further levels to the same value.
#[derive(Debug, Clone, PartialEq)]
pub struct CGSmilesIR {
    /// Resolution levels, coarsest first. Level 0 is the block that was
    /// written.
    pub levels: Vec<CGGraph>,
    /// Byte range of the whole parsed string.
    pub span: Span,
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
    /// Partial charge in elementary charge units `e` (`CGsmiles` `q`,
    /// positional slot 2), `None` when the notation wrote none.
    ///
    /// This is **not** a formal charge: the formal charge of an atom is
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
