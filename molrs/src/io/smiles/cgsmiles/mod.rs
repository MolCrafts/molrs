//! `CGsmiles` coarse-graph notation: parser and intermediate representation.
//!
//! `CGsmiles` writes a molecule at a *coarse-grained* resolution: a block
//! `{…}` of named **beads** — single nodes each standing for a whole fragment
//! of the molecule — joined like atoms in SMILES. `{[#PEO][#PEO][#PEO]}` is a
//! three-bead trimer of poly(ethylene oxide); each `#NAME` is a fragment name
//! that a later stage resolves into atoms. This module parses one such block
//! into a [`CGSmilesIR`] and does nothing else — no fragment resolution, no
//! atoms, no `MolGraph`.
//!
//! # What the block grammar says
//!
//! * A node is `[#NAME]`, optionally annotated `[#NAME;q=-0.5;kind=ether]`.
//!   Two `[#A]` occurrences are two distinct nodes sharing a name.
//! * Adjacency is a bond. The symbols `-`, `=`, `#`, `$` write the
//!   multiplicities 1..=4, and the default is 1. One symbol serves every
//!   position it may be written in: between two nodes, before a ring marker,
//!   around a branch, before `|` — where it becomes the bond that joins each
//!   repeated copy to the previous one — and before a bonding-descriptor
//!   bracket, where it becomes the order the descriptor's later pairing will
//!   take.
//! * `( … )` branches and nests, attaching to the node written before `(`.
//! * A bare digit `1`..`9`, or `%` and exactly two digits, opens a ring marker
//!   that its second occurrence closes; the ring bond takes the order written
//!   before the *opening* marker, and a marker number is reusable once closed.
//!   A different order written before the *closing* marker is a conflict
//!   ([`SmilesErrorKind::RingBondConflict`]), not an override.
//! * `|n` repeats the preceding unit `n` times in total, chaining the copies.
//!   After `)` the unit is the whole branch **including its anchor**, so
//!   `{[#A]([#B][#C])|2}` is six nodes, not four.
//! * Node annotations bind positionally: slot 1 is the `#NAME`, slot 2 is `q`
//!   (partial charge), slot 3 is `w` (mapping weight). `[#A;0.5]` therefore
//!   means `[#A;q=0.5]`, and a fourth positional field has no slot to take.
//!   Keys the table does not reserve are kept verbatim — except `x`, the
//!   dialect's chirality key, which is refused (below).
//!
//! Units: `charge` is a **partial charge in elementary charge units `e`**, not
//! a formal charge (`.claude/notes/science.md`); `w` is dimensionless; a bond
//! order is a dimensionless multiplicity 1..=4.
//!
//! # Refusals of valid notation (choices of this first implementation)
//!
//! None of these is a property of the notation — each is this implementation
//! declining to model something, loudly rather than silently. ("A later
//! `cgsmiles-*` link" names a follow-up specification in
//! `.claude/specs/`, the chain of documents this parser was built from.)
//!
//! * **Bond order 0**, written `.` (`{[#A].[#B]}`): a virtual edge between
//!   virtual particles. Refused with [`SmilesErrorKind::CgInvalidBondOrder`]
//!   until a later `cgsmiles-*` link models virtual particles, which needs a
//!   level-side representation rather than a fifth [`CGBondOrder`] variant.
//! * **A non-default mapping weight `w`**, keyword (`[#A;w=2]`) or positional
//!   (`[#A;0;0.5]`): refused with
//!   [`SmilesErrorKind::CgUnsupportedAnnotation`] until a later `cgsmiles-*`
//!   link models mapping weights, together with the pending
//!   schema-vocabulary spec that decides where a weight column lives. The
//!   default `w = 1` parses, and is dropped rather than retained.
//! * **A chirality annotation `x`** (`[#A;x=S]`): refused with the same
//!   [`SmilesErrorKind::CgUnsupportedAnnotation`], because a coarse-grained
//!   bead has no stereocentre for molrs to place and silently keeping the key
//!   as free text would claim otherwise.
//!
//! # Invariants and caveats
//!
//! * **One block, one level.** This version of the parser accepts exactly one
//!   block, so `levels.len() == 1`; text after the closing `}` is
//!   [`SmilesErrorKind::TrailingCharacters`].
//! * **Replayed spans are shared.** `|n` re-reads the unit's bytes by
//!   rewinding the one scanner over the full input, so every copy carries the
//!   *template's* [`Span`](crate::io::smiles::Span): spans within a level are
//!   neither disjoint nor monotonic, and a diagnostic raised inside a copy
//!   points at the template's text.
//! * **Descriptors are cloned per copy, ring markers are not replayable.** A
//!   bonding descriptor is a pairing *class*, consumed one occurrence at a
//!   time, so three copies of `[$1]` are three ports of class `$1`. A ring
//!   marker is a one-shot identity, so opening or closing one inside a
//!   repeated unit is [`SmilesErrorKind::CgRepeatOnRingMarker`].
//! * **Edges are in parse order**, the ring-closure edge emitted where its
//!   closing marker is read. The reference implementation iterates its graph
//!   library's adjacency order, so a ring-bearing string indexes edges
//!   differently there.
//!
//! # References
//!
//! Primary source: the `CGsmiles` documentation, cgsmiles.readthedocs.io,
//! *Basic graph description* and *Multiple resolutions*. Reference
//! implementation: `CGsmiles @ 910c9ee`, `read_cgsmiles.py:134` (the
//! bond-order table) and `dialects.py` (the reserved-annotation table).
//! The `CGsmiles` paper — Grünewald, F. et al., J. Chem. Inf. Model. **65**
//! (2025), DOI 10.1021/acs.jcim.5c00064 — is cited for provenance only: it was
//! not readable when this module was written, so no rule above rests on it.
//!
//! [`SmilesErrorKind::CgInvalidBondOrder`]: crate::io::smiles::SmilesErrorKind::CgInvalidBondOrder
//! [`SmilesErrorKind::CgUnsupportedAnnotation`]: crate::io::smiles::SmilesErrorKind::CgUnsupportedAnnotation
//! [`SmilesErrorKind::CgRepeatOnRingMarker`]: crate::io::smiles::SmilesErrorKind::CgRepeatOnRingMarker
//! [`SmilesErrorKind::RingBondConflict`]: crate::io::smiles::SmilesErrorKind::RingBondConflict
//! [`SmilesErrorKind::TrailingCharacters`]: crate::io::smiles::SmilesErrorKind::TrailingCharacters

mod ast;
mod parser;

use crate::io::smiles::error::SmilesError;
use parser::CgParser;

pub use ast::{CGBondOrder, CGEdge, CGGraph, CGNode, CGSmilesIR};

/// Parse one `CGsmiles` coarse-graph block into its intermediate
/// representation.
///
/// `CGsmiles` is a line notation that writes a molecule at a *coarse-grained*
/// resolution: where SMILES names atoms, `CGsmiles` names **beads**, each one
/// standing for a whole named fragment of the molecule. A block `{…}` is one
/// resolution level, so `{[#PEO][#PEO][#PEO]}` is a chain of three beads of
/// the fragment called `PEO`. Inside a block, `[#NAME]` is a node, adjacency
/// is a bond, the symbols `-`, `=`, `#` and `$` write bond multiplicities
/// 1..=4 (the default is 1), `( … )` branches, a bare digit `1`..`9` or `%`
/// and two digits opens a ring marker that its second occurrence closes, `|n`
/// repeats the preceding unit `n` times in total, and `;`-separated
/// annotations decorate a node — `[#A;q=-0.5]`, or positionally `[#A;-0.5]`,
/// since annotation slot 2 is `q`.
///
/// The result is that one block and nothing more: node names are **not**
/// resolved to fragments, nothing is expanded into atoms, and no molecular
/// graph or frame is built.
///
/// Units: a node's [`charge`](CGNode::charge) is a **partial charge in
/// elementary charge units `e`** — the fractional force-field charge carried
/// by a bead, not the integer formal charge of an atom. A bond order is a
/// dimensionless multiplicity, 1..=4.
///
/// # Valid notation this parser refuses
///
/// Three constructs are legal `CGsmiles` that molrs does not yet model, and
/// are refused loudly rather than dropped silently: bond order 0, written `.`
/// (`{[#A].[#B]}`), which denotes a virtual edge between virtual particles;
/// a mapping weight `w` at any value other than its default 1, keyword
/// (`[#A;w=2]`) or positional (`[#A;0;0.5]`); and a chirality annotation
/// (`[#A;x=S]`), a coarse bead having no stereocentre to place. Everything
/// else the annotation table does not reserve is kept verbatim in
/// [`CGNode::annotations`].
///
/// # Invariant and caveats
///
/// This version of the parser accepts exactly one block, so
/// `levels.len() == 1`, and text after the closing `}` is
/// [`TrailingCharacters`]. Nodes and edges are in parse order, a ring-closure
/// edge appearing where its closing marker was read — the reference
/// implementation orders edges by its graph library's adjacency instead, so a
/// ring-bearing string indexes edges differently there. The copies `|n` makes
/// are re-read from the template's bytes, so each copy **shares** the
/// template's [`Span`](crate::io::smiles::Span): spans within a level are
/// neither disjoint nor monotonic, and a diagnostic raised inside a copy
/// points at the template's text. A bonding descriptor inside a repeated unit
/// is cloned into every copy — a descriptor is a pairing *class*, consumed one
/// occurrence at a time — whereas a ring marker is a one-shot identity and
/// cannot be replayed at all ([`CgRepeatOnRingMarker`]).
///
/// A free function, matching
/// [`parse_smiles`](crate::io::smiles::parse_smiles) and
/// [`parse_smarts`](crate::io::smiles::parse_smarts): per-notation parsing in
/// this module is stateless, and the parser that does the work is a private
/// type with exactly one user-visible step.
///
/// # Errors
///
/// Returns a [`SmilesError`] stamped
/// [`Notation::CGsmiles`](crate::io::smiles::Notation::CGsmiles) and carrying
/// the whole of `text`, so the rendered message reads `CGsmiles parse error at
/// position N: …` with a caret under column N.
///
/// Eleven kinds name inputs only this notation has: [`CgEmptyBlock`] (`{}`),
/// [`CgMalformedAnnotation`] (`{[#A;]}`, `{[#A;=1]}`, `{[#A;q=x]}`, or a
/// fourth positional field), [`CgUnsupportedAnnotation`] (the weight and
/// chirality refusals above), [`CgAnnotationOnWildcard`] (`{[#*;q=1]}`, the
/// wildcard bead standing for *any* fragment and so having no properties of
/// its own), [`CgInvalidBondOrder`] (`.`, or any other non-bond character in
/// bond position), [`CgInvalidRepeatCount`] (`{[#A]|0}`, `{[#A]|x}`),
/// [`CgDuplicateEdge`] (a ring closure re-forming a pair the graph already
/// has, or bonding a node to itself), [`CgRepeatOnBranchedNode`] (`|` inside
/// an open branch, or after a node carrying two branches),
/// [`CgRepeatOnRingMarker`], [`CgDanglingBond`] (`{[#A]=}`) and
/// [`CgInvalidRingMarker`] (`%` not followed by exactly two digits).
///
/// Eight kinds are shared with SMILES and SMARTS, and are told apart from them
/// by [`SmilesError::notation`] rather than by their name: [`EmptyInput`] (an
/// empty `text`), [`UnexpectedChar`], [`UnexpectedEnd`] (the block never
/// closes), [`UnclosedBracket`] (a `[…]` never closes), [`UnclosedBranch`],
/// [`UnmatchedRingClosure`] (a marker still open at `}`),
/// [`RingBondConflict`] and [`TrailingCharacters`].
///
/// A bonding descriptor written beside a node is checked by the same grammar
/// rules the SMILES fragment dialect uses, so [`BondInsideDescriptor`]
/// (`{[#A][$=]}`), [`InvalidDescriptorLabel`] (`{[#A][$a+]}`) and
/// [`DanglingDescriptor`] (`{[$][#A]}`) can be raised too, stamped as
/// `CGsmiles`.
///
/// [`TrailingCharacters`]: crate::io::smiles::SmilesErrorKind::TrailingCharacters
/// [`CgEmptyBlock`]: crate::io::smiles::SmilesErrorKind::CgEmptyBlock
/// [`CgMalformedAnnotation`]: crate::io::smiles::SmilesErrorKind::CgMalformedAnnotation
/// [`CgUnsupportedAnnotation`]: crate::io::smiles::SmilesErrorKind::CgUnsupportedAnnotation
/// [`CgAnnotationOnWildcard`]: crate::io::smiles::SmilesErrorKind::CgAnnotationOnWildcard
/// [`CgInvalidBondOrder`]: crate::io::smiles::SmilesErrorKind::CgInvalidBondOrder
/// [`CgInvalidRepeatCount`]: crate::io::smiles::SmilesErrorKind::CgInvalidRepeatCount
/// [`CgDuplicateEdge`]: crate::io::smiles::SmilesErrorKind::CgDuplicateEdge
/// [`CgRepeatOnBranchedNode`]: crate::io::smiles::SmilesErrorKind::CgRepeatOnBranchedNode
/// [`CgRepeatOnRingMarker`]: crate::io::smiles::SmilesErrorKind::CgRepeatOnRingMarker
/// [`CgDanglingBond`]: crate::io::smiles::SmilesErrorKind::CgDanglingBond
/// [`CgInvalidRingMarker`]: crate::io::smiles::SmilesErrorKind::CgInvalidRingMarker
/// [`EmptyInput`]: crate::io::smiles::SmilesErrorKind::EmptyInput
/// [`UnexpectedChar`]: crate::io::smiles::SmilesErrorKind::UnexpectedChar
/// [`UnexpectedEnd`]: crate::io::smiles::SmilesErrorKind::UnexpectedEnd
/// [`UnclosedBracket`]: crate::io::smiles::SmilesErrorKind::UnclosedBracket
/// [`UnclosedBranch`]: crate::io::smiles::SmilesErrorKind::UnclosedBranch
/// [`UnmatchedRingClosure`]: crate::io::smiles::SmilesErrorKind::UnmatchedRingClosure
/// [`RingBondConflict`]: crate::io::smiles::SmilesErrorKind::RingBondConflict
/// [`BondInsideDescriptor`]: crate::io::smiles::SmilesErrorKind::BondInsideDescriptor
/// [`InvalidDescriptorLabel`]: crate::io::smiles::SmilesErrorKind::InvalidDescriptorLabel
/// [`DanglingDescriptor`]: crate::io::smiles::SmilesErrorKind::DanglingDescriptor
/// [`SmilesError::notation`]: crate::io::smiles::SmilesError::notation
///
/// # Examples
///
/// ```
/// use molrs::io::smiles::parse_cgsmiles;
///
/// let ir = parse_cgsmiles("{[#PEO][#PEO][#PEO]}").unwrap();
/// assert_eq!(ir.levels.len(), 1);
/// assert_eq!(ir.levels[0].nodes.len(), 3);
/// assert_eq!(ir.levels[0].nodes[0].name, "PEO");
/// assert_eq!(ir.levels[0].edges.len(), 2);
/// assert_eq!(ir.levels[0].edges[0].order.multiplicity(), 1);
/// ```
pub fn parse_cgsmiles(text: &str) -> Result<CGSmilesIR, SmilesError> {
    CgParser::new(text).parse()
}
