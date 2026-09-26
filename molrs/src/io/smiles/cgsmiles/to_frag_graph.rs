//! [`CGSmilesIR::to_frag_graph`]: the lowest level's node names and last resolved pair list as a unit-level [`FragGraph`](molrs::system::frag_graph::FragGraph).

use crate::io::smiles::cgsmiles::ast::{CGSmilesIR, PairEnd};
use crate::io::smiles::cgsmiles::to_atomistic::LowestLevel;
use crate::io::smiles::cgsmiles::to_fragment::cg_build;
use crate::io::smiles::error::SmilesError;
use molrs::system::frag_graph::{FragEdge, FragGraph};

impl CGSmilesIR {
    /// The unit-level topology this string writes: one node per lowest-level
    /// node, named after its fragment, and one edge per resolved pair of the
    /// lowest level, mapped verbatim.
    ///
    /// The result names templates by fragment name and ports by ordinal; it
    /// holds no atoms. To build the molecule, store one template per name in a
    /// `builder::FragLibrary` (for example from
    /// [`to_fragment`](Self::to_fragment)) and hand this graph to a
    /// `builder::Assembler`.
    ///
    /// A pair `PairEnd::Body { instance, port }` on each side becomes the edge
    /// `(a, b, port_a, port_b)`; `port` is the descriptor index on the body,
    /// which is the port's ordinal in the template
    /// [`Fragment::ordered_ports`](molrs::system::fragment::Fragment::ordered_ports)
    /// order.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgNotExpandable`](crate::io::smiles::SmilesErrorKind::CgNotExpandable)
    /// for a base-only string (no fragment table, so no body carries a port).
    ///
    /// [`SmilesErrorKind::CgBuild`](crate::io::smiles::SmilesErrorKind::CgBuild)
    /// when the pair lists and levels are misaligned, when a lowest-level
    /// pair end is a [`PairEnd::Sub`], or when [`FragGraph::new`] refuses the
    /// edges (an instance out of range, a self-edge, a port used twice).
    pub fn to_frag_graph(&self) -> Result<FragGraph, SmilesError> {
        let LowestLevel { level, pairs, .. } = self.lowest_level()?;
        let body = |end: &PairEnd| match *end {
            PairEnd::Body { instance, port } => Ok((instance, port)),
            PairEnd::Sub { .. } => Err(cg_build(
                self.span,
                format!("{end:?} is not a last-level port"),
            )),
        };
        let edges = pairs
            .iter()
            .map(|pair| {
                let (a, port_a) = body(&pair.src)?;
                let (b, port_b) = body(&pair.dst)?;
                Ok(FragEdge {
                    a,
                    b,
                    port_a,
                    port_b,
                })
            })
            .collect::<Result<Vec<_>, SmilesError>>()?;
        let nodes = level.nodes.iter().map(|n| n.name.clone()).collect();
        FragGraph::new(nodes, edges)
            .map_err(|e| cg_build(self.span, format!("unit-level graph: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use crate::io::smiles::{SmilesErrorKind, parse_cgsmiles};
    use molrs::system::frag_graph::FragEdge;

    // Expected values follow `.claude/specs/assembly-04-fraggraph.md`
    // § Design 6: nodes are the lowest level's names and edges are the last
    // pair list mapped verbatim. The edges are hand-derived below, not read
    // off the IR.

    /// F2: an OH–PEO–PEO–PEO–OH chain; four coarse bonds, four pairs.
    const F2: &str = "{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}";

    fn edge(a: usize, b: usize, port_a: usize, port_b: usize) -> FragEdge {
        FragEdge {
            a,
            b,
            port_a,
            port_b,
        }
    }

    /// Hand-derived: instances OH0, PEO1, PEO2, PEO3, OH4; `[$]O` has port 0,
    /// `[$]COC[$]` ports 0 and 1 (descriptor index). Every descriptor is `$`
    /// with no label and single order, so any two free ports pair, and the
    /// written edges (0,1), (1,2), (2,3), (3,4) take the first free pair in
    /// written order: (0,1) takes OH0:0 and PEO1:0; (1,2) finds PEO1:0 used
    /// and takes PEO1:1 with PEO2:0; likewise (2,3) PEO2:1–PEO3:0 and (3,4)
    /// PEO3:1–OH4:0.
    #[test]
    fn maps_lowest_names_and_last_pairs_verbatim() {
        let ir = parse_cgsmiles(F2).expect("F2 parses");
        let graph = ir.to_frag_graph().expect("F2 converts");

        assert_eq!(
            graph.nodes(),
            ["OH", "PEO", "PEO", "PEO", "OH"].map(String::from)
        );
        assert_eq!(
            graph.edges(),
            &[
                edge(0, 1, 0, 0),
                edge(1, 2, 1, 0),
                edge(2, 3, 1, 0),
                edge(3, 4, 1, 0),
            ]
        );
    }

    #[test]
    fn rejects_a_base_only_ir() {
        let ir = parse_cgsmiles("{[#A][#B]}").expect("a base-only string must parse");
        let err = ir
            .to_frag_graph()
            .expect_err("a base-only string has no resolved body ports");
        assert!(
            matches!(err.kind, SmilesErrorKind::CgNotExpandable(_)),
            "kind was {:?}",
            err.kind
        );
    }
}
