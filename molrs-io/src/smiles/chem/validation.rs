//! Shared post-parse validation helpers used by both SMILES and SMARTS.
//!
//! Validation that depends only on the shared AST vocabulary lives here.
//! Language-specific checks (element-symbol validity for SMILES, query-primitive
//! well-formedness for SMARTS) live in the sibling `smiles::validate` and
//! `smarts::validate` modules.

use std::collections::HashMap;

use crate::smiles::chem::ast::{Chain, ChainElement, SmilesIR, Span};
use crate::smiles::error::{SmilesError, SmilesErrorKind};

/// Ensure every ring-closure digit is opened and closed exactly once.
///
/// This check applies equally to SMILES and SMARTS because ring closure is a
/// syntactic construct of the shared grammar.
pub(crate) fn validate_ring_closures(mol: &SmilesIR, input: &str) -> Result<(), SmilesError> {
    let mut open: HashMap<u16, Span> = HashMap::new();

    for component in &mol.components {
        collect_ring_closures(component, &mut open);
    }

    if let Some((&rnum, &span)) = open.iter().next() {
        return Err(SmilesError::new(
            SmilesErrorKind::UnmatchedRingClosure(rnum),
            span,
            input,
        ));
    }

    Ok(())
}

fn collect_ring_closures(chain: &Chain, open: &mut HashMap<u16, Span>) {
    for elem in &chain.tail {
        match elem {
            ChainElement::RingClosure { rnum, span, .. } => {
                if open.remove(rnum).is_none() {
                    open.insert(*rnum, *span);
                }
            }
            ChainElement::Branch { chain, .. } => {
                collect_ring_closures(chain, open);
            }
            ChainElement::BondedAtom { .. } => {}
        }
    }
}
