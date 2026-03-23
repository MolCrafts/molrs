//! Post-parse semantic validation for SMILES / SMARTS ASTs.
//!
//! The parser enforces syntactic correctness (balanced brackets, valid grammar).
//! This module adds semantic checks that require knowledge beyond the grammar:
//!
//! * Ring closures must come in matched pairs.
//! * Element symbols must be valid (checked via [`Element::by_symbol`]).
//! * Organic-subset atoms must be from the allowed set.

use std::collections::HashMap;

use super::ast::*;
use super::error::{SmilesError, SmilesErrorKind};
use crate::element::Element;

/// Validate a parsed SMILES molecule.
///
/// Returns `Ok(())` if valid, or the first validation error found.
pub fn validate_smiles(mol: &SmilesIR, input: &str) -> Result<(), SmilesError> {
    validate_ring_closures(mol, input)?;
    validate_elements(mol, input)?;
    Ok(())
}

/// Validate a parsed SMARTS molecule (less strict on element symbols).
pub fn validate_smarts(mol: &SmilesIR, input: &str) -> Result<(), SmilesError> {
    validate_ring_closures(mol, input)?;
    // SMARTS allows query primitives inside brackets, so we skip
    // element validation for Query atoms.
    Ok(())
}

// ---------------------------------------------------------------------------
// Ring-closure validation
// ---------------------------------------------------------------------------

/// Ensure every ring-closure digit is opened and closed exactly once.
fn validate_ring_closures(mol: &SmilesIR, input: &str) -> Result<(), SmilesError> {
    let mut open: HashMap<u16, Span> = HashMap::new();

    for component in &mol.components {
        collect_ring_closures(component, &mut open);
    }

    // Any remaining open closures are unmatched.
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

// ---------------------------------------------------------------------------
// Element validation
// ---------------------------------------------------------------------------

/// Validate that all element symbols refer to real elements.
fn validate_elements(mol: &SmilesIR, input: &str) -> Result<(), SmilesError> {
    for component in &mol.components {
        validate_chain_elements(component, input)?;
    }
    Ok(())
}

fn validate_chain_elements(chain: &Chain, input: &str) -> Result<(), SmilesError> {
    validate_atom_element(&chain.head, input)?;
    for elem in &chain.tail {
        match elem {
            ChainElement::BondedAtom { atom, .. } => {
                validate_atom_element(atom, input)?;
            }
            ChainElement::Branch { chain, .. } => {
                validate_chain_elements(chain, input)?;
            }
            ChainElement::RingClosure { .. } => {}
        }
    }
    Ok(())
}

fn validate_atom_element(atom: &AtomNode, input: &str) -> Result<(), SmilesError> {
    match &atom.spec {
        AtomSpec::Organic { symbol, .. } => {
            // Organic subset symbols are already validated by the parser
            // (only recognised symbols are produced). This is a safety net.
            validate_symbol(symbol, atom.span, input)?;
        }
        AtomSpec::Bracket { symbol, .. } => match symbol {
            BracketSymbol::Element { symbol, .. } => {
                validate_symbol(symbol, atom.span, input)?;
            }
            BracketSymbol::Any | BracketSymbol::Aliphatic | BracketSymbol::Aromatic => {}
        },
        AtomSpec::Wildcard => {}
        AtomSpec::Query(_) => {
            // SMARTS query atoms may contain primitives — skip deep validation
            // here as it is handled by the SMARTS-specific validator.
        }
    }
    Ok(())
}

fn validate_symbol(symbol: &str, span: Span, input: &str) -> Result<(), SmilesError> {
    // Handle aromatic symbols (lowercase single char).
    let lookup = if symbol.len() == 1 && symbol.chars().next().unwrap().is_ascii_lowercase() {
        // Aromatic: capitalise for Element lookup.
        let upper: String = symbol.to_ascii_uppercase();
        Element::by_symbol(&upper)
    } else {
        Element::by_symbol(symbol)
    };

    if lookup.is_none() {
        return Err(SmilesError::new(
            SmilesErrorKind::InvalidElement(symbol.to_owned()),
            span,
            input,
        ));
    }
    Ok(())
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::super::parser::parse_smiles;
    use super::*;

    #[test]
    fn test_valid_smiles() {
        let mol = parse_smiles("C1CCCCC1").unwrap();
        assert!(validate_smiles(&mol, "C1CCCCC1").is_ok());
    }

    #[test]
    fn test_unmatched_ring_closure() {
        let mol = parse_smiles("CC1CC").unwrap();
        let err = validate_smiles(&mol, "CC1CC").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnmatchedRingClosure(1)));
    }

    #[test]
    fn test_valid_elements() {
        let mol = parse_smiles("[Fe+2]").unwrap();
        assert!(validate_smiles(&mol, "[Fe+2]").is_ok());
    }

    #[test]
    fn test_valid_aromatic_element() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        assert!(validate_smiles(&mol, "c1ccccc1").is_ok());
    }

    #[test]
    fn test_multiple_ring_closures() {
        // Naphthalene: two fused rings
        let mol = parse_smiles("c1ccc2ccccc2c1").unwrap();
        assert!(validate_smiles(&mol, "c1ccc2ccccc2c1").is_ok());
    }

    #[test]
    fn test_disconnected_valid() {
        let mol = parse_smiles("[Na+].[Cl-]").unwrap();
        assert!(validate_smiles(&mol, "[Na+].[Cl-]").is_ok());
    }
}
