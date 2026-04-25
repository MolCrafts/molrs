//! Post-parse semantic validation for SMILES ASTs.
//!
//! The parser enforces syntactic correctness (balanced brackets, valid grammar).
//! This module adds SMILES-specific semantic checks:
//!
//! * Ring closures must come in matched pairs (shared with SMARTS via
//!   [`chem::validation::validate_ring_closures`](crate::smiles::chem::validation)).
//! * Element symbols must refer to real elements (SMILES-specific; SMARTS
//!   permits query primitives in their place).

use crate::smiles::chem::ast::*;
use crate::smiles::chem::validation::validate_ring_closures;
use crate::smiles::error::{SmilesError, SmilesErrorKind};
use molrs::element::Element;

/// Validate a parsed SMILES molecule.
///
/// Returns `Ok(())` if valid, or the first validation error found.
pub fn validate_smiles(mol: &SmilesIR, input: &str) -> Result<(), SmilesError> {
    validate_ring_closures(mol, input)?;
    validate_elements(mol, input)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Element validation (SMILES-specific)
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
    let lookup = if symbol.len() == 1 && symbol.chars().next().unwrap().is_ascii_lowercase() {
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
    use super::*;
    use crate::smiles::parser::parse_smiles;

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
        let mol = parse_smiles("c1ccc2ccccc2c1").unwrap();
        assert!(validate_smiles(&mol, "c1ccc2ccccc2c1").is_ok());
    }

    #[test]
    fn test_disconnected_valid() {
        let mol = parse_smiles("[Na+].[Cl-]").unwrap();
        assert!(validate_smiles(&mol, "[Na+].[Cl-]").is_ok());
    }
}
