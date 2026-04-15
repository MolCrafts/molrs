//! Post-parse semantic validation for SMARTS ASTs.
//!
//! SMARTS validation is laxer than SMILES on element symbols — query primitives
//! may stand in for concrete elements — but ring-closure correctness applies
//! equally and is delegated to
//! [`chem::validation::validate_ring_closures`](crate::chem::validation).

use crate::chem::ast::SmilesIR;
use crate::chem::validation::validate_ring_closures;
use crate::error::SmilesError;

/// Validate a parsed SMARTS pattern.
///
/// Currently checks ring-closure pairing only. Deeper query-primitive checks
/// (impossible combinations, contradictory logical queries) are a future
/// enhancement tracked with the SMARTS matcher rollout.
pub fn validate_smarts(mol: &SmilesIR, input: &str) -> Result<(), SmilesError> {
    validate_ring_closures(mol, input)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SmilesErrorKind;
    use crate::parser::parse_smarts;

    #[test]
    fn test_valid_smarts_pattern() {
        let pat = parse_smarts("[C;X4]").unwrap();
        assert!(validate_smarts(&pat, "[C;X4]").is_ok());
    }

    #[test]
    fn test_smarts_ring_closure_mismatch() {
        let pat = parse_smarts("C1CC").unwrap();
        let err = validate_smarts(&pat, "C1CC").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnmatchedRingClosure(1)));
    }
}
