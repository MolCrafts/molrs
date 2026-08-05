//! Write [`SmilesIR`] back to SMILES / SMARTS strings.
//!
//! Pure syntax: no chemical policy. Graph → IR lives in [`super::from_atomistic`].

use crate::io::smiles::chem::ast::*;
use crate::io::smiles::error::{SmilesError, SmilesErrorKind};

/// Write a concrete SMILES string. Query atoms / SMARTS-only bond ops → Err.
pub fn write_smiles(ir: &SmilesIR) -> Result<String, SmilesError> {
    write_ir(ir, Mode::Smiles)
}

/// Write a SMARTS string (SMILES is a subset).
pub fn write_smarts(ir: &SmilesIR) -> Result<String, SmilesError> {
    write_ir(ir, Mode::Smarts)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Smiles,
    Smarts,
}

fn write_ir(ir: &SmilesIR, mode: Mode) -> Result<String, SmilesError> {
    if ir.components.is_empty() {
        return Err(SmilesError::new(
            SmilesErrorKind::Emit("empty SmilesIR".into()),
            ir.span,
            "",
        ));
    }
    let mut out = String::new();
    for (i, chain) in ir.components.iter().enumerate() {
        if i > 0 {
            out.push('.');
        }
        write_chain(&mut out, chain, mode, /*in_branch*/ false)?;
    }
    Ok(out)
}

fn write_chain(
    out: &mut String,
    chain: &Chain,
    mode: Mode,
    _in_branch: bool,
) -> Result<(), SmilesError> {
    write_atom(out, &chain.head, mode)?;
    for elem in &chain.tail {
        match elem {
            ChainElement::BondedAtom { bond, atom } => {
                write_bond(out, bond.as_ref(), mode, /*omit_default_single*/ true)?;
                write_atom(out, atom, mode)?;
            }
            ChainElement::Branch { bond, chain, .. } => {
                out.push('(');
                write_bond(out, bond.as_ref(), mode, true)?;
                write_chain(out, chain, mode, true)?;
                out.push(')');
            }
            ChainElement::RingClosure { bond, rnum, .. } => {
                write_bond(out, bond.as_ref(), mode, true)?;
                write_rnum(out, *rnum);
            }
        }
    }
    Ok(())
}

fn write_rnum(out: &mut String, rnum: u16) {
    if rnum < 10 {
        out.push(char::from(b'0' + rnum as u8));
    } else {
        out.push('%');
        out.push_str(&rnum.to_string());
    }
}

fn write_atom(out: &mut String, node: &AtomNode, mode: Mode) -> Result<(), SmilesError> {
    match &node.spec {
        AtomSpec::Organic { symbol, aromatic } => {
            if *aromatic {
                for c in symbol.chars() {
                    out.push(c.to_ascii_lowercase());
                }
            } else {
                out.push_str(symbol);
            }
            Ok(())
        }
        AtomSpec::Wildcard => {
            out.push('*');
            Ok(())
        }
        AtomSpec::Bracket {
            isotope,
            symbol,
            chirality,
            hcount,
            charge,
            atom_class,
        } => {
            out.push('[');
            if let Some(iso) = isotope {
                out.push_str(&iso.to_string());
            }
            match symbol {
                BracketSymbol::Element { symbol, aromatic } => {
                    if *aromatic {
                        for c in symbol.chars() {
                            out.push(c.to_ascii_lowercase());
                        }
                    } else {
                        out.push_str(symbol);
                    }
                }
                BracketSymbol::Any => out.push('*'),
                BracketSymbol::Aliphatic => out.push('A'),
                BracketSymbol::Aromatic => out.push('a'),
            }
            if let Some(ch) = chirality {
                match ch {
                    Chirality::CounterClockwise => out.push('@'),
                    Chirality::Clockwise => out.push_str("@@"),
                }
            }
            if let Some(h) = hcount {
                out.push('H');
                if *h != 1 {
                    out.push_str(&h.to_string());
                }
            }
            if let Some(c) = charge {
                write_charge(out, *c);
            }
            if let Some(cls) = atom_class {
                out.push(':');
                out.push_str(&cls.to_string());
            }
            out.push(']');
            Ok(())
        }
        AtomSpec::Query(q) => {
            if mode == Mode::Smiles {
                return Err(SmilesError::new(
                    SmilesErrorKind::InvalidQueryPrimitive(
                        "SMARTS query atoms cannot be written as SMILES".into(),
                    ),
                    node.span,
                    "",
                ));
            }
            out.push('[');
            write_atom_query(out, q)?;
            out.push(']');
            Ok(())
        }
    }
}

fn write_charge(out: &mut String, c: i8) {
    if c == 0 {
        return;
    }
    if c > 0 {
        out.push('+');
        if c != 1 {
            out.push_str(&c.to_string());
        }
    } else {
        out.push('-');
        let a = (-c) as u8;
        if a != 1 {
            out.push_str(&a.to_string());
        }
    }
}

fn write_atom_query(out: &mut String, q: &AtomQuery) -> Result<(), SmilesError> {
    match q {
        AtomQuery::Primitive(p) => write_primitive(out, p),
        AtomQuery::Not(inner) => {
            out.push('!');
            write_atom_query(out, inner)
        }
        AtomQuery::And(parts) => {
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    // implicit AND is fine between primitives; use & for clarity
                    // only when needed — Daylight allows juxtaposition.
                }
                write_atom_query(out, p)?;
            }
            Ok(())
        }
        AtomQuery::Or(parts) => {
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_atom_query(out, p)?;
            }
            Ok(())
        }
        AtomQuery::LowAnd(parts) => {
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.push(';');
                }
                write_atom_query(out, p)?;
            }
            Ok(())
        }
    }
}

fn write_primitive(out: &mut String, p: &AtomPrimitive) -> Result<(), SmilesError> {
    match p {
        AtomPrimitive::Element { symbol, aromatic } => {
            // `#6` is stored as symbol `"#6"` by the parser; write through.
            if symbol.starts_with('#') {
                out.push_str(symbol);
            } else if *aromatic {
                for c in symbol.chars() {
                    out.push(c.to_ascii_lowercase());
                }
            } else {
                out.push_str(symbol);
            }
            Ok(())
        }
        AtomPrimitive::Wildcard => {
            out.push('*');
            Ok(())
        }
        AtomPrimitive::Aliphatic => {
            out.push('A');
            Ok(())
        }
        AtomPrimitive::Aromatic => {
            out.push('a');
            Ok(())
        }
        AtomPrimitive::Degree(n) => {
            out.push('D');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::TotalConnections(n) => {
            out.push('X');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::HCount(n) => {
            out.push('H');
            if *n != 1 {
                out.push_str(&n.to_string());
            }
            Ok(())
        }
        AtomPrimitive::ImplicitH(n) => {
            out.push('h');
            if *n != 1 {
                out.push_str(&n.to_string());
            }
            Ok(())
        }
        AtomPrimitive::RingMembership(None) => {
            out.push('R');
            Ok(())
        }
        AtomPrimitive::RingMembership(Some(n)) => {
            out.push('R');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::RingSize(n) => {
            out.push('r');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::Valence(n) => {
            out.push('v');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::Charge(c) => {
            write_charge(out, *c);
            Ok(())
        }
        AtomPrimitive::Isotope(iso) => {
            out.push_str(&iso.to_string());
            Ok(())
        }
        AtomPrimitive::AtomClass(cls) => {
            out.push(':');
            out.push_str(&cls.to_string());
            Ok(())
        }
        AtomPrimitive::Chirality(ch) => {
            match ch {
                Chirality::CounterClockwise => out.push('@'),
                Chirality::Clockwise => out.push_str("@@"),
            }
            Ok(())
        }
        AtomPrimitive::Recursive(ir) => {
            out.push_str("$(");
            // recursive body is a full SMILES/SMARTS molecule fragment
            let body = write_smarts(ir)?;
            out.push_str(&body);
            out.push(')');
            Ok(())
        }
    }
}

fn write_bond(
    out: &mut String,
    bond: Option<&BondQuery>,
    mode: Mode,
    omit_default_single: bool,
) -> Result<(), SmilesError> {
    let Some(q) = bond else {
        return Ok(());
    };
    match q {
        BondQuery::Kind(k) => {
            write_bond_kind(out, *k, omit_default_single);
            Ok(())
        }
        BondQuery::Not(inner) => {
            if mode == Mode::Smiles {
                return Err(smiles_bond_err());
            }
            out.push('!');
            write_bond(out, Some(inner), mode, false)
        }
        BondQuery::And(parts) | BondQuery::Or(parts) => {
            if mode == Mode::Smiles {
                return Err(smiles_bond_err());
            }
            let sep = if matches!(q, BondQuery::And(_)) {
                '&'
            } else {
                ','
            };
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.push(sep);
                }
                write_bond(out, Some(p), mode, false)?;
            }
            Ok(())
        }
    }
}

fn smiles_bond_err() -> SmilesError {
    SmilesError::new(
        SmilesErrorKind::InvalidQueryPrimitive(
            "SMARTS bond query cannot be written as SMILES".into(),
        ),
        Span::new(0, 0),
        "",
    )
}

fn write_bond_kind(out: &mut String, k: BondKind, omit_default_single: bool) {
    match k {
        BondKind::Single if omit_default_single => {}
        BondKind::Single => out.push('-'),
        BondKind::Double => out.push('='),
        BondKind::Triple => out.push('#'),
        BondKind::Quadruple => out.push('$'),
        BondKind::Aromatic => out.push(':'),
        BondKind::Up => out.push('/'),
        BondKind::Down => out.push('\\'),
        BondKind::Any => out.push('~'),
        BondKind::Ring => out.push('@'),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parser::parse_smarts;
    use crate::io::smiles::smiles::parse_smiles;

    #[test]
    fn write_smiles_ethanol_stable() {
        let ir = parse_smiles("CCO").unwrap();
        let s1 = write_smiles(&ir).unwrap();
        let ir2 = parse_smiles(&s1).unwrap();
        let s2 = write_smiles(&ir2).unwrap();
        assert_eq!(s1, s2);
        assert!(!s1.is_empty());
    }

    #[test]
    fn write_smiles_acetic_and_benzene() {
        for src in ["C(=O)O", "c1ccccc1", "[NH4+]", "CCO.O"] {
            let ir = parse_smiles(src).unwrap();
            let s = write_smiles(&ir).unwrap();
            let ir2 = parse_smiles(&s).unwrap();
            assert_eq!(write_smiles(&ir2).unwrap(), s, "src={src} wrote={s}");
        }
    }

    #[test]
    fn write_smiles_rejects_query() {
        // Explicit OR query cannot be concrete SMILES.
        let ir = parse_smarts("[C,N]").unwrap();
        assert!(write_smiles(&ir).is_err(), "wrote {:?}", write_smiles(&ir));
    }

    #[test]
    fn write_smarts_query_and_recursive() {
        for src in ["[#6;D3]", "[C;$(C=O)]"] {
            let ir = parse_smarts(src).unwrap();
            let s = write_smarts(&ir).unwrap();
            let ir2 = parse_smarts(&s).unwrap();
            let _ = write_smarts(&ir2).unwrap();
        }
    }
}
