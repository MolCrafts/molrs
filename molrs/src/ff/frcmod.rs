//! `frcmod` parameter-delta format.
//!
//! The format is small but it is part of the force-field boundary, so molrs
//! keeps it as a typed object instead of ad-hoc parser code in callers.

use std::fmt::Write;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq)]
pub struct Frcmod {
    pub title: String,
    pub masses: Vec<FrcmodMass>,
    pub bonds: Vec<FrcmodBond>,
    pub angles: Vec<FrcmodAngle>,
    pub dihedrals: Vec<FrcmodDihedral>,
    pub impropers: Vec<FrcmodImproper>,
    pub nonbonded: Vec<FrcmodNonbonded>,
}

impl Frcmod {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            masses: Vec::new(),
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            impropers: Vec::new(),
            nonbonded: Vec::new(),
        }
    }

    pub fn parse_str(input: &str) -> Result<Self, String> {
        let mut out = Self::new("");
        let mut title = Vec::new();
        let mut section = Section::Preamble;

        for (idx, raw) in input.lines().enumerate() {
            let lineno = idx + 1;
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            if let Some(next) = Section::from_heading(line) {
                section = next;
                continue;
            }

            match section {
                Section::Preamble => title.push(line.to_owned()),
                Section::Mass => out.masses.push(parse_mass(line, lineno)?),
                Section::Bond => out.bonds.push(parse_bond(line, lineno)?),
                Section::Angle => out.angles.push(parse_angle(line, lineno)?),
                Section::Dihedral => out.dihedrals.push(parse_dihedral(line, lineno)?),
                Section::Improper => out.impropers.push(parse_improper(line, lineno)?),
                Section::Nonbonded => out.nonbonded.push(parse_nonbonded(line, lineno)?),
            }
        }

        out.title = title.join(" ");
        Ok(out)
    }

    pub fn write_string(&self) -> String {
        let mut out = String::new();
        if self.title.trim().is_empty() {
            out.push_str("frcmod\n");
        } else {
            out.push_str(self.title.trim());
            out.push('\n');
        }

        if !self.masses.is_empty() {
            out.push_str("\nMASS\n");
            for row in &self.masses {
                match row.polarizability {
                    Some(pol) => {
                        let _ =
                            writeln!(out, "{:<8} {:>12.6} {:>12.6}", row.atom_type, row.mass, pol);
                    }
                    None => {
                        let _ = writeln!(out, "{:<8} {:>12.6}", row.atom_type, row.mass);
                    }
                }
            }
        }

        if !self.bonds.is_empty() {
            out.push_str("\nBOND\n");
            for row in &self.bonds {
                let _ = writeln!(
                    out,
                    "{:<17} {:>12.6} {:>12.6}",
                    pattern(&[&row.i, &row.j]),
                    row.force_constant,
                    row.equilibrium
                );
            }
        }

        if !self.angles.is_empty() {
            out.push_str("\nANGLE\n");
            for row in &self.angles {
                let _ = writeln!(
                    out,
                    "{:<17} {:>12.6} {:>12.6}",
                    pattern(&[&row.i, &row.j, &row.k]),
                    row.force_constant,
                    row.equilibrium_deg
                );
            }
        }

        if !self.dihedrals.is_empty() {
            out.push_str("\nDIHE\n");
            for row in &self.dihedrals {
                let _ = writeln!(
                    out,
                    "{:<17} {:>6.0} {:>12.6} {:>12.6} {:>12.6}",
                    pattern(&[&row.i, &row.j, &row.k, &row.l]),
                    row.divisor,
                    row.barrier,
                    row.phase_deg,
                    row.periodicity
                );
            }
        }

        if !self.impropers.is_empty() {
            out.push_str("\nIMPROPER\n");
            for row in &self.impropers {
                let _ = writeln!(
                    out,
                    "{:<17} {:>12.6} {:>12.6} {:>12.6}",
                    pattern(&[&row.i, &row.j, &row.k, &row.l]),
                    row.barrier,
                    row.phase_deg,
                    row.periodicity
                );
            }
        }

        if !self.nonbonded.is_empty() {
            out.push_str("\nNONBON\n");
            for row in &self.nonbonded {
                let _ = writeln!(
                    out,
                    "{:<8} {:>12.6} {:>12.6}",
                    row.atom_type, row.radius, row.epsilon
                );
            }
        }

        out
    }
}

impl FromStr for Frcmod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_str(s)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrcmodMass {
    pub atom_type: String,
    pub mass: f64,
    pub polarizability: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrcmodBond {
    pub i: String,
    pub j: String,
    pub force_constant: f64,
    pub equilibrium: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrcmodAngle {
    pub i: String,
    pub j: String,
    pub k: String,
    pub force_constant: f64,
    pub equilibrium_deg: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrcmodDihedral {
    pub i: String,
    pub j: String,
    pub k: String,
    pub l: String,
    pub divisor: f64,
    pub barrier: f64,
    pub phase_deg: f64,
    pub periodicity: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrcmodImproper {
    pub i: String,
    pub j: String,
    pub k: String,
    pub l: String,
    pub barrier: f64,
    pub phase_deg: f64,
    pub periodicity: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrcmodNonbonded {
    pub atom_type: String,
    pub radius: f64,
    pub epsilon: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Preamble,
    Mass,
    Bond,
    Angle,
    Dihedral,
    Improper,
    Nonbonded,
}

impl Section {
    fn from_heading(line: &str) -> Option<Self> {
        let heading = line.split_whitespace().next()?.to_ascii_uppercase();
        match heading.as_str() {
            "MASS" => Some(Self::Mass),
            "BOND" => Some(Self::Bond),
            "ANGLE" => Some(Self::Angle),
            "DIHE" | "DIHEDRAL" => Some(Self::Dihedral),
            "IMPROPER" | "IMPR" => Some(Self::Improper),
            "NONBON" | "NONB" | "NONBONDED" => Some(Self::Nonbonded),
            _ => None,
        }
    }
}

fn strip_comment(line: &str) -> &str {
    let hash = line.find('#').unwrap_or(line.len());
    let bang = line.find('!').unwrap_or(line.len());
    &line[..hash.min(bang)]
}

fn parse_mass(line: &str, lineno: usize) -> Result<FrcmodMass, String> {
    let fields: Vec<_> = line.split_whitespace().collect();
    if fields.len() < 2 {
        return Err(format!("invalid MASS row at line {lineno}: {line}"));
    }
    Ok(FrcmodMass {
        atom_type: fields[0].to_owned(),
        mass: parse_f64(fields[1], "MASS mass", lineno)?,
        polarizability: fields.get(2).and_then(|v| v.parse::<f64>().ok()),
    })
}

fn parse_bond(line: &str, lineno: usize) -> Result<FrcmodBond, String> {
    let (atoms, values) = parse_labeled_values(line, 2, 2, "BOND", lineno)?;
    Ok(FrcmodBond {
        i: atoms[0].clone(),
        j: atoms[1].clone(),
        force_constant: values[0],
        equilibrium: values[1],
    })
}

fn parse_angle(line: &str, lineno: usize) -> Result<FrcmodAngle, String> {
    let (atoms, values) = parse_labeled_values(line, 3, 2, "ANGLE", lineno)?;
    Ok(FrcmodAngle {
        i: atoms[0].clone(),
        j: atoms[1].clone(),
        k: atoms[2].clone(),
        force_constant: values[0],
        equilibrium_deg: values[1],
    })
}

fn parse_dihedral(line: &str, lineno: usize) -> Result<FrcmodDihedral, String> {
    let (atoms, values) = parse_labeled_values(line, 4, 4, "DIHE", lineno)?;
    Ok(FrcmodDihedral {
        i: atoms[0].clone(),
        j: atoms[1].clone(),
        k: atoms[2].clone(),
        l: atoms[3].clone(),
        divisor: values[0],
        barrier: values[1],
        phase_deg: values[2],
        periodicity: values[3],
    })
}

fn parse_improper(line: &str, lineno: usize) -> Result<FrcmodImproper, String> {
    let (atoms, values) = parse_labeled_values(line, 4, 3, "IMPROPER", lineno)?;
    Ok(FrcmodImproper {
        i: atoms[0].clone(),
        j: atoms[1].clone(),
        k: atoms[2].clone(),
        l: atoms[3].clone(),
        barrier: values[0],
        phase_deg: values[1],
        periodicity: values[2],
    })
}

fn parse_nonbonded(line: &str, lineno: usize) -> Result<FrcmodNonbonded, String> {
    let fields: Vec<_> = line.split_whitespace().collect();
    if fields.len() < 3 {
        return Err(format!("invalid NONBON row at line {lineno}: {line}"));
    }
    Ok(FrcmodNonbonded {
        atom_type: fields[0].to_owned(),
        radius: parse_f64(fields[1], "NONBON radius", lineno)?,
        epsilon: parse_f64(fields[2], "NONBON epsilon", lineno)?,
    })
}

fn parse_labeled_values(
    line: &str,
    arity: usize,
    nvalues: usize,
    section: &str,
    lineno: usize,
) -> Result<(Vec<String>, Vec<f64>), String> {
    let fields: Vec<_> = line.split_whitespace().collect();
    for split in 1..fields.len() {
        if fields.len() < split + nvalues {
            break;
        }
        let mut values = Vec::with_capacity(nvalues);
        let mut all_numeric = true;
        for field in &fields[split..split + nvalues] {
            match field.parse::<f64>() {
                Ok(v) => values.push(v),
                Err(_) => {
                    all_numeric = false;
                    break;
                }
            }
        }
        if !all_numeric {
            continue;
        }
        let label = fields[..split].join("");
        if let Ok(atoms) = parse_pattern(&label, arity) {
            return Ok((atoms, values));
        }
    }
    Err(format!("invalid {section} row at line {lineno}: {line}"))
}

fn parse_pattern(label: &str, arity: usize) -> Result<Vec<String>, String> {
    let atoms: Vec<String> = label
        .split('-')
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect();
    if atoms.len() == arity {
        Ok(atoms)
    } else {
        Err(format!("expected {arity} atom-type labels in `{label}`"))
    }
}

fn parse_f64(input: &str, context: &str, lineno: usize) -> Result<f64, String> {
    input
        .parse::<f64>()
        .map_err(|e| format!("invalid {context} at line {lineno}: {e}"))
}

fn pattern(atoms: &[&str]) -> String {
    atoms.join("-")
}
