//! LAMMPS data file format reader and writer.
//!
//! Implements support for LAMMPS data files as specified in:
//! https://docs.lammps.org/read_data.html
//!
//! # Supported Features
//!
//! - Header section: atom/bond counts, box dimensions, type counts
//! - Atoms section: atomic, full, molecular, charge styles
//! - Bonds section
//! - Type Labels sections: Atom Type Labels, Bond Type Labels
//! - Orthorhombic and triclinic simulation boxes
//!
//! # Unsupported Features (can be added later)
//!
//! - Coeffs sections (Pair Coeffs, Bond Coeffs, etc.)
//! - Angles, Dihedrals, Impropers sections
//! - Velocities section
//! - Special atom styles (ellipsoid, sphere, etc.)
//!
//! # Examples
//!
//! ```no_run
//! use molrs::io::lammps_data::{read_lammps_data, write_lammps_data};
//!
//! # fn main() -> std::io::Result<()> {
//! // Read LAMMPS data file
//! let frame = read_lammps_data("system.data")?;
//!
//! // Access atoms
//! let atoms = frame.get("atoms").expect("atoms block");
//! let x = atoms.get_float("x").expect("x coordinates");
//!
//! // Write to new file
//! write_lammps_data("output.data", &frame)?;
//! # Ok(())
//! # }```

use crate::block::Block;
use crate::frame::Frame;
use crate::io::reader::{FrameReader, Reader};
use crate::io::writer::FrameWriter;
use crate::types::{F, I, U};
use ndarray::{Array1, IxDyn};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::Path;

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse a line into whitespace-separated tokens
fn tokenize(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

fn err_mapper<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

// ============================================================================
// Data Structures
// ============================================================================

/// Header information from LAMMPS data file
#[derive(Debug, Clone, Default)]
struct LAMMPSHeader {
    num_atoms: usize,
    num_bonds: usize,
    num_angles: usize,
    num_dihedrals: usize,
    num_atom_types: usize,
    num_bond_types: usize,
    num_angle_types: usize,
    num_dihedral_types: usize,
    xlo: f64,
    xhi: f64,
    ylo: f64,
    yhi: f64,
    zlo: f64,
    zhi: f64,
    xy: Option<f64>,
    xz: Option<f64>,
    yz: Option<f64>,
}

/// Atom data from Atoms section
#[derive(Debug, Clone)]
struct AtomData {
    id: I,
    molecule_id: Option<I>,
    atom_type: String, // Can be numeric or label
    charge: Option<F>,
    x: F,
    y: F,
    z: F,
}

/// Bond data from Bonds section
#[derive(Debug, Clone)]
struct BondData {
    _id: I,
    bond_type: String, // Can be numeric or label
    atom_i: I,
    atom_j: I,
}

/// Angle data from Angles section
#[derive(Debug, Clone)]
struct AngleData {
    _id: I,
    angle_type: String, // Can be numeric or label
    atom_i: I,
    atom_j: I,
    atom_k: I,
}

/// Dihedral data from Dihedrals section
#[derive(Debug, Clone)]
struct DihedralData {
    _id: I,
    dihedral_type: String, // Can be numeric or label
    atom_i: I,
    atom_j: I,
    atom_k: I,
    atom_l: I,
}

// ============================================================================
// Parser Functions
// ============================================================================

/// Parse header section and return the first section line encountered
fn parse_header_with_first_section<R: BufRead>(
    reader: &mut R,
) -> std::io::Result<(LAMMPSHeader, Option<String>)> {
    let mut header = LAMMPSHeader::default();
    let mut line = String::new();

    // Skip first line (comment)
    reader.read_line(&mut line)?;
    line.clear();

    loop {
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            return Ok((header, None));
        }

        let trimmed = line.trim();

        // Skip blank lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            line.clear();
            continue;
        }

        // Check if this looks like a section header (capitalized word followed by optional comment)
        // Section headers: "Atoms", "Bonds", "Atom Type Labels", etc.
        // Header keywords end with lowercase: "atoms", "bonds", "xlo xhi", etc.
        let tokens = tokenize(trimmed);
        if tokens.is_empty() {
            line.clear();
            continue;
        }

        // If first token starts with uppercase and is not followed by a number, it's likely a section
        if let Some(first_char) = tokens[0].chars().next()
            && first_char.is_uppercase()
        {
            // This is a section header, return it along with the header
            return Ok((header, Some(line.clone())));
        }

        // Parse header keywords
        match tokens.last() {
            Some(&"atoms") if tokens.len() >= 2 => {
                header.num_atoms = tokens[0].parse().map_err(err_mapper)?;
            }
            Some(&"bonds") if tokens.len() >= 2 => {
                header.num_bonds = tokens[0].parse().map_err(err_mapper)?;
            }
            Some(&"angles") if tokens.len() >= 2 => {
                header.num_angles = tokens[0].parse().map_err(err_mapper)?;
            }
            Some(&"dihedrals") if tokens.len() >= 2 => {
                header.num_dihedrals = tokens[0].parse().map_err(err_mapper)?;
            }
            Some(&"types") if tokens.len() >= 3 => {
                let count: usize = tokens[0].parse().map_err(err_mapper)?;
                match tokens[1] {
                    "atom" => header.num_atom_types = count,
                    "bond" => header.num_bond_types = count,
                    "angle" => header.num_angle_types = count,
                    "dihedral" => header.num_dihedral_types = count,
                    _ => {}
                }
            }
            Some(&"xhi") if tokens.len() >= 4 && tokens[2] == "xlo" => {
                header.xlo = tokens[0].parse().map_err(err_mapper)?;
                header.xhi = tokens[1].parse().map_err(err_mapper)?;
            }
            Some(&"yhi") if tokens.len() >= 4 && tokens[2] == "ylo" => {
                header.ylo = tokens[0].parse().map_err(err_mapper)?;
                header.yhi = tokens[1].parse().map_err(err_mapper)?;
            }
            Some(&"zhi") if tokens.len() >= 4 && tokens[2] == "zlo" => {
                header.zlo = tokens[0].parse().map_err(err_mapper)?;
                header.zhi = tokens[1].parse().map_err(err_mapper)?;
            }
            Some(&"yz") if tokens.len() >= 6 && tokens[3] == "xy" && tokens[4] == "xz" => {
                header.xy = Some(tokens[0].parse().map_err(err_mapper)?);
                header.xz = Some(tokens[1].parse().map_err(err_mapper)?);
                header.yz = Some(tokens[2].parse().map_err(err_mapper)?);
            }
            _ => {}
        }

        line.clear();
    }
}

/// Parse Type Labels section and return the next section line if encountered
fn parse_type_labels<R: BufRead>(
    reader: &mut R,
) -> std::io::Result<(HashMap<String, String>, Option<String>)> {
    let mut labels = HashMap::new();
    let mut line = String::new();

    loop {
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            return Ok((labels, None));
        }

        let trimmed = line.trim();

        // Stop at blank line or new section
        if trimmed.is_empty() {
            line.clear();
            continue;
        }

        // Check if this is a new section header
        if trimmed.chars().next().is_some_and(|c| c.is_uppercase()) {
            // Check if it looks like a section header (not a type label)
            let tokens = tokenize(trimmed);
            if tokens.len() >= 2 {
                // Could be either "Atoms" or "1 C" (type label)
                // Type labels have numeric first token
                if tokens[0].parse::<i64>().is_err() {
                    // First token is not a number, so it's a section header
                    return Ok((labels, Some(line.clone())));
                }
            } else if tokens.len() == 1 {
                // Single word starting with uppercase is a section header
                return Ok((labels, Some(line.clone())));
            }
        }

        let tokens = tokenize(trimmed);
        if tokens.len() >= 2 {
            let type_id = tokens[0].to_string();
            let label = tokens[1].to_string();
            labels.insert(type_id, label);
        }

        line.clear();
    }
}

/// Parse Atoms section
fn parse_atoms<R: BufRead>(
    reader: &mut R,
    num_atoms: usize,
    _atom_type_labels: &HashMap<String, String>,
) -> std::io::Result<Vec<AtomData>> {
    let mut atoms = Vec::with_capacity(num_atoms);
    let mut line = String::new();

    while atoms.len() < num_atoms {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let tokens = tokenize(trimmed);

        // Detect atom style by number of columns
        // atomic: id type x y z (5)
        // charge: id type q x y z (6)
        // molecular: id mol-id type x y z (6)
        // full: id mol-id type q x y z (7)

        let atom = if tokens.len() == 5 {
            // atomic style
            AtomData {
                id: tokens[0].parse::<I>().map_err(err_mapper)?,
                molecule_id: None,
                atom_type: tokens[1].to_string(),
                charge: None,
                x: tokens[2].parse::<F>().map_err(err_mapper)?,
                y: tokens[3].parse::<F>().map_err(err_mapper)?,
                z: tokens[4].parse::<F>().map_err(err_mapper)?,
            }
        } else if tokens.len() == 6 {
            // Could be charge or molecular - try to detect
            // If token[2] parses as float, it's charge style
            if tokens[2].parse::<F>().is_ok() {
                // charge style
                AtomData {
                    id: tokens[0].parse::<I>().map_err(err_mapper)?,
                    molecule_id: None,
                    atom_type: tokens[1].to_string(),
                    charge: Some(tokens[2].parse::<F>().map_err(err_mapper)?),
                    x: tokens[3].parse::<F>().map_err(err_mapper)?,
                    y: tokens[4].parse::<F>().map_err(err_mapper)?,
                    z: tokens[5].parse::<F>().map_err(err_mapper)?,
                }
            } else {
                // molecular style
                AtomData {
                    id: tokens[0].parse::<I>().map_err(err_mapper)?,
                    molecule_id: Some(tokens[1].parse::<I>().map_err(err_mapper)?),
                    atom_type: tokens[2].to_string(),
                    charge: None,
                    x: tokens[3].parse::<F>().map_err(err_mapper)?,
                    y: tokens[4].parse::<F>().map_err(err_mapper)?,
                    z: tokens[5].parse::<F>().map_err(err_mapper)?,
                }
            }
        } else if tokens.len() >= 7 {
            // full style
            AtomData {
                id: tokens[0].parse::<I>().map_err(err_mapper)?,
                molecule_id: Some(tokens[1].parse::<I>().map_err(err_mapper)?),
                atom_type: tokens[2].to_string(),
                charge: Some(tokens[3].parse::<F>().map_err(err_mapper)?),
                x: tokens[4].parse::<F>().map_err(err_mapper)?,
                y: tokens[5].parse::<F>().map_err(err_mapper)?,
                z: tokens[6].parse::<F>().map_err(err_mapper)?,
            }
        } else {
            return Err(err_mapper(format!(
                "Invalid Atoms line: expected 5-7 columns, got {}",
                tokens.len()
            )));
        };

        atoms.push(atom);
        line.clear();
    }

    Ok(atoms)
}

/// Parse Bonds section
fn parse_bonds<R: BufRead>(
    reader: &mut R,
    num_bonds: usize,
    _bond_type_labels: &HashMap<String, String>,
) -> std::io::Result<Vec<BondData>> {
    let mut bonds = Vec::with_capacity(num_bonds);
    let mut line = String::new();

    while bonds.len() < num_bonds {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let tokens = tokenize(trimmed);
        if tokens.len() < 4 {
            return Err(err_mapper(format!(
                "Invalid Bonds line: expected 4 columns, got {}",
                tokens.len()
            )));
        }

        let bond = BondData {
            _id: tokens[0].parse::<I>().map_err(err_mapper)?,
            bond_type: tokens[1].to_string(),
            atom_i: tokens[2].parse::<I>().map_err(err_mapper)?,
            atom_j: tokens[3].parse::<I>().map_err(err_mapper)?,
        };

        bonds.push(bond);
        line.clear();
    }

    Ok(bonds)
}

/// Parse Angles section
fn parse_angles<R: BufRead>(
    reader: &mut R,
    num_angles: usize,
    _angle_type_labels: &HashMap<String, String>,
) -> std::io::Result<Vec<AngleData>> {
    let mut angles = Vec::with_capacity(num_angles);
    let mut line = String::new();

    while angles.len() < num_angles {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let tokens = tokenize(trimmed);
        if tokens.len() < 5 {
            return Err(err_mapper(format!(
                "Invalid Angles line: expected 5 columns, got {}",
                tokens.len()
            )));
        }

        let angle = AngleData {
            _id: tokens[0].parse::<I>().map_err(err_mapper)?,
            angle_type: tokens[1].to_string(),
            atom_i: tokens[2].parse::<I>().map_err(err_mapper)?,
            atom_j: tokens[3].parse::<I>().map_err(err_mapper)?,
            atom_k: tokens[4].parse::<I>().map_err(err_mapper)?,
        };

        angles.push(angle);
    }

    Ok(angles)
}

/// Parse Dihedrals section
fn parse_dihedrals<R: BufRead>(
    reader: &mut R,
    num_dihedrals: usize,
    _dihedral_type_labels: &HashMap<String, String>,
) -> std::io::Result<Vec<DihedralData>> {
    let mut dihedrals = Vec::with_capacity(num_dihedrals);
    let mut line = String::new();

    while dihedrals.len() < num_dihedrals {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let tokens = tokenize(trimmed);
        if tokens.len() < 6 {
            return Err(err_mapper(format!(
                "Invalid Dihedrals line: expected 6 columns, got {}",
                tokens.len()
            )));
        }

        let dihedral = DihedralData {
            _id: tokens[0].parse::<I>().map_err(err_mapper)?,
            dihedral_type: tokens[1].to_string(),
            atom_i: tokens[2].parse::<I>().map_err(err_mapper)?,
            atom_j: tokens[3].parse::<I>().map_err(err_mapper)?,
            atom_k: tokens[4].parse::<I>().map_err(err_mapper)?,
            atom_l: tokens[5].parse::<I>().map_err(err_mapper)?,
        };

        dihedrals.push(dihedral);
    }

    Ok(dihedrals)
}

/// Build Frame from parsed data
#[allow(clippy::too_many_arguments)]
fn build_frame(
    header: &LAMMPSHeader,
    atoms: Vec<AtomData>,
    bonds: Vec<BondData>,
    angles: Vec<AngleData>,
    dihedrals: Vec<DihedralData>,
    atom_type_labels: HashMap<String, String>,
    bond_type_labels: HashMap<String, String>,
    angle_type_labels: HashMap<String, String>,
    dihedral_type_labels: HashMap<String, String>,
) -> std::io::Result<Frame> {
    let mut frame = Frame::new();

    // Build atoms block
    if !atoms.is_empty() {
        let n = atoms.len();
        let mut atom_block = Block::new();

        let mut x_vec = Vec::with_capacity(n);
        let mut y_vec = Vec::with_capacity(n);
        let mut z_vec = Vec::with_capacity(n);
        let mut id_vec = Vec::with_capacity(n);
        let mut type_vec = Vec::with_capacity(n);
        let mut charge_vec = Vec::with_capacity(n);
        let mut mol_id_vec = Vec::with_capacity(n);

        let mut has_charge = false;
        let mut has_mol_id = false;

        // Build atom ID to index map for bonds
        let mut atom_id_map = HashMap::new();

        for (idx, atom) in atoms.iter().enumerate() {
            x_vec.push(atom.x);
            y_vec.push(atom.y);
            z_vec.push(atom.z);
            id_vec.push(atom.id);

            // Convert type label to numeric if it's a label
            let type_num = if let Ok(num) = atom.atom_type.parse::<I>() {
                num
            } else {
                // It's a label, find the numeric type
                atom_type_labels
                    .iter()
                    .find(|(_, label)| *label == &atom.atom_type)
                    .and_then(|(id, _)| id.parse::<I>().ok())
                    .unwrap_or(1)
            };
            type_vec.push(type_num);

            if let Some(q) = atom.charge {
                charge_vec.push(q);
                has_charge = true;
            } else {
                charge_vec.push(0.0 as F);
            }

            if let Some(mol_id) = atom.molecule_id {
                mol_id_vec.push(mol_id);
                has_mol_id = true;
            } else {
                mol_id_vec.push(0);
            }

            atom_id_map.insert(atom.id, idx as U);
        }

        // Insert coordinate arrays
        let to_arr_float = |v: Vec<F>| -> std::io::Result<ndarray::ArrayD<F>> {
            Ok(Array1::from_vec(v)
                .into_shape_with_order(IxDyn(&[n]))
                .map_err(err_mapper)?
                .into_dyn())
        };
        let to_arr_int = |v: Vec<I>| -> std::io::Result<ndarray::ArrayD<I>> {
            Ok(Array1::from_vec(v)
                .into_shape_with_order(IxDyn(&[n]))
                .map_err(err_mapper)?
                .into_dyn())
        };

        atom_block
            .insert("x", to_arr_float(x_vec)?)
            .map_err(err_mapper)?;
        atom_block
            .insert("y", to_arr_float(y_vec)?)
            .map_err(err_mapper)?;
        atom_block
            .insert("z", to_arr_float(z_vec)?)
            .map_err(err_mapper)?;
        atom_block
            .insert("id", to_arr_int(id_vec)?)
            .map_err(err_mapper)?;
        atom_block
            .insert("type", to_arr_int(type_vec)?)
            .map_err(err_mapper)?;

        if has_charge {
            atom_block
                .insert("charge", to_arr_float(charge_vec)?)
                .map_err(err_mapper)?;
        }
        if has_mol_id {
            atom_block
                .insert("molecule_id", to_arr_int(mol_id_vec)?)
                .map_err(err_mapper)?;
        }

        frame.insert("atoms", atom_block);

        // Build bonds block
        if !bonds.is_empty() {
            let bn = bonds.len();
            let mut bond_block = Block::new();

            let mut atom_i_vec = Vec::with_capacity(bn);
            let mut atom_j_vec = Vec::with_capacity(bn);
            let mut bond_type_vec = Vec::with_capacity(bn);

            for bond in &bonds {
                // Map atom IDs to indices
                let idx_i = *atom_id_map.get(&bond.atom_i).ok_or_else(|| {
                    err_mapper(format!("Bond references unknown atom ID: {}", bond.atom_i))
                })?;
                let idx_j = *atom_id_map.get(&bond.atom_j).ok_or_else(|| {
                    err_mapper(format!("Bond references unknown atom ID: {}", bond.atom_j))
                })?;

                atom_i_vec.push(idx_i);
                atom_j_vec.push(idx_j);

                // Convert bond type label to numeric if it's a label
                let type_num = if let Ok(num) = bond.bond_type.parse::<I>() {
                    num
                } else {
                    bond_type_labels
                        .iter()
                        .find(|(_, label)| *label == &bond.bond_type)
                        .and_then(|(id, _)| id.parse::<I>().ok())
                        .unwrap_or(1)
                };
                bond_type_vec.push(type_num);
            }

            let mk_barr_u = |v: Vec<U>| -> std::io::Result<ndarray::ArrayD<U>> {
                Array1::from_vec(v)
                    .into_shape_with_order(IxDyn(&[bn]))
                    .map_err(err_mapper)
                    .map(|a| a.into_dyn())
            };
            let mk_barr_i = |v: Vec<I>| -> std::io::Result<ndarray::ArrayD<I>> {
                Array1::from_vec(v)
                    .into_shape_with_order(IxDyn(&[bn]))
                    .map_err(err_mapper)
                    .map(|a| a.into_dyn())
            };

            bond_block
                .insert("atomi", mk_barr_u(atom_i_vec)?)
                .map_err(err_mapper)?;
            bond_block
                .insert("atomj", mk_barr_u(atom_j_vec)?)
                .map_err(err_mapper)?;
            bond_block
                .insert("type", mk_barr_i(bond_type_vec)?)
                .map_err(err_mapper)?;

            frame.insert("bonds", bond_block);
        }

        // Build angles block
        if !angles.is_empty() {
            let an = angles.len();
            let mut angle_block = Block::new();

            let mut atom_i_vec = Vec::with_capacity(an);
            let mut atom_j_vec = Vec::with_capacity(an);
            let mut atom_k_vec = Vec::with_capacity(an);
            let mut angle_type_vec = Vec::with_capacity(an);

            for angle in &angles {
                let idx_i = *atom_id_map.get(&angle.atom_i).ok_or_else(|| {
                    err_mapper(format!(
                        "Angle references unknown atom ID: {}",
                        angle.atom_i
                    ))
                })?;
                let idx_j = *atom_id_map.get(&angle.atom_j).ok_or_else(|| {
                    err_mapper(format!(
                        "Angle references unknown atom ID: {}",
                        angle.atom_j
                    ))
                })?;
                let idx_k = *atom_id_map.get(&angle.atom_k).ok_or_else(|| {
                    err_mapper(format!(
                        "Angle references unknown atom ID: {}",
                        angle.atom_k
                    ))
                })?;

                atom_i_vec.push(idx_i);
                atom_j_vec.push(idx_j);
                atom_k_vec.push(idx_k);

                let type_num = if let Ok(num) = angle.angle_type.parse::<I>() {
                    num
                } else {
                    angle_type_labels
                        .iter()
                        .find(|(_, label)| *label == &angle.angle_type)
                        .and_then(|(id, _)| id.parse::<I>().ok())
                        .unwrap_or(1)
                };
                angle_type_vec.push(type_num);
            }

            let mk_aarr_u = |v: Vec<U>| -> std::io::Result<ndarray::ArrayD<U>> {
                Array1::from_vec(v)
                    .into_shape_with_order(IxDyn(&[an]))
                    .map_err(err_mapper)
                    .map(|a| a.into_dyn())
            };
            let mk_aarr_i = |v: Vec<I>| -> std::io::Result<ndarray::ArrayD<I>> {
                Array1::from_vec(v)
                    .into_shape_with_order(IxDyn(&[an]))
                    .map_err(err_mapper)
                    .map(|a| a.into_dyn())
            };

            angle_block
                .insert("atomi", mk_aarr_u(atom_i_vec)?)
                .map_err(err_mapper)?;
            angle_block
                .insert("atomj", mk_aarr_u(atom_j_vec)?)
                .map_err(err_mapper)?;
            angle_block
                .insert("atomk", mk_aarr_u(atom_k_vec)?)
                .map_err(err_mapper)?;
            angle_block
                .insert("type", mk_aarr_i(angle_type_vec)?)
                .map_err(err_mapper)?;

            frame.insert("angles", angle_block);
        }

        // Build dihedrals block
        if !dihedrals.is_empty() {
            let dn = dihedrals.len();
            let mut dihedral_block = Block::new();

            let mut atom_i_vec = Vec::with_capacity(dn);
            let mut atom_j_vec = Vec::with_capacity(dn);
            let mut atom_k_vec = Vec::with_capacity(dn);
            let mut atom_l_vec = Vec::with_capacity(dn);
            let mut dihedral_type_vec = Vec::with_capacity(dn);

            for dihedral in &dihedrals {
                let idx_i = *atom_id_map.get(&dihedral.atom_i).ok_or_else(|| {
                    err_mapper(format!(
                        "Dihedral references unknown atom ID: {}",
                        dihedral.atom_i
                    ))
                })?;
                let idx_j = *atom_id_map.get(&dihedral.atom_j).ok_or_else(|| {
                    err_mapper(format!(
                        "Dihedral references unknown atom ID: {}",
                        dihedral.atom_j
                    ))
                })?;
                let idx_k = *atom_id_map.get(&dihedral.atom_k).ok_or_else(|| {
                    err_mapper(format!(
                        "Dihedral references unknown atom ID: {}",
                        dihedral.atom_k
                    ))
                })?;
                let idx_l = *atom_id_map.get(&dihedral.atom_l).ok_or_else(|| {
                    err_mapper(format!(
                        "Dihedral references unknown atom ID: {}",
                        dihedral.atom_l
                    ))
                })?;

                atom_i_vec.push(idx_i);
                atom_j_vec.push(idx_j);
                atom_k_vec.push(idx_k);
                atom_l_vec.push(idx_l);

                let type_num = if let Ok(num) = dihedral.dihedral_type.parse::<I>() {
                    num
                } else {
                    dihedral_type_labels
                        .iter()
                        .find(|(_, label)| *label == &dihedral.dihedral_type)
                        .and_then(|(id, _)| id.parse::<I>().ok())
                        .unwrap_or(1)
                };
                dihedral_type_vec.push(type_num);
            }

            let mk_darr_u = |v: Vec<U>| -> std::io::Result<ndarray::ArrayD<U>> {
                Array1::from_vec(v)
                    .into_shape_with_order(IxDyn(&[dn]))
                    .map_err(err_mapper)
                    .map(|a| a.into_dyn())
            };
            let mk_darr_i = |v: Vec<I>| -> std::io::Result<ndarray::ArrayD<I>> {
                Array1::from_vec(v)
                    .into_shape_with_order(IxDyn(&[dn]))
                    .map_err(err_mapper)
                    .map(|a| a.into_dyn())
            };

            dihedral_block
                .insert("atomi", mk_darr_u(atom_i_vec)?)
                .map_err(err_mapper)?;
            dihedral_block
                .insert("atomj", mk_darr_u(atom_j_vec)?)
                .map_err(err_mapper)?;
            dihedral_block
                .insert("atomk", mk_darr_u(atom_k_vec)?)
                .map_err(err_mapper)?;
            dihedral_block
                .insert("atoml", mk_darr_u(atom_l_vec)?)
                .map_err(err_mapper)?;
            dihedral_block
                .insert("type", mk_darr_i(dihedral_type_vec)?)
                .map_err(err_mapper)?;

            frame.insert("dihedrals", dihedral_block);
        }
    }

    // Add box metadata
    let lx = header.xhi - header.xlo;
    let ly = header.yhi - header.ylo;
    let lz = header.zhi - header.zlo;
    frame
        .meta
        .insert("box".to_string(), format!("{} {} {}", lx, ly, lz));
    frame.meta.insert(
        "box_origin".to_string(),
        format!("{} {} {}", header.xlo, header.ylo, header.zlo),
    );

    if let (Some(xy), Some(xz), Some(yz)) = (header.xy, header.xz, header.yz) {
        frame
            .meta
            .insert("box_tilt".to_string(), format!("{} {} {}", xy, xz, yz));
    }

    // Add type labels to metadata
    if !atom_type_labels.is_empty() {
        let labels_str = atom_type_labels
            .iter()
            .map(|(id, label)| format!("{}:{}", id, label))
            .collect::<Vec<_>>()
            .join(",");
        frame
            .meta
            .insert("atom_type_labels".to_string(), labels_str);
    }

    if !bond_type_labels.is_empty() {
        let labels_str = bond_type_labels
            .iter()
            .map(|(id, label)| format!("{}:{}", id, label))
            .collect::<Vec<_>>()
            .join(",");
        frame
            .meta
            .insert("bond_type_labels".to_string(), labels_str);
    }

    if !angle_type_labels.is_empty() {
        let labels_str = angle_type_labels
            .iter()
            .map(|(id, label)| format!("{}:{}", id, label))
            .collect::<Vec<_>>()
            .join(",");
        frame
            .meta
            .insert("angle_type_labels".to_string(), labels_str);
    }

    if !dihedral_type_labels.is_empty() {
        let labels_str = dihedral_type_labels
            .iter()
            .map(|(id, label)| format!("{}:{}", id, label))
            .collect::<Vec<_>>()
            .join(",");
        frame
            .meta
            .insert("dihedral_type_labels".to_string(), labels_str);
    }

    Ok(frame)
}

// ============================================================================
// Reader
// ============================================================================

/// LAMMPS data file reader implementing the unified `Reader` interface.
///
/// LAMMPS data files are single-frame files; use `read_frame()` to load the frame.
/// and `read(n)` for n > 0 returns `Ok(None)`.
pub struct LAMMPSDataReader<R: BufRead + Seek> {
    reader: R,
    frame: OnceCell<Option<Frame>>,
    returned: bool,
}

impl<R: BufRead + Seek> LAMMPSDataReader<R> {
    /// Create a new LAMMPS data file reader
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            frame: OnceCell::new(),
            returned: false,
        }
    }

    /// Parse the entire file and return the frame
    fn parse_file(&mut self) -> std::io::Result<Option<Frame>> {
        self.reader.seek(SeekFrom::Start(0))?;

        // Parse header returns when it hits a section header line
        // We need to handle that first line
        let (header, first_section_line) = parse_header_with_first_section(&mut self.reader)?;

        let mut atom_type_labels = HashMap::new();
        let mut bond_type_labels = HashMap::new();
        let mut angle_type_labels = HashMap::new();
        let mut dihedral_type_labels = HashMap::new();
        let mut atoms = Vec::new();
        let mut bonds = Vec::new();
        let mut angles = Vec::new();
        let mut dihedrals = Vec::new();

        // Process the first section line that parse_header found
        let mut next_section = first_section_line;

        while let Some(line) = next_section.take() {
            let trimmed = line.trim();
            if trimmed.starts_with("Atom Type Labels") {
                let (labels, next) = parse_type_labels(&mut self.reader)?;
                atom_type_labels = labels;
                next_section = next;
            } else if trimmed.starts_with("Bond Type Labels") {
                let (labels, next) = parse_type_labels(&mut self.reader)?;
                bond_type_labels = labels;
                next_section = next;
            } else if trimmed.starts_with("Angle Type Labels") {
                let (labels, next) = parse_type_labels(&mut self.reader)?;
                angle_type_labels = labels;
                next_section = next;
            } else if trimmed.starts_with("Dihedral Type Labels") {
                let (labels, next) = parse_type_labels(&mut self.reader)?;
                dihedral_type_labels = labels;
                next_section = next;
            } else if trimmed.starts_with("Atoms") {
                atoms = parse_atoms(&mut self.reader, header.num_atoms, &atom_type_labels)?;
                break; // Atoms section doesn't return next section
            } else if trimmed.starts_with("Bonds") {
                bonds = parse_bonds(&mut self.reader, header.num_bonds, &bond_type_labels)?;
                break; // Bonds section doesn't return next section
            } else if trimmed.starts_with("Angles") {
                angles = parse_angles(&mut self.reader, header.num_angles, &angle_type_labels)?;
                break; // Angles section doesn't return next section
            } else if trimmed.starts_with("Dihedrals") {
                dihedrals = parse_dihedrals(
                    &mut self.reader,
                    header.num_dihedrals,
                    &dihedral_type_labels,
                )?;
                break; // Dihedrals section doesn't return next section
            } else {
                break; // Unknown section
            }
        }

        // Continue reading remaining sections
        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = self.reader.read_line(&mut line)?;
            if bytes_read == 0 {
                break;
            }

            let trimmed = line.trim();

            // Skip blank lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Check for section headers
            if trimmed.starts_with("Atom Type Labels") {
                let (labels, _) = parse_type_labels(&mut self.reader)?;
                atom_type_labels = labels;
            } else if trimmed.starts_with("Bond Type Labels") {
                let (labels, _) = parse_type_labels(&mut self.reader)?;
                bond_type_labels = labels;
            } else if trimmed.starts_with("Angle Type Labels") {
                let (labels, _) = parse_type_labels(&mut self.reader)?;
                angle_type_labels = labels;
            } else if trimmed.starts_with("Dihedral Type Labels") {
                let (labels, _) = parse_type_labels(&mut self.reader)?;
                dihedral_type_labels = labels;
            } else if trimmed.starts_with("Atoms") {
                atoms = parse_atoms(&mut self.reader, header.num_atoms, &atom_type_labels)?;
            } else if trimmed.starts_with("Bonds") {
                bonds = parse_bonds(&mut self.reader, header.num_bonds, &bond_type_labels)?;
            } else if trimmed.starts_with("Angles") {
                angles = parse_angles(&mut self.reader, header.num_angles, &angle_type_labels)?;
            } else if trimmed.starts_with("Dihedrals") {
                dihedrals = parse_dihedrals(
                    &mut self.reader,
                    header.num_dihedrals,
                    &dihedral_type_labels,
                )?;
            }
            // Ignore other sections (Masses, Impropers, etc.)
        }

        if atoms.is_empty() && header.num_atoms > 0 {
            return Err(err_mapper("No atoms found in file"));
        }

        let frame = build_frame(
            &header,
            atoms,
            bonds,
            angles,
            dihedrals,
            atom_type_labels,
            bond_type_labels,
            angle_type_labels,
            dihedral_type_labels,
        )?;
        Ok(Some(frame))
    }
}

impl<R: BufRead + Seek> Reader for LAMMPSDataReader<R> {
    type R = R;
    type Frame = Frame;

    fn new(reader: R) -> Self {
        Self::new(reader)
    }
}

impl<R: BufRead + Seek> FrameReader for LAMMPSDataReader<R> {
    fn read_frame(&mut self) -> std::io::Result<Option<Self::Frame>> {
        if self.returned {
            return Ok(None);
        }
        // Parse on first access and cache
        if self.frame.get().is_none() {
            let frame = self.parse_file()?;
            let _ = self.frame.set(frame);
        }

        self.returned = true;
        Ok(self.frame.get().unwrap().clone())
    }
}

// ============================================================================
// Writer
// ============================================================================

/// LAMMPS data file writer implementing the `FrameWriter` interface.
pub struct LAMMPSDataWriter<W: Write> {
    writer: W,
}

impl<W: Write> crate::io::writer::Writer for LAMMPSDataWriter<W> {
    type W = W;
    type FrameLike = Frame;

    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> LAMMPSDataWriter<W> {
    /// Create a new LAMMPS data file writer
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for LAMMPSDataWriter<W> {
    fn write_frame(&mut self, frame: &Frame) -> std::io::Result<()> {
        // Write comment line
        writeln!(self.writer, "# LAMMPS data file generated by molrs")?;
        writeln!(self.writer)?;

        // Get atoms block
        let atoms = frame
            .get("atoms")
            .ok_or_else(|| err_mapper("Frame must contain 'atoms' block"))?;

        let num_atoms = atoms.nrows().unwrap_or(0);

        // Get bonds block if present
        let bonds = frame.get("bonds");
        let num_bonds = bonds.and_then(|b| b.nrows()).unwrap_or(0);

        // Determine number of types
        let atom_types = atoms
            .get_int("type")
            .ok_or_else(|| err_mapper("Atoms block must contain 'type' column"))?;
        let num_atom_types = atom_types.iter().max().copied().unwrap_or(1) as usize;

        let num_bond_types = if let Some(bonds) = bonds {
            if let Some(bond_types) = bonds.get_int("type") {
                bond_types.iter().max().copied().unwrap_or(1) as usize
            } else {
                0
            }
        } else {
            0
        };

        // Write header
        writeln!(self.writer, "{} atoms", num_atoms)?;
        if num_bonds > 0 {
            writeln!(self.writer, "{} bonds", num_bonds)?;
        }
        writeln!(self.writer, "{} atom types", num_atom_types)?;
        if num_bond_types > 0 {
            writeln!(self.writer, "{} bond types", num_bond_types)?;
        }
        writeln!(self.writer)?;

        // Write box dimensions
        let box_str = frame
            .meta
            .get("box")
            .ok_or_else(|| err_mapper("Frame metadata must contain 'box'"))?;
        let default_origin = "0.0 0.0 0.0".to_string();
        let box_origin_str = frame.meta.get("box_origin").unwrap_or(&default_origin);

        let box_dims: Vec<f64> = box_str
            .split_whitespace()
            .map(|s| s.parse().map_err(err_mapper))
            .collect::<Result<_, _>>()?;
        let box_origin: Vec<f64> = box_origin_str
            .split_whitespace()
            .map(|s| s.parse().map_err(err_mapper))
            .collect::<Result<_, _>>()?;

        if box_dims.len() != 3 || box_origin.len() != 3 {
            return Err(err_mapper("Invalid box dimensions in metadata"));
        }

        writeln!(
            self.writer,
            "{} {} xlo xhi",
            box_origin[0],
            box_origin[0] + box_dims[0]
        )?;
        writeln!(
            self.writer,
            "{} {} ylo yhi",
            box_origin[1],
            box_origin[1] + box_dims[1]
        )?;
        writeln!(
            self.writer,
            "{} {} zlo zhi",
            box_origin[2],
            box_origin[2] + box_dims[2]
        )?;

        // Write tilt factors if present
        if let Some(tilt_str) = frame.meta.get("box_tilt") {
            writeln!(self.writer, "{} xy xz yz", tilt_str)?;
        }
        writeln!(self.writer)?;

        // Write type labels if present
        if let Some(atom_labels_str) = frame.meta.get("atom_type_labels") {
            writeln!(self.writer, "Atom Type Labels")?;
            writeln!(self.writer)?;
            for pair in atom_labels_str.split(',') {
                let parts: Vec<&str> = pair.split(':').collect();
                if parts.len() == 2 {
                    writeln!(self.writer, "{} {}", parts[0], parts[1])?;
                }
            }
            writeln!(self.writer)?;
        }

        if let Some(bond_labels_str) = frame.meta.get("bond_type_labels") {
            writeln!(self.writer, "Bond Type Labels")?;
            writeln!(self.writer)?;
            for pair in bond_labels_str.split(',') {
                let parts: Vec<&str> = pair.split(':').collect();
                if parts.len() == 2 {
                    writeln!(self.writer, "{} {}", parts[0], parts[1])?;
                }
            }
            writeln!(self.writer)?;
        }

        // Write Atoms section
        let x = atoms
            .get_float("x")
            .ok_or_else(|| err_mapper("Missing 'x' column"))?;
        let y = atoms
            .get_float("y")
            .ok_or_else(|| err_mapper("Missing 'y' column"))?;
        let z = atoms
            .get_float("z")
            .ok_or_else(|| err_mapper("Missing 'z' column"))?;
        let ids = atoms
            .get_int("id")
            .ok_or_else(|| err_mapper("Missing 'id' column"))?;

        let has_charge = atoms.get_float("charge").is_some();
        let has_mol_id = atoms.get_int("molecule_id").is_some();

        // Determine atom style
        let atom_style = if has_mol_id && has_charge {
            "full"
        } else if has_mol_id {
            "molecular"
        } else if has_charge {
            "charge"
        } else {
            "atomic"
        };

        writeln!(self.writer, "Atoms # {}", atom_style)?;
        writeln!(self.writer)?;

        for i in 0..num_atoms {
            write!(self.writer, "{}", ids[i])?;

            if has_mol_id {
                let mol_ids = atoms.get_int("molecule_id").unwrap();
                write!(self.writer, " {}", mol_ids[i])?;
            }

            write!(self.writer, " {}", atom_types[i])?;

            if has_charge {
                let charges = atoms.get_float("charge").unwrap();
                write!(self.writer, " {}", charges[i])?;
            }

            writeln!(self.writer, " {} {} {}", x[i], y[i], z[i])?;
        }
        writeln!(self.writer)?;

        // Write Bonds section if present
        if let Some(bonds) = bonds
            && num_bonds > 0
        {
            writeln!(self.writer, "Bonds")?;
            writeln!(self.writer)?;

            let atom_i = bonds
                .get_uint("atomi")
                .ok_or_else(|| err_mapper("Missing 'atom_i' column"))?;
            let atom_j = bonds
                .get_uint("atomj")
                .ok_or_else(|| err_mapper("Missing 'atom_j' column"))?;
            let bond_types = bonds
                .get_int("type")
                .ok_or_else(|| err_mapper("Missing 'type' column"))?;

            // Convert indices back to IDs
            for i in 0..num_bonds {
                let id_i = ids[atom_i[i] as usize];
                let id_j = ids[atom_j[i] as usize];
                writeln!(self.writer, "{} {} {} {}", i + 1, bond_types[i], id_i, id_j)?;
            }
            writeln!(self.writer)?;
        }

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Read a LAMMPS data file from a path
pub fn read_lammps_data<P: AsRef<Path>>(path: P) -> std::io::Result<Frame> {
    let file = File::open(path)?;
    let mut reader = LAMMPSDataReader::new(BufReader::new(file));
    reader
        .read_frame()?
        .ok_or_else(|| err_mapper("No frame found in LAMMPS data file"))
}

/// Write a Frame to a LAMMPS data file
pub fn write_lammps_data<P: AsRef<Path>>(path: P, frame: &Frame) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = LAMMPSDataWriter::new(file);
    writer.write_frame(frame)
}
