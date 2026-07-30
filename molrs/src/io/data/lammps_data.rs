//! LAMMPS data file format reader and writer.
//!
//! Specs: <https://docs.lammps.org/read_data.html>,
//! <https://docs.lammps.org/atom_style.html>
//!
//! Atom-style layouts and shared helpers live in [`crate::io::lammps`].
//! Atoms are streamed straight into typed column buffers (no intermediate
//! per-atom struct), which cuts peak memory on large systems.

use crate::io::lammps::atom_style::{
    AtomStyleLayout, DataField, field_column_key, infer_write_style, is_int_token,
    is_noninteger_float_token, layout_for_atom_style, layout_from_column_count,
    parse_atoms_style_hint,
};
use crate::io::lammps::box_bounds::{BoxBounds, simbox_from_bounds};
use crate::io::lammps::common::{
    OptCol, TypeRef, err_mapper, insert_f, insert_i, insert_u, invert_type_labels, labels_to_meta,
    parse_f, parse_i, tokenize,
};
use crate::io::reader::{FrameReader, Reader};
use crate::io::streaming::{FrameIndexBuilder, FrameIndexEntry};
use crate::io::writer::FrameWriter;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::store::frame_access::FrameAccess;
use molrs::store::keys;
use molrs::types::{F, I, Pbc3, U};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Seek, SeekFrom, Write};
use std::path::Path;

// ============================================================================
// Header
// ============================================================================

#[derive(Debug, Clone, Default)]
struct LAMMPSHeader {
    num_atoms: usize,
    num_bonds: usize,
    num_angles: usize,
    num_dihedrals: usize,
    num_impropers: usize,
    bounds: BoxBounds,
}

// ============================================================================
// Streaming atom columns
// ============================================================================

/// Column-oriented atom buffer filled directly from the Atoms section.
struct AtomColumns {
    id: Vec<I>,
    type_refs: Vec<TypeRef>,
    x: Vec<F>,
    y: Vec<F>,
    z: Vec<F>,
    mol: OptCol<I>,
    charge: OptCol<F>,
    bodyflag: OptCol<I>,
    mass: OptCol<F>,
    diameter: OptCol<F>,
    density: OptCol<F>,
    volume: OptCol<F>,
    shape_flag: OptCol<I>,
    mux: OptCol<F>,
    muy: OptCol<F>,
    muz: OptCol<F>,
    spx: OptCol<F>,
    spy: OptCol<F>,
    spz: OptCol<F>,
    sp: OptCol<F>,
    rho: OptCol<F>,
    esph: OptCol<F>,
    cv: OptCol<F>,
    theta: OptCol<F>,
    espin: OptCol<I>,
    eradius: OptCol<F>,
    status: OptCol<I>,
    energy: OptCol<F>,
    template_index: OptCol<I>,
    template_atom: OptCol<I>,
    edpd_temp: OptCol<F>,
    edpd_cv: OptCol<F>,
    smd_volume: OptCol<F>,
    smd_mass: OptCol<F>,
    smd_kradius: OptCol<F>,
    smd_cradius: OptCol<F>,
    smd_x0: OptCol<F>,
    smd_y0: OptCol<F>,
    smd_z0: OptCol<F>,
    area: OptCol<F>,
    ed: OptCol<F>,
    em: OptCol<F>,
    epsilon: OptCol<F>,
    curvature: OptCol<F>,
    ix: OptCol<I>,
    iy: OptCol<I>,
    iz: OptCol<I>,
    vx: OptCol<F>,
    vy: OptCol<F>,
    vz: OptCol<F>,
}

impl AtomColumns {
    fn with_capacity(n: usize) -> Self {
        Self {
            id: Vec::with_capacity(n),
            type_refs: Vec::with_capacity(n),
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
            z: Vec::with_capacity(n),
            mol: OptCol::with_capacity(n),
            charge: OptCol::with_capacity(n),
            bodyflag: OptCol::with_capacity(n),
            mass: OptCol::with_capacity(n),
            diameter: OptCol::with_capacity(n),
            density: OptCol::with_capacity(n),
            volume: OptCol::with_capacity(n),
            shape_flag: OptCol::with_capacity(n),
            mux: OptCol::with_capacity(n),
            muy: OptCol::with_capacity(n),
            muz: OptCol::with_capacity(n),
            spx: OptCol::with_capacity(n),
            spy: OptCol::with_capacity(n),
            spz: OptCol::with_capacity(n),
            sp: OptCol::with_capacity(n),
            rho: OptCol::with_capacity(n),
            esph: OptCol::with_capacity(n),
            cv: OptCol::with_capacity(n),
            theta: OptCol::with_capacity(n),
            espin: OptCol::with_capacity(n),
            eradius: OptCol::with_capacity(n),
            status: OptCol::with_capacity(n),
            energy: OptCol::with_capacity(n),
            template_index: OptCol::with_capacity(n),
            template_atom: OptCol::with_capacity(n),
            edpd_temp: OptCol::with_capacity(n),
            edpd_cv: OptCol::with_capacity(n),
            smd_volume: OptCol::with_capacity(n),
            smd_mass: OptCol::with_capacity(n),
            smd_kradius: OptCol::with_capacity(n),
            smd_cradius: OptCol::with_capacity(n),
            smd_x0: OptCol::with_capacity(n),
            smd_y0: OptCol::with_capacity(n),
            smd_z0: OptCol::with_capacity(n),
            area: OptCol::with_capacity(n),
            ed: OptCol::with_capacity(n),
            em: OptCol::with_capacity(n),
            epsilon: OptCol::with_capacity(n),
            curvature: OptCol::with_capacity(n),
            ix: OptCol::with_capacity(n),
            iy: OptCol::with_capacity(n),
            iz: OptCol::with_capacity(n),
            vx: OptCol::with_capacity(n),
            vy: OptCol::with_capacity(n),
            vz: OptCol::with_capacity(n),
        }
    }

    fn len(&self) -> usize {
        self.id.len()
    }

    fn push_defaults_for_absent_optionals(&mut self) {
        // After streaming a row that only touches present fields, pad optionals
        // that were not written this row. Called once per row after field walk.
        let n = self.id.len();
        let pad_i = |c: &mut OptCol<I>| c.data.resize(n, 0);
        let pad_f = |c: &mut OptCol<F>| c.data.resize(n, 0.0);
        pad_i(&mut self.mol);
        pad_f(&mut self.charge);
        pad_i(&mut self.bodyflag);
        pad_f(&mut self.mass);
        pad_f(&mut self.diameter);
        pad_f(&mut self.density);
        pad_f(&mut self.volume);
        pad_i(&mut self.shape_flag);
        pad_f(&mut self.mux);
        pad_f(&mut self.muy);
        pad_f(&mut self.muz);
        pad_f(&mut self.spx);
        pad_f(&mut self.spy);
        pad_f(&mut self.spz);
        pad_f(&mut self.sp);
        pad_f(&mut self.rho);
        pad_f(&mut self.esph);
        pad_f(&mut self.cv);
        pad_f(&mut self.theta);
        pad_i(&mut self.espin);
        pad_f(&mut self.eradius);
        pad_i(&mut self.status);
        pad_f(&mut self.energy);
        pad_i(&mut self.template_index);
        pad_i(&mut self.template_atom);
        pad_f(&mut self.edpd_temp);
        pad_f(&mut self.edpd_cv);
        pad_f(&mut self.smd_volume);
        pad_f(&mut self.smd_mass);
        pad_f(&mut self.smd_kradius);
        pad_f(&mut self.smd_cradius);
        pad_f(&mut self.smd_x0);
        pad_f(&mut self.smd_y0);
        pad_f(&mut self.smd_z0);
        pad_f(&mut self.area);
        pad_f(&mut self.ed);
        pad_f(&mut self.em);
        pad_f(&mut self.epsilon);
        pad_f(&mut self.curvature);
        pad_i(&mut self.ix);
        pad_i(&mut self.iy);
        pad_i(&mut self.iz);
        pad_f(&mut self.vx);
        pad_f(&mut self.vy);
        pad_f(&mut self.vz);
    }

    fn into_block(self, atom_type_labels: &HashMap<String, String>) -> std::io::Result<Block> {
        let n = self.id.len();
        let label_to_id = invert_type_labels(atom_type_labels);
        let types: Vec<I> = self
            .type_refs
            .iter()
            .map(|t| t.resolve(&label_to_id))
            .collect();

        let mut block = Block::new();
        insert_i(&mut block, keys::ID, self.id, n)?;
        insert_i(&mut block, keys::TYPE, types, n)?;
        insert_f(&mut block, keys::X, self.x, n)?;
        insert_f(&mut block, keys::Y, self.y, n)?;
        insert_f(&mut block, keys::Z, self.z, n)?;

        macro_rules! opt_i {
            ($col:expr, $key:expr) => {
                if $col.present {
                    insert_i(&mut block, $key, $col.data, n)?;
                }
            };
        }
        macro_rules! opt_f {
            ($col:expr, $key:expr) => {
                if $col.present {
                    insert_f(&mut block, $key, $col.data, n)?;
                }
            };
        }

        opt_i!(self.mol, keys::MOL_ID);
        opt_f!(self.charge, keys::CHARGE);
        opt_i!(self.bodyflag, "bodyflag");
        opt_f!(self.mass, keys::MASS);
        opt_f!(self.diameter, "diameter");
        opt_f!(self.density, "density");
        opt_f!(self.volume, "volume");
        opt_i!(self.shape_flag, "shape_flag");
        opt_f!(self.mux, keys::MUX);
        opt_f!(self.muy, keys::MUY);
        opt_f!(self.muz, keys::MUZ);
        opt_f!(self.spx, "spx");
        opt_f!(self.spy, "spy");
        opt_f!(self.spz, "spz");
        opt_f!(self.sp, "sp");
        opt_f!(self.rho, "rho");
        opt_f!(self.esph, "esph");
        opt_f!(self.cv, "cv");
        opt_f!(self.theta, "theta");
        opt_i!(self.espin, "espin");
        opt_f!(self.eradius, "eradius");
        opt_i!(self.status, "status");
        opt_f!(self.energy, "energy");
        opt_i!(self.template_index, "template_index");
        opt_i!(self.template_atom, "template_atom");
        opt_f!(self.edpd_temp, "edpd_temp");
        opt_f!(self.edpd_cv, "edpd_cv");
        opt_f!(self.smd_volume, "smd_volume");
        opt_f!(self.smd_mass, "smd_mass");
        opt_f!(self.smd_kradius, "smd_kradius");
        opt_f!(self.smd_cradius, "smd_cradius");
        opt_f!(self.smd_x0, "x0");
        opt_f!(self.smd_y0, "y0");
        opt_f!(self.smd_z0, "z0");
        opt_f!(self.area, "area");
        opt_f!(self.ed, "ed");
        opt_f!(self.em, "em");
        opt_f!(self.epsilon, "epsilon");
        opt_f!(self.curvature, "curvature");
        opt_i!(self.ix, "ix");
        opt_i!(self.iy, "iy");
        opt_i!(self.iz, "iz");
        opt_f!(self.vx, keys::VX);
        opt_f!(self.vy, keys::VY);
        opt_f!(self.vz, keys::VZ);

        Ok(block)
    }
}

// ============================================================================
// Topology
// ============================================================================

struct TopologyTerm {
    type_ref: TypeRef,
    members: [I; 4],
    n_members: u8,
}

impl TopologyTerm {
    fn members(&self) -> &[I] {
        &self.members[..self.n_members as usize]
    }
}

// ============================================================================
// Atoms line → columns
// ============================================================================

fn resolve_layout(
    tokens: &[&str],
    known: Option<AtomStyleLayout>,
    style_known: bool,
) -> std::io::Result<AtomStyleLayout> {
    let mut layout = match known {
        Some(l) => l,
        None => layout_from_column_count(tokens.len())?,
    };

    // charge vs molecular when style unknown and col count is 6 or 9.
    if !style_known && (tokens.len() == 6 || tokens.len() == 9) && tokens.len() >= 3 {
        let token2 = tokens[2];
        let looks_like_charge = is_noninteger_float_token(token2) || !is_int_token(token2);
        let looks_like_molecular = is_int_token(tokens[1]) && is_int_token(token2);
        if looks_like_molecular && !looks_like_charge {
            layout = layout_for_atom_style("molecular").unwrap();
        } else if looks_like_charge {
            layout = layout_for_atom_style("charge").unwrap();
        }
    }
    Ok(layout)
}

fn push_atom_line(
    cols: &mut AtomColumns,
    tokens: &[&str],
    layout: AtomStyleLayout,
) -> std::io::Result<()> {
    let n = tokens.len();
    let min = layout.min_cols();
    if n < min {
        return Err(err_mapper(format!(
            "Invalid Atoms line: expected at least {min} columns, got {n}"
        )));
    }

    let image = if n == min + 3 {
        Some([
            parse_i(tokens[n - 3])?,
            parse_i(tokens[n - 2])?,
            parse_i(tokens[n - 1])?,
        ])
    } else if n == min {
        None
    } else if layout.flexible_tail && n > min {
        if n >= min + 3
            && is_int_token(tokens[n - 3])
            && is_int_token(tokens[n - 2])
            && is_int_token(tokens[n - 1])
        {
            Some([
                parse_i(tokens[n - 3])?,
                parse_i(tokens[n - 2])?,
                parse_i(tokens[n - 1])?,
            ])
        } else {
            None
        }
    } else {
        return Err(err_mapper(format!(
            "Invalid Atoms line: got {n} columns, layout expects {min} or {} \
             (with image flags)",
            min + 3
        )));
    };

    // Required xyz placeholders — filled by field walk.
    let mut got_id = false;
    let mut got_type = false;
    let mut got_x = false;

    for (i, field) in layout.fields.iter().enumerate() {
        let tok = tokens[i];
        match field {
            DataField::Id => {
                cols.id.push(parse_i(tok)?);
                got_id = true;
            }
            DataField::Type => {
                cols.type_refs.push(TypeRef::parse(tok));
                got_type = true;
            }
            DataField::Mol => cols.mol.push(parse_i(tok)?),
            DataField::Charge => cols.charge.push(parse_f(tok)?),
            DataField::X => {
                cols.x.push(parse_f(tok)?);
                got_x = true;
            }
            DataField::Y => cols.y.push(parse_f(tok)?),
            DataField::Z => cols.z.push(parse_f(tok)?),
            DataField::Bodyflag => cols.bodyflag.push(parse_i(tok)?),
            DataField::Mass => cols.mass.push(parse_f(tok)?),
            DataField::Diameter => cols.diameter.push(parse_f(tok)?),
            DataField::Density => cols.density.push(parse_f(tok)?),
            DataField::Volume => cols.volume.push(parse_f(tok)?),
            DataField::ShapeFlag => cols.shape_flag.push(parse_i(tok)?),
            DataField::Mux => cols.mux.push(parse_f(tok)?),
            DataField::Muy => cols.muy.push(parse_f(tok)?),
            DataField::Muz => cols.muz.push(parse_f(tok)?),
            DataField::Spx => cols.spx.push(parse_f(tok)?),
            DataField::Spy => cols.spy.push(parse_f(tok)?),
            DataField::Spz => cols.spz.push(parse_f(tok)?),
            DataField::Sp => cols.sp.push(parse_f(tok)?),
            DataField::Rho => cols.rho.push(parse_f(tok)?),
            DataField::Esph => cols.esph.push(parse_f(tok)?),
            DataField::Cv => cols.cv.push(parse_f(tok)?),
            DataField::Theta => cols.theta.push(parse_f(tok)?),
            DataField::Espin => cols.espin.push(parse_i(tok)?),
            DataField::Eradius => cols.eradius.push(parse_f(tok)?),
            DataField::Status => cols.status.push(parse_i(tok)?),
            DataField::Energy => cols.energy.push(parse_f(tok)?),
            DataField::TemplateIndex => cols.template_index.push(parse_i(tok)?),
            DataField::TemplateAtom => cols.template_atom.push(parse_i(tok)?),
            DataField::EdpdTemp => cols.edpd_temp.push(parse_f(tok)?),
            DataField::EdpdCv => cols.edpd_cv.push(parse_f(tok)?),
            DataField::SmdVolume => cols.smd_volume.push(parse_f(tok)?),
            DataField::SmdMass => cols.smd_mass.push(parse_f(tok)?),
            DataField::SmdKradius => cols.smd_kradius.push(parse_f(tok)?),
            DataField::SmdCradius => cols.smd_cradius.push(parse_f(tok)?),
            DataField::SmdX0 => cols.smd_x0.push(parse_f(tok)?),
            DataField::SmdY0 => cols.smd_y0.push(parse_f(tok)?),
            DataField::SmdZ0 => cols.smd_z0.push(parse_f(tok)?),
            DataField::Area => cols.area.push(parse_f(tok)?),
            DataField::Ed => cols.ed.push(parse_f(tok)?),
            DataField::Em => cols.em.push(parse_f(tok)?),
            DataField::Epsilon => cols.epsilon.push(parse_f(tok)?),
            DataField::Curvature => cols.curvature.push(parse_f(tok)?),
        }
    }

    if !got_id || !got_type || !got_x {
        return Err(err_mapper(
            "Invalid atom style layout: missing id, type, or x field",
        ));
    }
    // y/z must match x length
    if cols.y.len() != cols.x.len() || cols.z.len() != cols.x.len() {
        return Err(err_mapper("Invalid Atoms line: incomplete xyz"));
    }

    match image {
        Some([a, b, c]) => {
            cols.ix.push(a);
            cols.iy.push(b);
            cols.iz.push(c);
        }
        None => {
            // leave unset; pad_defaults fills zeros without marking present
        }
    }

    cols.push_defaults_for_absent_optionals();
    Ok(())
}

// ============================================================================
// Section parsers
// ============================================================================

fn parse_header_with_first_section<R: BufRead>(
    reader: &mut R,
) -> std::io::Result<(LAMMPSHeader, Option<String>)> {
    let mut header = LAMMPSHeader::default();
    let mut line = String::new();

    reader.read_line(&mut line)?; // comment
    line.clear();

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok((header, None));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens = tokenize(trimmed);
        if tokens.is_empty() {
            continue;
        }
        if tokens[0].chars().next().is_some_and(|c| c.is_uppercase()) {
            return Ok((header, Some(line.clone())));
        }
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
            Some(&"impropers") if tokens.len() >= 2 => {
                header.num_impropers = tokens[0].parse().map_err(err_mapper)?;
            }
            Some(&"xhi") if tokens.len() >= 4 && tokens[2] == "xlo" => {
                header.bounds.xlo = tokens[0].parse().map_err(err_mapper)?;
                header.bounds.xhi = tokens[1].parse().map_err(err_mapper)?;
            }
            Some(&"yhi") if tokens.len() >= 4 && tokens[2] == "ylo" => {
                header.bounds.ylo = tokens[0].parse().map_err(err_mapper)?;
                header.bounds.yhi = tokens[1].parse().map_err(err_mapper)?;
            }
            Some(&"zhi") if tokens.len() >= 4 && tokens[2] == "zlo" => {
                header.bounds.zlo = tokens[0].parse().map_err(err_mapper)?;
                header.bounds.zhi = tokens[1].parse().map_err(err_mapper)?;
            }
            Some(&"yz") if tokens.len() >= 6 && tokens[3] == "xy" && tokens[4] == "xz" => {
                header.bounds.xy = Some(tokens[0].parse().map_err(err_mapper)?);
                header.bounds.xz = Some(tokens[1].parse().map_err(err_mapper)?);
                header.bounds.yz = Some(tokens[2].parse().map_err(err_mapper)?);
            }
            _ => {}
        }
    }
}

fn parse_type_labels<R: BufRead>(
    reader: &mut R,
) -> std::io::Result<(HashMap<String, String>, Option<String>)> {
    let mut labels = HashMap::new();
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok((labels, None));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let tokens = tokenize(trimmed);
        if trimmed.chars().next().is_some_and(|c| c.is_uppercase()) {
            if tokens.len() >= 2 {
                if tokens[0].parse::<i64>().is_err() {
                    return Ok((labels, Some(line.clone())));
                }
            } else if tokens.len() == 1 {
                return Ok((labels, Some(line.clone())));
            }
        }
        if tokens.len() >= 2 {
            labels.insert(tokens[0].to_string(), tokens[1].to_string());
        }
    }
}

/// Masses section: `type mass` per line → map type-id → mass.
fn parse_masses<R: BufRead>(reader: &mut R) -> std::io::Result<HashMap<I, F>> {
    let mut masses = HashMap::new();
    let mut line = String::new();
    // Read until blank line or next section (uppercase non-numeric).
    // We don't know the count; stop at section header.
    // But we're called mid-stream after "Masses" — consume lines until blank
    // then stop, OR until next section. Actually blank lines appear between
    // Masses header and first row, and after last row before next section.
    // Strategy: read while lines look like "int float"; stop at uppercase
    // section or empty after having seen data. For simplicity read until
    // we hit a line that doesn't match type-mass, then if it's a section
    // the caller re-dispatches — but we already consumed it.
    //
    // Better: peek-style by returning leftover section line like type_labels.
    // For Masses LAMMPS always has blank line after last mass before next
    // section. Read until blank-after-data or EOF; section after blank is
    // handled by outer loop.
    let mut saw_data = false;
    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if saw_data {
                break;
            }
            continue;
        }
        if trimmed.starts_with('#') {
            continue;
        }
        if trimmed.chars().next().is_some_and(|c| c.is_uppercase())
            && tokenize(trimmed)
                .first()
                .is_some_and(|t| t.parse::<i64>().is_err())
        {
            // Hit next section without blank — shouldn't happen for Masses,
            // but don't consume: we can't put it back. Outer loop already
            // lost this line. Rare. Ignore and stop.
            break;
        }
        let tokens = tokenize(trimmed);
        if tokens.len() >= 2
            && let (Ok(tid), Ok(m)) = (tokens[0].parse::<I>(), tokens[1].parse::<F>())
        {
            masses.insert(tid, m);
            saw_data = true;
            continue;
        }
        if saw_data {
            break;
        }
    }
    Ok(masses)
}

fn parse_atoms_streamed<R: BufRead>(
    reader: &mut R,
    num_atoms: usize,
    style_hint: Option<&str>,
) -> std::io::Result<AtomColumns> {
    let mut cols = AtomColumns::with_capacity(num_atoms);
    let mut line = String::new();
    let known = style_hint.and_then(layout_for_atom_style);
    let style_known = known.is_some();

    while cols.len() < num_atoms {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens = tokenize(trimmed);
        let layout = resolve_layout(&tokens, known, style_known)?;
        push_atom_line(&mut cols, &tokens, layout)?;
    }
    Ok(cols)
}

fn parse_topology_section<R: BufRead>(
    reader: &mut R,
    count: usize,
    n_members: usize,
    section: &str,
) -> std::io::Result<Vec<TopologyTerm>> {
    let min_cols = 2 + n_members;
    let mut terms = Vec::with_capacity(count);
    let mut line = String::new();
    while terms.len() < count {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens = tokenize(trimmed);
        if tokens.len() < min_cols {
            return Err(err_mapper(format!(
                "Invalid {section} line: expected {min_cols} columns, got {}",
                tokens.len()
            )));
        }
        let mut members = [0 as I; 4];
        for (i, slot) in members.iter_mut().enumerate().take(n_members) {
            *slot = parse_i(tokens[2 + i])?;
        }
        terms.push(TopologyTerm {
            type_ref: TypeRef::parse(tokens[1]),
            members,
            n_members: n_members as u8,
        });
    }
    Ok(terms)
}

/// Velocities: `id vx vy vz` — merge into atom columns by id.
fn parse_velocities_into<R: BufRead>(
    reader: &mut R,
    cols: &mut AtomColumns,
) -> std::io::Result<()> {
    let n = cols.len();
    if n == 0 {
        return Ok(());
    }
    // Ensure velocity columns exist and are zeroed.
    cols.vx.data.resize(n, 0.0);
    cols.vy.data.resize(n, 0.0);
    cols.vz.data.resize(n, 0.0);
    cols.vx.present = true;
    cols.vy.present = true;
    cols.vz.present = true;

    let id_to_idx: HashMap<I, usize> = cols.id.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let mut line = String::new();
    let mut filled = 0usize;
    while filled < n {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if filled > 0 {
                break;
            }
            continue;
        }
        if trimmed.starts_with('#') {
            continue;
        }
        if trimmed.chars().next().is_some_and(|c| c.is_uppercase())
            && tokenize(trimmed)
                .first()
                .is_some_and(|t| t.parse::<i64>().is_err())
        {
            break;
        }
        let tokens = tokenize(trimmed);
        if tokens.len() < 4 {
            continue;
        }
        let id = parse_i(tokens[0])?;
        if let Some(&idx) = id_to_idx.get(&id) {
            cols.vx.data[idx] = parse_f(tokens[1])?;
            cols.vy.data[idx] = parse_f(tokens[2])?;
            cols.vz.data[idx] = parse_f(tokens[3])?;
            filled += 1;
        }
    }
    Ok(())
}

// ============================================================================
// Frame assembly
// ============================================================================

fn insert_topology_block(
    frame: &mut Frame,
    block_name: &str,
    kind: &str,
    terms: &[TopologyTerm],
    atom_keys: &[&str],
    atom_id_map: &HashMap<I, U>,
    label_to_id: &HashMap<String, I>,
) -> std::io::Result<()> {
    if terms.is_empty() {
        return Ok(());
    }
    let n = terms.len();
    let n_members = atom_keys.len();
    let mut member_cols: Vec<Vec<U>> = (0..n_members).map(|_| Vec::with_capacity(n)).collect();
    let mut types = Vec::with_capacity(n);

    for term in terms {
        for (i, &atom_id) in term.members().iter().enumerate() {
            let idx = atom_id_map.get(&atom_id).copied().ok_or_else(|| {
                err_mapper(format!("{kind} references unknown atom ID: {atom_id}"))
            })?;
            member_cols[i].push(idx);
        }
        types.push(term.type_ref.resolve(label_to_id));
    }

    let mut block = Block::new();
    for (key, col) in atom_keys.iter().zip(member_cols) {
        insert_u(&mut block, key, col, n)?;
    }
    insert_i(&mut block, keys::TYPE, types, n)?;
    frame.insert(block_name, block);
    Ok(())
}

struct ParsedData {
    header: LAMMPSHeader,
    atoms: AtomColumns,
    bonds: Vec<TopologyTerm>,
    angles: Vec<TopologyTerm>,
    dihedrals: Vec<TopologyTerm>,
    impropers: Vec<TopologyTerm>,
    type_masses: HashMap<I, F>,
    atom_type_labels: HashMap<String, String>,
    bond_type_labels: HashMap<String, String>,
    angle_type_labels: HashMap<String, String>,
    dihedral_type_labels: HashMap<String, String>,
    improper_type_labels: HashMap<String, String>,
}

fn build_frame(mut data: ParsedData) -> std::io::Result<Frame> {
    let mut frame = Frame::new();

    // Apply per-type Masses when no per-atom mass column was set.
    if !data.atoms.mass.present && !data.type_masses.is_empty() {
        let label_to_id = invert_type_labels(&data.atom_type_labels);
        let n = data.atoms.len();
        data.atoms.mass.data.resize(n, 0.0);
        for (i, tref) in data.atoms.type_refs.iter().enumerate() {
            let tid = tref.resolve(&label_to_id);
            data.atoms.mass.data[i] = data.type_masses.get(&tid).copied().unwrap_or(0.0);
        }
        data.atoms.mass.present = true;
    }

    // atom id → row index for topology remapping
    let atom_id_map: HashMap<I, U> = data
        .atoms
        .id
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as U))
        .collect();

    if data.atoms.len() > 0 {
        let atom_block = data.atoms.into_block(&data.atom_type_labels)?;
        frame.insert("atoms", atom_block);

        insert_topology_block(
            &mut frame,
            "bonds",
            "Bond",
            &data.bonds,
            &[keys::ATOMI, keys::ATOMJ],
            &atom_id_map,
            &invert_type_labels(&data.bond_type_labels),
        )?;
        insert_topology_block(
            &mut frame,
            "angles",
            "Angle",
            &data.angles,
            &[keys::ATOMI, keys::ATOMJ, keys::ATOMK],
            &atom_id_map,
            &invert_type_labels(&data.angle_type_labels),
        )?;
        insert_topology_block(
            &mut frame,
            "dihedrals",
            "Dihedral",
            &data.dihedrals,
            &[keys::ATOMI, keys::ATOMJ, keys::ATOMK, keys::ATOML],
            &atom_id_map,
            &invert_type_labels(&data.dihedral_type_labels),
        )?;
        insert_topology_block(
            &mut frame,
            "impropers",
            "Improper",
            &data.impropers,
            &[keys::ATOMI, keys::ATOMJ, keys::ATOMK, keys::ATOML],
            &atom_id_map,
            &invert_type_labels(&data.improper_type_labels),
        )?;
    }

    let pbc: Pbc3 = [true, true, true];
    if let Some(sb) = simbox_from_bounds(&data.header.bounds, pbc)? {
        frame.simbox = Some(sb);
    }

    for (key, labels) in [
        ("atom_type_labels", &data.atom_type_labels),
        ("bond_type_labels", &data.bond_type_labels),
        ("angle_type_labels", &data.angle_type_labels),
        ("dihedral_type_labels", &data.dihedral_type_labels),
        ("improper_type_labels", &data.improper_type_labels),
    ] {
        if let Some(s) = labels_to_meta(labels) {
            frame.meta.insert(key.to_string(), s);
        }
    }

    Ok(frame)
}

// ============================================================================
// Section dispatch
// ============================================================================

fn dispatch_section<R: BufRead>(
    header_line: &str,
    reader: &mut R,
    data: &mut ParsedData,
) -> std::io::Result<Option<String>> {
    let trimmed = header_line.trim();

    if trimmed.starts_with("Atom Type Labels") {
        let (labels, next) = parse_type_labels(reader)?;
        data.atom_type_labels = labels;
        return Ok(next);
    }
    if trimmed.starts_with("Bond Type Labels") {
        let (labels, next) = parse_type_labels(reader)?;
        data.bond_type_labels = labels;
        return Ok(next);
    }
    if trimmed.starts_with("Angle Type Labels") {
        let (labels, next) = parse_type_labels(reader)?;
        data.angle_type_labels = labels;
        return Ok(next);
    }
    if trimmed.starts_with("Dihedral Type Labels") {
        let (labels, next) = parse_type_labels(reader)?;
        data.dihedral_type_labels = labels;
        return Ok(next);
    }
    if trimmed.starts_with("Improper Type Labels") {
        let (labels, next) = parse_type_labels(reader)?;
        data.improper_type_labels = labels;
        return Ok(next);
    }
    if trimmed.starts_with("Masses") {
        data.type_masses = parse_masses(reader)?;
        return Ok(None);
    }
    if trimmed.starts_with("Atoms") {
        let hint = parse_atoms_style_hint(trimmed);
        data.atoms = parse_atoms_streamed(reader, data.header.num_atoms, hint.as_deref())?;
        return Ok(None);
    }
    if trimmed.starts_with("Velocities") {
        parse_velocities_into(reader, &mut data.atoms)?;
        return Ok(None);
    }
    if trimmed.starts_with("Bonds") {
        data.bonds = parse_topology_section(reader, data.header.num_bonds, 2, "Bonds")?;
        return Ok(None);
    }
    if trimmed.starts_with("Angles") {
        data.angles = parse_topology_section(reader, data.header.num_angles, 3, "Angles")?;
        return Ok(None);
    }
    if trimmed.starts_with("Dihedrals") {
        data.dihedrals = parse_topology_section(reader, data.header.num_dihedrals, 4, "Dihedrals")?;
        return Ok(None);
    }
    if trimmed.starts_with("Impropers") {
        data.impropers = parse_topology_section(reader, data.header.num_impropers, 4, "Impropers")?;
        return Ok(None);
    }
    Ok(None)
}

fn is_section_header(trimmed: &str) -> bool {
    !trimmed.is_empty()
        && !trimmed.starts_with('#')
        && trimmed.chars().next().is_some_and(|c| c.is_uppercase())
}

// ============================================================================
// Reader
// ============================================================================

pub struct LAMMPSDataReader<R: BufRead + Seek> {
    reader: R,
    frame: OnceCell<Option<Frame>>,
    returned: bool,
}

impl<R: BufRead + Seek> LAMMPSDataReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            frame: OnceCell::new(),
            returned: false,
        }
    }

    fn parse_file(&mut self) -> std::io::Result<Option<Frame>> {
        self.reader.seek(SeekFrom::Start(0))?;
        let (header, first) = parse_header_with_first_section(&mut self.reader)?;
        let mut data = ParsedData {
            header,
            atoms: AtomColumns::with_capacity(0),
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            impropers: Vec::new(),
            type_masses: HashMap::new(),
            atom_type_labels: HashMap::new(),
            bond_type_labels: HashMap::new(),
            angle_type_labels: HashMap::new(),
            dihedral_type_labels: HashMap::new(),
            improper_type_labels: HashMap::new(),
        };

        let mut pending = first;
        while let Some(line) = pending.take() {
            pending = dispatch_section(&line, &mut self.reader, &mut data)?;
        }

        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();
            if !is_section_header(trimmed) {
                continue;
            }
            let mut next = dispatch_section(trimmed, &mut self.reader, &mut data)?;
            while let Some(hdr) = next.take() {
                next = dispatch_section(&hdr, &mut self.reader, &mut data)?;
            }
        }

        if data.atoms.len() == 0 && data.header.num_atoms > 0 {
            return Err(err_mapper("No atoms found in file"));
        }
        Ok(Some(build_frame(data)?))
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
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for LAMMPSDataWriter<W> {
    fn write_frame(&mut self, frame: &Frame) -> std::io::Result<()> {
        write_lammps_data_frame(&mut self.writer, frame)
    }
}

fn write_meta_type_labels<W: Write>(
    writer: &mut W,
    section: &str,
    raw: Option<&str>,
) -> std::io::Result<()> {
    let Some(labels_str) = raw else {
        return Ok(());
    };
    writeln!(writer, "{section}")?;
    writeln!(writer)?;
    for pair in labels_str.split(',') {
        let mut parts = pair.splitn(2, ':');
        if let (Some(id), Some(label)) = (parts.next(), parts.next()) {
            writeln!(writer, "{id} {label}")?;
        }
    }
    writeln!(writer)?;
    Ok(())
}

/// Does the atoms block have an int or float column for this data-file field?
fn frame_has_atom_field(frame: &impl FrameAccess, field: DataField) -> bool {
    let key = field_column_key(field);
    // Core fields checked separately; mol accepts legacy name.
    if field == DataField::Mol {
        return frame.get_int("atoms", keys::MOL_ID).is_some()
            || frame.get_int("atoms", "molecule_id").is_some();
    }
    frame.get_int("atoms", key).is_some() || frame.get_float("atoms", key).is_some()
}

fn write_atom_field_value<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
    field: DataField,
    i: usize,
) -> std::io::Result<()> {
    let key = field_column_key(field);
    match field {
        DataField::Id
        | DataField::Type
        | DataField::Bodyflag
        | DataField::ShapeFlag
        | DataField::Espin
        | DataField::Status
        | DataField::TemplateIndex
        | DataField::TemplateAtom => {
            let col = frame
                .get_int("atoms", key)
                .ok_or_else(|| err_mapper(format!("Missing integer column '{key}'")))?;
            write!(writer, " {}", col[i])?;
        }
        DataField::Mol => {
            let col = frame
                .get_int("atoms", keys::MOL_ID)
                .or_else(|| frame.get_int("atoms", "molecule_id"))
                .ok_or_else(|| err_mapper("Missing mol_id column"))?;
            write!(writer, " {}", col[i])?;
        }
        _ => {
            let col = frame
                .get_float("atoms", key)
                .ok_or_else(|| err_mapper(format!("Missing float column '{key}'")))?;
            write!(writer, " {}", col[i])?;
        }
    }
    Ok(())
}

fn write_topology_section<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
    section: &str,
    block: &str,
    n_members: usize,
    ids: &ndarray::ArrayViewD<'_, I>,
) -> std::io::Result<()> {
    let n = frame
        .visit_block(block, |b| b.nrows().unwrap_or(0))
        .unwrap_or(0);
    if n == 0 {
        return Ok(());
    }
    let types = frame
        .get_int(block, keys::TYPE)
        .ok_or_else(|| err_mapper(format!("Missing '{block}.type'")))?;
    let keys_ep = &keys::ENDPOINTS[..n_members];
    let mut cols = Vec::with_capacity(n_members);
    for k in keys_ep {
        cols.push(
            frame
                .get_uint(block, k)
                .ok_or_else(|| err_mapper(format!("Missing '{block}.{k}'")))?,
        );
    }

    writeln!(writer, "{section}")?;
    writeln!(writer)?;
    for i in 0..n {
        write!(writer, "{} {}", i + 1, types[i])?;
        for col in &cols {
            let atom_id = ids[col[i] as usize];
            write!(writer, " {atom_id}")?;
        }
        writeln!(writer)?;
    }
    writeln!(writer)?;
    Ok(())
}

fn write_lammps_data_frame<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
) -> std::io::Result<()> {
    writeln!(writer, "# LAMMPS data file generated by molrs")?;
    writeln!(writer)?;

    let atom_types = frame
        .get_int("atoms", keys::TYPE)
        .ok_or_else(|| err_mapper("Atoms block must contain 'type' column"))?;
    let num_atoms = atom_types.shape().first().copied().unwrap_or(0);
    let num_atom_types = atom_types.iter().max().copied().unwrap_or(1) as usize;

    let ids = frame
        .get_int("atoms", keys::ID)
        .ok_or_else(|| err_mapper("Missing 'id' column"))?;
    // Ensure core coords exist (required for every write style).
    let _x = frame
        .get_float("atoms", keys::X)
        .ok_or_else(|| err_mapper("Missing 'x' column"))?;
    let _y = frame
        .get_float("atoms", keys::Y)
        .ok_or_else(|| err_mapper("Missing 'y' column"))?;
    let _z = frame
        .get_float("atoms", keys::Z)
        .ok_or_else(|| err_mapper("Missing 'z' column"))?;

    let num_bonds = frame
        .visit_block("bonds", |b| b.nrows().unwrap_or(0))
        .unwrap_or(0);
    let num_angles = frame
        .visit_block("angles", |b| b.nrows().unwrap_or(0))
        .unwrap_or(0);
    let num_dihedrals = frame
        .visit_block("dihedrals", |b| b.nrows().unwrap_or(0))
        .unwrap_or(0);
    let num_impropers = frame
        .visit_block("impropers", |b| b.nrows().unwrap_or(0))
        .unwrap_or(0);

    let num_bond_types = frame
        .get_int("bonds", keys::TYPE)
        .map(|t| t.iter().max().copied().unwrap_or(1) as usize)
        .unwrap_or(0);
    let num_angle_types = frame
        .get_int("angles", keys::TYPE)
        .map(|t| t.iter().max().copied().unwrap_or(1) as usize)
        .unwrap_or(0);
    let num_dihedral_types = frame
        .get_int("dihedrals", keys::TYPE)
        .map(|t| t.iter().max().copied().unwrap_or(1) as usize)
        .unwrap_or(0);
    let num_improper_types = frame
        .get_int("impropers", keys::TYPE)
        .map(|t| t.iter().max().copied().unwrap_or(1) as usize)
        .unwrap_or(0);

    writeln!(writer, "{num_atoms} atoms")?;
    if num_bonds > 0 {
        writeln!(writer, "{num_bonds} bonds")?;
    }
    if num_angles > 0 {
        writeln!(writer, "{num_angles} angles")?;
    }
    if num_dihedrals > 0 {
        writeln!(writer, "{num_dihedrals} dihedrals")?;
    }
    if num_impropers > 0 {
        writeln!(writer, "{num_impropers} impropers")?;
    }
    writeln!(writer, "{num_atom_types} atom types")?;
    if num_bond_types > 0 {
        writeln!(writer, "{num_bond_types} bond types")?;
    }
    if num_angle_types > 0 {
        writeln!(writer, "{num_angle_types} angle types")?;
    }
    if num_dihedral_types > 0 {
        writeln!(writer, "{num_dihedral_types} dihedral types")?;
    }
    if num_improper_types > 0 {
        writeln!(writer, "{num_improper_types} improper types")?;
    }
    writeln!(writer)?;

    let (box_origin, box_lengths, tilts) = if let Some(sb) = frame.simbox_ref() {
        let o = sb.origin_view();
        let l = sb.lengths();
        let t = sb.tilts();
        ([o[0], o[1], o[2]], [l[0], l[1], l[2]], [t[0], t[1], t[2]])
    } else {
        ([0.0; 3], [1.0; 3], [0.0; 3])
    };
    writeln!(
        writer,
        "{} {} xlo xhi",
        box_origin[0],
        box_origin[0] + box_lengths[0]
    )?;
    writeln!(
        writer,
        "{} {} ylo yhi",
        box_origin[1],
        box_origin[1] + box_lengths[1]
    )?;
    writeln!(
        writer,
        "{} {} zlo zhi",
        box_origin[2],
        box_origin[2] + box_lengths[2]
    )?;
    if tilts.iter().any(|&t| t != 0.0) {
        writeln!(writer, "{} {} {} xy xz yz", tilts[0], tilts[1], tilts[2])?;
    }
    writeln!(writer)?;

    let meta = frame.meta_ref();
    write_meta_type_labels(
        writer,
        "Atom Type Labels",
        meta.get("atom_type_labels").and_then(|v| v.as_str()),
    )?;
    write_meta_type_labels(
        writer,
        "Bond Type Labels",
        meta.get("bond_type_labels").and_then(|v| v.as_str()),
    )?;

    // Masses: emit per-type mass when a mass column exists and style is not
    // body (body carries per-atom mass on the Atoms line).
    let masses = frame.get_float("atoms", keys::MASS);
    let (style_name, layout) = infer_write_style(|f| frame_has_atom_field(frame, f));
    let body_style = style_name == "body";
    if let Some(mass_col) = masses.as_ref()
        && !body_style
    {
        writeln!(writer, "Masses")?;
        writeln!(writer)?;
        // First-seen mass per type.
        let mut seen = vec![false; num_atom_types + 1];
        let mut type_mass = vec![0.0_f64; num_atom_types + 1];
        for i in 0..num_atoms {
            let t = atom_types[i] as usize;
            if t > 0 && t <= num_atom_types && !seen[t] {
                seen[t] = true;
                type_mass[t] = mass_col[i];
            }
        }
        for (t, m) in type_mass
            .iter()
            .enumerate()
            .take(num_atom_types + 1)
            .skip(1)
        {
            writeln!(writer, "{t} {m}")?;
        }
        writeln!(writer)?;
    }

    let has_image = frame.get_int("atoms", "ix").is_some()
        && frame.get_int("atoms", "iy").is_some()
        && frame.get_int("atoms", "iz").is_some();

    writeln!(writer, "Atoms # {style_name}")?;
    writeln!(writer)?;
    for i in 0..num_atoms {
        // First field is always id without leading space.
        let id_col = frame
            .get_int("atoms", keys::ID)
            .ok_or_else(|| err_mapper("Missing id"))?;
        write!(writer, "{}", id_col[i])?;
        for &field in layout.fields.iter().skip(1) {
            // Id already written; remaining fields including Type/xyz.
            if field == DataField::Id {
                continue;
            }
            write_atom_field_value(writer, frame, field, i)?;
        }
        // Ensure Type was in layout (always is). Core X/Y/Z always present.
        if has_image {
            let ix = frame.get_int("atoms", "ix").unwrap();
            let iy = frame.get_int("atoms", "iy").unwrap();
            let iz = frame.get_int("atoms", "iz").unwrap();
            write!(writer, " {} {} {}", ix[i], iy[i], iz[i])?;
        }
        writeln!(writer)?;
    }
    writeln!(writer)?;

    // Velocities section when all three components exist.
    if frame.get_float("atoms", keys::VX).is_some()
        && frame.get_float("atoms", keys::VY).is_some()
        && frame.get_float("atoms", keys::VZ).is_some()
    {
        let vx = frame.get_float("atoms", keys::VX).unwrap();
        let vy = frame.get_float("atoms", keys::VY).unwrap();
        let vz = frame.get_float("atoms", keys::VZ).unwrap();
        writeln!(writer, "Velocities")?;
        writeln!(writer)?;
        for i in 0..num_atoms {
            writeln!(writer, "{} {} {} {}", ids[i], vx[i], vy[i], vz[i])?;
        }
        writeln!(writer)?;
    }

    write_topology_section(writer, frame, "Bonds", "bonds", 2, &ids)?;
    write_topology_section(writer, frame, "Angles", "angles", 3, &ids)?;
    write_topology_section(writer, frame, "Dihedrals", "dihedrals", 4, &ids)?;
    write_topology_section(writer, frame, "Impropers", "impropers", 4, &ids)?;

    Ok(())
}

// ============================================================================
// Public API + streaming index
// ============================================================================

pub fn read_lammps_data<P: AsRef<Path>>(path: P) -> std::io::Result<Frame> {
    let file = File::open(path)?;
    let mut reader = LAMMPSDataReader::new(BufReader::new(file));
    reader
        .read_frame()?
        .ok_or_else(|| err_mapper("No frame found in LAMMPS data file"))
}

pub fn write_lammps_data<P: AsRef<Path>>(path: P, frame: &impl FrameAccess) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    write_lammps_data_frame(&mut writer, frame)
}

pub fn parse_frame_bytes(bytes: &[u8]) -> std::io::Result<Frame> {
    let mut reader = LAMMPSDataReader::new(Cursor::new(bytes));
    reader
        .read_frame()?
        .ok_or_else(|| err_mapper("No frame found in LAMMPS data slice"))
}

pub struct LammpsDataIndexBuilder {
    bytes_seen: u64,
}

impl Default for LammpsDataIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LammpsDataIndexBuilder {
    pub fn new() -> Self {
        Self { bytes_seen: 0 }
    }
}

impl FrameIndexBuilder for LammpsDataIndexBuilder {
    fn feed(&mut self, chunk: &[u8], global_offset: u64) {
        self.bytes_seen = global_offset.saturating_add(chunk.len() as u64);
    }
    fn drain(&mut self) -> Vec<FrameIndexEntry> {
        Vec::new()
    }
    fn finish(self: Box<Self>) -> std::io::Result<Vec<FrameIndexEntry>> {
        if self.bytes_seen == 0 {
            return Ok(Vec::new());
        }
        if self.bytes_seen > u32::MAX as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "LAMMPS data file exceeds 4 GiB",
            ));
        }
        Ok(vec![FrameIndexEntry {
            byte_offset: 0,
            byte_len: self.bytes_seen as u32,
        }])
    }
    fn bytes_seen(&self) -> u64 {
        self.bytes_seen
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod streaming_tests {
    use super::*;

    const TINY_DATA: &str = concat!(
        "LAMMPS data file\n\n2 atoms\n1 atom types\n\n",
        "0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n",
        "Atoms\n\n1 1 0.0 0.0 0.0\n2 1 1.0 0.0 0.0\n",
    );

    fn build_chunked(bytes: &[u8], cs: usize) -> Vec<FrameIndexEntry> {
        let mut b = Box::new(LammpsDataIndexBuilder::new());
        let mut off: u64 = 0;
        let mut out = Vec::new();
        for piece in bytes.chunks(cs.max(1)) {
            b.feed(piece, off);
            off += piece.len() as u64;
            out.extend(b.drain());
        }
        out.extend(b.finish().expect("finish"));
        out
    }

    #[test]
    fn lammps_data_streaming_single_frame() {
        let bytes = TINY_DATA.as_bytes();
        let one = build_chunked(bytes, bytes.len());
        for cs in [1usize, 7, 31, 64, bytes.len()] {
            assert_eq!(one, build_chunked(bytes, cs));
        }
        let frame = parse_frame_bytes(bytes).expect("parse");
        assert_eq!(frame.get("atoms").unwrap().nrows().unwrap(), 2);
    }
}

#[cfg(test)]
mod atom_style_tests {
    use super::*;
    use molrs::store::frame_access::FrameAccess;

    fn parse_text(text: &str) -> Frame {
        parse_frame_bytes(text.as_bytes()).expect("parse")
    }

    fn xyz(frame: &Frame, i: usize) -> (f64, f64, f64) {
        (
            frame.get_float("atoms", keys::X).unwrap()[i],
            frame.get_float("atoms", keys::Y).unwrap()[i],
            frame.get_float("atoms", keys::Z).unwrap()[i],
        )
    }

    #[test]
    fn normalize_strips_accelerator_suffixes() {
        use crate::io::lammps::atom_style::normalize_atom_style;
        assert_eq!(normalize_atom_style("angle/kk"), "angle");
        assert_eq!(normalize_atom_style("bpm/sphere"), "bpm/sphere");
        assert_eq!(normalize_atom_style("ANGLE/KK"), "angle");
    }

    #[test]
    fn style_hint_parses_angle_kk() {
        assert_eq!(
            parse_atoms_style_hint("Atoms # angle/kk"),
            Some("angle".into())
        );
        assert_eq!(
            parse_atoms_style_hint("Atoms # hybrid charge bond"),
            Some("hybrid".into())
        );
    }

    #[test]
    fn angle_kk_with_image_flags() {
        let text = concat!(
            "LAMMPS data file\n\n2 atoms\n1 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Atoms # angle/kk\n\n",
            "1 42 1 1.5 2.5 3.5 0 0 1\n",
            "2 0 2 4.0 5.0 6.0 1 -1 0\n",
        );
        let frame = parse_text(text);
        assert_eq!(frame.get_int("atoms", keys::MOL_ID).unwrap()[0], 42);
        assert_eq!(xyz(&frame, 0), (1.5, 2.5, 3.5));
        assert_eq!(frame.get_int("atoms", "iz").unwrap()[0], 1);
        assert!(frame.get_float("atoms", keys::CHARGE).is_none());
    }

    #[test]
    fn bond_and_molecular_styles() {
        for style in ["bond", "molecular", "angle"] {
            let text = format!(
                "LAMMPS data file\n\n1 atoms\n1 atom types\n\n\
                 0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n\
                 Atoms # {style}\n\n7 3 2 0.1 0.2 0.3\n"
            );
            let frame = parse_text(&text);
            assert_eq!(frame.get_int("atoms", keys::MOL_ID).unwrap()[0], 3);
            assert_eq!(xyz(&frame, 0), (0.1, 0.2, 0.3));
        }
    }

    #[test]
    fn full_charge_atomic_sphere_body_dipole() {
        let full = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # full\n\n1 2 3 -0.8 1.0 2.0 3.0 1 0 -1\n",
        );
        let f = parse_text(full);
        assert!((f.get_float("atoms", keys::CHARGE).unwrap()[0] + 0.8).abs() < 1e-12);
        assert_eq!(xyz(&f, 0), (1.0, 2.0, 3.0));
        assert_eq!(f.get_int("atoms", "ix").unwrap()[0], 1);

        let charge = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # charge\n\n5 1 -0.5 9.0 8.0 7.0\n",
        );
        let f = parse_text(charge);
        assert_eq!(xyz(&f, 0), (9.0, 8.0, 7.0));

        let sphere = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # sphere\n\n1 1 1.0 2.5 3.0 4.0 5.0\n",
        );
        let f = parse_text(sphere);
        assert_eq!(xyz(&f, 0), (3.0, 4.0, 5.0));
        assert!((f.get_float("atoms", "diameter").unwrap()[0] - 1.0).abs() < 1e-12);

        let body = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # body\n\n1 1 1 6.0 -1.5 -2.5 0.0 1 2 0\n",
        );
        let f = parse_text(body);
        assert!((f.get_float("atoms", keys::MASS).unwrap()[0] - 6.0).abs() < 1e-12);
        assert_eq!(xyz(&f, 0), (-1.5, -2.5, 0.0));

        let dipole = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # dipole\n\n1 1 0.5 1.0 2.0 3.0 0.1 0.2 0.3\n",
        );
        let f = parse_text(dipole);
        assert!((f.get_float("atoms", keys::MUX).unwrap()[0] - 0.1).abs() < 1e-12);
        assert_eq!(xyz(&f, 0), (1.0, 2.0, 3.0));
    }

    #[test]
    fn pe_angle_kk_sample_and_masses() {
        let text = concat!(
            "LAMMPS data file\n\n2 atoms\n2 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Masses\n\n1 12.0\n2 1.0\n\n",
            "Atoms # angle\n\n",
            "182153 45539 1 1.9 1.8 1.6 0 0 1\n",
            "10 0 2 0.5 0.5 0.5 0 0 0\n",
        );
        let frame = parse_text(text);
        assert_eq!(frame.get_int("atoms", keys::MOL_ID).unwrap()[0], 45539);
        assert!((frame.get_float("atoms", keys::MASS).unwrap()[0] - 12.0).abs() < 1e-12);
        assert!((frame.get_float("atoms", keys::MASS).unwrap()[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fixtures_body_and_full() {
        let root =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests-data/lammps-data");
        let body = read_lammps_data(root.join("data.body")).expect("body");
        assert!(body.get_int("atoms", "bodyflag").is_some());
        let full = read_lammps_data(root.join("molid.lmp")).expect("molid");
        assert!(full.get_int("atoms", keys::MOL_ID).is_some());
        assert!(full.get_float("atoms", keys::CHARGE).is_some());
    }

    #[test]
    fn topology_bonds_angles() {
        let text = concat!(
            "LAMMPS data file\n\n3 atoms\n2 bonds\n1 angles\n1 atom types\n",
            "1 bond types\n1 angle types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Atoms # molecular\n\n",
            "1 1 1 0 0 0\n2 1 1 1 0 0\n3 1 1 2 0 0\n\n",
            "Bonds\n\n1 1 1 2\n2 1 2 3\n\n",
            "Angles\n\n1 1 1 2 3\n",
        );
        let frame = parse_text(text);
        assert_eq!(frame.get("bonds").unwrap().nrows().unwrap(), 2);
        assert_eq!(
            (
                frame.get_uint("bonds", keys::ATOMI).unwrap()[0],
                frame.get_uint("bonds", keys::ATOMJ).unwrap()[0]
            ),
            (0, 1)
        );
    }

    #[test]
    fn nine_columns_without_hint_is_molecular() {
        let text = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms\n\n1 42 1 1.5 2.5 3.5 0 0 1\n",
        );
        let frame = parse_text(text);
        assert_eq!(frame.get_int("atoms", keys::MOL_ID).unwrap()[0], 42);
        assert_eq!(xyz(&frame, 0), (1.5, 2.5, 3.5));
    }

    #[test]
    fn write_round_trip_full_with_image_and_velocities() {
        let text = concat!(
            "LAMMPS data file\n\n2 atoms\n1 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Masses\n\n1 12.0\n\n",
            "Atoms # full\n\n",
            "1 1 1 -0.5 1.0 2.0 3.0 0 0 1\n",
            "2 1 1 0.5 4.0 5.0 6.0 1 -1 0\n",
            "\nVelocities\n\n",
            "1 0.1 0.2 0.3\n",
            "2 -0.1 0.0 0.5\n",
        );
        let frame = parse_text(text);
        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("Atoms # full"), "{out}");
        assert!(out.contains("Velocities"), "{out}");
        assert!(out.contains("Masses"), "{out}");
        // Image flags preserved
        assert!(out.contains("0 0 1") || out.contains(" 0 0 1\n"), "{out}");

        let frame2 = parse_frame_bytes(out.as_bytes()).expect("re-read");
        assert_eq!(frame2.get("atoms").unwrap().nrows().unwrap(), 2);
        assert!((frame2.get_float("atoms", keys::CHARGE).unwrap()[0] + 0.5).abs() < 1e-12);
        assert!((frame2.get_float("atoms", keys::VX).unwrap()[0] - 0.1).abs() < 1e-12);
        assert_eq!(frame2.get_int("atoms", "iz").unwrap()[0], 1);
    }

    #[test]
    fn write_round_trip_sphere_and_dipole() {
        let sphere = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # sphere\n\n1 1 1.0 2.5 3.0 4.0 5.0\n",
        );
        let frame = parse_text(sphere);
        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("Atoms # sphere"), "{out}");
        let f2 = parse_frame_bytes(out.as_bytes()).unwrap();
        assert!((f2.get_float("atoms", "diameter").unwrap()[0] - 1.0).abs() < 1e-12);
        assert_eq!(
            (
                f2.get_float("atoms", keys::X).unwrap()[0],
                f2.get_float("atoms", keys::Y).unwrap()[0],
                f2.get_float("atoms", keys::Z).unwrap()[0],
            ),
            (3.0, 4.0, 5.0)
        );

        let dipole = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # dipole\n\n1 1 0.5 1.0 2.0 3.0 0.1 0.2 0.3\n",
        );
        let frame = parse_text(dipole);
        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("Atoms # dipole"), "{out}");
        let f2 = parse_frame_bytes(out.as_bytes()).unwrap();
        assert!((f2.get_float("atoms", keys::MUX).unwrap()[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn write_round_trip_topology_angles() {
        let text = concat!(
            "LAMMPS data file\n\n3 atoms\n2 bonds\n1 angles\n1 atom types\n",
            "1 bond types\n1 angle types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Atoms # molecular\n\n",
            "1 1 1 0 0 0\n2 1 1 1 0 0\n3 1 1 2 0 0\n\n",
            "Bonds\n\n1 1 1 2\n2 1 2 3\n\n",
            "Angles\n\n1 1 1 2 3\n",
        );
        let frame = parse_text(text);
        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("2 bonds"), "{out}");
        assert!(out.contains("1 angles"), "{out}");
        assert!(out.contains("Atoms # molecular"), "{out}");
        let f2 = parse_frame_bytes(out.as_bytes()).unwrap();
        assert_eq!(f2.get("bonds").unwrap().nrows().unwrap(), 2);
        assert_eq!(f2.get("angles").unwrap().nrows().unwrap(), 1);
    }

    #[test]
    fn dump_aliases_q_and_mol() {
        use crate::io::lammps::common::{canonical_dump_column, native_dump_column};
        assert_eq!(canonical_dump_column("q"), keys::CHARGE);
        assert_eq!(canonical_dump_column("mol"), keys::MOL_ID);
        assert_eq!(canonical_dump_column("molecule"), keys::MOL_ID);
        assert_eq!(native_dump_column(keys::CHARGE), "q");
        assert_eq!(native_dump_column(keys::MOL_ID), "mol");
        assert_eq!(canonical_dump_column("spin"), "espin");
    }
}
