//! LAMMPS dump trajectory file reader and writer.
//!
//! Implements support for LAMMPS dump files as output by the `dump` command:
//! <https://docs.lammps.org/dump.html>
//!
//! Column names in `ITEM: ATOMS …` are the source of truth (unlike data files,
//! which encode layout via `atom_style`). Shared helpers — error mapping,
//! column aliases (`q`→`charge`, `mol`→`mol_id`), SimBox construction — live
//! in the internal `io::lammps` module and are reused by the data-file reader.
//!
//! # Supported Features
//!
//! - Multi-frame trajectory reading with random access via `TrajectoryReader`
//! - Orthogonal and triclinic simulation boxes
//! - Automatic column type detection (integer vs float) with promote-on-demand
//! - Boundary condition flag parsing (`pp`, `ff`, `ss`, etc.)
//! - Gzip-compressed files via `open_lammps_dump`
//! - Canonical field rename for style-related dump columns (`q`, `mol`, …)
//!
//! # Examples
//!
//! ```no_run
//! use molrs::io::trajectory::lammps_dump::{read_lammps_dump, open_lammps_dump, write_lammps_dump};
//!
//! # fn main() -> std::io::Result<()> {
//! let frames = read_lammps_dump("trajectory.lammpstrj")?;
//! use molrs::io::reader::TrajectoryReader;
//! let mut reader = open_lammps_dump("trajectory.lammpstrj")?;
//! let frame_5 = reader.read_step(5)?;
//! write_lammps_dump("output.lammpstrj", &frames, None)?;
//! # Ok(())
//! # }
//! ```

use crate::io::lammps::box_bounds::{BoxBounds, pbc_from_boundary_tokens, simbox_from_bounds};
use crate::io::lammps::common::{
    canonical_dump_column, err_mapper, insert_f, insert_i, insert_str, insert_u,
    is_integer_dump_column, is_string_dump_column, native_dump_column,
};
use crate::io::reader::{FrameIndex, FrameReader, ReadSeek, Reader, TrajectoryReader};
use crate::io::writer::{FrameWriter, Writer};
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::store::frame_access::FrameAccess;
use molrs::types::Idx;
use molrs::types::{F, I};
use once_cell::sync::OnceCell;
use std::fs::File;
use std::io::{BufRead, Seek, SeekFrom, Write};
use std::path::Path;

// ============================================================================
// Helpers
// ============================================================================

/// Column type classification for LAMMPS dump columns.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ColumnType {
    Integer,
    /// Unsigned — canonical ids and relation endpoints.
    Unsigned,
    Float,
    String,
}

/// Whether a frame's data section came from the per-atom (`dump
/// atom/custom`) or per-entry (`dump local`) flavor of the LAMMPS dump
/// format. Picked by which header keyword starts the count line:
/// `ITEM: NUMBER OF ATOMS` vs. one of [`LOCAL_LABELS`]. Determines the
/// destination block name on the resulting [`Frame`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum BlockKind {
    Atoms,
    Entries,
}

/// Section labels a `dump local` frame may carry, in the `ITEM: NUMBER OF
/// <LABEL>` / `ITEM: <LABEL> col…` pair.
///
/// `ENTRIES` is what LAMMPS writes by default; the rest come from
/// [`dump_modify label <LABEL>`], which OVITO's LAMMPS-dump-local reader
/// documents as the thing to set — its manual tells users to write
/// `dump_modify bond_dump label BONDS`, so files in the wild carry `BONDS`
/// at least as often as the default. Accepting only `ENTRIES` rejected
/// exactly the setup that manual recommends.
///
/// The label says what the rows *mean*; it does not change how they parse,
/// and it does not earn the rows a contract-bearing block name — see the
/// note at the `block_name` binding below. All of them land in `entries`.
///
/// [`dump_modify label <LABEL>`]: https://docs.lammps.org/dump_modify.html
/// Reference: <https://www.ovito.org/manual/reference/file_formats/input/lammps_dump_local.html>
const LOCAL_LABELS: &[&str] = &[
    "ENTRIES",
    "BONDS",
    "ANGLES",
    "DIHEDRALS",
    "IMPROPERS",
    "NEIGHBORS",
];

/// The `dump local` label in a `ITEM: NUMBER OF <LABEL>` line, if any.
fn local_label_of(count_header: &str) -> Option<&'static str> {
    let tail = count_header.strip_prefix("ITEM: NUMBER OF ")?.trim();
    LOCAL_LABELS.iter().copied().find(|label| tail == *label)
}

/// Classify a LAMMPS dump column by **canonical** name (post-alias).
///
/// Used by the *writer* to pick a per-column print format. Reader-side
/// typing is value-based (promote-on-demand) because dump column names are
/// user-defined (`c_X[N]`, `f_reax[1]`, …). Integer/string sets come from
/// the dump custom + property/atom attribute lists (see `is_*_dump_column`).
fn classify_column(name: &str) -> ColumnType {
    // A canonical key's dtype is declared by the vocabulary, not guessed from
    // its name. Without this, `id` and `mol_id` are stored signed because the
    // hardcoded name list says "integer", and every consumer reading them as
    // unsigned silently sees nothing.
    if let Some(spec) = molrs::store::schema::column(&canonical_dump_column(name)) {
        return match spec.dtype {
            molrs::store::block::DType::Float => ColumnType::Float,
            molrs::store::block::DType::String => ColumnType::String,
            molrs::store::block::DType::UInt => ColumnType::Unsigned,
            _ => ColumnType::Integer,
        };
    }
    if is_integer_dump_column(name) {
        ColumnType::Integer
    } else if is_string_dump_column(name) {
        ColumnType::String
    } else {
        ColumnType::Float
    }
}

#[inline]
fn canonical_column_name(name: &str) -> String {
    canonical_dump_column(name)
}

#[inline]
fn native_column_name(name: &str) -> &str {
    native_dump_column(name)
}

// ============================================================================
// Parsing
// ============================================================================

/// Parsed box bounds from a single LAMMPS dump frame header.
#[derive(Debug, Clone)]
struct DumpBoxBounds {
    xlo: f64,
    xhi: f64,
    ylo: f64,
    yhi: f64,
    zlo: f64,
    zhi: f64,
    xy: Option<f64>,
    xz: Option<f64>,
    yz: Option<f64>,
    boundary_raw: [String; 3],
}

impl DumpBoxBounds {
    /// Parse the BOX BOUNDS header line to detect triclinic and boundary flags.
    ///
    /// Format: `ITEM: BOX BOUNDS [xy xz yz] bb bb bb`
    /// where bb is pp, ff, ss, fs, sf, etc.
    fn parse_header(header: &str) -> std::io::Result<(bool, [String; 3])> {
        // Strip "ITEM: BOX BOUNDS" prefix
        let rest = header.strip_prefix("ITEM: BOX BOUNDS").unwrap_or("").trim();

        let tokens: Vec<&str> = rest.split_whitespace().collect();

        // Detect triclinic: header contains "xy xz yz" before boundary flags
        let (is_triclinic, boundary_tokens) =
            if tokens.len() >= 6 && tokens[0] == "xy" && tokens[1] == "xz" && tokens[2] == "yz" {
                (true, &tokens[3..])
            } else {
                (false, tokens.as_slice())
            };

        let boundary_raw = if boundary_tokens.len() >= 3 {
            [
                boundary_tokens[0].to_string(),
                boundary_tokens[1].to_string(),
                boundary_tokens[2].to_string(),
            ]
        } else {
            ["pp".to_string(), "pp".to_string(), "pp".to_string()]
        };

        Ok((is_triclinic, boundary_raw))
    }

    /// Parse 3 box bound lines (orthogonal or triclinic).
    fn parse_lines<R: BufRead>(
        reader: &mut R,
        is_triclinic: bool,
        boundary_raw: [String; 3],
    ) -> std::io::Result<Self> {
        let mut line = String::new();

        // Line 1: xlo xhi [xy]
        line.clear();
        reader.read_line(&mut line)?;
        let vals: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().map_err(err_mapper))
            .collect::<Result<_, _>>()?;

        if vals.len() < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "box line 1: expected at least 2 values",
            ));
        }
        let (xlo_bound, xhi_bound) = (vals[0], vals[1]);
        let xy = if is_triclinic {
            Some(*vals.get(2).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "triclinic box line 1: missing tilt factor xy",
                )
            })?)
        } else {
            None
        };

        // Line 2: ylo yhi [xz]
        line.clear();
        reader.read_line(&mut line)?;
        let vals: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().map_err(err_mapper))
            .collect::<Result<_, _>>()?;

        if vals.len() < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "box line 2: expected at least 2 values",
            ));
        }
        let (ylo_bound, yhi_bound) = (vals[0], vals[1]);
        let xz = if is_triclinic {
            Some(*vals.get(2).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "triclinic box line 2: missing tilt factor xz",
                )
            })?)
        } else {
            None
        };

        // Line 3: zlo zhi [yz]
        line.clear();
        reader.read_line(&mut line)?;
        let vals: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().map_err(err_mapper))
            .collect::<Result<_, _>>()?;

        if vals.len() < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "box line 3: expected at least 2 values",
            ));
        }
        let (zlo_bound, zhi_bound) = (vals[0], vals[1]);
        let yz = if is_triclinic {
            Some(*vals.get(2).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "triclinic box line 3: missing tilt factor yz",
                )
            })?)
        } else {
            None
        };

        // For triclinic, convert bounds to actual box limits
        // See: https://docs.lammps.org/Howto_triclinic.html
        if is_triclinic {
            let xy_v = xy.unwrap_or(0.0);
            let xz_v = xz.unwrap_or(0.0);
            let yz_v = yz.unwrap_or(0.0);

            let xlo = xlo_bound - f64::min(0.0, f64::min(xy_v, f64::min(xz_v, xy_v + xz_v)));
            let xhi = xhi_bound - f64::max(0.0, f64::max(xy_v, f64::max(xz_v, xy_v + xz_v)));
            let ylo = ylo_bound - f64::min(0.0, yz_v);
            let yhi = yhi_bound - f64::max(0.0, yz_v);
            let zlo = zlo_bound;
            let zhi = zhi_bound;

            Ok(Self {
                xlo,
                xhi,
                ylo,
                yhi,
                zlo,
                zhi,
                xy,
                xz,
                yz,
                boundary_raw,
            })
        } else {
            Ok(Self {
                xlo: xlo_bound,
                xhi: xhi_bound,
                ylo: ylo_bound,
                yhi: yhi_bound,
                zlo: zlo_bound,
                zhi: zhi_bound,
                xy: None,
                xz: None,
                yz: None,
                boundary_raw,
            })
        }
    }
}

/// Parse a single LAMMPS dump frame from the current reader position.
///
/// Returns `Ok(None)` on EOF.
fn parse_single_frame<R: BufRead>(reader: &mut R) -> std::io::Result<Option<Frame>> {
    let mut line = String::new();

    // -- ITEM: TIMESTEP (skip optional ITEM: UNITS / ITEM: TIME headers) --
    let timestep: i64 = loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(None); // EOF
        }
        let trimmed = line.trim();
        if trimmed.starts_with("ITEM: TIMESTEP") {
            line.clear();
            reader.read_line(&mut line)?;
            break line.trim().parse().map_err(err_mapper)?;
        }
        if trimmed.starts_with("ITEM:") {
            // Unknown optional ITEM (e.g. UNITS, TIME) — skip its value line.
            line.clear();
            reader.read_line(&mut line)?;
        } else {
            return Err(err_mapper(format!(
                "Expected 'ITEM: TIMESTEP', got: {}",
                trimmed
            )));
        }
    };

    // -- ITEM: NUMBER OF ATOMS  /  ITEM: NUMBER OF <LOCAL_LABELS> --
    //
    // Two flavors of LAMMPS dump output share this parser:
    //   * `dump atom/custom` writes per-atom rows under
    //     `ITEM: NUMBER OF ATOMS` + `ITEM: ATOMS …`.
    //   * `dump local` writes per-bond / per-angle / per-pair-distance rows
    //     under `ITEM: NUMBER OF <LABEL>` + `ITEM: <LABEL> …`, where LABEL
    //     is `ENTRIES` by default and anything in [`LOCAL_LABELS`] once
    //     `dump_modify label` has been used.
    //
    // The per-row schema is identical (whitespace-separated tokens, one
    // line per row), so we accept either header keyword and stash a
    // `BlockKind` discriminator plus the label to pick the destination
    // block name and the data-header keyword when we build the Frame.
    line.clear();
    reader.read_line(&mut line)?;
    let (block_kind, local_label) = if line.trim().starts_with("ITEM: NUMBER OF ATOMS") {
        (BlockKind::Atoms, "ATOMS")
    } else if let Some(label) = local_label_of(line.trim()) {
        (BlockKind::Entries, label)
    } else {
        return Err(err_mapper(format!(
            "Expected 'ITEM: NUMBER OF ATOMS' or 'ITEM: NUMBER OF <{}>', got: {}",
            LOCAL_LABELS.join("|"),
            line.trim()
        )));
    };

    line.clear();
    reader.read_line(&mut line)?;
    let nrows: usize = line.trim().parse().map_err(err_mapper)?;

    // -- ITEM: BOX BOUNDS --
    line.clear();
    reader.read_line(&mut line)?;
    if !line.trim().starts_with("ITEM: BOX BOUNDS") {
        return Err(err_mapper(format!(
            "Expected 'ITEM: BOX BOUNDS', got: {}",
            line.trim()
        )));
    }

    let (is_triclinic, boundary_raw) = DumpBoxBounds::parse_header(line.trim())?;
    let bounds = DumpBoxBounds::parse_lines(reader, is_triclinic, boundary_raw)?;

    // -- ITEM: ATOMS  /  ITEM: <label> --
    //
    // The data header repeats the label from the count line, so a frame
    // that counted BONDS must name BONDS here too.
    line.clear();
    reader.read_line(&mut line)?;
    let header_keyword = format!("ITEM: {}", local_label);
    if !line.trim().starts_with(header_keyword.as_str()) {
        return Err(err_mapper(format!(
            "Expected '{}', got: {}",
            header_keyword,
            line.trim()
        )));
    }

    // Extract column names from "ITEM: <keyword> col1 col2 ..."
    let header_tail = line
        .trim()
        .strip_prefix(header_keyword.as_str())
        .unwrap_or("")
        .trim();
    let col_names: Vec<String> = header_tail
        .split_whitespace()
        .map(canonical_column_name)
        .collect();

    if col_names.is_empty() {
        return Err(err_mapper(format!(
            "{} header has no column names",
            header_keyword
        )));
    }

    let ncols = col_names.len();

    // Atoms are kept in file order, NOT sorted by `id`. Per-row order
    // out of LAMMPS's `dump custom`/`dump local` reflects each MPI
    // rank's local atom storage; bonds emitted by `compute property/
    // local` in companion `dump local` files are typically indexed by
    // that same per-row position rather than by atom id. Re-sorting
    // atom rows on read would break those bond mappings — bonds.dump's
    // batom1/batom2 (or equivalent) point at "the atom written at file
    // row K", not at "the atom whose id is K". Keep both flavors in
    // file order and let the user pick a 0-/1-based offset in the
    // BondColumnRemap dialog if needed.
    //
    // Note: this means atom rows can shuffle across frames if MPI
    // rebalancing happens. That's a property of the LAMMPS dump
    // protocol — the user can add `dump_modify sort id` to their LAMMPS
    // script to stabilize order on the writer side.

    // Per-column typed buffers. Exactly one of int/float/str is `Some`
    // at any moment for each column; the active one matches `col_types[i]`.
    // Promotion (Integer → Float → String) drains the old buffer and
    // converts each existing value to the wider type before continuing
    // — see the match arms below.
    // Seed from the vocabulary, not from Integer: promote-on-demand widens a
    // column when a token does not fit, but it can never *narrow*, so a
    // canonical Float column whose file happens to hold whole numbers would
    // stay Integer forever. Unspec'd columns still start at Integer and promote.
    // Canonical keys are typed by the vocabulary; everything else keeps the
    // promote-on-demand behaviour (start Integer, widen when a token does not
    // fit), because a `dump local` column like `batom1` has no spec and its
    // type is genuinely a property of the data.
    let mut col_types: Vec<ColumnType> = col_names
        .iter()
        .map(|n| {
            if molrs::store::schema::column(&canonical_dump_column(n)).is_some() {
                classify_column(n)
            } else {
                ColumnType::Integer
            }
        })
        .collect();
    // Buffers follow the seeded type. Previously every column started Integer
    // so only `int_cols` was pre-allocated and the promotion path allocated the
    // others; seeding from the vocabulary means a column can *begin* as Float
    // or String, and its buffer has to exist before the first row.
    let mut int_cols: Vec<Option<Vec<I>>> = col_types
        .iter()
        .map(|t| {
            matches!(t, ColumnType::Integer | ColumnType::Unsigned)
                .then(|| Vec::with_capacity(nrows))
        })
        .collect();
    let mut float_cols: Vec<Option<Vec<F>>> = col_types
        .iter()
        .map(|t| matches!(t, ColumnType::Float).then(|| Vec::with_capacity(nrows)))
        .collect();
    let mut str_cols: Vec<Option<Vec<std::string::String>>> = col_types
        .iter()
        .map(|t| matches!(t, ColumnType::String).then(|| Vec::with_capacity(nrows)))
        .collect();

    // --- Single pass: walk rows in file order, push into typed columns ---
    //
    // Promote-on-demand value-based typing: every column starts at
    // Integer (the narrowest); the first token that doesn't parse as
    // the current type triggers a one-shot promotion of that column's
    // already-collected values to the wider type, and the loop
    // continues with the new type cached in `col_types[i]`. Per-cell
    // cost is one `i64`/`f64::parse` in the steady state; promotions
    // happen at most twice per column over the whole file (Integer →
    // Float once, Float → String once) and are bounded O(rows-already-
    // collected).
    //
    // Why promote-on-demand instead of "probe row 0, lock in types,
    // dispatch the rest": LAMMPS' `%g` float format prints exact zeros
    // as `0` (no decimal point, no exponent), so a column whose first
    // atom sits at the origin would lock as Integer and then panic on
    // row 2's `0.693361`.
    for row in 0..nrows {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Err(err_mapper(format!(
                "Unexpected EOF at row {} (expected {})",
                row, nrows
            )));
        }

        let mut tokens = line.split_whitespace();
        for i in 0..ncols {
            let token = tokens.next().ok_or_else(|| {
                err_mapper(format!("Row {} has fewer than {} tokens", row, ncols))
            })?;
            match col_types[i] {
                ColumnType::Integer | ColumnType::Unsigned => {
                    if let Ok(v) = token.parse::<I>() {
                        int_cols[i].as_mut().unwrap().push(v);
                    } else if let Ok(v) = token.parse::<F>() {
                        // Integer → Float: lift accumulated ints into a
                        // Vec<F> and continue with float storage. Cast
                        // is lossless for values in `i32`/`u32` range
                        // and acceptable elsewhere — the column already
                        // committed to numeric.
                        let drained = int_cols[i].take().unwrap();
                        let mut promoted: Vec<F> = Vec::with_capacity(nrows);
                        for prev in drained {
                            promoted.push(prev as F);
                        }
                        promoted.push(v);
                        float_cols[i] = Some(promoted);
                        col_types[i] = ColumnType::Float;
                    } else {
                        // Integer → String: stringify accumulated ints.
                        let drained = int_cols[i].take().unwrap();
                        let mut promoted: Vec<std::string::String> = Vec::with_capacity(nrows);
                        for prev in drained {
                            promoted.push(prev.to_string());
                        }
                        promoted.push(token.to_owned());
                        str_cols[i] = Some(promoted);
                        col_types[i] = ColumnType::String;
                    }
                }
                ColumnType::Float => {
                    if let Ok(v) = token.parse::<F>() {
                        float_cols[i].as_mut().unwrap().push(v);
                    } else {
                        // Float → String: stringify accumulated floats.
                        let drained = float_cols[i].take().unwrap();
                        let mut promoted: Vec<std::string::String> = Vec::with_capacity(nrows);
                        for prev in drained {
                            promoted.push(prev.to_string());
                        }
                        promoted.push(token.to_owned());
                        str_cols[i] = Some(promoted);
                        col_types[i] = ColumnType::String;
                    }
                }
                ColumnType::String => {
                    str_cols[i].as_mut().unwrap().push(token.to_owned());
                }
            }
        }
    }

    // Build Frame
    let mut frame = Frame::new();
    let mut data_block = Block::new();

    for (i, name) in col_names.iter().enumerate() {
        match col_types[i] {
            ColumnType::Integer => {
                insert_i(
                    &mut data_block,
                    name.as_str(),
                    int_cols[i].take().unwrap(),
                    nrows,
                )?;
            }
            ColumnType::Unsigned => {
                // Parsed through the signed buffer, stored unsigned: the
                // vocabulary declares these keys UInt and a negative value in
                // one is a malformed file, not a representable state.
                let raw = int_cols[i].take().unwrap();
                if let Some(bad) = raw.iter().find(|&&v| v < 0) {
                    return Err(err_mapper(format!(
                        "column '{}' is unsigned in the Frame schema but the dump holds {bad}",
                        name.as_str()
                    )));
                }
                insert_u(
                    &mut data_block,
                    name.as_str(),
                    raw.into_iter().map(|v| v as Idx).collect(),
                    nrows,
                )?;
            }
            ColumnType::Float => {
                insert_f(
                    &mut data_block,
                    name.as_str(),
                    float_cols[i].take().unwrap(),
                    nrows,
                )?;
            }
            ColumnType::String => {
                insert_str(
                    &mut data_block,
                    name.as_str(),
                    str_cols[i].take().unwrap(),
                    nrows,
                )?;
            }
        }
    }

    // ENTRIES (`dump local`) lands in "entries", not "bonds".
    //
    // It used to be called "bonds", which claimed the canonical bonds contract
    // — `atomi`/`atomj`, 0-indexed into `atoms` — while carrying whatever
    // columns the file happened to name (`batom1`/`batom2`, angles, pair
    // distances, …). Worse, as the note above records, those endpoints point
    // at "the atom written at file row K" with a 0-/1-based offset the *user*
    // resolves, so they are not 0-based row indices at all.
    //
    // A block that does not satisfy a contract must not take its name; the
    // schema check on read is what made this visible. Column names stay
    // as-is, and downstream stays gated on the canonical endpoints, so a
    // consumer that wants bonds still gets nothing here — it just gets
    // nothing under an honest name.
    let block_name = match block_kind {
        BlockKind::Atoms => "atoms",
        BlockKind::Entries => "entries",
    };
    frame.insert(block_name, data_block);

    // Timestep is frame-level metadata, not a box property.
    frame.meta.insert("timestep", timestep);

    // What LAMMPS itself says the local rows mean. Column names are
    // user-defined — `dump local c_bond[1] c_bond[2]` is the default spelling
    // and carries no meaning at all — so the label is the only reliable
    // signal, and a consumer must read it here rather than re-guess from the
    // header. Kept as metadata rather than a block name for the reason given
    // at `block_name` above: the rows still do not satisfy any
    // contract-bearing block's schema.
    if block_kind == BlockKind::Entries {
        frame.meta.insert("dump_local_label", local_label);
    }

    // Shared SimBox builder (same path as the data-file reader).
    let shared_bounds = BoxBounds {
        xlo: bounds.xlo,
        xhi: bounds.xhi,
        ylo: bounds.ylo,
        yhi: bounds.yhi,
        zlo: bounds.zlo,
        zhi: bounds.zhi,
        xy: bounds.xy,
        xz: bounds.xz,
        yz: bounds.yz,
        has_x: true,
        has_y: true,
        has_z: true,
    };
    let pbc = pbc_from_boundary_tokens(&bounds.boundary_raw);
    if let Some(simbox) = simbox_from_bounds(&shared_bounds, pbc)? {
        frame.simbox = Some(simbox);
    }

    Ok(Some(frame))
}

// ============================================================================
// Reader
// ============================================================================

/// LAMMPS dump trajectory reader implementing `TrajectoryReader` for random access.
///
/// Supports multi-frame dump files with lazy index building for random access.
///
/// # Examples
///
/// ```no_run
/// use molrs::io::trajectory::lammps_dump::open_lammps_dump;
/// use molrs::io::reader::TrajectoryReader;
///
/// # fn main() -> std::io::Result<()> {
/// let mut reader = open_lammps_dump("traj.lammpstrj")?;
/// let n = reader.len()?;
/// println!("Trajectory has {} frames", n);
/// let frame = reader.read_step(0)?.expect("first frame");
/// # Ok(())
/// # }
/// ```
pub struct LAMMPSTrajReader<R: BufRead> {
    reader: R,
    index: OnceCell<FrameIndex>,
}

impl<R: BufRead + Seek> LAMMPSTrajReader<R> {
    /// Create a new LAMMPS dump reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            index: OnceCell::new(),
        }
    }

    /// Build frame index by scanning for `ITEM: TIMESTEP` markers.
    fn build_index_impl(&mut self) -> std::io::Result<()> {
        if self.index.get().is_some() {
            return Ok(());
        }

        let start_pos = self.reader.stream_position()?;
        self.reader.seek(SeekFrom::Start(0))?;

        let mut frame_index = FrameIndex::new();
        let mut current_pos: u64 = 0;
        let mut line = String::new();

        loop {
            // Record potential frame start
            let frame_start = current_pos;

            // Read next line
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break; // EOF
            }
            current_pos += bytes as u64;

            if !line.trim().starts_with("ITEM: TIMESTEP") {
                continue;
            }

            // Found a frame start
            frame_index.add_frame(frame_start);

            // Skip timestep value
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            current_pos += bytes as u64;

            // Skip "ITEM: NUMBER OF ATOMS" or "ITEM: NUMBER OF ENTRIES"
            // — both flavors share this scaffolding (see parse_single_frame).
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            current_pos += bytes as u64;

            // Read row count (natoms for ATOMS, nentries for ENTRIES).
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            current_pos += bytes as u64;
            let nrows: usize = line.trim().parse().unwrap_or(0);

            // Skip "ITEM: BOX BOUNDS ..."
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            current_pos += bytes as u64;

            // Skip 3 box lines
            for _ in 0..3 {
                line.clear();
                let bytes = self.reader.read_line(&mut line)?;
                if bytes == 0 {
                    break;
                }
                current_pos += bytes as u64;
            }

            // Skip "ITEM: ATOMS ..." or "ITEM: ENTRIES ..."
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            current_pos += bytes as u64;

            // Skip nrows data lines
            for _ in 0..nrows {
                line.clear();
                let bytes = self.reader.read_line(&mut line)?;
                if bytes == 0 {
                    break;
                }
                current_pos += bytes as u64;
            }
        }

        self.reader.seek(SeekFrom::Start(start_pos))?;
        self.index
            .set(frame_index)
            .map_err(|_| std::io::Error::other("failed to set index"))?;

        Ok(())
    }

    /// Read frame at a specific byte offset.
    fn read_at_offset(&mut self, offset: u64) -> std::io::Result<Option<Frame>> {
        self.reader.seek(SeekFrom::Start(offset))?;
        parse_single_frame(&mut self.reader)
    }
}

impl<R: BufRead + Seek> Reader for LAMMPSTrajReader<R> {
    type R = R;

    fn new(reader: Self::R) -> Self {
        Self::new(reader)
    }
}

impl<R: BufRead + Seek> FrameReader for LAMMPSTrajReader<R> {
    fn read(&mut self) -> std::io::Result<Option<Frame>> {
        // Validate on the way out: a frame that violates the vocabulary
        // is a malformed file or a reader bug, not a result to return.
        crate::io::reader::validated(parse_single_frame(&mut self.reader)?)
    }
}

impl<R: BufRead + Seek> TrajectoryReader for LAMMPSTrajReader<R> {
    fn build_index(&mut self) -> std::io::Result<()> {
        self.build_index_impl()
    }

    fn read_step(&mut self, step: usize) -> std::io::Result<Option<Frame>> {
        if self.index.get().is_none() {
            self.build_index_impl()?;
        }

        let index = self.index.get().unwrap();
        if step >= index.len() {
            return Ok(None);
        }

        let offset = index.get(step).unwrap();
        self.read_at_offset(offset)
    }

    fn len(&mut self) -> std::io::Result<usize> {
        if self.index.get().is_none() {
            self.build_index_impl()?;
        }
        Ok(self.index.get().unwrap().len())
    }
}

// ============================================================================
// Writer
// ============================================================================

/// LAMMPS dump trajectory writer.
///
/// Writes frames in LAMMPS dump format. Call `write_frame` for each timestep.
///
/// # Examples
///
/// ```no_run
/// use molrs::io::trajectory::lammps_dump::write_lammps_dump;
/// use molrs::store::frame::Frame;
///
/// # fn main() -> std::io::Result<()> {
/// let frames: Vec<Frame> = vec![];
/// write_lammps_dump("output.lammpstrj", &frames, None)?;
/// # Ok(())
/// # }
/// ```
pub struct LAMMPSDumpWriter<W: Write> {
    writer: W,
}

impl<W: Write> LAMMPSDumpWriter<W> {
    /// Create a new LAMMPS dump writer.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> Writer for LAMMPSDumpWriter<W> {
    type W = W;

    fn new(writer: Self::W) -> Self {
        Self::new(writer)
    }
}

impl<W: Write> FrameWriter for LAMMPSDumpWriter<W> {
    fn write(&mut self, frame: &Frame) -> std::io::Result<()> {
        // Refuse to emit a frame that violates the vocabulary: a bad file
        // looks fine and is found wrong later, by whatever reads it.
        crate::io::writer::check_before_write(frame)?;
        write_lammps_dump_frame(&mut self.writer, frame, None)
    }
}

/// Write a single frame in LAMMPS dump format.
///
/// Accepts any type implementing [`FrameAccess`], including both [`Frame`] and
/// [`FrameView`](molrs::store::frame_view::FrameView).
///
/// `columns` is the caller's `dump custom` line: `Some` writes exactly those
/// columns in that order, `None` writes every column the block holds.
fn write_lammps_dump_frame<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
    columns: Option<&[&str]>,
) -> std::io::Result<()> {
    let natoms = frame
        .visit_block("atoms", |b| b.nrows().unwrap_or(0))
        .ok_or_else(|| err_mapper("Frame must contain 'atoms' block"))?;

    let meta = frame.meta_ref();

    // -- Timestep --
    let timestep = meta
        .get("timestep")
        .and_then(|value| value.as_i64())
        .unwrap_or(0);
    writeln!(writer, "ITEM: TIMESTEP")?;
    writeln!(writer, "{}", timestep)?;

    // -- Number of atoms --
    writeln!(writer, "ITEM: NUMBER OF ATOMS")?;
    writeln!(writer, "{}", natoms)?;
    write_dump_box_bounds(writer, frame)?;

    // -- Atoms --
    // Determine column ordering and write per-row data via visit_block
    let atom_lines: Vec<String> = frame
        .visit_block("atoms", |atoms| -> std::io::Result<Vec<String>> {
            let col_names = atoms.column_keys();
            let ordered: Vec<String> = match columns {
                Some(chosen) => select_dump_columns(chosen, &col_names)?,
                None => {
                    let mut ordered: Vec<&str> = Vec::with_capacity(col_names.len());

                    if col_names.contains(&"id") {
                        ordered.push("id");
                    }
                    if col_names.contains(&"type") {
                        ordered.push("type");
                    }

                    let mut remaining: Vec<&str> = col_names
                        .iter()
                        .filter(|&&n| n != "id" && n != "type")
                        .copied()
                        .collect();
                    remaining.sort();
                    ordered.extend(remaining);
                    ordered.into_iter().map(str::to_string).collect()
                }
            };

            // `ordered` holds canonical keys, used to look values up in the
            // block. The header and the type heuristic both speak LAMMPS's
            // native names, so translate on the way out.
            let native: Vec<&str> = ordered.iter().map(|n| native_column_name(n)).collect();
            let header = format!("ITEM: ATOMS {}", native.join(" "));
            let col_types: Vec<ColumnType> = native.iter().map(|n| classify_column(n)).collect();

            let mut lines = Vec::with_capacity(natoms + 1);
            lines.push(header);

            for row in 0..natoms {
                let mut parts = Vec::with_capacity(ordered.len());
                for (ci, name) in ordered.iter().map(String::as_str).enumerate() {
                    let s = match col_types[ci] {
                        ColumnType::Unsigned => {
                            if let Some(arr) = atoms.get_uint_view(name) {
                                format!("{}", arr[row])
                            } else {
                                String::new()
                            }
                        }
                        ColumnType::Integer => {
                            if let Some(arr) = atoms.get_int_view(name) {
                                format!("{}", arr[row])
                            } else if let Some(arr) = atoms.get_float_view(name) {
                                format!("{}", arr[row] as I)
                            } else {
                                "0".to_string()
                            }
                        }
                        ColumnType::Float => {
                            if let Some(arr) = atoms.get_float_view(name) {
                                format!("{:.6}", arr[row])
                            } else if let Some(arr) = atoms.get_int_view(name) {
                                format!("{:.6}", arr[row] as F)
                            } else {
                                "0.000000".to_string()
                            }
                        }
                        ColumnType::String => {
                            if let Some(arr) = atoms.get_string_view(name) {
                                arr[row].clone()
                            } else {
                                "X".to_string()
                            }
                        }
                    };
                    parts.push(s);
                }
                lines.push(parts.join(" "));
            }
            Ok(lines)
        })
        .transpose()?
        .unwrap_or_default();

    for line in &atom_lines {
        writeln!(writer, "{}", line)?;
    }

    Ok(())
}

/// Resolve a caller's `dump custom` column list against the `atoms` block.
///
/// Names may be native (`mol`, `q`, `type`) or canonical (`mol_id`, `charge`,
/// `type_id`); the returned keys are canonical, in the order asked for. A name
/// the block cannot supply is an error: a dump silently missing the column the
/// caller named is found wrong later, by whatever reads it.
fn select_dump_columns(chosen: &[&str], col_names: &[&str]) -> std::io::Result<Vec<String>> {
    if chosen.is_empty() {
        return Err(err_mapper("dump column list must name at least one column"));
    }
    chosen
        .iter()
        .map(|name| {
            let key = canonical_column_name(name);
            if col_names.contains(&key.as_str()) {
                Ok(key)
            } else {
                let have: Vec<&str> = col_names.iter().map(|n| native_column_name(n)).collect();
                Err(err_mapper(format!(
                    "dump column '{}' is not in the 'atoms' block (have: {})",
                    name,
                    have.join(" ")
                )))
            }
        })
        .collect()
}

/// Write a single frame as LAMMPS `dump local` (OVITO Load Trajectory bonds).
///
/// `ITEM: NUMBER OF ENTRIES` + `ITEM: ENTRIES batom1 batom2 [btype]`.
/// Rows come from the `entries` block if present, otherwise from canonical
/// `bonds` (`atomi`/`atomj` 0-based, emitted as 1-based atom ids).
///
/// See <https://www.ovito.org/manual/reference/file_formats/input/lammps_dump_local.html>
fn write_lammps_dump_local_frame<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
) -> std::io::Result<()> {
    crate::io::writer::check_before_write(frame)?;

    let timestep = frame
        .meta_ref()
        .get("timestep")
        .and_then(|value| value.as_i64())
        .unwrap_or(0);
    writeln!(writer, "ITEM: TIMESTEP")?;
    writeln!(writer, "{}", timestep)?;

    let from_entries = frame.contains_block("entries");
    let nentries = if from_entries {
        frame
            .visit_block("entries", |b| b.nrows().unwrap_or(0))
            .unwrap_or(0)
    } else {
        frame
            .visit_block("bonds", |b| b.nrows().unwrap_or(0))
            .unwrap_or(0)
    };
    if !from_entries && nentries == 0 && !frame.contains_block("bonds") {
        return Err(err_mapper("dump local needs a 'bonds' or 'entries' block"));
    }

    writeln!(writer, "ITEM: NUMBER OF ENTRIES")?;
    writeln!(writer, "{}", nentries)?;
    write_dump_box_bounds(writer, frame)?;

    if from_entries {
        let lines: Vec<String> = frame
            .visit_block("entries", |entries| {
                dump_block_lines(entries, nentries, "ITEM: ENTRIES")
            })
            .unwrap_or_default();
        for line in &lines {
            writeln!(writer, "{}", line)?;
        }
        return Ok(());
    }

    let atomi = frame
        .get_uint("bonds", "atomi")
        .ok_or_else(|| err_mapper("bonds block missing atomi"))?;
    let atomj = frame
        .get_uint("bonds", "atomj")
        .ok_or_else(|| err_mapper("bonds block missing atomj"))?;
    let atomi = atomi
        .as_slice()
        .ok_or_else(|| err_mapper("bonds.atomi is not contiguous"))?;
    let atomj = atomj
        .as_slice()
        .ok_or_else(|| err_mapper("bonds.atomj is not contiguous"))?;
    let btype = frame
        .get_uint("bonds", "type_id")
        .and_then(|a| a.as_slice().map(|s| s.to_vec()));
    let atom_ids = frame
        .get_uint("atoms", "id")
        .and_then(|a| a.as_slice().map(|s| s.to_vec()));

    let id_of = |idx: Idx| -> Idx {
        let i = idx as usize;
        if let Some(ref ids) = atom_ids {
            ids.get(i).copied().unwrap_or(idx)
        } else {
            idx + 1
        }
    };

    if btype.is_some() {
        writeln!(writer, "ITEM: ENTRIES batom1 batom2 btype")?;
    } else {
        writeln!(writer, "ITEM: ENTRIES batom1 batom2")?;
    }
    for row in 0..nentries {
        let a = id_of(atomi[row]);
        let b = id_of(atomj[row]);
        if let Some(ref t) = btype {
            writeln!(writer, "{} {} {}", a, b, t[row])?;
        } else {
            writeln!(writer, "{} {}", a, b)?;
        }
    }
    Ok(())
}

fn dump_block_lines(
    block: &dyn crate::store::block::access::BlockAccess,
    nrows: usize,
    header_prefix: &str,
) -> Vec<String> {
    let col_names = block.column_keys();
    let mut ordered: Vec<&str> = col_names.to_vec();
    ordered.sort();
    let native: Vec<&str> = ordered.iter().map(|n| native_column_name(n)).collect();
    let header = format!("{} {}", header_prefix, native.join(" "));
    let col_types: Vec<ColumnType> = native.iter().map(|n| classify_column(n)).collect();
    let mut lines = Vec::with_capacity(nrows + 1);
    lines.push(header);
    for row in 0..nrows {
        let mut parts = Vec::with_capacity(ordered.len());
        for (ci, &name) in ordered.iter().enumerate() {
            let s = match col_types[ci] {
                ColumnType::Unsigned => block
                    .get_uint_view(name)
                    .map(|arr| format!("{}", arr[row]))
                    .unwrap_or_default(),
                ColumnType::Integer => {
                    if let Some(arr) = block.get_int_view(name) {
                        format!("{}", arr[row])
                    } else if let Some(arr) = block.get_float_view(name) {
                        format!("{}", arr[row] as I)
                    } else {
                        "0".to_string()
                    }
                }
                ColumnType::Float => {
                    if let Some(arr) = block.get_float_view(name) {
                        format!("{:.6}", arr[row])
                    } else if let Some(arr) = block.get_int_view(name) {
                        format!("{:.6}", arr[row] as F)
                    } else {
                        "0.000000".to_string()
                    }
                }
                ColumnType::String => block
                    .get_string_view(name)
                    .map(|arr| arr[row].clone())
                    .unwrap_or_else(|| "X".to_string()),
            };
            parts.push(s);
        }
        lines.push(parts.join(" "));
    }
    lines
}

fn write_dump_box_bounds<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
) -> std::io::Result<()> {
    let simbox = frame
        .simbox_ref()
        .ok_or_else(|| err_mapper("Frame must have a simbox"))?;

    let h = simbox.h_view();
    let o = simbox.origin_view();
    let pbc_flags = simbox.pbc();

    let lx = h[[0, 0]];
    let ly = h[[1, 1]];
    let lz = h[[2, 2]];
    let xy = h[[0, 1]];
    let xz = h[[0, 2]];
    let yz = h[[1, 2]];
    let xlo = o[0];
    let ylo = o[1];
    let zlo = o[2];
    let xhi = xlo + lx;
    let yhi = ylo + ly;
    let zhi = zlo + lz;

    let pbc_str = format!(
        "{} {} {}",
        if pbc_flags[0] { "pp" } else { "ff" },
        if pbc_flags[1] { "pp" } else { "ff" },
        if pbc_flags[2] { "pp" } else { "ff" },
    );

    let is_triclinic = xy != 0.0 || xz != 0.0 || yz != 0.0;
    if is_triclinic {
        let xlo_bound = xlo + f64::min(0.0, f64::min(xy, f64::min(xz, xy + xz)));
        let xhi_bound = xhi + f64::max(0.0, f64::max(xy, f64::max(xz, xy + xz)));
        let ylo_bound = ylo + f64::min(0.0, yz);
        let yhi_bound = yhi + f64::max(0.0, yz);

        writeln!(writer, "ITEM: BOX BOUNDS xy xz yz {}", pbc_str)?;
        writeln!(writer, "{} {} {}", xlo_bound, xhi_bound, xy)?;
        writeln!(writer, "{} {} {}", ylo_bound, yhi_bound, xz)?;
        writeln!(writer, "{} {} {}", zlo, zhi, yz)?;
    } else {
        writeln!(writer, "ITEM: BOX BOUNDS {}", pbc_str)?;
        writeln!(writer, "{} {}", xlo, xhi)?;
        writeln!(writer, "{} {}", ylo, yhi)?;
        writeln!(writer, "{} {}", zlo, zhi)?;
    }
    Ok(())
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Read all frames from a LAMMPS dump file.
///
/// For large trajectories, prefer `open_lammps_dump` with `TrajectoryReader::read_step`
/// for random access without loading all frames into memory.
pub fn read_lammps_dump<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Frame>> {
    let reader = crate::io::reader::open_seekable(path)?;
    let mut dump_reader = LAMMPSTrajReader::new(reader);
    crate::io::reader::collect_frames(&mut dump_reader)
}

/// Open a LAMMPS dump file for trajectory-style random access.
///
/// Returns a reader implementing `TrajectoryReader`. The index is built lazily
/// on first call to `read_step` or `len`.
pub fn open_lammps_dump<P: AsRef<Path>>(
    path: P,
) -> std::io::Result<LAMMPSTrajReader<Box<dyn ReadSeek>>> {
    let reader = crate::io::reader::open_seekable(path)?;
    Ok(LAMMPSTrajReader::new(reader))
}

/// Write frames to a LAMMPS dump file.
///
/// Accepts a slice of any type implementing [`FrameAccess`], including
/// `&[Frame]`.
///
/// `columns` is the `dump custom` column line: `Some(&["id", "element", "x",
/// "y", "z"])` writes exactly those, in that order, and errors on a name the
/// frame cannot supply; `None` writes every column the `atoms` block holds
/// (`id`, `type`, then the rest sorted).
pub fn write_lammps_dump<P: AsRef<Path>, FA: FrameAccess>(
    path: P,
    frames: &[FA],
    columns: Option<&[&str]>,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for frame in frames {
        write_lammps_dump_frame(&mut writer, frame, columns)?;
    }
    Ok(())
}

/// Write frames as LAMMPS `dump local` (OVITO Load Trajectory bond overlay).
///
/// Column names `batom1` / `batom2` / `btype` match OVITO's automatic mapping.
/// See <https://www.ovito.org/manual/reference/pipelines/modifiers/load_trajectory.html>
pub fn write_lammps_dump_local<P: AsRef<Path>, FA: FrameAccess>(
    path: P,
    frames: &[FA],
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for frame in frames {
        write_lammps_dump_local_frame(&mut writer, frame)?;
    }
    Ok(())
}

// ============================================================================
// Streaming
// ============================================================================

use crate::io::streaming::{FrameIndexBuilder, FrameIndexEntry, LineAccumulator};
use std::io::Cursor;

/// Parse exactly one LAMMPS dump frame from a tightly-bounded byte slice.
///
/// `bytes` must be the slice produced by [`LammpsDumpIndexBuilder`] for one
/// frame: it begins with an `ITEM: TIMESTEP` (or an optional `ITEM: UNITS` /
/// `ITEM: TIME` header preceding it) and ends just before the next frame's
/// `ITEM: TIMESTEP` or at EOF. The frame must be self-contained.
pub fn parse_frame_bytes(bytes: &[u8]) -> std::io::Result<Frame> {
    let mut cursor = Cursor::new(bytes);
    parse_single_frame(&mut cursor)?.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "LAMMPS dump frame slice is empty",
        )
    })
}

/// Streaming frame indexer for LAMMPS dump files.
///
/// Detects frame boundaries by scanning for `ITEM: TIMESTEP` lines. The
/// builder tolerates chunk boundaries that split lines (LF or CRLF) and
/// frames that span multiple chunks.
pub struct LammpsDumpIndexBuilder {
    lines: LineAccumulator,
    /// Offset of the most-recent unfinalized frame's first byte, if any.
    pending_frame_start: Option<u64>,
    /// Frames finalized (i.e. their successor's `ITEM: TIMESTEP` has been
    /// observed) but not yet drained.
    pending_entries: Vec<FrameIndexEntry>,
}

impl Default for LammpsDumpIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LammpsDumpIndexBuilder {
    pub fn new() -> Self {
        Self {
            lines: LineAccumulator::new(),
            pending_frame_start: None,
            pending_entries: Vec::new(),
        }
    }
}

impl FrameIndexBuilder for LammpsDumpIndexBuilder {
    fn feed(&mut self, chunk: &[u8], global_offset: u64) {
        let pending_frame_start = &mut self.pending_frame_start;
        let pending_entries = &mut self.pending_entries;
        self.lines
            .feed(chunk, global_offset, |line, line_offset, _line_len| {
                if !line.trim_start().starts_with("ITEM: TIMESTEP") {
                    return;
                }
                if let Some(prev) = pending_frame_start.replace(line_offset) {
                    let len = (line_offset - prev) as u32;
                    pending_entries.push(FrameIndexEntry {
                        byte_offset: prev,
                        byte_len: len,
                    });
                }
            });
    }

    fn drain(&mut self) -> Vec<FrameIndexEntry> {
        std::mem::take(&mut self.pending_entries)
    }

    fn finish(mut self: Box<Self>) -> std::io::Result<Vec<FrameIndexEntry>> {
        self.lines.check_line_budget()?;
        let pending_frame_start = &mut self.pending_frame_start;
        let pending_entries = &mut self.pending_entries;
        self.lines.finish(|line, line_offset, _len| {
            if !line.trim_start().starts_with("ITEM: TIMESTEP") {
                return;
            }
            if let Some(prev) = pending_frame_start.replace(line_offset) {
                let len = (line_offset - prev) as u32;
                pending_entries.push(FrameIndexEntry {
                    byte_offset: prev,
                    byte_len: len,
                });
            }
        });
        let bytes_seen = self.lines.bytes_seen();
        if let Some(prev) = self.pending_frame_start.take() {
            let span = bytes_seen.saturating_sub(prev);
            if span > u32::MAX as u64 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "LAMMPS dump frame size exceeds 4 GiB",
                ));
            }
            self.pending_entries.push(FrameIndexEntry {
                byte_offset: prev,
                byte_len: span as u32,
            });
        }
        Ok(std::mem::take(&mut self.pending_entries))
    }

    fn bytes_seen(&self) -> u64 {
        self.lines.bytes_seen()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::store::keys;
    use std::io::Cursor;

    /// Multi-frame dump (2 frames) — used only by index/random-access/iter tests.
    const MULTI_DUMP: &str = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.5 2.5 3.5
2 1 4.5 5.5 6.5
";

    fn cursor(s: &str) -> Cursor<Vec<u8>> {
        Cursor::new(s.as_bytes().to_vec())
    }

    #[test]
    fn test_classify_column() {
        // `id` is UInt in the vocabulary, so classification follows the schema
        // rather than the hardcoded integer-name list.
        assert_eq!(classify_column("id"), ColumnType::Unsigned);
        // A dump's `type` column canonicalizes to `type_id` (UInt).
        assert_eq!(classify_column("type"), ColumnType::Unsigned);
        // `mol` canonicalizes to `mol_id` (UInt).
        assert_eq!(classify_column("mol"), ColumnType::Unsigned);
        assert_eq!(classify_column("ix"), ColumnType::Integer);
        assert_eq!(classify_column("x"), ColumnType::Float);
        assert_eq!(classify_column("vx"), ColumnType::Float);
        assert_eq!(classify_column("q"), ColumnType::Float);
        assert_eq!(classify_column("c_pe"), ColumnType::Float);
        assert_eq!(classify_column("f_reax[1]"), ColumnType::Float);
    }

    #[test]
    fn test_build_index() {
        let mut reader = LAMMPSTrajReader::new(cursor(MULTI_DUMP));
        reader.build_index().unwrap();

        assert_eq!(reader.len().unwrap(), 2);
    }

    #[test]
    fn test_read_step_random_access() {
        let mut reader = LAMMPSTrajReader::new(cursor(MULTI_DUMP));

        // Read step 1 first (out of order)
        let f1 = reader.read_step(1).unwrap().expect("step 1");
        assert_eq!(f1.meta.get("timestep").unwrap().as_i64(), Some(100));

        // Then step 0
        let f0 = reader.read_step(0).unwrap().expect("step 0");
        assert_eq!(f0.meta.get("timestep").unwrap().as_i64(), Some(0));

        // Out of bounds
        assert!(reader.read_step(5).unwrap().is_none());
    }

    /// Per-bond `dump local` (OVITO-compatible). The header keywords
    /// `NUMBER OF ENTRIES` and `ENTRIES` substitute for the per-atom
    /// flavor's `NUMBER OF ATOMS` / `ATOMS`. The parsed columns land
    /// in a `bonds` block (vs `atoms` for the per-atom flavor) so a
    /// downstream pipeline can route the two through different
    /// renderers without inspecting the dump variant directly.
    #[test]
    fn test_dump_local_entries_form() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ENTRIES
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ENTRIES c_1[1] c_1[2] c_1[3]
1 1 2
2 2 3
3 3 4
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frames = crate::io::reader::collect_frames(&mut reader).unwrap();
        assert_eq!(frames.len(), 1);
        // ENTRIES lands in "entries" — it does not satisfy the canonical
        // `bonds` contract (0-based `atomi`/`atomj`), so it does not take that name.
        assert!(frames[0].get("atoms").is_none());
        let entries = frames[0].get("entries").expect("entries block present");
        assert_eq!(entries.nrows(), Some(3));
        // Column names preserved as-is from the file.
        assert!(entries.dtype("c_1[1]").is_some());
        assert!(entries.dtype("c_1[2]").is_some());
        assert!(entries.dtype("c_1[3]").is_some());
    }

    /// `dump_modify … label BONDS` is what OVITO's LAMMPS-dump-local manual
    /// tells users to set, so files in the wild carry it at least as often as
    /// the LAMMPS default `ENTRIES`. Accepting only `ENTRIES` rejected exactly
    /// the setup that manual recommends.
    #[test]
    fn test_dump_local_accepts_dump_modify_labels() {
        for label in [
            "ENTRIES",
            "BONDS",
            "ANGLES",
            "DIHEDRALS",
            "IMPROPERS",
            "NEIGHBORS",
        ] {
            let dump = format!(
                "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF {label}
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: {label} c_1[1] c_1[2]
1 2
2 3
"
            );
            let mut reader = LAMMPSTrajReader::new(cursor(&dump));
            let frames = crate::io::reader::collect_frames(&mut reader)
                .unwrap_or_else(|e| panic!("label {label} rejected: {e}"));
            assert_eq!(frames.len(), 1, "label {label}");
            // Every label lands in `entries`: the label says what the rows
            // mean, it does not make them satisfy a contract-bearing schema.
            let entries = frames[0]
                .get("entries")
                .unwrap_or_else(|| panic!("label {label} produced no entries block"));
            assert_eq!(entries.nrows(), Some(2), "label {label}");
            assert_eq!(
                frames[0]
                    .meta
                    .get("dump_local_label")
                    .and_then(|v| v.as_str()),
                Some(label),
                "label {label} not recorded in meta"
            );
        }
    }

    /// The label is the only signal that survives default column naming, so
    /// it must be readable even when the columns say nothing.
    #[test]
    fn test_dump_local_label_absent_for_per_atom_dump() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id x y z
1 1.0 2.0 3.0
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frames = crate::io::reader::collect_frames(&mut reader).unwrap();
        assert!(frames[0].meta.get("dump_local_label").is_none());
    }

    /// A mismatched pair — counted as BONDS, then a data header that says
    /// ENTRIES — is a malformed file, not something to paper over.
    #[test]
    fn test_dump_local_label_must_match_between_header_lines() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF BONDS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ENTRIES batom1 batom2
1 2
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let err = crate::io::reader::collect_frames(&mut reader)
            .expect_err("mismatched labels must not parse");
        assert!(
            err.to_string().contains("ITEM: BONDS"),
            "error should name the expected header, got: {err}"
        );
    }

    /// An unknown label is rejected with the accepted set named, so the
    /// message tells the user which `dump_modify label` values work.
    #[test]
    fn test_dump_local_unknown_label_rejected() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF WIDGETS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: WIDGETS a b
1 2
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let err = crate::io::reader::collect_frames(&mut reader)
            .expect_err("unknown label must not parse");
        let text = err.to_string();
        assert!(text.contains("BONDS"), "message should list labels: {text}");
        assert!(
            text.contains("ENTRIES"),
            "message should list labels: {text}"
        );
    }

    #[test]
    fn test_atoms_keep_file_order() {
        // Atoms are NOT sorted by `id` on read. LAMMPS' companion
        // `dump local` outputs (e.g. bonds.dump from `compute property/
        // local`) typically reference atoms by the same per-row position
        // they occupy in this dump, not by atom id. Re-sorting on read
        // would break those bond mappings on every frame switch. Users
        // who want a canonical row order should add `dump_modify sort
        // id` on the LAMMPS side; the reader trusts whatever ordering
        // the writer chose.
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
3 1 9.0 0.0 0.0
1 1 1.0 0.0 0.0
2 1 5.0 0.0 0.0
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frames = crate::io::reader::collect_frames(&mut reader).unwrap();
        let atoms = frames[0].get("atoms").expect("atoms block");
        let ids = atoms.get_uint("id").expect("id column");
        let xs = atoms.get_float("x").expect("x column");
        // File order preserved: 3, 1, 2 (matching x: 9.0, 1.0, 5.0).
        assert_eq!(ids.as_slice().unwrap(), &[3, 1, 2]);
        assert_eq!(xs.as_slice().unwrap(), &[9.0, 1.0, 5.0]);
    }

    #[test]
    fn test_entries_keep_file_order() {
        // ENTRIES blocks (dump local) have no `id` column — bonds are
        // identified by their endpoint atom IDs, not by row position.
        // File order is preserved.
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ENTRIES
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ENTRIES batom1 batom2 btype
3 4 1
1 2 1
2 3 1
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frames = crate::io::reader::collect_frames(&mut reader).unwrap();
        let entries = frames[0].get("entries").expect("entries block");
        let batom1 = entries.get_int("batom1").expect("batom1");
        // File order: 3, 1, 2 (no sort applied).
        assert_eq!(batom1.as_slice().unwrap(), &[3, 1, 2]);
    }

    #[test]
    fn test_variable_atom_count() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
3 2 7.0 8.0 9.0
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frames = crate::io::reader::collect_frames(&mut reader).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].get("atoms").unwrap().nrows(), Some(2));
        assert_eq!(frames[1].get("atoms").unwrap().nrows(), Some(3));
    }

    #[test]
    fn test_custom_columns() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z vx vy vz q c_pe
1 1 1.0 2.0 3.0 0.1 0.2 0.3 -0.5 -10.5
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frame = reader.read().unwrap().expect("parse");
        let atoms = frame.get("atoms").unwrap();

        // Custom columns should be float. LAMMPS's `q` is renamed to the
        // canonical `charge` on the way out of the reader.
        let q = atoms.get_float(keys::CHARGE).expect("charge column");
        assert!((q[0] - (-0.5)).abs() < 1e-6);

        let pe = atoms.get_float("c_pe").expect("c_pe column");
        assert!((pe[0] - (-10.5)).abs() < 1e-4);

        // Velocities should be float
        let vx = atoms.get_float("vx").expect("vx column");
        assert!((vx[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_preserves_unwrapped_coords_without_synthesizing_xyz() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type xu yu zu
1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frame = reader.read().unwrap().expect("parse");
        let atoms = frame.get("atoms").expect("atoms");

        let x = atoms.get_float("xu").expect("xu");
        let y = atoms.get_float("yu").expect("yu");
        let z = atoms.get_float("zu").expect("zu");

        assert_eq!(x.iter().copied().collect::<Vec<_>>(), vec![1.0, 4.0]);
        assert_eq!(y.iter().copied().collect::<Vec<_>>(), vec![2.0, 5.0]);
        assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![3.0, 6.0]);
        assert!(
            atoms.get_float("x").is_none(),
            "reader should not synthesize x/y/z from xu/yu/zu"
        );
    }

    #[test]
    fn test_preserves_scaled_coords_without_synthesizing_xyz() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
1.0 11.0
2.0 22.0
3.0 43.0
ITEM: ATOMS id type xs ys zs
1 1 0.0 0.0 0.0
2 1 0.5 0.5 0.5
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frame = reader.read().unwrap().expect("parse");
        let atoms = frame.get("atoms").expect("atoms");

        let x = atoms.get_float("xs").expect("xs");
        let y = atoms.get_float("ys").expect("ys");
        let z = atoms.get_float("zs").expect("zs");

        assert_eq!(x.iter().copied().collect::<Vec<_>>(), vec![0.0, 0.5]);
        assert_eq!(y.iter().copied().collect::<Vec<_>>(), vec![0.0, 0.5]);
        assert_eq!(z.iter().copied().collect::<Vec<_>>(), vec![0.0, 0.5]);
        assert!(
            atoms.get_float("x").is_none(),
            "reader should preserve source columns only"
        );
    }

    #[test]
    fn test_preserves_triclinic_scaled_coords_as_read() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0 14.0 1.5
0.0 23.5 2.5
0.0 30.0 3.5
ITEM: ATOMS id type xs ys zs
1 1 0.25 0.5 0.75
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frame = reader.read().unwrap().expect("parse");
        let atoms = frame.get("atoms").expect("atoms");

        let x = atoms.get_float("xs").expect("xs");
        let y = atoms.get_float("ys").expect("ys");
        let z = atoms.get_float("zs").expect("zs");

        assert!((x[0] - 0.25).abs() < 1e-6);
        assert!((y[0] - 0.5).abs() < 1e-6);
        assert!((z[0] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_preserves_mixed_scaled_and_real_coords() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type xs yu zu
1 1 0.5 5.0 5.0
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frame = reader.read().unwrap().expect("mixed coords parse");
        let atoms = frame.get("atoms").expect("atoms");
        assert_eq!(atoms.get_float("xs").expect("xs")[0], 0.5);
        assert_eq!(atoms.get_float("yu").expect("yu")[0], 5.0);
        assert_eq!(atoms.get_float("zu").expect("zu")[0], 5.0);
    }

    #[test]
    fn test_allows_frames_without_coordinate_columns() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type q
1 1 -0.5
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frame = reader.read().unwrap().expect("parse");
        let atoms = frame.get("atoms").expect("atoms");
        assert_eq!(atoms.get_float(keys::CHARGE).expect("charge")[0], -0.5);
    }

    #[test]
    fn lammps_native_columns_are_canonicalized_on_read() {
        // `q` and `mol` are LAMMPS-native spellings; a frame must expose only
        // the canonical `charge` / `mol_id`, per `store::keys`.
        let dump = "ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type mol q x y z
1 1 7 -0.5 1.0 2.0 3.0
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frame = reader.read().unwrap().expect("parse");
        let atoms = frame.get("atoms").expect("atoms");

        assert_eq!(atoms.get_float(keys::CHARGE).expect("charge")[0], -0.5);
        assert_eq!(atoms.get_uint(keys::MOL_ID).expect("mol_id")[0], 7);
        assert!(
            atoms.get("q").is_none(),
            "raw `q` must not survive the reader"
        );
        assert!(
            atoms.get("mol").is_none(),
            "raw `mol` must not survive the reader"
        );
    }

    #[test]
    fn canonical_columns_round_trip_to_lammps_native_names() {
        assert_eq!(canonical_column_name("q"), keys::CHARGE);
        assert_eq!(canonical_column_name("mol"), keys::MOL_ID);
        assert_eq!(canonical_column_name("vx"), keys::VX);
        assert_eq!(native_column_name(keys::CHARGE), "q");
        assert_eq!(native_column_name(keys::MOL_ID), "mol");
        assert_eq!(native_column_name(keys::VX), keys::VX);
    }

    #[test]
    fn test_empty_input() {
        let mut reader = LAMMPSTrajReader::new(cursor(""));
        assert!(reader.read().unwrap().is_none());
    }

    #[test]
    fn test_empty_index() {
        let mut reader = LAMMPSTrajReader::new(cursor(""));
        reader.build_index().unwrap();
        assert_eq!(reader.len().unwrap(), 0);
    }

    #[test]
    fn test_boundary_flags() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS ff pp ss
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
";
        let mut reader = LAMMPSTrajReader::new(cursor(dump));
        let frame = reader.read().unwrap().expect("parse");
        let pbc = frame.simbox.as_ref().expect("simbox").pbc();
        // ff pp ss → [false, true, false]
        assert_eq!(pbc, [false, true, false]);
    }

    #[test]
    fn test_iter() {
        let mut reader = LAMMPSTrajReader::new(cursor(MULTI_DUMP));
        reader.build_index().unwrap();
        let mut count = 0;
        for result in reader.iter() {
            result.unwrap();
            count += 1;
        }
        assert_eq!(count, 2);
    }

    // -----------------------------------------------------------------
    // Streaming index tests
    // -----------------------------------------------------------------

    fn build_index_in_chunks(bytes: &[u8], chunk_size: usize) -> Vec<FrameIndexEntry> {
        let mut builder = Box::new(LammpsDumpIndexBuilder::new());
        let mut offset: u64 = 0;
        let mut out: Vec<FrameIndexEntry> = Vec::new();
        for piece in bytes.chunks(chunk_size.max(1)) {
            builder.feed(piece, offset);
            offset += piece.len() as u64;
            out.extend(builder.drain());
        }
        out.extend(builder.finish().expect("finish"));
        out
    }

    #[test]
    fn streaming_single_shot_matches_legacy() {
        let bytes = MULTI_DUMP.as_bytes();
        let entries = build_index_in_chunks(bytes, bytes.len());
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].byte_offset, 0);
        // Reconstruct a frame from each entry slice.
        for entry in &entries {
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            let frame = parse_frame_bytes(&bytes[lo..hi]).expect("parse_frame_bytes");
            assert!(frame.get("atoms").is_some());
        }
    }

    #[test]
    fn streaming_chunked_indices_are_identical() {
        let bytes = MULTI_DUMP.as_bytes();
        let one_shot = build_index_in_chunks(bytes, bytes.len());
        for cs in [1usize, 7, 13, 31, 64, 1024] {
            let chunked = build_index_in_chunks(bytes, cs);
            assert_eq!(
                one_shot, chunked,
                "chunk size {} produced different index",
                cs
            );
        }
    }

    /// Edge case: chunk boundary lands inside the literal "ITEM: TIMESTEP".
    #[test]
    fn streaming_boundary_inside_timestep_literal() {
        let bytes = MULTI_DUMP.as_bytes();
        // Find first "ITEM: TIMESTEP" position in second frame.
        let second = bytes
            .windows(b"ITEM: TIMESTEP".len())
            .position(|w| w == b"ITEM: TIMESTEP")
            .and_then(|first| {
                bytes[first + 1..]
                    .windows(b"ITEM: TIMESTEP".len())
                    .position(|w| w == b"ITEM: TIMESTEP")
                    .map(|p| first + 1 + p)
            })
            .expect("two TIMESTEP markers");
        // Split ~7 bytes into the literal.
        let split = second + 7;
        let mut builder = Box::new(LammpsDumpIndexBuilder::new());
        builder.feed(&bytes[..split], 0);
        builder.feed(&bytes[split..], split as u64);
        let mut entries = builder.drain();
        entries.extend(builder.finish().expect("finish"));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].byte_offset as usize, second);
    }

    /// Edge case: file with `ITEM: UNITS` preceding `ITEM: TIMESTEP`. The
    /// indexer must NOT treat `ITEM: UNITS` as a frame boundary; only the
    /// `ITEM: TIMESTEP` line is the boundary marker. The leading `ITEM: UNITS`
    /// prefix is part of the first frame's body.
    #[test]
    fn streaming_handles_units_header_before_timestep() {
        let dump = "\
ITEM: UNITS
metal
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
";
        let bytes = dump.as_bytes();
        let entries = build_index_in_chunks(bytes, bytes.len());
        // The frame starts at the `ITEM: TIMESTEP` line, NOT byte 0.
        assert_eq!(entries.len(), 1);
        let lo = entries[0].byte_offset as usize;
        assert!(bytes[lo..].starts_with(b"ITEM: TIMESTEP"));
        // The slice rooted at byte_offset must be parseable.
        parse_frame_bytes(&bytes[lo..lo + entries[0].byte_len as usize])
            .expect("parse the units-prefixed frame");
    }

    /// Edge case: CRLF line endings.
    #[test]
    fn streaming_handles_crlf_line_endings() {
        let dump = MULTI_DUMP.replace('\n', "\r\n");
        let bytes = dump.as_bytes();
        let entries = build_index_in_chunks(bytes, bytes.len());
        assert_eq!(entries.len(), 2);
        for entry in &entries {
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            parse_frame_bytes(&bytes[lo..hi]).expect("parse CRLF frame");
        }
    }

    /// Edge case: missing trailing newline on the final atom line.
    #[test]
    fn streaming_handles_missing_trailing_newline() {
        // Build a single-frame dump without a trailing newline.
        let dump = "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0.0 10.0\n0.0 10.0\n0.0 10.0\nITEM: ATOMS id type x y z\n1 1 1.0 2.0 3.0";
        let bytes = dump.as_bytes();
        let entries = build_index_in_chunks(bytes, bytes.len());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].byte_offset, 0);
        assert_eq!(entries[0].byte_len as usize, bytes.len());
        parse_frame_bytes(bytes).expect("parse no-trailing-newline");
    }

    /// Edge case: chunk boundary at byte 0 of a TIMESTEP line.
    #[test]
    fn streaming_boundary_at_timestep_start_byte() {
        let bytes = MULTI_DUMP.as_bytes();
        let second = bytes
            .windows(b"ITEM: TIMESTEP".len())
            .position(|w| w == b"ITEM: TIMESTEP")
            .and_then(|first| {
                bytes[first + 1..]
                    .windows(b"ITEM: TIMESTEP".len())
                    .position(|w| w == b"ITEM: TIMESTEP")
                    .map(|p| first + 1 + p)
            })
            .expect("two TIMESTEP markers");
        let mut builder = Box::new(LammpsDumpIndexBuilder::new());
        builder.feed(&bytes[..second], 0);
        builder.feed(&bytes[second..], second as u64);
        let mut entries = builder.drain();
        entries.extend(builder.finish().expect("finish"));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].byte_offset, 0);
        assert_eq!(entries[1].byte_offset as usize, second);
    }

    #[test]
    fn write_dump_local_from_bonds_roundtrip() {
        use molrs::spatial::simbox::SimBox;
        use ndarray::{Array1, array};

        let mut atoms = Block::new();
        atoms
            .insert(
                "id",
                Array1::from_vec(vec![1 as Idx, 2 as Idx, 3 as Idx]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert("x", Array1::from_vec(vec![0.0 as F, 1.0, 2.0]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![0.0 as F; 3]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![0.0 as F; 3]).into_dyn())
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert(
                "atomi",
                Array1::from_vec(vec![0 as Idx, 1 as Idx]).into_dyn(),
            )
            .unwrap();
        bonds
            .insert(
                "atomj",
                Array1::from_vec(vec![1 as Idx, 2 as Idx]).into_dyn(),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);
        frame.simbox =
            Some(SimBox::cube(10.0, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bonds.dump");
        write_lammps_dump_local(&path, &[frame]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("ITEM: NUMBER OF ENTRIES"));
        assert!(text.contains("ITEM: ENTRIES batom1 batom2"));
        assert!(text.contains("1 2"));
        assert!(text.contains("2 3"));
        let loaded = read_lammps_dump(&path).unwrap();
        let entries = loaded[0].get("entries").expect("entries");
        assert_eq!(entries.nrows(), Some(2));
        assert!(entries.dtype("batom1").is_some());
        assert!(entries.dtype("batom2").is_some());
    }

    /// A frame carrying more than a viewer needs, for the column-choice tests.
    fn wide_frame() -> Frame {
        use molrs::spatial::simbox::SimBox;
        use ndarray::{Array1, array};

        let mut atoms = Block::new();
        atoms
            .insert("id", Array1::from_vec(vec![1 as Idx, 2]).into_dyn())
            .unwrap();
        atoms
            .insert("mol_id", Array1::from_vec(vec![1 as Idx, 1]).into_dyn())
            .unwrap();
        atoms
            .insert("mass", Array1::from_vec(vec![16.0 as F, 1.008]).into_dyn())
            .unwrap();
        insert_str(&mut atoms, "element", vec!["O".into(), "H".into()], 2).unwrap();
        for (key, values) in [("x", [0.0 as F, 1.0]), ("y", [0.0, 2.0]), ("z", [0.0, 3.0])] {
            atoms
                .insert(key, Array1::from_vec(values.to_vec()).into_dyn())
                .unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.simbox =
            Some(SimBox::cube(10.0, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap());
        frame
    }

    #[test]
    fn write_dump_columns_writes_only_what_was_asked_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chosen.lammpstrj");
        write_lammps_dump(
            &path,
            &[wide_frame()],
            Some(&["id", "element", "mol", "x", "y", "z"]),
        )
        .unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("ITEM: ATOMS id element mol x y z\n"));
        assert!(!text.contains("mass"));
        assert!(text.contains("1 O 1 0.000000 0.000000 0.000000\n"));
    }

    #[test]
    fn write_dump_columns_rejects_a_column_the_frame_lacks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.lammpstrj");
        let err = write_lammps_dump(&path, &[wide_frame()], Some(&["id", "q"])).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("'q'"), "{message}");
        assert!(message.contains("element"), "{message}");
    }

    #[test]
    fn write_dump_columns_rejects_an_empty_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.lammpstrj");
        assert!(write_lammps_dump(&path, &[wide_frame()], Some(&[])).is_err());
    }
}
