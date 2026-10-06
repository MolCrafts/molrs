//! Reading a `molrec_version` 1 record under version 2 — the per-style
//! force-field parameter conversion the mrec reader (`io::mrec`) applies.
//!
//! It lives in `ff` because each rule is a fact about a style's parameters
//! (a harmonic `k` halved, an angle in radians become degrees); the reader
//! only applies it. Without the `ff` feature a version-1 record is refused.
//!
//! Contract: molrec `docs/spec/overview.md` (versions) and
//! `docs/spec/forcefield.md` ("Reading a version-1 record"). Version 2 changed
//! what some stored numbers mean — the force-field IR adopted LAMMPS's
//! definitions — so a version-1 record is never read as version 2: every
//! number whose meaning changed is converted, exactly, or the record is
//! refused. A store without `molrec_version` predates version 1 and is read
//! by the same rules.
//!
//! # What changed between version 1 and version 2
//!
//! In the `forcefield` section (version 1's angle unit `U` is
//! `units.angle`, else the radian every version-1 preset gives):
//!
//! | Style | Version 1 | Version 2 |
//! |---|---|---|
//! | `units` | `angle` = `U` (presets: radian) | `angle` = `degree` |
//! | `bond harmonic`, `drude harmonic` | `k` of ½k(x − x0)² | `k` = k₁ / 2 |
//! | `angle harmonic` | `k` of ½k(θ − θ0)² per U², `theta0` in U | `k` = k₁ / 2 per rad², `theta0` in degrees |
//! | `angle class2` | `theta0` in U, `k2..k4` per Uⁿ | degrees, per radⁿ |
//! | `improper harmonic` | `chi0` in U, `k` per U² | degrees, per rad² |
//! | `dihedral periodic`, `improper periodic`, `improper trefoil` | `phase`, `phase<m>` in U | degrees |
//! | `dihedral charmm` | `phase` in U | degrees |
//! | `dihedral class2` | `phi1..phi3` in U | degrees |
//! | `bond morse` | `D` | `d0` |
//!
//! and molrs ≤ 0.15's own styles (outside the version-1 registry):
//! `dihedral fourier` is `dihedral periodic`; `pair morse` `D0` is `d0`;
//! `pair thole` `a_thole` is `damp`; `angle mmff_angle` / `uff_angle`
//! `theta0` is in degrees; `improper mmff_oop` / `uff_inversion` list the
//! centre first (it was second), so their `itom` / `jtom` swap.
//!
//! In frames (`system`, `frame`, every trajectory frame): a relation row's
//! parameter columns convert as the parameter of the same name of the row's
//! style (linking rule 2: its `style` column, else every table of its
//! category holding its `type`), and a row of an out-of-plane style swaps
//! `atomi` / `atomj`. A row no style resolves converts its angle-valued
//! columns (`theta0`, `chi0`, `phase`, `phase<m>`, `phi1..phi3`) from `U`
//! (the radian when the record has no force field), and an `impropers` row
//! carrying `koop` (MMFF) or `K` (UFF) is an out-of-plane row.
//!
//! # What is refused
//!
//! A `pair14` style (version 1 priced 1-4 pairs from it, unweighted; version
//! 2 has no force-field form for that); an `improper periodic` row with
//! per-term columns (version 2's is one term); an `expression` on a style
//! this conversion changes; a style of an angular category (`angle`,
//! `dihedral`, `improper`) that neither registry nor molrs ≤ 0.15 knows and
//! that carries parameters (its angle values and per-angle constants cannot
//! be told apart); `units.angle` beside a preset other than the radian
//! (version 1 refused it too), or a unit other than the radian and the
//! degree; a renamed parameter as a relation column; and a relation cell
//! two resolved styles would convert differently.
//!
//! Unchanged: `dihedral charmm` `w` (LAMMPS's 1-4 weight in both; version 2
//! also prices it), `improper cvff` (cos nχ = cos nφ), every other registered
//! style, `special_bonds`, `cmap` grids, and the atom order of every other
//! improper style (each prices the dihedral of its atoms as listed in both
//! versions).

use std::collections::HashSet;

use indexmap::IndexMap;
use serde_json::Value as JsonValue;

use crate::error::MolRsError;
use crate::store::Column;
use crate::store::Frame;
use crate::store::forcefield_section::{is_parameter_column, unit_preset};
use crate::store::{Block, DType};
use crate::store::{ForceFieldSection, style_block_name};

fn refuse(message: impl std::fmt::Display) -> MolRsError {
    MolRsError::validation(format!("molrec_version 1 record: {message}"))
}

/// Version 1's angle unit: what its angle values are in, and what the angle
/// part of its force constants is per.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AngleUnit {
    Radian,
    Degree,
    /// `units` resolves no angle unit: angle values keep their numbers.
    Unresolved,
}

impl AngleUnit {
    fn parse(text: &str) -> Result<Self, MolRsError> {
        match text {
            "radian" | "radians" | "rad" => Ok(Self::Radian),
            "degree" | "degrees" | "deg" => Ok(Self::Degree),
            other => Err(refuse(format!(
                "units.angle {other:?}: this reader converts the radian and the degree"
            ))),
        }
    }

    /// An angle value in degrees.
    fn value(self, x: f64) -> f64 {
        match self {
            Self::Radian => x.to_degrees(),
            Self::Degree | Self::Unresolved => x,
        }
    }

    /// A constant per angle unitⁿ, per radianⁿ.
    fn per_angle(self, mut k: f64, power: u8) -> f64 {
        if self == Self::Degree {
            for _ in 0..power {
                k = k.to_degrees();
            }
        }
        k
    }
}

/// How one version-1 parameter becomes its version-2 self.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Conv {
    /// A number: halved when `half` (the ½ of a ½k form), its angle part
    /// re-expressed per radian^`per_angle`.
    Scale { half: bool, per_angle: u8 },
    /// An angle value: into degrees.
    Angle,
    /// The same number under another name.
    Rename(&'static str),
}

/// What happens to a version-1 style as a whole.
#[derive(Debug, Clone, Copy, PartialEq)]
enum StyleRule {
    /// Its numbers mean the same in both versions.
    Unchanged,
    /// Some of its columns convert ([`column_conv`]); `centre_second` when its
    /// rows listed an out-of-plane centre second.
    Converted { centre_second: bool },
    /// Renamed (`dihedral fourier` → `periodic`), then converted as that.
    Renamed(&'static str),
    /// No version-2 form.
    Refused(&'static str),
    /// Neither registry nor molrs ≤ 0.15 knows it.
    Unknown,
}

fn style_rule(category: &str, style: &str) -> StyleRule {
    use StyleRule::*;
    let plain = Converted {
        centre_second: false,
    };
    match (category, style) {
        ("bond" | "drude", "harmonic")
        | ("bond", "morse")
        | ("angle", "harmonic" | "class2" | "mmff_angle" | "uff_angle")
        | ("dihedral", "periodic" | "charmm" | "class2")
        | ("improper", "harmonic" | "periodic" | "trefoil")
        | ("pair", "morse" | "thole") => plain,
        ("improper", "mmff_oop" | "uff_inversion") => Converted {
            centre_second: true,
        },
        ("dihedral", "fourier") => Renamed("periodic"),
        ("pair14", _) => Refused(
            "version 1 priced 1-4 pairs from a pair14 table, unweighted; version 2 has no \
             force-field form for that (per-pair overrides on the system's pairs rows are one)",
        ),
        ("bond", "class2" | "mmff_bond" | "uff_bond")
        | ("angle", "mmff_stbn")
        | (
            "dihedral",
            "opls" | "rb" | "harmonic" | "multi/harmonic" | "mmff_torsion" | "uff_torsion",
        )
        | ("improper", "cvff") => Unchanged,
        ("angle" | "dihedral" | "improper", _) => Unknown,
        // Outside the angular categories no number is an angle value or a
        // per-angle constant, so version 1 and version 2 read it alike.
        _ => Unchanged,
    }
}

/// The version-2 name of the version-1 style `style` of `category`.
fn version2_style<'a>(category: &str, style: &'a str) -> &'a str {
    match style_rule(category, style) {
        StyleRule::Renamed(to) => to,
        _ => style,
    }
}

/// `phase<m>`, `k<m>`, `periodicity<m>`: `stem` followed by a term number.
fn term_of(column: &str, stem: &str) -> bool {
    column
        .strip_prefix(stem)
        .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

/// The version-1 angle values a relation row carries whatever its style.
fn is_angle_value(column: &str) -> bool {
    matches!(
        column,
        "theta0" | "chi0" | "phase" | "phi1" | "phi2" | "phi3"
    ) || term_of(column, "phase")
}

/// The conversion of `column` of a converted `(category, style)`, `None`
/// when the column keeps its number.
fn column_conv(category: &str, style: &str, column: &str) -> Result<Option<Conv>, MolRsError> {
    use Conv::*;
    let half = Scale {
        half: true,
        per_angle: 0,
    };
    Ok(match (category, style, column) {
        ("bond" | "drude", "harmonic", "k") => Some(half),
        ("bond", "morse", "D") => Some(Rename("d0")),
        ("angle", "harmonic", "k") => Some(Scale {
            half: true,
            per_angle: 2,
        }),
        ("angle", "class2", "k2") => Some(Scale {
            half: false,
            per_angle: 2,
        }),
        ("angle", "class2", "k3") => Some(Scale {
            half: false,
            per_angle: 3,
        }),
        ("angle", "class2", "k4") => Some(Scale {
            half: false,
            per_angle: 4,
        }),
        ("improper", "harmonic", "k") => Some(Scale {
            half: false,
            per_angle: 2,
        }),
        ("angle", "harmonic" | "class2" | "mmff_angle" | "uff_angle", "theta0")
        | ("improper", "harmonic", "chi0")
        | ("dihedral" | "improper", "charmm" | "periodic", "phase")
        | ("dihedral", "class2", "phi1" | "phi2" | "phi3") => Some(Angle),
        ("dihedral", "periodic", c) | ("improper", "trefoil", c)
            if c == "phase" || term_of(c, "phase") =>
        {
            Some(Angle)
        }
        ("improper", "periodic", c)
            if term_of(c, "k") || term_of(c, "periodicity") || term_of(c, "phase") =>
        {
            return Err(refuse(format!(
                "improper periodic column {c:?} is a term of a multi-term improper; version 2's \
                 improper periodic is one term"
            )));
        }
        ("pair", "morse", "D0") => Some(Rename("d0")),
        ("pair", "thole", "a_thole") => Some(Rename("damp")),
        _ => None,
    })
}

/// Apply `conv` (not a rename) to the non-null cells `rows` of float column
/// `column` of `block`.
fn convert_cells(
    block: &mut Block,
    column: &str,
    conv: Conv,
    unit: AngleUnit,
    rows: &[usize],
) -> Result<(), MolRsError> {
    let valid = block.validity(column).map(<[bool]>::to_vec);
    let values = block
        .get_mut(column)
        .and_then(|c| c.as_float_mut())
        .ok_or_else(|| refuse(format!("parameter {column:?} is no f64 column")))?;
    for &row in rows {
        if valid.as_ref().is_some_and(|mask| !mask[row]) {
            continue;
        }
        let x = &mut values.as_slice_mut().expect("a 1-D column is contiguous")[row];
        *x = match conv {
            Conv::Scale { half, per_angle } => {
                unit.per_angle(if half { *x * 0.5 } else { *x }, per_angle)
            }
            Conv::Angle => unit.value(*x),
            Conv::Rename(_) => unreachable!("renames are not cell conversions"),
        };
    }
    Ok(())
}

/// Swap `a` and `b` of `block` in `rows` (an out-of-plane improper's centre
/// moves from second to first).
fn swap_rows<T: Clone>(
    block: &mut Block,
    a: &str,
    b: &str,
    rows: &[usize],
    project: impl Fn(&mut Column) -> Option<&mut ndarray::ArrayD<T>>,
) -> Result<(), MolRsError> {
    let mut column = |key: &str| -> Result<Vec<T>, MolRsError> {
        let values = block
            .get_mut(key)
            .and_then(&project)
            .ok_or_else(|| refuse(format!("an out-of-plane improper has no {key} column")))?;
        Ok(values.iter().cloned().collect())
    };
    let (mut first, mut second) = (column(a)?, column(b)?);
    for &row in rows {
        std::mem::swap(&mut first[row], &mut second[row]);
    }
    for (key, values) in [(a, first), (b, second)] {
        let target = block.get_mut(key).and_then(&project).expect("read above");
        for (cell, value) in target.iter_mut().zip(values) {
            *cell = value;
        }
    }
    Ok(())
}

/// The conversion context of one version-1 record: its angle unit and its
/// force field's tables, to resolve each relation row's style.
#[derive(Debug, Clone)]
pub struct V1Upgrade {
    unit: AngleUnit,
    /// `(category, style, type names)` of every table, version-1 names.
    tables: Vec<(String, String, HashSet<String>)>,
}

impl V1Upgrade {
    /// The context of a version-1 record whose `forcefield` section, as
    /// stored, is `forcefield` (`None`: the record has none).
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Validation`] when `units` states an angle unit beside
    /// a preset other than the radian, or one this reader cannot convert.
    pub fn new(forcefield: Option<&ForceFieldSection>) -> Result<Self, MolRsError> {
        let Some(section) = forcefield else {
            return Ok(Self {
                unit: AngleUnit::Radian,
                tables: Vec::new(),
            });
        };
        let units = section.document.get("units").and_then(JsonValue::as_object);
        let stated = units
            .and_then(|u| u.get("angle"))
            .and_then(JsonValue::as_str);
        let preset = units
            .and_then(|u| u.get("preset"))
            .and_then(JsonValue::as_str)
            .filter(|p| unit_preset(p).is_some());
        let unit = match (preset, stated) {
            (_, Some(text)) => AngleUnit::parse(text)?,
            (Some(_), None) => AngleUnit::Radian,
            (None, None) => AngleUnit::Unresolved,
        };
        if let (Some(preset), Some(text)) = (preset, stated)
            && unit != AngleUnit::Radian
        {
            return Err(refuse(format!(
                "units.angle {text:?} disagrees with preset {preset:?}, whose version-1 angle is \
                 the radian"
            )));
        }
        let mut tables = Vec::new();
        for style in section.styles()? {
            let names = section
                .tables
                .get(&style.block_name())
                .and_then(|t| t.get("name"))
                .and_then(|c| c.as_string())
                .map(|names| names.iter().cloned().collect())
                .unwrap_or_default();
            tables.push((style.category.to_owned(), style.style.to_owned(), names));
        }
        Ok(Self { unit, tables })
    }

    /// The version-2 form of the version-1 `forcefield` section `v1`. The
    /// result is not validated; the caller validates it as any section.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Validation`] naming what has no exact version-2 form
    /// (module docs).
    pub fn forcefield(&self, v1: &ForceFieldSection) -> Result<ForceFieldSection, MolRsError> {
        let mut document = v1.document.clone();
        if self.unit != AngleUnit::Unresolved
            && let Some(units) = document.get_mut("units").and_then(JsonValue::as_object_mut)
        {
            units.insert("angle".into(), "degree".into());
        }
        let mut tables: IndexMap<String, Block> = v1.tables.clone();
        let raw: Vec<JsonValue> = v1
            .document
            .get("styles")
            .and_then(JsonValue::as_array)
            .cloned()
            .unwrap_or_default();
        let mut styles = Vec::with_capacity(raw.len());
        for (entry, fields) in v1.styles()?.into_iter().zip(raw) {
            let block = entry.block_name();
            let mut style = entry.style;
            let mut rule = style_rule(entry.category, style);
            let mut renamed = None;
            if let StyleRule::Renamed(to) = rule {
                let target = style_block_name(entry.category, to);
                if tables.contains_key(&target) {
                    return Err(refuse(format!(
                        "{}/{style} is {}/{to}, which the force field also holds",
                        entry.category, entry.category
                    )));
                }
                style = to;
                rule = style_rule(entry.category, to);
                renamed = Some((block.clone(), target));
            }
            let table = tables.get_mut(&block);
            let has_params = entry.params.is_some_and(|p| !p.is_empty())
                || table
                    .as_ref()
                    .is_some_and(|t| t.keys().any(is_parameter_column));
            match rule {
                StyleRule::Unchanged => {}
                StyleRule::Refused(why) => {
                    return Err(refuse(format!("{}/{style}: {why}", entry.category)));
                }
                StyleRule::Unknown => {
                    if has_params || entry.expression.is_some() {
                        return Err(refuse(format!(
                            "{}/{style} is no style this reader knows; its angle values and \
                             per-angle constants cannot be told apart from its other numbers",
                            entry.category
                        )));
                    }
                }
                StyleRule::Renamed(_) => unreachable!("a rename resolves to a converted style"),
                StyleRule::Converted { centre_second } => {
                    if entry.expression.is_some() {
                        return Err(refuse(format!(
                            "{}/{style} carries an expression in version 1's meaning of its \
                             parameters",
                            entry.category
                        )));
                    }
                    if let Some(table) = table {
                        self.convert_table(entry.category, style, table, centre_second)?;
                    }
                }
            }
            if let Some((from, to)) = renamed {
                let index = tables.get_index_of(&from).expect("the style's table");
                let (_, table) = tables.shift_remove_index(index).expect("present");
                tables.shift_insert(index, to, table);
            }
            let mut fields = fields;
            if let Some(fields) = fields.as_object_mut() {
                fields.insert("style".into(), style.into());
            }
            styles.push(fields);
        }
        if document.contains_key("styles") {
            document.insert("styles".into(), JsonValue::Array(styles));
        }
        Ok(ForceFieldSection { document, tables })
    }

    fn convert_table(
        &self,
        category: &str,
        style: &str,
        table: &mut Block,
        centre_second: bool,
    ) -> Result<(), MolRsError> {
        let rows: Vec<usize> = (0..table.nrows().unwrap_or(0)).collect();
        let columns: Vec<String> = table.keys().map(str::to_owned).collect();
        for column in &columns {
            match column_conv(category, style, column)? {
                None => {}
                Some(Conv::Rename(to)) => {
                    if table.contains_key(to) {
                        return Err(refuse(format!(
                            "{category}/{style} holds both {column:?} and {to:?}"
                        )));
                    }
                    table
                        .rename_column(column, to)
                        .map_err(|e| refuse(format!("{category}/{style}: {e}")))?;
                }
                Some(conv) => convert_cells(table, column, conv, self.unit, &rows)
                    .map_err(|e| refuse(format!("{category}/{style}: {e}")))?,
            }
        }
        if centre_second && !rows.is_empty() {
            swap_rows(table, "itom", "jtom", &rows, Column::as_string_mut)?;
        }
        Ok(())
    }

    /// The version-1 styles of `category` a relation row resolves to: its
    /// `style` cell, else every table of the category holding its `type`.
    fn resolve<'a>(
        &'a self,
        category: &str,
        style: Option<&'a str>,
        name: Option<&str>,
    ) -> Vec<&'a str> {
        match style {
            Some(style) => vec![style],
            None => self
                .tables
                .iter()
                .filter(|(c, _, names)| c == category && name.is_some_and(|n| names.contains(n)))
                .map(|(_, s, _)| s.as_str())
                .collect(),
        }
    }

    /// Convert the relation blocks of the version-1 frame `frame` in place.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Validation`] for a relation column with no exact
    /// version-2 form (module docs).
    pub fn frame(&self, frame: &mut Frame) -> Result<(), MolRsError> {
        for (block_name, category) in [
            ("bonds", "bond"),
            ("angles", "angle"),
            ("dihedrals", "dihedral"),
            ("impropers", "improper"),
            ("drudes", "drude"),
            ("constraints", "constraint"),
        ] {
            if let Some(block) = frame.get_mut(block_name) {
                self.relation(block_name, category, block)?;
            }
        }
        Ok(())
    }

    fn relation(
        &self,
        block_name: &str,
        category: &str,
        block: &mut Block,
    ) -> Result<(), MolRsError> {
        let nrows = block.nrows().unwrap_or(0);
        let cell = |block: &Block, key: &str, row: usize| -> Option<String> {
            let valid = block.validity(key).is_none_or(|mask| mask[row]);
            block
                .get(key)
                .and_then(|c| c.as_string())
                .filter(|_| valid)
                .map(|values| values[[row]].clone())
        };
        let non_null = |block: &Block, key: &str, row: usize| -> bool {
            block.contains_key(key) && block.validity(key).is_none_or(|mask| mask[row])
        };
        // Each row's version-2 styles (a renamed style under its new name).
        let resolved: Vec<Vec<String>> = (0..nrows)
            .map(|row| {
                let style = cell(block, "style", row);
                let name = cell(block, "type", row);
                self.resolve(category, style.as_deref(), name.as_deref())
                    .into_iter()
                    .map(|s| version2_style(category, s).to_owned())
                    .collect()
            })
            .collect();

        let columns: Vec<String> = block.keys().map(str::to_owned).collect();
        for column in &columns {
            if matches!(column.as_str(), "type" | "type_id" | "style") || column.starts_with("atom")
            {
                continue;
            }
            let mut groups: Vec<(Conv, Vec<usize>)> = Vec::new();
            for (row, styles) in resolved.iter().enumerate() {
                if !non_null(block, column, row) {
                    continue;
                }
                // An angle value is one in every style that names it.
                let conv = if is_angle_value(column) {
                    Some(Conv::Angle)
                } else {
                    let mut convs = Vec::with_capacity(styles.len());
                    for style in styles {
                        convs.push(self.effective(category, style, column)?);
                    }
                    if convs.windows(2).any(|w| w[0] != w[1]) {
                        return Err(refuse(format!(
                            "{block_name} row {row}: its styles {styles:?} read column \
                             {column:?} differently"
                        )));
                    }
                    convs.first().copied().flatten()
                };
                let Some(conv) = conv else { continue };
                if let Conv::Rename(to) = conv {
                    return Err(refuse(format!(
                        "{block_name} column {column:?} is {to:?} in version 2; a relation \
                         column is not renamed row by row"
                    )));
                }
                match groups.iter_mut().find(|(c, _)| *c == conv) {
                    Some((_, rows)) => rows.push(row),
                    None => groups.push((conv, vec![row])),
                }
            }
            for (conv, rows) in groups {
                convert_cells(block, column, conv, self.unit, &rows)
                    .map_err(|e| refuse(format!("{block_name}: {e}")))?;
            }
        }

        // A renamed style is named by its version-2 name.
        if let Some(styles) = block.get_mut("style").and_then(Column::as_string_mut) {
            for style in styles.iter_mut() {
                let renamed = version2_style(category, style);
                if renamed != style.as_str() {
                    *style = renamed.to_owned();
                }
            }
        }

        if category == "improper" {
            let mut centre_second = Vec::new();
            for (row, styles) in resolved.iter().enumerate() {
                let swaps = if styles.is_empty() {
                    non_null(block, "koop", row) || non_null(block, "K", row)
                } else {
                    let out_of_plane = |s: &String| {
                        matches!(
                            style_rule(category, s),
                            StyleRule::Converted {
                                centre_second: true
                            }
                        )
                    };
                    if styles.iter().any(out_of_plane) && !styles.iter().all(out_of_plane) {
                        return Err(refuse(format!(
                            "{block_name} row {row}: its styles {styles:?} put the centre in \
                             different places"
                        )));
                    }
                    styles.iter().any(out_of_plane)
                };
                if swaps {
                    centre_second.push(row);
                }
            }
            // A column subset without both endpoints has nothing to reorder.
            let endpoints = block.contains_key("atomi") && block.contains_key("atomj");
            if endpoints && !centre_second.is_empty() {
                if block.get("atomi").map(Column::dtype) != Some(DType::UInt) {
                    return Err(refuse(format!("{block_name}: atomi is no u64 column")));
                }
                swap_rows(block, "atomi", "atomj", &centre_second, Column::as_uint_mut)?;
            }
        }
        Ok(())
    }

    /// The conversion `style` gives `column`, as it acts on a number: a
    /// constant per angle unit re-expressed per radian is no conversion when
    /// the unit is the radian.
    fn effective(
        &self,
        category: &str,
        style: &str,
        column: &str,
    ) -> Result<Option<Conv>, MolRsError> {
        let conv = match style_rule(category, style) {
            StyleRule::Converted { .. } => column_conv(category, style, column)?,
            _ => None,
        };
        Ok(match conv {
            Some(Conv::Scale {
                half: false,
                per_angle,
            }) if per_angle == 0 || self.unit != AngleUnit::Degree => None,
            other => other,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ForceFieldSection;
    use ndarray::ArrayD;
    use serde_json::json;
    use std::f64::consts::PI;

    fn strings(values: &[&str]) -> Column {
        Column::from_string(
            ArrayD::from_shape_vec(
                vec![values.len()],
                values.iter().map(|s| s.to_string()).collect(),
            )
            .unwrap(),
        )
    }

    fn floats(values: &[f64]) -> Column {
        Column::from_float(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn uints(values: &[u64]) -> Column {
        Column::from_uint(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn table(columns: Vec<(&str, Column)>) -> Block {
        let mut block = Block::new();
        for (name, column) in columns {
            block.insert_column(name, column).unwrap();
        }
        block
    }

    /// One row of `category/style` named `name` over `ends`, with `params`.
    fn row(ends: &[&str], params: &[(&str, f64)]) -> Block {
        let mut columns = vec![("name", strings(&["t"]))];
        for (endpoint, end) in ["itom", "jtom", "ktom", "ltom"].iter().zip(ends) {
            columns.push((endpoint, strings(&[end])));
        }
        for (key, value) in params {
            columns.push((key, floats(&[*value])));
        }
        table(columns)
    }

    /// A version-1 section: `units`, and one table per `(category, style,
    /// table)`.
    fn section(units: JsonValue, styles: Vec<(&str, &str, Block)>) -> ForceFieldSection {
        let mut tables = IndexMap::new();
        let mut entries = Vec::new();
        for (category, style, block) in styles {
            entries.push(json!({"category": category, "style": style}));
            tables.insert(style_block_name(category, style), block);
        }
        ForceFieldSection {
            document: json!({"name": "v1", "units": units, "styles": entries})
                .as_object()
                .unwrap()
                .clone(),
            tables,
        }
    }

    fn real() -> JsonValue {
        json!({"preset": "real", "length": "angstrom", "energy": "kcal/mol",
               "angle": "radian", "charge": "e", "mass": "dalton"})
    }

    fn upgraded(v1: &ForceFieldSection) -> ForceFieldSection {
        let v2 = V1Upgrade::new(Some(v1)).unwrap().forcefield(v1).unwrap();
        v2.validate().unwrap();
        v2
    }

    fn value(ff: &ForceFieldSection, block: &str, column: &str) -> f64 {
        ff.tables[block][column].as_float().unwrap()[[0]]
    }

    #[test]
    fn every_changed_parameter_converts_exactly() {
        let ab = ["A", "B"];
        let abc = ["A", "B", "C"];
        let abcd = ["A", "B", "C", "D"];
        let v1 = section(
            real(),
            vec![
                ("bond", "harmonic", row(&ab, &[("k", 600.0), ("r0", 1.5)])),
                ("bond", "morse", row(&ab, &[("D", 90.0), ("alpha", 2.0)])),
                ("drude", "harmonic", row(&ab, &[("k", 500.0)])),
                (
                    "angle",
                    "harmonic",
                    row(&abc, &[("k", 100.0), ("theta0", 1.9)]),
                ),
                (
                    "angle",
                    "class2",
                    row(&abc, &[("theta0", 2.0), ("k2", 50.0), ("k4", 4.0)]),
                ),
                (
                    "dihedral",
                    "periodic",
                    row(&abcd, &[("k1", 0.5), ("phase1", PI), ("phase2", 0.3)]),
                ),
                (
                    "dihedral",
                    "charmm",
                    row(&abcd, &[("k", 0.4), ("phase", 0.2), ("w", 0.5)]),
                ),
                (
                    "dihedral",
                    "class2",
                    row(&abcd, &[("phi1", 0.1), ("phi3", -0.4)]),
                ),
                (
                    "improper",
                    "harmonic",
                    row(&abcd, &[("k", 12.0), ("chi0", 0.11)]),
                ),
                (
                    "improper",
                    "periodic",
                    row(&abcd, &[("k", 1.1), ("phase", PI)]),
                ),
                (
                    "improper",
                    "cvff",
                    row(&abcd, &[("k", 1.0), ("sign", -1.0)]),
                ),
                ("improper", "mmff_oop", row(&abcd, &[("koop", 0.05)])),
                (
                    "pair",
                    "morse",
                    row(&["A", "A"], &[("D0", 0.1), ("alpha", 1.5)]),
                ),
                (
                    "pair",
                    "thole",
                    row(&["A", "A"], &[("alpha", 1.0), ("a_thole", 2.6)]),
                ),
            ],
        );
        let v2 = upgraded(&v1);
        assert_eq!(v2.document["units"]["angle"], "degree");
        assert_eq!(value(&v2, "bond.harmonic", "k"), 300.0);
        assert_eq!(value(&v2, "bond.harmonic", "r0"), 1.5);
        assert_eq!(value(&v2, "bond.morse", "d0"), 90.0);
        assert!(!v2.tables["bond.morse"].contains_key("D"));
        assert_eq!(value(&v2, "drude.harmonic", "k"), 250.0);
        assert_eq!(value(&v2, "angle.harmonic", "k"), 50.0);
        assert_eq!(value(&v2, "angle.harmonic", "theta0"), 1.9_f64.to_degrees());
        assert_eq!(value(&v2, "angle.class2", "theta0"), 2.0_f64.to_degrees());
        assert_eq!(value(&v2, "angle.class2", "k2"), 50.0);
        assert_eq!(value(&v2, "dihedral.periodic", "phase1"), 180.0);
        assert_eq!(value(&v2, "dihedral.periodic", "k1"), 0.5);
        assert_eq!(
            value(&v2, "dihedral.periodic", "phase2"),
            0.3_f64.to_degrees()
        );
        assert_eq!(value(&v2, "dihedral.charmm", "phase"), 0.2_f64.to_degrees());
        assert_eq!(value(&v2, "dihedral.charmm", "w"), 0.5);
        assert_eq!(
            value(&v2, "dihedral.class2", "phi3"),
            (-0.4_f64).to_degrees()
        );
        assert_eq!(value(&v2, "improper.harmonic", "k"), 12.0);
        assert_eq!(
            value(&v2, "improper.harmonic", "chi0"),
            0.11_f64.to_degrees()
        );
        assert_eq!(value(&v2, "improper.periodic", "phase"), 180.0);
        assert_eq!(value(&v2, "improper.cvff", "sign"), -1.0);
        assert_eq!(value(&v2, "pair.morse", "d0"), 0.1);
        assert_eq!(value(&v2, "pair.thole", "damp"), 2.6);
        let oop = &v2.tables["improper.mmff_oop"];
        assert_eq!(oop["itom"].as_string().unwrap()[[0]], "B");
        assert_eq!(oop["jtom"].as_string().unwrap()[[0]], "A");
        assert_eq!(oop["ktom"].as_string().unwrap()[[0]], "C");
    }

    /// `dihedral fourier` is `dihedral periodic`: the style entry, its table's
    /// block name and a frame's `style` cells move to the new name.
    #[test]
    fn dihedral_fourier_is_dihedral_periodic() {
        let abcd = ["A", "B", "C", "D"];
        let v1 = section(
            real(),
            vec![(
                "dihedral",
                "fourier",
                row(&abcd, &[("k1", 0.5), ("phase1", 0.7)]),
            )],
        );
        let v2 = upgraded(&v1);
        assert_eq!(v2.document["styles"][0]["style"], "periodic");
        assert!(!v2.tables.contains_key("dihedral.fourier"));
        assert_eq!(
            value(&v2, "dihedral.periodic", "phase1"),
            0.7_f64.to_degrees()
        );

        let both = section(
            real(),
            vec![
                ("dihedral", "fourier", row(&abcd, &[("k1", 0.5)])),
                ("dihedral", "periodic", row(&abcd, &[("k", 0.5)])),
            ],
        );
        let err = V1Upgrade::new(Some(&both))
            .unwrap()
            .forcefield(&both)
            .unwrap_err()
            .to_string();
        assert!(err.contains("also holds"), "{err}");
    }

    /// The `lj` preset stated no angle (molrs 0.15 wrote `{"preset": "lj"}`):
    /// its version-1 angle is the radian all the same.
    #[test]
    fn units_lj_and_stated_units_resolve_the_version_1_angle() {
        let abc = ["A", "B", "C"];
        let lj = section(
            json!({"preset": "lj"}),
            vec![(
                "angle",
                "harmonic",
                row(&abc, &[("k", 40.0), ("theta0", 2.0)]),
            )],
        );
        let v2 = upgraded(&lj);
        assert_eq!(
            v2.document["units"],
            json!({"preset": "lj", "angle": "degree"})
        );
        assert_eq!(value(&v2, "angle.harmonic", "theta0"), 2.0_f64.to_degrees());

        // A degree statement without a preset: values stay, per-degree
        // constants become per-radian ones.
        let degree = section(
            json!({"energy": "kcal/mol", "angle": "degree"}),
            vec![(
                "angle",
                "harmonic",
                row(&abc, &[("k", 0.02), ("theta0", 109.5)]),
            )],
        );
        let v2 = upgraded(&degree);
        assert_eq!(value(&v2, "angle.harmonic", "theta0"), 109.5);
        assert_eq!(
            value(&v2, "angle.harmonic", "k"),
            (0.02 * 0.5_f64).to_degrees().to_degrees()
        );

        // No angle unit at all: the values have none in either version.
        let none = section(
            json!({"energy": "kcal/mol"}),
            vec![(
                "angle",
                "harmonic",
                row(&abc, &[("k", 40.0), ("theta0", 2.0)]),
            )],
        );
        let v2 = upgraded(&none);
        assert_eq!(value(&v2, "angle.harmonic", "theta0"), 2.0);
        assert_eq!(value(&v2, "angle.harmonic", "k"), 20.0);
        assert!(v2.document["units"].get("angle").is_none());

        for (units, why) in [
            (json!({"preset": "real", "angle": "degree"}), "disagrees"),
            (json!({"angle": "grad"}), "converts the radian"),
        ] {
            let bad = section(units, vec![]);
            let err = V1Upgrade::new(Some(&bad)).unwrap_err().to_string();
            assert!(err.contains(why), "{err}");
        }
    }

    #[test]
    fn what_has_no_exact_version_2_form_is_refused() {
        let ab = ["A", "B"];
        let abc = ["A", "B", "C"];
        let abcd = ["A", "B", "C", "D"];
        let expression = {
            let mut ff = section(real(), vec![("bond", "harmonic", row(&ab, &[("k", 1.0)]))]);
            ff.document["styles"][0]["expression"] = json!("0.5*k*(r-r0)^2");
            ff
        };
        for (v1, why) in [
            (
                section(
                    real(),
                    vec![("pair14", "lj/cut", row(&["A", "A"], &[("epsilon", 0.1)]))],
                ),
                "pair14",
            ),
            (
                section(
                    real(),
                    vec![(
                        "improper",
                        "periodic",
                        row(&abcd, &[("k1", 1.0), ("phase1", 0.0)]),
                    )],
                ),
                "one term",
            ),
            (expression, "expression"),
            (
                section(
                    real(),
                    vec![("angle", "cosine/squared", row(&abc, &[("k", 1.0)]))],
                ),
                "no style this reader knows",
            ),
        ] {
            let err = V1Upgrade::new(Some(&v1))
                .unwrap()
                .forcefield(&v1)
                .unwrap_err()
                .to_string();
            assert!(err.contains(why), "{why}: {err}");
        }

        // An unknown style elsewhere, or an unknown angular style with no
        // numbers, has nothing to misread.
        let fine = section(
            real(),
            vec![
                ("pair", "coul/tt", row(&["A", "A"], &[("b", 1.0)])),
                ("angle", "cosine/squared", row(&abc, &[])),
            ],
        );
        upgraded(&fine);
    }

    fn mmff_frame() -> Frame {
        let mut frame = Frame::new();
        frame.insert(
            "angles",
            table(vec![
                ("atomi", uints(&[0, 1])),
                ("atomj", uints(&[1, 2])),
                ("atomk", uints(&[2, 3])),
                ("theta0", floats(&[1.9, 2.0])),
                ("ka", floats(&[0.7, 0.8])),
                ("type", strings(&["0_1_2_1", "0_2_2_2"])),
            ]),
        );
        frame.insert(
            "impropers",
            table(vec![
                ("atomi", uints(&[0, 4])),
                ("atomj", uints(&[1, 1])),
                ("atomk", uints(&[4, 5])),
                ("atoml", uints(&[5, 0])),
                ("koop", floats(&[0.05, 0.05])),
                ("type", strings(&["1_3_7_10", "1_3_7_10"])),
            ]),
        );
        frame
    }

    fn column_u(frame: &Frame, block: &str, column: &str) -> Vec<u64> {
        frame[block][column]
            .as_uint()
            .unwrap()
            .iter()
            .copied()
            .collect()
    }

    /// Without a force field, the frame's own columns convert it: `theta0`
    /// to degrees, and `koop` rows are out-of-plane rows, centre first.
    #[test]
    fn an_mmff_frame_converts_by_its_own_columns() {
        let mut frame = mmff_frame();
        V1Upgrade::new(None).unwrap().frame(&mut frame).unwrap();
        let theta0 = frame["angles"]["theta0"].as_float().unwrap();
        assert_eq!(theta0[[0]], 1.9_f64.to_degrees());
        assert_eq!(theta0[[1]], 2.0_f64.to_degrees());
        assert_eq!(frame["angles"]["ka"].as_float().unwrap()[[0]], 0.7);
        assert_eq!(column_u(&frame, "impropers", "atomi"), [1, 1]);
        assert_eq!(column_u(&frame, "impropers", "atomj"), [0, 4]);
        assert_eq!(column_u(&frame, "impropers", "atomk"), [4, 5]);
    }

    /// With a force field, a row's style decides: a per-instance `k` on a
    /// harmonic bond row is halved, the same column on a class2 row is not;
    /// a `style` column naming `fourier` names `periodic`.
    #[test]
    fn relation_columns_convert_as_their_rows_style() {
        let ab = ["A", "B"];
        let abcd = ["A", "B", "C", "D"];
        let mut harmonic = row(&ab, &[("k", 1.0)]);
        harmonic.get_mut("name").unwrap().as_string_mut().unwrap()[[0]] = "h".into();
        let mut class2 = row(&ab, &[("k2", 1.0)]);
        class2.get_mut("name").unwrap().as_string_mut().unwrap()[[0]] = "c".into();
        let v1 = section(
            real(),
            vec![
                ("bond", "harmonic", harmonic),
                ("bond", "class2", class2),
                ("dihedral", "fourier", row(&abcd, &[("k1", 1.0)])),
            ],
        );
        let upgrade = V1Upgrade::new(Some(&v1)).unwrap();
        let mut frame = Frame::new();
        frame.insert(
            "bonds",
            table(vec![
                ("atomi", uints(&[0, 1])),
                ("atomj", uints(&[1, 2])),
                ("type", strings(&["h", "c"])),
                ("k", floats(&[600.0, 600.0])),
            ]),
        );
        frame.insert(
            "dihedrals",
            table(vec![
                ("atomi", uints(&[0])),
                ("atomj", uints(&[1])),
                ("atomk", uints(&[2])),
                ("atoml", uints(&[3])),
                ("type", strings(&["t"])),
                ("style", strings(&["fourier"])),
                ("phase1", floats(&[PI])),
            ]),
        );
        upgrade.frame(&mut frame).unwrap();
        let k = frame["bonds"]["k"].as_float().unwrap();
        assert_eq!((k[[0]], k[[1]]), (300.0, 600.0));
        assert_eq!(
            frame["dihedrals"]["style"].as_string().unwrap()[[0]],
            "periodic"
        );
        assert_eq!(frame["dihedrals"]["phase1"].as_float().unwrap()[[0]], 180.0);

        // A renamed parameter has no per-row form.
        let mut frame = Frame::new();
        frame.insert(
            "bonds",
            table(vec![
                ("atomi", uints(&[0])),
                ("atomj", uints(&[1])),
                ("style", strings(&["morse"])),
                ("D", floats(&[90.0])),
            ]),
        );
        let err = upgrade.frame(&mut frame).unwrap_err().to_string();
        assert!(err.contains("\"d0\""), "{err}");
    }

    /// A null cell is no value: it is left as it is.
    #[test]
    fn a_null_cell_is_not_converted() {
        let mut frame = mmff_frame();
        frame
            .get_mut("angles")
            .unwrap()
            .set_validity("theta0", vec![true, false])
            .unwrap();
        V1Upgrade::new(None).unwrap().frame(&mut frame).unwrap();
        let theta0 = frame["angles"]["theta0"].as_float().unwrap();
        assert_eq!((theta0[[0]], theta0[[1]]), (1.9_f64.to_degrees(), 2.0));
    }
}
