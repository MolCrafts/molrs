//! The `forcefield` section of a MolRec record — the force-field document and
//! one table per style, as data.
//!
//! Contract: molrec `docs/spec/forcefield.md`. The section is frame-shaped: its
//! attribute map is the **document** (`name`, `units`, `source`,
//! `special_bonds`, the ordered `styles` list, and any key a producer adds),
//! and every style has one [`Block`] — its **table**, one row per type — at
//! the block name [`style_block_name`] gives it.
//!
//! [`ForceFieldSection`] is neutral: it knows nothing of the force-field
//! kernels. It holds what a store holds, keeps every key and table it does not
//! interpret, and converts no unit. `ff::forcefield::ForceField::to_section` /
//! `from_section` (feature `ff`) map it onto a compilable force field.
//!
//! # What [`ForceFieldSection::validate`] refuses
//!
//! - a document without a string `name`, or with `units` stating neither a
//!   preset nor any quantity, or a quantity that is not the preset's own;
//! - a malformed `source`, `special_bonds` or style entry (a category outside
//!   `^[a-z][a-z0-9_]*$`, an empty style, a non-finite or non-scalar param, a
//!   `params.special` other than `lj` / `coul`, a `params.mixing` that is no
//!   combining rule, an `endpoint_key` other than `type` / `class` /
//!   `smirks`);
//! - a `(category, style)` listed twice, or a style without its table;
//! - a style table with a structural shape, without a unique never-null string
//!   `name`, with endpoint columns other than its category's (or, keyed by
//!   `smirks`, any endpoint column or no never-null `smirks`), or with a column
//!   that is not `f64` / `string` (an annotation column: `string`; a canonical
//!   key: that key's dtype), has trailing axes, or declares a precision — the
//!   one exception being a `cmap` table's [`CMAP_GRID`] column, `f64[T, N, N]`
//!   with `N ≥ 2` and every value of a non-null row finite;
//! - a `class`-keyed style beside an atom table without `class`;
//! - a `pair` or `pair14` table (endpoint-keyed) with two rows on one
//!   unordered `{itom, jtom}` that differ in a parameter
//!   ([`check_pair_restatements`]).
//!
//! A table no style names is unknown content: kept, never checked.

use indexmap::IndexMap;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::MolRsError;
use crate::store::block::{Block, DType};

/// The endpoint columns of a style table, in position order.
pub const ENDPOINT_COLUMNS: [&str; 5] = ["itom", "jtom", "ktom", "ltom", "mtom"];

/// The one parameter column with trailing axes: a `cmap` row's correction
/// table, `f64[T, N, N]` — an `N × N` grid over the two dihedrals, every row
/// of the table sharing `N`.
pub const CMAP_GRID: &str = "grid";

/// The annotation columns of a style table: `string`, nullable.
pub const ANNOTATION_COLUMNS: [&str; 7] = [
    "class",
    "element",
    "smarts",
    "smirks",
    "overrides",
    "desc",
    "doi",
];

/// The categories whose rows are found by their unordered `{itom, jtom}`, not
/// by name (molrec forcefield, linking rule 3).
pub const PAIR_CATEGORIES: [&str; 2] = ["pair", "pair14"];

/// Whether a style-table column is a parameter: not `name`, an endpoint or an
/// annotation column. Rows restating one pair are compared on these alone.
pub fn is_parameter_column(column: &str) -> bool {
    column != "name" && !ENDPOINT_COLUMNS.contains(&column) && !ANNOTATION_COLUMNS.contains(&column)
}

/// The combining rules `params.mixing` may name on a van-der-Waals pair style.
pub const MIXING_RULES: [&str; 3] = ["arithmetic", "geometric", "sixthpower"];

/// The quantities `units` may state, in the order [`unit_preset`] lists them.
pub const UNIT_QUANTITIES: [&str; 6] = ["length", "energy", "angle", "charge", "mass", "time"];

/// The unit of each of [`UNIT_QUANTITIES`] in the preset `name`, or `None`
/// when `name` is no preset. A `None` entry is a quantity the preset gives no
/// unit (reduced `lj`). The angle is a radian in every preset.
pub fn unit_preset(name: &str) -> Option<[Option<&'static str>; 6]> {
    Some(match name {
        "real" => [
            Some("angstrom"),
            Some("kcal/mol"),
            Some("radian"),
            Some("e"),
            Some("dalton"),
            Some("fs"),
        ],
        "metal" => [
            Some("angstrom"),
            Some("eV"),
            Some("radian"),
            Some("e"),
            Some("dalton"),
            Some("ps"),
        ],
        "si" => [
            Some("m"),
            Some("J"),
            Some("radian"),
            Some("C"),
            Some("kg"),
            Some("s"),
        ],
        "cgs" => [
            Some("cm"),
            Some("erg"),
            Some("radian"),
            Some("statcoulomb"),
            Some("g"),
            Some("s"),
        ],
        "electron" => [
            Some("bohr"),
            Some("hartree"),
            Some("radian"),
            Some("e"),
            Some("dalton"),
            Some("fs"),
        ],
        "micro" => [
            Some("micrometer"),
            Some("picogram * micrometer**2 / microsecond**2"),
            Some("radian"),
            Some("picocoulomb"),
            Some("picogram"),
            Some("microsecond"),
        ],
        "nano" => [
            Some("nm"),
            Some("attogram * nm**2 / ns**2"),
            Some("radian"),
            Some("e"),
            Some("attogram"),
            Some("ns"),
        ],
        "lj" => [None, None, Some("radian"), None, None, None],
        _ => return None,
    })
}

/// How many endpoints a row of `category` names, for a category of the
/// chapter's table; `None` for any other (its arity is its endpoint prefix).
pub fn category_arity(category: &str) -> Option<usize> {
    Some(match category {
        "atom" | "virtual_site" => 0,
        "bond" | "pair" | "pair14" | "constraint" | "drude" => 2,
        "angle" => 3,
        "dihedral" | "improper" => 4,
        "cmap" => 5,
        _ => return None,
    })
}

/// Whether `byte` stays verbatim in a style's block name.
fn unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'
}

/// The block a style's table lives at: `<category>.<encoded style>`, every
/// byte of the UTF-8 style outside `A-Z a-z 0-9 - _` written `%XX` (uppercase
/// hex). `lj/cut/coul/long` under `pair` is `pair.lj%2Fcut%2Fcoul%2Flong`.
pub fn style_block_name(category: &str, style: &str) -> String {
    use std::fmt::Write as _;
    let mut name = String::with_capacity(category.len() + 1 + style.len());
    name.push_str(category);
    name.push('.');
    for &byte in style.as_bytes() {
        if unreserved(byte) {
            name.push(byte as char);
        } else {
            let _ = write!(name, "%{byte:02X}");
        }
    }
    name
}

/// `(category, style)` of a style table's block name: the inverse of
/// [`style_block_name`], refusing every other spelling (a lowercase escape, a
/// reserved byte written verbatim, an unreserved byte escaped).
///
/// # Errors
///
/// A [`MolRsError::Validation`] naming `name`.
pub fn parse_style_block_name(name: &str) -> Result<(String, String), MolRsError> {
    let bad = |why: &str| MolRsError::validation(format!("{name:?} is no style table name: {why}"));
    let (category, encoded) = name
        .split_once('.')
        .filter(|(category, _)| !category.is_empty())
        .ok_or_else(|| bad("expected <category>.<style>"))?;
    let bytes = encoded.as_bytes();
    let mut raw = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let digits = bytes
                .get(i + 1..i + 3)
                .filter(|d| {
                    d.iter()
                        .all(|b| b.is_ascii_digit() || (b'A'..=b'F').contains(b))
                })
                .ok_or_else(|| bad("a % is followed by two uppercase hex digits"))?;
            let hex = std::str::from_utf8(digits).expect("ASCII hex digits");
            raw.push(u8::from_str_radix(hex, 16).expect("two hex digits"));
            i += 3;
        } else if unreserved(bytes[i]) {
            raw.push(bytes[i]);
            i += 1;
        } else {
            return Err(bad("a byte outside A-Z a-z 0-9 - _ is written as %XX"));
        }
    }
    let style = String::from_utf8(raw).map_err(|_| bad("the style is not UTF-8"))?;
    if style_block_name(category, &style) != name {
        return Err(bad("an unreserved byte is escaped"));
    }
    Ok((category.to_owned(), style))
}

/// What the endpoint strings of a style's rows name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EndpointKey {
    /// The `name` of an atom-table row (the default).
    #[default]
    Type,
    /// The `class` of an atom-table row; every atom table carries `class`.
    Class,
    /// Nothing: the table has no endpoint columns, each row's `smirks`
    /// assigns it.
    Smirks,
}

impl EndpointKey {
    /// The document spelling (`"type"`, `"class"`, `"smirks"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Type => "type",
            Self::Class => "class",
            Self::Smirks => "smirks",
        }
    }

    /// The key a document spells `text`, or `None`.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "type" => Some(Self::Type),
            "class" => Some(Self::Class),
            "smirks" => Some(Self::Smirks),
            _ => None,
        }
    }
}

/// One entry of the document's `styles` list, as stored (borrowed from the
/// document).
#[derive(Debug, Clone)]
pub struct StyleEntry<'a> {
    /// What the style's rows parameterize (`bond`, `pair`, …).
    pub category: &'a str,
    /// The functional form within the category (`harmonic`, `lj/cut`).
    pub style: &'a str,
    /// Style-level parameters, each a finite number or a string.
    pub params: Option<&'a JsonMap<String, JsonValue>>,
    /// The energy expression, verbatim.
    pub expression: Option<&'a str>,
    /// What a row's endpoints name.
    pub endpoint_key: EndpointKey,
}

impl StyleEntry<'_> {
    /// The block name of this style's table.
    pub fn block_name(&self) -> String {
        style_block_name(self.category, self.style)
    }
}

/// The `forcefield` root section: the document and one table per style.
///
/// `document` is the section group's attribute map, verbatim and in its
/// stored key order; `tables` maps a block name to its table. Both keep what
/// this build does not interpret.
#[derive(Debug, Clone, Default)]
pub struct ForceFieldSection {
    /// The force-field document (`name`, `units`, `styles`, …).
    pub document: JsonMap<String, JsonValue>,
    /// Style tables (and unknown blocks), keyed by block name.
    pub tables: IndexMap<String, Block>,
}

fn invalid(message: String) -> MolRsError {
    MolRsError::validation(format!("forcefield: {message}"))
}

impl ForceFieldSection {
    /// The force field's `name`, when the document carries one.
    pub fn name(&self) -> Option<&str> {
        self.document.get("name").and_then(JsonValue::as_str)
    }

    /// The `styles` list, in document order (empty when absent).
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Validation`] when `styles` is not a list of style
    /// entries ([`validate`](Self::validate) says what one is).
    pub fn styles(&self) -> Result<Vec<StyleEntry<'_>>, MolRsError> {
        let Some(styles) = self.document.get("styles") else {
            return Ok(Vec::new());
        };
        let styles = styles
            .as_array()
            .ok_or_else(|| invalid(format!("styles is a list, found {styles}")))?;
        styles.iter().map(style_entry).collect()
    }

    /// The table of the `(category, style)` style, by its block name.
    pub fn table(&self, category: &str, style: &str) -> Option<&Block> {
        self.tables.get(&style_block_name(category, style))
    }

    /// Check the section against the chapter's rules (see the module docs).
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Validation`] naming the first rule broken.
    pub fn validate(&self) -> Result<(), MolRsError> {
        match self.document.get("name") {
            Some(JsonValue::String(_)) => {}
            Some(other) => return Err(invalid(format!("name is a string, found {other}"))),
            None => return Err(invalid("the document has no name".into())),
        }
        check_units(self.document.get("units"))?;
        if let Some(source) = self.document.get("source") {
            check_source(source)?;
        }
        if let Some(special) = self.document.get("special_bonds") {
            check_special_bonds(special)?;
        }

        let styles = self.styles()?;
        let mut seen = std::collections::HashSet::new();
        for style in &styles {
            if !seen.insert((style.category, style.style)) {
                return Err(invalid(format!(
                    "style {}/{} is listed twice",
                    style.category, style.style
                )));
            }
        }
        let class_keyed = styles.iter().any(|s| s.endpoint_key == EndpointKey::Class);
        for style in &styles {
            let block = style.block_name();
            let table = self.tables.get(&block).ok_or_else(|| {
                invalid(format!(
                    "style {}/{} has no table {block:?}",
                    style.category, style.style
                ))
            })?;
            check_style_table(style, &block, table, class_keyed)?;
        }
        Ok(())
    }
}

fn style_entry(value: &JsonValue) -> Result<StyleEntry<'_>, MolRsError> {
    let entry = value
        .as_object()
        .ok_or_else(|| invalid(format!("a styles entry is an object, found {value}")))?;
    let text = |key: &str| -> Result<Option<&str>, MolRsError> {
        match entry.get(key) {
            None | Some(JsonValue::Null) => Ok(None),
            Some(JsonValue::String(s)) => Ok(Some(s)),
            Some(other) => Err(invalid(format!(
                "a style's {key} is a string, found {other}"
            ))),
        }
    };
    let category = text("category")?
        .ok_or_else(|| invalid(format!("a styles entry has no category: {value}")))?;
    let mut chars = category.bytes();
    if !chars.next().is_some_and(|b| b.is_ascii_lowercase())
        || !chars.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    {
        return Err(invalid(format!(
            "category {category:?} does not match ^[a-z][a-z0-9_]*$"
        )));
    }
    let style = text("style")?
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid(format!("a {category} style has no non-empty style name")))?;
    let params = match entry.get("params") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Object(params)) => Some(params),
        Some(other) => {
            return Err(invalid(format!(
                "params of {category}/{style} is an object, found {other}"
            )));
        }
    };
    for (key, param) in params.into_iter().flatten() {
        let finite = param.as_f64().is_some_and(f64::is_finite);
        if !(finite || param.is_string()) {
            return Err(invalid(format!(
                "param {key:?} of {category}/{style} is a finite number or a string, found \
                 {param}"
            )));
        }
    }
    let param_str = |key: &str| params.and_then(|p| p.get(key)).and_then(JsonValue::as_str);
    if let Some(special) = params.and_then(|p| p.get("special"))
        && !matches!(special.as_str(), Some("lj" | "coul"))
    {
        return Err(invalid(format!(
            "params.special of {category}/{style} is \"lj\" or \"coul\", found {special}"
        )));
    }
    if let Some(mixing) = params.and_then(|p| p.get("mixing"))
        && !param_str("mixing").is_some_and(|m| MIXING_RULES.contains(&m))
    {
        return Err(invalid(format!(
            "params.mixing of {category}/{style} is one of {MIXING_RULES:?}, found {mixing}"
        )));
    }
    let endpoint_key = match text("endpoint_key")? {
        None => EndpointKey::Type,
        Some(key) => EndpointKey::parse(key).ok_or_else(|| {
            invalid(format!(
                "endpoint_key of {category}/{style} is type, class or smirks, found {key:?}"
            ))
        })?,
    };
    Ok(StyleEntry {
        category,
        style,
        params,
        expression: text("expression")?,
        endpoint_key,
    })
}

fn check_units(units: Option<&JsonValue>) -> Result<(), MolRsError> {
    let units = units
        .ok_or_else(|| invalid("the document has no units".into()))?
        .as_object()
        .ok_or_else(|| invalid("units is an object".into()))?;
    let text = |key: &str| -> Result<Option<&str>, MolRsError> {
        match units.get(key) {
            None | Some(JsonValue::Null) => Ok(None),
            Some(JsonValue::String(s)) => Ok(Some(s)),
            Some(other) => Err(invalid(format!("units.{key} is a string, found {other}"))),
        }
    };
    let preset = match text("preset")? {
        None => None,
        Some(name) => Some(
            unit_preset(name)
                .map(|table| (name, table))
                .ok_or_else(|| invalid(format!("units.preset {name:?} is no unit preset")))?,
        ),
    };
    let mut stated = 0;
    for (i, quantity) in UNIT_QUANTITIES.iter().enumerate() {
        let Some(unit) = text(quantity)? else {
            continue;
        };
        stated += 1;
        if let Some((name, table)) = preset
            && table[i] != Some(unit)
        {
            return Err(invalid(format!(
                "units.{quantity} {unit:?} disagrees with preset {name:?} ({})",
                table[i].unwrap_or("no unit")
            )));
        }
    }
    if preset.is_none() && stated == 0 {
        return Err(invalid(
            "units states a preset or at least one quantity".into(),
        ));
    }
    Ok(())
}

fn check_source(source: &JsonValue) -> Result<(), MolRsError> {
    let source = source
        .as_object()
        .ok_or_else(|| invalid(format!("source is an object, found {source}")))?;
    if !source.get("format").is_some_and(JsonValue::is_string) {
        return Err(invalid("source.format is a string".into()));
    }
    if let Some(uri) = source.get("uri")
        && !uri.is_string()
        && !uri.is_null()
    {
        return Err(invalid(format!("source.uri is a string, found {uri}")));
    }
    if let Some(sha) = source.get("sha256").filter(|v| !v.is_null()) {
        let ok = sha.as_str().is_some_and(|s| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        });
        if !ok {
            return Err(invalid(format!(
                "source.sha256 is 64 lowercase hex digits, found {sha}"
            )));
        }
    }
    Ok(())
}

fn check_special_bonds(special: &JsonValue) -> Result<(), MolRsError> {
    let special = special
        .as_object()
        .ok_or_else(|| invalid(format!("special_bonds is an object, found {special}")))?;
    for key in special.keys() {
        if key != "lj" && key != "coul" {
            return Err(invalid(format!(
                "special_bonds holds lj and coul only, found {key:?}"
            )));
        }
    }
    for key in ["lj", "coul"] {
        let weights = special.get(key).and_then(JsonValue::as_array);
        let ok = weights.is_some_and(|w| {
            w.len() == 3 && w.iter().all(|v| v.as_f64().is_some_and(f64::is_finite))
        });
        if !ok {
            return Err(invalid(format!(
                "special_bonds.{key} is three finite weights [w12, w13, w14]"
            )));
        }
    }
    Ok(())
}

fn check_style_table(
    style: &StyleEntry<'_>,
    block: &str,
    table: &Block,
    class_keyed: bool,
) -> Result<(), MolRsError> {
    let fail = |why: String| invalid(format!("table {block:?}: {why}"));
    if table.structural_shape().is_some() {
        return Err(fail("a table of types declares no structural shape".into()));
    }
    let names = table
        .get("name")
        .ok_or_else(|| fail("no name column".into()))?;
    let names = names
        .as_string()
        .filter(|_| table.validity("name").is_none())
        .ok_or_else(|| fail("name is a string column, never null".into()))?;
    let mut seen = std::collections::HashSet::with_capacity(names.len());
    if let Some(dup) = names.iter().find(|name| !seen.insert(name.as_str())) {
        return Err(fail(format!("type name {dup:?} is not unique")));
    }

    let present: Vec<&str> = ENDPOINT_COLUMNS
        .iter()
        .copied()
        .filter(|column| table.contains_key(column))
        .collect();
    match (style.endpoint_key, category_arity(style.category)) {
        (EndpointKey::Smirks, _) => {
            let smirks = table.get("smirks").and_then(|c| c.as_string());
            if !present.is_empty() || smirks.is_none() || table.validity("smirks").is_some() {
                return Err(fail(
                    "smirks-keyed: no endpoint columns, and a smirks column never null".into(),
                ));
            }
        }
        (_, Some(arity)) => {
            if present != ENDPOINT_COLUMNS[..arity] {
                return Err(fail(format!(
                    "a {} row names endpoints {:?}, found {present:?}",
                    style.category,
                    &ENDPOINT_COLUMNS[..arity]
                )));
            }
        }
        (_, None) => {
            if present != ENDPOINT_COLUMNS[..present.len()] {
                return Err(fail(format!(
                    "endpoint columns {present:?} are no prefix of itom..mtom"
                )));
            }
        }
    }
    for endpoint in &present {
        let ok = table.dtype(endpoint) == Some(DType::String) && table.validity(endpoint).is_none();
        if !ok {
            return Err(fail(format!(
                "endpoint {endpoint} is a string column, never null"
            )));
        }
    }

    for (column, values) in table.iter() {
        if column == "name" || ENDPOINT_COLUMNS.contains(&column) {
            continue;
        }
        if style.category == "cmap" && column == CMAP_GRID {
            check_cmap_grid(table, values).map_err(fail)?;
            continue;
        }
        let allowed: &[DType] = if ANNOTATION_COLUMNS.contains(&column) {
            &[DType::String]
        } else if let Some(spec) = crate::store::schema::column(column) {
            std::slice::from_ref(&spec.dtype)
        } else {
            &[DType::Float, DType::String]
        };
        let ok = allowed.contains(&values.dtype())
            && values.shape().len() == 1
            && table.precision(column).is_none();
        if !ok {
            let names: Vec<&str> = allowed.iter().map(DType::name).collect();
            return Err(fail(format!(
                "parameter {column:?} is {}[T] with no precision, found {}{:?}",
                names.join(" or "),
                values.dtype().name(),
                &values.shape()[1..]
            )));
        }
    }
    if class_keyed && style.category == "atom" && !table.contains_key("class") {
        return Err(fail(
            "a class-keyed style links through atom classes; this atom table has no class".into(),
        ));
    }
    if PAIR_CATEGORIES.contains(&style.category) && present == ENDPOINT_COLUMNS[..2] {
        check_pair_restatements(table).map_err(fail)?;
    }
    Ok(())
}

/// The trailing-axis exception: a `cmap` table's [`CMAP_GRID`] column is
/// `f64[T, N, N]`, `N ≥ 2`, with no precision, and every value of a row that
/// is not null is finite.
fn check_cmap_grid(table: &Block, values: &crate::store::block::Column) -> Result<(), String> {
    let shape = values.shape();
    let grid = values
        .as_float()
        .filter(|_| table.precision(CMAP_GRID).is_none())
        .filter(|_| matches!(shape, [_, n, m] if n == m && *n >= 2))
        .ok_or_else(|| {
            format!(
                "the cmap grid is f64[T, N, N] with N >= 2 and no precision, found {}{:?}",
                values.dtype().name(),
                &shape[1..]
            )
        })?;
    let validity = table.validity(CMAP_GRID);
    for (row, cells) in grid.outer_iter().enumerate() {
        if validity.is_some_and(|mask| !mask[row]) {
            continue;
        }
        if let Some(bad) = cells.iter().find(|v| !v.is_finite()) {
            return Err(format!(
                "the cmap grid of row {row} holds {bad}; it is finite"
            ));
        }
    }
    Ok(())
}

/// Linking rule 3: a pair table prices each unordered `{itom, jtom}` once.
///
/// Rows restating a pair, in either order, are one row when every
/// [parameter column](is_parameter_column) holds equal values in both (a null
/// equal only to a null); any other restatement is refused, naming both rows
/// and the parameters they differ in. `table` is a validated pair table: a
/// never-null string `name`, `itom` and `jtom`, every column 1-D. A
/// `smirks`-keyed table has no endpoints and is not checked.
///
/// # Errors
///
/// The reason, without the table's name (the caller prefixes it).
pub fn check_pair_restatements(table: &Block) -> Result<(), String> {
    let strings = |column: &str| {
        table
            .get(column)
            .and_then(|c| c.as_string())
            .expect("a validated pair table: a string column")
    };
    let (names, itom, jtom) = (strings("name"), strings("itom"), strings("jtom"));
    let params: Vec<(&str, &crate::store::block::Column, Option<&[bool]>)> = table
        .iter()
        .filter(|(column, _)| is_parameter_column(column))
        .map(|(column, values)| (column, values, table.validity(column)))
        .collect();
    let mut first_row = std::collections::HashMap::with_capacity(names.len());
    for row in 0..names.len() {
        let (a, b) = (itom[[row]].as_str(), jtom[[row]].as_str());
        let pair = if a <= b { (a, b) } else { (b, a) };
        let first = *first_row.entry(pair).or_insert(row);
        if first == row {
            continue;
        }
        let mut differ: Vec<&str> = params
            .iter()
            .filter(|(_, values, validity)| {
                let valid = |r: usize| validity.is_none_or(|mask| mask[r]);
                match (valid(first), valid(row)) {
                    (true, true) => !values.rows_equal(first, row),
                    (x, y) => x != y,
                }
            })
            .map(|(column, ..)| *column)
            .collect();
        if !differ.is_empty() {
            differ.sort_unstable();
            return Err(format!(
                "rows {:?} and {:?} both price the pair {{{:?}, {:?}}} and differ in {differ:?}",
                names[[first]],
                names[[row]],
                pair.0,
                pair.1
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::block::Column;
    use ndarray::ArrayD;
    use serde_json::json;

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

    fn table(columns: Vec<(&str, Column)>) -> Block {
        let mut block = Block::new();
        for (name, column) in columns {
            block.insert_column(name, column).unwrap();
        }
        block
    }

    fn base() -> ForceFieldSection {
        let document = json!({
            "name": "test",
            "units": {"preset": "real"},
            "styles": [
                {"category": "atom", "style": "full"},
                {"category": "bond", "style": "harmonic"},
            ],
        });
        let mut tables = IndexMap::new();
        tables.insert(
            "atom.full".to_owned(),
            table(vec![
                ("name", strings(&["CT", "HC"])),
                ("mass", floats(&[12.0, 1.0])),
            ]),
        );
        tables.insert(
            "bond.harmonic".to_owned(),
            table(vec![
                ("name", strings(&["CT-HC"])),
                ("itom", strings(&["CT"])),
                ("jtom", strings(&["HC"])),
                ("k", floats(&[680.0])),
                ("r0", floats(&[1.09])),
            ]),
        );
        ForceFieldSection {
            document: document.as_object().unwrap().clone(),
            tables,
        }
    }

    #[test]
    fn a_style_block_name_escapes_every_reserved_byte() {
        assert_eq!(style_block_name("bond", "harmonic"), "bond.harmonic");
        assert_eq!(
            style_block_name("pair", "lj/cut/coul/long"),
            "pair.lj%2Fcut%2Fcoul%2Flong"
        );
        assert_eq!(style_block_name("x", "a.b c"), "x.a%2Eb%20c");
        assert_eq!(style_block_name("x", "é"), "x.%C3%A9");
    }

    #[test]
    fn a_style_block_name_parses_back_and_refuses_other_spellings() {
        for (category, style) in [
            ("pair", "lj/cut/coul/long"),
            ("x", "é ü"),
            ("bond", "m-o_r"),
        ] {
            let name = style_block_name(category, style);
            assert_eq!(
                parse_style_block_name(&name).unwrap(),
                (category.to_owned(), style.to_owned())
            );
        }
        for bad in [
            "nodot",
            ".style",
            "pair.lj/cut",
            "pair.lj%2fcut",
            "pair.lj%2",
            "pair.%41",
        ] {
            assert!(parse_style_block_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_well_formed_section_validates() {
        base().validate().unwrap();
    }

    #[test]
    fn units_state_a_preset_or_a_quantity_and_agree_with_the_preset() {
        let mut ff = base();
        ff.document.insert("units".into(), json!({}));
        assert!(ff.validate().is_err());
        ff.document.insert(
            "units".into(),
            json!({"preset": "real", "energy": "kJ/mol"}),
        );
        assert!(ff.validate().is_err());
        ff.document.insert(
            "units".into(),
            json!({"preset": "real", "energy": "kcal/mol"}),
        );
        ff.validate().unwrap();
        ff.document
            .insert("units".into(), json!({"length": "nm", "energy": "kJ/mol"}));
        ff.validate().unwrap();
        ff.document.insert("units".into(), json!({"preset": "lj"}));
        ff.validate().unwrap();
        ff.document
            .insert("units".into(), json!({"preset": "lj", "length": "nm"}));
        assert!(ff.validate().is_err());
    }

    #[test]
    fn each_table_rule_is_refused() {
        let mut ff = base();
        ff.document["styles"]
            .as_array_mut()
            .unwrap()
            .push(json!({"category": "bond", "style": "harmonic"}));
        assert!(ff.validate().is_err(), "duplicate style");

        let mut ff = base();
        ff.tables.shift_remove("bond.harmonic");
        assert!(ff.validate().is_err(), "missing table");

        let mut ff = base();
        ff.tables["bond.harmonic"] = table(vec![
            ("name", strings(&["a", "a"])),
            ("itom", strings(&["CT", "CT"])),
            ("jtom", strings(&["HC", "HC"])),
        ]);
        assert!(ff.validate().is_err(), "duplicate type name");

        let mut ff = base();
        ff.tables["bond.harmonic"] =
            table(vec![("name", strings(&["a"])), ("itom", strings(&["CT"]))]);
        assert!(ff.validate().is_err(), "wrong arity");

        let mut ff = base();
        let n = Column::from_i64(ArrayD::from_shape_vec(vec![1], vec![3i64]).unwrap());
        ff.tables["bond.harmonic"].insert_column("n", n).unwrap();
        assert!(ff.validate().is_err(), "param dtype");

        let mut ff = base();
        ff.tables["bond.harmonic"]
            .set_validity("name", vec![false])
            .unwrap();
        assert!(ff.validate().is_err(), "null name");

        let mut ff = base();
        ff.document["styles"][1]["endpoint_key"] = json!("class");
        assert!(ff.validate().is_err(), "class key without class");
        ff.tables["atom.full"]
            .insert_column("class", strings(&["CT", "HC"]))
            .unwrap();
        ff.validate().unwrap();
    }

    #[test]
    fn style_params_are_scalars_with_reserved_meanings() {
        let mut ff = base();
        ff.document["styles"][1]["params"] = json!({"mixing": "geometric", "cutoff": 10.0});
        ff.validate().unwrap();
        ff.document["styles"][1]["params"] = json!({"mixing": "lorentz"});
        assert!(ff.validate().is_err());
        ff.document["styles"][1]["params"] = json!({"special": "vdw"});
        assert!(ff.validate().is_err());
        ff.document["styles"][1]["params"] = json!({"grid": [1, 2]});
        assert!(ff.validate().is_err());
    }

    #[test]
    fn an_unknown_category_takes_its_endpoint_prefix_and_an_unknown_table_is_kept() {
        let mut ff = base();
        ff.document["styles"]
            .as_array_mut()
            .unwrap()
            .push(json!({"category": "cross_term", "style": "custom"}));
        ff.tables.insert(
            "cross_term.custom".into(),
            table(vec![
                ("name", strings(&["c"])),
                ("itom", strings(&["C"])),
                ("jtom", strings(&["N"])),
                ("grid", strings(&["24x24"])),
            ]),
        );
        ff.tables.insert(
            "notes.free%20text".into(),
            table(vec![("text", strings(&["anything"]))]),
        );
        ff.validate().unwrap();
        ff.tables["cross_term.custom"]
            .insert_column("ltom", strings(&["C"]))
            .unwrap();
        assert!(ff.validate().is_err(), "itom jtom ltom is no prefix");
    }

    fn grids(shape: &[usize], values: Vec<f64>) -> Column {
        Column::from_float(ArrayD::from_shape_vec(shape.to_vec(), values).unwrap())
    }

    /// `base` with a `cmap charmm` style of two rows, whose `grid` column is
    /// `grid`.
    fn with_cmap(grid: Column) -> ForceFieldSection {
        let mut ff = base();
        ff.document["styles"]
            .as_array_mut()
            .unwrap()
            .push(json!({"category": "cmap", "style": "charmm"}));
        let mut columns = vec![("name", strings(&["c1", "c2"]))];
        for endpoint in ENDPOINT_COLUMNS {
            columns.push((endpoint, strings(&["C", "N"])));
        }
        columns.push(("grid", grid));
        ff.tables
            .insert(style_block_name("cmap", "charmm"), table(columns));
        ff
    }

    #[test]
    fn a_cmap_row_names_five_endpoints_and_its_grid_is_square() {
        assert_eq!(category_arity("cmap"), Some(5));
        let square = |n: usize| grids(&[2, n, n], vec![0.25; 2 * n * n]);
        with_cmap(square(2)).validate().unwrap();
        with_cmap(square(24)).validate().unwrap();

        let mut ff = with_cmap(square(3));
        ff.tables["cmap.charmm"].remove("mtom");
        assert!(ff.validate().is_err(), "four endpoints");
    }

    #[test]
    fn a_ragged_or_malformed_cmap_grid_is_refused() {
        for (grid, why) in [
            (grids(&[2, 3, 4], vec![0.0; 24]), "not square"),
            (grids(&[2, 1, 1], vec![0.0; 2]), "N < 2"),
            (grids(&[2, 4], vec![0.0; 8]), "one trailing axis"),
            (grids(&[2], vec![0.0; 2]), "no trailing axes"),
            (grids(&[2, 2, 2, 2], vec![0.0; 16]), "three trailing axes"),
            (strings(&["24x24", "24x24"]), "a string"),
        ] {
            let err = with_cmap(grid).validate().unwrap_err().to_string();
            assert!(err.contains("cmap grid"), "{why}: {err}");
        }
        let mut values = vec![0.0; 8];
        values[5] = f64::NAN;
        let err = with_cmap(grids(&[2, 2, 2], values.clone()))
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("row 1"), "{err}");
        // A null row's filler is not a value.
        let mut ff = with_cmap(grids(&[2, 2, 2], values));
        ff.tables["cmap.charmm"]
            .set_validity("grid", vec![true, false])
            .unwrap();
        ff.validate().unwrap();
    }

    /// Only a cmap table's `grid` carries trailing axes: another cmap column,
    /// or a `grid` anywhere else, is refused.
    #[test]
    fn a_trailing_axis_anywhere_else_is_refused() {
        let mut ff = with_cmap(grids(&[2, 2, 2], vec![0.0; 8]));
        ff.tables["cmap.charmm"]
            .insert_column("other", grids(&[2, 2, 2], vec![0.0; 8]))
            .unwrap();
        assert!(ff.validate().is_err(), "another cmap column");

        let mut ff = base();
        ff.tables["bond.harmonic"]
            .insert_column("grid", grids(&[1, 2, 2], vec![0.0; 4]))
            .unwrap();
        assert!(ff.validate().is_err(), "a bond grid");
    }

    /// `base` with a `pair lj/cut` style (category `category`) whose rows
    /// are `(name, itom, jtom, epsilon)`, `sigma` 2.0 throughout.
    fn with_pairs(category: &str, rows: &[(&str, &str, &str, f64)]) -> ForceFieldSection {
        let mut ff = base();
        ff.document["styles"]
            .as_array_mut()
            .unwrap()
            .push(json!({"category": category, "style": "lj/cut"}));
        let column = |i: usize| -> Vec<&str> { rows.iter().map(|r| [r.0, r.1, r.2][i]).collect() };
        let epsilon: Vec<f64> = rows.iter().map(|r| r.3).collect();
        ff.tables.insert(
            style_block_name(category, "lj/cut"),
            table(vec![
                ("name", strings(&column(0))),
                ("itom", strings(&column(1))),
                ("jtom", strings(&column(2))),
                ("epsilon", floats(&epsilon)),
                ("sigma", floats(&vec![2.0; rows.len()])),
            ]),
        );
        ff
    }

    fn pair_table<'a>(ff: &'a mut ForceFieldSection, category: &str) -> &'a mut Block {
        ff.tables
            .get_mut(&style_block_name(category, "lj/cut"))
            .unwrap()
    }

    #[test]
    fn a_reversed_pair_restated_with_other_params_is_refused() {
        for category in PAIR_CATEGORIES {
            let rows = [
                ("A", "A", "A", 0.1),
                ("A-B", "A", "B", 0.9),
                ("B-A", "B", "A", 0.8),
            ];
            let err = with_pairs(category, &rows)
                .validate()
                .unwrap_err()
                .to_string();
            assert!(err.contains("\"A-B\" and \"B-A\""), "{err}");
            assert!(err.contains("[\"epsilon\"]"), "{err}");
        }
    }

    /// Two names on one ordered pair are no less one pair: the last of them
    /// would otherwise silently win.
    #[test]
    fn a_pair_restated_in_the_same_order_with_other_params_is_refused() {
        let rows = [("nbfix-1", "A", "B", 0.9), ("nbfix-2", "A", "B", 0.8)];
        assert!(with_pairs("pair", &rows).validate().is_err());
        let rows = [("A", "A", "A", 0.1), ("A-again", "A", "A", 0.2)];
        assert!(with_pairs("pair", &rows).validate().is_err(), "self pair");
    }

    #[test]
    fn an_equal_restatement_is_one_row() {
        let rows = [
            ("A", "A", "A", 0.1),
            ("A-B", "A", "B", 0.9),
            ("B-A", "B", "A", 0.9),
            ("A-B-again", "A", "B", 0.9),
        ];
        with_pairs("pair", &rows).validate().unwrap();
    }

    /// A null equals a null and nothing else.
    #[test]
    fn a_null_parameter_differs_from_a_value_and_equals_a_null() {
        let rows = [("A-B", "A", "B", 0.9), ("B-A", "B", "A", 0.9)];
        let mut ff = with_pairs("pair", &rows);
        pair_table(&mut ff, "pair")
            .set_validity("epsilon", vec![true, false])
            .unwrap();
        let err = ff.validate().unwrap_err().to_string();
        assert!(err.contains("epsilon"), "{err}");
        pair_table(&mut ff, "pair")
            .set_validity("epsilon", vec![false, false])
            .unwrap();
        ff.validate().unwrap();
    }

    /// `name` and the annotation columns are no parameters; every other
    /// column, a string one included, is.
    #[test]
    fn names_and_annotations_take_no_part_but_a_string_parameter_does() {
        let rows = [("A-B", "A", "B", 0.9), ("B-A", "B", "A", 0.9)];
        let mut ff = with_pairs("pair", &rows);
        for annotation in ANNOTATION_COLUMNS {
            pair_table(&mut ff, "pair")
                .insert_column(annotation, strings(&["one", "other"]))
                .unwrap();
        }
        ff.validate().unwrap();
        pair_table(&mut ff, "pair")
            .insert_column("flavour", strings(&["one", "other"]))
            .unwrap();
        let err = ff.validate().unwrap_err().to_string();
        assert!(err.contains("[\"flavour\"]"), "{err}");
    }

    /// Only `pair` and `pair14` rows are found by their endpoints: a bond
    /// table may hold two names on one pair, and a smirks-keyed pair table
    /// has no endpoints to compare.
    #[test]
    fn other_tables_are_not_checked() {
        let mut ff = base();
        ff.tables["bond.harmonic"] = table(vec![
            ("name", strings(&["CT-HC", "HC-CT"])),
            ("itom", strings(&["CT", "HC"])),
            ("jtom", strings(&["HC", "CT"])),
            ("k", floats(&[680.0, 340.0])),
        ]);
        ff.validate().unwrap();

        let mut ff = with_pairs("pair", &[]);
        ff.document["styles"][2]["endpoint_key"] = json!("smirks");
        ff.tables[&style_block_name("pair", "lj/cut")] = table(vec![
            ("name", strings(&["n1", "n2"])),
            ("smirks", strings(&["[#6:1]", "[#6:1]"])),
            ("epsilon", floats(&[0.1, 0.2])),
        ]);
        ff.validate().unwrap();
    }

    #[test]
    fn a_smirks_keyed_table_has_no_endpoints() {
        let mut ff = base();
        ff.document["styles"][1]["endpoint_key"] = json!("smirks");
        assert!(ff.validate().is_err());
        ff.tables["bond.harmonic"] = table(vec![
            ("name", strings(&["b1"])),
            ("smirks", strings(&["[#6:1]-[#1:2]"])),
            ("k", floats(&[680.0])),
        ]);
        ff.validate().unwrap();
    }
}
