//! [`ForceField`] ↔ [`ForceFieldSection`]: a force field as the `forcefield`
//! section of a record (molrec `docs/spec/forcefield.md`).
//!
//! | molrs | section |
//! |---|---|
//! | [`ForceField::name`] | `document.name` |
//! | [`ForceField::units`] (declared, or the `real` default) | `document.units`: `{preset, length, energy, angle, charge, mass}` from the preset's table (`lj`: `{preset}` alone) |
//! | [`ForceField::declared_special_bonds`] | `document.special_bonds`; absent ⇔ `None` |
//! | [`ForceField::styles`], in order | `document.styles`, in order |
//! | [`Style::category`], [`Style::name`] | `category`, `style` |
//! | [`Style::params`] numeric / string | `params` numbers / strings |
//! | the string style params `expression`, `endpoint_key` | the entry fields of those names |
//! | [`Style::type_rows`], in definition order | the rows of the table at [`style_block_name`] |
//! | row name, endpoints (a pair always two) | `name`, `itom`…`ltom` |
//! | row [`Params`] numeric / string | `f64` / `string` columns, after `name` and the endpoints, keys sorted bytewise |
//! | a numeric param under a canonical non-`f64` key (`atomic_number`, `id`, …) | a column of that key's dtype; [`ForceField::from_section`] reads it back as `f64` |
//! | a key a row does not carry | null in that row |
//!
//! `from_section(to_section(ff))` is `ff` with its units declared, and
//! `to_section(from_section(s))` is `s` for every section `to_section`
//! produced. Units are never converted: a section in a unit system that is no
//! molrs preset is refused, not rescaled.
//!
//! [`Style::category`]: super::Style::category
//! [`Style::name`]: super::Style::name
//! [`Style::params`]: super::Style::params
//! [`Style::type_rows`]: super::Style::type_rows

use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;
use ndarray::ArrayD;
use serde_json::{Map as JsonMap, Value as JsonValue, json};

use super::{ForceField, Params, SpecialBonds, Style};
use molrs::store::block::{Block, Column, DType};
use molrs::store::forcefield_section::{
    ENDPOINT_COLUMNS, EndpointKey, ForceFieldSection, UNIT_QUANTITIES, style_block_name,
    unit_preset,
};

/// The string style params that are entry fields of `document.styles`.
const ENTRY_FIELDS: [&str; 2] = ["expression", "endpoint_key"];

/// The quantities `to_section` states beside a preset.
const STATED_QUANTITIES: [&str; 5] = ["length", "energy", "angle", "charge", "mass"];

/// One value of a row's or a style's params.
#[derive(Debug, Clone, PartialEq)]
enum Value<'a> {
    Number(f64),
    Text(&'a str),
}

/// The params of `params` as one sorted map, refusing a key that is both a
/// number and a string.
fn merged<'a>(
    params: &'a Params,
    what: &dyn Fn() -> String,
) -> Result<BTreeMap<&'a str, Value<'a>>, String> {
    let mut out: BTreeMap<&str, Value<'_>> =
        params.iter().map(|(k, v)| (k, Value::Number(v))).collect();
    for (key, text) in params.iter_strings() {
        if out.insert(key, Value::Text(text)).is_some() {
            return Err(format!(
                "{}: param {key:?} is both a number and a string",
                what()
            ));
        }
    }
    Ok(out)
}

fn units_document(units: &str) -> Result<JsonValue, String> {
    let table = unit_preset(units).ok_or_else(|| {
        format!("units {units:?} is no unit preset the forcefield section can state")
    })?;
    let mut out = JsonMap::new();
    out.insert("preset".into(), units.into());
    if units != "lj" {
        for (quantity, unit) in UNIT_QUANTITIES.iter().zip(table) {
            if let Some(unit) = unit
                && STATED_QUANTITIES.contains(quantity)
            {
                out.insert((*quantity).into(), unit.into());
            }
        }
    }
    Ok(JsonValue::Object(out))
}

fn style_entry(style: &Style) -> Result<JsonValue, String> {
    let what = || format!("{}/{} style params", style.category(), style.name());
    let mut entry = JsonMap::new();
    entry.insert("category".into(), style.category().into());
    entry.insert("style".into(), style.name().into());
    let mut params = JsonMap::new();
    let mut fields = JsonMap::new();
    for (key, value) in merged(style.params(), &what)? {
        match value {
            Value::Text(text) if ENTRY_FIELDS.contains(&key) => {
                fields.insert(key.into(), text.into());
            }
            Value::Number(_) if ENTRY_FIELDS.contains(&key) => {
                return Err(format!("{}: {key:?} is a string", what()));
            }
            Value::Number(n) => {
                let number = serde_json::Number::from_f64(n)
                    .ok_or_else(|| format!("{}: {key:?} = {n} is not finite", what()))?;
                params.insert(key.into(), JsonValue::Number(number));
            }
            Value::Text(text) => {
                params.insert(key.into(), text.into());
            }
        }
    }
    if !params.is_empty() {
        entry.insert("params".into(), JsonValue::Object(params));
    }
    if let Some(expression) = fields.remove("expression") {
        entry.insert("expression".into(), expression);
    }
    if let Some(key) = fields.remove("endpoint_key")
        && key != "type"
    {
        entry.insert("endpoint_key".into(), key);
    }
    Ok(JsonValue::Object(entry))
}

fn strings(values: Vec<String>) -> Column {
    let n = values.len();
    Column::from_string(ArrayD::from_shape_vec(vec![n], values).expect("one value per row"))
}

/// The column of one param across the rows, at the dtype it is stored at.
fn param_column(
    key: &str,
    cells: &[Option<Value<'_>>],
    what: &dyn Fn() -> String,
) -> Result<(Column, Vec<bool>), String> {
    let validity: Vec<bool> = cells.iter().map(Option::is_some).collect();
    let numeric = cells
        .iter()
        .flatten()
        .any(|v| matches!(v, Value::Number(_)));
    let text = cells.iter().flatten().any(|v| matches!(v, Value::Text(_)));
    if numeric && text {
        return Err(format!(
            "{}: param {key:?} is a number in one row and a string in another",
            what()
        ));
    }
    let canonical = molrs::store::schema::column(key).map(|spec| spec.dtype);
    let column = if text {
        if canonical.is_some_and(|dtype| dtype != DType::String) {
            return Err(format!(
                "{}: param {key:?} is a string, and the canonical key {key:?} is not",
                what()
            ));
        }
        strings(
            cells
                .iter()
                .map(|c| match c {
                    Some(Value::Text(t)) => (*t).to_owned(),
                    _ => String::new(),
                })
                .collect(),
        )
    } else {
        let numbers: Vec<f64> = cells
            .iter()
            .map(|c| match c {
                Some(Value::Number(n)) => *n,
                _ => 0.0,
            })
            .collect();
        let shape = vec![numbers.len()];
        match canonical {
            None | Some(DType::Float) => Column::from_float(
                ArrayD::from_shape_vec(shape, numbers).expect("one value per row"),
            ),
            Some(DType::UInt) => {
                let mut values = Vec::with_capacity(numbers.len());
                for (n, valid) in numbers.iter().zip(&validity) {
                    if *valid && !(n.fract() == 0.0 && *n >= 0.0 && *n < u64::MAX as f64) {
                        return Err(format!(
                            "{}: param {key:?} = {n} is no non-negative integer, and the \
                             canonical key {key:?} is u64",
                            what()
                        ));
                    }
                    values.push(*n as u64);
                }
                Column::from_uint(ArrayD::from_shape_vec(shape, values).expect("one per row"))
            }
            Some(other) => {
                return Err(format!(
                    "{}: param {key:?} is a number, and the canonical key {key:?} is {}",
                    what(),
                    other.name()
                ));
            }
        }
    };
    Ok((column, validity))
}

fn style_table(style: &Style) -> Result<Block, String> {
    let what = || format!("{}/{}", style.category(), style.name());
    let rows = style.type_rows();
    let arity = match style.category() {
        "atom" => 0,
        "bond" | "pair" => 2,
        "angle" => 3,
        _ => 4,
    };
    let mut block = Block::new();
    let column_err = |e: molrs::store::block::BlockError| format!("{}: {e}", what());
    block
        .insert_column(
            "name",
            strings(rows.iter().map(|(name, _, _)| (*name).to_owned()).collect()),
        )
        .map_err(column_err)?;
    for (position, endpoint) in ENDPOINT_COLUMNS[..arity].iter().enumerate() {
        let values = rows
            .iter()
            .map(|(_, ends, _)| ends[position].to_owned())
            .collect();
        block
            .insert_column(*endpoint, strings(values))
            .map_err(column_err)?;
    }

    let mut per_row = Vec::with_capacity(rows.len());
    let mut keys = BTreeSet::new();
    for (name, _, params) in &rows {
        let row = merged(params, &|| format!("{} type {name:?}", what()))?;
        keys.extend(row.keys().copied());
        per_row.push(row);
    }
    for key in keys {
        if key == "name" || ENDPOINT_COLUMNS.contains(&key) {
            return Err(format!(
                "{}: a param named {key:?} would shadow the table's own column",
                what()
            ));
        }
        let cells: Vec<Option<Value<'_>>> = per_row.iter().map(|r| r.get(key).cloned()).collect();
        let (column, validity) = param_column(key, &cells, &what)?;
        block.insert_column(key, column).map_err(column_err)?;
        block.set_validity(key, validity).map_err(column_err)?;
    }
    Ok(block)
}

/// The preset a section's `units` name, whether by `preset` or by stating
/// exactly one preset's length and energy (and nothing that disagrees).
fn molrs_units(units: &JsonMap<String, JsonValue>) -> Result<String, String> {
    if let Some(preset) = units.get("preset").and_then(JsonValue::as_str) {
        return Ok(preset.to_owned());
    }
    let stated: Vec<(usize, &str)> = UNIT_QUANTITIES
        .iter()
        .enumerate()
        .filter_map(|(i, q)| units.get(*q).and_then(JsonValue::as_str).map(|u| (i, u)))
        .collect();
    let states = |q: &str| stated.iter().any(|(i, _)| UNIT_QUANTITIES[*i] == q);
    if states("length") && states("energy") {
        let matching: Vec<&str> = ["real", "metal", "si", "cgs", "electron", "micro", "nano"]
            .into_iter()
            .filter(|name| {
                let table = unit_preset(name).expect("a preset");
                stated.iter().all(|(i, unit)| table[*i] == Some(*unit))
            })
            .collect();
        if let [only] = matching[..] {
            return Ok(only.to_owned());
        }
    }
    Err(format!(
        "units {} are no unit preset; molrs does not convert a force field's units",
        JsonValue::Object(units.clone())
    ))
}

fn special_bonds(value: &JsonValue) -> SpecialBonds {
    let weights = |key: &str| {
        let w = value[key].as_array().expect("validated: three weights");
        [0, 1, 2].map(|i| w[i].as_f64().expect("validated: finite weights"))
    };
    SpecialBonds {
        lj: weights("lj"),
        coul: weights("coul"),
    }
}

/// The params of row `row` of `table`: every column but `name` and the
/// endpoints, where not null.
fn row_params(table: &Block, row: usize) -> Params {
    let mut params = Params::new();
    for (key, column) in table.iter() {
        if key == "name" || ENDPOINT_COLUMNS.contains(&key) {
            continue;
        }
        if table.validity(key).is_some_and(|mask| !mask[row]) {
            continue;
        }
        if let Some(values) = column.as_string() {
            params.set_str(key, &values[[row]]);
        } else if let Some(values) = column.as_float() {
            params.set(key, values[[row]]);
        } else if let Some(values) = column.as_uint() {
            params.set(key, values[[row]] as f64);
        } else if let Some(values) = column.as_int() {
            params.set(key, f64::from(values[[row]]));
        } else if let Some(values) = column.as_i64() {
            params.set(key, values[[row]] as f64);
        }
    }
    params
}

impl ForceField {
    /// This force field as a record's `forcefield` section (see the module
    /// docs for the mapping).
    ///
    /// # Errors
    ///
    /// An `Err` naming the style and key when the force field has no section
    /// form: units that are no preset; a non-finite style param; a param key
    /// that is a number in one row and a string in another, both in one
    /// [`Params`], or `name` / an endpoint column; a param under a canonical
    /// key at another dtype (a non-integral `atomic_number`, a numeric
    /// `element`); or anything [`ForceFieldSection::validate`] refuses.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::ff::forcefield::{ForceField, Params};
    ///
    /// let mut ff = ForceField::new("example");
    /// ff.def_style("bond", "harmonic", Params::new())
    ///     .unwrap()
    ///     .def_type("A-B", &["A", "B"], Params::from_pairs(&[("k", 300.0), ("r0", 1.5)]))
    ///     .unwrap();
    /// let section = ff.to_section().unwrap();
    /// assert!(section.table("bond", "harmonic").is_some());
    /// let back = ForceField::from_section(&section).unwrap();
    /// assert_eq!(back.get_bondtypes()[0].params.get("k"), Some(300.0));
    /// ```
    pub fn to_section(&self) -> Result<ForceFieldSection, String> {
        let mut document = JsonMap::new();
        document.insert("name".into(), self.name.clone().into());
        document.insert("units".into(), units_document(self.units())?);
        if let Some(sb) = self.declared_special_bonds() {
            document.insert(
                "special_bonds".into(),
                json!({"lj": sb.lj.to_vec(), "coul": sb.coul.to_vec()}),
            );
        }
        let mut entries = Vec::with_capacity(self.styles().len());
        let mut tables = IndexMap::with_capacity(self.styles().len());
        for style in self.styles() {
            entries.push(style_entry(style)?);
            tables.insert(
                style_block_name(style.category(), style.name()),
                style_table(style)?,
            );
        }
        document.insert("styles".into(), JsonValue::Array(entries));
        let section = ForceFieldSection { document, tables };
        section.validate().map_err(|e| e.to_string())?;
        Ok(section)
    }

    /// The force field a `forcefield` section describes (see the module docs
    /// for the mapping).
    ///
    /// The section's units become the force field's declared units; they are
    /// not converted. A table no style names and a document key this build
    /// does not know are unknown content the force field has no place for:
    /// they stay with the section.
    ///
    /// # Errors
    ///
    /// An `Err` when the section fails [`ForceFieldSection::validate`], or
    /// has no [`ForceField`] form: a category outside `atom bond angle
    /// dihedral improper pair`; units that are no preset (stated by
    /// `preset`, or by one preset's own length and energy); a `smirks`-keyed
    /// style; a `class`-keyed style whose endpoints are not all atom-type
    /// names.
    pub fn from_section(section: &ForceFieldSection) -> Result<ForceField, String> {
        section.validate().map_err(|e| e.to_string())?;
        let doc = &section.document;
        let mut ff = ForceField::new(section.name().expect("validated: a name"));
        let units = doc["units"].as_object().expect("validated: units");
        ff.set_units(&molrs_units(units)?);
        if let Some(sb) = doc.get("special_bonds") {
            ff.set_special_bonds(special_bonds(sb));
        }

        let styles = section.styles().map_err(|e| e.to_string())?;
        let atom_names: BTreeSet<&str> = styles
            .iter()
            .filter(|s| s.category == "atom")
            .filter_map(|s| section.tables.get(&s.block_name()))
            .filter_map(|t| t.get("name").and_then(Column::as_string))
            .flat_map(|names| names.iter().map(String::as_str))
            .collect();
        for entry in &styles {
            let what = format!("{}/{}", entry.category, entry.style);
            let mut params = Params::new();
            for (key, value) in entry.params.into_iter().flatten() {
                match value {
                    JsonValue::String(text) => params.set_str(key, text),
                    number => params.set(key, number.as_f64().expect("validated: a number")),
                }
            }
            if let Some(expression) = entry.expression {
                params.set_str("expression", expression);
            }
            let table = &section.tables[&entry.block_name()];
            match entry.endpoint_key {
                EndpointKey::Type => {}
                EndpointKey::Smirks => {
                    return Err(format!(
                        "{what} is smirks-keyed; a molrs force field keys rows by atom type"
                    ));
                }
                EndpointKey::Class => {
                    for endpoint in ENDPOINT_COLUMNS {
                        let Some(values) = table.get(endpoint).and_then(Column::as_string) else {
                            continue;
                        };
                        if let Some(class) = values
                            .iter()
                            .find(|v| !v.is_empty() && !atom_names.contains(v.as_str()))
                        {
                            return Err(format!(
                                "{what} is class-keyed and its endpoint {class:?} is no atom \
                                 type; a molrs force field keys rows by atom type"
                            ));
                        }
                    }
                    params.set_str("endpoint_key", EndpointKey::Class.as_str());
                }
            }
            let style = ff
                .def_style(entry.category, entry.style, params)
                .map_err(|e| format!("{what}: {e}"))?;
            let names = table
                .get("name")
                .and_then(Column::as_string)
                .expect("validated: a name column");
            let endpoint_columns: Vec<&ArrayD<String>> = ENDPOINT_COLUMNS
                .iter()
                .filter_map(|c| table.get(c).and_then(Column::as_string))
                .collect();
            for (row, name) in names.iter().enumerate() {
                let endpoints: Vec<&str> = endpoint_columns
                    .iter()
                    .map(|column| column[[row]].as_str())
                    .collect();
                style
                    .def_type(name, &endpoints, row_params(table, row))
                    .map_err(|e| format!("{what}: {e}"))?;
            }
        }
        Ok(ff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::readers::ForceFieldReader;
    use crate::ff::forcefield::tests::assert_same_definitions;
    use crate::ff::typifier::Typifier;

    /// `ff` with its units declared: what `from_section(to_section(ff))`
    /// gives back.
    fn declared(ff: &ForceField) -> ForceField {
        let mut out = ff.clone();
        out.set_units(ff.units());
        out
    }

    fn same_section(a: &ForceFieldSection, b: &ForceFieldSection, what: &str) {
        assert_eq!(a.document, b.document, "{what}: document");
        assert_eq!(
            a.tables.keys().collect::<Vec<_>>(),
            b.tables.keys().collect::<Vec<_>>(),
            "{what}: tables"
        );
        for (name, x) in &a.tables {
            let y = &b.tables[name];
            assert_eq!(format!("{x:?}"), format!("{y:?}"), "{what}: {name} layout");
            for (key, column) in x.iter() {
                let other = y.get(key).unwrap();
                assert_eq!(x.validity(key), y.validity(key), "{what}: {name}.{key}");
                match (column.as_float(), other.as_float()) {
                    (Some(p), Some(q)) => {
                        let bits =
                            |a: &ArrayD<f64>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                        assert_eq!(bits(p), bits(q), "{what}: {name}.{key}");
                    }
                    _ => assert_eq!(
                        (column.as_string(), column.as_uint()),
                        (other.as_string(), other.as_uint()),
                        "{what}: {name}.{key}"
                    ),
                }
            }
        }
    }

    /// The round-trip property, both ways: `from_section ∘ to_section` is the
    /// identity up to declared units, and `to_section ∘ from_section` the
    /// identity on the section. Through a `*.mrec` store too, where it can be
    /// written.
    fn round_trips(ff: &ForceField, what: &str) {
        let section = ff
            .to_section()
            .unwrap_or_else(|e| panic!("{what}: to_section: {e}"));
        let back = ForceField::from_section(&section)
            .unwrap_or_else(|e| panic!("{what}: from_section: {e}"));
        assert_same_definitions(&declared(ff), &back);
        let again = back.to_section().unwrap();
        same_section(&section, &again, what);

        #[cfg(feature = "filesystem")]
        {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("ff.mrec");
            molrs::io::mrec::write_forcefield_file(&path, &section, None).unwrap();
            let stored = molrs::io::mrec::read_forcefield_file(&path)
                .unwrap()
                .unwrap();
            assert_eq!(stored.document, section.document, "{what}: stored document");
            let from_store = ForceField::from_section(&stored)
                .unwrap_or_else(|e| panic!("{what}: from the store: {e}"));
            assert_same_definitions(&declared(ff), &from_store);
        }
    }

    #[test]
    fn the_built_in_libraries_round_trip() {
        use crate::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier};
        use crate::ff::typifier::{
            AtdParameterSet, AtdTypifier, ElementTypifier, GaffParameterSet, GaffTypifier,
            OPLSAATypifier, UFFTypifier,
        };
        round_trips(OPLSAATypifier::oplsaa().library(), "OPLS-AA");
        round_trips(GaffTypifier::new(GaffParameterSet::Gaff).library(), "GAFF");
        round_trips(
            GaffTypifier::new(GaffParameterSet::Gaff2).library(),
            "GAFF2",
        );
        for set in [
            AtdParameterSet::Bcc,
            AtdParameterSet::Abcg2,
            AtdParameterSet::Gas,
            AtdParameterSet::Gff,
        ] {
            round_trips(AtdTypifier::new(set).library(), &format!("ATD {set:?}"));
        }
        round_trips(MMFF94Typifier::new().library(), "MMFF94");
        round_trips(MMFF94STypifier::new().library(), "MMFF94s");
        round_trips(UFFTypifier::new().library(), "UFF");
        round_trips(ElementTypifier::new().library(), "element");
    }

    #[test]
    fn a_lammps_read_force_field_round_trips() {
        let text = "special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333\n\
                    pair_style lj/cut/coul/long 10.0 10.0\n\
                    pair_modify mix arithmetic\n\
                    pair_coeff c3 c3 0.107800 3.397710\n\
                    pair_coeff oh oh 0.093000 3.242871\n\
                    bond_style harmonic\n\
                    bond_coeff c3-oh 314.1 1.426\n\
                    angle_style harmonic\n\
                    angle_coeff c3-c3-oh 76.79 109.66\n\
                    dihedral_style fourier\n\
                    dihedral_coeff c3-c3-oh-ho 2 0.16 3 0.0 0.25 1 0.0\n\
                    improper_style harmonic\n\
                    improper_coeff c3-oh-c3-c3 1.1 180.0\n";
        let ff = crate::ff::forcefield::readers::lammps::LammpsFfReader::new()
            .read_str(text)
            .unwrap();
        round_trips(&ff, "LAMMPS");
    }

    #[test]
    fn a_gromacs_read_force_field_round_trips() {
        let text = "[ defaults ]\n1 2 yes 0.5 0.8333\n\
                    [ atomtypes ]\n\
                    CT CT 6 12.011 -0.18 A 0.35 0.276144\n\
                    HC HC 1 1.008 0.06 A 0.25 0.12552\n\
                    [ bondtypes ]\nCT HC 1 0.109 284512.0\n\
                    [ angletypes ]\nHC CT HC 1 107.8 276.144\n\
                    [ dihedraltypes ]\nX CT CT X 1 0.0 0.6276 3\n";
        let ff = crate::ff::forcefield::readers::gromacs::GromacsTopFfReader::new()
            .read_str(text)
            .unwrap();
        // The GROMACS bond_type is the atom table's `class`.
        let section = ff.to_section().unwrap();
        let atoms = section.table("atom", "full").unwrap();
        assert_eq!(atoms.dtype("class"), Some(DType::String));
        assert_eq!(atoms.dtype("atomic_number"), Some(DType::UInt));
        // `X` is the empty-string wildcard.
        let dihedrals = section.table("dihedral", "periodic").unwrap();
        let itom = dihedrals.get("itom").and_then(Column::as_string).unwrap();
        assert_eq!(itom[[0]], "");
        round_trips(&ff, "GROMACS");
    }

    #[test]
    fn an_openmm_read_force_field_round_trips_with_its_string_params() {
        let xml = r#"<ForceField name="mini" combining_rule="geometric">
  <AtomTypes>
    <Type name="opls_135" class="CT" element="C" mass="12.011" def="[C;X4](C)(H)(H)H" desc="alkane CH3"/>
    <Type name="opls_140" class="HC" element="H" mass="1.008" def="H[C;X4]" overrides="opls_135"/>
  </AtomTypes>
  <HarmonicBondForce>
    <Bond class1="CT" class2="HC" length="0.109" k="284512.0"/>
  </HarmonicBondForce>
  <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
    <Atom type="opls_135" charge="-0.18" sigma="0.35" epsilon="0.276144"/>
    <Atom type="opls_140" charge="0.06" sigma="0.25" epsilon="0.12552"/>
  </NonbondedForce>
</ForceField>"#;
        let ff = crate::ff::forcefield::readers::opls::OplsXmlReader::new()
            .read_str(xml)
            .unwrap();
        let section = ff.to_section().unwrap();
        let atoms = section.table("atom", "full").unwrap();
        for annotation in ["class", "element", "smarts"] {
            assert_eq!(atoms.dtype(annotation), Some(DType::String), "{annotation}");
        }
        assert!(atoms.validity("overrides").is_some(), "absent on opls_135");
        round_trips(&ff, "OpenMM XML");
    }

    #[test]
    fn style_level_strings_become_entry_fields_and_mixing_a_param() {
        let mut ff = ForceField::new("custom");
        let mut lj = Params::from_pairs(&[("cutoff", 10.0)]);
        lj.set_str("mixing", "geometric");
        ff.def_style("pair", "lj/cut", lj).unwrap();
        let mut fene = Params::new();
        fene.set_str("expression", "-0.5*K*R0^2*log(1-(r/R0)^2)");
        ff.def_style("bond", "fene", fene).unwrap();
        let section = ff.to_section().unwrap();
        let styles = &section.document["styles"];
        assert_eq!(
            styles[0]["params"],
            json!({"cutoff": 10.0, "mixing": "geometric"})
        );
        assert_eq!(
            styles[1]["expression"],
            json!("-0.5*K*R0^2*log(1-(r/R0)^2)")
        );
        assert!(styles[1].get("params").is_none());
        assert_eq!(
            section.document["units"],
            json!({"preset": "real", "length": "angstrom", "energy": "kcal/mol",
                   "angle": "radian", "charge": "e", "mass": "dalton"})
        );
        round_trips(&ff, "custom");
    }

    #[test]
    fn a_section_molrs_cannot_hold_is_refused_not_approximated() {
        let mut base = ForceField::new("t");
        base.def_style("bond", "harmonic", Params::new()).unwrap();
        let section = base.to_section().unwrap();

        let mut nm = section.clone();
        nm.document["units"] = json!({"length": "nm", "energy": "kJ/mol"});
        assert!(ForceField::from_section(&nm).unwrap_err().contains("units"));

        let mut stated = section.clone();
        stated.document["units"] = json!({"length": "angstrom", "energy": "eV"});
        assert_eq!(ForceField::from_section(&stated).unwrap().units(), "metal");

        let mut cmap = section.clone();
        cmap.document["styles"] = json!([{"category": "cmap", "style": "charmm"}]);
        let mut grid = Block::new();
        grid.insert_column("name", strings(vec![])).unwrap();
        cmap.tables.insert("cmap.charmm".into(), grid);
        assert!(
            ForceField::from_section(&cmap)
                .unwrap_err()
                .contains("cmap")
        );

        let mut smirks = section;
        smirks.document["styles"][0]["endpoint_key"] = json!("smirks");
        let mut rows = Block::new();
        rows.insert_column("name", strings(vec!["b1".into()]))
            .unwrap();
        rows.insert_column("smirks", strings(vec!["[#6:1]-[#1:2]".into()]))
            .unwrap();
        smirks.tables.insert("bond.harmonic".into(), rows);
        assert!(
            ForceField::from_section(&smirks)
                .unwrap_err()
                .contains("smirks")
        );
    }

    #[test]
    fn a_force_field_without_a_section_form_is_refused_by_name() {
        let mut ff = ForceField::new("t");
        let mut a = Params::from_pairs(&[("k", 1.0)]);
        a.set_str("k", "one");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("A-B", &["A", "B"], a)
            .unwrap();
        assert!(ff.to_section().unwrap_err().contains("\"k\""));

        let mut ff = ForceField::new("t");
        let mut odd = Params::new();
        odd.set_str("w", "x");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("A-B", &["A", "B"], Params::from_pairs(&[("w", 1.0)]))
            .unwrap()
            .def_type("A-C", &["A", "C"], odd)
            .unwrap();
        assert!(ff.to_section().unwrap_err().contains("\"w\""));

        let mut ff = ForceField::new("t");
        ff.set_units("furlongs");
        assert!(ff.to_section().unwrap_err().contains("furlongs"));
    }
}
