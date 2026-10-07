//! The molrs force-field XML: a [`ForceField`] written style by style.
//!
//! One element per style, `<BondStyle>`, `<AngleStyle>`, `<DihedralStyle>`,
//! `<ImproperStyle>` or `<PairStyle>`, named by its `name` attribute; a pair
//! style's other numeric attributes are its style parameters. A `<Type>`'s
//! `name` is stored verbatim and its endpoints are its `class1` … `class4`
//! attributes (one or two for a pair style); its other numeric attributes are
//! its parameters:
//!
//! ```xml
//! <ForceField name="TIP3P">
//!   <BondStyle name="harmonic">
//!     <Type name="OW-HW" class1="OW" class2="HW" k="450.0" r0="0.9572" />
//!   </BondStyle>
//! </ForceField>
//! ```
//!
//! [`read_molrs_xml_forcefield`] reads this layout and nothing else: an element
//! of another layout (OpenMM's `<HarmonicBondForce>`, MMFF's `<VdWParams>`) is
//! an error naming the door that reads it. [`write_molrs_xml_forcefield`] is
//! its inverse: it writes what the reader reads, and refuses by name what the
//! layout cannot hold (another category, a string or array parameter, a
//! declared unit system or special-bonds weights, a style parameter of a
//! bonded style).

use std::fmt::Write as _;

use crate::ff::forcefield::{ForceField, Params, Style};
use crate::io::xml_attribute::{attr_str, children_named, forcefield_root, numeric_attrs};

/// The style elements of the layout, by category.
const STYLE_ELEMENTS: [(&str, &str); 5] = [
    ("bond", "BondStyle"),
    ("angle", "AngleStyle"),
    ("dihedral", "DihedralStyle"),
    ("improper", "ImproperStyle"),
    ("pair", "PairStyle"),
];

/// Read a [`ForceField`] from a molrs force-field XML file.
pub fn read_molrs_xml_forcefield(path: &str) -> Result<ForceField, String> {
    let xml = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
    read_molrs_xml_forcefield_str(&xml)
}

/// Read a [`ForceField`] from molrs force-field XML text.
///
/// # Errors
///
/// The root is not `<ForceField>`, an element is not a style element of this
/// layout, an attribute is missing, or a `<Type>` does not fit its category.
pub fn read_molrs_xml_forcefield_str(xml: &str) -> Result<ForceField, String> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| format!("XML parse error: {}", e))?;
    let root = forcefield_root(&doc)?;
    let mut ff = ForceField::new(root.attribute("name").unwrap_or("unnamed"));
    for child in root.children().filter(|n| n.is_element()) {
        if !read_style_element(&mut ff, &child)? {
            return Err(format!(
                "<{}> is not an element of the molrs force-field XML (read an OpenMM \
                 force field with read_openmm_xml_forcefield, an MMFF parameter set with \
                 read_mmff_xml_forcefield)",
                child.tag_name().name()
            ));
        }
    }
    Ok(ff)
}

/// Define the style `node` states on `ff`, when `node` is a style element of
/// this layout; `Ok(false)` when it is not one.
pub(crate) fn read_style_element(
    ff: &mut ForceField,
    node: &roxmltree::Node,
) -> Result<bool, String> {
    let tag = node.tag_name().name();
    let Some((category, _)) = STYLE_ELEMENTS.iter().find(|(_, t)| *t == tag) else {
        return Ok(false);
    };
    if *category == "pair" {
        parse_generic_pair_style(ff, node)?;
    } else {
        parse_generic_style(ff, node, category)?;
    }
    Ok(true)
}

pub(crate) fn parse_generic_style(
    ff: &mut ForceField,
    node: &roxmltree::Node,
    category: &str,
) -> Result<(), String> {
    let style_name = attr_str(node, "name")?;
    let style = ff
        .def_style(category, style_name, Params::new())
        .map_err(|e| e.to_string())?;
    parse_generic_types(style, node)
}

pub(crate) fn parse_generic_pair_style(
    ff: &mut ForceField,
    node: &roxmltree::Node,
) -> Result<(), String> {
    let style_name = attr_str(node, "name")?;
    let style_params = numeric_attrs(node, &["name"]);
    let style = ff
        .def_style("pair", style_name, Params::from_pairs(&style_params))
        .map_err(|e| e.to_string())?;
    parse_generic_types(style, node)
}

/// The endpoint attributes of a generic `<Type>`, in order.
const ENDPOINT_ATTRS: [&str; 4] = ["class1", "class2", "class3", "class4"];

/// Define every `<Type>` child of a generic style element on `style`.
///
/// The name is the `name` attribute, stored verbatim; the endpoints are the
/// `class1` … `class4` attributes present, in order (none for an atom style,
/// one or two for a pair style). A name is never split into endpoints: a
/// `<Type>` whose endpoint count does not fit the category is an error.
fn parse_generic_types(style: &mut Style, node: &roxmltree::Node) -> Result<(), String> {
    let mut skip = vec!["name"];
    skip.extend(ENDPOINT_ATTRS);
    for type_node in children_named(node, "Type") {
        let name = attr_str(&type_node, "name")?;
        let endpoints: Vec<&str> = ENDPOINT_ATTRS
            .iter()
            .map_while(|attr| type_node.attribute(*attr))
            .collect();
        let params = numeric_attrs(&type_node, &skip);
        style
            .def_type(name, &endpoints, Params::from_pairs(&params))
            .map_err(|e| format!("<Type name={name:?}>: {e}"))?;
    }
    Ok(())
}

/// Write `ff` as a molrs force-field XML file.
pub fn write_molrs_xml_forcefield(path: &str, ff: &ForceField) -> Result<(), String> {
    let text = write_molrs_xml_forcefield_str(ff)?;
    std::fs::write(path, text).map_err(|e| format!("write {}: {}", path, e))
}

/// Write `ff` as molrs force-field XML text — the inverse of
/// [`read_molrs_xml_forcefield_str`].
///
/// # Errors
///
/// What the layout cannot hold, by name: a style of a category other than
/// bond, angle, dihedral, improper and pair; a style parameter of a bonded
/// style; a string or array parameter; a declared unit system or declared
/// special-bonds weights.
pub fn write_molrs_xml_forcefield_str(ff: &ForceField) -> Result<String, String> {
    if let Some(units) = ff.declared_units() {
        return Err(format!(
            "the molrs force-field XML holds no unit system; `{}` declares `{units}`",
            ff.name
        ));
    }
    if ff.declared_special_bonds().is_some() {
        return Err(format!(
            "the molrs force-field XML holds no special-bonds weights; `{}` declares them",
            ff.name
        ));
    }
    let mut out = String::new();
    writeln!(out, "<ForceField name=\"{}\">", escape(&ff.name)).unwrap();
    for style in ff.styles() {
        write_style(&mut out, style)?;
    }
    out.push_str("</ForceField>\n");
    Ok(out)
}

fn write_style(out: &mut String, style: &Style) -> Result<(), String> {
    let category = style.category();
    let Some((_, tag)) = STYLE_ELEMENTS.iter().find(|(c, _)| *c == category) else {
        return Err(format!(
            "the molrs force-field XML has no element for {category} style `{}`",
            style.name()
        ));
    };
    let what = format!("{category} style `{}`", style.name());
    let mut head = format!("  <{tag} name=\"{}\"", escape(style.name()));
    if category == "pair" {
        head.push_str(&numeric_attributes(style.params(), &what)?);
    } else if style.params().iter().next().is_some() || has_non_numeric(style.params()) {
        return Err(format!(
            "{what}: a bonded style's parameters have no place in the molrs force-field XML"
        ));
    }
    let rows = style.type_rows();
    if rows.is_empty() {
        writeln!(out, "{head} />").unwrap();
        return Ok(());
    }
    writeln!(out, "{head}>").unwrap();
    for (name, endpoints, params) in rows {
        let mut line = format!("    <Type name=\"{}\"", escape(name));
        for (i, endpoint) in endpoints.iter().enumerate() {
            write!(line, " class{}=\"{}\"", i + 1, escape(endpoint)).unwrap();
        }
        line.push_str(&numeric_attributes(
            params,
            &format!("{what}, type `{name}`"),
        )?);
        writeln!(out, "{line} />").unwrap();
    }
    writeln!(out, "  </{tag}>").unwrap();
    Ok(())
}

fn has_non_numeric(params: &Params) -> bool {
    params.iter_strings().next().is_some() || params.iter_arrays().next().is_some()
}

/// ` key="value"` for every number of `params`, by key, in a form that reads
/// back to the same `f64`; a string or array parameter is an error.
fn numeric_attributes(params: &Params, what: &str) -> Result<String, String> {
    if let Some((key, _)) = params.iter_strings().next() {
        return Err(format!(
            "{what}: string parameter `{key}` has no place in the molrs force-field XML"
        ));
    }
    if let Some((key, _)) = params.iter_arrays().next() {
        return Err(format!(
            "{what}: array parameter `{key}` has no place in the molrs force-field XML"
        ));
    }
    let mut numbers: Vec<(&str, f64)> = params.iter().collect();
    numbers.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::new();
    for (key, value) in numbers {
        write!(out, " {key}=\"{value:?}\"").unwrap();
    }
    Ok(out)
}

/// `text` with the five XML special characters escaped.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_bond_style() {
        let xml = r#"
        <ForceField name="test">
          <BondStyle name="harmonic">
            <Type name="CT-OH" class1="CT" class2="OH" k="300.0" r0="1.4" />
            <Type name="CT-CT" class1="CT" class2="CT" k="268.0" r0="1.529" />
          </BondStyle>
        </ForceField>
        "#;

        let ff = read_molrs_xml_forcefield_str(xml).unwrap();
        assert_eq!(ff.name, "test");
        assert_eq!(ff.get_bondtypes().len(), 2);

        let style = ff.get_style("bond", "harmonic").unwrap();
        let bt = style.get_bondtype("CT", "OH").unwrap();
        assert_eq!(bt.params.get("k"), Some(300.0));
        assert_eq!(bt.params.get("r0"), Some(1.4));
    }

    #[test]
    fn test_generic_angle_style() {
        let xml = r#"
        <ForceField name="test">
          <AngleStyle name="harmonic">
            <Type name="HW-OW-HW" class1="HW" class2="OW" class3="HW" k="55.0" theta0="104.52" />
          </AngleStyle>
        </ForceField>
        "#;

        let ff = read_molrs_xml_forcefield_str(xml).unwrap();
        let types = ff.get_angletypes();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].params.get("theta0"), Some(104.52));
    }

    #[test]
    fn test_generic_pair_style() {
        let xml = r#"
        <ForceField name="test">
          <PairStyle name="lj/cut" cutoff="10.0">
            <Type name="CT" class1="CT" epsilon="0.066" sigma="3.5" />
          </PairStyle>
        </ForceField>
        "#;

        let ff = read_molrs_xml_forcefield_str(xml).unwrap();
        let style = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(style.params().get("cutoff"), Some(10.0));
        assert_eq!(ff.get_pairtypes().len(), 1);
    }

    #[test]
    fn test_generic_dihedral_style() {
        let xml = r#"
        <ForceField name="test">
          <DihedralStyle name="opls">
            <Type name="HC-CT-CT-HC" class1="HC" class2="CT" class3="CT" class4="HC" k1="0.0" k2="0.0" k3="0.3" />
          </DihedralStyle>
        </ForceField>
        "#;

        let ff = read_molrs_xml_forcefield_str(xml).unwrap();
        let styles = ff.get_styles("dihedral");
        assert_eq!(styles.len(), 1);
    }

    /// The endpoints are the `class` attributes, never the name: a bond named
    /// `anything` on `CT`, `OH` holds `CT`, `OH`, and the numeric-looking
    /// `class1="1"` is an endpoint, not a param.
    #[test]
    fn generic_type_endpoints_are_the_class_attributes() {
        let xml = r#"
        <ForceField name="test">
          <BondStyle name="harmonic">
            <Type name="anything" class1="CT" class2="OH" k="300.0" r0="1.4" />
            <Type name="0_1_5" class1="1" class2="5" k="4.258" r0="1.5" />
          </BondStyle>
        </ForceField>
        "#;

        let ff = read_molrs_xml_forcefield_str(xml).unwrap();
        let style = ff.get_style("bond", "harmonic").unwrap();
        assert_eq!(
            style.type_endpoints("anything"),
            Some(vec!["CT".to_string(), "OH".to_string()])
        );
        assert_eq!(
            style.type_endpoints("0_1_5"),
            Some(vec!["1".to_string(), "5".to_string()])
        );
        let bt = style.get_bondtype("5", "1").unwrap();
        assert_eq!(bt.params.get("class1"), None);
        assert_eq!(bt.params.get("k"), Some(4.258));
    }

    /// A generic bonded `<Type>` without its `class` attributes is an error
    /// naming the type: the name `CT-OH` is not read as endpoints.
    #[test]
    fn generic_type_without_class_attributes_is_an_error_naming_it() {
        let xml = r#"
        <ForceField name="test">
          <BondStyle name="harmonic">
            <Type name="CT-OH" k="300.0" r0="1.4" />
          </BondStyle>
        </ForceField>
        "#;

        let err = read_molrs_xml_forcefield_str(xml).unwrap_err();
        assert!(err.contains("CT-OH"), "{err}");
    }

    #[test]
    fn test_invalid_root() {
        let xml = r#"<NotForceField name="test" />"#;
        let err = read_molrs_xml_forcefield_str(xml).unwrap_err();
        assert!(err.contains("Root element must be <ForceField>"));
    }

    #[test]
    fn test_missing_attribute() {
        let xml = r#"
        <ForceField name="test">
          <BondStyle>
            <Type name="A-B" class1="A" class2="B" k0="1.0" />
          </BondStyle>
        </ForceField>
        "#;
        let err = read_molrs_xml_forcefield_str(xml).unwrap_err();
        assert!(err.contains("missing attribute 'name'"));
    }

    /// An element of another layout is refused by name, not skipped.
    #[test]
    fn an_openmm_element_is_refused_naming_its_door() {
        let xml = r#"<ForceField name="t"><HarmonicBondForce /></ForceField>"#;
        let err = read_molrs_xml_forcefield_str(xml).unwrap_err();
        assert!(
            err.contains("HarmonicBondForce") && err.contains("read_openmm_xml_forcefield"),
            "{err}"
        );
    }

    /// The writer is the reader's inverse: what it writes reads back equal.
    #[test]
    fn write_then_read_is_the_same_force_field() {
        let xml = r#"
        <ForceField name="round &amp; trip">
          <BondStyle name="harmonic">
            <Type name="CT-OH" class1="CT" class2="OH" k="300.0" r0="1.4" />
          </BondStyle>
          <AngleStyle name="harmonic">
            <Type name="HW-OW-HW" class1="HW" class2="OW" class3="HW" k="55.0" theta0="104.52" />
          </AngleStyle>
          <PairStyle name="lj/cut" cutoff="10.0">
            <Type name="CT" class1="CT" epsilon="0.066" sigma="3.5" />
          </PairStyle>
        </ForceField>
        "#;
        let ff = read_molrs_xml_forcefield_str(xml).unwrap();
        let text = write_molrs_xml_forcefield_str(&ff).unwrap();
        let back = read_molrs_xml_forcefield_str(&text).unwrap();
        assert_eq!(back.name, "round & trip");
        assert_eq!(write_molrs_xml_forcefield_str(&back).unwrap(), text);
        let bt = back
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("OH", "CT")
            .unwrap();
        assert_eq!(bt.params.get("r0"), Some(1.4));
        assert_eq!(
            back.get_style("pair", "lj/cut")
                .unwrap()
                .params()
                .get("cutoff"),
            Some(10.0)
        );
    }

    /// What the layout cannot hold is refused by name.
    #[test]
    fn the_writer_refuses_what_the_layout_cannot_hold() {
        let mut ff = ForceField::new("t");
        ff.def_style("bond", "harmonic", Params::from_pairs(&[("k0", 1.0)]))
            .unwrap();
        let err = write_molrs_xml_forcefield_str(&ff).unwrap_err();
        assert!(err.contains("bond style `harmonic`"), "{err}");

        let mut ff = ForceField::new("t");
        ff.set_units("metal");
        let err = write_molrs_xml_forcefield_str(&ff).unwrap_err();
        assert!(err.contains("metal"), "{err}");
    }
}
