//! Attribute reading shared by the force-field XML readers
//! ([`read_molrs_xml_forcefield`](crate::io::read_molrs_xml_forcefield),
//! [`read_mmff_xml_forcefield`](crate::io::read_mmff_xml_forcefield) and the
//! OpenMM XML typing reader): the `<ForceField>` root, required and optional
//! attributes, and the numeric attributes of an element.

/// The `<ForceField>` root of a parsed document, or why it is not one.
pub(crate) fn forcefield_root<'a, 'i>(
    doc: &'a roxmltree::Document<'i>,
) -> Result<roxmltree::Node<'a, 'i>, String> {
    let root = doc.root_element();
    if root.tag_name().name() != "ForceField" {
        return Err(format!(
            "Root element must be <ForceField>, got <{}>",
            root.tag_name().name()
        ));
    }
    Ok(root)
}

pub(crate) fn attr_str<'a>(node: &'a roxmltree::Node, name: &str) -> Result<&'a str, String> {
    node.attribute(name)
        .ok_or_else(|| format!("<{}> missing attribute '{}'", node.tag_name().name(), name))
}

pub(crate) fn attr_f64(node: &roxmltree::Node, name: &str) -> Result<f64, String> {
    let s = attr_str(node, name)?;
    s.parse::<f64>().map_err(|e| {
        format!(
            "<{}> attribute '{}' = {:?}: {}",
            node.tag_name().name(),
            name,
            s,
            e
        )
    })
}

pub(crate) fn attr_u32(node: &roxmltree::Node, name: &str) -> Result<u32, String> {
    let s = attr_str(node, name)?;
    s.parse::<u32>().map_err(|e| {
        format!(
            "<{}> attribute '{}' = {:?}: {}",
            node.tag_name().name(),
            name,
            s,
            e
        )
    })
}

pub(crate) fn opt_attr_f64(node: &roxmltree::Node, name: &str) -> Option<f64> {
    node.attribute(name).and_then(|s| s.parse::<f64>().ok())
}

pub(crate) fn children_named<'a>(
    parent: &'a roxmltree::Node<'a, 'a>,
    tag: &'a str,
) -> impl Iterator<Item = roxmltree::Node<'a, 'a>> {
    parent
        .children()
        .filter(move |n| n.is_element() && n.tag_name().name() == tag)
}

pub(crate) fn numeric_attrs<'a>(node: &'a roxmltree::Node, skip: &[&str]) -> Vec<(&'a str, f64)> {
    let mut result = Vec::new();
    for attr in node.attributes() {
        if skip.contains(&attr.name()) {
            continue;
        }
        if let Ok(v) = attr.value().parse::<f64>() {
            result.push((attr.name(), v));
        }
    }
    result
}
