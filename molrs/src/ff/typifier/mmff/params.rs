//! MMFF94 typing data: per-type atom properties.

use std::collections::HashMap;

use crate::ff::forcefield::xml::{attr_u32, forcefield_root};

/// MMFF94 atom property row (from `<AtomProperties>` in the XML).
#[derive(Debug, Clone)]
pub struct MMFFAtomProp {
    pub type_id: u32,
    pub atno: u32,
    pub crd: u32,
    pub val: u32,
    pub pilp: u32,
    pub mltb: u32,
    pub arom: u32,
    pub linh: u32,
    pub sbmb: u32,
}

/// Parsed MMFF typing metadata: atom properties indexed by type id.
///
/// Separate from [`ForceField`](crate::ff::forcefield::ForceField) because these
/// are typing metadata, not potential parameters. Loaded from the same XML but
/// used only during topology classification (e.g. the `sbmb` flag drives MMFF
/// bond-type assignment), not during energy evaluation.
#[derive(Debug, Clone)]
pub struct MMFFParams {
    /// Atom properties indexed by type_id.
    pub(crate) props: HashMap<u32, MMFFAtomProp>,
}

impl MMFFParams {
    /// Create a new `MMFFParams` from pre-parsed atom properties.
    pub fn new(props: HashMap<u32, MMFFAtomProp>) -> Self {
        Self { props }
    }

    /// Look up atom property by type_id.
    pub fn get_prop(&self, type_id: u32) -> Option<&MMFFAtomProp> {
        self.props.get(&type_id)
    }
}

/// Parse [`MMFFParams`] from the `<AtomProperties>` section of an MMFF XML
/// string — the typing half of a caller's MMFF XML (the potential half is
/// [`read_forcefield_xml_str`](crate::ff::forcefield::xml::read_forcefield_xml_str)).
pub(super) fn read_params_xml_str(xml: &str) -> Result<MMFFParams, String> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| format!("XML parse error: {}", e))?;

    let root = forcefield_root(&doc)?;

    let mut props = HashMap::new();

    for child in root.children().filter(|n| n.is_element()) {
        if child.tag_name().name() == "AtomProperties" {
            for prop_node in child
                .children()
                .filter(|n| n.is_element() && n.tag_name().name() == "Prop")
            {
                let p = parse_atom_prop(&prop_node)?;
                props.insert(p.type_id, p);
            }
        }
    }

    if props.is_empty() {
        return Err("No <AtomProperties> found in XML".to_string());
    }

    Ok(MMFFParams::new(props))
}

fn parse_atom_prop(node: &roxmltree::Node) -> Result<MMFFAtomProp, String> {
    Ok(MMFFAtomProp {
        type_id: attr_u32(node, "type")?,
        atno: attr_u32(node, "atno")?,
        crd: attr_u32(node, "crd")?,
        val: attr_u32(node, "val")?,
        pilp: attr_u32(node, "pilp")?,
        mltb: attr_u32(node, "mltb")?,
        arom: attr_u32(node, "arom")?,
        linh: attr_u32(node, "linh")?,
        sbmb: attr_u32(node, "sbmb")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MMFF atom-property section, from a caller's XML.
    #[test]
    fn test_mmff_params_xml() {
        let xml = r#"
        <ForceField name="MMFF94">
          <AtomProperties>
            <Prop type="1" atno="6" crd="4" val="4" pilp="0" mltb="0" arom="0" linh="0" sbmb="0" />
          </AtomProperties>
        </ForceField>
        "#;

        let params = read_params_xml_str(xml).unwrap();
        let p = params.get_prop(1).expect("type 1 is in the parsed table");
        assert_eq!((p.atno, p.crd, p.val), (6, 4, 4));
    }
}
