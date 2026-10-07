//! The MMFF parameter-set XML, as `scripts/mmff_to_xml.py` writes it.

use crate::ff::forcefield::{ForceField, Params, SpecialBonds};
use crate::ff::params::mmff::MmffProp;
use crate::ff::params::mmff::encode_da;
use crate::ff::typifier::mmff::MmffAtomProperties;
use crate::io::molrs_xml::parse_style_element;
use crate::io::xml_attribute::{attr_f64, attr_u32, children_named, forcefield_root, opt_attr_f64};

/// Read an MMFF [`ForceField`] from an MMFF parameter-set XML file.
pub fn read_mmff_xml_forcefield(path: &str) -> Result<ForceField, String> {
    let xml = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
    read_mmff_xml_forcefield_str(&xml)
}

/// Read an MMFF [`ForceField`] from MMFF parameter-set XML text: the declared
/// styles, `<VdWParams>` and `<ElectrostaticParams>`.
///
/// Only the two sections MMFF actually resolves from type rows are tables; its
/// bonded terms are `ParamSource::PerInstance` styles (the typifier bakes their
/// parameters into Frame columns), so they are *declared*, through the molrs
/// force-field XML's style elements, and carry no rows:
///
/// ```xml
/// <ForceField name="MMFF94">
///   <BondStyle name="mmff_bond" />                       <!-- per-instance: no rows -->
///   <VdWParams B="0.2" Beta="12.0" DARAD="0.8" DAEPS="0.5">
///     <VdW type="1" alpha="1.05" n_eff="2.49" a_i="3.89" g_i="1.282" da="-" />
///   </VdWParams>                                         <!-- a real 95-row table -->
///   <ElectrostaticParams coulomb="332.0716" dielectric="1.0" delta="0.05" scale14="0.75" />
///   <AtomProperties> … </AtomProperties>                 <!-- the typing half -->
/// </ForceField>
/// ```
///
/// `<ElectrostaticParams>` declares the **generic** buffered-Coulomb pair style
/// `coul/cut` — `E = coulomb·qᵢqⱼ / (dielectric·(r + delta))`. MMFF owns no
/// electrostatic kernel; the section above is a *parameterization* of that one,
/// and `delta = 0` degenerates it into the textbook Coulomb.
///
/// This reads the force-field half; [`read_mmff_xml_params_str`] reads the
/// typing half (`<AtomProperties>`), and `Mmff94Typifier::from_parts` takes
/// the two.
///
/// # Errors
///
/// The root is not `<ForceField>`, an attribute is missing, or an element is
/// none of the layout's.
pub fn read_mmff_xml_forcefield_str(xml: &str) -> Result<ForceField, String> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| format!("XML parse error: {}", e))?;
    let root = forcefield_root(&doc)?;
    let mut ff = ForceField::new(root.attribute("name").unwrap_or("unnamed"));
    for child in root.children().filter(|n| n.is_element()) {
        match child.tag_name().name() {
            "VdWParams" => parse_mmff_vdw(&mut ff, &child)?,
            "ElectrostaticParams" => parse_electrostatics(&mut ff, &child)?,
            // The typing half and the informational tables: not force-field rows.
            "AtomTypes"
            | "AtomProperties"
            | "EquivalenceTable"
            | "BondChargeIncrements"
            | "PartialBondChargeIncrements"
            | "DefaultStretchBend"
            | "EmpiricalBondRules" => {}
            other => {
                if !parse_style_element(&mut ff, &child)? {
                    return Err(format!(
                        "<{other}> is not an element of the MMFF parameter-set XML"
                    ));
                }
            }
        }
    }
    Ok(ff)
}

fn parse_mmff_vdw(ff: &mut ForceField, node: &roxmltree::Node) -> Result<(), String> {
    let mut style_params: Vec<(&str, f64)> = Vec::new();
    for attr_name in ["B", "Beta", "DARAD", "DAEPS"] {
        if let Some(v) = opt_attr_f64(node, attr_name) {
            style_params.push((attr_name, v));
        }
    }

    let style = ff
        .def_style("pair", "mmff_vdw", Params::from_pairs(&style_params))
        .map_err(|e| e.to_string())?;

    for vdw in children_named(node, "VdW") {
        let atype = attr_f64(&vdw, "type")?;
        let alpha = attr_f64(&vdw, "alpha")?;
        let n_eff = attr_f64(&vdw, "n_eff")?;
        let a_i = attr_f64(&vdw, "a_i")?;
        let g_i = attr_f64(&vdw, "g_i")?;
        // MMFF's `DA` column ("D" donor / "A" acceptor / "-" neither). It selects
        // the two hydrogen-bond corrections in the combining rule (donor R*
        // suppression, donor-acceptor R*/eps scaling) — dropping it silently
        // over-estimates the vdW energy of every H-bonding molecule (urea +0.59,
        // NMA +0.12 kcal/mol). Absent attribute => "-", MMFF's own default.
        let da = encode_da(vdw.attribute("da").unwrap_or("-"));

        let type_name = format!("{}", atype as u32);
        style
            .def_type(
                &type_name,
                &[&type_name],
                Params::from_pairs(&[
                    ("alpha", alpha),
                    ("n_eff", n_eff),
                    ("a_i", a_i),
                    ("g_i", g_i),
                    ("da", da),
                ]),
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Parse `<ElectrostaticParams coulomb="332.0716" dielectric="1.0" delta="0.05"
/// scale14="0.75"/>` into the **generic** `pair/coul/cut` style and the force
/// field's 1-4 weights.
///
/// ```text
/// E = coulomb * qi*qj / (dielectric * (R + delta))
/// ```
///
/// The term has no per-type rows at all — the charges are per-atom and the typifier
/// bakes them onto the frame — so the style carries only these scalars. That is
/// exactly why it once went missing: the kernel had been in the registry since it
/// was written, but no force field ever *defined* the style, so no consumer of the
/// documented `ForceField` API could compute electrostatics (caffeine was off by
/// 150 kcal/mol). Declaring it as a section keeps the numbers in the data file,
/// symmetric with `<VdWParams>`, rather than conjuring a style out of literals
/// inside the reader.
///
/// # `coulomb` is required
///
/// The Coulomb constant is **force-field data**: MMFF uses Halgren's 332.0716 and
/// OPLS/LAMMPS use CODATA's 332.06371, a difference above the RDKit parity
/// tolerance. Both are correct — the force field decides — so neither this reader
/// nor the kernel may pick one. A section that does not state it is an error, not a
/// silent default. (`delta` and `dielectric` keep genuine semantic defaults here:
/// "no buffer" and "vacuum" are meaningful things for a data file to leave unsaid,
/// and stating them is still what every shipped force field does.)
///
/// `scale14` also sets [`SpecialBonds`]: coulomb 1-4 is scaled by it (0.75 for MMFF)
/// while **vdW 1-4 is left unscaled (1.0)** — MMFF scales only electrostatics, and
/// its torsion parameters were fitted against unscaled 1-4 vdW. The 1-2 / 1-3
/// weights stay at the default 0.0 (molrs excludes those pairs by omitting them from
/// the neighbour list).
fn parse_electrostatics(ff: &mut ForceField, node: &roxmltree::Node) -> Result<(), String> {
    let coulomb = attr_f64(node, "coulomb")?;
    let dielectric = opt_attr_f64(node, "dielectric").unwrap_or(1.0);
    let delta = opt_attr_f64(node, "delta").unwrap_or(0.0);
    let scale14 = opt_attr_f64(node, "scale14").unwrap_or(1.0);

    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, 1.0],
        coul: [0.0, 0.0, scale14],
    });
    ff.def_style(
        "pair",
        "coul/cut",
        Params::from_pairs(&[
            ("coulomb", coulomb),
            ("dielectric", dielectric),
            ("delta", delta),
        ]),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Parse [`MmffAtomProperties`] from the `<AtomProperties>` section of an MMFF XML
/// string — the typing half of a caller's MMFF XML (the potential half is
/// [`read_mmff_xml_forcefield_str`]); `Mmff94Typifier::from_parts` takes the two.
pub fn read_mmff_xml_params_str(xml: &str) -> Result<MmffAtomProperties, String> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| format!("XML parse error: {}", e))?;

    let root = forcefield_root(&doc)?;

    let mut props = Vec::new();

    for child in root.children().filter(|n| n.is_element()) {
        if child.tag_name().name() == "AtomProperties" {
            for prop_node in child
                .children()
                .filter(|n| n.is_element() && n.tag_name().name() == "Prop")
            {
                props.push(parse_atom_prop(&prop_node)?);
            }
        }
    }

    if props.is_empty() {
        return Err("No <AtomProperties> found in XML".to_string());
    }

    Ok(MmffAtomProperties::new(props))
}

fn parse_atom_prop(node: &roxmltree::Node) -> Result<MmffProp, String> {
    let byte = |name: &str| -> Result<u8, String> {
        let v = attr_u32(node, name)?;
        u8::try_from(v).map_err(|_| format!("<Prop {name}=\"{v}\">: MMFF's tables hold 0..=255"))
    };
    Ok(MmffProp {
        atom_type: byte("type")?,
        atno: byte("atno")?,
        crd: byte("crd")?,
        val: byte("val")?,
        pilp: byte("pilp")?,
        mltb: byte("mltb")?,
        arom: byte("arom")?,
        linh: byte("linh")?,
        sbmb: byte("sbmb")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The four unit tests that lived here drove the MMFF type-def readers —
    // `<BondStretchParams>`, `<AngleBendParams>`, `<TorsionParams>`,
    // `<OutOfPlaneParams>` — and asserted the rows they produced. Those readers,
    // and the 4,065 rows they parsed across the shipped parameter files, are
    // deleted: no kernel ever read one of them. MMFF's bonded styles are
    // `ParamSource::PerInstance` (their parameters are resolved per interaction by
    // the typifier and baked into Frame columns), so they are declared through the
    // generic `<BondStyle>` / `<AngleStyle>` / `<DihedralStyle>` / `<ImproperStyle>`
    // elements with no `<Type>` children — which `test_mmff_per_instance_styles`
    // below covers.
    //
    // `<VdWParams>` is a real 95-row table and keeps its reader and its test.

    /// MMFF's per-instance styles are DECLARED, and carry no type rows.
    #[test]
    fn test_mmff_per_instance_styles() {
        let xml = r#"
        <ForceField name="MMFF94">
          <BondStyle name="mmff_bond" />
          <AngleStyle name="mmff_angle" />
          <AngleStyle name="mmff_stbn" />
          <DihedralStyle name="mmff_torsion" />
          <ImproperStyle name="mmff_oop" />
        </ForceField>
        "#;

        let ff = read_mmff_xml_forcefield_str(xml).unwrap();
        assert_eq!(ff.get_styles("bond").len(), 1);
        assert_eq!(ff.get_styles("angle").len(), 2);
        assert_eq!(ff.get_styles("dihedral")[0].name(), "mmff_torsion");
        assert_eq!(ff.get_styles("improper")[0].name(), "mmff_oop");

        // Declared, but table-free: every parameter comes from the Frame.
        assert!(ff.get_bondtypes().is_empty());
        assert!(ff.get_angletypes().is_empty());
        assert!(ff.get_dihedraltypes().is_empty());
        assert!(ff.get_impropertypes().is_empty());
    }

    #[test]
    fn test_mmff_vdw() {
        let xml = r#"
        <ForceField name="MMFF94">
          <VdWParams B="0.12" Beta="12.0" DARAD="0.8" DAEPS="0.5">
            <VdW type="1" alpha="1.050" n_eff="2.490" a_i="3.890" g_i="1.282" />
          </VdWParams>
        </ForceField>
        "#;

        let ff = read_mmff_xml_forcefield_str(xml).unwrap();
        let style = ff.get_style("pair", "mmff_vdw").unwrap();
        assert_eq!(style.params().get("B"), Some(0.12));
        assert_eq!(style.params().get("Beta"), Some(12.0));

        let types = ff.get_pairtypes();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].params.get("alpha"), Some(1.05));
    }

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

        let params = read_mmff_xml_params_str(xml).unwrap();
        let p = params.get(1).expect("type 1 is in the parsed table");
        assert_eq!((p.atno, p.crd, p.val), (6, 4, 4));
    }
}
