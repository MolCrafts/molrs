//! OpenMM-style force-field XML writer — the inverse of
//! [`OplsXmlReader`](crate::ff::forcefield::readers::opls::OplsXmlReader).
//!
//! The schema is OpenMM's, and so are the units: lengths in **nm**, energies in
//! **kJ/mol**, angles and phases in **radians**. molrs stores Å, kcal/mol and
//! radians, so every length and energy is converted at this boundary — the
//! exact inverse of the reader's table (bond `k` × 4.184 × 100, angle `k` and
//! every torsion / pair energy × 4.184, lengths ÷ 10). OPLS dihedrals stored as
//! the 4-cosine `k1..k4` are written as Ryckaert–Bellemans `c0..c5` (GROMACS
//! Eqs. 200–201); periodic terms use `k{m}/periodicity{m}/phase{m}`.

use super::ForceFieldWriter;
use crate::ff::forcefield::{ForceField, StyleDefs};

/// kcal/mol → kJ/mol.
const KJ_PER_KCAL: f64 = 4.184;
/// Å → nm.
const NM_PER_ANGSTROM: f64 = 0.1;

/// Writer for OpenMM-style `<ForceField>` XML.
#[derive(Debug, Clone)]
pub struct XmlForceFieldWriter {
    pub precision: usize,
}

impl Default for XmlForceFieldWriter {
    fn default() -> Self {
        Self { precision: 6 }
    }
}

impl XmlForceFieldWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }

    fn fmt_f(&self, v: f64) -> String {
        format!("{:.*}", self.precision, v)
    }

    fn esc(&self, s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('"', "&quot;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    }
}

impl ForceFieldWriter for XmlForceFieldWriter {
    fn write_str(&self, ff: &ForceField) -> Result<String, String> {
        let mut out = String::from("<?xml version='1.0' encoding='utf-8'?>\n");
        out.push_str(&format!(
            "<ForceField name=\"{}\">\n",
            self.esc(if ff.name.is_empty() {
                "MolPy"
            } else {
                &ff.name
            })
        ));

        // AtomTypes
        let mut atoms_xml = String::new();
        for style in ff.get_styles("atom") {
            let StyleDefs::Atom(types) = &style.defs else {
                continue;
            };
            let mut sorted: Vec<_> = types.iter().collect();
            sorted.sort_by(|a, b| a.name.cmp(&b.name));
            for t in sorted {
                let mut attrs = Vec::new();
                let type_ = t
                    .params
                    .iter_strings()
                    .find(|(k, _)| *k == "type_")
                    .map(|(_, v)| v)
                    .unwrap_or(t.name.as_str());
                let class_ = t
                    .params
                    .iter_strings()
                    .find(|(k, _)| *k == "class_")
                    .map(|(_, v)| v);
                if type_ != "*" {
                    attrs.push(format!("name=\"{}\"", self.esc(type_)));
                }
                if let Some(c) = class_
                    && c != "*"
                {
                    attrs.push(format!("class=\"{}\"", self.esc(c)));
                }
                for (xml_key, kw_key) in [
                    ("element", "element"),
                    ("mass", "mass"),
                    ("def", "def_"),
                    ("desc", "desc"),
                    ("doi", "doi"),
                    ("overrides", "overrides"),
                ] {
                    if let Some(v) = t
                        .params
                        .iter_strings()
                        .find(|(k, _)| *k == kw_key)
                        .map(|(_, v)| v)
                    {
                        attrs.push(format!("{xml_key}=\"{}\"", self.esc(v)));
                    } else if let Some(v) = t.params.get(kw_key) {
                        attrs.push(format!("{xml_key}=\"{}\"", self.fmt_f(v)));
                    }
                }
                if attrs.iter().all(|a| !a.starts_with("name=")) {
                    attrs.insert(0, format!("name=\"{}\"", self.esc(&t.name)));
                }
                atoms_xml.push_str(&format!("    <Type {}/>\n", attrs.join(" ")));
            }
        }
        if !atoms_xml.is_empty() {
            out.push_str("  <AtomTypes>\n");
            out.push_str(&atoms_xml);
            out.push_str("  </AtomTypes>\n");
        }

        // Bonds
        for style in ff.get_styles("bond") {
            if style.name != "harmonic" && !style.name.is_empty() {
                continue;
            }
            let StyleDefs::Bond(types) = &style.defs else {
                continue;
            };
            if types.is_empty() {
                continue;
            }
            out.push_str("  <HarmonicBondForce>\n");
            for bt in types {
                // Å → nm; kcal/mol/Å² → kJ/mol/nm².
                let r0 = bt.params.get("r0").unwrap_or(0.0) * NM_PER_ANGSTROM;
                let k = bt.params.get("k").unwrap_or(0.0) * KJ_PER_KCAL
                    / (NM_PER_ANGSTROM * NM_PER_ANGSTROM);
                out.push_str(&format!(
                    "    <Bond class1=\"{}\" class2=\"{}\" length=\"{}\" k=\"{}\"/>\n",
                    self.esc(&bt.itom),
                    self.esc(&bt.jtom),
                    self.fmt_f(r0),
                    self.fmt_f(k)
                ));
            }
            out.push_str("  </HarmonicBondForce>\n");
        }

        // Angles
        for style in ff.get_styles("angle") {
            if style.name != "harmonic" && !style.name.is_empty() {
                continue;
            }
            let StyleDefs::Angle(types) = &style.defs else {
                continue;
            };
            if types.is_empty() {
                continue;
            }
            out.push_str("  <HarmonicAngleForce>\n");
            for at in types {
                let theta0 = at.params.get("theta0").unwrap_or(0.0);
                let k = at.params.get("k").unwrap_or(0.0) * KJ_PER_KCAL;
                out.push_str(&format!(
                    "    <Angle class1=\"{}\" class2=\"{}\" class3=\"{}\" angle=\"{}\" k=\"{}\"/>\n",
                    self.esc(&at.itom),
                    self.esc(&at.jtom),
                    self.esc(&at.ktom),
                    self.fmt_f(theta0),
                    self.fmt_f(k)
                ));
            }
            out.push_str("  </HarmonicAngleForce>\n");
        }

        // Dihedrals: opls → RB, else Periodic
        for style in ff.get_styles("dihedral") {
            let StyleDefs::Dihedral(types) = &style.defs else {
                continue;
            };
            if types.is_empty() {
                continue;
            }
            if style.name == "opls" {
                out.push_str("  <RBTorsionForce>\n");
                for dt in types {
                    let mut attrs = format!(
                        "class1=\"{}\" class2=\"{}\" class3=\"{}\" class4=\"{}\"",
                        self.esc(&dt.itom),
                        self.esc(&dt.jtom),
                        self.esc(&dt.ktom),
                        self.esc(&dt.ltom)
                    );
                    for (i, c) in opls_to_rb(&dt.params).iter().enumerate() {
                        attrs.push_str(&format!(" c{i}=\"{}\"", self.fmt_f(*c)));
                    }
                    out.push_str(&format!("    <Proper {attrs}/>\n"));
                }
                out.push_str("  </RBTorsionForce>\n");
            } else {
                out.push_str("  <PeriodicTorsionForce>\n");
                for dt in types {
                    let mut attrs = format!(
                        "class1=\"{}\" class2=\"{}\" class3=\"{}\" class4=\"{}\"",
                        self.esc(&dt.itom),
                        self.esc(&dt.jtom),
                        self.esc(&dt.ktom),
                        self.esc(&dt.ltom)
                    );
                    // canonical per-term keys k{m}/periodicity{m}/phase{m}
                    for m in 1..10 {
                        let k = dt.params.get(&format!("k{m}"));
                        let n = dt.params.get(&format!("periodicity{m}"));
                        let d = dt.params.get(&format!("phase{m}"));
                        match (k, n, d) {
                            (Some(k), Some(n), Some(d)) => {
                                attrs.push_str(&format!(
                                    " periodicity{m}=\"{}\" k{m}=\"{}\" phase{m}=\"{}\"",
                                    n as i64,
                                    self.fmt_f(k * KJ_PER_KCAL),
                                    self.fmt_f(d)
                                ));
                            }
                            _ => break,
                        }
                    }
                    // single-term k/periodicity/phase
                    if !attrs.contains("periodicity1=")
                        && let (Some(k), Some(n), Some(d)) = (
                            dt.params.get("k"),
                            dt.params.get("periodicity"),
                            dt.params.get("phase"),
                        )
                    {
                        attrs.push_str(&format!(
                            " periodicity1=\"{}\" k1=\"{}\" phase1=\"{}\"",
                            n as i64,
                            self.fmt_f(k * KJ_PER_KCAL),
                            self.fmt_f(d)
                        ));
                    }
                    out.push_str(&format!("    <Proper {attrs}/>\n"));
                }
                out.push_str("  </PeriodicTorsionForce>\n");
            }
        }

        // Impropers periodic
        for style in ff.get_styles("improper") {
            let StyleDefs::Improper(types) = &style.defs else {
                continue;
            };
            if types.is_empty() {
                continue;
            }
            out.push_str("  <PeriodicImproperForce>\n");
            for it in types {
                let k = it.params.get("k").unwrap_or(0.0) * KJ_PER_KCAL;
                let n = it.params.get("periodicity").unwrap_or(0.0);
                let d = it.params.get("phase").unwrap_or(0.0);
                out.push_str(&format!(
                    "    <Improper class1=\"{}\" class2=\"{}\" class3=\"{}\" class4=\"{}\" periodicity1=\"{}\" k1=\"{}\" phase1=\"{}\"/>\n",
                    self.esc(&it.itom),
                    self.esc(&it.jtom),
                    self.esc(&it.ktom),
                    self.esc(&it.ltom),
                    n as i64,
                    self.fmt_f(k),
                    self.fmt_f(d)
                ));
            }
            out.push_str("  </PeriodicImproperForce>\n");
        }

        // Nonbonded
        for style in ff.get_styles("pair") {
            if !(style.name.contains("lj") || style.name.is_empty()) {
                continue;
            }
            let StyleDefs::Pair(types) = &style.defs else {
                continue;
            };
            if types.is_empty() {
                continue;
            }
            let coul14 = ff.special_bonds().coul_14();
            let lj14 = ff.special_bonds().lj_14();
            out.push_str(&format!(
                "  <NonbondedForce coulomb14scale=\"{}\" lj14scale=\"{}\">\n",
                self.fmt_f(coul14),
                self.fmt_f(lj14)
            ));
            for pt in types {
                let eps = pt.params.get("epsilon").unwrap_or(0.0) * KJ_PER_KCAL;
                let sig = pt.params.get("sigma").unwrap_or(0.0) * NM_PER_ANGSTROM;
                let chg = pt.params.get("charge").unwrap_or(0.0);
                out.push_str(&format!(
                    "    <Atom type=\"{}\" charge=\"{}\" sigma=\"{}\" epsilon=\"{}\"/>\n",
                    self.esc(&pt.itom),
                    self.fmt_f(chg),
                    self.fmt_f(sig),
                    self.fmt_f(eps)
                ));
            }
            out.push_str("  </NonbondedForce>\n");
        }

        out.push_str("</ForceField>\n");
        Ok(out)
    }
}

/// OPLS 4-cosine `k1..k4` (kcal/mol) → Ryckaert–Bellemans `c0..c5` (kJ/mol):
/// the inverse of the reader's `rb_to_opls` (GROMACS Eqs. 200–201). A type
/// that already carries `c0..c5` (in kcal/mol) is passed through converted.
fn opls_to_rb(params: &crate::ff::forcefield::Params) -> [f64; 6] {
    if params.get("k1").is_none() {
        let mut c = [0.0; 6];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = params.get(&format!("c{i}")).unwrap_or(0.0) * KJ_PER_KCAL;
        }
        return c;
    }
    let f = |k: &str| params.get(k).unwrap_or(0.0) * KJ_PER_KCAL;
    let (f1, f2, f3, f4) = (f("k1"), f("k2"), f("k3"), f("k4"));
    [
        f2 + 0.5 * (f1 + f3),
        0.5 * (-f1 + 3.0 * f3),
        -f2 + 4.0 * f4,
        -2.0 * f3,
        -4.0 * f4,
        0.0,
    ]
}

pub fn write_forcefield_xml(path: &str, ff: &ForceField, precision: usize) -> Result<(), String> {
    XmlForceFieldWriter::new()
        .with_precision(precision)
        .write(ff, path)
}

pub fn write_forcefield_xml_str(ff: &ForceField, precision: usize) -> Result<String, String> {
    XmlForceFieldWriter::new()
        .with_precision(precision)
        .write_str(ff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::xml::read_forcefield_xml_str;
    use crate::ff::forcefield::{ForceField, Params, Style};

    fn style<'a>(ff: &'a ForceField, category: &str) -> &'a Style {
        ff.styles()
            .iter()
            .find(|s| s.category() == category)
            .expect(category)
    }

    fn type_params(style: &Style, name: &str) -> Params {
        style
            .defs
            .collect_type_params()
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p)
            .expect(name)
    }

    fn small_ff() -> ForceField {
        let mut ff = ForceField::new("tiny");
        ff.def_atomstyle("full")
            .def_type("CT", &[("mass", 12.011), ("charge", -0.18)]);
        ff.def_bondstyle("harmonic")
            .def_type("CT-CT", &[("k", 268.0), ("r0", 1.529)]);
        ff.def_anglestyle("harmonic")
            .def_type("CT-CT-CT", &[("k", 58.35), ("theta0", 1.9670)]);
        ff
    }

    #[test]
    fn what_is_written_reads_back_with_the_same_styles_and_parameters() {
        let ff = small_ff();
        let xml = write_forcefield_xml_str(&ff, 6).unwrap();
        let back = read_forcefield_xml_str(&xml).unwrap();
        assert_eq!(back.name, "tiny");
        let bond = style(&back, "bond");
        assert_eq!(bond.name, "harmonic");
        let bt = type_params(bond, "CT-CT");
        assert!((bt.get("k").unwrap() - 268.0).abs() < 1e-9);
        assert!((bt.get("r0").unwrap() - 1.529).abs() < 1e-9);
        let at = type_params(style(&back, "angle"), "CT-CT-CT");
        assert!((at.get("theta0").unwrap() - 1.9670).abs() < 1e-9);
    }

    #[test]
    fn opls_torsions_and_pairs_round_trip_through_the_openmm_units() {
        let mut ff = ForceField::new("opls");
        ff.def_dihedralstyle("opls").def_type(
            "CT-CT-CT-CT",
            &[("k1", 1.3), ("k2", -0.05), ("k3", 0.2), ("k4", 0.0)],
        );
        ff.def_pairstyle("lj/cut", &[]).def_type(
            "opls_135",
            &[("charge", -0.18), ("sigma", 3.5), ("epsilon", 0.066)],
        );
        let xml = write_forcefield_xml_str(&ff, 8).unwrap();
        let back = read_forcefield_xml_str(&xml).unwrap();
        let dt = type_params(style(&back, "dihedral"), "CT-CT-CT-CT");
        for (key, want) in [("k1", 1.3), ("k2", -0.05), ("k3", 0.2), ("k4", 0.0)] {
            assert!((dt.get(key).unwrap() - want).abs() < 1e-6, "{key}");
        }
        let lj = back
            .styles()
            .iter()
            .find(|s| s.category() == "pair" && s.name == "lj/cut")
            .expect("lj/cut");
        let StyleDefs::Pair(types) = &lj.defs else {
            panic!("lj/cut holds pair types");
        };
        let pt = types
            .iter()
            .find(|t| t.itom == "opls_135")
            .expect("opls_135");
        assert!((pt.params.get("sigma").unwrap() - 3.5).abs() < 1e-6);
        assert!((pt.params.get("epsilon").unwrap() - 0.066).abs() < 1e-6);
    }

    #[test]
    fn the_precision_bounds_the_written_decimals() {
        let ff = small_ff();
        // r0 = 1.529 Å is written as 0.1529 nm.
        let coarse = write_forcefield_xml_str(&ff, 2).unwrap();
        assert!(coarse.contains("length=\"0.15\""), "{coarse}");
        let fine = write_forcefield_xml_str(&ff, 4).unwrap();
        assert!(fine.contains("length=\"0.1529\""), "{fine}");
    }
}
