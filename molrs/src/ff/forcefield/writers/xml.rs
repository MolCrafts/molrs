//! OpenMM-style force-field XML writer (inverse of molrs XML / OPLS readers).
//!
//! Emits molrs store units as stored (Å, kcal/mol, radians) into the native
//! molrs/OpenMM-ish schema used by [`crate::ff::forcefield::xml`]. Angle
//! equilibria / phases are written in **radians** (store form). OPLS dihedrals
//! write `c0..c5` when present; periodic terms use `k{m}/n{m}/d{m}`.

use super::ForceFieldWriter;
use crate::ff::forcefield::{ForceField, StyleDefs};

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
                let r0 = bt.params.get("r0").unwrap_or(0.0);
                let k = bt.params.get("k").unwrap_or(0.0);
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
                let k = at.params.get("k").unwrap_or(0.0);
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
                    for i in 0..6 {
                        let key = format!("c{i}");
                        let v = dt.params.get(&key).unwrap_or(0.0);
                        attrs.push_str(&format!(" c{i}=\"{}\"", self.fmt_f(v)));
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
                    // k{m}/n{m}/d{m} or periodicity{m}/k{m}/phase{m}
                    for m in 1..10 {
                        let k = dt.params.get(&format!("k{m}"));
                        let n = dt
                            .params
                            .get(&format!("n{m}"))
                            .or_else(|| dt.params.get(&format!("periodicity{m}")));
                        let d = dt
                            .params
                            .get(&format!("d{m}"))
                            .or_else(|| dt.params.get(&format!("phase{m}")));
                        match (k, n, d) {
                            (Some(k), Some(n), Some(d)) => {
                                attrs.push_str(&format!(
                                    " periodicity{m}=\"{}\" k{m}=\"{}\" phase{m}=\"{}\"",
                                    n as i64,
                                    self.fmt_f(k),
                                    self.fmt_f(d)
                                ));
                            }
                            _ => break,
                        }
                    }
                    // single-term k/n/d
                    if !attrs.contains("periodicity1=")
                        && let (Some(k), Some(n), Some(d)) =
                            (dt.params.get("k"), dt.params.get("n"), dt.params.get("d"))
                    {
                        attrs.push_str(&format!(
                            " periodicity1=\"{}\" k1=\"{}\" phase1=\"{}\"",
                            n as i64,
                            self.fmt_f(k),
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
                let k = it.params.get("k").unwrap_or(0.0);
                let n = it.params.get("n").unwrap_or(0.0);
                let d = it.params.get("d").unwrap_or(0.0);
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
            let coul14 = style.params.get("coulomb14scale").unwrap_or(0.5);
            let lj14 = style.params.get("lj14scale").unwrap_or(0.5);
            out.push_str(&format!(
                "  <NonbondedForce coulomb14scale=\"{}\" lj14scale=\"{}\">\n",
                self.fmt_f(coul14),
                self.fmt_f(lj14)
            ));
            for pt in types {
                let eps = pt.params.get("epsilon").unwrap_or(0.0);
                let sig = pt.params.get("sigma").unwrap_or(0.0);
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
