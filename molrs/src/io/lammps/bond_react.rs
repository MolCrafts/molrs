//! LAMMPS `fix bond/react` (REACTER) input files.
//!
//! A `fix bond/react` reaction is a pre-reaction template, the same atoms
//! after the reaction, and a *map file* pairing them: which two atoms initiate
//! the reaction, which template atoms sit on the template's edge (bonded to
//! atoms the template leaves out), which atoms the reaction deletes, and the
//! pre ↔ post equivalence of every atom. [`BondReactTemplate`] holds one
//! reaction; [`write_lammps_bond_react_map`] writes its map file, and
//! [`write_lammps_bond_react_system`] writes the whole file set a reactive run
//! reads:
//!
//! - `{stem}.data` — the system ([`write_lammps_data`](super::lammps_data::write_lammps_data));
//! - `{name}_pre.mol` / `{name}_post.mol` — the templates, as LAMMPS molecule
//!   files ([`write_lammps_molecule`]);
//! - `{name}.map` — the map file.
//!
//! `fix bond/react` matches template atoms to system atoms by **type id**, so
//! the data file and every template must number types the same way. The
//! writer collects every type label the system and the templates use (the
//! string `type` columns) into one inventory per block, declares it in the
//! data file ([`TypeLabels::declare`]) and numbers the templates from it. The
//! returned [`BondReactSystem::labels`] is that numbering; write the force
//! field's coefficients with it (`LammpsForcefieldWriter::new(&labels)` in `ff`) so
//! the `*.ff` include covers every type, template-only ones included.
//!
//! Template atoms are paired by an integer `react_id` atom column, the same
//! value in the pre and the post template. A template's atom ids are its row
//! numbers (1-based): the map file speaks in them.
//!
//! References: LAMMPS `fix bond/react`, <https://docs.lammps.org/fix_bond_react.html>;
//! Gissinger, Jensen & Wise, *Polymer* **128** (2017) 211;
//! *Macromolecules* **53** (2020) 9953.

use crate::io::invalid_data;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::TypeLabels;
use molrs::core::keys;
use molrs::op::{F, Idx};
use ndarray::Array1;

use crate::io::lammps::data::write_lammps_data_with_masses;
use crate::io::lammps::molecule::write_lammps_molecule;

use crate::core::keys::REACT_ID;

/// The typed blocks a template and the system share a numbering for.
const TYPED_BLOCKS: [&str; 5] = ["atoms", "bonds", "angles", "dihedrals", "impropers"];

/// One `fix bond/react` reaction: the template before and after, and the
/// atoms the map file names, by `react_id`.
#[derive(Debug, Clone)]
pub struct BondReactTemplate {
    /// Pre-reaction template; its `atoms` carry the integer `react_id`.
    pub pre: Frame,
    /// Post-reaction template: the same `react_id`s, the new topology.
    pub post: Frame,
    /// The two atoms that initiate the reaction (`InitiatorIDs`), in order.
    pub initiators: [i64; 2],
    /// Atoms bonded to atoms outside the template (`EdgeIDs`). One that is
    /// also an initiator, or is not in `pre`, is left out of the map.
    pub edges: Vec<i64>,
    /// Atoms the reaction deletes (`DeleteIDs`). One not in `pre` is left out.
    pub deleted: Vec<i64>,
}

/// The `react_id` of every atom row of `frame`, in row order.
fn react_ids(frame: &Frame, which: &str) -> Result<Vec<i64>> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data(format!("the {which} template has no atoms")))?;
    let column = atoms.get(REACT_ID).ok_or_else(|| {
        invalid_data(format!(
            "the {which} template's atoms have no '{REACT_ID}' column pairing them"
        ))
    })?;
    if let Some(v) = column.as_int() {
        Ok(v.iter().map(|&x| x as i64).collect())
    } else if let Some(v) = column.as_uint() {
        Ok(v.iter().map(|&x| x as i64).collect())
    } else if let Some(v) = column.as_i64() {
        Ok(v.iter().copied().collect())
    } else if let Some(v) = column.as_u32() {
        Ok(v.iter().map(|&x| x as i64).collect())
    } else {
        Err(invalid_data(format!(
            "the {which} template's '{REACT_ID}' column must hold integers"
        )))
    }
}

/// `react_id` → 1-based template atom id; a repeated `react_id` is an error.
fn index(ids: &[i64], which: &str) -> Result<HashMap<i64, usize>> {
    let mut out = HashMap::with_capacity(ids.len());
    for (row, &rid) in ids.iter().enumerate() {
        if out.insert(rid, row + 1).is_some() {
            return Err(invalid_data(format!(
                "react_id {rid} appears twice in the {which} template"
            )));
        }
    }
    Ok(out)
}

impl BondReactTemplate {
    /// The map file's text.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidData`] when either template lacks an integer
    /// `react_id` column or repeats a value, when the pre and post templates
    /// do not hold the same `react_id`s, or when an initiator is not in the
    /// pre template.
    pub fn map_text(&self) -> Result<String> {
        let pre = react_ids(&self.pre, "pre")?;
        let post = react_ids(&self.post, "post")?;
        let pre_index = index(&pre, "pre")?;
        let post_index = index(&post, "post")?;

        let pre_set: BTreeSet<i64> = pre.iter().copied().collect();
        let post_set: BTreeSet<i64> = post.iter().copied().collect();
        if pre_set != post_set {
            return Err(invalid_data(format!(
                "the pre and post templates hold different atoms: missing in post \
                 {:?}, missing in pre {:?}",
                pre_set.difference(&post_set).collect::<Vec<_>>(),
                post_set.difference(&pre_set).collect::<Vec<_>>(),
            )));
        }

        let mut initiators = [0usize; 2];
        for (slot, rid) in initiators.iter_mut().zip(self.initiators) {
            *slot = *pre_index.get(&rid).ok_or_else(|| {
                invalid_data(format!(
                    "initiator atom (react_id={rid}) is not in the pre template; \
                     extract a wider local environment"
                ))
            })?;
        }
        let edges: Vec<usize> = self
            .edges
            .iter()
            .filter(|rid| !self.initiators.contains(rid))
            .filter_map(|rid| pre_index.get(rid).copied())
            .collect();
        let deleted: Vec<usize> = self
            .deleted
            .iter()
            .filter_map(|rid| pre_index.get(rid).copied())
            .collect();

        let mut text = String::new();
        // `write!` into a String cannot fail.
        let _ = writeln!(text, "# auto-generated map file for fix bond/react\n");
        let _ = writeln!(text, "{} equivalences", pre.len());
        let _ = writeln!(text, "{} edgeIDs", edges.len());
        let _ = writeln!(text, "{} deleteIDs\n", deleted.len());
        let _ = writeln!(text, "InitiatorIDs\n");
        for id in initiators {
            let _ = writeln!(text, "{id}");
        }
        let _ = writeln!(text, "\nEdgeIDs\n");
        for id in &edges {
            let _ = writeln!(text, "{id}");
        }
        let _ = writeln!(text, "\nDeleteIDs\n");
        for id in &deleted {
            let _ = writeln!(text, "{id}");
        }
        let _ = writeln!(text, "\nEquivalences\n");
        for (row, rid) in pre.iter().enumerate() {
            let _ = writeln!(text, "{}   {}", row + 1, post_index[rid]);
        }
        Ok(text)
    }
}

/// Write `{base_path}.map`, the map file of `template`.
///
/// # Errors
///
/// [`BondReactTemplate::map_text`]'s, or the file's I/O error.
pub fn write_lammps_bond_react_map(
    template: &BondReactTemplate,
    base_path: impl AsRef<Path>,
) -> Result<()> {
    let text = template.map_text()?;
    let mut path = base_path.as_ref().as_os_str().to_owned();
    path.push(".map");
    std::fs::write(PathBuf::from(path), text)
}

/// Topology rows a template lost because their type is not a label of the
/// unified inventory (an empty, `"None"` or purely numeric `type`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedRows {
    /// The reaction's name.
    pub template: String,
    /// The block the rows were in (`"bonds"`, `"angles"`, …).
    pub block: String,
    /// How many rows were dropped, summed over the pre and post templates.
    pub rows: usize,
}

/// What [`write_lammps_bond_react_system`] wrote.
#[derive(Debug, Clone)]
pub struct BondReactSystem {
    /// The type numbering shared by the data file and every template; write
    /// the force-field include with it.
    pub labels: TypeLabels,
    /// The data file, `{workdir}/{stem}.data`.
    pub data_path: PathBuf,
    /// The force-field include path this file set expects, `{workdir}/{stem}.ff`
    /// (not written here: coefficients are `ff`'s). The input reads it after
    /// the data file, so write it without a `units` line
    /// (`LammpsForcefieldWriteOptions::skip_units`).
    pub ff_path: PathBuf,
    /// Template topology rows left out for want of a type label.
    pub dropped: Vec<DroppedRows>,
}

/// Whether `label` names a type: non-empty, not the `"None"` placeholder,
/// and not a bare number (which is already an id, not a label).
fn is_label(label: &str) -> bool {
    !label.is_empty() && label != "None" && !label.bytes().all(|b| b.is_ascii_digit())
}

/// Every type label `frames` use, per typed block, sorted.
fn unified_labels(frames: &[&Frame]) -> HashMap<&'static str, Vec<String>> {
    TYPED_BLOCKS
        .iter()
        .map(|&block| {
            let mut all = BTreeSet::new();
            for frame in frames {
                if let Some(types) = frame
                    .get(block)
                    .and_then(|b| b.get(keys::TYPE))
                    .and_then(|c| c.as_string())
                {
                    all.extend(types.iter().filter(|t| is_label(t)).cloned());
                }
            }
            (block, all.into_iter().collect())
        })
        .collect()
}

/// The mass of each atom label the frames use: the first row's `mass`
/// column, else its element's periodic-table mass. The data file needs one
/// for every declared type, a type only a template uses included.
fn label_masses(frames: &[&Frame]) -> HashMap<String, F> {
    let mut out = HashMap::new();
    for frame in frames {
        let Some(atoms) = frame.get("atoms") else {
            continue;
        };
        let Some(types) = atoms.get(keys::TYPE).and_then(|c| c.as_string()) else {
            continue;
        };
        let mass = atoms.get(keys::MASS).and_then(|c| c.as_float());
        let element = atoms.get(keys::ELEMENT).and_then(|c| c.as_string());
        for (i, label) in types.iter().enumerate() {
            if out.contains_key(label) {
                continue;
            }
            let m = mass.map(|m| m[[i]]).or_else(|| {
                element
                    .and_then(|e| molrs::core::Element::by_symbol(&e[[i]]))
                    .map(|e| F::from(e.atomic_mass()))
            });
            if let Some(m) = m {
                out.insert(label.clone(), m);
            }
        }
    }
    out
}

/// A template frame ready for `write_lammps_molecule`: atom ids its row
/// numbers, every typed row's `type_id` its label's unified id, and topology
/// rows without a unified label dropped (their count per block returned).
fn numbered_template(
    frame: &Frame,
    ids: &HashMap<&'static str, HashMap<&str, Idx>>,
    which: &str,
) -> Result<(Frame, Vec<(&'static str, usize)>)> {
    let mut out = frame.clone();
    let mut dropped = Vec::new();
    for block_name in TYPED_BLOCKS {
        let Some(block) = out.get(block_name) else {
            continue;
        };
        let n = block.nrows().unwrap_or(0);
        if n == 0 {
            continue;
        }
        let Some(types) = block.get(keys::TYPE).and_then(|c| c.as_string()) else {
            continue;
        };
        let map = &ids[block_name];
        let labels: Vec<String> = types.iter().cloned().collect();
        let keep: Vec<usize> = (0..n)
            .filter(|&i| map.contains_key(labels[i].as_str()))
            .collect();
        let mut block: Block = if keep.len() < n {
            if block_name == "atoms" {
                let bad = labels.iter().find(|l| !map.contains_key(l.as_str()));
                return Err(invalid_data(format!(
                    "the {which} template has an atom without a type label ({bad:?}); \
                     type every template atom"
                )));
            }
            dropped.push((block_name, n - keep.len()));
            block
                .select_rows(&keep)
                .map_err(|e| invalid_data(e.to_string()))?
        } else {
            block.clone()
        };
        let type_ids: Vec<Idx> = keep.iter().map(|&i| map[labels[i].as_str()]).collect();
        block
            .insert(keys::TYPE_ID, Array1::from_vec(type_ids).into_dyn())
            .map_err(|e| invalid_data(e.to_string()))?;
        out.insert(block_name, block);
    }
    if let Some(atoms) = out.get_mut("atoms") {
        let n = atoms.nrows().unwrap_or(0) as Idx;
        atoms
            .insert(keys::ID, Array1::from_iter(1..=n).into_dyn())
            .map_err(|e| invalid_data(e.to_string()))?;
    }
    Ok((out, dropped))
}

/// Write the file set of a `fix bond/react` run into `workdir` (created if
/// missing): `{stem}.data` for `frame`, and `{name}_pre.mol`,
/// `{name}_post.mol` and `{name}.map` for each named template, `stem` being
/// `workdir`'s own file name.
///
/// Every type label `frame` and the templates use (their string `type`
/// columns) is declared in the data file, so the data file and the
/// templates number types alike; template topology rows whose type is not
/// such a label are left out of the molecule files and reported in
/// [`BondReactSystem::dropped`].
///
/// # Errors
///
/// [`ErrorKind::InvalidData`] for a template whose map is malformed
/// ([`BondReactTemplate::map_text`]) or that has an atom without a type
/// label; for a `frame` whose own labels are not the unified numbering
/// (a purely numeric label beside named ones); and any writer's error.
pub fn write_lammps_bond_react_system(
    workdir: impl AsRef<Path>,
    frame: &Frame,
    templates: &[(String, BondReactTemplate)],
) -> Result<BondReactSystem> {
    let workdir = workdir.as_ref();
    let stem = workdir
        .file_name()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("{} names no directory", workdir.display()),
            )
        })?
        .to_owned();
    let mut seen = HashSet::new();
    if let Some((name, _)) = templates.iter().find(|(name, _)| !seen.insert(name)) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("two templates are named {name:?}"),
        ));
    }
    // Every map is checked before any file is written.
    let maps = templates
        .iter()
        .map(|(_, t)| t.map_text())
        .collect::<Result<Vec<String>>>()?;

    let mut all: Vec<&Frame> = vec![frame];
    for (_, t) in templates {
        all.push(&t.pre);
        all.push(&t.post);
    }
    let unified = unified_labels(&all);

    let mut system = frame.clone();
    for block in TYPED_BLOCKS {
        TypeLabels::declare(&mut system, block, &unified[block]).map_err(invalid_data)?;
    }
    let labels = TypeLabels::from_frame(&system).map_err(invalid_data)?;
    for block in TYPED_BLOCKS {
        let declared = labels.block(block).and_then(|b| b.labels()).unwrap_or(&[]);
        if !unified[block].is_empty() && declared != unified[block].as_slice() {
            return Err(invalid_data(format!(
                "the system's {block} labels {declared:?} are not the unified numbering \
                 {:?}; give every {block} row a named type",
                unified[block]
            )));
        }
    }
    let ids: HashMap<&'static str, HashMap<&str, Idx>> = TYPED_BLOCKS
        .iter()
        .map(|&block| {
            let map = unified[block]
                .iter()
                .enumerate()
                .map(|(i, label)| (label.as_str(), (i + 1) as Idx))
                .collect();
            (block, map)
        })
        .collect();

    std::fs::create_dir_all(workdir)?;
    let base = workdir.join(&stem);
    let data_path = base.with_extension("data");
    write_lammps_data_with_masses(&data_path, &system, &label_masses(&all))?;

    let mut dropped = Vec::new();
    for ((name, template), map) in templates.iter().zip(maps) {
        let mut lost: HashMap<&'static str, usize> = HashMap::new();
        for (which, part) in [("pre", &template.pre), ("post", &template.post)] {
            let (numbered, gone) = numbered_template(part, &ids, which)?;
            for (block, rows) in gone {
                *lost.entry(block).or_default() += rows;
            }
            write_lammps_molecule(workdir.join(format!("{name}_{which}.mol")), &numbered)?;
        }
        for block in TYPED_BLOCKS {
            if let Some(&rows) = lost.get(block) {
                dropped.push(DroppedRows {
                    template: name.clone(),
                    block: block.to_owned(),
                    rows,
                });
            }
        }
        std::fs::write(workdir.join(format!("{name}.map")), map)?;
    }

    Ok(BondReactSystem {
        labels,
        ff_path: base.with_extension("ff"),
        data_path,
        dropped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> ndarray::ArrayD<String> {
        Array1::from_vec(values.iter().map(|s| s.to_string()).collect()).into_dyn()
    }

    /// `atoms` with `react_id`, `type`, coordinates, and `bonds` among them.
    fn template(types: &[&str], rids: &[i64], bonds: &[(Idx, Idx, &str)]) -> Frame {
        let n = types.len();
        let mut atoms = Block::new();
        atoms.insert("type", strings(types)).unwrap();
        atoms
            .insert(REACT_ID, Array1::from_vec(rids.to_vec()).into_dyn())
            .unwrap();
        let elements: Vec<&str> = types
            .iter()
            .map(|t| match *t {
                "oh" => "O",
                "hc" => "H",
                _ => "C",
            })
            .collect();
        atoms.insert("element", strings(&elements)).unwrap();
        for axis in ["x", "y", "z"] {
            atoms
                .insert(axis, Array1::from_vec(vec![0.0 as F; n]).into_dyn())
                .unwrap();
        }
        atoms
            .insert("charge", Array1::from_vec(vec![0.0 as F; n]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        if !bonds.is_empty() {
            let mut b = Block::new();
            b.insert(
                "atomi",
                Array1::from_iter(bonds.iter().map(|t| t.0)).into_dyn(),
            )
            .unwrap();
            b.insert(
                "atomj",
                Array1::from_iter(bonds.iter().map(|t| t.1)).into_dyn(),
            )
            .unwrap();
            b.insert(
                "type",
                strings(&bonds.iter().map(|t| t.2).collect::<Vec<_>>()),
            )
            .unwrap();
            frame.insert("bonds", b);
        }
        frame
    }

    /// c3 + oh → c3-oh: the bond type exists only in the post template.
    fn coupling() -> BondReactTemplate {
        BondReactTemplate {
            pre: template(&["c3", "oh", "hc"], &[1, 2, 3], &[(1, 2, "hc-oh")]),
            post: template(&["oh", "c3", "hc"], &[2, 1, 3], &[(0, 1, "c3-oh")]),
            initiators: [1, 2],
            edges: vec![1, 3],
            deleted: vec![3, 99],
        }
    }

    #[test]
    fn the_map_pairs_atoms_and_names_initiators_edges_and_deletions() {
        let text = coupling().map_text().unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines.contains(&"3 equivalences"));
        // edge 1 is an initiator and is left out.
        assert!(lines.contains(&"1 edgeIDs"));
        // deletion 99 is not in the template and is left out.
        assert!(lines.contains(&"1 deleteIDs"));
        let at = |section: &str| lines.iter().position(|l| *l == section).unwrap();
        assert!(at("InitiatorIDs") < at("EdgeIDs"));
        assert!(at("EdgeIDs") < at("DeleteIDs"));
        assert!(at("DeleteIDs") < at("Equivalences"));
        assert_eq!(
            &lines[at("InitiatorIDs") + 2..at("InitiatorIDs") + 4],
            ["1", "2"]
        );
        let eq: Vec<&str> = lines[at("Equivalences") + 2..].to_vec();
        assert_eq!(eq, ["1   2", "2   1", "3   3"]);
    }

    #[test]
    fn mismatched_templates_and_lost_initiators_are_refused() {
        let mut t = coupling();
        t.post = template(&["oh", "c3"], &[2, 1], &[]);
        assert_eq!(t.map_text().unwrap_err().kind(), ErrorKind::InvalidData);
        let mut t = coupling();
        t.initiators = [1, 7];
        assert!(t.map_text().unwrap_err().to_string().contains("react_id=7"));
        let mut t = coupling();
        t.pre = template(&["c3", "oh", "hc"], &[1, 1, 3], &[]);
        assert!(t.map_text().unwrap_err().to_string().contains("twice"));
    }

    #[test]
    fn the_system_declares_template_only_types_and_numbers_templates_alike() {
        let dir = std::env::temp_dir().join(format!("molrs-bond-react-{}", std::process::id()));
        let workdir = dir.join("rxn");
        let mut system = template(&["c3", "c3"], &[0, 0], &[(0, 1, "c3-c3")]);
        system
            .get_mut("atoms")
            .unwrap()
            .insert("mol_id", Array1::from_vec(vec![1 as Idx, 1]).into_dyn())
            .unwrap();
        let mut t = coupling();
        for part in [&mut t.pre, &mut t.post] {
            let n = part["atoms"].nrows().unwrap();
            part.get_mut("atoms")
                .unwrap()
                .insert("mol_id", Array1::from_vec(vec![1 as Idx; n]).into_dyn())
                .unwrap();
        }
        // An untyped template angle is dropped and reported.
        let mut angles = Block::new();
        for (k, v) in [("atomi", 0), ("atomj", 1), ("atomk", 2)] {
            angles
                .insert(k, Array1::from_vec(vec![v as Idx]).into_dyn())
                .unwrap();
        }
        angles.insert("type", strings(&["None"])).unwrap();
        t.post.insert("angles", angles);

        let out = write_lammps_bond_react_system(&workdir, &system, &[("rxn1".into(), t)]).unwrap();
        let data = std::fs::read_to_string(&out.data_path).unwrap();
        let pre = std::fs::read_to_string(workdir.join("rxn1_pre.mol")).unwrap();
        let post = std::fs::read_to_string(workdir.join("rxn1_post.mol")).unwrap();
        let map_exists = workdir.join("rxn1.map").exists();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(out.data_path, workdir.join("rxn.data"));
        assert_eq!(out.ff_path, workdir.join("rxn.ff"));
        assert!(map_exists);
        // atoms c3=1, hc=2, oh=3; bonds c3-c3=1, c3-oh=2, hc-oh=3.
        assert!(data.contains("3 atom types"), "{data}");
        // `oh` only a template uses: its mass is the template's, not 1.
        let masses = &data[data.find("Masses\n\n").unwrap()..];
        let oh: f64 = masses.lines().nth(4).unwrap()[2..].parse().unwrap();
        assert!((oh - 15.999).abs() < 0.01, "{data}");
        assert!(data.contains("3 bond types"), "{data}");
        assert!(
            data.contains("Bond Type Labels\n\n1 c3-c3\n2 c3-oh\n3 hc-oh\n"),
            "{data}"
        );
        assert!(pre.contains("Types\n\n1 1\n2 3\n3 2\n"), "{pre}");
        assert!(post.contains("Types\n\n1 3\n2 1\n3 2\n"), "{post}");
        assert!(post.contains("Bonds\n\n1 2 1 2\n"), "{post}");
        assert!(!post.contains("Angles"), "{post}");
        assert_eq!(
            out.dropped,
            [DroppedRows {
                template: "rxn1".into(),
                block: "angles".into(),
                rows: 1
            }]
        );
        let atoms = out.labels.block("atoms").unwrap();
        assert_eq!(atoms.labels().unwrap(), ["c3", "hc", "oh"]);
    }
}
