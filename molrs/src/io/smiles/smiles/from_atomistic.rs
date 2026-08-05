//! Build a concrete [`SmilesIR`] from an [`Atomistic`] graph.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::io::smiles::chem::ast::*;
use crate::io::smiles::error::{SmilesError, SmilesErrorKind};
use crate::io::smiles::smiles::options::{
    AromaticEmit, HydrogenEmit, MultiComponentEmit, SmilesEmitOptions,
};
use crate::io::smiles::smiles::write::write_smiles;
use molrs::Element;
use molrs::system::atomistic::{AtomId, Atomistic};
use molrs::system::bond::{BondNumber, BondType};
use molrs::system::molgraph::PropValue;

/// Convert a molecular graph into a concrete SMILES IR.
pub fn from_atomistic(mol: &Atomistic, opts: &SmilesEmitOptions) -> Result<SmilesIR, SmilesError> {
    if mol.n_atoms() == 0 {
        return Err(emit_err("empty molecule"));
    }

    let skip_h = matches!(
        opts.hydrogens,
        HydrogenEmit::OrganicSubset | HydrogenEmit::AsStored
    ) && opts.hydrogens != HydrogenEmit::ExplicitAll;

    let include_h_atoms = matches!(opts.hydrogens, HydrogenEmit::ExplicitAll);

    let mut components = connected_components(mol, include_h_atoms);
    if components.is_empty() {
        return Err(emit_err("no atoms to emit"));
    }

    // Sort components for stability when canonical.
    if opts.canonical {
        components.sort_by_key(|comp| {
            let mut labels: Vec<_> = comp
                .iter()
                .map(|&id| {
                    mol.get_atom(id)
                        .ok()
                        .and_then(|a| a.get_str("element").map(str::to_owned))
                        .unwrap_or_default()
                })
                .collect();
            labels.sort();
            (comp.len(), labels)
        });
    }

    match opts.multi_component {
        MultiComponentEmit::ErrorIfMultiple if components.len() > 1 => {
            return Err(emit_err(format!(
                "molecule has {} components; set multi_component=JoinDot or FirstOnly",
                components.len()
            )));
        }
        MultiComponentEmit::FirstOnly => {
            if let Some(root) = opts.root {
                components.retain(|c| c.contains(&root));
                if components.is_empty() {
                    return Err(emit_err("root atom not found in any component"));
                }
            } else {
                components.truncate(1);
            }
        }
        MultiComponentEmit::JoinDot | MultiComponentEmit::ErrorIfMultiple => {}
    }

    let mut chains = Vec::with_capacity(components.len());
    for comp in &components {
        let root = choose_root(mol, comp, opts)?;
        let chain = emit_component(mol, comp, root, opts, include_h_atoms, skip_h)?;
        chains.push(chain);
    }

    Ok(SmilesIR {
        components: chains,
        span: Span::new(0, 0),
    })
}

/// `from_atomistic` then [`write_smiles`].
pub fn write_atomistic_smiles(
    mol: &Atomistic,
    opts: &SmilesEmitOptions,
) -> Result<String, SmilesError> {
    let ir = from_atomistic(mol, opts)?;
    write_smiles(&ir)
}

fn emit_err(msg: impl Into<String>) -> SmilesError {
    SmilesError::new(SmilesErrorKind::Emit(msg.into()), Span::new(0, 0), "")
}

fn is_hydrogen(mol: &Atomistic, id: AtomId) -> bool {
    mol.get_atom(id)
        .ok()
        .and_then(|a| a.get_str("element").map(|s| s.eq_ignore_ascii_case("H")))
        .unwrap_or(false)
}

fn connected_components(mol: &Atomistic, include_h: bool) -> Vec<Vec<AtomId>> {
    let mut seen = HashSet::new();
    let mut comps = Vec::new();
    for (id, _) in mol.atoms() {
        if !include_h && is_hydrogen(mol, id) {
            continue;
        }
        if !seen.insert(id) {
            continue;
        }
        let mut q = VecDeque::new();
        let mut comp = Vec::new();
        q.push_back(id);
        while let Some(cur) = q.pop_front() {
            comp.push(cur);
            for (nb, _) in mol.neighbor_bonds(cur) {
                if !include_h && is_hydrogen(mol, nb) {
                    continue;
                }
                if seen.insert(nb) {
                    q.push_back(nb);
                }
            }
        }
        comps.push(comp);
    }
    comps
}

fn choose_root(
    mol: &Atomistic,
    comp: &[AtomId],
    opts: &SmilesEmitOptions,
) -> Result<AtomId, SmilesError> {
    if let Some(r) = opts.root {
        if comp.contains(&r) {
            return Ok(r);
        }
        return Err(emit_err("root atom not in component"));
    }
    if opts.canonical {
        let order = mol.canonical_order();
        for id in order {
            if comp.contains(&id) && !is_hydrogen(mol, id) {
                return Ok(id);
            }
        }
    }
    // Prefer non-H, lowest handle order in component.
    let mut ids = comp.to_vec();
    ids.sort_by_key(|id| molrs::system::molgraph::node_to_u64(*id));
    ids.into_iter()
        .find(|&id| !is_hydrogen(mol, id))
        .or_else(|| comp.first().copied())
        .ok_or_else(|| emit_err("empty component"))
}

#[derive(Default)]
struct Tree {
    /// parent[child] = Some(parent); root has None
    parent: HashMap<AtomId, Option<AtomId>>,
    children: HashMap<AtomId, Vec<AtomId>>,
    /// ring closures: atom → list of (rnum, optional bond kind for write)
    rings: HashMap<AtomId, Vec<(u16, Option<BondKind>)>>,
}

fn emit_component(
    mol: &Atomistic,
    comp: &[AtomId],
    root: AtomId,
    opts: &SmilesEmitOptions,
    include_h: bool,
    _skip_h_implicit: bool,
) -> Result<Chain, SmilesError> {
    let comp_set: HashSet<AtomId> = comp.iter().copied().collect();
    let tree = build_tree(mol, &comp_set, root, opts, include_h)?;
    build_chain_from(mol, root, &tree, opts)
}

fn build_tree(
    mol: &Atomistic,
    comp: &HashSet<AtomId>,
    root: AtomId,
    opts: &SmilesEmitOptions,
    include_h: bool,
) -> Result<Tree, SmilesError> {
    let mut tree = Tree::default();
    let mut parent_edge: HashMap<AtomId, AtomId> = HashMap::new();
    let mut stack = vec![root];
    tree.parent.insert(root, None);
    tree.children.insert(root, Vec::new());

    // DFS to build spanning tree
    let mut visited = HashSet::new();
    visited.insert(root);

    while let Some(cur) = stack.pop() {
        let mut nbs: Vec<AtomId> = mol
            .neighbor_bonds(cur)
            .map(|(nb, _)| nb)
            .filter(|nb| comp.contains(nb))
            .filter(|nb| include_h || !is_hydrogen(mol, *nb))
            .collect();

        if opts.canonical {
            let order = mol.canonical_order();
            let rank: HashMap<AtomId, usize> =
                order.iter().enumerate().map(|(i, id)| (*id, i)).collect();
            nbs.sort_by_key(|id| rank.get(id).copied().unwrap_or(usize::MAX));
        } else {
            nbs.sort_by_key(|id| molrs::system::molgraph::node_to_u64(*id));
        }

        for nb in nbs {
            if visited.insert(nb) {
                tree.parent.insert(nb, Some(cur));
                tree.children.entry(cur).or_default().push(nb);
                tree.children.entry(nb).or_default();
                parent_edge.insert(nb, cur);
                stack.push(nb);
            }
        }
    }

    // Non-tree edges → ring digits
    let mut next_rnum: u16 = 1;
    let mut assigned_edges: HashSet<(AtomId, AtomId)> = HashSet::new();
    for &a in comp {
        if !include_h && is_hydrogen(mol, a) {
            continue;
        }
        for (b, bid) in mol.neighbor_bonds(a) {
            if !comp.contains(&b) {
                continue;
            }
            if !include_h && is_hydrogen(mol, b) {
                continue;
            }
            let edge = if molrs::system::molgraph::node_to_u64(a)
                < molrs::system::molgraph::node_to_u64(b)
            {
                (a, b)
            } else {
                (b, a)
            };
            if !assigned_edges.insert(edge) {
                continue;
            }
            // tree edge?
            let is_tree = parent_edge.get(&b) == Some(&a) || parent_edge.get(&a) == Some(&b);
            if is_tree {
                continue;
            }
            let rnum = next_rnum;
            next_rnum += 1;
            if next_rnum > 99 {
                return Err(emit_err("too many ring closures (>99)"));
            }
            let bk = bond_kind_for(mol, bid, opts)?;
            tree.rings.entry(a).or_default().push((rnum, bk));
            tree.rings.entry(b).or_default().push((rnum, bk));
        }
    }

    Ok(tree)
}

fn bond_kind_for(
    mol: &Atomistic,
    bid: molrs::system::atomistic::BondId,
    opts: &SmilesEmitOptions,
) -> Result<Option<BondKind>, SmilesError> {
    let bt = mol.bond_type(bid);
    match opts.aromatic {
        AromaticEmit::AsMarked if bt.is_aromatic() => Ok(Some(BondKind::Aromatic)),
        AromaticEmit::KekuleOnly if bt.is_aromatic() => {
            // Prefer bond_number if set
            match mol.bond_number(bid) {
                BondNumber::Single => Ok(Some(BondKind::Single)),
                BondNumber::Double => Ok(Some(BondKind::Double)),
                BondNumber::Triple => Ok(Some(BondKind::Triple)),
                BondNumber::Quadruple => Ok(Some(BondKind::Quadruple)),
                BondNumber::Unknown => Err(emit_err(
                    "KekuleOnly requires integer bond_number on aromatic bonds",
                )),
            }
        }
        _ => match bt {
            BondType::Single => Ok(None), // default single omitted
            BondType::Double => Ok(Some(BondKind::Double)),
            BondType::Triple => Ok(Some(BondKind::Triple)),
            BondType::Aromatic => Ok(Some(BondKind::Aromatic)),
            BondType::Unknown => match mol.bond_number(bid) {
                BondNumber::Double => Ok(Some(BondKind::Double)),
                BondNumber::Triple => Ok(Some(BondKind::Triple)),
                BondNumber::Quadruple => Ok(Some(BondKind::Quadruple)),
                _ => Ok(None),
            },
        },
    }
}

fn find_bond(mol: &Atomistic, a: AtomId, b: AtomId) -> Option<molrs::system::atomistic::BondId> {
    mol.neighbor_bonds(a)
        .find(|(nb, _)| *nb == b)
        .map(|(_, bid)| bid)
}

fn build_chain_from(
    mol: &Atomistic,
    atom: AtomId,
    tree: &Tree,
    opts: &SmilesEmitOptions,
) -> Result<Chain, SmilesError> {
    let head = atom_node(mol, atom, opts)?;
    let mut tail = Vec::new();

    // Ring closures on this atom first (Daylight: after atom, before branches)
    if let Some(rings) = tree.rings.get(&atom) {
        for (rnum, bk) in rings {
            tail.push(ChainElement::RingClosure {
                bond: bk.map(BondQuery::Kind),
                rnum: *rnum,
                span: Span::new(0, 0),
            });
        }
    }

    let children = tree.children.get(&atom).cloned().unwrap_or_default();
    if children.is_empty() {
        return Ok(Chain { head, tail });
    }

    // All but last child are branches; last continues the main chain
    let (branches, last) = children.split_at(children.len().saturating_sub(1));
    for &ch in branches {
        let bond = find_bond(mol, atom, ch)
            .map(|bid| bond_kind_for(mol, bid, opts))
            .transpose()?
            .flatten();
        let sub = build_chain_from(mol, ch, tree, opts)?;
        tail.push(ChainElement::Branch {
            bond: bond.map(BondQuery::Kind),
            chain: sub,
            span: Span::new(0, 0),
        });
    }
    if let Some(&ch) = last.first() {
        let bond = find_bond(mol, atom, ch)
            .map(|bid| bond_kind_for(mol, bid, opts))
            .transpose()?
            .flatten();
        let sub = build_chain_from(mol, ch, tree, opts)?;
        // Flatten: bond + head of sub + sub.tail into main chain
        tail.push(ChainElement::BondedAtom {
            bond: bond.map(BondQuery::Kind),
            atom: sub.head,
        });
        tail.extend(sub.tail);
    }

    Ok(Chain { head, tail })
}

fn atom_is_aromatic(mol: &Atomistic, id: AtomId) -> bool {
    mol.get_atom(id)
        .ok()
        .and_then(|a| match a.get("is_aromatic") {
            Some(PropValue::Int(v)) if *v != 0 => Some(true),
            Some(PropValue::F64(v)) if *v != 0.0 => Some(true),
            Some(PropValue::Bool(true)) => Some(true),
            _ => None,
        })
        .unwrap_or(false)
}

fn formal_charge(mol: &Atomistic, id: AtomId) -> Option<i8> {
    mol.get_atom(id).ok().and_then(|a| {
        a.get_f64("formal_charge")
            .or_else(|| a.get_f64("charge"))
            .map(|v| v as i8)
    })
}

fn h_count_prop(mol: &Atomistic, id: AtomId) -> Option<u8> {
    mol.get_atom(id)
        .ok()
        .and_then(|a| a.get_f64("h_count").map(|v| v as u8))
}

fn isotope_prop(mol: &Atomistic, id: AtomId) -> Option<u16> {
    mol.get_atom(id)
        .ok()
        .and_then(|a| a.get_f64("isotope").map(|v| v as u16))
}

fn organic_subset_ok(sym: &str) -> bool {
    matches!(
        sym,
        "B" | "C"
            | "N"
            | "O"
            | "P"
            | "S"
            | "F"
            | "Cl"
            | "Br"
            | "I"
            | "b"
            | "c"
            | "n"
            | "o"
            | "p"
            | "s"
    )
}

fn atom_node(
    mol: &Atomistic,
    id: AtomId,
    opts: &SmilesEmitOptions,
) -> Result<AtomNode, SmilesError> {
    let atom = mol.get_atom(id).map_err(|e| emit_err(e.to_string()))?;
    let sym_raw = atom
        .get_str("element")
        .ok_or_else(|| emit_err("atom missing element"))?
        .to_owned();
    let sym = Element::by_symbol(&sym_raw)
        .map(|e| e.symbol().to_owned())
        .unwrap_or(sym_raw);

    let aromatic = matches!(opts.aromatic, AromaticEmit::AsMarked) && atom_is_aromatic(mol, id);
    let charge = formal_charge(mol, id).filter(|&c| c != 0);
    let isotope = isotope_prop(mol, id);
    let hcount = match opts.hydrogens {
        HydrogenEmit::OrganicSubset => None,
        HydrogenEmit::ExplicitAll => Some(h_count_prop(mol, id).unwrap_or(0)),
        HydrogenEmit::AsStored => h_count_prop(mol, id),
    };
    let stereo = if opts.include_stereo {
        atom.get_str("stereo").and_then(|s| match s {
            "CW" | "@@" => Some(Chirality::Clockwise),
            "CCW" | "@" => Some(Chirality::CounterClockwise),
            _ => None,
        })
    } else {
        None
    };

    let need_bracket = !opts.organic_subset
        || !organic_subset_ok(&sym)
        || charge.is_some()
        || isotope.is_some()
        || hcount.is_some()
        || stereo.is_some()
        || sym == "*";

    let spec = if need_bracket {
        AtomSpec::Bracket {
            isotope,
            symbol: BracketSymbol::Element {
                symbol: sym,
                aromatic,
            },
            chirality: stereo,
            hcount,
            charge,
            atom_class: None,
        }
    } else {
        AtomSpec::Organic {
            symbol: sym,
            aromatic,
        }
    };

    Ok(AtomNode {
        spec,
        span: Span::new(0, 0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::smiles::parse_smiles;
    use crate::io::smiles::smiles::to_atomistic::to_atomistic;

    #[test]
    fn ethanol_round_trip_stable() {
        let mol = to_atomistic(&parse_smiles("CCO").unwrap()).unwrap();
        let opts = SmilesEmitOptions::default();
        let s1 = write_atomistic_smiles(&mol, &opts).unwrap();
        let mol2 = to_atomistic(&parse_smiles(&s1).unwrap()).unwrap();
        let s2 = write_atomistic_smiles(&mol2, &opts).unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn acetic_and_benzene() {
        let opts = SmilesEmitOptions::default();
        for src in ["CC(=O)O", "c1ccccc1"] {
            let mol = to_atomistic(&parse_smiles(src).unwrap()).unwrap();
            let s = write_atomistic_smiles(&mol, &opts).unwrap();
            assert!(parse_smiles(&s).is_ok(), "src={src} wrote={s}");
            let mol2 = to_atomistic(&parse_smiles(&s).unwrap()).unwrap();
            let s2 = write_atomistic_smiles(&mol2, &opts).unwrap();
            assert_eq!(s, s2);
        }
    }

    #[test]
    fn multi_component_error_and_join() {
        let mol = to_atomistic(&parse_smiles("CCO.O").unwrap()).unwrap();
        let err_opts = SmilesEmitOptions::default();
        assert!(write_atomistic_smiles(&mol, &err_opts).is_err());

        let join = SmilesEmitOptions {
            multi_component: MultiComponentEmit::JoinDot,
            ..Default::default()
        };
        let s = write_atomistic_smiles(&mol, &join).unwrap();
        assert!(s.contains('.'), "got {s}");
    }

    #[test]
    fn hydrogens_flag_changes_output() {
        let mol = to_atomistic(&parse_smiles("CCO").unwrap()).unwrap();
        let a = SmilesEmitOptions {
            hydrogens: HydrogenEmit::OrganicSubset,
            ..Default::default()
        };
        let b = SmilesEmitOptions {
            hydrogens: HydrogenEmit::ExplicitAll,
            ..Default::default()
        };
        // ExplicitAll still has no H atoms in graph from to_atomistic, but
        // may force brackets with H0 — at least options path runs.
        let sa = write_atomistic_smiles(&mol, &a).unwrap();
        let sb = write_atomistic_smiles(&mol, &b).unwrap();
        assert!(!sa.is_empty());
        assert!(!sb.is_empty());
        // Bracket form under ExplicitAll for every atom
        assert!(sb.contains('['), "explicit expected brackets: {sb}");
    }

    #[test]
    fn no_stereo_when_flag_false() {
        // Bracket stereo marker if present on graph; default flag must not invent @.
        let mol = to_atomistic(&parse_smiles("CCO").unwrap()).unwrap();
        let opts = SmilesEmitOptions::default(); // include_stereo = false
        let s = write_atomistic_smiles(&mol, &opts).unwrap();
        assert!(!s.contains('@'), "got {s}");
    }
}
