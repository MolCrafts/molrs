//! Fragment-rule embedding (stage-1 coordinate construction).
//!
//! The implementation follows the same algorithm family used by classic
//! builder pipeline: split by rotatable bonds into rigid fragments, embed
//! rigid fragments first (including ring templates), then stitch fragments
//! through connector bonds.

use std::collections::{HashMap, HashSet, VecDeque};

use rand::Rng;

use super::fragment_data::{RigidTemplate, rigid_templates, ring_templates};
use super::geom::{
    add, arbitrary_perpendicular, cross, dot, norm, normalize, random_unit, rotate_about_axis,
    scale, sub,
};
use molrs::element::Element;
use molrs::error::MolRsError;
use molrs::molgraph::{AtomId, MolGraph};
use molrs::rings::find_rings;
use molrs::rotatable::detect_rotatable_bonds;

/// Summary metrics returned by the embedding stage.
#[derive(Debug, Clone)]
pub(crate) struct BuildSummary {
    pub placed_atoms: usize,
    pub warnings: Vec<String>,
}

/// Embed initial 3D coordinates using a fragment/rule strategy.
pub(crate) fn embed_fragment_rules(
    mol: &mut MolGraph,
    rng: &mut impl Rng,
) -> Result<BuildSummary, MolRsError> {
    let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    if atom_ids.is_empty() {
        return Err(MolRsError::validation(
            "cannot build 3D coordinates for empty molecule",
        ));
    }
    let n = atom_ids.len();
    let id_to_idx: HashMap<AtomId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut warnings = Vec::new();
    let rotatable: HashSet<(usize, usize)> = detect_rotatable_bonds(mol)
        .into_iter()
        .filter_map(|(a, b)| {
            let i = id_to_idx.get(&a).copied()?;
            let j = id_to_idx.get(&b).copied()?;
            Some(canonical_pair(i, j))
        })
        .collect();
    let (fragments, frag_of) = build_rigid_fragments(mol, &atom_ids, &id_to_idx, &rotatable);
    if fragments.len() > 1 {
        warnings.push(format!(
            "fragment-rules embedding split molecule into {} rigid fragment(s)",
            fragments.len()
        ));
    }

    let ring_idx = ring_atom_indices(mol, &id_to_idx);
    let mut local_coords = vec![[0.0_f64; 3]; n];
    for frag_atoms in &fragments {
        embed_rigid_fragment_local(
            mol,
            &atom_ids,
            &id_to_idx,
            frag_atoms,
            &ring_idx,
            &mut local_coords,
            rng,
            &mut warnings,
        );
    }

    let mut coords = assemble_fragments(
        mol,
        &atom_ids,
        &id_to_idx,
        &fragments,
        &frag_of,
        &local_coords,
        rng,
    );

    // Final geometric cleanup.
    relax_bond_lengths(mol, &atom_ids, &id_to_idx, &mut coords, 30);
    for c in &mut coords {
        c[2] += rng.random_range(-0.01..0.01);
    }
    recenter_subset(&mut coords, &(0..n).collect::<Vec<_>>());

    for (i, atom_id) in atom_ids.iter().copied().enumerate() {
        let atom = mol.get_atom_mut(atom_id)?;
        if atom.get_str("element").is_none() {
            warnings.push(format!(
                "atom {:?} has no symbol; using generic geometry defaults",
                atom_id
            ));
        }
        atom.set("x", coords[i][0]);
        atom.set("y", coords[i][1]);
        atom.set("z", coords[i][2]);
    }

    Ok(BuildSummary {
        placed_atoms: n,
        warnings,
    })
}

fn build_rigid_fragments(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
    rotatable: &HashSet<(usize, usize)>,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = atom_ids.len();
    let mut visited = vec![false; n];
    let mut fragments = Vec::new();
    let mut frag_of = vec![usize::MAX; n];

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut q = VecDeque::new();
        let mut frag = Vec::new();
        visited[start] = true;
        q.push_back(start);
        while let Some(i) = q.pop_front() {
            frag.push(i);
            for nid in mol.neighbors(atom_ids[i]) {
                let Some(j) = id_to_idx.get(&nid).copied() else {
                    continue;
                };
                if rotatable.contains(&canonical_pair(i, j)) {
                    continue;
                }
                if !visited[j] {
                    visited[j] = true;
                    q.push_back(j);
                }
            }
        }
        let fid = fragments.len();
        for &idx in &frag {
            frag_of[idx] = fid;
        }
        fragments.push(frag);
    }

    (fragments, frag_of)
}

fn ring_atom_indices(mol: &MolGraph, id_to_idx: &HashMap<AtomId, usize>) -> Vec<Vec<usize>> {
    find_rings(mol)
        .rings()
        .iter()
        .map(|ring| {
            ring.iter()
                .filter_map(|id| id_to_idx.get(id).copied())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn embed_rigid_fragment_local(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
    frag_atoms: &[usize],
    ring_idx: &[Vec<usize>],
    local_coords: &mut [[f64; 3]],
    rng: &mut impl Rng,
    warnings: &mut Vec<String>,
) {
    if frag_atoms.is_empty() {
        return;
    }
    let n = atom_ids.len();
    let mut in_fragment = vec![false; n];
    for &idx in frag_atoms {
        in_fragment[idx] = true;
    }

    let mut placed = vec![false; n];
    let mut queue = VecDeque::new();

    let has_ring = ring_idx
        .iter()
        .any(|ring| ring.len() >= 3 && ring.iter().all(|&idx| in_fragment[idx]));
    if !has_ring
        && let Some(template_name) =
            try_embed_rigid_template(mol, atom_ids, id_to_idx, frag_atoms, local_coords, rng)
    {
        warnings.push(format!(
            "fragment-rules embedded rigid template \"{}\"",
            template_name
        ));
        return;
    }

    // Ring templates first: this matches the fragment/ring-first behavior.
    let mut seeded_rings = 0usize;
    let mut ring_offset = 0.0_f64;
    for ring in ring_idx {
        if ring.len() < 3 || !ring.iter().all(|&idx| in_fragment[idx]) {
            continue;
        }
        if ring.iter().any(|&idx| placed[idx]) {
            continue;
        }
        seeded_rings += 1;
        place_regular_ring_template(mol, atom_ids, ring, local_coords, ring_offset);
        let mut max_r = 1.2_f64;
        for &idx in ring {
            placed[idx] = true;
            queue.push_back(idx);
            max_r = max_r.max(norm(local_coords[idx]));
        }
        ring_offset += 2.5 * max_r + 1.0;
    }
    if seeded_rings > 0 {
        warnings.push(format!(
            "fragment-rules embedded {} ring template(s) in rigid fragment",
            seeded_rings
        ));
    }

    if queue.is_empty() {
        let root = frag_atoms[0];
        local_coords[root] = [0.0, 0.0, 0.0];
        placed[root] = true;
        queue.push_back(root);
    }

    while let Some(parent_idx) = queue.pop_front() {
        let parent_id = atom_ids[parent_idx];
        let mut neighbors: Vec<usize> = mol
            .neighbors(parent_id)
            .filter_map(|nid| id_to_idx.get(&nid).copied())
            .filter(|&idx| in_fragment[idx])
            .collect();
        neighbors.sort_unstable();

        let mut child_ord = 0usize;
        for child_idx in neighbors {
            if placed[child_idx] {
                continue;
            }
            let child_id = atom_ids[child_idx];
            let pos = propose_child_position_fragment(
                mol,
                atom_ids,
                id_to_idx,
                &in_fragment,
                local_coords,
                &placed,
                parent_idx,
                child_idx,
                parent_id,
                child_id,
                child_ord,
                rng,
            );
            local_coords[child_idx] = pos;
            placed[child_idx] = true;
            queue.push_back(child_idx);
            child_ord += 1;
        }
    }

    for &idx in frag_atoms {
        if placed[idx] {
            continue;
        }
        local_coords[idx] = scale(random_unit(rng), 0.4);
        placed[idx] = true;
    }

    recenter_subset(local_coords, frag_atoms);
}

fn place_regular_ring_template(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    ring: &[usize],
    coords: &mut [[f64; 3]],
    x_offset: f64,
) {
    let n = ring.len();
    let ring_symbols = ring
        .iter()
        .map(|&idx| {
            mol.get_atom(atom_ids[idx])
                .ok()
                .and_then(|a| a.get_str("element"))
                .unwrap_or("C")
                .to_string()
        })
        .collect::<Vec<_>>();

    if let Some(mut tpl_coords) = match_ring_template(&ring_symbols) {
        let target_avg = average_ring_bond_length(mol, atom_ids, ring);
        let tpl_avg = average_closed_path_bond_length(&tpl_coords);
        let s = if tpl_avg > 1.0e-8 {
            target_avg / tpl_avg
        } else {
            1.0
        };
        for c in &mut tpl_coords {
            *c = scale(*c, s);
        }
        let c0 = centroid(&tpl_coords);
        for c in &mut tpl_coords {
            *c = sub(*c, c0);
        }
        for (k, &idx) in ring.iter().enumerate() {
            coords[idx] = add(tpl_coords[k], [x_offset, 0.0, 0.0]);
        }
        return;
    }

    // Fallback rule template if no matching fragment entry exists.
    let avg_bond = average_ring_bond_length(mol, atom_ids, ring);
    let radius = avg_bond / (2.0 * (std::f64::consts::PI / n as f64).sin()).max(1.0e-6);
    let aromatic = ring
        .iter()
        .enumerate()
        .all(|(k, &i)| bond_order_between(mol, atom_ids[i], atom_ids[ring[(k + 1) % n]]) >= 1.3);
    for (k, &idx) in ring.iter().enumerate() {
        let ang = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
        let z = if aromatic {
            0.0
        } else {
            0.05 * (k as f64).sin()
        };
        coords[idx] = [x_offset + radius * ang.cos(), radius * ang.sin(), z];
    }
}

fn match_ring_template(ring_symbols: &[String]) -> Option<Vec<[f64; 3]>> {
    let n = ring_symbols.len();
    for tpl in ring_templates().iter().filter(|t| t.symbols.len() == n) {
        // forward orientation
        for shift in 0..n {
            if ring_symbols
                .iter()
                .enumerate()
                .all(|(i, sym)| ring_symbol_matches(&tpl.symbols[(shift + i) % n], sym))
            {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(tpl.coords[(shift + i) % n]);
                }
                return Some(out);
            }
        }
        // reverse orientation
        for shift in 0..n {
            if ring_symbols
                .iter()
                .enumerate()
                .all(|(i, sym)| ring_symbol_matches(&tpl.symbols[(shift + n - i) % n], sym))
            {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(tpl.coords[(shift + n - i) % n]);
                }
                return Some(out);
            }
        }
    }
    None
}

fn ring_symbol_matches(template_symbol: &str, ring_symbol: &str) -> bool {
    template_symbol == "*" || template_symbol.eq_ignore_ascii_case(ring_symbol)
}

fn try_embed_rigid_template(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
    frag_atoms: &[usize],
    coords: &mut [[f64; 3]],
    rng: &mut impl Rng,
) -> Option<String> {
    let templates = rigid_templates();
    if templates.is_empty() || frag_atoms.is_empty() {
        return None;
    }

    let frag = build_fragment_signature(mol, atom_ids, id_to_idx, frag_atoms);
    for tpl in templates
        .iter()
        .filter(|t| t.symbols.len() == frag_atoms.len())
    {
        if !multiset_equal(&frag.symbols, &tpl.symbols) {
            continue;
        }
        let tpl_adj = template_adjacency(tpl);
        let tpl_deg = tpl_adj
            .iter()
            .map(|row| row.iter().filter(|&&b| b).count())
            .collect::<Vec<_>>();
        if !multiset_equal_usize(&frag.degrees, &tpl_deg) {
            continue;
        }

        let mapping = find_isomorphism(
            &frag.adj,
            &frag.symbols,
            &frag.degrees,
            &tpl_adj,
            &tpl.symbols,
            &tpl_deg,
        )?;
        place_template_on_fragment(
            mol,
            atom_ids,
            frag_atoms,
            &frag.edges,
            tpl,
            &mapping,
            coords,
            rng,
        );
        return Some(tpl.name.clone());
    }
    None
}

#[derive(Debug)]
struct FragmentSignature {
    symbols: Vec<String>,
    degrees: Vec<usize>,
    adj: Vec<Vec<bool>>,
    edges: Vec<(usize, usize)>,
}

fn build_fragment_signature(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
    frag_atoms: &[usize],
) -> FragmentSignature {
    let n = frag_atoms.len();
    let local_of = frag_atoms
        .iter()
        .enumerate()
        .map(|(l, &g)| (g, l))
        .collect::<HashMap<_, _>>();
    let mut adj = vec![vec![false; n]; n];
    let mut edges = Vec::new();
    let mut symbols = Vec::with_capacity(n);

    for &g in frag_atoms {
        let sym = mol
            .get_atom(atom_ids[g])
            .ok()
            .and_then(|a| a.get_str("element"))
            .unwrap_or("X")
            .to_string();
        symbols.push(sym);
    }

    for (li, &gi) in frag_atoms.iter().enumerate() {
        for nid in mol.neighbors(atom_ids[gi]) {
            let Some(gj) = id_to_idx.get(&nid).copied() else {
                continue;
            };
            let Some(&lj) = local_of.get(&gj) else {
                continue;
            };
            if li == lj || adj[li][lj] {
                continue;
            }
            adj[li][lj] = true;
            adj[lj][li] = true;
            if li < lj {
                edges.push((li, lj));
            } else {
                edges.push((lj, li));
            }
        }
    }

    edges.sort_unstable();
    edges.dedup();
    let degrees = adj
        .iter()
        .map(|row| row.iter().filter(|&&b| b).count())
        .collect::<Vec<_>>();

    FragmentSignature {
        symbols,
        degrees,
        adj,
        edges,
    }
}

fn template_adjacency(tpl: &RigidTemplate) -> Vec<Vec<bool>> {
    let n = tpl.symbols.len();
    let mut adj = vec![vec![false; n]; n];
    for &(i, j, _) in &tpl.bonds {
        if i < n && j < n && i != j {
            adj[i][j] = true;
            adj[j][i] = true;
        }
    }
    adj
}

fn find_isomorphism(
    frag_adj: &[Vec<bool>],
    frag_symbols: &[String],
    frag_deg: &[usize],
    tpl_adj: &[Vec<bool>],
    tpl_symbols: &[String],
    tpl_deg: &[usize],
) -> Option<Vec<usize>> {
    let n = frag_symbols.len();
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by_key(|&i| (usize::MAX - frag_deg[i], frag_symbols[i].clone()));

    let mut map_frag_to_tpl = vec![usize::MAX; n];
    let mut used_tpl = vec![false; n];

    #[allow(clippy::too_many_arguments)]
    fn backtrack(
        depth: usize,
        order: &[usize],
        map_frag_to_tpl: &mut [usize],
        used_tpl: &mut [bool],
        frag_adj: &[Vec<bool>],
        frag_symbols: &[String],
        frag_deg: &[usize],
        tpl_adj: &[Vec<bool>],
        tpl_symbols: &[String],
        tpl_deg: &[usize],
    ) -> bool {
        if depth == order.len() {
            return true;
        }
        let f = order[depth];
        for t in 0..tpl_symbols.len() {
            if used_tpl[t] {
                continue;
            }
            if frag_symbols[f] != tpl_symbols[t] || frag_deg[f] != tpl_deg[t] {
                continue;
            }

            let mut ok = true;
            for ff in 0..map_frag_to_tpl.len() {
                let tt = map_frag_to_tpl[ff];
                if tt == usize::MAX {
                    continue;
                }
                if frag_adj[f][ff] != tpl_adj[t][tt] {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }

            map_frag_to_tpl[f] = t;
            used_tpl[t] = true;
            if backtrack(
                depth + 1,
                order,
                map_frag_to_tpl,
                used_tpl,
                frag_adj,
                frag_symbols,
                frag_deg,
                tpl_adj,
                tpl_symbols,
                tpl_deg,
            ) {
                return true;
            }
            map_frag_to_tpl[f] = usize::MAX;
            used_tpl[t] = false;
        }
        false
    }

    if backtrack(
        0,
        &order,
        &mut map_frag_to_tpl,
        &mut used_tpl,
        frag_adj,
        frag_symbols,
        frag_deg,
        tpl_adj,
        tpl_symbols,
        tpl_deg,
    ) {
        Some(map_frag_to_tpl)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn place_template_on_fragment(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    frag_atoms: &[usize],
    frag_edges: &[(usize, usize)],
    tpl: &RigidTemplate,
    frag_to_tpl: &[usize],
    coords: &mut [[f64; 3]],
    rng: &mut impl Rng,
) {
    let n = frag_atoms.len();
    if n == 0 {
        return;
    }

    let mut frag_coords = vec![[0.0_f64; 3]; n];
    for lf in 0..n {
        frag_coords[lf] = tpl.coords[frag_to_tpl[lf]];
    }

    let tpl_avg = average_template_bond_length(
        &frag_coords,
        &template_edges_in_frag_order(tpl, frag_to_tpl),
    );
    let frag_avg = average_fragment_target_bond_length(mol, atom_ids, frag_atoms, frag_edges);
    if tpl_avg > 1.0e-8 {
        let s = frag_avg / tpl_avg;
        for c in &mut frag_coords {
            *c = scale(*c, s);
        }
    }

    let c0 = centroid(&frag_coords);
    for c in &mut frag_coords {
        *c = sub(*c, c0);
    }

    let axis = random_unit(rng);
    let angle = rng.random_range(0.0..(2.0 * std::f64::consts::PI));
    for c in &mut frag_coords {
        *c = rotate_about_axis(*c, axis, angle);
    }

    for (lf, &gidx) in frag_atoms.iter().enumerate() {
        coords[gidx] = frag_coords[lf];
    }
}

fn template_edges_in_frag_order(tpl: &RigidTemplate, frag_to_tpl: &[usize]) -> Vec<(usize, usize)> {
    let mut tpl_to_frag = vec![usize::MAX; tpl.symbols.len()];
    for (f, &t) in frag_to_tpl.iter().enumerate() {
        if t < tpl_to_frag.len() {
            tpl_to_frag[t] = f;
        }
    }

    let mut edges = Vec::new();
    for &(i, j, _) in &tpl.bonds {
        if i >= tpl_to_frag.len() || j >= tpl_to_frag.len() {
            continue;
        }
        let fi = tpl_to_frag[i];
        let fj = tpl_to_frag[j];
        if fi == usize::MAX || fj == usize::MAX || fi == fj {
            continue;
        }
        edges.push(canonical_pair(fi, fj));
    }
    edges.sort_unstable();
    edges.dedup();
    edges
}

fn average_template_bond_length(coords: &[[f64; 3]], edges: &[(usize, usize)]) -> f64 {
    if edges.is_empty() {
        return 1.4;
    }
    let mut s = 0.0;
    for &(i, j) in edges {
        s += norm(sub(coords[i], coords[j]));
    }
    s / edges.len() as f64
}

fn average_fragment_target_bond_length(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    frag_atoms: &[usize],
    edges: &[(usize, usize)],
) -> f64 {
    if edges.is_empty() {
        return 1.4;
    }
    let mut s = 0.0;
    for &(li, lj) in edges {
        s += ideal_bond_length(mol, atom_ids[frag_atoms[li]], atom_ids[frag_atoms[lj]]);
    }
    s / edges.len() as f64
}

fn multiset_equal(a: &[String], b: &[String]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut aa = a.to_vec();
    let mut bb = b.to_vec();
    aa.sort();
    bb.sort();
    aa == bb
}

fn multiset_equal_usize(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut aa = a.to_vec();
    let mut bb = b.to_vec();
    aa.sort_unstable();
    bb.sort_unstable();
    aa == bb
}

fn average_ring_bond_length(mol: &MolGraph, atom_ids: &[AtomId], ring: &[usize]) -> f64 {
    if ring.is_empty() {
        return 1.4;
    }
    let mut s = 0.0;
    for k in 0..ring.len() {
        let i = ring[k];
        let j = ring[(k + 1) % ring.len()];
        s += ideal_bond_length(mol, atom_ids[i], atom_ids[j]);
    }
    s / ring.len() as f64
}

fn average_closed_path_bond_length(coords: &[[f64; 3]]) -> f64 {
    if coords.is_empty() {
        return 1.4;
    }
    let mut s = 0.0;
    for i in 0..coords.len() {
        s += norm(sub(coords[(i + 1) % coords.len()], coords[i]));
    }
    s / coords.len() as f64
}

fn centroid(coords: &[[f64; 3]]) -> [f64; 3] {
    if coords.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut c = [0.0, 0.0, 0.0];
    for &p in coords {
        c = add(c, p);
    }
    scale(c, 1.0 / coords.len() as f64)
}

fn assemble_fragments(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
    fragments: &[Vec<usize>],
    frag_of: &[usize],
    local_coords: &[[f64; 3]],
    rng: &mut impl Rng,
) -> Vec<[f64; 3]> {
    let n = atom_ids.len();
    let mut frag_edges: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); fragments.len()];
    for (_, bond) in mol.bonds() {
        let Some(i) = id_to_idx.get(&bond.atoms[0]).copied() else {
            continue;
        };
        let Some(j) = id_to_idx.get(&bond.atoms[1]).copied() else {
            continue;
        };
        let fi = frag_of[i];
        let fj = frag_of[j];
        if fi == fj {
            continue;
        }
        frag_edges[fi].push((fj, i, j));
        frag_edges[fj].push((fi, j, i));
    }

    let mut coords = vec![[0.0_f64; 3]; n];
    let mut frag_placed = vec![false; fragments.len()];
    let mut component_offset = 0.0_f64;

    for root in 0..fragments.len() {
        if frag_placed[root] {
            continue;
        }

        for &idx in &fragments[root] {
            coords[idx] = add(local_coords[idx], [component_offset, 0.0, 0.0]);
        }
        frag_placed[root] = true;
        let mut q = VecDeque::new();
        q.push_back(root);
        let mut component_atoms = fragments[root].len();

        while let Some(f) = q.pop_front() {
            for &(g, atom_f, atom_g) in &frag_edges[f] {
                if frag_placed[g] {
                    continue;
                }
                let pos_f = coords[atom_f];
                let dir = connection_direction(&coords, &fragments[f], atom_f, rng);
                let target_len = ideal_bond_length(mol, atom_ids[atom_f], atom_ids[atom_g]);
                let target_g = add(pos_f, scale(dir, target_len));

                let centroid_g = centroid_of_fragment(local_coords, &fragments[g]);
                let origin_local = local_coords[atom_g];
                let mut v_local = sub(centroid_g, origin_local);
                if norm(v_local) < 1.0e-8 {
                    if let Some(&other) = fragments[g].iter().find(|&&idx| idx != atom_g) {
                        v_local = sub(local_coords[other], origin_local);
                    } else {
                        v_local = [1.0, 0.0, 0.0];
                    }
                }
                let torsion = rng.random_range(0.0..(2.0 * std::f64::consts::PI));

                for &idx in &fragments[g] {
                    let rel = sub(local_coords[idx], origin_local);
                    let rel_aligned = rotate_from_to(rel, v_local, dir);
                    let rel_twisted = rotate_about_axis(rel_aligned, dir, torsion);
                    coords[idx] = add(target_g, rel_twisted);
                }

                frag_placed[g] = true;
                component_atoms += fragments[g].len();
                q.push_back(g);
            }
        }

        component_offset += 6.0 + 1.8 * (component_atoms as f64).sqrt();
    }

    coords
}

fn connection_direction(
    coords: &[[f64; 3]],
    frag_atoms: &[usize],
    anchor: usize,
    rng: &mut impl Rng,
) -> [f64; 3] {
    let centroid = centroid_of_fragment(coords, frag_atoms);
    let mut dir = sub(coords[anchor], centroid);
    if norm(dir) < 1.0e-8
        && let Some(&other) = frag_atoms.iter().find(|&&idx| idx != anchor)
    {
        dir = sub(coords[anchor], coords[other]);
    }
    if norm(dir) < 1.0e-8 {
        dir = random_unit(rng);
    }
    normalize(dir)
}

fn centroid_of_fragment(coords: &[[f64; 3]], frag_atoms: &[usize]) -> [f64; 3] {
    if frag_atoms.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let n = frag_atoms.len() as f64;
    let mut c = [0.0, 0.0, 0.0];
    for &idx in frag_atoms {
        c = add(c, coords[idx]);
    }
    scale(c, 1.0 / n)
}

fn rotate_from_to(v: [f64; 3], from: [f64; 3], to: [f64; 3]) -> [f64; 3] {
    let u = normalize(from);
    let w = normalize(to);
    let c = cross(u, w);
    let s = norm(c);
    let d = dot(u, w).clamp(-1.0, 1.0);

    if s < 1.0e-10 {
        if d > 0.0 {
            return v;
        }
        let axis = arbitrary_perpendicular(u);
        return rotate_about_axis(v, axis, std::f64::consts::PI);
    }

    let axis = scale(c, 1.0 / s);
    let angle = s.atan2(d);
    rotate_about_axis(v, axis, angle)
}

#[allow(clippy::too_many_arguments)]
fn propose_child_position_fragment(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
    in_fragment: &[bool],
    coords: &[[f64; 3]],
    placed: &[bool],
    parent_idx: usize,
    child_idx: usize,
    parent_id: AtomId,
    child_id: AtomId,
    child_ord: usize,
    rng: &mut impl Rng,
) -> [f64; 3] {
    let parent = coords[parent_idx];
    let bond_len = ideal_bond_length(mol, parent_id, child_id);

    let mut placed_neighbors: Vec<usize> = mol
        .neighbors(parent_id)
        .filter_map(|nid| id_to_idx.get(&nid).copied())
        .filter(|&idx| idx != child_idx && in_fragment[idx] && placed[idx])
        .collect();
    placed_neighbors.sort_unstable();

    let direction = if placed_neighbors.is_empty() {
        if child_ord == 0 {
            [1.0, 0.0, 0.0]
        } else {
            random_unit(rng)
        }
    } else if placed_neighbors.len() == 1 {
        let ref_vec = normalize(sub(parent, coords[placed_neighbors[0]]));
        let theta = ideal_angle_for_center(mol, parent_id);
        let phi = rng.random_range(0.0..(2.0 * std::f64::consts::PI));
        let perp = arbitrary_perpendicular(ref_vec);
        let ring = rotate_about_axis(perp, ref_vec, phi);
        normalize(add(scale(ref_vec, theta.cos()), scale(ring, theta.sin())))
    } else {
        let mut away = [0.0, 0.0, 0.0];
        for &nidx in &placed_neighbors {
            away = add(away, normalize(sub(parent, coords[nidx])));
        }
        normalize(add(away, scale(random_unit(rng), 0.20)))
    };

    let mut candidate = add(parent, scale(direction, bond_len));
    let mut anchors: Vec<usize> = mol
        .neighbors(child_id)
        .filter_map(|nid| id_to_idx.get(&nid).copied())
        .filter(|&idx| idx != parent_idx && in_fragment[idx] && placed[idx])
        .collect();
    anchors.sort_unstable();

    if !anchors.is_empty() {
        for _ in 0..3 {
            for &aidx in &anchors {
                let target = ideal_bond_length(mol, atom_ids[aidx], child_id);
                let v = sub(coords[aidx], candidate);
                let d = norm(v);
                if d > 1.0e-8 {
                    candidate = add(candidate, scale(normalize(v), (d - target) * 0.35));
                }
            }
        }
        candidate = add(parent, scale(normalize(sub(candidate, parent)), bond_len));
    }

    candidate
}

fn recenter_subset(coords: &mut [[f64; 3]], subset: &[usize]) {
    if subset.is_empty() {
        return;
    }
    let n = subset.len() as f64;
    let mut c = [0.0, 0.0, 0.0];
    for &idx in subset {
        c = add(c, coords[idx]);
    }
    c = scale(c, 1.0 / n);
    for &idx in subset {
        coords[idx] = sub(coords[idx], c);
    }
}

fn relax_bond_lengths(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
    coords: &mut [[f64; 3]],
    iterations: usize,
) {
    for _ in 0..iterations {
        for (_, bond) in mol.bonds() {
            let Some(i) = id_to_idx.get(&bond.atoms[0]).copied() else {
                continue;
            };
            let Some(j) = id_to_idx.get(&bond.atoms[1]).copied() else {
                continue;
            };
            let target = ideal_bond_length(mol, atom_ids[i], atom_ids[j]);
            let dvec = sub(coords[j], coords[i]);
            let d = norm(dvec);
            if d < 1.0e-10 {
                continue;
            }
            let corr = (d - target) * 0.25;
            let shift = scale(normalize(dvec), corr);
            coords[i] = add(coords[i], shift);
            coords[j] = sub(coords[j], shift);
        }
    }
}

fn ideal_angle_for_center(mol: &MolGraph, center: AtomId) -> f64 {
    let degree = mol.neighbors(center).count();
    let max_order = mol
        .neighbor_bonds(center)
        .map(|(_, order)| order)
        .fold(1.0_f64, f64::max);

    if max_order >= 2.5 || (degree == 2 && max_order >= 2.0) {
        std::f64::consts::PI
    } else if max_order >= 1.5 || degree == 3 {
        120_f64.to_radians()
    } else {
        109.47_f64.to_radians()
    }
}

fn ideal_bond_length(mol: &MolGraph, a: AtomId, b: AtomId) -> f64 {
    let sym_a = mol.get_atom(a).ok().and_then(|x| x.get_str("element"));
    let sym_b = mol.get_atom(b).ok().and_then(|x| x.get_str("element"));
    let r_a = sym_a
        .and_then(Element::by_symbol)
        .map(|e| e.covalent_radius() as f64)
        .unwrap_or(0.77);
    let r_b = sym_b
        .and_then(Element::by_symbol)
        .map(|e| e.covalent_radius() as f64)
        .unwrap_or(0.77);
    let order = bond_order_between(mol, a, b);

    let mut length = r_a + r_b;
    if order >= 2.5 {
        length *= 0.86;
    } else if order >= 1.5 {
        length *= 0.92;
    }
    length.clamp(0.9, 2.2)
}

fn bond_order_between(mol: &MolGraph, a: AtomId, b: AtomId) -> f64 {
    mol.neighbor_bonds(a)
        .find_map(|(nbr, order)| (nbr == b).then_some(order))
        .unwrap_or(1.0)
}

fn canonical_pair(i: usize, j: usize) -> (usize, usize) {
    if i < j { (i, j) } else { (j, i) }
}
