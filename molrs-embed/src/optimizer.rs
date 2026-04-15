//! Lightweight geometry optimizer used by the Embed pipeline.
//!
//! Energy terms (all first-principles, no empirical force-field parameters):
//! - Bond springs: k(r − r₀)²  with r₀ from covalent radii
//! - Angle springs: k(d − d₀)² as 1-3 distance constraint from VSEPR
//! - Pair repulsion: k(σ/r)¹² soft wall for non-bonded contacts
//! - **sp2 planarity (OOP)**: k·h² penalising out-of-plane displacement
//! - **Chiral volume**: k·max(0, −sign·V)² enforcing tetrahedral chirality
//!
//! Optimisers: steepest descent, conjugate gradients, **L-BFGS**.

use std::collections::{HashMap, HashSet, VecDeque};

use super::geom::{add, cross, dot, norm, normalize, scale, sub};
use molrs::element::Element;
use molrs::error::MolRsError;
use molrs::molgraph::{AtomId, MolGraph};
use molrs::stereo::{TetrahedralStereo, assign_stereo_from_3d, find_chiral_centers};

// ───────────────────── energy term structs ─────────────────────

#[derive(Debug, Clone, Copy)]
struct SpringTerm {
    i: usize,
    j: usize,
    r0: f64,
    k: f64,
}

#[derive(Debug, Clone, Copy)]
struct PairRepulsion {
    i: usize,
    j: usize,
    sigma: f64,
    k: f64,
}

/// Out-of-plane term for sp2 planarity.
/// Plane defined by (center, i, j); atom k should lie in that plane.
#[derive(Debug, Clone, Copy)]
struct OopTerm {
    center: usize,
    i: usize,
    j: usize,
    k: usize,
    k_oop: f64,
}

/// Chiral volume constraint.
/// V = (r[n1]−r[n0]) · ((r[n2]−r[n0]) × (r[n3]−r[n0]))
/// Penalised when sign(V) ≠ target_sign.
#[derive(Debug, Clone, Copy)]
struct ChiralTerm {
    neighbors: [usize; 4],
    target_sign: f64,
    k: f64,
}

// ───────────────────── EnergyModel ─────────────────────

/// Internal energy model for Embed minimization.
#[derive(Debug, Clone)]
pub(crate) struct EnergyModel {
    atom_ids: Vec<AtomId>,
    bond_terms: Vec<SpringTerm>,
    angle_terms: Vec<SpringTerm>,
    pair_terms: Vec<PairRepulsion>,
    oop_terms: Vec<OopTerm>,
    chiral_terms: Vec<ChiralTerm>,
}

/// Result of a minimization run.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MinResult {
    pub steps: usize,
    pub converged: bool,
    pub energy: f64,
}

impl EnergyModel {
    /// Build an optimization model from molecular topology and elements.
    pub(crate) fn from_mol(mol: &MolGraph) -> Self {
        let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
        let id_to_idx: HashMap<AtomId, usize> = atom_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let mut bond_terms = Vec::new();
        let mut excluded_pairs: HashSet<(usize, usize)> = HashSet::new();
        let mut vdw = vec![1.7_f64; atom_ids.len()];

        for (i, atom_id) in atom_ids.iter().copied().enumerate() {
            let radius = mol
                .get_atom(atom_id)
                .ok()
                .and_then(|a| a.get_str("element"))
                .and_then(Element::by_symbol)
                .map(|e| e.vdw_radius() as f64)
                .unwrap_or(1.7);
            vdw[i] = radius;
        }

        for (_, bond) in mol.bonds() {
            let ai = bond.atoms[0];
            let aj = bond.atoms[1];
            let Some(&i) = id_to_idx.get(&ai) else {
                continue;
            };
            let Some(&j) = id_to_idx.get(&aj) else {
                continue;
            };
            let order = bond_order_between(mol, ai, aj);
            let r0 = ideal_bond_length(mol, ai, aj);
            let k = 160.0 * order.clamp(1.0, 3.0);
            bond_terms.push(SpringTerm { i, j, r0, k });
            excluded_pairs.insert(minmax(i, j));
        }

        let mut angle_terms = Vec::new();
        for center in atom_ids.iter().copied() {
            let mut neigh: Vec<usize> = mol
                .neighbors(center)
                .filter_map(|nid| id_to_idx.get(&nid).copied())
                .collect();
            neigh.sort_unstable();
            if neigh.len() < 2 {
                continue;
            }
            let theta0 = ideal_angle_for_center(mol, center);
            for a in 0..neigh.len() {
                for b in (a + 1)..neigh.len() {
                    let i = neigh[a];
                    let k = neigh[b];
                    excluded_pairs.insert(minmax(i, k));
                    let r1 = ideal_bond_length(mol, atom_ids[i], center);
                    let r2 = ideal_bond_length(mol, center, atom_ids[k]);
                    let d0 = (r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * theta0.cos())
                        .max(0.1)
                        .sqrt();
                    angle_terms.push(SpringTerm {
                        i,
                        j: k,
                        r0: d0,
                        k: 18.0,
                    });
                }
            }
        }

        let mut pair_terms = Vec::new();
        for i in 0..atom_ids.len() {
            for j in (i + 1)..atom_ids.len() {
                if excluded_pairs.contains(&minmax(i, j)) {
                    continue;
                }
                let sigma = 0.70 * (vdw[i] + vdw[j]);
                pair_terms.push(PairRepulsion {
                    i,
                    j,
                    sigma,
                    k: 0.015,
                });
            }
        }

        // ── sp2 planarity (OOP) terms ──
        let mut oop_terms = Vec::new();
        for (idx, &atom_id) in atom_ids.iter().enumerate() {
            let theta = ideal_angle_for_center(mol, atom_id);
            // Only sp2 centres (≈ 120°).
            if (theta - 2.094).abs() > 0.1 {
                continue;
            }
            let neighbors: Vec<usize> = mol
                .neighbors(atom_id)
                .filter_map(|nid| id_to_idx.get(&nid).copied())
                .collect();
            if neighbors.len() != 3 {
                continue;
            }
            // For each neighbour as the "out-of-plane" atom, plane = (center, other two).
            for m in 0..3 {
                let out = neighbors[m];
                let in1 = neighbors[(m + 1) % 3];
                let in2 = neighbors[(m + 2) % 3];
                oop_terms.push(OopTerm {
                    center: idx,
                    i: in1,
                    j: in2,
                    k: out,
                    k_oop: 20.0,
                });
            }
        }

        Self {
            atom_ids,
            bond_terms,
            angle_terms,
            pair_terms,
            oop_terms,
            chiral_terms: Vec::new(),
        }
    }

    /// Add chiral-volume constraints inferred from the molecule's 3D geometry
    /// or explicit stereo annotations.
    pub(crate) fn with_chiral_constraints(mut self, mol: &MolGraph) -> Self {
        let id_to_idx: HashMap<AtomId, usize> = self
            .atom_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let chiral_centers = find_chiral_centers(mol);
        let geo_stereo = assign_stereo_from_3d(mol);

        for center_id in chiral_centers {
            // Prefer explicit annotation, fall back to geometry.
            let annotated = mol
                .get_atom(center_id)
                .ok()
                .and_then(|a| a.get_str("stereo").map(|s| s.to_string()));
            let target_sign = match annotated.as_deref() {
                Some("CW") => -1.0,
                Some("CCW") => 1.0,
                _ => match geo_stereo.get(&center_id) {
                    Some(TetrahedralStereo::CW) => -1.0,
                    Some(TetrahedralStereo::CCW) => 1.0,
                    _ => continue,
                },
            };

            let neighbors: Vec<usize> = mol
                .neighbors(center_id)
                .filter_map(|nid| id_to_idx.get(&nid).copied())
                .collect();
            if neighbors.len() >= 4 {
                self.chiral_terms.push(ChiralTerm {
                    neighbors: [neighbors[0], neighbors[1], neighbors[2], neighbors[3]],
                    target_sign,
                    k: 30.0,
                });
            }
        }

        self
    }

    pub(crate) fn read_coords_from_mol(&self, mol: &MolGraph) -> Vec<[f64; 3]> {
        let mut coords = vec![[0.0; 3]; self.atom_ids.len()];
        for (i, atom_id) in self.atom_ids.iter().copied().enumerate() {
            if let Ok(atom) = mol.get_atom(atom_id) {
                coords[i] = [
                    atom.get_f64("x").unwrap_or(0.0),
                    atom.get_f64("y").unwrap_or(0.0),
                    atom.get_f64("z").unwrap_or(0.0),
                ];
            }
        }
        coords
    }

    pub(crate) fn write_coords_to_mol(
        &self,
        mol: &mut MolGraph,
        coords: &[[f64; 3]],
    ) -> Result<(), MolRsError> {
        for (i, atom_id) in self.atom_ids.iter().copied().enumerate() {
            let atom = mol.get_atom_mut(atom_id)?;
            atom.set("x", coords[i][0]);
            atom.set("y", coords[i][1]);
            atom.set("z", coords[i][2]);
        }
        Ok(())
    }

    pub(crate) fn energy(&self, coords: &[[f64; 3]]) -> f64 {
        let mut e = 0.0;
        for t in &self.bond_terms {
            e += spring_energy(*t, coords);
        }
        for t in &self.angle_terms {
            e += spring_energy(*t, coords);
        }
        for t in &self.pair_terms {
            e += repulsion_energy(*t, coords);
        }
        for t in &self.oop_terms {
            e += oop_energy(t, coords);
        }
        for t in &self.chiral_terms {
            e += chiral_energy(t, coords);
        }
        e
    }

    pub(crate) fn energy_and_gradient(&self, coords: &[[f64; 3]], grad: &mut [[f64; 3]]) -> f64 {
        for g in grad.iter_mut() {
            *g = [0.0, 0.0, 0.0];
        }
        let mut e = 0.0;

        for t in &self.bond_terms {
            e += spring_energy_grad(*t, coords, grad);
        }
        for t in &self.angle_terms {
            e += spring_energy_grad(*t, coords, grad);
        }
        for t in &self.pair_terms {
            e += repulsion_energy_grad(*t, coords, grad);
        }
        for t in &self.oop_terms {
            e += oop_energy_grad(t, coords, grad);
        }
        for t in &self.chiral_terms {
            e += chiral_energy_grad(t, coords, grad);
        }

        e
    }
}

// ───────────────────── steepest descent ─────────────────────

pub(crate) fn steepest_descent(
    model: &EnergyModel,
    coords: &mut [[f64; 3]],
    max_steps: usize,
    init_step: f64,
    grad_tol: f64,
) -> MinResult {
    if coords.is_empty() {
        return MinResult {
            steps: 0,
            converged: true,
            energy: 0.0,
        };
    }

    let mut grad = vec![[0.0; 3]; coords.len()];
    let mut direction = vec![[0.0; 3]; coords.len()];
    let mut trial = coords.to_vec();
    let mut energy = model.energy_and_gradient(coords, &mut grad);

    for step in 0..max_steps {
        let rms = grad_rms(&grad);
        if rms < grad_tol {
            return MinResult {
                steps: step,
                converged: true,
                energy,
            };
        }

        for i in 0..coords.len() {
            direction[i] = scale(grad[i], -1.0);
        }

        let (accepted, e_next) =
            line_search(model, coords, &direction, &mut trial, energy, init_step, 10);
        if !accepted {
            return MinResult {
                steps: step + 1,
                converged: false,
                energy,
            };
        }
        energy = e_next;
        model.energy_and_gradient(coords, &mut grad);
    }

    MinResult {
        steps: max_steps,
        converged: grad_rms(&grad) < grad_tol,
        energy,
    }
}

// ───────────────────── conjugate gradients ─────────────────────

#[allow(dead_code)]
pub(crate) fn conjugate_gradients(
    model: &EnergyModel,
    coords: &mut [[f64; 3]],
    max_steps: usize,
    init_step: f64,
    grad_tol: f64,
) -> MinResult {
    if coords.is_empty() {
        return MinResult {
            steps: 0,
            converged: true,
            energy: 0.0,
        };
    }

    let mut grad = vec![[0.0; 3]; coords.len()];
    let mut grad_prev = vec![[0.0; 3]; coords.len()];
    let mut direction = vec![[0.0; 3]; coords.len()];
    let mut trial = coords.to_vec();

    let mut energy = model.energy_and_gradient(coords, &mut grad);
    for i in 0..coords.len() {
        direction[i] = scale(grad[i], -1.0);
    }

    for step in 0..max_steps {
        let rms = grad_rms(&grad);
        if rms < grad_tol {
            return MinResult {
                steps: step,
                converged: true,
                energy,
            };
        }

        let (accepted, e_next) =
            line_search(model, coords, &direction, &mut trial, energy, init_step, 10);
        if !accepted {
            return MinResult {
                steps: step + 1,
                converged: false,
                energy,
            };
        }
        energy = e_next;

        grad_prev.copy_from_slice(&grad);
        model.energy_and_gradient(coords, &mut grad);

        let mut numer = 0.0;
        let mut denom = 0.0;
        for i in 0..coords.len() {
            let g = grad[i];
            let gp = grad_prev[i];
            numer += dot(g, sub(g, gp));
            denom += dot(gp, gp);
        }
        let beta = if denom > 1e-20 {
            (numer / denom).max(0.0)
        } else {
            0.0
        };

        for i in 0..coords.len() {
            direction[i] = add(scale(grad[i], -1.0), scale(direction[i], beta));
        }

        // Restart if direction is no longer downhill.
        let mut dg = 0.0;
        for i in 0..coords.len() {
            dg += dot(direction[i], grad[i]);
        }
        if dg >= 0.0 {
            for i in 0..coords.len() {
                direction[i] = scale(grad[i], -1.0);
            }
        }
    }

    MinResult {
        steps: max_steps,
        converged: grad_rms(&grad) < grad_tol,
        energy,
    }
}

// ───────────────────── L-BFGS ─────────────────────

/// Limited-memory BFGS optimizer with two-loop recursion.
pub(crate) fn lbfgs(
    model: &EnergyModel,
    coords: &mut [[f64; 3]],
    max_steps: usize,
    grad_tol: f64,
) -> MinResult {
    const M: usize = 7; // history depth

    if coords.is_empty() {
        return MinResult {
            steps: 0,
            converged: true,
            energy: 0.0,
        };
    }

    let n = coords.len();
    let mut grad = vec![[0.0; 3]; n];
    let mut energy = model.energy_and_gradient(coords, &mut grad);

    let mut s_hist: VecDeque<Vec<[f64; 3]>> = VecDeque::new();
    let mut y_hist: VecDeque<Vec<[f64; 3]>> = VecDeque::new();
    let mut rho_hist: VecDeque<f64> = VecDeque::new();

    let mut x_prev = coords.to_vec();
    let mut g_prev = grad.clone();
    let mut trial = coords.to_vec();

    for step in 0..max_steps {
        let rms = grad_rms(&grad);
        if rms < grad_tol {
            return MinResult {
                steps: step,
                converged: true,
                energy,
            };
        }

        // ── two-loop recursion ──
        let mut q = grad.clone();
        let hist_len = s_hist.len();
        let mut alphas = vec![0.0_f64; hist_len];

        for i in (0..hist_len).rev() {
            alphas[i] = rho_hist[i] * flat_dot(&s_hist[i], &q);
            flat_axpy(&mut q, -alphas[i], &y_hist[i]);
        }

        // Initial Hessian approximation: H₀ = γ I.
        let gamma = if hist_len > 0 {
            let sy = flat_dot(&s_hist[hist_len - 1], &y_hist[hist_len - 1]);
            let yy = flat_dot(&y_hist[hist_len - 1], &y_hist[hist_len - 1]);
            if yy > 1e-20 { sy / yy } else { 1.0 }
        } else {
            1.0
        };

        // r = γ q  (apply H₀)
        let mut r: Vec<[f64; 3]> = q.iter().map(|&qi| scale(qi, gamma)).collect();

        for i in 0..hist_len {
            let beta = rho_hist[i] * flat_dot(&y_hist[i], &r);
            flat_axpy(&mut r, alphas[i] - beta, &s_hist[i]);
        }

        // direction = −r
        let direction: Vec<[f64; 3]> = r.iter().map(|&ri| scale(ri, -1.0)).collect();

        // Line search.
        let init_step = 1.0_f64.min(1.0 / (rms + 1e-10));
        let (accepted, e_next) =
            line_search(model, coords, &direction, &mut trial, energy, init_step, 15);

        if !accepted {
            // Reset history and try steepest descent step.
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            let sd_dir: Vec<[f64; 3]> = grad.iter().map(|&gi| scale(gi, -1.0)).collect();
            let (accepted2, e_next2) =
                line_search(model, coords, &sd_dir, &mut trial, energy, 0.01, 10);
            if !accepted2 {
                return MinResult {
                    steps: step + 1,
                    converged: false,
                    energy,
                };
            }
            energy = e_next2;
            x_prev.copy_from_slice(coords);
            g_prev.copy_from_slice(&grad);
            model.energy_and_gradient(coords, &mut grad);
            continue;
        }
        energy = e_next;

        // Update history: s = x_new − x_old, y = g_new − g_old.
        let s: Vec<[f64; 3]> = (0..n).map(|i| sub(coords[i], x_prev[i])).collect();
        x_prev.copy_from_slice(coords);
        g_prev.copy_from_slice(&grad);
        model.energy_and_gradient(coords, &mut grad);
        let y: Vec<[f64; 3]> = (0..n).map(|i| sub(grad[i], g_prev[i])).collect();

        let sy = flat_dot(&s, &y);
        if sy > 1e-20 {
            if s_hist.len() >= M {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            s_hist.push_back(s);
            y_hist.push_back(y);
            rho_hist.push_back(1.0 / sy);
        }
    }

    MinResult {
        steps: max_steps,
        converged: grad_rms(&grad) < grad_tol,
        energy,
    }
}

// ───────────────────── line search & helpers ─────────────────────

fn line_search(
    model: &EnergyModel,
    coords: &mut [[f64; 3]],
    direction: &[[f64; 3]],
    trial: &mut [[f64; 3]],
    energy: f64,
    init_step: f64,
    max_backtracks: usize,
) -> (bool, f64) {
    let mut step = init_step;
    for _ in 0..max_backtracks {
        for i in 0..coords.len() {
            trial[i] = add(coords[i], scale(direction[i], step));
        }
        let e_trial = model.energy(trial);
        if e_trial < energy {
            coords.copy_from_slice(trial);
            return (true, e_trial);
        }
        step *= 0.5;
    }
    (false, energy)
}

fn grad_rms(grad: &[[f64; 3]]) -> f64 {
    if grad.is_empty() {
        return 0.0;
    }
    let s2: f64 = grad.iter().map(|g| dot(*g, *g)).sum();
    (s2 / grad.len() as f64).sqrt()
}

fn flat_dot(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| dot(*ai, *bi)).sum()
}

fn flat_axpy(y: &mut [[f64; 3]], alpha: f64, x: &[[f64; 3]]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = add(*yi, scale(*xi, alpha));
    }
}

// ───────────────────── spring energy / gradient ─────────────────────

fn spring_energy(term: SpringTerm, coords: &[[f64; 3]]) -> f64 {
    let d = sub(coords[term.i], coords[term.j]);
    let r = norm(d).max(1e-10);
    let dr = r - term.r0;
    term.k * dr * dr
}

fn spring_energy_grad(term: SpringTerm, coords: &[[f64; 3]], grad: &mut [[f64; 3]]) -> f64 {
    let d = sub(coords[term.i], coords[term.j]);
    let r = norm(d).max(1e-10);
    let dr = r - term.r0;
    let e = term.k * dr * dr;
    let gscale = 2.0 * term.k * dr / r;
    let g = scale(d, gscale);
    grad[term.i] = add(grad[term.i], g);
    grad[term.j] = sub(grad[term.j], g);
    e
}

// ───────────────────── pair repulsion ─────────────────────

fn repulsion_energy(term: PairRepulsion, coords: &[[f64; 3]]) -> f64 {
    let d = sub(coords[term.i], coords[term.j]);
    let r = norm(d).max(1e-10);
    if r >= term.sigma {
        return 0.0;
    }
    let x = term.sigma / r;
    let x6 = x.powi(6);
    term.k * x6 * x6
}

fn repulsion_energy_grad(term: PairRepulsion, coords: &[[f64; 3]], grad: &mut [[f64; 3]]) -> f64 {
    let d = sub(coords[term.i], coords[term.j]);
    let r = norm(d).max(1e-10);
    if r >= term.sigma {
        return 0.0;
    }
    let x = term.sigma / r;
    let x6 = x.powi(6);
    let x12 = x6 * x6;
    let e = term.k * x12;
    let d_e_dr = -12.0 * term.k * x12 / r;
    let g = scale(normalize(d), d_e_dr);
    grad[term.i] = add(grad[term.i], g);
    grad[term.j] = sub(grad[term.j], g);
    e
}

// ───────────────────── out-of-plane (sp2 planarity) ─────────────────────

/// E_oop = k · h²  where h is the height of atom `k` above the plane
/// defined by (center, i, j).
fn oop_energy(term: &OopTerm, coords: &[[f64; 3]]) -> f64 {
    let u = sub(coords[term.i], coords[term.center]);
    let v = sub(coords[term.j], coords[term.center]);
    let w = sub(coords[term.k], coords[term.center]);
    let n = cross(u, v);
    let n_len = norm(n);
    if n_len < 1e-12 {
        return 0.0;
    }
    let n_hat = scale(n, 1.0 / n_len);
    let h = dot(w, n_hat);
    term.k_oop * h * h
}

/// Analytical gradient of the OOP energy.
///
/// h = w · n̂,  n = u × v,  u = r_i − r_c,  v = r_j − r_c,  w = r_k − r_c
///
/// ∂h/∂r_k = n̂
/// ∂h/∂r_i = (v × w_⊥) / |n|   where w_⊥ = w − h n̂
/// ∂h/∂r_j = (w_⊥ × u) / |n|
/// ∂h/∂r_c = −(∂h/∂r_k + ∂h/∂r_i + ∂h/∂r_j)
fn oop_energy_grad(term: &OopTerm, coords: &[[f64; 3]], grad: &mut [[f64; 3]]) -> f64 {
    let c = term.center;
    let u = sub(coords[term.i], coords[c]);
    let v = sub(coords[term.j], coords[c]);
    let w = sub(coords[term.k], coords[c]);
    let n = cross(u, v);
    let n_len = norm(n);
    if n_len < 1e-12 {
        return 0.0;
    }
    let inv_n = 1.0 / n_len;
    let n_hat = scale(n, inv_n);
    let h = dot(w, n_hat);
    let e = term.k_oop * h * h;
    let g_scale = 2.0 * term.k_oop * h;

    let w_perp = sub(w, scale(n_hat, h));
    let dh_dk = n_hat;
    let dh_di = scale(cross(v, w_perp), inv_n);
    let dh_dj = scale(cross(w_perp, u), inv_n);
    let dh_dc = scale(add(add(dh_dk, dh_di), dh_dj), -1.0);

    grad[term.k] = add(grad[term.k], scale(dh_dk, g_scale));
    grad[term.i] = add(grad[term.i], scale(dh_di, g_scale));
    grad[term.j] = add(grad[term.j], scale(dh_dj, g_scale));
    grad[c] = add(grad[c], scale(dh_dc, g_scale));
    e
}

// ───────────────────── chiral volume constraint ─────────────────────

/// E_chiral = k · max(0, −sign · V)²
/// V = (r[n1]−r[n0]) · ((r[n2]−r[n0]) × (r[n3]−r[n0]))
fn chiral_energy(term: &ChiralTerm, coords: &[[f64; 3]]) -> f64 {
    let [n0, n1, n2, n3] = term.neighbors;
    let u = sub(coords[n1], coords[n0]);
    let v = sub(coords[n2], coords[n0]);
    let w = sub(coords[n3], coords[n0]);
    let vol = dot(u, cross(v, w));
    let f = -term.target_sign * vol;
    if f > 0.0 { term.k * f * f } else { 0.0 }
}

/// Analytical gradient of chiral volume penalty.
///
/// ∂V/∂r[n1] = v × w
/// ∂V/∂r[n2] = w × u
/// ∂V/∂r[n3] = u × v
/// ∂V/∂r[n0] = −(∂V/∂r[n1] + ∂V/∂r[n2] + ∂V/∂r[n3])
///
/// ∂E/∂x = 2k · V · ∂V/∂x   (only when chirality is wrong)
fn chiral_energy_grad(term: &ChiralTerm, coords: &[[f64; 3]], grad: &mut [[f64; 3]]) -> f64 {
    let [n0, n1, n2, n3] = term.neighbors;
    let u = sub(coords[n1], coords[n0]);
    let v = sub(coords[n2], coords[n0]);
    let w = sub(coords[n3], coords[n0]);
    let vol = dot(u, cross(v, w));
    let f = -term.target_sign * vol;
    if f <= 0.0 {
        return 0.0;
    }
    let e = term.k * f * f;
    // ∂E/∂V = 2k · f · (−target_sign) = −2k · target_sign · f
    // but f = −target_sign · V, so ∂E/∂V = 2k · V.
    let g_scale = 2.0 * term.k * vol;

    let dv_n1 = cross(v, w);
    let dv_n2 = cross(w, u);
    let dv_n3 = cross(u, v);
    let dv_n0 = scale(add(add(dv_n1, dv_n2), dv_n3), -1.0);

    grad[n0] = add(grad[n0], scale(dv_n0, g_scale));
    grad[n1] = add(grad[n1], scale(dv_n1, g_scale));
    grad[n2] = add(grad[n2], scale(dv_n2, g_scale));
    grad[n3] = add(grad[n3], scale(dv_n3, g_scale));
    e
}

// ───────────────────── element helpers ─────────────────────

fn minmax(i: usize, j: usize) -> (usize, usize) {
    if i < j { (i, j) } else { (j, i) }
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
