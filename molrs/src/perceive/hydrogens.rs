//! Hydrogen addition for molecular graphs.
//!
//! Hydrogen addition ([`Perceive::find_hydrogens`](crate::perceive::Perceive::find_hydrogens))
//! computes the number of implicit hydrogens each heavy atom requires (based on its element's default valences and the sum of its current
//! bond orders) and returns a **new** [`Atomistic`] with explicit H atoms added.
//!
//! [`remove_hydrogens`] does the inverse: it returns a new [`Atomistic`] with
//! all terminal explicit hydrogen atoms removed — terminal by `bonds`-kind
//! degree, and never a hydrogen that is the handle of a port.
//!
//! # Immutability
//! The original `MolGraph` is never mutated; a clone is returned.
//!
//! # Bond-order convention
//! Localized bond counts are read from the bond's `bond_number`, and
//! aromaticity from its `bond_type` — the two are separate questions.
//! If the property is absent the bond is assumed to be a single bond (1.0).
//! Aromatic bonds should be stored as 1.5.
//!
//! # Formal-charge correction
//! A formal charge is folded into the element identity, not into the bond
//! demand: the valence list of `Z − formal_charge` is used. This is RDKit's
//! `getEffectiveAtomicNum` rule and gets the group-13/14 cation case right
//! (e.g. `[CH3+]` → C(Z=6) − (+1) = B(Z=5), valence 3 → 3 H, rather than the
//! naive `bond_order_sum − formal_charge` which over-counts to 5 H). For the
//! late atoms N/O/F the two formulations happen to agree, but for early atoms
//! (B, C, Si, …) they diverge, which is exactly the bug this rule fixes.

use std::collections::HashSet;

use crate::core::Atom;
use crate::core::Atomistic;
use crate::core::BondOrder;
use crate::core::NodeId;
use crate::op::vec3::{cross, norm};
use molrs::core::Element;
use molrs::core::MolRsError;

/// Name of the relation kind whose members mark a fragment attachment point.
///
/// The kind [`MolGraph::add_port`](crate::core::MolGraph::add_port)
/// registers ([`crate::core::keys::PORTS`]); it is matched by name because
/// any graph may carry ports.
const PORTS_KIND: &str = "ports";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return a new [`Atomistic`] with explicit hydrogen atoms added to every
/// heavy atom that has unfilled valence.
///
/// Hydrogen atoms already present (symbol == "H") are not modified.
///
/// When a heavy atom has `x`/`y`/`z` components, each new H is placed at a
/// standard X–H length along a tetrahedral valence-completing direction
/// (initial geometry only — force fields may refine). When the heavy lacks
/// coordinates, H is added without `x`/`y`/`z` (topology-only path).
///
/// Starts from [`Clone`] of `mol` (handles preserved on the skeleton); only
/// new H atoms and bonds are appended, so parent angles/dihedrals remain.
///
/// # Errors
///
/// [`MolRsError`] when bonding a fresh hydrogen to its heavy atom fails, which
/// means the graph handed in holds a stale atom handle. No public constructor
/// of [`Atomistic`] can produce such a graph, so the case is unreachable in
/// practice; the `Result` is here so that the invariant is *returned* rather
/// than asserted by a panic on a caller's thread.
pub(crate) fn add_hydrogens(mol: &Atomistic) -> Result<Atomistic, MolRsError> {
    let mut new_mol = mol.clone();

    // Mass (amu) of the hydrogens this call appends, read once from the
    // periodic table: a literal is a second home for a number `Element` owns.
    let hydrogen = Element::by_symbol("H")
        .ok_or_else(|| MolRsError::validation("the periodic table has no entry for element H"))?;
    let h_mass = f64::from(hydrogen.atomic_mass());

    // Collect (atom_id, n_implicit_h) for all heavy atoms up front so that
    // we don't hold a borrow while mutating.
    let additions: Vec<(NodeId, u32)> = new_mol
        .atoms()
        .filter_map(|(id, atom)| {
            let sym = atom.get_str("element")?;
            if sym.eq_ignore_ascii_case("H") {
                return None; // skip existing hydrogens
            }
            let n = implicit_h_count(&new_mol, id)?;
            if n == 0 { None } else { Some((id, n)) }
        })
        .collect();

    for (heavy_id, n) in additions {
        let heavy = new_mol.get_atom(heavy_id).ok();
        let place = heavy.as_ref().and_then(|a| {
            Some((
                a.get_f64("x")?,
                a.get_f64("y")?,
                a.get_f64("z")?,
                a.get_str("element").unwrap_or("C"),
            ))
        });

        let positions: Vec<[f64; 3]> = if let Some((hx, hy, hz, elem)) = place {
            let p = [hx, hy, hz];
            let mut existing: Vec<[f64; 3]> = Vec::new();
            for (nb, _) in new_mol.neighbor_bonds(heavy_id) {
                if let Ok(na) = new_mol.get_atom(nb)
                    && let (Some(x), Some(y), Some(z)) =
                        (na.get_f64("x"), na.get_f64("y"), na.get_f64("z"))
                {
                    existing.push(unit([x - p[0], y - p[1], z - p[2]]));
                }
            }
            let dirs = cap_directions(&existing, n as usize);
            let len = cap_length(elem);
            dirs.into_iter()
                .map(|d| [p[0] + len * d[0], p[1] + len * d[1], p[2] + len * d[2]])
                .collect()
        } else {
            vec![[0.0, 0.0, 0.0]; n as usize] // placeholder; coords omitted below
        };

        let place_coords = place.is_some();
        for pos in positions.iter().take(n as usize) {
            let mut h = Atom::new();
            h.set("element", "H");
            h.set("mass", h_mass);
            if place_coords {
                h.set("x", pos[0]);
                h.set("y", pos[1]);
                h.set("z", pos[2]);
            }
            let h_id = new_mol.add_atom(h);
            // `Atomistic::add_bond` already stamps the bond Single/Single, so
            // there is nothing left to set. Both endpoints exist (one is the
            // atom just added), so the only error `add_bond` has — an unknown
            // endpoint — is a broken invariant; it travels out to the caller
            // rather than being swallowed, which would leave the new hydrogen
            // floating unbonded.
            new_mol.add_bond(heavy_id, h_id)?;
        }
    }

    Ok(new_mol)
}

// ---------------------------------------------------------------------------
// Initial X–H geometry (port of molpy.core.capping; values are starting
// guesses for downstream minimization, not equilibrium force-field lengths).
// ---------------------------------------------------------------------------

/// Cap X–H bond length (Å) keyed by heavy element; default 1.0 Å.
fn cap_length(element: &str) -> f64 {
    match element {
        "C" | "c" => 1.09,
        "N" | "n" => 1.01,
        "O" | "o" => 0.96,
        "S" | "s" => 1.34,
        _ => 1.0,
    }
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    if n > 1e-9 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        v
    }
}

fn orthogonal(v: [f64; 3]) -> [f64; 3] {
    let seed = if v[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    unit(cross(v, seed))
}

/// `k` unit directions completing ~sp3 (tetrahedral) coordination.
fn cap_directions(existing: &[[f64; 3]], k: usize) -> Vec<[f64; 3]> {
    let n = existing.len();
    let mut caps: Vec<[f64; 3]> = if n >= 3 {
        let s = [
            existing[0][0] + existing[1][0] + existing[2][0],
            existing[0][1] + existing[1][1] + existing[2][1],
            existing[0][2] + existing[1][2] + existing[2][2],
        ];
        vec![unit([-s[0], -s[1], -s[2]])]
    } else if n == 2 {
        let u1 = existing[0];
        let u2 = existing[1];
        let bisector = unit([-(u1[0] + u2[0]), -(u1[1] + u2[1]), -(u1[2] + u2[2])]);
        let cr = cross(u1, u2);
        let normal = if norm(cr) < 1e-6 {
            orthogonal(u1)
        } else {
            unit(cr)
        };
        let half = 54.75_f64.to_radians();
        let (c, s) = (half.cos(), half.sin());
        vec![
            unit([
                c * bisector[0] + s * normal[0],
                c * bisector[1] + s * normal[1],
                c * bisector[2] + s * normal[2],
            ]),
            unit([
                c * bisector[0] - s * normal[0],
                c * bisector[1] - s * normal[1],
                c * bisector[2] - s * normal[2],
            ]),
        ]
    } else if n == 1 {
        let u = existing[0];
        let e1 = orthogonal(u);
        let e2 = unit(cross(u, e1));
        let theta = 109.47_f64.to_radians();
        let (ct, st) = (theta.cos(), theta.sin());
        [0.0_f64, 120.0, 240.0]
            .into_iter()
            .map(|phi_deg| {
                let phi = phi_deg.to_radians();
                let (cp, sp) = (phi.cos(), phi.sin());
                unit([
                    ct * u[0] + st * (cp * e1[0] + sp * e2[0]),
                    ct * u[1] + st * (cp * e1[1] + sp * e2[1]),
                    ct * u[2] + st * (cp * e1[2] + sp * e2[2]),
                ])
            })
            .collect()
    } else {
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
        .into_iter()
        .map(unit)
        .collect()
    };
    caps.truncate(k);
    // If fewer directions than k (shouldn't for tetrahedral cases), pad with +x-ish
    while caps.len() < k {
        caps.push([1.0, 0.0, 0.0]);
    }
    caps
}

/// Return a new [`Atomistic`] with all terminal explicit hydrogen atoms removed.
///
/// # The rule
///
/// An explicit hydrogen is stripped when **both** hold:
///
/// 1. its degree is exactly one, counted over the `bonds` kind alone
///    ([`Atomistic::neighbor_bonds`]) — the standard cheminformatics
///    convention for a "non-bridging" H;
/// 2. it takes part in no relation of the kind named `ports`.
///
/// # Why the degree is counted over `bonds` only
///
/// [`MolGraph::neighbors`](crate::core::MolGraph::neighbors) is
/// kind-blind: it walks every arity-2 relation on the graph, so a hydrogen
/// that a caller also recorded in some other 2-ary kind reads as degree two
/// and is spared for a reason that has nothing to do with its bonding. The
/// question clause 1 asks is chemical — how many bonds does this hydrogen
/// have — so it is asked of the bond kind.
///
/// # Why a port handle is kept
///
/// A port is `(anchor, handle)`: the handle is a real bonded hydrogen that
/// *also* carries the attachment point where another fragment joins. Removing
/// it would cascade-delete the port row and leave the fragment with no
/// recorded join site, so the handle is kept however few bonds it has.
///
/// The two clauses are independent on purpose. Before they were separated,
/// handles survived only as a side effect of clause 1 being kind-blind: the
/// port relation inflated a handle's neighbour count to two. That made an
/// ordinary repletion hydrogen on a ported anchor and a port handle
/// indistinguishable to anyone reading the code, and tied the survival of
/// every port to the arity of the kind that records it.
///
/// Incident bonds, angles, and dihedrals of a removed hydrogen are
/// cascade-deleted by [`Atomistic::remove_atom`].
///
/// The original `MolGraph` is never mutated; a clone is returned.
///
/// # Errors
///
/// [`MolRsError`] when removing a hydrogen fails, which means the graph handed
/// in holds a stale atom handle: the handles stripped here were read out of
/// this same graph a moment ago, they are distinct, and removing one node
/// never despawns another. No public constructor of [`Atomistic`] can produce
/// such a graph, so the case is unreachable in practice; the `Result` is here
/// so that the invariant is *returned* rather than asserted by a panic on a
/// caller's thread.
pub fn remove_hydrogens(mol: &Atomistic) -> Result<Atomistic, MolRsError> {
    let mut new_mol = mol.clone();

    // Every node named by a `ports` relation, gathered before the first
    // removal. The relations are scanned rather than the adjacency index
    // because that index holds arity-2 kinds only, and "participates in a
    // port" must not depend on how wide a port row happens to be.
    let port_nodes: HashSet<NodeId> = match new_mol.kind_id(PORTS_KIND) {
        Some(kind) => new_mol
            .relations(kind)
            .flat_map(|(_, rel)| rel.nodes.into_iter())
            .collect(),
        None => HashSet::new(),
    };

    let h_ids: Vec<NodeId> = new_mol
        .atoms()
        .filter_map(|(id, atom)| {
            let sym = atom.get_str("element")?;
            if !sym.eq_ignore_ascii_case("H") || port_nodes.contains(&id) {
                return None;
            }
            if new_mol.neighbor_bonds(id).count() == 1 {
                Some(id)
            } else {
                None
            }
        })
        .collect();

    for h_id in h_ids {
        // `remove_atom`'s only failure is an unknown handle, and every handle
        // here is live (see `# Errors`). Returning it keeps a broken node
        // table a value the caller can handle instead of a panic.
        new_mol.remove_atom(h_id)?;
    }
    Ok(new_mol)
}

// ---------------------------------------------------------------------------
// Implicit-H calculation
// ---------------------------------------------------------------------------

/// Compute the number of hydrogens to add to `atom_id`.
///
/// Returns `None` if the atom has no recognisable element symbol or if its
/// element has no defined default valences (e.g. noble gases).
pub fn implicit_h_count(mol: &Atomistic, atom_id: NodeId) -> Option<u32> {
    let atom = mol.get_atom(atom_id).ok()?;

    // A declared hydrogen count (SMILES bracket atom) is exact: `[nH]` has one
    // hydrogen and `[C]` has none, whatever the valence model would prefer.
    if let Some(h) = atom.get("h_count").and_then(|v| v.as_f64()) {
        return Some(h.max(0.0).round() as u32);
    }

    let sym = atom.get_str("element")?;
    let element = Element::by_symbol(sym)?;

    // RDKit charged-atom valence rule (`getEffectiveAtomicNum` +
    // `calculateImplicitValence` in `Code/GraphMol/Atom.cpp`):
    //
    //   1. Z_eff = Z − formal_charge  (cation → element one place earlier;
    //      anion → one place later). The valence list is taken from Z_eff,
    //      NOT from the bare element with a charge-adjusted demand.
    //   2. demand = sum of incident bond orders (no charge term here).
    //   3. target = smallest Z_eff valence ≥ demand.
    //   4. implicit_h = target − demand.
    //
    // This is what makes early atoms (B, C, Si, …) and late atoms (N, O, F)
    // behave asymmetrically under charge:
    //   [CH3+]  Z 6−(+1)=5 (B), valences [3], demand 0 → 3 H
    //   [CH3-]  Z 6−(−1)=7 (N), valences [3,5], demand 0 → 3 H
    //   [NH4+]  Z 7−(+1)=6 (C), valences [4], demand 0 → 4 H
    //   [BH4-]  Z 5−(−1)=6 (C), valences [4], demand 0 → 4 H
    //   [OH-]   Z 8−(−1)=9 (F), valences [1], demand 0 → 1 H
    //   [NH2-]  Z 7−(−1)=8 (O), valences [2], demand 0 → 2 H
    // `formal_charge` is stored as an i32-typed column, so read it through the
    // coercing `as_f64` (matching `bond_order_sum`'s order read); the strict
    // `get_f64` only matches `PropValue::F64` and would silently miss the Int
    // variant, treating every charged atom as neutral (e.g. protonating [N-]).
    let formal_charge = atom
        .get("formal_charge")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .round() as i32;

    // Fold the charge into the element identity, then read that element's
    // valence list. An out-of-range shift (or an element with no valence
    // model) means we add no hydrogens.
    let effective = element.effective_atomic_number(formal_charge)?;
    let valences = effective.default_valences();
    if valences.is_empty() {
        return None; // noble gas / effective element with no valence model
    }

    // Sum of bond orders connected to this atom (the explicit valence).
    let demand: f64 = valence_demand(mol, atom_id, valences[0]);

    // Select the smallest allowed valence ≥ the (un-charge-adjusted) demand.
    let target = valences
        .iter()
        .copied()
        .find(|&v| v as f64 >= demand - 1e-6);

    let target = target?; // if demand exceeds all valences, add nothing
    let n = target as f64 - demand;
    if n <= 0.5 {
        Some(0)
    } else {
        Some(n.round() as u32)
    }
}

/// Explicit valence of `atom_id` — the demand its existing bonds already place
/// on `lowest_valence`, the smallest valence its (charge-adjusted) element has.
///
/// # Aromatic bonds
///
/// An aromatic bond is stored with order `1.5`, but that number is a *bond*
/// property and summing it does not give an atom's valence: in every Kekulé
/// structure an aromatic atom has one σ bond per aromatic neighbour, plus at
/// most one π bond. Summing 1.5 per bond bills a ring atom with two aromatic
/// neighbours for two half-π bonds it does not both have, and bills a
/// lone-pair donor for a π bond it does not have at all — which is how
/// thiophene's S reaches 3.0, takes the S valence of 4, and grows a spurious
/// S–H.
///
/// So aromatic bonds are counted as the σ frame, and the π bond is added back
/// exactly once — only when the σ frame leaves room for it. That single test
/// separates the two donor classes without an element table:
///
/// | atom | σ | lowest valence | π? | demand | H |
/// |---|---|---|---|---|---|
/// | benzene C–H     | 2 | 4 | yes | 3 | 1 |
/// | substituted C   | 3 | 4 | yes | 4 | 0 |
/// | pyridine N      | 2 | 3 | yes | 3 | 0 |
/// | furan O         | 2 | 2 | no  | 2 | 0 |
/// | thiophene S     | 2 | 2 | no  | 2 | 0 |
/// | aromatic C=O    | 4 | 4 | no  | 4 | 0 |
///
/// (Pyrrole-type N reaches this path only when its H was *not* declared; a
/// declared `h_count` short-circuits in [`implicit_h_count`].)
///
/// A graph with integral Kekulé orders has no aromatic bonds and is summed
/// unchanged.
fn valence_demand(mol: &Atomistic, atom_id: NodeId, lowest_valence: u8) -> f64 {
    // The two facts are read from their own places: how many bonds this is
    // (the localized number) and whether it is delocalized (the class).
    let bonds: Vec<(BondOrder, f64)> = mol
        .incident_bond_ids(atom_id)
        .map(|(bid, _)| {
            let number = mol.bond_number(bid).count().max(1) as f64;
            (mol.bond_type(bid), number)
        })
        .collect();

    let n_aromatic = bonds.iter().filter(|(t, _)| t.is_aromatic()).count();
    // An aromatic bond contributes its sigma bond here; the extra pi bond is
    // added once below if the atom is still short of its lowest valence.
    let sigma: f64 = bonds
        .iter()
        .map(|(t, n)| if t.is_aromatic() { 1.0 } else { *n })
        .sum();

    if n_aromatic > 0 && sigma < lowest_valence as f64 - 1e-6 {
        sigma + 1.0
    } else {
        sigma
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BondNumber;

    fn atom(sym: &str) -> Atom {
        let mut a = Atom::new();
        a.set("element", sym);
        a
    }

    fn bond_with_order(mol: &mut Atomistic, a: NodeId, b: NodeId, order: f64) {
        let bid = mol.add_bond(a, b).expect("fixture bond");
        // The old float encoding, expressed in the two facts it conflated:
        // 1.5 meant "aromatic", every integer meant a localized count.
        if (order - 1.5).abs() < 1e-6 {
            mol.set_bond_class(bid, BondOrder::Aromatic, BondNumber::Unknown)
                .expect("fixture bond class");
        } else {
            mol.set_bond_type(bid, BondOrder::from_code(order.round() as u32))
                .expect("fixture bond type");
        }
    }

    #[test]
    fn test_methane_skeleton() {
        // Isolated C — should get 4 H.
        let mut g = Atomistic::new();
        g.add_atom(atom("C"));
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        // original unchanged
        assert_eq!(g.n_atoms(), 1);
        // result has C + 4H
        assert_eq!(result.n_atoms(), 5);
        assert_eq!(result.n_bonds(), 4);
        let n_h = result
            .atoms()
            .filter(|(_, a)| a.get_str("element") == Some("H"))
            .count();
        assert_eq!(n_h, 4);
    }

    #[test]
    fn test_ethane_c_c() {
        // C-C single bond: each C needs 3 H.
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 1.0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 8); // 2C + 6H
    }

    #[test]
    fn test_ethylene_c_double_c() {
        // C=C double bond: each C needs 2 H.
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 2.0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 6); // 2C + 4H
    }

    #[test]
    fn test_benzene_aromatic() {
        // 6-membered ring with bond order 1.5: each C should get 1 H.
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..6).map(|_| g.add_atom(atom("C"))).collect();
        for i in 0..6 {
            bond_with_order(&mut g, ids[i], ids[(i + 1) % 6], 1.5);
        }
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 12); // 6C + 6H
    }

    #[test]
    fn test_benzene_kekule() {
        // Kekule benzene: alternating single/double bonds.
        // Each C has bond_order_sum = 1+2 = 3, needs 1 H. Total = 6 H.
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..6).map(|_| g.add_atom(atom("C"))).collect();
        let orders = [2.0, 1.0, 2.0, 1.0, 2.0, 1.0];
        for i in 0..6 {
            bond_with_order(&mut g, ids[i], ids[(i + 1) % 6], orders[i]);
        }
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        let n_h = result
            .atoms()
            .filter(|(_, a)| a.get_str("element") == Some("H"))
            .count();
        assert_eq!(n_h, 6, "Kekule benzene should get 6 H, got {}", n_h);
        assert_eq!(result.n_atoms(), 12); // 6C + 6H
    }

    #[test]
    fn test_ethylene_round_trip_frame() {
        // C=C → to_frame → from_frame → add_hydrogens should give 4H not 6H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 2.0);
        let frame = g.to_frame().expect("a schema-conforming graph converts");
        let g2 = Atomistic::from_frame(&frame).unwrap();
        let result = add_hydrogens(&g2).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 6, "C=C round-trip should give 2C + 4H");
    }

    #[test]
    fn test_acetylene_round_trip_frame() {
        // C#C → to_frame → from_frame → add_hydrogens should give 2H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 3.0);
        let frame = g.to_frame().expect("a schema-conforming graph converts");
        let g2 = Atomistic::from_frame(&frame).unwrap();
        let result = add_hydrogens(&g2).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 4, "C#C round-trip should give 2C + 2H");
    }

    #[test]
    fn test_water() {
        // Isolated O → 2 H
        let mut g = Atomistic::new();
        g.add_atom(atom("O"));
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 3);
    }

    #[test]
    fn test_ammonia_like() {
        // N with 1 bond → 2 H  (valence 3)
        let mut g = Atomistic::new();
        let n = g.add_atom(atom("N"));
        let c = g.add_atom(atom("C"));
        bond_with_order(&mut g, n, c, 1.0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        // N gets 2H, C gets 3H, total = 2C + 2H(on N) + 3H(on C) = 2+5 = 7
        assert_eq!(result.n_atoms(), 7);
    }

    #[test]
    fn test_nh4_plus() {
        // NH4+: formal_charge=1 on N → needs 4 H
        let mut g = Atomistic::new();
        let mut n_atom = Atom::new();
        n_atom.set("element", "N");
        n_atom.set("formal_charge", 1.0_f64);
        let n = g.add_atom(n_atom);
        let count = implicit_h_count(&g, n).unwrap();
        assert_eq!(count, 4);
    }

    /// Build a single charged heavy atom (no heavy neighbours) and return its
    /// implicit-H count.
    fn charged_atom_h(sym: &str, fc: f64) -> u32 {
        let mut g = Atomistic::new();
        let mut a = Atom::new();
        a.set("element", sym);
        a.set("formal_charge", fc);
        let id = g.add_atom(a);
        implicit_h_count(&g, id).unwrap_or(0)
    }

    #[test]
    fn charged_single_atoms_take_their_valence_hydrogens() {
        // Charged single-heavy-atom species (the cases the old
        // bond_order_sum - formal_charge rule got wrong for group-13/14):
        assert_eq!(charged_atom_h("C", 1.0), 3, "[CH3+] -> 3 H");
        assert_eq!(charged_atom_h("C", -1.0), 3, "[CH3-] -> 3 H");
        assert_eq!(charged_atom_h("B", -1.0), 4, "[BH4-] -> 4 H");
        assert_eq!(charged_atom_h("N", 1.0), 4, "[NH4+] -> 4 H");
        assert_eq!(charged_atom_h("O", -1.0), 1, "[OH-] -> 1 H");
        assert_eq!(charged_atom_h("N", -1.0), 2, "[NH2-] -> 2 H");

        // Neutral references (unchanged by the fix):
        assert_eq!(charged_atom_h("C", 0.0), 4, "methane C -> 4 H");
        assert_eq!(charged_atom_h("O", 0.0), 2, "water O -> 2 H");
        assert_eq!(charged_atom_h("N", 0.0), 3, "ammonia N -> 3 H");
    }

    /// Like `charged_atom_h` but stores `formal_charge` as the canonical
    /// **integer** column (`PropValue::Int`) — what the parsers and the i32-typed
    /// graph schema actually emit. Guards the regression where `implicit_h_count`
    /// read the charge via the strict `get_f64` (F64-only) and silently treated
    /// every charged atom as neutral — e.g. protonating the sulfonimide [N-] in
    /// TFSI/ANI and breaking antechamber's charge balance.
    fn charged_atom_h_int(sym: &str, fc: i32) -> u32 {
        let mut g = Atomistic::new();
        let mut a = Atom::new();
        a.set("element", sym);
        a.set("formal_charge", fc); // i32 == type alias `I` → PropValue::Int
        let id = g.add_atom(a);
        implicit_h_count(&g, id).unwrap_or(0)
    }

    #[test]
    fn test_int_formal_charge_parity() {
        // Same expectations as `charged_single_atoms_take_their_valence_hydrogens`, but the
        // charge is an Int prop (the real on-graph representation), not f64.
        assert_eq!(charged_atom_h_int("N", -1), 2, "[NH2-] (int fc) -> 2 H");
        assert_eq!(charged_atom_h_int("N", 1), 4, "[NH4+] (int fc) -> 4 H");
        assert_eq!(charged_atom_h_int("O", -1), 1, "[OH-] (int fc) -> 1 H");
        assert_eq!(charged_atom_h_int("C", 1), 3, "[CH3+] (int fc) -> 3 H");
        assert_eq!(charged_atom_h_int("N", 0), 3, "neutral N -> 3 H");
    }

    #[test]
    fn test_sulfonimide_anion_not_protonated() {
        // The exact TFSI/ANI failure: a deprotonated sulfonimide N (two single
        // bonds, int formal_charge -1) must add NO hydrogen. Before the fix the
        // Int charge was missed → N read as neutral (valence 3, demand 2) → 1 H.
        let mut g = Atomistic::new();
        let mut n_atom = Atom::new();
        n_atom.set("element", "N");
        n_atom.set("formal_charge", -1_i32);
        let n = g.add_atom(n_atom);
        let s1 = g.add_atom(atom("S"));
        let s2 = g.add_atom(atom("S"));
        bond_with_order(&mut g, n, s1, 1.0);
        bond_with_order(&mut g, n, s2, 1.0);
        assert_eq!(h_at(&g, n), 0, "sulfonimide [N-] with two bonds -> 0 H");
    }

    /// Helper: implicit-H on `atom_id` of a built graph.
    fn h_at(g: &Atomistic, id: NodeId) -> u32 {
        implicit_h_count(g, id).unwrap_or(0)
    }

    #[test]
    fn bonded_atoms_take_their_remaining_valence_hydrogens() {
        // ethane CC: each C has bos 1 -> 3 H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 1.0);
        assert_eq!(h_at(&g, c1), 3, "ethane C -> 3 H");
        assert_eq!(h_at(&g, c2), 3, "ethane C -> 3 H");

        // ethylene C=C: each C has bos 2 -> 2 H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 2.0);
        assert_eq!(h_at(&g, c1), 2, "ethylene C -> 2 H");

        // benzene (aromatic, bos 1.5+1.5=3): each C -> 1 H
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..6).map(|_| g.add_atom(atom("C"))).collect();
        for i in 0..6 {
            bond_with_order(&mut g, ids[i], ids[(i + 1) % 6], 1.5);
        }
        assert_eq!(h_at(&g, ids[0]), 1, "benzene C -> 1 H");

        // acetate CC(=O)[O-]: methyl C -> 3, carbonyl C -> 0,
        // carbonyl O (=O) -> 0, [O-] (single bond, fc -1) -> 0
        let mut g = Atomistic::new();
        let c_me = g.add_atom(atom("C"));
        let c_carb = g.add_atom(atom("C"));
        let o_dbl = g.add_atom(atom("O"));
        let mut o_minus = Atom::new();
        o_minus.set("element", "O");
        o_minus.set("formal_charge", -1.0_f64);
        let o_minus = g.add_atom(o_minus);
        bond_with_order(&mut g, c_me, c_carb, 1.0);
        bond_with_order(&mut g, c_carb, o_dbl, 2.0);
        bond_with_order(&mut g, c_carb, o_minus, 1.0);
        assert_eq!(h_at(&g, c_me), 3, "acetate methyl C -> 3 H");
        assert_eq!(h_at(&g, c_carb), 0, "acetate carbonyl C -> 0 H");
        assert_eq!(h_at(&g, o_dbl), 0, "acetate =O -> 0 H");
        assert_eq!(h_at(&g, o_minus), 0, "acetate [O-] -> 0 H");
    }

    #[test]
    fn test_no_double_h_on_existing_hydrogen() {
        // Existing H atoms should not get more H added.
        let mut g = Atomistic::new();
        let c = g.add_atom(atom("C"));
        let h = g.add_atom(atom("H"));
        bond_with_order(&mut g, c, h, 1.0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        // C had 1 bond, needs 3 more H; H should remain unchanged
        let n_h = result
            .atoms()
            .filter(|(_, a)| a.get_str("element") == Some("H"))
            .count();
        assert_eq!(n_h, 4); // 1 original + 3 new
    }

    // ── remove_hydrogens tests ──────────────────────────────────────────────

    #[test]
    fn test_remove_hydrogens_methane() {
        // C + 4H → remove → 1 atom (C only), 0 bonds
        let mut g = Atomistic::new();
        g.add_atom(atom("C"));
        let with_h = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(with_h.n_atoms(), 5);
        let stripped =
            remove_hydrogens(&with_h).expect("stripping succeeds on a well-formed graph");
        assert_eq!(stripped.n_atoms(), 1);
        assert_eq!(stripped.n_bonds(), 0);
    }

    #[test]
    fn test_remove_hydrogens_ethane() {
        // 2C + 6H → remove → 2 atoms, 1 bond (C-C preserved)
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 1.0);
        let with_h = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(with_h.n_atoms(), 8);
        let stripped =
            remove_hydrogens(&with_h).expect("stripping succeeds on a well-formed graph");
        assert_eq!(stripped.n_atoms(), 2);
        assert_eq!(stripped.n_bonds(), 1);
    }

    #[test]
    fn test_remove_hydrogens_immutable() {
        // Original graph must remain unchanged after remove_hydrogens
        let mut g = Atomistic::new();
        g.add_atom(atom("C"));
        let with_h = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        let before = with_h.n_atoms();
        remove_hydrogens(&with_h).expect("stripping succeeds on a well-formed graph");
        assert_eq!(with_h.n_atoms(), before);
    }

    #[test]
    fn test_remove_hydrogens_no_h_present() {
        // C=C without any H → unchanged
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 2.0);
        let stripped = remove_hydrogens(&g).expect("stripping succeeds on a well-formed graph");
        assert_eq!(stripped.n_atoms(), 2);
        assert_eq!(stripped.n_bonds(), 1);
    }

    #[test]
    fn test_remove_hydrogens_cascades_angles() {
        // Build C with H and an angle involving H, then remove H → angle gone
        let mut g = Atomistic::new();
        let c = g.add_atom(atom("C"));
        let h1 = g.add_atom(atom("H"));
        let h2 = g.add_atom(atom("H"));
        bond_with_order(&mut g, c, h1, 1.0);
        bond_with_order(&mut g, c, h2, 1.0);
        g.add_angle(h1, c, h2).expect("add angle");
        assert_eq!(g.n_angles(), 1);
        let stripped = remove_hydrogens(&g).expect("stripping succeeds on a well-formed graph");
        assert_eq!(stripped.n_atoms(), 1);
        assert_eq!(stripped.n_bonds(), 0);
        assert_eq!(stripped.n_angles(), 0);
    }

    #[test]
    fn test_add_hydrogens_places_coords_when_heavy_has_xyz() {
        let mut g = Atomistic::new();
        g.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 5);
        let mut n_h = 0;
        for (id, a) in result.atoms() {
            if a.get_str("element") != Some("H") {
                continue;
            }
            n_h += 1;
            let x = a.get_f64("x").expect("H must have x");
            let y = a.get_f64("y").expect("H must have y");
            let z = a.get_f64("z").expect("H must have z");
            let dist = (x * x + y * y + z * z).sqrt();
            assert!(
                (dist - 1.09).abs() < 0.02,
                "C–H distance {dist} for H {id:?}"
            );
        }
        assert_eq!(n_h, 4);
    }

    #[test]
    fn test_add_hydrogens_no_xyz_when_heavy_lacks_coords() {
        let mut g = Atomistic::new();
        g.add_atom(atom("C"));
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        for (_, a) in result.atoms() {
            if a.get_str("element") == Some("H") {
                assert!(a.get_f64("x").is_none());
            }
        }
    }

    #[test]
    fn test_add_hydrogens_preserves_parent_angles() {
        let mut g = Atomistic::new();
        let c1 = g.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let c2 = g.add_atom_xyz("C", 1.5, 0.0, 0.0);
        let c3 = g.add_atom_xyz("C", 3.0, 0.0, 0.0);
        bond_with_order(&mut g, c1, c2, 1.0);
        bond_with_order(&mut g, c2, c3, 1.0);
        g.generate_topology(true, false, false, false).unwrap();
        let n_ang = g.n_angles();
        assert!(n_ang > 0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert!(result.n_angles() >= n_ang);
    }

    #[test]
    fn test_explicit_h_count_is_authoritative() {
        // A declared H count (SMILES bracket atom) is exact — do not top it up
        // to the element's default valence.
        let mut g = Atomistic::new();
        let mut c = Atom::new();
        c.set("element", "C");
        c.set("h_count", 2.0_f64);
        let id = g.add_atom(c);
        assert_eq!(implicit_h_count(&g, id), Some(2));
    }

    /// An added hydrogen's mass is the element's mass, read from the periodic
    /// table rather than written out as a literal, so the two can never drift.
    #[test]
    fn added_hydrogen_carries_the_element_mass() {
        let mut g = Atomistic::new();
        g.add_atom(atom("C"));
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");

        let expected = f64::from(
            Element::by_symbol("H")
                .expect("H is an element")
                .atomic_mass(),
        );
        let mut checked = 0;
        for (_, a) in result.atoms() {
            if a.get_str("element") == Some("H") {
                assert_eq!(a.get_f64("mass"), Some(expected));
                checked += 1;
            }
        }
        assert_eq!(checked, 4, "methane's four hydrogens were all checked");
    }

    // ---- ports ride along untouched ---------------------------------------

    #[test]
    fn add_hydrogens_keeps_the_ports_kind_and_caps_a_ported_c_c_o() {
        // C0-C1-O2 with two handle hydrogens: H3 on C0, H4 on O2. Each handle
        // is a real bonded H that is *additionally* recorded as a 2-ary `ports`
        // relation `[anchor, handle]`. No `h_count` / `formal_charge` is set,
        // so the valence model runs.
        //
        // Hand-derived atom count (every bond is single, and `valence_demand`
        // bills each bond at least 1, so the result does not depend on the
        // bond class):
        //   C0: C1 + H3 = 2 bonds, valence 4 -> 2 new H
        //   C1: C0 + O2 = 2 bonds, valence 4 -> 2 new H
        //   O2: C1 + H4 = 2 bonds, valence 2 -> 0 new H
        //   total = 3 heavy + 2 handles + 4 added = 9 atoms
        let mut g = Atomistic::new();
        let c0 = g.add_atom(atom("C"));
        let c1 = g.add_atom(atom("C"));
        let o2 = g.add_atom(atom("O"));
        let h3 = g.add_atom(atom("H"));
        let h4 = g.add_atom(atom("H"));
        bond_with_order(&mut g, c0, c1, 1.0);
        bond_with_order(&mut g, c1, o2, 1.0);
        bond_with_order(&mut g, c0, h3, 1.0);
        bond_with_order(&mut g, o2, h4, 1.0);

        // The `ports` kind rides on the bare `MolGraph`; this test asserts on
        // that graph and never promotes.
        let ports = g.register_kind("ports", 2);
        g.add_relation(ports, &[c0, h3])
            .expect("(anchor, handle) is a 2-ary relation");
        g.add_relation(ports, &[o2, h4])
            .expect("(anchor, handle) is a 2-ary relation");

        let before: Vec<NodeId> = g.atoms().map(|(id, _)| id).collect();
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");

        assert_eq!(result.n_atoms(), 9, "3 heavy + 2 handles + 4 added");

        let ports_after = result
            .kind_id("ports")
            .expect("the ports kind survives add_hydrogens");
        assert_eq!(
            result.n_relations(ports_after),
            2,
            "no added H bond was written into the ports kind"
        );

        let mut added = 0;
        for (id, a) in result.atoms() {
            if before.contains(&id) {
                continue;
            }
            assert_eq!(
                a.get_str("element"),
                Some("H"),
                "add_hydrogens only appends hydrogens"
            );
            assert_eq!(
                result.neighbor_bonds(id).count(),
                1,
                "each added H carries exactly one bond"
            );
            added += 1;
        }
        assert_eq!(added, 4, "two on C0, two on C1, none on the hydroxyl O");
    }

    /// The rule: degree is counted over the `bonds` kind **only**, and an H
    /// that is the handle of a port is kept whatever that degree says.
    ///
    /// C0-C1-O2 with two port handles (H3 on C0, H4 on O2 — real bonded H
    /// additionally recorded as 2-ary `ports` relations) and two plain
    /// repletion hydrogens (H5 on C0, H6 on C1). H5 shares its anchor with the
    /// handle H3, so the fixture separates the two halves of the rule: a
    /// kind-blind degree keeps H5 out of the removal set only by accident of
    /// the port relation, and a bonds-only degree without the port exemption
    /// strips H3 and H4 along with it. Only "bonds-only degree **plus** port
    /// exemption" leaves exactly the two handles.
    #[test]
    fn remove_hydrogens_keeps_port_handles_and_strips_a_plain_h_on_the_same_anchor() {
        let mut g = Atomistic::new();
        let c0 = g.add_atom(atom("C"));
        let c1 = g.add_atom(atom("C"));
        let o2 = g.add_atom(atom("O"));
        let h3 = g.add_atom(atom("H"));
        let h4 = g.add_atom(atom("H"));
        let h5 = g.add_atom(atom("H"));
        let h6 = g.add_atom(atom("H"));
        bond_with_order(&mut g, c0, c1, 1.0);
        bond_with_order(&mut g, c1, o2, 1.0);
        bond_with_order(&mut g, c0, h3, 1.0);
        bond_with_order(&mut g, o2, h4, 1.0);
        bond_with_order(&mut g, c0, h5, 1.0);
        bond_with_order(&mut g, c1, h6, 1.0);

        let ports = g.register_kind("ports", 2);
        g.add_relation(ports, &[c0, h3])
            .expect("(anchor, handle) is a 2-ary relation");
        g.add_relation(ports, &[o2, h4])
            .expect("(anchor, handle) is a 2-ary relation");

        let result = remove_hydrogens(&g).expect("stripping succeeds on a well-formed graph");

        assert_eq!(g.n_atoms(), 7, "the input graph is never mutated");
        assert_eq!(result.n_atoms(), 5, "3 heavy + 2 handles");
        assert_eq!(
            result.n_bonds(),
            4,
            "C0-C1, C1-O2 and the two handle bonds survive"
        );
        let n_h = result
            .atoms()
            .filter(|(_, a)| a.get_str("element") == Some("H"))
            .count();
        assert_eq!(n_h, 2, "both handles kept, both plain hydrogens removed");

        let ports_after = result
            .kind_id("ports")
            .expect("the ports kind survives remove_hydrogens");
        assert_eq!(
            result.n_relations(ports_after),
            2,
            "no port row was cascade-deleted with a removed handle"
        );
    }

    /// A port handle is a terminal hydrogen the pass must *not* remove. That
    /// exemption is a data condition, not a failure: the call returns `Ok`
    /// and the handle is still there.
    #[test]
    fn remove_hydrogens_returns_ok_when_a_port_handle_is_bonded_to_a_heavy_atom() {
        let mut g = Atomistic::new();
        let c0 = g.add_atom(atom("C"));
        let h1 = g.add_atom(atom("H"));
        bond_with_order(&mut g, c0, h1, 1.0);

        let ports = g.register_kind("ports", 2);
        g.add_relation(ports, &[c0, h1])
            .expect("(anchor, handle) is a 2-ary relation");

        let result = remove_hydrogens(&g).expect("an exempt handle is not an error");
        assert_eq!(result.n_atoms(), 2, "the port handle is kept");
    }

    #[test]
    fn implicit_h_count_on_a_long_alkane_reads_only_incident_bonds() {
        // Guards the O(degree) incident-bond read in `valence_demand`: on a
        // 2000-carbon chain the middle carbon has exactly two C-C bonds out of
        // 1999, so it must see a bond-order sum of 2 and take 2 H; each end
        // carbon sees one bond and takes 3.
        const N: usize = 2000;
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..N).map(|_| g.add_atom(atom("C"))).collect();
        for pair in ids.windows(2) {
            bond_with_order(&mut g, pair[0], pair[1], 1.0);
        }
        assert_eq!(g.n_bonds(), N - 1);

        assert_eq!(implicit_h_count(&g, ids[N / 2]), Some(2));
        assert_eq!(implicit_h_count(&g, ids[0]), Some(3));
        assert_eq!(implicit_h_count(&g, ids[N - 1]), Some(3));
    }
}

// ---------------------------------------------------------------------------
// Molecular-formula tests over real SMILES input.
//
// These are the end-to-end gate on the aromatic valence model: a hand-built
// graph can be given whatever bond orders make the arithmetic work, so only
// notation-driven fixtures can catch a parser that mis-declares aromaticity.
// ---------------------------------------------------------------------------
#[cfg(all(test, feature = "smiles"))]
mod smiles_formula_tests {
    use super::*;
    use crate::io::smiles::SmilesIR;
    use std::collections::BTreeMap;

    /// Element → count of the hydrogen-completed molecule.
    fn formula(smiles: &str) -> BTreeMap<String, usize> {
        let ir = SmilesIR::parse(smiles).unwrap_or_else(|e| panic!("parse {smiles:?}: {e}"));
        let mol = ir
            .to_atomistic()
            .unwrap_or_else(|e| panic!("to_atomistic {smiles:?}: {e}"));
        let with_h = add_hydrogens(&mol).expect("repletion succeeds on a well-formed graph");
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (_, atom) in with_h.atoms() {
            let sym = atom.get_str("element").expect("element").to_owned();
            *counts.entry(sym).or_default() += 1;
        }
        counts
    }

    fn assert_formula(smiles: &str, expected: &[(&str, usize)]) {
        let got = formula(smiles);
        let want: BTreeMap<String, usize> = expected
            .iter()
            .map(|(s, n)| ((*s).to_owned(), *n))
            .collect();
        assert_eq!(got, want, "formula of {smiles:?}");
    }

    #[test]
    fn test_aspirin_formula() {
        // C9H8O4 — 21 atoms.
        assert_formula("CC(=O)Oc1ccccc1C(=O)O", &[("C", 9), ("H", 8), ("O", 4)]);
    }

    #[test]
    fn test_benzene_formula() {
        assert_formula("c1ccccc1", &[("C", 6), ("H", 6)]);
    }

    #[test]
    fn test_toluene_formula() {
        assert_formula("Cc1ccccc1", &[("C", 7), ("H", 8)]);
    }

    #[test]
    fn test_naphthalene_formula() {
        // Fusion carbons carry three aromatic bonds and take no hydrogen.
        assert_formula("c1ccc2ccccc2c1", &[("C", 10), ("H", 8)]);
    }

    #[test]
    fn test_pyridine_formula() {
        // One-electron donor N: three ring σ+π valences, no N–H.
        assert_formula("c1ccncc1", &[("C", 5), ("H", 5), ("N", 1)]);
    }

    #[test]
    fn test_pyrrole_formula() {
        // Lone-pair donor N — the H is declared by the bracket and must survive.
        assert_formula("c1cc[nH]c1", &[("C", 4), ("H", 5), ("N", 1)]);
    }

    #[test]
    fn test_furan_formula() {
        assert_formula("c1ccoc1", &[("C", 4), ("H", 4), ("O", 1)]);
    }

    #[test]
    fn test_thiophene_formula() {
        // S has valences [2,4,6]: the lone-pair donor must not reach for 4.
        assert_formula("c1ccsc1", &[("C", 4), ("H", 4), ("S", 1)]);
    }

    #[test]
    fn test_caffeine_formula() {
        assert_formula(
            "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
            &[("C", 8), ("H", 10), ("N", 4), ("O", 2)],
        );
    }

    #[test]
    fn test_ethanol_formula() {
        assert_formula("CCO", &[("C", 2), ("H", 6), ("O", 1)]);
    }

    #[test]
    fn test_acetate_anion_formula() {
        assert_formula("CC(=O)[O-]", &[("C", 2), ("H", 3), ("O", 2)]);
    }

    #[test]
    fn test_biphenyl_formula() {
        // The explicit single bond between the two rings is not aromatic.
        assert_formula("c1ccccc1-c1ccccc1", &[("C", 12), ("H", 10)]);
    }

    #[test]
    fn test_indole_formula() {
        assert_formula("c1ccc2[nH]ccc2c1", &[("C", 8), ("H", 7), ("N", 1)]);
    }

    /// A `Frame` round trip must not invent hydrogen counts.
    ///
    /// `CCO` is organic subset, so no atom declares an `h_count` and
    /// repletion is free to read the valence model. A column that spells an
    /// unset cell as the default `0` turns that freedom into a declared
    /// "this atom has no hydrogens", and the round-tripped molecule comes
    /// back bare. Ethanol is C2H6O either way.
    #[test]
    fn a_frame_round_trip_keeps_the_repletion_count_of_an_organic_subset_smiles() {
        let ir = SmilesIR::parse("CCO").expect("CCO parses");
        let mol = ir
            .to_atomistic()
            .expect("CCO converts to an atomistic graph");
        let direct = add_hydrogens(&mol).expect("repletion succeeds on a well-formed graph");
        assert_eq!(direct.n_atoms(), 9, "C2H6O is nine atoms");

        let read_back =
            Atomistic::from_frame(&mol.to_frame().expect("a schema-conforming graph converts"))
                .expect("an atomistic frame reads back");
        assert_eq!(
            add_hydrogens(&read_back)
                .expect("repletion succeeds on a well-formed graph")
                .n_atoms(),
            9
        );
    }

    /// The same rule where the column really is partial: `[CH3]` declares
    /// three hydrogens and the plain `C` declares none, so `h_count` is set
    /// on one atom of two. Writing the unset cell as `0` tells repletion the
    /// plain carbon is already saturated, and ethane comes back as C2H3.
    #[test]
    fn a_frame_round_trip_keeps_an_undeclared_h_count_undeclared() {
        let ir = SmilesIR::parse("C[CH3]").expect("C[CH3] parses");
        let mol = ir
            .to_atomistic()
            .expect("C[CH3] converts to an atomistic graph");
        let direct = add_hydrogens(&mol).expect("repletion succeeds on a well-formed graph");
        assert_eq!(direct.n_atoms(), 8, "ethane is 2 C + 6 H");

        let read_back =
            Atomistic::from_frame(&mol.to_frame().expect("a schema-conforming graph converts"))
                .expect("an atomistic frame reads back");
        assert_eq!(
            add_hydrogens(&read_back)
                .expect("repletion succeeds on a well-formed graph")
                .n_atoms(),
            8
        );
    }
}
