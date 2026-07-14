//! MMFF atom typing — annotates an [`Atomistic`] with MMFF type labels, partial
//! charges, and the per-instance force constants the kernels read.
//!
//! This is the **typifier** half of MMFF: it takes a molecular graph and returns
//! a *labeled* graph (atoms typed + charged; bonds/angles/dihedrals/impropers
//! labeled). Materializing that graph into a [`Frame`](molrs::store::frame::Frame)
//! for the generic `ForceField::to_potentials` path is the caller's job (via
//! [`Atomistic::to_frame`]); building the neighbour list is the consumer's. Atom
//! types + partial charges are reused from the RDKit-validated MMFF front-end
//! ([`MmffMolProperties`]); the bond/angle/dihedral type-label rules are
//! MMFF-specific and live here.
//!
//! # The variant is a parameter, never a constant
//!
//! Every parameter resolved here is resolved **for the caller's [`MmffVariant`]**.
//! This is the path that bakes `koop` (impropers) and `(v1, v2, v3)` (dihedrals)
//! into the Frame columns that `mmff_oop` / `mmff_torsion` consume, so a hardcoded
//! variant here silently produces MMFF94 numbers no matter which typifier the user
//! constructed — the exact bug `MMFF94STypifier` exists to make impossible.

use std::collections::HashMap;

use molrs::system::molgraph::PropValue;
use molrs::{AtomId, Atomistic};

use crate::ff::mmff::energy::params as eparams;
use crate::ff::mmff::{MmffMolProperties, MmffVariant};

use super::classify::{
    resolve_angle_label, resolve_oop_label, typify_angle, typify_bond, typify_dihedral,
};
use super::params::MMFFParams;

/// Annotate `mol` with MMFF type labels + partial charges for `variant`,
/// returning the labeled [`Atomistic`]:
/// - atoms: `type` (MMFF numeric type as string) + `charge` (MMFF partial charge)
/// - bonds: `type` (e.g. `"0_1_5"`) + `kb` / `r0`
/// - angles: `type` / `stbn_type` (e.g. `"0_1_2_1"`) — enumerated — + `ka` /
///   `theta0` (radians) / `kba_ijk` / `kba_kji` / `r0_ij` / `r0_kj` / `linear`
///   (0/1: the central atom is a linear centre, `linh != 0`)
/// - dihedrals: `type` (e.g. `"0_5_1_1_5"`) — enumerated — + the Fourier
///   coefficients `v1` / `v2` / `v3` (kcal·mol⁻¹), **variant-dependent**
/// - impropers: `type` = canonical MMFF out-of-plane key (e.g. `"0_37_37_37"`) +
///   `koop` (md·Å·rad⁻², **variant-dependent**); three Wilson rows per trigonal
///   centre, centre in the `atomj` position, sharing one `koop`
///
/// `variant` is supplied by the typifier front door
/// ([`MMFF94Typifier`](super::MMFF94Typifier) /
/// [`MMFF94STypifier`](super::MMFF94STypifier)) and is threaded to **every**
/// parameter lookup below. Atom types and charges are variant-independent by
/// construction (MMFF94 and MMFF94s share all 95 types); `koop` and `(v1, v2, v3)`
/// are not.
///
/// The caller converts the result with [`Atomistic::to_frame`], builds the
/// neighbour list, and calls `to_potentials`.
pub(crate) fn annotate_mmff(
    mol: &Atomistic,
    params: &MMFFParams,
    variant: MmffVariant,
) -> Result<Atomistic, String> {
    // Reuse the RDKit-validated front-end for atom types + MMFF partial charges.
    // Its per-atom index is the molecule's atom iteration order — the same order
    // as `atom_ids` below.
    let props = MmffMolProperties::compute(mol, variant).map_err(|e| e.to_string())?;

    let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    let idx_of: HashMap<AtomId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let type_of = |aid: AtomId| -> u32 { props.atom_type(idx_of[&aid]) as u32 };

    // Build the MMFF topology + numeric type vector once, to drive the
    // RDKit-validated per-instance parameter resolution (reused from the energy
    // path) when baking numeric parameters onto each interaction below.
    let topo = {
        let base =
            crate::ff::mmff::topo::Topo::build(mol).map_err(|s| format!("MMFF Topo: {s}"))?;
        crate::ff::mmff::aromaticity::set_mmff_aromaticity(&base)
    };
    let types_u8: Vec<u8> = (0..atom_ids.len()).map(|i| props.atom_type(i)).collect();

    let mut out = mol.clone();

    // 1. Atoms: validated MMFF numeric type + MMFF partial charge.
    for (i, &aid) in atom_ids.iter().enumerate() {
        out.set_atom(aid, "type", format!("{}", props.atom_type(i)))
            .map_err(|e| e.to_string())?;
        out.set_atom(aid, "charge", props.partial_charge(i))
            .map_err(|e| e.to_string())?;
    }

    // 2. Bonds: typify the MMFF bond type; cache it for angle/dihedral typing.
    let bond_rows: Vec<(_, AtomId, AtomId, f64)> = mol
        .bonds()
        .map(|(bid, bond)| {
            let order = match bond.props.get("order") {
                Some(PropValue::F64(v)) => *v,
                _ => 1.0,
            };
            (bid, bond.nodes[0], bond.nodes[1], order)
        })
        .collect();
    let mut bt_map: HashMap<(usize, usize), u32> = HashMap::new();
    for (bid, a, b, order) in &bond_rows {
        let (ia, ib) = (idx_of[a], idx_of[b]);
        let (t1, t2) = (type_of(*a), type_of(*b));
        let bt = typify_bond(t1, t2, *order, params);
        let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        out.set_bond_prop(*bid, "type", format!("{}_{}_{}", bt, lo, hi))
            .map_err(|e| e.to_string())?;
        // Bake the per-bond numeric parameters (table → equivalence → empirical),
        // resolved exactly as the RDKit-validated energy path does.
        let (kb, r0) = crate::ff::mmff::energy::params::bond_params(&topo, &types_u8, ia, ib)
            .map(|bp| (bp.kb, bp.r0))
            .unwrap_or((0.0, 0.0));
        out.set_bond_prop(*bid, "kb", kb)
            .map_err(|e| e.to_string())?;
        out.set_bond_prop(*bid, "r0", r0)
            .map_err(|e| e.to_string())?;
        bt_map.insert((ia.min(ib), ia.max(ib)), bt);
    }
    let get_bt = |ia: usize, ib: usize| -> u32 {
        bt_map.get(&(ia.min(ib), ia.max(ib))).copied().unwrap_or(0)
    };

    // 3. Enumerate angles + dihedrals on the graph (impropers handled below).
    crate::ff::typifier::topology::typify_bonded_topology(&mut out)?;

    // 4. Angles: typify + label. Collect first — `set_angle_prop` borrows `out`
    //    mutably while `angles()` borrows it immutably.
    let angle_rows: Vec<_> = out
        .angles()
        .map(|(id, a)| (id, a.nodes[0], a.nodes[1], a.nodes[2]))
        .collect();
    for (id, a, b, c) in angle_rows {
        let (ia, ib, ic) = (idx_of[&a], idx_of[&b], idx_of[&c]);
        let at = typify_angle(get_bt(ia, ib), get_bt(ib, ic));
        let label = resolve_angle_label(at, type_of(a), type_of(b), type_of(c));
        out.set_angle_prop(id, "type", label.clone())
            .map_err(|e| e.to_string())?;
        out.set_angle_prop(id, "stbn_type", label)
            .map_err(|e| e.to_string())?;
        // Linear-centre flag, from the CENTRAL atom's `linh` property (nitrile,
        // alkyne, allene, isocyanate…). It selects a different functional form for
        // the bend — `E = 143.9325·ka·(1 + cos θ)` instead of the cubic expansion
        // about theta0 — and suppresses the stretch-bend term at that centre; both
        // kernels (`mmff_angle`, `mmff_stbn`) read this one column. Baked as 0/1
        // rather than a bool because `MolGraph::to_frame` carries only f64 / i32 /
        // string columns into the Frame; a bool would be silently dropped.
        let linear = params
            .get_prop(type_of(b))
            .map(|p| p.linh != 0)
            .unwrap_or(false);
        out.set_angle_prop(id, "linear", PropValue::Int(i32::from(linear)))
            .map_err(|e| e.to_string())?;
        // Bake per-instance numeric params (table → equivalence → empirical),
        // resolved exactly as the RDKit-validated energy path does. `theta0`
        // comes back in degrees; the angle/stretch-bend kernels consume radians.
        let (ka, theta0) = eparams::angle_params(&topo, &types_u8, ia, ib, ic)
            .map(|p| (p.ka, p.theta0.to_radians()))
            .unwrap_or((0.0, 0.0));
        out.set_angle_prop(id, "ka", ka)
            .map_err(|e| e.to_string())?;
        out.set_angle_prop(id, "theta0", theta0)
            .map_err(|e| e.to_string())?;
        // Stretch-bend force constants — `stretch_bend_params` carries the
        // `dfsb` period-row default fallback that the shared-table path lacked
        // (the benzene `mmff_stbn: unknown` blocker). The two reference bond
        // lengths are per-bond r0, taken straight from the bond resolver.
        let (kba_ijk, kba_kji) = eparams::stretch_bend_params(&topo, &types_u8, ia, ib, ic)
            .map(|(s, _, _, _)| (s.kba_ijk, s.kba_kji))
            .unwrap_or((0.0, 0.0));
        let r0_ij = eparams::bond_params(&topo, &types_u8, ia, ib)
            .map(|b| b.r0)
            .unwrap_or(0.0);
        let r0_kj = eparams::bond_params(&topo, &types_u8, ic, ib)
            .map(|b| b.r0)
            .unwrap_or(0.0);
        out.set_angle_prop(id, "kba_ijk", kba_ijk)
            .map_err(|e| e.to_string())?;
        out.set_angle_prop(id, "kba_kji", kba_kji)
            .map_err(|e| e.to_string())?;
        out.set_angle_prop(id, "r0_ij", r0_ij)
            .map_err(|e| e.to_string())?;
        out.set_angle_prop(id, "r0_kj", r0_kj)
            .map_err(|e| e.to_string())?;
    }

    // 5. Dihedrals: typify + label.
    let dih_rows: Vec<_> = out
        .dihedrals()
        .map(|(id, d)| (id, d.nodes[0], d.nodes[1], d.nodes[2], d.nodes[3]))
        .collect();
    for (id, a, b, c, d) in dih_rows {
        let (ia, ib, ic, il) = (idx_of[&a], idx_of[&b], idx_of[&c], idx_of[&d]);
        let tt = typify_dihedral(get_bt(ia, ib), get_bt(ib, ic), get_bt(ic, il));
        let label = format!(
            "{}_{}_{}_{}_{}",
            tt,
            type_of(a),
            type_of(b),
            type_of(c),
            type_of(d)
        );
        out.set_dihedral_prop(id, "type", label)
            .map_err(|e| e.to_string())?;
        // Bake per-instance Fourier coefficients (table → empirical), resolved
        // via the RDKit-validated energy path for the caller's variant; the kernel
        // reads the columns. 42 of these rows are re-parameterised by MMFF94s.
        let (v1, v2, v3) = eparams::torsion_params(variant, &topo, &types_u8, ia, ib, ic, il)
            .map(|t| (t.v1, t.v2, t.v3))
            .unwrap_or((0.0, 0.0, 0.0));
        out.set_dihedral_prop(id, "v1", v1)
            .map_err(|e| e.to_string())?;
        out.set_dihedral_prop(id, "v2", v2)
            .map_err(|e| e.to_string())?;
        out.set_dihedral_prop(id, "v3", v3)
            .map_err(|e| e.to_string())?;
    }

    // 6. Out-of-plane (Wilson) terms — MMFF-specific enumeration. Only atoms with
    //    *exactly three* neighbours are trigonal centres; each contributes three
    //    Wilson permutations that share one `koop`. The centre is placed in the
    //    second (`atomj`) position to match the `mmff_oop` kernel, which treats
    //    `atomj` as the centre (mirroring the RDKit-validated energy path). The
    //    `type` label is the canonical OOP key (peripherals equivalence-degraded +
    //    sorted), so the kernel resolves `koop` by exact match; centres for which
    //    MMFF defines no out-of-plane term are skipped.
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); atom_ids.len()];
    for (_, a, b, _) in &bond_rows {
        adjacency[idx_of[a]].push(idx_of[b]);
        adjacency[idx_of[b]].push(idx_of[a]);
    }
    for center in 0..atom_ids.len() {
        if adjacency[center].len() != 3 {
            continue;
        }
        let (a, b, c) = (
            adjacency[center][0],
            adjacency[center][1],
            adjacency[center][2],
        );
        let center_id = atom_ids[center];
        let Some(label) = resolve_oop_label(
            type_of(center_id),
            [
                type_of(atom_ids[a]),
                type_of(atom_ids[b]),
                type_of(atom_ids[c]),
            ],
        ) else {
            continue;
        };
        // Per-centre out-of-plane force constant `koop` (md·Å·rad⁻²), shared by all
        // three Wilson permutations — the OOP lookup is symmetric in the peripheral
        // atoms — resolved via the RDKit-validated energy path for the caller's
        // variant; the kernel reads the column and evaluates
        // `E_oop = 0.5 · 143.9325 · koop · χ²` with χ in radians. This is the one
        // number MMFF94s changes on a delocalised trivalent nitrogen.
        let koop = eparams::oop_koop(variant, &types_u8, a, center, b, c).unwrap_or(0.0);
        // Three Wilson permutations (i, k, l) with the centre fixed in atomj.
        for &(i, k, l) in &[(a, b, c), (a, c, b), (b, c, a)] {
            let id = out
                .add_improper(atom_ids[i], center_id, atom_ids[k], atom_ids[l])
                .map_err(|e| e.to_string())?;
            out.set_improper_prop(id, "type", label.clone())
                .map_err(|e| e.to_string())?;
            out.set_improper_prop(id, "koop", koop)
                .map_err(|e| e.to_string())?;
        }
    }

    Ok(out)
}
