//! MMFF atom typing — annotates an [`Atomistic`] with MMFF type labels, partial
//! charges, and the per-instance force constants the kernels read.
//!
//! This is the **typifier** half of MMFF: it takes a molecular graph and returns
//! a *labeled* graph (atoms typed + charged; bonds/angles/dihedrals/impropers
//! labeled). Materializing that graph into a [`Frame`](molrs::store::frame::Frame)
//! for the generic `ForceField::to_potentials` path is the caller's job (via
//! [`Atomistic::to_frame`]); building the neighbour list is the consumer's. Atom
//! types + partial charges are reused from the RDKit-validated MMFF front-end
//! ([`MmffMolProperties`]).
//!
//! # One resolver, for the numbers AND the labels
//!
//! Every number and every type code on this graph comes from
//! [`crate::ff::mmff::params`] — the RDKit-faithful resolver, with the ring rules,
//! the four-level equivalence degradation and the empirical fallbacks. There used
//! to be a second classifier (`typifier/mmff/classify.rs`) that produced the
//! *labels* while the resolver produced the *parameters*, so a single row could
//! carry a force constant from one rule set and a type code from another. It was
//! also wrong — it read raw bond orders, so an aromatic bond came out as bond
//! type 1 (RDKit: 0, because after aromaticity perception the bond is `AROMATIC`,
//! not `SINGLE`), and its `typify_angle(bt_ij, bt_jk)` could not see ring
//! membership at all, so a cyclopropane C-C-C angle could never reach its true
//! type 3. It is deleted.
//!
//! The labels are now only provenance — the per-instance kernels
//! ([`ParamSource::PerInstance`](crate::ff::potential::ParamSource)) read Frame
//! *columns*, not labels — but "only provenance" is not a licence to be wrong.
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

use crate::ff::mmff::params as eparams;
use crate::ff::mmff::topo::Topo;
use crate::ff::mmff::{MmffMolProperties, MmffVariant};

use super::params::MMFFParams;

/// Everything the six annotation steps below share: the molecule's atom order,
/// the MMFF topology (with aromaticity perceived) and numeric atom types the
/// resolver keys off, and the caller's variant.
///
/// Assembled once. `Topo::build` + `set_mmff_aromaticity` is the expensive part
/// of MMFF typing and every step needs the result, so it is not re-derived.
struct MmffContext<'a> {
    /// Molecule atom-iteration order — the index space `props` / `types` use.
    atom_ids: Vec<AtomId>,
    idx_of: HashMap<AtomId, usize>,
    props: MmffMolProperties,
    /// MMFF topology with perceived aromaticity — the resolver's ring / bond-order
    /// source, and the reason the type codes below can see what `classify.rs` could not.
    topo: Topo,
    /// Numeric MMFF atom types, indexed as `atom_ids`.
    types: Vec<u8>,
    /// Typing metadata (atom-type properties), for the linear-centre flag.
    params: &'a MMFFParams,
    variant: MmffVariant,
}

impl MmffContext<'_> {
    /// MMFF numeric type of an atom, by id.
    fn type_of(&self, aid: AtomId) -> u32 {
        self.props.atom_type(self.idx_of[&aid]) as u32
    }

    /// Zero-based index of an atom, by id.
    fn idx(&self, aid: AtomId) -> usize {
        self.idx_of[&aid]
    }
}

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
/// The label grammar is `{type_code}_{atom_types...}` — the MMFF type code first,
/// then the atom types the row resolved through. Both halves come from the same
/// resolver call as the row's numbers.
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
    let ctx = build_context(mol, params, variant)?;
    let mut out = mol.clone();

    annotate_atoms(&mut out, &ctx)?;
    annotate_bonds(&mut out, &ctx)?;

    // Enumerate angles + dihedrals on the graph (impropers are MMFF-specific and
    // are enumerated by `annotate_impropers` below).
    crate::ff::typifier::topology::typify_bonded_topology(&mut out)?;

    annotate_angles(&mut out, &ctx)?;
    annotate_dihedrals(&mut out, &ctx)?;
    annotate_impropers(&mut out, &ctx)?;

    Ok(out)
}

/// The shared front-end: atom types, partial charges, MMFF topology.
fn build_context<'a>(
    mol: &Atomistic,
    params: &'a MMFFParams,
    variant: MmffVariant,
) -> Result<MmffContext<'a>, String> {
    // The RDKit-validated front-end for atom types + MMFF partial charges. Its
    // per-atom index is the molecule's atom iteration order — the same order as
    // `atom_ids`.
    let props = MmffMolProperties::compute(mol, variant).map_err(|e| e.to_string())?;

    let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    let idx_of: HashMap<AtomId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // The MMFF topology drives every per-instance parameter and type-code lookup
    // below. Aromaticity is *perceived* here — which is precisely the fact the
    // deleted classifier never saw, because it was handed raw bond orders instead.
    let base = Topo::build(mol).map_err(|s| format!("MMFF Topo: {s}"))?;
    let topo = crate::ff::mmff::aromaticity::set_mmff_aromaticity(&base);
    let types: Vec<u8> = (0..atom_ids.len()).map(|i| props.atom_type(i)).collect();

    Ok(MmffContext {
        atom_ids,
        idx_of,
        props,
        topo,
        types,
        params,
        variant,
    })
}

// --- 1. Atoms ------------------------------------------------------------

/// Validated MMFF numeric type + MMFF partial charge on every atom.
fn annotate_atoms(out: &mut Atomistic, ctx: &MmffContext) -> Result<(), String> {
    for (i, &aid) in ctx.atom_ids.iter().enumerate() {
        out.set_atom(aid, "type", format!("{}", ctx.props.atom_type(i)))
            .map_err(|e| e.to_string())?;
        out.set_atom(aid, "charge", ctx.props.partial_charge(i))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

// --- 2. Bonds ------------------------------------------------------------

/// MMFF bond type + the per-bond `kb` / `r0` (table → equivalence → empirical).
fn annotate_bonds(out: &mut Atomistic, ctx: &MmffContext) -> Result<(), String> {
    let bond_rows: Vec<(_, AtomId, AtomId)> = out
        .bonds()
        .map(|(bid, bond)| (bid, bond.nodes[0], bond.nodes[1]))
        .collect();

    for (bid, a, b) in bond_rows {
        let (ia, ib) = (ctx.idx(a), ctx.idx(b));
        let (t1, t2) = (ctx.type_of(a), ctx.type_of(b));
        let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        let bt = eparams::bond_type(&ctx.topo, &ctx.types, ia, ib);

        out.set_bond_prop(bid, "type", format!("{bt}_{lo}_{hi}"))
            .map_err(|e| e.to_string())?;

        let (kb, r0) = eparams::bond_params(&ctx.topo, &ctx.types, ia, ib)
            .map(|bp| (bp.kb, bp.r0))
            .unwrap_or((0.0, 0.0));
        out.set_bond_prop(bid, "kb", kb)
            .map_err(|e| e.to_string())?;
        out.set_bond_prop(bid, "r0", r0)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

// --- 3. Angles (+ stretch-bend) ------------------------------------------

/// MMFF angle type + `ka` / `theta0` / the stretch-bend constants and their two
/// reference bond lengths, plus the linear-centre flag.
///
/// Collected first: `set_angle_prop` borrows `out` mutably while `angles()`
/// borrows it immutably.
fn annotate_angles(out: &mut Atomistic, ctx: &MmffContext) -> Result<(), String> {
    let angle_rows: Vec<_> = out
        .angles()
        .map(|(id, a)| (id, a.nodes[0], a.nodes[1], a.nodes[2]))
        .collect();

    for (id, a, b, c) in angle_rows {
        let (ia, ib, ic) = (ctx.idx(a), ctx.idx(b), ctx.idx(c));
        // The ring-aware angle type: an angle inside a 3-/4-membered ring is
        // promoted to 3..8, which is exactly what a `(bt_ij, bt_jk)` signature
        // could never express.
        let at = eparams::angle_type(&ctx.topo, &ctx.types, ia, ib, ic);
        let label = format!(
            "{at}_{}_{}_{}",
            ctx.type_of(a),
            ctx.type_of(b),
            ctx.type_of(c)
        );
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
        let linear = ctx
            .params
            .get_prop(ctx.type_of(b))
            .map(|p| p.linh != 0)
            .unwrap_or(false);
        out.set_angle_prop(id, "linear", PropValue::Int(i32::from(linear)))
            .map_err(|e| e.to_string())?;

        // `theta0` comes back in degrees; the angle / stretch-bend kernels consume
        // radians (molrs internal-radians convention).
        let (ka, theta0) = eparams::angle_params(&ctx.topo, &ctx.types, ia, ib, ic)
            .map(|p| (p.ka, p.theta0.to_radians()))
            .unwrap_or((0.0, 0.0));
        out.set_angle_prop(id, "ka", ka)
            .map_err(|e| e.to_string())?;
        out.set_angle_prop(id, "theta0", theta0)
            .map_err(|e| e.to_string())?;

        // Stretch-bend force constants — `stretch_bend_params` carries the `dfsb`
        // period-row default fallback that a table-keyed path lacks (the benzene
        // `mmff_stbn: unknown` blocker). The two reference bond lengths are the
        // per-bond r0, taken straight from the bond resolver.
        let (kba_ijk, kba_kji) = eparams::stretch_bend_params(&ctx.topo, &ctx.types, ia, ib, ic)
            .map(|(s, _, _, _)| (s.kba_ijk, s.kba_kji))
            .unwrap_or((0.0, 0.0));
        let r0_ij = eparams::bond_params(&ctx.topo, &ctx.types, ia, ib)
            .map(|b| b.r0)
            .unwrap_or(0.0);
        let r0_kj = eparams::bond_params(&ctx.topo, &ctx.types, ic, ib)
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
    Ok(())
}

// --- 4. Dihedrals --------------------------------------------------------

/// MMFF torsion type + the variant's Fourier coefficients `(v1, v2, v3)`.
///
/// 42 of the torsion rows are re-parameterised by MMFF94s, all on delocalised
/// trivalent nitrogen — which is why the variant has to reach this lookup.
fn annotate_dihedrals(out: &mut Atomistic, ctx: &MmffContext) -> Result<(), String> {
    let dih_rows: Vec<_> = out
        .dihedrals()
        .map(|(id, d)| (id, d.nodes[0], d.nodes[1], d.nodes[2], d.nodes[3]))
        .collect();

    for (id, a, b, c, d) in dih_rows {
        let (ia, ib, ic, il) = (ctx.idx(a), ctx.idx(b), ctx.idx(c), ctx.idx(d));
        // `torsion_type` returns `(principal, secondary)`; the principal code is
        // the one that names the row, and the 4-/5-ring promotions live in it.
        let (tt, _secondary) = eparams::torsion_type(&ctx.topo, &ctx.types, ia, ib, ic, il);
        let label = format!(
            "{tt}_{}_{}_{}_{}",
            ctx.type_of(a),
            ctx.type_of(b),
            ctx.type_of(c),
            ctx.type_of(d)
        );
        out.set_dihedral_prop(id, "type", label)
            .map_err(|e| e.to_string())?;

        let (v1, v2, v3) =
            eparams::torsion_params(ctx.variant, &ctx.topo, &ctx.types, ia, ib, ic, il)
                .map(|t| (t.v1, t.v2, t.v3))
                .unwrap_or((0.0, 0.0, 0.0));
        out.set_dihedral_prop(id, "v1", v1)
            .map_err(|e| e.to_string())?;
        out.set_dihedral_prop(id, "v2", v2)
            .map_err(|e| e.to_string())?;
        out.set_dihedral_prop(id, "v3", v3)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

// --- 5. Out-of-plane (Wilson) --------------------------------------------

/// MMFF-specific out-of-plane enumeration.
///
/// Only atoms with *exactly three* neighbours are trigonal centres; each
/// contributes three Wilson permutations that share one `koop`. The centre is
/// placed in the second (`atomj`) position to match the `mmff_oop` kernel, which
/// treats `atomj` as the centre. The `type` label is the canonical OOP key that
/// [`eparams::oop_params`] matched on (peripherals equivalence-degraded and
/// sorted), so the label names the row the `koop` came from; centres for which
/// MMFF defines no out-of-plane term are skipped.
fn annotate_impropers(out: &mut Atomistic, ctx: &MmffContext) -> Result<(), String> {
    let n = ctx.atom_ids.len();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (_, bond) in out.bonds() {
        let (a, b) = (ctx.idx(bond.nodes[0]), ctx.idx(bond.nodes[1]));
        adjacency[a].push(b);
        adjacency[b].push(a);
    }

    for (center, neighbours) in adjacency.iter().enumerate() {
        // Exactly three neighbours — the definition of a trigonal centre — said as
        // a pattern, so the arity check and the destructuring cannot disagree.
        let &[a, b, c] = &neighbours[..] else {
            continue;
        };

        // Per-centre out-of-plane force constant `koop` (md·Å·rad⁻²), shared by all
        // three Wilson permutations — the OOP lookup is symmetric in the peripheral
        // atoms — resolved for the caller's variant; the kernel reads the column and
        // evaluates `E_oop = 0.5 · 143.9325 · koop · χ²` with χ in radians. This is
        // the one number MMFF94s changes on a delocalised trivalent nitrogen.
        let Some((label, koop)) = eparams::oop_params(ctx.variant, &ctx.types, a, center, b, c)
        else {
            continue;
        };

        let center_id = ctx.atom_ids[center];
        for &(i, k, l) in &[(a, b, c), (a, c, b), (b, c, a)] {
            let id = out
                .add_improper(ctx.atom_ids[i], center_id, ctx.atom_ids[k], ctx.atom_ids[l])
                .map_err(|e| e.to_string())?;
            out.set_improper_prop(id, "type", label.clone())
                .map_err(|e| e.to_string())?;
            out.set_improper_prop(id, "koop", koop)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
