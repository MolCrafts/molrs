//! Emit option flags for graph → SMILES / local SMARTS.
//!
//! Every science/representation choice is an explicit field — no silent policy.

use molrs::system::atomistic::AtomId;

/// Options for [`super::from_atomistic`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmilesEmitOptions {
    /// Use WL [`canonical_order`](molrs::system::atomistic::Atomistic::canonical_order)
    /// for root selection and branch ordering.
    pub canonical: bool,
    /// Override root atom; when set, root selection ignores `canonical` root pick
    /// (branch order may still use canonical colors).
    pub root: Option<AtomId>,
    /// Aromatic emission style.
    pub aromatic: AromaticEmit,
    /// How hydrogens appear in the string.
    pub hydrogens: HydrogenEmit,
    /// Emit tetrahedral / double-bond stereo markers when present on the graph.
    pub include_stereo: bool,
    /// Multi-component graph policy.
    pub multi_component: MultiComponentEmit,
    /// Prefer Daylight organic-subset letters when legal.
    pub organic_subset: bool,
}

/// Aromatic write policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AromaticEmit {
    /// Honour `is_aromatic` / aromatic bond markers on the graph.
    AsMarked,
    /// Ignore aromatic markers; require integer bond numbers.
    KekuleOnly,
}

/// Hydrogen write policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HydrogenEmit {
    /// Daylight organic-subset omission of implicit H; skip explicit H atoms.
    OrganicSubset,
    /// Write every atom including H as bracket atoms.
    ExplicitAll,
    /// Use stored `h_count` / explicit H neighbours only.
    AsStored,
}

/// Multi-component policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiComponentEmit {
    /// Error if more than one connected component (safe default for systems).
    ErrorIfMultiple,
    /// Join components with `'.'`.
    JoinDot,
    /// Emit only the component containing `root`, or the first canonical component.
    FirstOnly,
}

impl Default for SmilesEmitOptions {
    fn default() -> Self {
        Self {
            canonical: true,
            root: None,
            aromatic: AromaticEmit::AsMarked,
            hydrogens: HydrogenEmit::OrganicSubset,
            include_stereo: false,
            multi_component: MultiComponentEmit::ErrorIfMultiple,
            organic_subset: true,
        }
    }
}

/// Neighbour encoding style for local SMARTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborStyle {
    /// Spanning-tree branches from the centre (chain form).
    Chain,
    /// Centre atom with recursive `$(...)` environments.
    Recursive,
}

/// Options for [`super::local_smarts`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSmartsOptions {
    /// Bond depth from the centre atom. Must be >= 1.
    pub reach: u32,
    /// Encode centre as `[#Z]` rather than element symbol.
    pub atomic_number: bool,
    pub include_degree: bool,
    pub include_h_count: bool,
    pub include_charge: bool,
    pub include_aromatic: bool,
    /// `[R]` / `[Rn]` — makes molpy TypeScope unbounded when true.
    pub include_ring_membership: bool,
    pub include_ring_size: bool,
    pub include_explicit_h_atoms: bool,
    pub include_bond_orders: bool,
    pub neighbor_style: NeighborStyle,
    pub canonical_neighbor_order: bool,
}

impl Default for LocalSmartsOptions {
    fn default() -> Self {
        Self {
            reach: 1,
            atomic_number: true,
            include_degree: true,
            include_h_count: true,
            include_charge: true,
            include_aromatic: true,
            include_ring_membership: false,
            include_ring_size: false,
            include_explicit_h_atoms: false,
            include_bond_orders: true,
            neighbor_style: NeighborStyle::Chain,
            canonical_neighbor_order: true,
        }
    }
}
