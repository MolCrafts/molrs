//! MMFF94 typing data: per-type atom properties.

use std::collections::HashMap;

/// MMFF94 atom property row (from `<AtomProperties>` in the XML).
#[derive(Debug, Clone)]
pub struct MMFFAtomProp {
    pub type_id: u32,
    pub atno: u32,
    pub crd: u32,
    pub val: u32,
    pub pilp: u32,
    pub mltb: u32,
    pub arom: u32,
    pub linh: u32,
    pub sbmb: u32,
}

/// Parsed MMFF typing metadata: atom properties indexed by type id.
///
/// Separate from [`ForceField`](crate::ff::forcefield::ForceField) because these
/// are typing metadata, not potential parameters. Loaded from the same XML but
/// used only during topology classification (e.g. the `sbmb` flag drives MMFF
/// bond-type assignment), not during energy evaluation.
#[derive(Debug, Clone)]
pub struct MMFFParams {
    /// Atom properties indexed by type_id.
    pub(crate) props: HashMap<u32, MMFFAtomProp>,
}

impl MMFFParams {
    /// Create a new `MMFFParams` from pre-parsed atom properties.
    pub fn new(props: HashMap<u32, MMFFAtomProp>) -> Self {
        Self { props }
    }

    /// Look up atom property by type_id.
    pub fn get_prop(&self, type_id: u32) -> Option<&MMFFAtomProp> {
        self.props.get(&type_id)
    }
}
