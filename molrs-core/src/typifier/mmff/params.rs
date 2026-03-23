//! MMFF94 typing data structures: atom properties and equivalence tables.

use std::collections::HashMap;

/// Key type for the MMFF property index: (atno, crd, val, pilp, mltb, arom, linh, sbmb).
pub(crate) type PropKey = (u32, u32, u32, u32, u32, u32, u32, u32);

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

/// MMFF94 equivalence table row (from `<EquivalenceTable>` in the XML).
#[derive(Debug, Clone)]
pub struct MMFFEquiv {
    pub type_id: u32,
    pub eq1: u32,
    pub eq2: u32,
    pub eq3: u32,
    pub eq4: u32,
}

/// Parsed MMFF typing metadata (atom properties + equivalence table).
///
/// Separate from [`ForceField`](crate::forcefield::ForceField) because these
/// are typing metadata, not potential parameters. Loaded from the same XML but
/// used only during typing, not during energy evaluation.
#[derive(Debug, Clone)]
pub struct MMFFParams {
    /// Atom properties indexed by type_id.
    pub(crate) props: HashMap<u32, MMFFAtomProp>,
    /// Properties indexed by (atno, crd, val, pilp, mltb, arom, linh, sbmb).
    pub(crate) prop_index: HashMap<PropKey, Vec<u32>>,
    /// Equivalence table indexed by type_id.
    pub(crate) equiv: HashMap<u32, MMFFEquiv>,
}

impl MMFFParams {
    /// Create a new `MMFFParams` from pre-parsed data.
    pub fn new(
        props: HashMap<u32, MMFFAtomProp>,
        prop_index: HashMap<PropKey, Vec<u32>>,
        equiv: HashMap<u32, MMFFEquiv>,
    ) -> Self {
        Self {
            props,
            prop_index,
            equiv,
        }
    }

    /// Look up atom property by type_id.
    pub fn get_prop(&self, type_id: u32) -> Option<&MMFFAtomProp> {
        self.props.get(&type_id)
    }

    /// Look up equivalence by type_id.
    pub fn get_equiv(&self, type_id: u32) -> Option<&MMFFEquiv> {
        self.equiv.get(&type_id)
    }

    /// Find candidate type_ids matching a property key.
    pub fn find_types(&self, key: PropKey) -> &[u32] {
        self.prop_index
            .get(&key)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}
