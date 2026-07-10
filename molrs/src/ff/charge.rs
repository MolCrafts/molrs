//! Charge assignment facade over struct-based molecular typifiers.
//!
//! Charge models that need topology labels should still live as typifiers
//! (`Typifier<Mol = Atomistic>`). `ChargeAssigner` is the small orchestration
//! layer that runs such a model and reports the charges it wrote.

use molrs::Atomistic;
use molrs::store::frame::Frame;
use molrs::store::keys;

use crate::ff::typifier::Typifier;

/// Result summary for a charge-assignment pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ChargeResult {
    pub method: String,
    pub charges: Vec<f64>,
    pub total_charge: f64,
}

/// Struct-first charge assignment facade.
pub struct ChargeAssigner<T> {
    model: T,
    method: String,
}

impl<T> ChargeAssigner<T> {
    pub fn new(model: T) -> Self {
        Self {
            model,
            method: "typifier".to_owned(),
        }
    }

    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.method = method.into();
        self
    }

    pub fn model(&self) -> &T {
        &self.model
    }
}

impl<T> ChargeAssigner<T>
where
    T: Typifier<Mol = Atomistic>,
{
    pub fn assign_atomistic(&self, mol: &mut Atomistic) -> Result<ChargeResult, String> {
        let typed = self.model.typify(mol)?;
        let result = collect_charge_result(&typed, &self.method)?;
        *mol = typed;
        Ok(result)
    }

    /// Assign charges to a frame by round-tripping through `Atomistic`.
    ///
    /// This preserves the molrs graph/Frame boundary used by existing
    /// typifiers. Non-graph frame blocks are out of scope for this first facade.
    pub fn assign_frame(&self, frame: &mut Frame) -> Result<ChargeResult, String> {
        let mut mol = Atomistic::from_frame(frame).map_err(|e| e.to_string())?;
        let result = self.assign_atomistic(&mut mol)?;
        *frame = mol.to_frame();
        Ok(result)
    }
}

fn collect_charge_result(mol: &Atomistic, method: &str) -> Result<ChargeResult, String> {
    let mut charges = Vec::with_capacity(mol.n_atoms());
    for (_, atom) in mol.atoms() {
        let Some(q) = atom.get_f64(keys::CHARGE) else {
            return Err("charge assignment produced an atom without `charge`".to_owned());
        };
        charges.push(q);
    }
    let total_charge = charges.iter().sum();
    Ok(ChargeResult {
        method: method.to_owned(),
        charges,
        total_charge,
    })
}
