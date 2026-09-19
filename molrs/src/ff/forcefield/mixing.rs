//! How a force field says per-atom-type Lennard-Jones parameters combine.
//!
//! A combining rule is a **force-field** declaration, not a kernel constant —
//! it is what `<ForceField combining_rule>` and LAMMPS's `pair_modify mix` set,
//! and reading an OPLS pack under Lorentz-Berthelot silently shifts every σ.
//! It therefore lives with the force field that declares it, and the kernels
//! read it from there; the reverse — a reader importing a rule out of a pair
//! kernel — put a cycle between `ff::forcefield` and `ff::potential`.

use molrs::types::F;

/// How per-atom-type ε/σ combine into a pair's ε/σ.
///
/// The style-level **string** param `mixing` selects it. LAMMPS spells the same
/// choice `pair_modify mix <rule>`; OpenMM / foyer XML spells it
/// `combining_rule` on `<ForceField>`. It is a force-field property, not a
/// kernel constant: OPLS-AA is `geometric` on both ε and σ, AMBER / GAFF and
/// CHARMM are `arithmetic` (Lorentz-Berthelot), COMPASS / class2 is
/// `sixthpower`. Reading an OPLS pack with Lorentz-Berthelot silently shifts
/// every σ — hence the explicit knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mixing {
    /// `ε = √(εᵢεⱼ)`, `σ = ½(σᵢ + σⱼ)` — Lorentz-Berthelot (AMBER, CHARMM).
    Arithmetic,
    /// `ε = √(εᵢεⱼ)`, `σ = √(σᵢσⱼ)` — OPLS-AA.
    Geometric,
    /// `σ⁶ = ½(σᵢ⁶ + σⱼ⁶)`, `ε = 2√(εᵢεⱼ)·σᵢ³σⱼ³/(σᵢ⁶ + σⱼ⁶)` — COMPASS / class2.
    SixthPower,
}

impl Mixing {
    /// Parse the canonical spelling, accepting LAMMPS' and foyer's synonyms.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "arithmetic" | "lorentz" | "lorentz-berthelot" => Ok(Self::Arithmetic),
            "geometric" => Ok(Self::Geometric),
            "sixthpower" => Ok(Self::SixthPower),
            other => Err(format!(
                "unknown mixing rule '{other}'                  (expected arithmetic | geometric | sixthpower)"
            )),
        }
    }

    /// Combine one pair's per-type `(ε, σ)` into the pair's `(ε, σ)`.
    pub fn combine(self, (eps_i, sig_i): (F, F), (eps_j, sig_j): (F, F)) -> (F, F) {
        match self {
            Self::Arithmetic => ((eps_i * eps_j).sqrt(), 0.5 * (sig_i + sig_j)),
            Self::Geometric => ((eps_i * eps_j).sqrt(), (sig_i * sig_j).sqrt()),
            Self::SixthPower => {
                let (si3, sj3) = (sig_i.powi(3), sig_j.powi(3));
                let (si6, sj6) = (si3 * si3, sj3 * sj3);
                let denom = si6 + sj6;
                if denom <= 0.0 {
                    return (0.0, 0.0);
                }
                let eps = 2.0 * (eps_i * eps_j).sqrt() * si3 * sj3 / denom;
                (eps, (0.5 * denom).powf(1.0 / 6.0))
            }
        }
    }
}
