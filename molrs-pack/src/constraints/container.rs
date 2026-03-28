//! Unified constraints entrypoint used by optimization layers.

use crate::context::PackContext;
use crate::objective::{compute_f, compute_fg, compute_g};
use molrs::types::F;

/// Evaluation mode for constraints/objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    /// Function value only.
    FOnly,
    /// Gradient only.
    GradientOnly,
    /// Function + gradient.
    FAndGradient,
    /// Restmol mode (same compute path as F+G, semantically explicit for callers).
    RestMol,
}

/// Unified evaluation output.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalOutput {
    pub f_total: F,
    pub fdist_max: F,
    pub frest_max: F,
}

/// Container facade for all constraints-related objective evaluation.
///
/// This struct is intentionally zero-sized: all state is carried by `PackContext`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Constraints;

impl Constraints {
    #[allow(clippy::needless_pass_by_value)]
    pub fn evaluate(
        self,
        x: &[F],
        ctx: &mut PackContext,
        mode: EvalMode,
        gradient: Option<&mut [F]>,
    ) -> EvalOutput {
        let mut f_total = 0.0;

        match mode {
            EvalMode::FOnly => {
                f_total = compute_f(x, ctx);
            }
            EvalMode::GradientOnly => {
                if let Some(g) = gradient {
                    compute_g(x, ctx, g);
                } else {
                    debug_assert!(false, "GradientOnly mode requires gradient buffer");
                }
            }
            EvalMode::FAndGradient | EvalMode::RestMol => {
                if let Some(g) = gradient {
                    f_total = compute_fg(x, ctx, g);
                } else {
                    debug_assert!(false, "FAndGradient/RestMol mode requires gradient buffer");
                }
            }
        }

        EvalOutput {
            f_total,
            fdist_max: ctx.fdist,
            frest_max: ctx.frest,
        }
    }

    #[inline]
    pub fn violation(self, x: &[F], ctx: &mut PackContext) -> EvalOutput {
        self.evaluate(x, ctx, EvalMode::FOnly, None)
    }
}
