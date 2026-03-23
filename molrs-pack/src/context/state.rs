//! Runtime state views over [`PackContext`](super::PackContext).

use super::PackContext;
use molrs::types::F;

/// Read-only runtime state view.
pub struct RuntimeState<'a> {
    pub(crate) ctx: &'a PackContext,
}

impl<'a> RuntimeState<'a> {
    #[inline]
    pub fn fdist(&self) -> F {
        self.ctx.fdist
    }

    #[inline]
    pub fn frest(&self) -> F {
        self.ctx.frest
    }

    #[inline]
    pub fn ncf(&self) -> usize {
        self.ctx.ncf()
    }

    #[inline]
    pub fn ncg(&self) -> usize {
        self.ctx.ncg()
    }
}

/// Mutable runtime state view.
pub struct RuntimeStateMut<'a> {
    pub(crate) ctx: &'a mut PackContext,
}

impl<'a> RuntimeStateMut<'a> {
    #[inline]
    pub fn set_move_flag(&mut self, enabled: bool) {
        self.ctx.move_flag = enabled;
    }

    #[inline]
    pub fn set_init1(&mut self, enabled: bool) {
        self.ctx.init1 = enabled;
    }
}
