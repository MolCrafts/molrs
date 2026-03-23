//! Read-only model/topology view over [`PackContext`](super::PackContext).

use super::PackContext;

/// Read-only model view.
pub struct ModelData<'a> {
    pub(crate) ctx: &'a PackContext,
}

impl<'a> ModelData<'a> {
    #[inline]
    pub fn ntype(&self) -> usize {
        self.ctx.ntype
    }

    #[inline]
    pub fn ntotmol(&self) -> usize {
        self.ctx.ntotmol
    }

    #[inline]
    pub fn ntotat(&self) -> usize {
        self.ctx.ntotat
    }

    #[inline]
    pub fn nmols(&self) -> &[usize] {
        &self.ctx.nmols
    }

    #[inline]
    pub fn natoms(&self) -> &[usize] {
        &self.ctx.natoms
    }
}
