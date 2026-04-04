//! Unified access trait for owned and borrowed frame types.
//!
//! [`FrameAccess`] provides a common read-only interface implemented by both
//! [`Frame`] and [`FrameView`], enabling generic code that works with either.

use std::collections::HashMap;

use ndarray::ArrayViewD;

use crate::block::access::BlockAccess;
use crate::frame::Frame;
use crate::frame_view::FrameView;
use crate::region::simbox::SimBox;
use crate::types::{F, I, U};

/// Unified read-only access for [`Frame`] and [`FrameView`].
///
/// Provides direct access to typed column data across block boundaries,
/// as well as metadata and simulation box references.
pub trait FrameAccess {
    /// Gets a float array view from a column inside a block.
    fn get_float(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, F>>;
    /// Gets an int array view from a column inside a block.
    fn get_int(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, I>>;
    /// Gets a bool array view from a column inside a block.
    fn get_bool(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, bool>>;
    /// Gets a uint array view from a column inside a block.
    fn get_uint(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, U>>;
    /// Gets a u8 array view from a column inside a block.
    fn get_u8(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, u8>>;
    /// Gets a string array view from a column inside a block.
    fn get_string(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, String>>;
    /// Returns a reference to the simulation box, if present.
    fn simbox_ref(&self) -> Option<&SimBox>;
    /// Returns a reference to the metadata map.
    fn meta_ref(&self) -> &HashMap<String, String>;
    /// Returns block keys as a `Vec`.
    fn block_keys(&self) -> Vec<&str>;
    /// Returns `true` if the frame contains the specified block key.
    fn contains_block(&self, key: &str) -> bool;
    /// Number of blocks.
    fn block_count(&self) -> usize;
    /// Returns `true` if the frame contains no blocks.
    fn is_empty(&self) -> bool;
    /// Visits a block by key through the [`BlockAccess`] trait, using the visitor pattern
    /// to avoid lifetime/return-type issues. Returns `None` if the block does not exist.
    fn visit_block<R>(&self, key: &str, f: impl FnOnce(&dyn BlockAccess) -> R) -> Option<R>;
}

impl FrameAccess for Frame {
    fn get_float(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, F>> {
        self.get(block_key)
            .and_then(|b| b.get_float(col_key))
            .map(|a| a.view())
    }

    fn get_int(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, I>> {
        self.get(block_key)
            .and_then(|b| b.get_int(col_key))
            .map(|a| a.view())
    }

    fn get_bool(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, bool>> {
        self.get(block_key)
            .and_then(|b| b.get_bool(col_key))
            .map(|a| a.view())
    }

    fn get_uint(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, U>> {
        self.get(block_key)
            .and_then(|b| b.get_uint(col_key))
            .map(|a| a.view())
    }

    fn get_u8(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, u8>> {
        self.get(block_key)
            .and_then(|b| b.get_u8(col_key))
            .map(|a| a.view())
    }

    fn get_string(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, String>> {
        self.get(block_key)
            .and_then(|b| b.get_string(col_key))
            .map(|a| a.view())
    }

    fn simbox_ref(&self) -> Option<&SimBox> {
        self.simbox.as_ref()
    }

    fn meta_ref(&self) -> &HashMap<String, String> {
        &self.meta
    }

    fn block_keys(&self) -> Vec<&str> {
        self.keys().collect()
    }

    fn contains_block(&self, key: &str) -> bool {
        self.contains_key(key)
    }

    fn block_count(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        Frame::is_empty(self)
    }

    fn visit_block<R>(&self, key: &str, f: impl FnOnce(&dyn BlockAccess) -> R) -> Option<R> {
        self.get(key).map(|block| f(block))
    }
}

impl FrameAccess for FrameView<'_> {
    fn get_float(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, F>> {
        self.get(block_key).and_then(|b| b.get_float(col_key))
    }

    fn get_int(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, I>> {
        self.get(block_key).and_then(|b| b.get_int(col_key))
    }

    fn get_bool(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, bool>> {
        self.get(block_key).and_then(|b| b.get_bool(col_key))
    }

    fn get_uint(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, U>> {
        self.get(block_key).and_then(|b| b.get_uint(col_key))
    }

    fn get_u8(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, u8>> {
        self.get(block_key).and_then(|b| b.get_u8(col_key))
    }

    fn get_string(&self, block_key: &str, col_key: &str) -> Option<ArrayViewD<'_, String>> {
        self.get(block_key).and_then(|b| b.get_string(col_key))
    }

    fn simbox_ref(&self) -> Option<&SimBox> {
        self.simbox
    }

    fn meta_ref(&self) -> &HashMap<String, String> {
        self.meta
    }

    fn block_keys(&self) -> Vec<&str> {
        self.keys().copied().collect()
    }

    fn contains_block(&self, key: &str) -> bool {
        self.contains_key(key)
    }

    fn block_count(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        FrameView::is_empty(self)
    }

    fn visit_block<R>(&self, key: &str, f: impl FnOnce(&dyn BlockAccess) -> R) -> Option<R> {
        self.get(key).map(|block_view| f(block_view))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::types::{F, I};
    use ndarray::Array1;

    fn make_frame() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn())
            .unwrap();
        atoms
            .insert("id", Array1::from_vec(vec![10 as I, 20, 30]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);
        frame.meta.insert("title".into(), "Test".into());
        frame
    }

    #[test]
    fn test_frame_access_on_frame() {
        let frame = make_frame();
        assert!(FrameAccess::get_float(&frame, "atoms", "x").is_some());
        assert!(FrameAccess::get_int(&frame, "atoms", "id").is_some());
        assert!(FrameAccess::get_float(&frame, "atoms", "missing").is_none());
        assert!(FrameAccess::get_float(&frame, "missing", "x").is_none());
        assert_eq!(FrameAccess::block_count(&frame), 1);
        assert!(FrameAccess::contains_block(&frame, "atoms"));
        assert!(!FrameAccess::is_empty(&frame));
        assert_eq!(FrameAccess::meta_ref(&frame).get("title").unwrap(), "Test");
        assert!(FrameAccess::simbox_ref(&frame).is_none());
    }

    #[test]
    fn test_frame_access_on_frame_view() {
        let frame = make_frame();
        let view = FrameView::from(&frame);
        assert!(FrameAccess::get_float(&view, "atoms", "x").is_some());
        assert!(FrameAccess::get_int(&view, "atoms", "id").is_some());
        assert!(FrameAccess::get_float(&view, "atoms", "missing").is_none());
        assert_eq!(FrameAccess::block_count(&view), 1);
        assert!(FrameAccess::contains_block(&view, "atoms"));
        assert!(!FrameAccess::is_empty(&view));
        assert_eq!(FrameAccess::meta_ref(&view).get("title").unwrap(), "Test");
    }

    #[test]
    fn test_generic_function_with_frame_access() {
        fn get_x_data(f: &impl FrameAccess) -> Option<Vec<F>> {
            f.get_float("atoms", "x")
                .map(|a| a.iter().copied().collect())
        }

        let frame = make_frame();
        assert_eq!(get_x_data(&frame), Some(vec![1.0, 2.0, 3.0]));

        let view = FrameView::from(&frame);
        assert_eq!(get_x_data(&view), Some(vec![1.0, 2.0, 3.0]));
    }
}
