//! Zero-copy borrowed view of a [`Frame`].
//!
//! `FrameView<'a>` borrows blocks from a [`Frame`] as [`BlockView`]s without
//! copying any array data, providing read-only access with the same API surface
//! as `Frame`.

use std::collections::HashMap;

use crate::block::block_view::BlockView;
use crate::frame::Frame;
use crate::region::simbox::SimBox;

/// A borrowed, read-only view of a [`Frame`].
///
/// Block keys are `&str` references into the original `Frame`'s key strings.
/// Block values are [`BlockView`]s that borrow the underlying column data.
/// `simbox` and `meta` are borrowed by reference.
pub struct FrameView<'a> {
    map: HashMap<&'a str, BlockView<'a>>,
    /// Borrowed simulation box, if present.
    pub simbox: Option<&'a SimBox>,
    /// Borrowed metadata map.
    pub meta: &'a HashMap<String, String>,
}

impl<'a> FrameView<'a> {
    /// Construct a FrameView from parts.
    pub fn from_parts(
        map: HashMap<&'a str, BlockView<'a>>,
        simbox: Option<&'a SimBox>,
        meta: &'a HashMap<String, String>,
    ) -> Self {
        Self { map, simbox, meta }
    }

    /// Gets an immutable reference to the block view for `key` if present.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&BlockView<'a>> {
        self.map.get(key)
    }

    /// Returns an iterator over `(&str, &BlockView)`.
    pub fn iter(&self) -> impl Iterator<Item = (&&'a str, &BlockView<'a>)> {
        self.map.iter()
    }

    /// Returns an iterator over block keys.
    pub fn keys(&self) -> impl Iterator<Item = &&'a str> {
        self.map.keys()
    }

    /// Returns an iterator over block view references.
    pub fn values(&self) -> impl Iterator<Item = &BlockView<'a>> {
        self.map.values()
    }

    /// Number of blocks in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the view contains no blocks.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns `true` if the view contains the specified block key.
    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Creates an owned [`Frame`] by cloning all viewed data.
    pub fn to_owned(&self) -> Frame {
        let mut block_map = HashMap::with_capacity(self.map.len());
        for (&key, block_view) in &self.map {
            block_map.insert(key.to_string(), block_view.to_owned());
        }
        let mut frame = Frame::from_map(block_map);
        frame.simbox = self.simbox.cloned();
        frame.meta = self.meta.clone();
        frame
    }
}

impl<'a> From<&'a Frame> for FrameView<'a> {
    fn from(frame: &'a Frame) -> Self {
        let mut map = HashMap::with_capacity(frame.len());
        for (key, block) in frame.iter() {
            map.insert(key, BlockView::from(block));
        }
        FrameView {
            map,
            simbox: frame.simbox.as_ref(),
            meta: &frame.meta,
        }
    }
}

impl std::fmt::Debug for FrameView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("FrameView");

        let mut blocks_map = std::collections::BTreeMap::new();
        for (&k, b) in &self.map {
            blocks_map.insert(k, (b.nrows(), b.len()));
        }
        debug_struct.field("blocks", &blocks_map);

        if !self.meta.is_empty() {
            debug_struct.field("meta", &self.meta);
        }

        debug_struct.finish()
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

        let mut bonds = Block::new();
        bonds
            .insert("type", Array1::from_vec(vec![1 as I, 2]).into_dyn())
            .unwrap();
        frame.insert("bonds", bonds);

        frame.meta.insert("title".into(), "Test".into());
        frame
    }

    #[test]
    fn test_from_frame() {
        let frame = make_frame();
        let view = FrameView::from(&frame);
        assert_eq!(view.len(), 2);
        assert!(view.contains_key("atoms"));
        assert!(view.contains_key("bonds"));
        assert!(!view.is_empty());
    }

    #[test]
    fn test_get_block_view() {
        let frame = make_frame();
        let view = FrameView::from(&frame);

        let atoms = view.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(3));
        assert!(atoms.get_float("x").is_some());
    }

    #[test]
    fn test_meta_borrowed() {
        let frame = make_frame();
        let view = FrameView::from(&frame);
        assert_eq!(view.meta.get("title").unwrap(), "Test");
    }

    #[test]
    fn test_simbox_none() {
        let frame = make_frame();
        let view = FrameView::from(&frame);
        assert!(view.simbox.is_none());
    }

    #[test]
    fn test_to_owned_roundtrip() {
        let frame = make_frame();
        let view = FrameView::from(&frame);
        let owned = view.to_owned();

        assert_eq!(owned.len(), 2);
        assert!(owned.contains_key("atoms"));
        assert!(owned.contains_key("bonds"));
        assert_eq!(owned.meta.get("title").unwrap(), "Test");

        let atoms = owned.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(3));
        assert_eq!(
            atoms
                .get_float("x")
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_iter_keys() {
        let frame = make_frame();
        let view = FrameView::from(&frame);

        let keys: Vec<&&str> = view.keys().collect();
        assert_eq!(keys.len(), 2);

        let mut count = 0;
        for (_name, _block) in view.iter() {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_zero_copy() {
        let frame = make_frame();
        let view = FrameView::from(&frame);

        let orig_ptr = frame.get("atoms").unwrap().get_float("x").unwrap().as_ptr();
        let view_ptr = view.get("atoms").unwrap().get_float("x").unwrap().as_ptr();
        assert_eq!(orig_ptr, view_ptr);
    }
}
