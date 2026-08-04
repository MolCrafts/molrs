use std::io::Result;
use std::io::Write;

use crate::store::frame::Frame;

/// Generic writer for data destinations.
pub trait Writer {
    /// Underlying writer type.
    type W: Write;
    /// Construct a new writer from the underlying writer.
    fn new(writer: Self::W) -> Self;
}

/// A writer that emits one logical frame at a time.
///
/// # Shape
///
/// Mirrors [`FrameReader`](crate::io::reader::FrameReader): [`write`] takes a
/// [`Frame`], [`write_from`] takes anything that can become one. Serialization
/// lives in a stateless free function per format, not in this trait.
///
/// [`write`]: Self::write
/// [`write_from`]: Self::write_from
pub trait FrameWriter: Writer {
    /// Write one frame.
    ///
    /// The frame is checked against the Frame schema first. A non-conforming
    /// frame produces a file that looks fine and is wrong — the expensive kind
    /// of failure, found later by whatever reads it.
    fn write(&mut self, frame: &Frame) -> Result<()>;

    /// Write a value that can be expressed as a frame.
    fn write_from<T: ToFrame>(&mut self, value: &T) -> Result<()> {
        self.write(&value.to_frame()?)
    }
}

/// Produce a [`Frame`] from a borrowed value.
///
/// The source side of [`FrameWriter::write_from`].
pub trait ToFrame {
    /// Convert, or explain why this value cannot be expressed as a frame.
    fn to_frame(&self) -> Result<Frame>;
}

impl ToFrame for Frame {
    fn to_frame(&self) -> Result<Frame> {
        Ok(self.clone())
    }
}

impl ToFrame for crate::system::atomistic::Atomistic {
    fn to_frame(&self) -> Result<Frame> {
        // Disambiguate from the inherent `Atomistic::to_frame`, which this
        // wraps; calling `self.to_frame()` here would recurse.
        Ok(crate::system::atomistic::Atomistic::to_frame(self))
    }
}

/// Check a frame against the Frame schema before writing it.
///
/// Every [`FrameWriter::write_frame`] calls this first.
pub fn check_before_write<F: crate::store::frame_access::FrameAccess>(frame: &F) -> Result<()> {
    crate::store::schema::Validator::canonical()
        .validate(frame)
        .map_err(|report| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("refusing to write a frame that violates the Frame schema:\n{report}"),
            )
        })
}
