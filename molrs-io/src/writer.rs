use std::io::Result;
use std::io::Write;

/// Generic writer for data destinations.
pub trait Writer {
    /// Underlying writer type.
    type W: Write;
    /// The frame type consumed by this writer.
    type FrameLike;
    /// Construct a new writer from the underlying writer.
    fn new(writer: Self::W) -> Self;
}

/// A writer that can write one logical frame at a time.
pub trait FrameWriter: Writer {
    /// Write a single frame to the stream.
    fn write_frame(&mut self, frame: &Self::FrameLike) -> Result<()>;
}
