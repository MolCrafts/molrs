use crate::store::frame::Frame;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Result, Seek};
use std::path::Path;

/// Trait for readable and seekable buffers (used for trait objects)
pub trait ReadSeek: BufRead + Seek {}
impl<T: BufRead + Seek> ReadSeek for T {}

/// Open a plain text file and return a buffered reader.
pub fn open_txt(path: &str) -> Result<BufReader<File>> {
    let file = File::open(path)?;
    Ok(BufReader::new(file))
}

/// Open a gzip-compressed file and return a buffered reader over the decompressed stream.
pub fn open_gz(path: &str) -> Result<BufReader<GzDecoder<File>>> {
    let file = File::open(path)?;
    let decoder = GzDecoder::new(file);
    Ok(BufReader::new(decoder))
}

/// Reader for data sources returning frame-like records.
pub trait Reader {
    /// Underlying buffered reader type.
    type R: BufRead;
    /// Construct a new reader from the underlying buffered reader.
    fn new(reader: Self::R) -> Self;
}

/// Reader that yields one logical frame at a time.
///
/// # Shape
///
/// Parsing lives in a **stateless free function** per format, not in this
/// trait. [`read`](Self::read) calls it and returns a validated [`Frame`];
/// [`read_as`](Self::read_as) returns whatever type the caller wants. Multiple
/// frames are [`TrajectoryReader`]'s job, not this trait's.
pub trait FrameReader: Reader {
    /// Read one frame from the current position. `Ok(None)` on EOF.
    ///
    /// The frame is checked against the Frame schema before it is returned: a
    /// frame that violates the vocabulary is a malformed file or a reader bug,
    /// not a value to hand back as if it were fine.
    fn read(&mut self) -> Result<Option<Frame>>;

    /// Read one record as `T`.
    ///
    /// The default converts from [`read`](Self::read). A reader whose parser
    /// natively produces `T` should override this and return that value
    /// directly — `SmilesReader` parses to an [`Atomistic`], so
    /// `read_as::<Atomistic>()` hands it over instead of going
    /// `Atomistic -> Frame -> Atomistic`.
    ///
    /// [`Atomistic`]: crate::system::atomistic::Atomistic
    fn read_as<T: FromFrame>(&mut self) -> Result<Option<T>> {
        match self.read()? {
            Some(frame) => Ok(Some(T::from_frame(&frame)?)),
            None => Ok(None),
        }
    }
}

/// Build a value from a [`Frame`].
///
/// The target side of [`FrameReader::read_as`] and the source side of
/// [`FrameWriter::write_from`](crate::io::writer::FrameWriter::write_from).
pub trait FromFrame: Sized {
    /// Convert, or explain why the frame cannot express this type.
    fn from_frame(frame: &Frame) -> Result<Self>;

    /// Fast path for readers whose parser natively produces an [`Atomistic`].
    ///
    /// Defaults to going through a frame, which is right for every type that
    /// is not itself an `Atomistic`. [`Atomistic`] overrides it to return the
    /// value unchanged, so `SmilesReader::read_as::<Atomistic>()` costs no
    /// conversion at all.
    ///
    /// This is admittedly a bulge in an otherwise frame-shaped trait. The
    /// alternative — an associated "native type" on the reader — infects every
    /// one of the twelve file readers, all of which natively produce frames,
    /// to serve the one reader that does not.
    ///
    /// [`Atomistic`]: crate::system::atomistic::Atomistic
    fn from_atomistic(mol: crate::system::atomistic::Atomistic) -> Result<Self> {
        Self::from_frame(&mol.to_frame())
    }
}

impl FromFrame for Frame {
    fn from_frame(frame: &Frame) -> Result<Self> {
        Ok(frame.clone())
    }
}

impl FromFrame for crate::system::atomistic::Atomistic {
    fn from_frame(frame: &Frame) -> Result<Self> {
        crate::system::atomistic::Atomistic::from_frame(frame)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }

    fn from_atomistic(mol: crate::system::atomistic::Atomistic) -> Result<Self> {
        Ok(mol)
    }
}

/// Drain a [`FrameReader`] into a `Vec`.
///
/// `FrameReader::read_all` used to live on the trait, duplicating
/// [`TrajectoryReader`]'s job — multi-frame access is that trait's whole
/// purpose. Formats that are genuinely indexable should use
/// [`TrajectoryReader::iter`]; this is for the ones that can only stream
/// forward.
pub fn collect_frames<R: FrameReader>(reader: &mut R) -> Result<Vec<Frame>> {
    let mut out = Vec::new();
    while let Some(frame) = reader.read()? {
        out.push(frame);
    }
    Ok(out)
}

/// Frame index storing byte offsets for each frame in a trajectory
#[derive(Debug, Clone)]
pub struct FrameIndex {
    /// Byte offset of each frame start position
    pub offsets: Vec<u64>,
}

impl FrameIndex {
    /// Create a new empty frame index
    pub fn new() -> Self {
        Self {
            offsets: Vec::new(),
        }
    }

    /// Add a frame offset to the index
    pub fn add_frame(&mut self, offset: u64) {
        self.offsets.push(offset);
    }

    /// Get number of frames in index
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Get offset for a specific frame
    pub fn get(&self, step: usize) -> Option<u64> {
        self.offsets.get(step).copied()
    }
}

impl Default for FrameIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over frames in a trajectory reader
pub struct FrameIterator<'a, R: TrajectoryReader + ?Sized> {
    reader: &'a mut R,
    current: usize,
}

impl<'a, R: TrajectoryReader> Iterator for FrameIterator<'a, R> {
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_step(self.current) {
            Ok(Some(frame)) => {
                self.current += 1;
                Some(Ok(frame))
            }
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Reader over a trajectory-like file supporting random access by step.
pub trait TrajectoryReader: Reader {
    /// Build and cache an index mapping step numbers to byte offsets.
    fn build_index(&mut self) -> Result<()>;

    /// Read a frame at a given step index (0-based).
    fn read_step(&mut self, step: usize) -> Result<Option<Frame>>;

    /// Get total number of frames in the file.
    fn len(&mut self) -> Result<usize>;

    /// Check if the trajectory is empty (contains no frames).
    fn is_empty(&mut self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Create an iterator over all frames.
    fn iter(&mut self) -> FrameIterator<'_, Self>
    where
        Self: Sized,
    {
        FrameIterator {
            reader: self,
            current: 0,
        }
    }
}

/// Open a seekable file reader with automatic gzip detection based on extension.
///
/// Files with `.gz` extension are decompressed into memory to provide seekability.
///
/// # Examples
///
/// ```no_run
/// use molrs::io::reader::open_seekable;
///
/// # fn main() -> std::io::Result<()> {
/// // Opens and decompresses automatically (seekable)
/// let reader = open_seekable("data.xyz.gz")?;
///
/// // Direct read for uncompressed
/// let reader = open_seekable("data.xyz")?;
/// # Ok(())
/// # }
/// ```
pub fn open_seekable<P: AsRef<Path>>(path: P) -> Result<Box<dyn ReadSeek>> {
    let path = path.as_ref();
    let file = File::open(path)?;

    if path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
    {
        // For gzipped files, decompress into memory for seekability
        use std::io::Read;
        let decoder = GzDecoder::new(file);
        let mut content = Vec::new();
        let mut buf_decoder = BufReader::new(decoder);
        buf_decoder.read_to_end(&mut content)?;
        Ok(Box::new(std::io::Cursor::new(content)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Open a streaming file reader with automatic gzip detection based on extension.
///
/// Files with `.gz` extension are decompressed on the fly and are not seekable.
pub fn open_streaming<P: AsRef<Path>>(path: P) -> Result<Box<dyn BufRead>> {
    let path = path.as_ref();
    let file = File::open(path)?;

    if path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
    {
        let decoder = GzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Open a file with automatic gzip detection based on extension.
///
/// This is a compatibility wrapper that returns a seekable reader.
pub fn open_file<P: AsRef<Path>>(path: P) -> Result<Box<dyn ReadSeek>> {
    open_seekable(path)
}

/// Check a freshly-read frame against the Frame schema.
///
/// Every [`FrameReader::read_frame`] returns through this. The report names
/// every offending column at once, so a malformed file takes one round trip to
/// diagnose rather than one per bad column.
pub fn validated<F: crate::store::frame_access::FrameAccess>(
    frame: Option<F>,
) -> Result<Option<F>> {
    if let Some(ref f) = frame {
        crate::store::schema::Validator::canonical()
            .validate(f)
            .map_err(|report| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("frame does not conform to the Frame schema:\n{report}"),
                )
            })?;
    }
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::{open_seekable, open_streaming};
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::{BufRead, Write};
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("molrs_reader_test_{}", name));
        path
    }

    #[test]
    fn open_seekable_plain_text() {
        let path = temp_path("plain.txt");
        std::fs::write(&path, b"hello\n").expect("write temp");
        let mut reader = open_seekable(&path).expect("open seekable");
        let mut line = String::new();
        reader.read_line(&mut line).expect("read line");
        assert_eq!(line, "hello\n");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_streaming_gz() {
        let path = temp_path("data.txt.gz");
        let file = std::fs::File::create(&path).expect("create gz");
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(b"hello\n").expect("write gz");
        encoder.finish().expect("finish gz");

        let mut reader = open_streaming(&path).expect("open streaming");
        let mut line = String::new();
        reader.read_line(&mut line).expect("read line");
        assert_eq!(line, "hello\n");
        let _ = std::fs::remove_file(&path);
    }
}
