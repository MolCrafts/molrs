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

/// Iterator over the frames of a [`TrajectoryReader`], from step 0 until
/// [`TrajectoryReader::read_step`] yields `None`.
///
/// Yields `Result<Frame>`. An `Err` leaves the cursor where it was, so the
/// iteration must be stopped on the first one rather than polled past it.
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

/// Random access by step over an ordered sequence of frames, whatever it is
/// stored in.
///
/// A *trajectory* here is any ordered sequence of [`Frame`]s addressed by a
/// 0-based step index. The contract is deliberately **backend-neutral**: it
/// says nothing about files, byte offsets or seeking, so a reader over a DCD
/// file and a reader over a Zarr store (`io::zarr`'s `FrameSequence`) implement
/// the same three methods. That is also why [`Reader`] — which demands an
/// underlying `BufRead` — is **not** a supertrait of this one: a store-backed
/// reader has no byte stream to name.
///
/// Dropping that supertrait also made this trait **dyn-compatible** — a
/// capability callers now rely on, so re-adding one would break them.
/// [`Reader`] is itself dyn-incompatible (`fn new(Self::R) -> Self` takes no
/// receiver and returns `Self` by value), and a dyn-incompatible supertrait
/// makes `&mut dyn TrajectoryReader` illegal. The one method a trait object
/// cannot reach is [`iter`](Self::iter), which is `where Self: Sized`; a `dyn`
/// caller loops on [`read_step`](Self::read_step) until it yields `None`.
///
/// # Errors
///
/// All three required methods return `std::io::Result`. That is the lowest
/// common denominator, not a claim that every backend's failures are IO
/// failures: a backend with a richer error type flattens it in at this boundary
/// — lossy, and deliberately so, since the trait must not be captured by one
/// backend's error type. A caller who needs the structured error uses the
/// concrete reader's own doors.
pub trait TrajectoryReader {
    /// Build and cache whatever per-step index the backend needs for random
    /// access. File readers cache byte offsets; store-backed readers cache
    /// their per-step index arrays.
    ///
    /// Calling it is never *required* — an implementation may already have its
    /// index (a store-backed reader builds one when it opens) and answer with
    /// `Ok(())` — but a caller about to make many [`read_step`](Self::read_step)
    /// calls should call it once first, since a reader that scans lazily would
    /// otherwise pay for the scan on the first read.
    ///
    /// # Errors
    ///
    /// Whatever the backend raised while scanning or reading its index.
    fn build_index(&mut self) -> Result<()>;

    /// Read the frame at a given step index (0-based).
    ///
    /// `Ok(None)` means `step` is past the end — the sequence has no such
    /// frame. It is a terminator, not a failure, and it is how
    /// [`FrameIterator`] knows to stop; an implementation must not return
    /// `Ok(None)` for a step it merely could not decode.
    ///
    /// # Errors
    ///
    /// Whatever the backend raised while reading or decoding the frame.
    fn read_step(&mut self, step: usize) -> Result<Option<Frame>>;

    /// Total number of frames the reader can serve.
    ///
    /// Every index in `0..len()` yields `Some` from
    /// [`read_step`](Self::read_step), and every index at or past it yields
    /// `None`. A backend that commits in batches counts only what it has
    /// committed.
    ///
    /// # Errors
    ///
    /// Whatever the backend raised while determining the count; a reader that
    /// already knows it returns `Ok` unconditionally.
    fn len(&mut self) -> Result<usize>;

    /// Whether the trajectory holds no frames at all.
    ///
    /// # Errors
    ///
    /// [`len`](Self::len)'s, which this defers to.
    fn is_empty(&mut self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Iterate frames from step 0 until [`read_step`](Self::read_step) yields
    /// `None`.
    ///
    /// Each item is a `Result`. An `Err` does **not** advance the cursor and
    /// does not end the iteration, so polling past one yields the same error
    /// again: stop on the first `Err`, most simply by collecting into
    /// `Result<Vec<Frame>, _>`. Unavailable through a trait object — see the
    /// trait docs.
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
/// Sequential trajectory IO buffer size (bytes).
///
/// Default `BufReader` is 8 KiB — far too small for multi‑GB TRR/XTC sequential
/// scans (one header+payload per frame). 8 MiB amortizes syscalls / NFS RTT
/// without pinning excessive RAM per open handle.
const TRAJECTORY_BUF_CAPACITY: usize = 8 * 1024 * 1024;

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
        let mut buf_decoder = BufReader::with_capacity(TRAJECTORY_BUF_CAPACITY, decoder);
        buf_decoder.read_to_end(&mut content)?;
        Ok(Box::new(std::io::Cursor::new(content)))
    } else {
        Ok(Box::new(BufReader::with_capacity(
            TRAJECTORY_BUF_CAPACITY,
            file,
        )))
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
        Ok(Box::new(BufReader::with_capacity(
            TRAJECTORY_BUF_CAPACITY,
            decoder,
        )))
    } else {
        Ok(Box::new(BufReader::with_capacity(
            TRAJECTORY_BUF_CAPACITY,
            file,
        )))
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
/// Every [`FrameReader::read`] returns through this. The report names
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
    use super::{Frame, TrajectoryReader, open_seekable, open_streaming};
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::{BufRead, Write};
    use std::path::PathBuf;

    /// Number of frames the store-less reader below pretends to hold.
    const STORELESS_FRAMES: usize = 2;

    /// A [`TrajectoryReader`] backed by nothing at all — no file, no
    /// `BufRead`, no `Seek`.
    ///
    /// This is the guard for the Zarr-backed `FrameSequence`: a backend whose
    /// frames come out of a store rather than a byte stream must be able to
    /// implement [`TrajectoryReader`] without an underlying reader. It
    /// deliberately does **not** `impl Reader` (which would demand
    /// `type R: BufRead` and `fn new(Self::R)` that this type cannot supply);
    /// it only becomes legal once `Reader` stops being a supertrait of
    /// [`TrajectoryReader`].
    struct StorelessReader {
        frames: usize,
    }

    impl TrajectoryReader for StorelessReader {
        fn build_index(&mut self) -> std::io::Result<()> {
            Ok(())
        }

        fn read_step(&mut self, step: usize) -> std::io::Result<Option<Frame>> {
            if step < self.frames {
                Ok(Some(Frame::new()))
            } else {
                Ok(None)
            }
        }

        fn len(&mut self) -> std::io::Result<usize> {
            Ok(self.frames)
        }
    }

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

    #[test]
    fn trajectory_reader_needs_no_bufread_source() {
        let mut reader = StorelessReader {
            frames: STORELESS_FRAMES,
        };

        reader
            .build_index()
            .expect("build_index on a store-less reader");
        assert_eq!(reader.len().expect("len"), 2);
        assert!(reader.read_step(0).expect("read_step(0)").is_some());
        assert!(reader.read_step(1).expect("read_step(1)").is_some());
        assert!(reader.read_step(2).expect("read_step(2)").is_none());

        let frames: Vec<Frame> = reader
            .iter()
            .collect::<std::io::Result<Vec<Frame>>>()
            .expect("iterate a store-less reader");
        assert_eq!(frames.len(), 2);
    }
}
