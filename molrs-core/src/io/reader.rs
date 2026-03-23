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
    /// The frame type produced by this reader.
    type Frame;
    /// Construct a new reader from the underlying buffered reader.
    fn new(reader: Self::R) -> Self;
}

/// Reader that can read one logical frame at a time.
pub trait FrameReader: Reader {
    /// Read a single frame from the current stream position. Returns Ok(None) on EOF.
    fn read_frame(&mut self) -> Result<Option<Self::Frame>>;

    /// Read all frames from the stream.
    fn read_all(&mut self) -> Result<Vec<Self::Frame>> {
        let mut out = Vec::new();
        while let Some(frame) = self.read_frame()? {
            out.push(frame);
        }
        Ok(out)
    }
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
pub struct FrameIterator<'a, R: TrajReader + ?Sized> {
    reader: &'a mut R,
    current: usize,
}

impl<'a, R: TrajReader> Iterator for FrameIterator<'a, R> {
    type Item = Result<R::Frame>;

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
pub trait TrajReader: Reader {
    /// Build and cache an index mapping step numbers to byte offsets.
    fn build_index(&mut self) -> Result<()>;

    /// Read a frame at a given step index (0-based).
    fn read_step(&mut self, step: usize) -> Result<Option<Self::Frame>>;

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
