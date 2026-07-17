//! Streaming frame indexers shared across all trajectory formats.
//!
//! This module defines the [`FrameIndexBuilder`] trait used by the streaming
//! pipeline (worker → WASM → main thread). Callers feed the source file in
//! arbitrarily-sized chunks; each implementation maintains a chunk-boundary-safe
//! state machine that emits a [`FrameIndexEntry`] per parsed frame.
//!
//! Per-format implementations live in their own modules
//! (`lammps_dump`, `xyz`, `pdb`, `lammps_data`, `sdf`); each format also
//! exposes a `parse_frame_bytes(&[u8]) -> std::io::Result<Frame>` free
//! function that decodes exactly the byte slice produced by the matching
//! indexer.
//!
//! See `docs/specs/streaming-trajectory.md` (in the molvis repo) for the
//! full design.

/// One frame's location inside the source byte stream.
///
/// `byte_offset` is the absolute byte position of the frame's first byte
/// inside the source file. `byte_len` is the number of bytes that make up
/// the frame; the slice `source[byte_offset..byte_offset + byte_len as u64]`
/// must be a self-contained input that the matching `parse_frame_bytes`
/// function can decode into a single [`Frame`](molrs::store::frame::Frame).
///
/// `byte_len` is intentionally `u32`. Per-frame size is bounded — even very
/// large MD trajectories use frames in the kilobyte-to-megabyte range, and a
/// frame larger than 4 GiB is well outside the protocol's design envelope.
/// The global `byte_offset` carries `u64` so the protocol supports source
/// files up to 1 TB (well within `Number.MAX_SAFE_INTEGER` for the WASM/JS
/// bridge).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameIndexEntry {
    /// Absolute offset (in bytes) from the start of the source.
    pub byte_offset: u64,
    /// Length, in bytes, of this frame.
    pub byte_len: u32,
}

/// Stream-friendly indexer: callers feed the raw file as zero-copy chunks
/// in source order; the indexer produces frame entries as soon as it has
/// scanned past their tail bytes.
///
/// The trait is intentionally `&mut self` rather than consuming so that
/// implementations can reuse internal scratch (line buffer, partial-frame
/// state) across calls.
///
/// Lifecycle:
///   1. Construct with `new()`.
///   2. Call [`feed`](FrameIndexBuilder::feed) repeatedly with byte chunks
///      in source order. After each `feed`, optionally call
///      [`drain`](FrameIndexBuilder::drain) to harvest any frame entries
///      that have become finalized.
///   3. After the source reaches EOF, call
///      [`finish`](FrameIndexBuilder::finish) to flush the trailing frame
///      (if any) and consume the indexer.
///
/// Implementations MUST tolerate chunks that split lines (LF or CRLF in the
/// middle), and MUST tolerate frames that span multiple chunks. After
/// `finish`, further `feed` calls panic.
pub trait FrameIndexBuilder: Send {
    /// Push the next chunk of bytes. `global_offset` is the absolute byte
    /// position of `chunk[0]` inside the source stream.
    ///
    /// Implementations MUST tolerate chunks that split lines (LF or CRLF
    /// in the middle), and MUST tolerate frames that span multiple chunks.
    fn feed(&mut self, chunk: &[u8], global_offset: u64);

    /// Drain frame entries that have been fully observed since the last
    /// `drain` (or `feed` if no prior `drain`). Successive calls are
    /// monotonic — each entry is yielded exactly once.
    fn drain(&mut self) -> Vec<FrameIndexEntry>;

    /// Called once the source has reached EOF. Yields any trailing frame
    /// that was held back because its end wasn't yet observed, plus
    /// any frame entries still pending in the drain queue.
    ///
    /// After `finish`, the indexer is exhausted; further `feed` calls
    /// must panic.
    fn finish(self: Box<Self>) -> std::io::Result<Vec<FrameIndexEntry>>;

    /// How many bytes have been consumed so far. Used by the worker to
    /// drive `index-progress` reports.
    fn bytes_seen(&self) -> u64;
}

// ============================================================================
// Shared chunk-boundary helpers
// ============================================================================

/// Scratch state for "line-oriented" indexers. Every text format on the
/// streaming path is line-driven, so we factor the chunk-boundary logic
/// here.
///
/// Caller pattern:
/// ```ignore
/// let mut acc = LineAccumulator::default();
/// acc.feed(chunk, global_offset, |line, line_offset, line_len| {
///     // process one complete line; `line_offset` is the absolute byte
///     // position of the line's first byte; `line_len` is the number of
///     // bytes occupied (including any trailing \r\n or \n). The
///     // `line` slice does NOT include the trailing newline characters.
/// });
/// // ... eventually ...
/// acc.finish(|line, line_offset, line_len| { ... });
/// ```
///
/// Lines are emitted as `&str` if valid UTF-8, or `&[u8]` reinterpreted
/// via `String::from_utf8_lossy` (matching the semantics of `BufRead::read_line`).
/// We keep things in `&str` here because every format's parser is text-based.
#[derive(Default)]
pub(crate) struct LineAccumulator {
    /// Bytes carried over from a previous `feed` because the previous chunk
    /// did not end on a newline.
    carry: Vec<u8>,
    /// Absolute byte offset of `carry[0]`. Only meaningful when `!carry.is_empty()`.
    carry_offset: u64,
    /// Total bytes ever fed in (sum of all chunk lengths).
    bytes_seen: u64,
    /// Whether `finish` has been called.
    finished: bool,
}

impl LineAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes seen across all `feed` calls so far.
    pub fn bytes_seen(&self) -> u64 {
        self.bytes_seen
    }

    /// Push the next chunk. Each complete line is delivered via `f` as
    /// `f(line_text_without_trailing_newline, line_offset, line_byte_len)`.
    /// `line_byte_len` includes the trailing `\n` or `\r\n`.
    pub fn feed<F>(&mut self, chunk: &[u8], global_offset: u64, mut f: F)
    where
        F: FnMut(&str, u64, u32),
    {
        if self.finished {
            panic!("LineAccumulator::feed called after finish");
        }
        self.bytes_seen = global_offset.saturating_add(chunk.len() as u64);

        if chunk.is_empty() {
            return;
        }

        // If we have carry-over bytes, the "logical" buffer is carry ++ chunk.
        // We walk over the chunk only and synthesize line slices that point
        // into either `carry+chunk` (when the line spans the boundary) or
        // directly into `chunk`.
        //
        // Strategy: scan the chunk for `\n`. For each newline found at chunk
        // position `i`, the line ends at `chunk[..=i]`. If we have carry, the
        // first line is `carry ++ chunk[..=i]` — emit by buffering. Subsequent
        // lines are emitted directly from a window inside `chunk`.

        let mut start_in_chunk: usize = 0;

        for (i, &b) in chunk.iter().enumerate() {
            if b != b'\n' {
                continue;
            }
            // Line: previous-line-start .. i (inclusive of newline).
            let line_end_excl_in_chunk = i + 1;
            let line_byte_len_in_chunk = line_end_excl_in_chunk - start_in_chunk;

            if !self.carry.is_empty() && start_in_chunk == 0 {
                // First line crosses the chunk boundary: carry ++ chunk[..=i].
                let mut combined = std::mem::take(&mut self.carry);
                combined.extend_from_slice(&chunk[..line_end_excl_in_chunk]);
                let line_offset = self.carry_offset;
                let total_len = combined.len() as u32;
                let trimmed = trim_trailing_newline(&combined);
                let s = std::str::from_utf8(trimmed).unwrap_or("");
                f(s, line_offset, total_len);
            } else {
                let slice = &chunk[start_in_chunk..line_end_excl_in_chunk];
                let line_offset = global_offset + start_in_chunk as u64;
                let total_len = line_byte_len_in_chunk as u32;
                let trimmed = trim_trailing_newline(slice);
                let s = std::str::from_utf8(trimmed).unwrap_or("");
                f(s, line_offset, total_len);
            }
            start_in_chunk = line_end_excl_in_chunk;
        }

        // Anything between `start_in_chunk` and end-of-chunk is a partial
        // trailing line — carry it into the next feed.
        if start_in_chunk < chunk.len() {
            if self.carry.is_empty() {
                // Brand-new partial line; remember its start offset.
                self.carry_offset = global_offset + start_in_chunk as u64;
            }
            // (If carry was non-empty, carry_offset is already the offset
            // of the first carry byte from a previous chunk — the boundary
            // line whose newline we haven't seen yet.)
            self.carry.extend_from_slice(&chunk[start_in_chunk..]);
        } else if !self.carry.is_empty() && start_in_chunk == 0 {
            // The chunk had zero newlines, so the entire chunk is part of
            // an existing carry-over line. Append.
            self.carry.extend_from_slice(chunk);
        }
    }

    /// Flush trailing partial line (no terminating newline) as a final line.
    /// Idempotent — calling twice is harmless (subsequent calls do nothing).
    pub fn finish<F>(&mut self, mut f: F)
    where
        F: FnMut(&str, u64, u32),
    {
        if self.finished {
            return;
        }
        self.finished = true;
        if !self.carry.is_empty() {
            let line_offset = self.carry_offset;
            let total_len = self.carry.len() as u32;
            let s = std::str::from_utf8(&self.carry).unwrap_or("");
            // The trailing partial line has no terminator, so the `trim`
            // call would be a no-op. Pass the raw slice so callers' state
            // machines see exactly what they would have seen if the file
            // had ended without a newline (the most common edge case).
            f(s, line_offset, total_len);
            self.carry.clear();
        }
    }
}

/// Strip a trailing `\n` or `\r\n` from `s`, returning the shorter slice.
fn trim_trailing_newline(s: &[u8]) -> &[u8] {
    if let Some((&b'\n', rest)) = s.split_last() {
        if let Some((&b'\r', rest2)) = rest.split_last() {
            rest2
        } else {
            rest
        }
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect all lines emitted from feeding `s` byte-by-byte and confirm
    /// they match a single-shot feed.
    #[test]
    fn line_accumulator_byte_by_byte_matches_single_shot() {
        let s: &[u8] = b"abc\ndef\r\nghi\n";

        let mut want: Vec<(String, u64, u32)> = Vec::new();
        let mut single = LineAccumulator::new();
        single.feed(s, 0, |line, off, len| {
            want.push((line.to_string(), off, len));
        });
        single.finish(|line, off, len| want.push((line.to_string(), off, len)));

        let mut got: Vec<(String, u64, u32)> = Vec::new();
        let mut acc = LineAccumulator::new();
        for (i, &b) in s.iter().enumerate() {
            acc.feed(&[b], i as u64, |line, off, len| {
                got.push((line.to_string(), off, len));
            });
        }
        acc.finish(|line, off, len| got.push((line.to_string(), off, len)));

        assert_eq!(got, want);
    }

    /// Trailing partial line (no newline) must be emitted from `finish`.
    #[test]
    fn line_accumulator_handles_no_trailing_newline() {
        let s: &[u8] = b"abc\ndef";
        let mut acc = LineAccumulator::new();
        let mut got: Vec<(String, u64, u32)> = Vec::new();
        acc.feed(s, 0, |line, off, len| {
            got.push((line.to_string(), off, len));
        });
        acc.finish(|line, off, len| got.push((line.to_string(), off, len)));

        assert_eq!(got, vec![("abc".into(), 0, 4), ("def".into(), 4, 3)]);
    }

    /// Chunk boundary mid-CRLF: \r at end of chunk-1, \n at start of chunk-2.
    #[test]
    fn line_accumulator_handles_crlf_split_across_chunks() {
        let s: &[u8] = b"abc\r\ndef\n";
        let mut acc = LineAccumulator::new();
        let mut got: Vec<(String, u64, u32)> = Vec::new();
        acc.feed(&s[..4], 0, |line, off, len| {
            got.push((line.to_string(), off, len));
        });
        acc.feed(&s[4..], 4, |line, off, len| {
            got.push((line.to_string(), off, len));
        });
        acc.finish(|line, off, len| got.push((line.to_string(), off, len)));

        assert_eq!(got, vec![("abc".into(), 0, 5), ("def".into(), 5, 4)]);
    }
}
