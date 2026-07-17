//! GROMACS XTC binary trajectory reader and writer.
//!
//! XTC is the compressed GROMACS trajectory format: an XDR (big-endian) stream
//! of per-frame records carrying step, time, box, and **lossily compressed**
//! coordinates (nm). Compression quantizes each coordinate to an integer
//! (`round(x * precision)`), then bit-packs the integers, exploiting that
//! consecutive atoms are usually spatially close.
//!
//! # Per-frame header (XDR, big-endian)
//!
//! ```text
//! magic   i32 = 1995 (also accepts 2023, a forward-magic test variant)
//! natoms  i32
//! step    i32
//! time    f32
//! box     9 × f32 (3×3, nm)
//! ```
//!
//! followed by the coordinate block:
//!
//! ```text
//! size      i32 (== natoms)
//! if natoms <= 9:  3*natoms uncompressed f32
//! else:
//!   precision f32
//!   minint[3] i32   maxint[3] i32   smallidx i32
//!   nbytes    i32
//!   buf[nbytes] (XDR opaque, padded to 4)   -- the compressed bitstream
//! ```
//!
//! The compression codec (`magicints` table, `receivebits`/`receiveints`
//! decode, `sendbits`/`sendints` encode) is a clean-room reimplementation of
//! the documented `xdr3dfcoord` algorithm — not transcribed from xdrfile or
//! any GPL source. It is validated behaviourally against the real chemfiles
//! `tests-data/xtc/` fixtures.
//!
//! # Output Frame
//!
//! - `atoms` block: `id` (1-based), `x`/`y`/`z` (nm).
//! - `frame.simbox`: from the box (row-stored vectors → column-stored H).
//! - `frame.meta`: `step`, `time`, `precision`.

use crate::io::reader::{FrameReader, ReadSeek, Reader, TrajReader};
use crate::io::trajectory::xdr;
use crate::io::writer::{FrameWriter, Writer};
use molrs::spatial::region::simbox::SimBox;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::store::frame_access::FrameAccess;
use molrs::types::{F, I};
use ndarray::{Array1, Array2, IxDyn, array};
use once_cell::sync::OnceCell;
use std::fs::File;
use std::io::{BufRead, BufWriter, Read, Result, Seek, SeekFrom, Write};
use std::path::Path;

/// Classic XTC magic number.
const XTC_MAGIC: i32 = 1995;
/// Forward-magic variant exercised by `ubiquitin_faux2023magic.xtc`.
const XTC_MAGIC_2023: i32 = 2023;
const DIM: usize = 3;
const FIRSTIDX: i32 = 9;

/// Bit-width selector table for the small-integer encoding.
const MAGICINTS: [i32; 73] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 101, 128, 161, 203, 256,
    322, 406, 512, 645, 812, 1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321,
    13003, 16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031, 131072, 165140, 208063,
    262144, 330280, 416127, 524287, 660561, 832255, 1048576, 1321122, 1664510, 2097152, 2642245,
    3329021, 4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
];
/// Largest valid index into [`MAGICINTS`].
const LASTIDX: i32 = MAGICINTS.len() as i32 - 1;

fn invalid<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

fn unsupported<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, e.to_string())
}

// ---------------------------------------------------------------------------
// Bit-width helpers
// ---------------------------------------------------------------------------

/// Number of bits needed to store an integer in `[0, size)`.
fn sizeofint(size: u32) -> i32 {
    let mut num: u32 = 1;
    let mut nbits = 0i32;
    while size >= num && nbits < 32 {
        nbits += 1;
        num = num.wrapping_shl(1);
    }
    nbits
}

/// Number of bits needed to store `num_of_ints` integers `< sizes[i]` as a
/// single mixed-radix number.
fn sizeofints(num_of_ints: usize, sizes: &[u32; 3]) -> i32 {
    let mut bytes = [0u32; 32];
    let mut num_of_bytes = 1usize;
    bytes[0] = 1;
    for &size in sizes.iter().take(num_of_ints) {
        let mut tmp = 0u64;
        let mut bytecnt = 0usize;
        while bytecnt < num_of_bytes {
            tmp += bytes[bytecnt] as u64 * size as u64;
            bytes[bytecnt] = (tmp & 0xff) as u32;
            tmp >>= 8;
            bytecnt += 1;
        }
        while tmp != 0 {
            bytes[bytecnt] = (tmp & 0xff) as u32;
            bytecnt += 1;
            tmp >>= 8;
        }
        num_of_bytes = bytecnt;
    }
    let last = num_of_bytes - 1;
    let mut num = 1u32;
    let mut nbits = 0i32;
    while bytes[last] >= num {
        nbits += 1;
        num = num.wrapping_mul(2);
    }
    nbits + (last as i32) * 8
}

// ---------------------------------------------------------------------------
// Bit reader (decode)
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    buf: &'a [u8],
    cnt: usize,
    lastbits: i32,
    lastbyte: u32,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            cnt: 0,
            lastbits: 0,
            lastbyte: 0,
        }
    }

    fn next_byte(&mut self) -> Result<u32> {
        let b = *self
            .buf
            .get(self.cnt)
            .ok_or_else(|| invalid("XTC compressed buffer underrun"))?;
        self.cnt += 1;
        Ok(b as u32)
    }

    /// Receive `num_of_bits` bits, MSB-first.
    fn receivebits(&mut self, mut num_of_bits: i32) -> Result<i32> {
        let mut num: u32 = 0;
        let mask = if num_of_bits >= 32 {
            u32::MAX
        } else {
            (1u32 << num_of_bits) - 1
        };
        while num_of_bits >= 8 {
            self.lastbyte = (self.lastbyte << 8) | self.next_byte()?;
            num |= (self.lastbyte >> self.lastbits) << (num_of_bits - 8);
            num_of_bits -= 8;
        }
        if num_of_bits > 0 {
            if self.lastbits < num_of_bits {
                self.lastbits += 8;
                self.lastbyte = (self.lastbyte << 8) | self.next_byte()?;
            }
            self.lastbits -= num_of_bits;
            num |= (self.lastbyte >> self.lastbits) & ((1u32 << num_of_bits) - 1);
        }
        num &= mask;
        Ok(num as i32)
    }

    /// Receive `num_of_ints` integers packed with `num_of_bits` total bits.
    fn receiveints(
        &mut self,
        num_of_ints: usize,
        mut num_of_bits: i32,
        sizes: &[u32; 3],
        nums: &mut [i32; 3],
    ) -> Result<()> {
        let mut bytes = [0i32; 32];
        let mut num_of_bytes = 0usize;
        while num_of_bits > 8 {
            bytes[num_of_bytes] = self.receivebits(8)?;
            num_of_bytes += 1;
            num_of_bits -= 8;
        }
        if num_of_bits > 0 {
            bytes[num_of_bytes] = self.receivebits(num_of_bits)?;
            num_of_bytes += 1;
        }
        for i in (1..num_of_ints).rev() {
            let mut num = 0u32;
            for j in (0..num_of_bytes).rev() {
                num = (num << 8) | bytes[j] as u32;
                let p = num / sizes[i];
                bytes[j] = p as i32;
                num -= p * sizes[i];
            }
            nums[i] = num as i32;
        }
        nums[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

#[inline]
fn check_idx(idx: i32) -> Result<usize> {
    if !(0..=LASTIDX).contains(&idx) {
        return Err(invalid(format!("XTC smallidx {idx} out of range")));
    }
    Ok(idx as usize)
}

/// Decompress a frame's coordinate bitstream into an interleaved `[x,y,z,…]`
/// nm vector of length `3*natoms`.
fn decompress_coords(
    bytes: &[u8],
    natoms: usize,
    precision: f32,
    minint: [i32; 3],
    maxint: [i32; 3],
    smallidx_init: i32,
) -> Result<Vec<f64>> {
    let inv: f32 = if precision != 0.0 {
        1.0 / precision
    } else {
        1.0
    };
    let mut sizeint = [0u32; 3];
    for d in 0..DIM {
        if maxint[d] < minint[d] {
            return Err(invalid("XTC maxint < minint"));
        }
        sizeint[d] = (maxint[d] - minint[d]) as u32 + 1;
    }

    let mut bitsizeint = [0i32; 3];
    let bitsize: i32;
    if (sizeint[0] | sizeint[1] | sizeint[2]) > 0x00ff_ffff {
        bitsizeint[0] = sizeofint(sizeint[0]);
        bitsizeint[1] = sizeofint(sizeint[1]);
        bitsizeint[2] = sizeofint(sizeint[2]);
        bitsize = 0;
    } else {
        bitsize = sizeofints(DIM, &sizeint);
    }

    let mut smallidx = smallidx_init;
    let init_smaller_idx = FIRSTIDX.max(smallidx - 1);
    let mut smaller = MAGICINTS[check_idx(init_smaller_idx)?] / 2;
    let mut smallnum = MAGICINTS[check_idx(smallidx)?] / 2;
    let mut sizesmall = [MAGICINTS[check_idx(smallidx)?] as u32; 3];

    let mut br = BitReader::new(bytes);
    let mut out: Vec<f64> = Vec::with_capacity(natoms * DIM);
    let emit = |out: &mut Vec<f64>, c: &[i32; 3]| {
        out.push((c[0] as f32 * inv) as f64);
        out.push((c[1] as f32 * inv) as f64);
        out.push((c[2] as f32 * inv) as f64);
    };

    let mut i = 0usize;
    // `run` persists across atoms: a `flag == 0` bit means "same run length as
    // the previous atom", so it must NOT be reset each iteration.
    let mut run = 0i32;
    while i < natoms {
        let mut thiscoord = [0i32; 3];
        if bitsize == 0 {
            thiscoord[0] = br.receivebits(bitsizeint[0])?;
            thiscoord[1] = br.receivebits(bitsizeint[1])?;
            thiscoord[2] = br.receivebits(bitsizeint[2])?;
        } else {
            br.receiveints(DIM, bitsize, &sizeint, &mut thiscoord)?;
        }
        i += 1;
        thiscoord[0] += minint[0];
        thiscoord[1] += minint[1];
        thiscoord[2] += minint[2];
        let mut prevcoord = thiscoord;

        let flag = br.receivebits(1)?;
        let mut is_smaller = 0i32;
        if flag == 1 {
            run = br.receivebits(5)?;
            is_smaller = run % 3;
            run -= is_smaller;
            is_smaller -= 1;
        }
        if run > 0 {
            if i + (run as usize / DIM) > natoms {
                return Err(invalid("XTC run length exceeds atom count"));
            }
            let mut k = 0i32;
            while k < run {
                let mut rc = [0i32; 3];
                br.receiveints(DIM, smallidx, &sizesmall, &mut rc)?;
                i += 1;
                rc[0] += prevcoord[0] - smallnum;
                rc[1] += prevcoord[1] - smallnum;
                rc[2] += prevcoord[2] - smallnum;
                if k == 0 {
                    std::mem::swap(&mut rc, &mut prevcoord);
                    emit(&mut out, &prevcoord);
                } else {
                    prevcoord = rc;
                }
                emit(&mut out, &rc);
                k += 3;
            }
        } else {
            emit(&mut out, &thiscoord);
        }

        smallidx += is_smaller;
        if is_smaller < 0 {
            smallnum = smaller;
            smaller = if smallidx > FIRSTIDX {
                MAGICINTS[check_idx(smallidx - 1)?] / 2
            } else {
                0
            };
        } else if is_smaller > 0 {
            smaller = smallnum;
            smallnum = MAGICINTS[check_idx(smallidx)?] / 2;
        }
        sizesmall = [MAGICINTS[check_idx(smallidx)?] as u32; 3];
    }

    if out.len() != natoms * DIM {
        return Err(invalid(format!(
            "XTC decode produced {} coords, expected {}",
            out.len(),
            natoms * DIM
        )));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Bit writer (encode)
// ---------------------------------------------------------------------------

struct BitWriter {
    data: Vec<u8>,
    cnt: usize,
    lastbits: i32,
    lastbyte: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            cnt: 0,
            lastbits: 0,
            lastbyte: 0,
        }
    }

    fn put(&mut self, idx: usize, val: u8) {
        if idx >= self.data.len() {
            self.data.resize(idx + 1, 0);
        }
        self.data[idx] = val;
    }

    fn sendbits(&mut self, mut num_of_bits: i32, num: u32) {
        let mut cnt = self.cnt;
        let mut lastbyte = self.lastbyte;
        let mut lastbits = self.lastbits;
        while num_of_bits >= 8 {
            lastbyte = (lastbyte << 8) | ((num >> (num_of_bits - 8)) & 0xff);
            self.put(cnt, (lastbyte >> lastbits) as u8);
            cnt += 1;
            num_of_bits -= 8;
        }
        if num_of_bits > 0 {
            let mask = (1u32 << num_of_bits) - 1;
            lastbyte = (lastbyte << num_of_bits) | (num & mask);
            lastbits += num_of_bits;
            if lastbits >= 8 {
                lastbits -= 8;
                self.put(cnt, (lastbyte >> lastbits) as u8);
                cnt += 1;
            }
        }
        self.cnt = cnt;
        self.lastbyte = lastbyte;
        self.lastbits = lastbits;
        if lastbits > 0 {
            self.put(cnt, (lastbyte << (8 - lastbits)) as u8);
        }
    }

    fn sendints(
        &mut self,
        num_of_ints: usize,
        num_of_bits: i32,
        sizes: &[u32; 3],
        nums: &[u32; 3],
    ) {
        let mut bytes = [0u32; 32];
        let mut tmp = nums[0];
        let mut num_of_bytes = 0usize;
        loop {
            bytes[num_of_bytes] = tmp & 0xff;
            num_of_bytes += 1;
            tmp >>= 8;
            if tmp == 0 {
                break;
            }
        }
        for i in 1..num_of_ints {
            let mut carry = nums[i] as u64;
            let mut bytecnt = 0usize;
            while bytecnt < num_of_bytes {
                carry += bytes[bytecnt] as u64 * sizes[i] as u64;
                bytes[bytecnt] = (carry & 0xff) as u32;
                carry >>= 8;
                bytecnt += 1;
            }
            while carry != 0 {
                bytes[bytecnt] = (carry & 0xff) as u32;
                bytecnt += 1;
                carry >>= 8;
            }
            num_of_bytes = bytecnt;
        }
        if num_of_bits >= (num_of_bytes as i32) * 8 {
            for &b in bytes.iter().take(num_of_bytes) {
                self.sendbits(8, b);
            }
            self.sendbits(num_of_bits - (num_of_bytes as i32) * 8, 0);
        } else {
            for &b in bytes.iter().take(num_of_bytes - 1) {
                self.sendbits(8, b);
            }
            self.sendbits(
                num_of_bits - ((num_of_bytes - 1) as i32) * 8,
                bytes[num_of_bytes - 1],
            );
        }
    }

    fn finish(mut self) -> Vec<u8> {
        let mut nbytes = self.cnt;
        if self.lastbits != 0 {
            nbytes += 1;
        }
        if self.data.len() < nbytes {
            self.data.resize(nbytes, 0);
        }
        self.data.truncate(nbytes);
        self.data
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// `(minint, maxint, smallidx, compressed_bytes)` — the pieces the XTC writer
/// emits after the precision field.
type Compressed = ([i32; 3], [i32; 3], i32, Vec<u8>);

/// Compress an interleaved `[x,y,z,…]` nm coordinate vector.
fn compress_coords(coords: &[f64], natoms: usize, precision: f32) -> Result<Compressed> {
    let size = natoms;
    let maxabs = (i32::MAX - 2) as f32;

    let mut ip = vec![0i32; size * DIM];
    let mut minint = [i32::MAX; 3];
    let mut maxint = [i32::MIN; 3];
    // i64 so large coordinate spreads (e.g. `large_diff.xtc`) don't overflow.
    let mut mindiff = i64::MAX;
    let mut oldlint = [0i32; 3];
    for a in 0..size {
        let mut lint = [0i32; 3];
        for d in 0..DIM {
            let val = coords[a * DIM + d] as f32;
            let lf = if val >= 0.0 {
                val * precision + 0.5
            } else {
                val * precision - 0.5
            };
            if lf.abs() > maxabs {
                return Err(unsupported("XTC: coordinate too large for compression"));
            }
            let li = lf as i32;
            lint[d] = li;
            if li < minint[d] {
                minint[d] = li;
            }
            if li > maxint[d] {
                maxint[d] = li;
            }
            ip[a * DIM + d] = li;
        }
        if a > 0 {
            let diff = (oldlint[0] as i64 - lint[0] as i64).abs()
                + (oldlint[1] as i64 - lint[1] as i64).abs()
                + (oldlint[2] as i64 - lint[2] as i64).abs();
            if diff < mindiff {
                mindiff = diff;
            }
        }
        oldlint = lint;
    }

    let mut sizeint = [0u32; 3];
    for d in 0..DIM {
        sizeint[d] = (maxint[d] as i64 - minint[d] as i64) as u32 + 1;
    }
    let mut bitsizeint = [0i32; 3];
    let bitsize: i32;
    if (sizeint[0] | sizeint[1] | sizeint[2]) > 0x00ff_ffff {
        bitsizeint[0] = sizeofint(sizeint[0]);
        bitsizeint[1] = sizeofint(sizeint[1]);
        bitsizeint[2] = sizeofint(sizeint[2]);
        bitsize = 0;
    } else {
        bitsize = sizeofints(DIM, &sizeint);
    }

    let mut smallidx = FIRSTIDX;
    while smallidx < LASTIDX && (MAGICINTS[(smallidx + 1) as usize] as i64) < mindiff {
        smallidx += 1;
    }
    let smallidx_initial = smallidx;
    let maxidx = LASTIDX.min(smallidx + 8);
    let minidx = maxidx - 8;
    let mut smaller = MAGICINTS[FIRSTIDX.max(smallidx - 1) as usize] / 2;
    let mut smallnum = MAGICINTS[smallidx as usize] / 2;
    let mut sizesmall = [MAGICINTS[smallidx as usize] as u32; 3];
    let larger = MAGICINTS[maxidx as usize];

    let mut bw = BitWriter::new();
    let mut prevcoord = [0i32; 3];
    let mut prevrun = -1i32;
    let mut i = 0usize;
    while i < size {
        let tc = i * DIM;
        // i64 deltas avoid overflow on large-spread coordinates.
        let adelta = |a: i32, b: i32| (a as i64 - b as i64).abs();
        let mut is_small = 0i32;
        let larger = larger as i64;
        let smallnum_i = smallnum as i64;
        let mut is_smaller = if smallidx < maxidx
            && i >= 1
            && adelta(ip[tc], prevcoord[0]) < larger
            && adelta(ip[tc + 1], prevcoord[1]) < larger
            && adelta(ip[tc + 2], prevcoord[2]) < larger
        {
            1
        } else if smallidx > minidx {
            -1
        } else {
            0
        };

        if i + 1 < size
            && adelta(ip[tc], ip[tc + 3]) < smallnum_i
            && adelta(ip[tc + 1], ip[tc + 4]) < smallnum_i
            && adelta(ip[tc + 2], ip[tc + 5]) < smallnum_i
        {
            ip.swap(tc, tc + 3);
            ip.swap(tc + 1, tc + 4);
            ip.swap(tc + 2, tc + 5);
            is_small = 1;
        }

        let tmpcoord = [
            (ip[tc] as i64 - minint[0] as i64) as u32,
            (ip[tc + 1] as i64 - minint[1] as i64) as u32,
            (ip[tc + 2] as i64 - minint[2] as i64) as u32,
        ];
        if bitsize == 0 {
            bw.sendbits(bitsizeint[0], tmpcoord[0]);
            bw.sendbits(bitsizeint[1], tmpcoord[1]);
            bw.sendbits(bitsizeint[2], tmpcoord[2]);
        } else {
            bw.sendints(DIM, bitsize, &sizeint, &tmpcoord);
        }
        prevcoord = [ip[tc], ip[tc + 1], ip[tc + 2]];
        let mut tc2 = tc + DIM;
        i += 1;

        let mut run = 0i32;
        let mut runbuf = [0u32; 30];
        if is_small == 0 && is_smaller == -1 {
            is_smaller = 0;
        }
        while is_small != 0 && run < 8 * 3 {
            if is_smaller == -1 {
                let d0 = (ip[tc2] - prevcoord[0]) as i64;
                let d1 = (ip[tc2 + 1] - prevcoord[1]) as i64;
                let d2 = (ip[tc2 + 2] - prevcoord[2]) as i64;
                if d0 * d0 + d1 * d1 + d2 * d2 >= (smaller as i64) * (smaller as i64) {
                    is_smaller = 0;
                }
            }
            runbuf[run as usize] = (ip[tc2] as i64 - prevcoord[0] as i64 + smallnum as i64) as u32;
            run += 1;
            runbuf[run as usize] =
                (ip[tc2 + 1] as i64 - prevcoord[1] as i64 + smallnum as i64) as u32;
            run += 1;
            runbuf[run as usize] =
                (ip[tc2 + 2] as i64 - prevcoord[2] as i64 + smallnum as i64) as u32;
            run += 1;
            prevcoord = [ip[tc2], ip[tc2 + 1], ip[tc2 + 2]];
            i += 1;
            tc2 += DIM;
            is_small = 0;
            if i < size
                && adelta(ip[tc2], prevcoord[0]) < smallnum_i
                && adelta(ip[tc2 + 1], prevcoord[1]) < smallnum_i
                && adelta(ip[tc2 + 2], prevcoord[2]) < smallnum_i
            {
                is_small = 1;
            }
        }

        if run != prevrun || is_smaller != 0 {
            prevrun = run;
            bw.sendbits(1, 1);
            bw.sendbits(5, (run + is_smaller + 1) as u32);
        } else {
            bw.sendbits(1, 0);
        }
        let mut k = 0usize;
        while (k as i32) < run {
            let chunk = [runbuf[k], runbuf[k + 1], runbuf[k + 2]];
            bw.sendints(DIM, smallidx, &sizesmall, &chunk);
            k += DIM;
        }
        if is_smaller != 0 {
            smallidx += is_smaller;
            if is_smaller > 0 {
                smaller = smallnum;
                smallnum = MAGICINTS[smallidx as usize] / 2;
            } else {
                smallnum = smaller;
                smaller = MAGICINTS[(smallidx - 1) as usize] / 2;
            }
            sizesmall = [MAGICINTS[smallidx as usize] as u32; 3];
        }
    }

    Ok((minint, maxint, smallidx_initial, bw.finish()))
}

// ---------------------------------------------------------------------------
// Frame parsing
// ---------------------------------------------------------------------------

/// Parsed XTC per-frame header (everything up to the coordinate block).
#[derive(Debug, Clone)]
struct XtcHeader {
    natoms: usize,
    step: i32,
    time: f32,
    boxv: [f32; 9],
    /// The 2023 magic stores the compressed byte count as a 64-bit integer
    /// (GROMACS' fix for frames whose compressed size exceeds 2 GiB).
    wide_nbytes: bool,
}

/// Read the compressed-buffer byte count (`i64` for the 2023 format, else
/// `i32`).
fn read_nbytes<R: Read>(r: &mut R, wide: bool) -> Result<usize> {
    let nbytes = if wide {
        xdr::read_i64(r)?
    } else {
        xdr::read_i32(r)? as i64
    };
    if nbytes < 0 {
        return Err(invalid(format!("negative XTC nbytes {nbytes}")));
    }
    Ok(nbytes as usize)
}

fn read_header<R: Read>(r: &mut R) -> Result<XtcHeader> {
    let magic = xdr::read_i32(r)?;
    if magic != XTC_MAGIC && magic != XTC_MAGIC_2023 {
        return Err(invalid(format!(
            "bad XTC magic {magic} (expected {XTC_MAGIC} or {XTC_MAGIC_2023})"
        )));
    }
    let natoms = xdr::read_i32(r)?;
    if natoms <= 0 {
        return Err(invalid(format!("invalid XTC natoms {natoms}")));
    }
    let step = xdr::read_i32(r)?;
    let time = xdr::read_f32(r)?;
    let mut boxv = [0f32; 9];
    for b in boxv.iter_mut() {
        *b = xdr::read_f32(r)?;
    }
    Ok(XtcHeader {
        natoms: natoms as usize,
        step,
        time,
        boxv,
        wide_nbytes: magic == XTC_MAGIC_2023,
    })
}

/// Read and decompress the coordinate block. Returns `(coords, precision)`.
fn read_coords<R: Read>(r: &mut R, natoms: usize, wide_nbytes: bool) -> Result<(Vec<f64>, f32)> {
    let size = xdr::read_i32(r)?;
    if size as usize != natoms {
        return Err(invalid(format!(
            "XTC coord size {size} disagrees with header natoms {natoms}"
        )));
    }
    if natoms <= 9 {
        let mut out = Vec::with_capacity(natoms * DIM);
        for _ in 0..natoms * DIM {
            out.push(xdr::read_f32(r)? as f64);
        }
        return Ok((out, 0.0));
    }
    let precision = xdr::read_f32(r)?;
    let mut minint = [0i32; 3];
    for m in minint.iter_mut() {
        *m = xdr::read_i32(r)?;
    }
    let mut maxint = [0i32; 3];
    for m in maxint.iter_mut() {
        *m = xdr::read_i32(r)?;
    }
    let smallidx = xdr::read_i32(r)?;
    let nbytes = read_nbytes(r, wide_nbytes)?;
    let buf = xdr::read_opaque(r, nbytes)?;
    let coords = decompress_coords(&buf, natoms, precision, minint, maxint, smallidx)?;
    Ok((coords, precision))
}

fn insert_float_col(block: &mut Block, key: &str, vals: Vec<F>) -> Result<()> {
    let n = vals.len();
    let arr = Array1::from_vec(vals)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(invalid)?;
    block.insert(key, arr).map_err(invalid)
}

fn build_simbox(boxv: &[f32; 9]) -> Option<Result<SimBox>> {
    if boxv.iter().all(|&v| v == 0.0) {
        return None;
    }
    // Row-stored vectors → column-stored H: H[r][c] = box[c*3 + r].
    let h = Array2::from_shape_fn((DIM, DIM), |(r, c)| boxv[c * DIM + r] as F);
    let origin = array![0.0 as F, 0.0, 0.0];
    Some(SimBox::new(h, origin, [true; 3]).map_err(|e| invalid(format!("XTC box: {e:?}"))))
}

fn parse_frame_at<R: BufRead + Seek>(r: &mut R, offset: u64) -> Result<Frame> {
    r.seek(SeekFrom::Start(offset))?;
    let hdr = read_header(r)?;
    let natoms = hdr.natoms;
    let (coords, precision) = read_coords(r, natoms, hdr.wide_nbytes)?;

    let mut atoms = Block::new();
    let id_arr = Array1::from_iter(1..=natoms as I)
        .into_shape_with_order(IxDyn(&[natoms]))
        .map_err(invalid)?;
    atoms.insert("id", id_arr).map_err(invalid)?;
    let mut x = Vec::with_capacity(natoms);
    let mut y = Vec::with_capacity(natoms);
    let mut z = Vec::with_capacity(natoms);
    for a in 0..natoms {
        x.push(coords[a * DIM] as F);
        y.push(coords[a * DIM + 1] as F);
        z.push(coords[a * DIM + 2] as F);
    }
    insert_float_col(&mut atoms, "x", x)?;
    insert_float_col(&mut atoms, "y", y)?;
    insert_float_col(&mut atoms, "z", z)?;

    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.simbox = match build_simbox(&hdr.boxv) {
        Some(res) => Some(res?),
        None => None,
    };
    frame.meta.insert("step", hdr.step);
    frame.meta.insert("time", hdr.time);
    if precision != 0.0 {
        frame.meta.insert("precision", precision);
    }
    Ok(frame)
}

/// Scan the file, recording each frame's start byte offset.
fn scan_offsets<R: BufRead + Seek>(r: &mut R) -> Result<Vec<u64>> {
    let end = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    let mut offsets = Vec::new();
    loop {
        let pos = r.stream_position()?;
        if pos >= end {
            break;
        }
        let hdr = match read_header(r) {
            Ok(h) => h,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                return Err(invalid(format!(
                    "XTC scan: header for frame {} at offset {pos} (end {end}): {e}",
                    offsets.len()
                )));
            }
        };
        let size = xdr::read_i32(r)?;
        if size as usize != hdr.natoms {
            return Err(invalid("XTC coord size mismatch during scan"));
        }
        if hdr.natoms <= 9 {
            r.seek(SeekFrom::Current((hdr.natoms * DIM * 4) as i64))?;
        } else {
            // skip precision(4) + minint(12) + maxint(12) + smallidx(4) = 32
            r.seek(SeekFrom::Current(32))?;
            let nbytes = read_nbytes(r, hdr.wide_nbytes)?;
            r.seek(SeekFrom::Current(xdr::pad4(nbytes) as i64))?;
        }
        offsets.push(pos);
    }
    Ok(offsets)
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// XTC trajectory reader with O(1) random access (offsets indexed lazily).
pub struct XtcReader<R: BufRead + Seek> {
    reader: R,
    offsets: OnceCell<Vec<u64>>,
    cursor: usize,
}

impl<R: BufRead + Seek> XtcReader<R> {
    /// Wrap `reader`; index building is deferred.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            offsets: OnceCell::new(),
            cursor: 0,
        }
    }

    fn ensure_index(&mut self) -> Result<()> {
        if self.offsets.get().is_some() {
            return Ok(());
        }
        let offs = scan_offsets(&mut self.reader)?;
        self.offsets
            .set(offs)
            .map_err(|_| std::io::Error::other("failed to set XTC index"))?;
        Ok(())
    }
}

impl<R: BufRead + Seek> Reader for XtcReader<R> {
    type R = R;
    type Frame = Frame;
    fn new(reader: Self::R) -> Self {
        Self::new(reader)
    }
}

impl<R: BufRead + Seek> FrameReader for XtcReader<R> {
    fn read_frame(&mut self) -> Result<Option<Frame>> {
        self.ensure_index()?;
        let cursor = self.cursor;
        let off = match self.offsets.get().and_then(|o| o.get(cursor).copied()) {
            Some(o) => o,
            None => return Ok(None),
        };
        let frame = parse_frame_at(&mut self.reader, off)?;
        self.cursor += 1;
        Ok(Some(frame))
    }
}

impl<R: BufRead + Seek> TrajReader for XtcReader<R> {
    fn build_index(&mut self) -> Result<()> {
        self.ensure_index()
    }

    fn read_step(&mut self, step: usize) -> Result<Option<Frame>> {
        self.ensure_index()?;
        let off = match self.offsets.get().and_then(|o| o.get(step).copied()) {
            Some(o) => o,
            None => return Ok(None),
        };
        parse_frame_at(&mut self.reader, off).map(Some)
    }

    fn len(&mut self) -> Result<usize> {
        self.ensure_index()?;
        Ok(self.offsets.get().expect("index set").len())
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

const DEFAULT_PRECISION: f32 = 1000.0;

fn axis<FA: FrameAccess>(frame: &FA, key: &str) -> Option<Vec<f64>> {
    frame
        .get_float("atoms", key)
        .map(|view| view.iter().copied().collect::<Vec<f64>>())
}

fn write_xtc_frame<W: Write, FA: FrameAccess>(w: &mut W, frame: &FA) -> Result<()> {
    let natoms = frame
        .visit_block("atoms", |a| a.nrows().unwrap_or(0))
        .ok_or_else(|| invalid("XTC write: frame has no atoms block"))?;
    if natoms == 0 {
        return Err(invalid("XTC write: atoms block is empty"));
    }
    let xs = axis(frame, "x").ok_or_else(|| invalid("XTC write: atoms.x missing"))?;
    let ys = axis(frame, "y").ok_or_else(|| invalid("XTC write: atoms.y missing"))?;
    let zs = axis(frame, "z").ok_or_else(|| invalid("XTC write: atoms.z missing"))?;
    if xs.len() != natoms || ys.len() != natoms || zs.len() != natoms {
        return Err(invalid("XTC write: coordinate columns disagree on length"));
    }

    let meta = frame.meta_ref();
    let step = meta
        .get("step")
        .and_then(|value| value.as_i32())
        .unwrap_or(0);
    let time = meta
        .get("time")
        .and_then(|value| value.as_f32())
        .unwrap_or(0.0);
    let precision: f32 = meta
        .get("precision")
        .and_then(|value| value.as_f32())
        .filter(|&p: &f32| p > 0.0)
        .unwrap_or(DEFAULT_PRECISION);

    xdr::write_i32(w, XTC_MAGIC)?;
    xdr::write_i32(w, natoms as i32)?;
    xdr::write_i32(w, step)?;
    xdr::write_f32(w, time)?;
    if let Some(sb) = frame.simbox_ref() {
        let h = sb.h_view().to_owned();
        for i in 0..DIM {
            for j in 0..DIM {
                xdr::write_f32(w, h[(j, i)] as f32)?;
            }
        }
    } else {
        for _ in 0..9 {
            xdr::write_f32(w, 0.0)?;
        }
    }

    xdr::write_i32(w, natoms as i32)?;
    if natoms <= 9 {
        for a in 0..natoms {
            xdr::write_f32(w, xs[a] as f32)?;
            xdr::write_f32(w, ys[a] as f32)?;
            xdr::write_f32(w, zs[a] as f32)?;
        }
        return Ok(());
    }

    let mut coords = Vec::with_capacity(natoms * DIM);
    for a in 0..natoms {
        coords.push(xs[a]);
        coords.push(ys[a]);
        coords.push(zs[a]);
    }
    let (minint, maxint, smallidx, buf) = compress_coords(&coords, natoms, precision)?;
    xdr::write_f32(w, precision)?;
    for v in minint {
        xdr::write_i32(w, v)?;
    }
    for v in maxint {
        xdr::write_i32(w, v)?;
    }
    xdr::write_i32(w, smallidx)?;
    xdr::write_i32(w, buf.len() as i32)?;
    xdr::write_opaque(w, &buf)?;
    Ok(())
}

/// XTC trajectory writer.
pub struct XtcWriter<W: Write> {
    writer: W,
}

impl<W: Write> XtcWriter<W> {
    /// Create a new XTC writer.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> Writer for XtcWriter<W> {
    type W = W;
    type FrameLike = Frame;
    fn new(writer: Self::W) -> Self {
        Self::new(writer)
    }
}

impl<W: Write> FrameWriter for XtcWriter<W> {
    fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        write_xtc_frame(&mut self.writer, frame)
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Read every frame of an XTC file into memory.
pub fn read_xtc<P: AsRef<Path>>(path: P) -> Result<Vec<Frame>> {
    let reader = crate::io::reader::open_seekable(path)?;
    XtcReader::new(reader).read_all()
}

/// Open an XTC file for trajectory-style random access.
pub fn open_xtc<P: AsRef<Path>>(path: P) -> Result<XtcReader<Box<dyn ReadSeek>>> {
    Ok(XtcReader::new(crate::io::reader::open_seekable(path)?))
}

/// Write a slice of frames to an XTC file.
pub fn write_xtc<P: AsRef<Path>, FA: FrameAccess>(path: P, frames: &[FA]) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    for f in frames {
        write_xtc_frame(&mut w, f)?;
    }
    w.flush()
}

// ---------------------------------------------------------------------------
// Tests (pure functions only — no tests-data reads, per CLAUDE.md IO rules)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizeofint_known_values() {
        assert_eq!(sizeofint(1), 1);
        assert_eq!(sizeofint(2), 2);
        assert_eq!(sizeofint(255), 8);
        assert_eq!(sizeofint(256), 9);
    }

    #[test]
    fn sizeofints_matches_reference() {
        // The reference algorithm is byte-aligned-conservative: it accumulates
        // the mixed-radix product byte by byte, so 8*8*8 = 512 spans 2 bytes
        // whose high byte (2) needs 2 bits → 2 + 8 = 10 (not the theoretical 9).
        // Reproducing this exactly is what lets us read real GROMACS streams.
        assert_eq!(sizeofints(3, &[8, 8, 8]), 10);
        assert_eq!(sizeofints(3, &[1, 1, 1]), 1);
        assert_eq!(sizeofints(3, &[256, 256, 256]), 25);
    }

    #[test]
    fn bit_round_trip_single_value() {
        for &(nbits, val) in &[(1u32, 1u32), (5, 19), (8, 200), (13, 5000), (20, 1_000_000)] {
            let mut bw = BitWriter::new();
            bw.sendbits(nbits as i32, val);
            let buf = bw.finish();
            let mut br = BitReader::new(&buf);
            assert_eq!(br.receivebits(nbits as i32).unwrap(), val as i32);
        }
    }

    #[test]
    fn ints_round_trip() {
        let sizes = [901u32, 9001, 90001];
        let nbits = sizeofints(3, &sizes);
        let nums = [123u32, 4567, 89012];
        let mut bw = BitWriter::new();
        bw.sendints(3, nbits, &sizes, &nums);
        let buf = bw.finish();
        let mut br = BitReader::new(&buf);
        let mut got = [0i32; 3];
        br.receiveints(3, nbits, &sizes, &mut got).unwrap();
        assert_eq!(got, [123, 4567, 89012]);
    }

    #[test]
    fn compress_decompress_round_trip() {
        // A handful of atoms (>9 so the compressed path runs), nm coords.
        let coords: Vec<f64> = (0..30)
            .map(|k| (k as f64) * 0.037 - 0.5)
            .collect::<Vec<_>>();
        let natoms = coords.len() / 3;
        let precision = 1000.0f32;
        let (minint, maxint, smallidx, buf) = compress_coords(&coords, natoms, precision).unwrap();
        let back = decompress_coords(&buf, natoms, precision, minint, maxint, smallidx).unwrap();
        assert_eq!(back.len(), coords.len());
        for (a, b) in coords.iter().zip(back.iter()) {
            assert!((a - b).abs() <= 1.0 / precision as f64 + 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn build_simbox_zero_is_none() {
        assert!(build_simbox(&[0.0; 9]).is_none());
        assert!(build_simbox(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]).is_some());
    }
}
