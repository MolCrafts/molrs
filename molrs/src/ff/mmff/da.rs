//! MMFF's `DA` column (`"D"` / `"A"` / `"-"`), as the small integer code that
//! travels on a `mmff_vdw` parameter row.
//!
//! One encoding with three writers and one reader: the molrs-native XML reader
//! and the embedded table transcribe the letter, and the `mmff_vdw` kernel
//! decodes it when it decides whether a pair is a hydrogen bond. It lives with
//! the rest of MMFF's vocabulary rather than inside the kernel, because a
//! reader reaching into `ff::potential` for it made `ff::forcefield` and
//! `ff::potential` name each other.

/// Neither hydrogen-bond donor nor acceptor (`DA` = `"-"`).
pub const DA_NEITHER: u8 = 0;
/// Hydrogen-bond **donor** (`DA` = `"D"`) — polar hydrogen.
pub const DA_DONOR: u8 = 1;
/// Hydrogen-bond **acceptor** (`DA` = `"A"`).
pub const DA_ACCEPTOR: u8 = 2;

/// Encode MMFF's `DA` column (`"D"` / `"A"` / `"-"`) as the small integer code
/// stored in a `mmff_vdw` [`PairType`](crate::ff::forcefield::PairType) row's
/// `da` param.
///
/// [`Params`] is a numeric bag, so a reader transcribes the letter here and
/// [`vdw_combining`] decodes it — one encoding, owned by its consumer. Anything
/// unrecognised is [`DA_NEITHER`], which is also MMFF's own default for the
/// 80-odd non-hydrogen-bonding types.
pub(crate) fn encode_da(raw: &str) -> f64 {
    match raw {
        "D" => DA_DONOR,
        "A" => DA_ACCEPTOR,
        _ => DA_NEITHER,
    }
    .into()
}

/// [`encode_da`] for the compiled table, whose `da` column is the letter's ASCII
/// byte ([`MmffVdW::da`](crate::ff::params::mmff::MmffVdW::da), as RDKit stores
/// it) rather than a string. Same three codes, same default — one mapping, two
/// spellings of the letter.
pub(crate) fn encode_da_byte(code: u8) -> f64 {
    match code {
        b'D' => DA_DONOR,
        b'A' => DA_ACCEPTOR,
        _ => DA_NEITHER,
    }
    .into()
}
