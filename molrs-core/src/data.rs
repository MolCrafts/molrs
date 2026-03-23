//! Embedded data files shipped with molrs-core.
//!
//! The MMFF94 parameter XML is compiled into the binary so callers never need
//! to locate a file on disk.

/// MMFF94 parameter XML (embedded at compile time).
pub const MMFF94_XML: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/mmff94.xml"));

/// MMFF94s parameter XML (embedded at compile time).
pub const MMFF94S_XML: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/mmff94s.xml"));
