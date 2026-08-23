//! The ABI contract of the molrs FFI layer.
//!
//! All cross-extension interop (molrs-python ↔ molpack, cxxapi consumers)
//! exchanges raw `molrs_ffi` handles, so both sides must embed a
//! layout-identical molrs core. The project rule is **minor-line = ABI
//! version**: every downstream shares one molrs minor line, and the layout of
//! every FFI-crossing type is frozen within a minor (see the layout snapshot
//! test below, which turns that rule into a CI gate). This module is the
//! single source of the versioned capsule names and the handshake line —
//! binders and consumers must take them from here, never hard-code them.

use std::ffi::{CStr, CString};
use std::sync::OnceLock;

/// The ABI line of this build: `major.minor` of the statically linked
/// `molcrafts-molrs` core (e.g. `"0.14"`).
///
/// Two extensions may exchange `molrs_ffi` handles if and only if their ABI
/// lines are equal. Patch versions may differ — layout is frozen within a
/// minor line.
pub fn abi_line() -> &'static str {
    static LINE: OnceLock<String> = OnceLock::new();
    LINE.get_or_init(|| {
        let mut parts = molrs::VERSION.split('.');
        let major = parts.next().expect("molrs::VERSION has a major component");
        let minor = parts.next().expect("molrs::VERSION has a minor component");
        format!("{major}.{minor}")
    })
}

fn versioned_name(prefix: &str) -> CString {
    CString::new(format!("{prefix}/{}", abi_line())).expect("capsule name has no NUL")
}

/// PyCapsule name for [`crate::FrameRef`] handles: `molrs.FrameRef/<abi_line>`.
///
/// The name carries the ABI line so a cross-minor handle exchange fails the
/// capsule name check (a clean `ValueError`) instead of dereferencing a
/// possibly drifted layout. Builds on the 0.13 line and earlier used the
/// unversioned name `molrs.FrameRef`.
pub fn frameref_capsule_name() -> &'static CStr {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| versioned_name("molrs.FrameRef"))
}

/// PyCapsule name for `ForceFieldRef` handles: `molrs.ForceFieldRef/<abi_line>`.
///
/// Not gated on the `ff` feature — it is only a string, and a core-only build
/// may still need to *recognize* (and reject) a force-field capsule.
pub fn forcefield_capsule_name() -> &'static CStr {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| versioned_name("molrs.ForceFieldRef"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_line_is_major_minor_of_molrs_version() {
        let expected: Vec<&str> = molrs::VERSION.split('.').take(2).collect();
        assert_eq!(abi_line(), expected.join("."));
        assert_eq!(abi_line().split('.').count(), 2);
    }

    #[test]
    fn capsule_names_carry_the_abi_line_and_differ() {
        let frame = frameref_capsule_name().to_str().expect("utf8");
        let ff = forcefield_capsule_name().to_str().expect("utf8");
        assert_eq!(frame, format!("molrs.FrameRef/{}", abi_line()));
        assert_eq!(ff, format!("molrs.ForceFieldRef/{}", abi_line()));
        assert_ne!(frame, ff);
    }
}

/// Layout snapshot gate — the enforcement of "minor-line = ABI version".
///
/// Every type a consumer dereferences through a `molrs_ffi` handle is listed
/// here with its size, alignment, and (where fields are nameable) offsets.
/// The report must equal the committed `layout.snapshot`, whose first line
/// declares the ABI line it was taken on.
///
/// If this test fails:
/// - **Same minor, layout changed** — the change breaks every already-shipped
///   wheel on this line. Revert it, or bump the minor version and refresh the
///   snapshot (first line included).
/// - **Toolchain update alone changed the report** — that is still an ABI
///   event for a pointer-crossing bridge: wheels built by the old rustc no
///   longer match. Bump the minor (and refresh) or pin the toolchain.
/// - **Minor was bumped** — refresh the snapshot: update the first line to the
///   new ABI line and paste the new report (printed in the failure message).
#[cfg(all(test, target_pointer_width = "64"))]
mod layout_snapshot {
    use std::mem::{align_of, offset_of, size_of};

    use crate::{BlockHandle, BlockRef, FrameId, FrameRef, SharedStore, Store};

    fn report() -> String {
        let mut out = format!("abi {}\n", super::abi_line());
        macro_rules! row {
            ($ty:ty) => {
                out.push_str(&format!(
                    "{} size={} align={}\n",
                    stringify!($ty),
                    size_of::<$ty>(),
                    align_of::<$ty>()
                ));
            };
        }
        macro_rules! field {
            ($ty:ty, $field:ident) => {
                out.push_str(&format!(
                    "{}.{} offset={}\n",
                    stringify!($ty),
                    stringify!($field),
                    offset_of!($ty, $field)
                ));
            };
        }

        // molrs-ffi handle types (this crate — private fields nameable here).
        row!(FrameId);
        row!(BlockHandle);
        field!(BlockHandle, frame_id);
        field!(BlockHandle, key);
        field!(BlockHandle, version);
        row!(FrameRef);
        field!(FrameRef, id);
        field!(FrameRef, store);
        row!(BlockRef);
        field!(BlockRef, handle);
        field!(BlockRef, store);
        row!(SharedStore);
        row!(Store);

        // molrs core types lent across the boundary (public fields only).
        row!(molrs::Frame);
        field!(molrs::Frame, meta);
        field!(molrs::Frame, simbox);
        row!(molrs::Block);
        row!(molrs::SimBox);

        // Force-field handle (only size/align — `ForceFieldRef.ff` is
        // module-private, and `Rc<T>` is pointer-sized regardless).
        #[cfg(feature = "ff")]
        {
            row!(crate::ForceFieldRef);
            row!(molrs::ff::ForceField);
        }

        out
    }

    #[test]
    #[cfg(feature = "ff")] // the committed snapshot is taken with --features ff
    fn layout_matches_committed_snapshot() {
        let expected = include_str!("layout.snapshot");
        let actual = report();
        assert_eq!(
            actual, expected,
            "\nFFI layout drifted from src/layout.snapshot.\n\
             Within a minor line this is FORBIDDEN (it breaks every shipped \
             wheel on the line): revert the layout change, or bump the minor \
             version and refresh the snapshot.\n\
             Actual report:\n{actual}"
        );
    }
}
