//! Control commands sent from streaming clients back to the simulation.
//!
//! This module is WASM-clean: it depends only on `serde` (and optional
//! MessagePack / JSON codecs at the call site). No tokio or socket types.

use serde::{Deserialize, Serialize};

/// A bidirectional control message from a visualization client to the server.
///
/// Wire format is externally tagged by `"type"` with snake_case variant names,
/// so a browser client can send `{"type":"pause"}` or MessagePack with the same
/// shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlCommand {
    /// Pause the producing simulation loop (interpretation is caller-defined).
    Pause,
    /// Resume after [`Pause`].
    Resume,
    /// Request a maximum stream rate in frames per second.
    SetFrameRate {
        /// Desired frames per second.
        hz: f64,
    },
    /// Restrict subsequent frames to a subset of atom indices.
    SetSubset {
        /// Atom ids to keep, in caller-defined order (preserved on the wire).
        atom_ids: Vec<u32>,
    },
    /// Ask the producer to emit a full key frame on the next send.
    RequestKeyFrame,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants() -> Vec<ControlCommand> {
        vec![
            ControlCommand::Pause,
            ControlCommand::Resume,
            ControlCommand::SetFrameRate { hz: 42.0 },
            ControlCommand::SetSubset {
                atom_ids: vec![1, 3, 5],
            },
            ControlCommand::RequestKeyFrame,
        ]
    }

    #[test]
    fn control_command_json_roundtrip() {
        for cmd in all_variants() {
            let bytes = serde_json::to_vec(&cmd).expect("json encode");
            let back: ControlCommand = serde_json::from_slice(&bytes).expect("json decode");
            assert_eq!(back, cmd);
        }
    }

    #[test]
    fn control_command_rmp_roundtrip() {
        for cmd in all_variants() {
            let bytes = rmp_serde::to_vec_named(&cmd).expect("rmp encode");
            let back: ControlCommand = rmp_serde::from_slice(&bytes).expect("rmp decode");
            assert_eq!(back, cmd);
        }
    }

    #[test]
    fn control_command_json_type_tags() {
        let pause = serde_json::to_value(ControlCommand::Pause).unwrap();
        assert_eq!(pause["type"], "pause");
        let resume = serde_json::to_value(ControlCommand::Resume).unwrap();
        assert_eq!(resume["type"], "resume");
        let rate = serde_json::to_value(ControlCommand::SetFrameRate { hz: 10.0 }).unwrap();
        assert_eq!(rate["type"], "set_frame_rate");
        let subset = serde_json::to_value(ControlCommand::SetSubset {
            atom_ids: vec![2, 4],
        })
        .unwrap();
        assert_eq!(subset["type"], "set_subset");
        assert_eq!(subset["atom_ids"], serde_json::json!([2, 4]));
        let key = serde_json::to_value(ControlCommand::RequestKeyFrame).unwrap();
        assert_eq!(key["type"], "request_key_frame");
    }

    #[test]
    fn set_subset_preserves_atom_id_order() {
        let cmd = ControlCommand::SetSubset {
            atom_ids: vec![9, 1, 7, 3],
        };
        let bytes = rmp_serde::to_vec_named(&cmd).unwrap();
        let back: ControlCommand = rmp_serde::from_slice(&bytes).unwrap();
        match back {
            ControlCommand::SetSubset { atom_ids } => assert_eq!(atom_ids, vec![9, 1, 7, 3]),
            other => panic!("unexpected variant: {other:?}"),
        }
    }
}
