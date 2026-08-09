---
title: net-streaming — WebSocket Frame streaming + bidirectional control for molrs
status: done
created: 2026-07-06
closed: 2026-08-04
---

# net-streaming — WebSocket Frame streaming + bidirectional control

> **STATUS (2026-07-08, rescoped for 0.7.0).** The **WASM-clean serialization
> foundation shipped in 0.7.0** as two standalone, always-available features —
> `serde` (`Serialize`/`Deserialize` for `Frame`/`Block`/`Column`/`SimBox`, in
> `src/serialize.rs`) and `stream` (MessagePack/JSON Frame wire encoding,
> `frame_to_bytes`/`bytes_to_frame` in `src/stream/mod.rs`). These deliver the
> lossless-round-trip goal (acceptance ac-002) through direct serde rather than
> the `WireFrame`/`Column::raw_bytes()` design sketched below. The **WebSocket
> networking + bidirectional-control layer** (the `net` feature: tokio runtime,
> `Publisher`, `ControlCommand`, crossbeam bridge — acceptance ac-003…ac-009)
> **shipped** under feature `net` (2026-08-04). Wire encoding reuses `stream`;
> `Publisher` + `ControlCommand` live in `molrs::stream`.

## Summary

Add a `net` feature-gated module to molrs that enables real-time Frame streaming over WebSocket. Simulation code (MD loops, trajectory generators) can broadcast `Frame` snapshots to browser-based visualization clients (like molvis) while the simulation runs. Clients may send control commands (pause, resume, set framerate, request keyframe, set atom subset) back to the simulation via the same WebSocket connection. The networking layer runs on a background tokio runtime behind a bounded crossbeam channel, so the simulation loop remains synchronous and never blocks on network writes. The serialization layer (`molrs::stream::ser`) is WASM-clean — it compiles under `wasm32-unknown-unknown` with no tokio dependency — so the same wire types can be reused in browser-side or FFI contexts.

## Design

### Module structure

```
molrs/src/net/
  mod.rs           -- re-exports; `pub mod ser;` always; bridge/server behind `#[cfg(not(target_arch = "wasm32"))]`
  ser.rs           -- WASM-clean: WireFrame / WireBlock / WireColumn / WireSimBox types + frame_to_wire_bytes()
  bridge.rs        -- #[cfg(not(wasm32))] sync→async channel bridge, StreamHandle
  server.rs        -- #[cfg(not(wasm32))] WebSocket accept loop, per-client fan-out, graceful shutdown
  message.rs       -- ControlCommand enum + serde; WASM-clean
```

### Wire types (ser.rs)

`Column` wraps `Arc<ColumnHolder<T>>` with a `ManuallyDrop` guard — it cannot derive `Serialize` directly. The net module defines intermediate wire types:

- `WireColumn` — carries the column's `dtype` tag, `shape` vec, and raw data bytes (`Vec<u8>`). Built by extracting contiguous byte slices via a new `Column::raw_bytes()` helper.
- `WireBlock` — maps block name to `WireColumn` vec.
- `WireSimBox` — serializable copy of `SimBox` fields (origin, basis, dimensions, pbc flags).
- `WireFrame` — block map + optional SimBox + metadata hashmap.

`frame_to_wire_bytes(frame, format) -> Vec<u8>` converts a Frame into rmp-serde (or JSON) bytes, ready for WebSocket transmission.

### Column::raw_bytes()

New public helper on `Column`:

```rust
impl Column {
    pub fn raw_bytes(&self) -> Option<Vec<u8>>;
}
```

For `Float`/`Int`/`UInt`/`U8`/`Bool` variants, returns a copy of the contiguous backing memory as `Vec<u8>`, using `as_slice_memory_order()` and bit-casting the element slice. For `String`, returns `None` (variable-length, non-contiguous). The copy is necessary because ndarray's memory order may not be simple; callers get an owned buffer they can send over the wire without lifetime constraints.

### Control commands (message.rs)

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlCommand {
    Pause,
    Resume,
    SetFrameRate { hz: f64 },
    SetSubset { atom_ids: Vec<u32> },
    RequestKeyFrame,
}
```

The client sends commands as JSON or MessagePack payloads with a `"type"` tag. The `recv_command()` API on `Publisher` polls for incoming commands from any client, returning `None` after a configurable timeout.

### Publisher API

```rust
#[derive(Clone)]
pub struct Publisher { /* Arc<Inner> */ }

impl Publisher {
    pub fn bind(addr: impl Into<String>) -> io::Result<Self>;
    pub fn bind_with(addr: impl Into<String>, config: PublisherConfig) -> io::Result<Self>;
    pub fn send(&self, frame: &Frame) -> Result<(), SendError>;
    pub async fn send_async(&self, frame: &Frame) -> Result<(), SendError>;
    pub fn client_count(&self) -> usize;
    pub async fn recv_command(&self) -> Option<ControlCommand>;
    pub fn shutdown(self) -> impl Future<Output = ()>;
}

pub struct PublisherConfig {
    pub format: MessageFormat,          // MessagePack | Json
    pub buffer_size: usize,             // crossbeam channel capacity; default 4
    pub max_frame_rate: f64,            // reserved, no-op in v1
}
```

### Threading model

Simulation thread (sync) → `crossbeam::bounded(buffer_size)` → Dedicated std::thread running a tokio runtime → `tokio::sync::broadcast` → Per-client `tokio::sync::mpsc` → `tokio_tungstenite::WebSocket` writes. The bounded crossbeam channel **drops the oldest frame** when full, so the simulation never blocks on a slow network client.

On `Publisher::Drop`, the background thread is joined and the tokio runtime is shut down gracefully.

### Feature flag

```toml
net = ["dep:tokio", "dep:tokio-tungstenite", "dep:rmp-serde", "dep:crossbeam-channel", "dep:bytes"]
```

**NOT** in `full`. The `full` feature enables all sub-system modules but excludes `net` to keep its heavy networking dependencies (tokio multi-thread runtime, tungstenite) opt-in.

### WASM strategy

`ser.rs` and `message.rs` avoid tokio / tokio-tungstenite / crossbeam-channel imports — they depend only on `serde`, `rmp-serde`, `serde_json`, and `bytes`. `bridge.rs` and `server.rs` are gated behind `#[cfg(not(target_arch = "wasm32"))]`. This lets wasm-pack consumers reuse the serialization layer without pulling in an entire async runtime.

## Files to create or modify

- `molrs/Cargo.toml` — add `net` feature + 5 optional deps (tokio, tokio-tungstenite, rmp-serde, crossbeam-channel, bytes) + `[[test]]` entry for `tests/net.rs`
- `molrs/src/lib.rs` — `#[cfg(feature = "net")] pub mod net;`
- `molrs/src/core/store/block/column.rs` — add `Column::raw_bytes() -> Option<Vec<u8>>` helper
- `molrs/src/net/mod.rs` (new) — module root, re-exports, feature-gated sub-module declarations
- `molrs/src/net/ser.rs` (new) — WireFrame/WireBlock/WireColumn/WireSimBox types, MessageFormat enum, `frame_to_wire_bytes()`
- `molrs/src/net/message.rs` (new) — ControlCommand enum + serde wire format
- `molrs/src/net/bridge.rs` (new) — sync→async bridge (crossbeam → tokio broadcast); `#[cfg(not(target_arch = "wasm32"))]`
- `molrs/src/net/server.rs` (new) — WebSocket accept loop, per-client fan-out, graceful shutdown; `#[cfg(not(target_arch = "wasm32"))]`
- `molrs/tests/net.rs` (new) — integration tests for net feature

## Tasks

- [ ] T1: Add `net` feature + 5 optional deps (tokio, tokio-tungstenite, rmp-serde, crossbeam-channel, bytes) to `molrs/Cargo.toml`; gate `pub mod net` in `molrs/src/lib.rs` with `#[cfg(feature = "net")]`; add `[[test]]` entry for `tests/net.rs` with `required-features = ["net"]`
- [ ] T2: Write failing inline `#[cfg(test)]` tests for `Column::raw_bytes()` — each numeric type returns `Some(Vec<u8>)` with correct byte-length; Strings return `None`
- [ ] T3: Implement `Column::raw_bytes() -> Option<Vec<u8>>` in `molrs/src/core/store/block/column.rs` using `as_slice_memory_order()` + bit-cast per variant
- [ ] T4: Write failing tests for wire-format serialization: build a Frame with varied Block types (float x/y/z, int serial, u8 type, bool flags, SimBox, meta), serialize round-trip via rmp-serde + via JSON, assert block names / nrows / dtypes / numeric values preserved within epsilon
- [ ] T5: Implement `WireFrame`/`WireBlock`/`WireColumn`/`WireSimBox` types, `MessageFormat` enum, and `frame_to_wire_bytes()` in `molrs/src/net/ser.rs`
- [ ] T6: Write failing tests for `ControlCommand`: every variant (Pause, Resume, SetFrameRate, SetSubset, RequestKeyFrame) round-trips through rmp-serde + JSON without data loss; SetSubset preserves atom_ids order
- [ ] T7: Implement `ControlCommand` enum + serde (tagged, `rename_all = "snake_case"`) in `molrs/src/net/message.rs`
- [ ] T8: Write failing integration tests for `Publisher` in `tests/net.rs`: bind to random port, client connects via tokio-tungstenite, `client_count == 1`; server `send(frame)` → client receives deserializable WireFrame; client sends ControlCommand → `server.recv_command()` returns it; bounded channel drops oldest frame when buffer is full
- [ ] T9: Implement `Publisher` (bind/bind_with/send/send_async/client_count/recv_command/shutdown) + sync→async bridge using crossbeam→tokio broadcast + WebSocket accept loop with per-client mpsc fan-out in `bridge.rs` + `server.rs`
- [ ] T10: Run full quality gate: `cargo fmt --all --check`, `cargo clippy --features net -- -D warnings`, `cargo check --features net`, `cargo test --features net`

## Testing strategy

- **Column::raw_bytes() unit tests** — for each dtype (Float, Int, UInt, U8, Bool, String): verify length matches `nrows * sizeof(T) * product(remaining shape dims)`. Multi-dimensional column where inner dims > 1 gives flat `Vec<u8>` of total byte count. String returns None.
- **Wire-format round-trip** — build a full Frame with at least 3 blocks ("atoms" with x/y/z float columns + serial int + type u8; "bonds" with uint i/j + u8 order; optional SimBox + meta key-value map). Serialize via `frame_to_wire_bytes()` with both MessagePack and JSON. Deserialize raw bytes into `WireFrame`. Assert block names, column count per block, nrows per column, dtype tags, and numeric values match (within f64 epsilon for floats).
- **ControlCommand round-trip** — all 5 variants: Pause, Resume, SetFrameRate {42.0}, SetSubset {[1,3,5]}, RequestKeyFrame. Serialize→deserialize→match.
- **Publisher in-process integration** — bind to `127.0.0.1:0`, spawn a tokio side-task that connects as a WebSocket client, send a Frame from the server side, assert client receives a matching message. From the client side, send `ControlCommand::Pause`, assert `recv_command()` yields `Some(Pause)`.
- **Bounded channel drop** — `PublisherConfig { buffer_size: 1 }`, send 3 frames without draining the client side; verify the simulation thread never blocks and the frame buffer drops oldest (the client eventually receives only the latest).
- **Graceful shutdown** — `Publisher::shutdown()` completes within 5 seconds; the background thread joins; no resources leaked.
- **WASM compilation** — `cargo check --target wasm32-unknown-unknown --features net` succeeds (verifies ser.rs and message.rs are WASM-clean).
- **Quality gate** — `cargo fmt --all --check`, `clippy -D warnings`, `check`, `test --features net` all exit 0.

## UI verification  <!-- optional, non-binding -->

This spec does not itself implement a frontend, but the protocol and server it produces are directly consumable by molvis:

- A dev-script MD loop calling `server.send(&frame)` at each time step should render live in a molvis browser tab connected to `ws://localhost:<port>`.
- Sending `{"type": "pause"}` from the browser console should appear on the Rust side via `recv_command()`.
- A second browser tab connecting should also receive frames (broadcast fan-out).

These checks are verified ad-hoc by the developer running `/mol:web` on the `live_stream` example; they are **non-binding** and never gate `done`.

## Out of scope

- **Delta / frame compression** (zstd, delta encoding between frames) — deferred
- **TLS / WSS** — the server listens on plain ws://; TLS is a follow-up
- **Connection recovery / reconnection logic** — client must reconnect on disconnect
- **HTTP/SSE fallback** — WebSocket-only in v1
- **`net` in `full` feature** — intentionally opt-in due to heavy dependencies
- **Schema caching** (send column metadata only on first frame, raw data only thereafter) — deferred; v1 sends self-describing messages
- **Frame rate throttling** — `max_frame_rate` config field is declared but not enforced; no-op in v1
- **Python / WASM client libraries** for consuming the stream — the wire types in ser.rs are reusable but no binding is provided
- **Multiple server instances** — single Publisher per process
