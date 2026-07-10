---
slug: net-streaming
criteria:
  - id: ac-001
    summary: raw_bytes 对每个数值 dtype 返回连续字节切片；String 返回 None
    type: code
    pass_when: |
      Column::raw_bytes() 在 Float(3) 上返回 Some(Vec<u8>) 长度为 3*8=24 字节；
      在 U8(5) 上为 5 字节；在 String(3) 上为 None。多维列的总字节数 = nrows * sizeof(T) * product(shape[1..])。
    status: pending
  - id: ac-002
    summary: WireFrame 序列化往返无损保留结构
    type: runtime
    pass_when: |
      一个完整 Frame（atoms block 含 x/y/z 浮点列 + serial 整型列 + type u8 列，bonds block 含 i/j uint + order u8 列，SimBox，meta）经 frame_to_wire_bytes() 和反序列化后，block 名称、列数、nrows、dtype 标签及数值与原始值一致（浮点在 f64::EPSILON 容差内）。
    status: pending
  - id: ac-003
    summary: ControlCommand 所有变体经 rmp-serde 往返无损
    type: code
    pass_when: |
      ControlCommand::Pause、Resume、SetFrameRate{hz:42.0}、SetSubset{atom_ids:[1,3,5]}、RequestKeyFrame 经 rmp-serde 和 JSON 序列化后反序列化得到匹配的变体，字段值不变。
    status: pending
  - id: ac-004
    summary: FrameServer 绑定端口并接受客户端连接
    type: runtime
    pass_when: |
      FrameServer::bind("127.0.0.1:0") 返回 Ok；tokio-tungstenite 客户端连接后 client_count == 1。
    status: pending
  - id: ac-005
    summary: FrameServer 向已连接的客户端发送帧
    type: runtime
    pass_when: |
      客户端连接后，server.send(frame) 返回 Ok；客户端收到可反序列化的消息，其内容与发送的原始 Frame 匹配。
    status: pending
  - id: ac-006
    summary: FrameServer 从客户端接收 ControlCommand
    type: runtime
    pass_when: |
      客户端发送序列化 ControlCommand::Pause 后，server.recv_command() 在超时内返回 Some(Pause)。
    status: pending
  - id: ac-007
    summary: 有界通道满时丢弃最旧帧，仿真不阻塞
    type: runtime
    pass_when: |
      ServerConfig{buffer_size:1} 下连续发送 3 帧，仿真线程不阻塞；客户端最终收到的帧为第 3 帧（最新）。
    status: pending
  - id: ac-008
    summary: ser 和 message 模块在 wasm32 下编译通过
    type: code
    pass_when: |
      cargo check --target wasm32-unknown-unknown --features net 成功退出，未引用 tokio/tungstenite 符号。
    status: pending
  - id: ac-009
    summary: 质量闸：fmt + clippy + check + test 全部通过
    type: runtime
    pass_when: |
      cargo fmt --all --check、cargo clippy --features net -- -D warnings、cargo check --features net、cargo test --features net 全部 exit 0。
    status: pending
---

> **Deferred (2026-07-08).** The serialization foundation (`serde` + `stream`
> features) shipped in 0.7.0 and meets the lossless-round-trip goal (ac-002) via
> direct serde; the `WireFrame`/`Column::raw_bytes()` design (ac-001) was
> superseded by it. The `net`-feature networking criteria (ac-003…ac-009) remain
> pending and are deferred to a later release. See `net-streaming.md` STATUS.

# Acceptance — net-streaming

## AC-001 — raw_bytes 对每个数值 dtype 返回连续字节切片；String 返回 None

`Column::raw_bytes()` 是新公开的辅助方法，供 `ser.rs` 中的 `WireColumn` 构建使用。它利用 ndarray 的 `as_slice_memory_order()` 获取底层 `&[T]` 并转换为 `&[u8]`。对于 String 变体，每个元素是独立堆分配的，无法获得连续字节表示，因此返回 `None`。

**测试位置**：`molrs/src/core/store/block/column.rs` 内联 `#[cfg(test)]`。

## AC-002 — WireFrame 序列化往返无损保留结构

验证 `ser.rs` 中间类型正确地将 `Frame` 转换为可序列化的 `WireFrame`，再通过 rmp-serde 和 JSON 往返后，结构信息完全保留。这是模拟代码→客户端数据通路的核心保障。

## AC-003 — ControlCommand 所有变体经 rmp-serde 往返无损

所有 5 个命令变体通过 rmp-serde 和 JSON 路径往返，确保 molvis（或任意客户端）发送的命令在 Rust 端能正确解析。

## AC-004 — FrameServer 绑定端口并接受客户端连接

基本连接生命周期：服务器绑定到随机端口，客户端通过 WebSocket 连接，服务器报告有 1 个客户端。

## AC-005 — FrameServer 向已连接的客户端发送帧

确保 `send()` 非阻塞，数据通过 WebSocket 传输，客户端正确收到帧数据。

## AC-006 — FrameServer 从客户端接收 ControlCommand

确保 `recv_command()` 从客户端接收反序列化的命令。这是双向控制的基础。

## AC-007 — 有界通道满时丢弃最旧帧，仿真不阻塞

验证 bounded channel 的 backpressure 行为：仿真线程不受慢客户端的影响。`buffer_size=1` 配置下，未消费的帧被丢弃，只有最新的帧保留。

## AC-008 — ser 和 message 模块在 wasm32 下编译通过

确保 `ser.rs` 和 `message.rs` 不引用 tokio 或特定于平台的网络 API，从而可以在 wasm 目标中复用。`bridge.rs` 和 `server.rs` 被 `#[cfg]` 排除。

## AC-009 — 质量闸：fmt + clippy + check + test 全部通过

项目的标准质量闸应用于新的 `net` 功能：格式正确、无 clippy 警告、编译通过、测试全部通过。
