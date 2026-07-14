# molrs — Spec Index

One row per spec produced by `/mol:spec`. Newest on top.

| Date | Slug | Status | Owner crate(s) | Summary |
|---|---|---|---|---|
| 2026-07-14 | chem-perceive-15-final-acceptance | approved | molcrafts-molrs, molrs-python | **整体验收**：这条链跑了 16 个 spec，每个只验证自己那一块，没有任何一个验证过整体。把五条"只此一份"的架构承诺（一个参数地点/一个感知层/一个插值 seam/一条 MMFF 路径/忽略 `tp` 的构造器不是 Style）变成自己不能豁免自己的门禁；端到端对着**外部** oracle（antechamber 37 分子 + RDKit MMFF 11 分子）跑通全链路；Python 与 Rust **逐位**一致。门禁全部是**反向**的（断言某物不存在）。铁律两条：**一道从没红过的门禁 = 没有门禁**；**验收里不许修任何东西**——它找到的每个缺陷都必须停下来另立 spec。 |
| 2026-07-06 | net-streaming | partial (networking deferred) | molcrafts-molrs | Serialization foundation SHIPPED 0.7.0 as the `serde` + `stream` features (Frame/Block/Column/SimBox serde + MessagePack/JSON `frame_to_bytes`, WASM-clean). WebSocket networking + bidirectional control (`net` feature: tokio runtime, `FrameServer`, `ControlCommand`, crossbeam bridge) DEFERRED to a later release. See net-streaming.md STATUS. |

<!--
Status values:
  draft      — spec written, not yet implemented
  in-flight  — /mol:impl started against this spec
  shipped    — merged to master
  superseded — replaced by a later spec (link it in Summary)
-->
