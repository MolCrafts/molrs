# molrs — Spec Index

One row per spec produced by `/mol:spec`. Newest on top.

| Date | Slug | Status | Owner crate(s) | Summary |
|---|---|---|---|---|
| 2026-07-15 | gaff-electrostatics | approved | molcrafts-molrs | **HIGH**。`chem-perceive-15` 的整体验收抓到：**GAFF 的力场根本没有库仑 style**，整条链对全部 37 个分子（含净电荷 −1 的乙酸根、+1 的甲铵/咪唑鎓）**静默丢掉静电能**。这是咖啡因那 150 kcal/mol 的洞，在另一个力场里。它藏得住是因为**每一段都被测过，组合从来没有被跑到能量**。证据一直在树里：`coul: [0,0,1/1.2]` —— 一个为不存在的库仑项声明的 1-4 缩放因子。 |
| 2026-07-15 | test-subset-assertions | approved | molcrafts-molrs | `chem-perceive-15` 抓到 4 处手写的 fixture 子集，而 `e_caffeine` / `e_big` **确实带离域氮却两个名单都不在**——从来没有任何东西断言过 MMFF94s 会改变咖啡因的能量。这是 `["e_ethane"]` 反模式的复发：**子集恰好排除了唯一可能失败的分子**。 |
| 2026-07-15 | vacuous-green-tests | approved | molcrafts-molrs | `chem-perceive-15` 抓到 `readers/opls.rs:36` 在输入缺失时打印 "skipping" 然后 `return`——**在 CI 里它什么都不断言，却计入覆盖率**。一个在 CI 里跳过自己的测试，就是一个从不运行的测试。 |
| 2026-07-14 | chem-perceive-15-final-acceptance | approved | molcrafts-molrs, molrs-python | **整体验收**：这条链跑了 16 个 spec，每个只验证自己那一块，没有任何一个验证过整体。把五条"只此一份"的架构承诺（一个参数地点/一个感知层/一个插值 seam/一条 MMFF 路径/忽略 `tp` 的构造器不是 Style）变成自己不能豁免自己的门禁；端到端对着**外部** oracle（antechamber 37 分子 + RDKit MMFF 11 分子）跑通全链路；Python 与 Rust **逐位**一致。门禁全部是**反向**的（断言某物不存在）。铁律两条：**一道从没红过的门禁 = 没有门禁**；**验收里不许修任何东西**——它找到的每个缺陷都必须停下来另立 spec。 |
| 2026-07-06 | net-streaming | partial (networking deferred) | molcrafts-molrs | Serialization foundation SHIPPED 0.7.0 as the `serde` + `stream` features (Frame/Block/Column/SimBox serde + MessagePack/JSON `frame_to_bytes`, WASM-clean). WebSocket networking + bidirectional control (`net` feature: tokio runtime, `FrameServer`, `ControlCommand`, crossbeam bridge) DEFERRED to a later release. See net-streaming.md STATUS. |

<!--
Status values:
  draft      — spec written, not yet implemented
  in-flight  — /mol:impl started against this spec
  shipped    — merged to master
  superseded — replaced by a later spec (link it in Summary)
-->
