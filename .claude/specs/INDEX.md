# molrs — Spec Index

One row per spec produced by `/mol:spec`. Newest on top.

| Date | Slug | Status | Owner crate(s) | Summary |
|---|---|---|---|---|
| 2026-07-12 | chem-perceive-04-equivalence | approved | molcrafts-molrs | `Perceive::find_equivalence_classes`：实现 antechamber 的 path-score 算法（式 I），**不是**自同构轨道——score 只依赖 (L, ΣZ) 故对路径顺序盲视，轨道是其真子集，用 WL/Morgan 必然偏离 oracle。修构象依赖/对称性破缺电荷（20/37 分子，最大 0.036 e）。链 4/13。 |
| 2026-07-12 | chem-perceive-05-atd-typifier | approved | molcrafts-molrs | 把 ATD/WILDATOM 规则引擎从 am1bcc.rs 抽出为参数化 `AtdTypifier`（一个引擎 + N 张生成表），BCC/ABCG2/GAS 原子类型跑绿。硬依赖 03——pyridine/imidazole 的 `missing BCC correction 17|25|10` 其实是原子类型 bug 的伪装。链 5/13。 |
| 2026-07-12 | chem-perceive-06-gaff-types | approved | molcrafts-molrs | 同一 AtdTypifier 再挂 GAFF/GAFF2/AMBER/SYBYL 四张 .DEF（引擎零改动，纯加表），各自 37/37 对 `antechamber -at <x>`。**显式推翻 2026-06-19 的「GAFF 只走 AmberTools」决定**（情况已变：ATD 引擎已存在且验证过），用户已授权。链 6/13。 |
| 2026-07-12 | chem-perceive-07-charge-trait | approved | molcrafts-molrs | `ChargeModel` trait 托起 Mulliken(QM直通) / BCC+ABCG2(QM+键增量) / Gasteiger(无QM) 的 2×2 泛化证明；BCC 换成纯函数 push API `correct(&mol,&am1)`；删掉三个实现全是假的 `AM1ChargeBackend` 拉模型 trait、删掉会让 molrs 偏离 antechamber 的 `normalize_total_charge`；终结 `keys::TYPE` 污染。链 7/13。 |
| 2026-07-12 | chem-perceive-08-gasteiger | approved | molcrafts-molrs | `GasteigerModel` 对齐 `antechamber -c gas`（37/37）：χ=a+b·q+c·q²，**`d` 列是 χ⁺ 分母不是四次项系数**（H 特例 20.02 ≠ a+b+c=12.85），阻尼收敛循环（非固定 6 次）。它**不需要 QM 输入**——这是 ChargeModel 抽象没有偷偷假设 QM 基电荷的证明。链 8/13。 |
| 2026-07-12 | chem-perceive-09-gaff-params | approved | molcrafts-molrs | gaff.dat(7312)/gaff2.dat(13181) → 提交的 Rust 静态表（含通配符行），用 GAFF/GAFF2 原子类型填 ForceField；精确命中即可，缺参报错（回退在 11）。硬 Task：实测 13k 行生成表的编译时间影响，过重就换 phf/二分布局而**不是**退回文本解析。链 9/13。 |
| 2026-07-12 | chem-perceive-10-parmchk-tables | approved | molcrafts-molrs | `gaff_equiv.json`(6159)/`gaff_empirical.json`(87) → 提交的 .rs，删掉 estimate/tables.rs 里 `include_str!` + 运行期 `serde_json.expect()`（FF 路径最后一处文本解析）。同时落地 parmchk2 frcmod 的 37 分子 RED oracle，供 11 转绿。链 10/13。 |
| 2026-07-12 | chem-perceive-11-param-estimate | approved | molcrafts-molrs | 在既有 `ParameterEstimator`/`ParameterInterpolator` 上原生实现 parmchk2 式缺参估计（精确→通配符→等价类替代→经验公式），**删掉 `Frcmod::parse_str`/`write_string`，molrs 不再依赖任何外部 frcmod 文件**。数据侧（gaff_equiv/gaff_empirical）本就在仓库里。链 11/13。 |
| 2026-07-12 | chem-perceive-12-cxx-bridge | approved | molrs-cxxapi | cxx bridge 改成返回 `Result<Vec<f64>>`——现在它声明 `-> Vec<f64>` 且函数体 `.expect()`，于是缺 BCC 参数这类**用户化学错误直接 abort 进程**而非抛可捕获的 C++ 异常；同时去掉 normalize 参数、加 parameter-set 选择器打通 ABCG2（现全文 0 命中）。跨仓：需 Atomiverse 配套改。链 12/13。 |
| 2026-07-12 | chem-perceive-13-python-bind | approved | molrs-python, molcrafts-molrs | 把 `Perceive`/`AtdTypifier`/`BccModel`/`MullikenModel`/`GasteigerModel` 暴露到 molrs-python，迁 `molrs::chem`→`molrs::perceive` 并删掉 01 的 compat alias。**Python 首次可达原生 AM1-BCC**（今天 molpy 只有 `antechamber -c bcc`），这是与 antechamber 对账的前提。链 13/13。 |
| 2026-07-12 | chem-perceive-14-all-tables | approved | molcrafts-molrs | 收尾「所有参数表 .rs 化」：mmff94/mmff94s/oplsaa + gen3d 的两个 fragment 库也转成 typed Rust 表，删空 `molrs/data/` 与全部 `include_str!`。**存在理由本身是纠错**——早前「编译时间会爆炸」的排除理由已被实测推翻（15,474 行 = +1071 KB / 0.37 s；而这些数据本来就以原始文本形式躺在二进制里，共 3974 KB）。纯表示层变更，数值零改动。链 14/14。 |
| 2026-07-11 | graph-sink-01-extract | done | molcrafts-molrs | Induced subgraph + multi-source `extract_ball` / leaf `extract_subgraph` (O(ball) regenerate path). Engine primitives for molpy region extract. Chain graph-sink 1/4. |
| 2026-07-11 | graph-sink-02-copy-merge | done | molcrafts-molrs | Lock copy = handle-preserving Clone; `merge` returns old→new node map; no identity-merge. Chain graph-sink 2/4. |
| 2026-07-11 | graph-sink-03-python-bind | done | molrs-python | PyO3: extract result type, merge→dict[int,int], copy contract tests. Blocks molpy wire-up. Chain graph-sink 3/4. |
| 2026-07-11 | graph-sink-04-hydrogens-coords | done | molcrafts-molrs | `add_hydrogens` places X–H xyz when heavy has coords (port molpy capping geometry). Chain graph-sink 4/4. |
| 2026-07-06 | net-streaming | partial (networking deferred) | molcrafts-molrs | Serialization foundation SHIPPED 0.7.0 as the `serde` + `stream` features (Frame/Block/Column/SimBox serde + MessagePack/JSON `frame_to_bytes`, WASM-clean). WebSocket networking + bidirectional control (`net` feature: tokio runtime, `FrameServer`, `ControlCommand`, crossbeam bridge) DEFERRED to a later release. See net-streaming.md STATUS. |
| 2026-07-05 | region-support-01-graph-hash | in-flight | molcrafts-molrs, molrs-python | Isomorphism-invariant structural graph hash (WL/Morgan) + canonical node order + `is_isomorphic` on `MolGraph` (AA+CG), exposed to Python. The dedup key for molpy incremental-typification (`AffectedRegion` hashes by it → identical polymer junctions retype once). Greenfield; reuses topo/neighbor kernels. See molpy notes/incremental-typification-design.md. |
| 2026-07-05 | region-support-02-reaction-touched | in-flight | molcrafts-molrs, molrs-python | `Reaction.apply` returns the touched atom handles (bond-forming endpoints + added atoms + deleted-atom surviving neighbors + prop-set atoms) so molpy can extract the retype-safe region. Small, on reaction-smarts-02. Return type None→list[int]. |
| 2026-07-05 | reaction-smarts-01-python-matcher | in-flight | molcrafts-molrs, molrs-python | Expose the existing atom-map-aware SMARTS engine (`molrs::SmartsPattern`, `core/chem/smarts/`, already parses `[C:1]`) to Python as `SmartsPattern` with map-keyed matches; add `PyAtomistic` graph-edit conveniences (`remove_atom`/`remove_bond`/`set_bond_order`/`copy`). Pure binding, no algorithm change. Consumed by molpy `crosslink-*`. |
| 2026-07-05 | reaction-smarts-02-smirks-applier | in-flight | molcrafts-molrs, molrs-python | Daylight reaction-SMARTS (SMIRKS) engine: parse `LHS>>RHS`, compile the atom-map diff to a `Transform` (Daylight SMIRKS semantics: pairwise maps preserved, unmapped LHS deleted, unmapped RHS added, bond diff→form/break/order), apply to one occurrence via existing core edits + `generate_topology`. Expose `Reaction` to Python. Greenfield, on top of 01. Permissive reaction SMARTS (no strict mode). Depends on 01. |

<!--
Status values:
  draft      — spec written, not yet implemented
  in-flight  — /mol:impl started against this spec
  shipped    — merged to master
  superseded — replaced by a later spec (link it in Summary)
-->
