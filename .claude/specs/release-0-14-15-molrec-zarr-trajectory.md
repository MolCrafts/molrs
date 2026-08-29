---
slug: release-0-14-15-molrec-zarr-trajectory
title: release-0-14-15-molrec-zarr-trajectory — 一种 frame 形状逐位往返，一条 trajectory 一次一帧地追加
status: in-progress
grilled: true
created: 2026-08-29
depends_on:
  - release-0-14-13-frame-store-naming
  - release-0-14-09-molpy-rebase
  - release-0-14-10-molpy-mirror
---

# release-0-14-15-molrec-zarr-trajectory — 一种形状，一条序列

## Summary

molrec 是记录契约,molrs 是它的 Zarr V3 参考实现,molpy 是消费者;三者今天在两件事上是错的。其一,frame 往返从未被证明:15 个 Column dtype 里 14 个与 `structural_shape` 零覆盖,`../molrec/tests/molrs_adapter.py` 没有被任何被收集的测试 import,两套实现从未被比较过——在这层掩护下藏着三个真缺陷(缺失 `boundary` 在 Python 读作全周期、在 Rust 读作全非周期;Zarr 读路径投机解一个只有 MessagePack 才写的 `{dtype, value}` 信封,于是用户一份普通的同形 JSON 被悄悄改写;写入端从不擦除,重写留下的陈旧节点会被读回)。其二,`trajectory/frames/<i>/` 每步写一个完整 frame 组,10 个原子列约 27 个文件/帧 ⇒ 10⁴ 帧约 15 万个文件,不压缩、不分片,而 `read_frame_from_store(i)` 与 `count_frames_in_store()` 都走 `read_record_store()`——读一帧解整条记录,molrs-wasm 每次 JS 调用付一次、构造器再付一次。本规范交付**一种** frame 形状(每个 dtype 逐位往返)与**一条** Trajectory(一次一帧追加,每帧 O(1),文件数有界,收工后可打包成单文件),并把 molrec 的契约文本与一致性套件、molpy 的薄门面一起带齐。顺带关掉一笔沉默的债:`io::store::zarr::{UnitSystem, Provenance}` 零调用点,而 02 号规范登记的"降为 core preset 消费者 + 把彻底删除记进 notes.md"这条跟进**从未执行**(今天 `notes.md` 里没有该条目)——本规范直接删除它们,并把删除与"兼容性顾虑不成立"的理由补记进 `notes.md`。

## Domain basis

无方程,不含物理推导 —— scientist 明确 N/A。但有两条**维护者已裁定**的域相关约束,写在这里是因为它们决定数据正确性,不是风格:

1. **缺失 `boundary` 属性 ⇒ 全周期(`[true, true, true]`)。** 这是 molrec / molpy `BoxModel` 的既有语义;Rust 侧 `read_simbox`(`molrs/src/io/store/zarr/frame_io.rs:300-311`)今天回落到 `[false, false, false]`。两边都是**故意**的,方向相反 —— 于是同一个 store 在两个实现里是两个不同的物理体系(周期 vs 真空)。Rust 向 molrec 对齐。
2. **dtype 逐位往返是硬性要求。** 下游精度研究(MD fp64-only)禁止任何有损编码:压缩只用无损 gzip,列宽只保留其到达时的宽度(CLAUDE.md § 2026-08-26 identity/宽度条目),读回断言一律 `assert_eq!` / `assert_array_equal`,不用容差。gzip 对未 shuffle 的 f64 坐标只有约 1.05–1.2× ——本规范如实写明这不是尺寸杠杆(杠杆是 byte-shuffle,列为路线图),但它相对今天的**不压缩**仍是净赢。

单位:本规范不引入任何带单位的量;`step` 是无量纲整数,`time` 沿用 `.claude/notes/science.md` 的 fs。

## Design

### 〇、命名:一个对象,两种访问形式(架构审查 🔴#1 的答复)

审查指出 `FrameSequence` / `FrameSequenceWriter` 与既有的 `Trajectory`(`molrs/src/core/store/trajectory.rs:20`)、`trait TrajectoryReader`(`molrs/src/io/reader.rs:187`)以及盘上的 `trajectory/` 组指的是同一个东西。**维护者裁定:名字保留**——这是既定方案里的词汇,"不要替用户发明命名"是本仓的常设裁定。但审查的实质是对的:三个名字并存必须有一条**能判别**的规则,否则下一个实现者要靠猜。规则如下,并且**必须**写进两个类型的模块文档(不是只写在规范里):

> 同一个对象(一条帧序列),三种身份,按**访问形式**命名而不是按内容命名:
> - **`Trajectory`** —— **急切的内存载体**:所有帧都已物化(`frames: Vec<Frame>` 加上可选的 step/time,`core/store/trajectory.rs:18-27`,其模块文档自称 "Backend-agnostic ... logical model")。它不知道 store 存在。
> - **`FrameSequence`** —— **惰性的 store 游标**:只读索引地打开(仅取各段的 `step_index` 与 `offset`),一次读**一**帧;`impl TrajectoryReader`,所以既有的 `FrameIterator` 直接可用。`to_trajectory()` 是从惰性形式到急切形式的**具名**转换,而不是第二种表示。
> - **`FrameSequenceWriter`** —— 它的**流式生产者**:一次追加一帧,`flush()` 提交。
>
> 盘上的 `trajectory/` 组是这条序列的**序列化形式**;三个 Rust 名字都不是它的第二个名字,它们是"内存急切 / store 惰性 / 流式写"三种访问形式。

### 一、目标形状(唯一形状)

```
trajectory/                       group,无属性
  step     i64  [nstep]           总是写——提交标记,最后扩展
  time     f64  [nstep]
  meta/<key>    typed [nstep] (或 [nstep][3|6|9]);属性 molrs_meta_dtype
  box/
    step_index u64  [n_box_updates]   ← 定胞 NVT 只写一条
    vectors    f64  [n_box_updates][3][3]
    origin     f64  [n_box_updates][3]
    boundary   bool [n_box_updates][3]
  <block>/                        声明了就带属性 structural_shape
    step_index u64  [n_updates]
    offset     u64  [n_updates+1] ← CSR 行指针,offset[0] = 0
    <column>        [total_rows][...trailing]
```

在第 i 帧解析 block B:在 `B/step_index` 上二分查找 ≤ i 的最大条目 → 行 j;该帧的行是 `offset[j]..offset[j+1]`;没有 ≤ i 的条目 ⇒ 该 block 在这一步不存在。`trajectory/` 的保留子节点:`step`、`time`、`meta`、`box`;block 内的保留列名:`offset`、`step_index`。

**硬切(决定 1)。** `trajectory/frames/<i>/` 从写入端与读取端一并删除。不留双解码路径:两种形状意味着两条读路径,而本规范的全部价值在于"只有一种形状被证明过"。

**布局只有一个属主(架构审查 🔴#2 的修正)。** 初稿把 `record_io.rs` 写成"原样迁入",这与硬切自相矛盾——`trajectory/` 段会变成无主之地,实现者很可能在 `record_io.rs` 里再写一个 CSR 编码器。更正:`record_io.rs` 是**迁移并重接**,不是原样迁入。具体地:

> `trajectory/` 布局的编码器**只有一个**、解码器**只有一个**,两者都在 `io/zarr/sequence.rs` 里。`record_io.rs` 的 `write_trajectory_section`(`:146`)与 `read_trajectory_section`(`:316`)**删除**;`write_trajectory_file` / `read_trajectory_file` 退化成**薄门**,把工作交给 `FrameSequenceWriter` / `FrameSequence`;`read_frame_from_store`(`:473`)与 `count_frames_in_store`(`:483`)删除。`record_io.rs` 保留的职责只剩:record 级的 meta / frame / system / status / observables 段,以及两扇吃 path 的 `FilesystemStore` 门。

**Ragged / CSR(决定 2)。** 只存 `offset`(前缀和),数量由 `diff(offset)` 得到,绝不同时存两者——两份相同信息必然有一天不一致。

**两条 sizing 规则(决定 3、5)。** 固定尺寸数组(frame/、system/、observables)逐字移植 molrec `chunking.py::plan`(`../molrec/src/molrec/chunking.py:30` `TARGET_CHUNK_BYTES = 512 KiB`、`:33` `SHARD_ABOVE = 4`);增长型数组在**创建时**固定 chunk / shard 尺度——`zarrs::Array::set_shape` 只增长前导轴、并用冻结的 chunk 元数据重建网格,创建后尺度改不了。注意 `plan()` 返回的 shard 横跨整个数组:对固定尺寸正确,对增长型不可用,这正是两条规则的分界。移植后的 `plan` **不返回裸元组**——两个同类型的 `Option<Vec<u64>>` 并排是调用点写反了也编译得过的形状——而返回一个私有具名结构 `ChunkPlan { chunks, shards }`;`ChunkPlan` 是私有模块里私有函数的返回类型,**不出现在任何公开面上**。

增长型的尺度统一按**行**给:每个 block 一个 `R`(rows-per-chunk),其全部列共用,由最宽的列决定 `R = max(1, floor(512 KiB / row_bytes_widest))`。**帧可以跨 chunk**(某帧的 `offset[j]..offset[j+1]` 允许跨越 chunk 边界)——这正是百万原子帧能工作的原因。"每 chunk c 帧"只是常数 N 的特例 `R = c·N`。稠密的每步数组(`step`、`time`、`meta/<key>`)按帧分块:`c = max(1, floor(512 KiB / frame_bytes))`。两者都下钳到 ≥ 1。

**声明式 schema(决定 6)。** 创建时钉死 blocks / columns / dtypes / trailing shape 的**并集**。后来的帧可以只出现其中一个子集——这就是稀疏的表达方式。

- 列的 dtype 与 trailing shape **只有一个来源:帧自己的列**。两个铸造口,都是**派生**的,都不接受手写 dtype:
  - `SequenceSchema::from_frame(&Frame)` —— 单帧便利式,从这一帧的列上抄下来。
  - `SequenceSchema::from_frames(&[Frame])` —— **并集铸造**:跨帧取 blocks / columns 的并集,同名列在两帧里 dtype 或 trailing shape 冲突时当场 `Err`(schema 违例规则提前到铸造时执行)。这是**异构 trajectory 的表达方式**——旧的每帧一组布局天然容得下逐帧不同的 blocks/列,新布局要靠并集容下它,所以 `write_trajectory_file` 用 `from_frames(&trajectory.frames)` 铸造,不用 `from_frame(&frames[0])`。
- `append` 只对 **`SequenceSchema`** 校验(未声明的 block / 列、dtype 变更、trailing shape 变更);**不**跑 `Validator::canonical`——canonical 的 Frame 校验仍是读写门的职责,与今天一致,append 不偷偷加一道语义门。稀疏是**块级**的:帧可以整块省略,但**在场的块必须带齐它声明的全部列**(缺列即 Err 指名——所有列共享一段 CSR 行区间,列级缺席无表示)。
- **step 与 time 的生产者(RED 批暴露的缺口,已裁)**:`Frame` 不携带 step 号与 time,所以 `append(&Frame)` 保持精简——step 自动取上一帧 + 1(从 0 起)、不写 time;显式门 `append_at(frame, step: i64, time: Option<f64>)` 供急切路径与真实 MD 步号使用:step 严格递增否则 Err,time **全有或全无**(首次 append 定死 `time` 数组是否存在,混用即 Err)。`write_trajectory_file` 按 `Trajectory` 自带的 step/time 选门,`Trajectory` 因此逐位往返;布局里的 `time f64 [nstep]` 是**可选子节点**——只在走过 `append_at(..., Some(t))` 的 run 里存在。
- **保留名在构造时就拒绝**,不是等到第一次 append:`SequenceSchema::from_frame` / `from_frames` / `FrameSequenceWriter::create` 见到 block 名为 `step`/`time`/`meta`/`box`、或列名为 `offset`/`step_index`,当场 `Err` 并指名。
- 已知限制,明写:**不支持运行中新增列**;一条跑到一半才决定记录力的 run 需要一个新 store。

**按段稀疏时间索引(决定 7)。** 每个 block 与 box 各带自己的 `step_index`,不变的段只写一次。代价跟着**变化**走而不是跟着**长度**走:恒定拓扑 1 条、反应/GC 拓扑每次变化 1 条、坐标 nstep 条(0..nstep-1,被 gzip 压扁)、定胞 1 条。这让"把拓扑手工拆进 system/"变成可选,并且能表达"有时会变的拓扑"——system/ 与稠密打包都表达不了。

**每步 meta(决定 8)。** `frame.meta` 的每步值落成带类型的 `trajectory/meta/<key>` 数组 `[nstep]`(或 `[nstep][3|6|9]`),属性 `molrs_meta_dtype`;dtype 由数组自己携带,于是精确性是免费的。一个帧**省略已声明的** meta key 是错误,除非该 key 在创建时声明了显式 fill 值 —— **没有隐式 NaN**。frame 段的 meta 仍是原始 JSON。复用 `MetaValue`(`molrs/src/core/store/meta.rs:11`),`dtype()`(`:42`)是稳定标签,20 个变体(含 `F64x6`/`F64x9`)给出全部 trailing shape;**不另立第二套 dtype 词汇**。

**构造与打开的语义(各自一件事,并且失败要响)。**

- `FrameSequenceWriter::create(store, schema)` —— **铸造并钉死** schema:在一个空路径上写出所有零长数组与属性。若该路径下已经存在一条序列(有 `trajectory/step`),返回 `Err` 并指名该路径;**绝不静默覆盖**。
- `FrameSequenceWriter::open(store)` —— **接上**一条已存在的序列:从盘上读回 schema 并校验(blocks / columns / dtypes / trailing shapes / chunk 与 shard 尺度),不一致时 `Err`,消息形如 `sequence schema mismatch at <path>: <what> expected <a>, found <b>`。
- `FrameSequence::open(store)` —— 只读、只取索引。

**提交语义与耐久性契约(决定 4 —— spike 已裁决:分支 A)。** `flush()` 是提交点:它让缓冲的行持久,并**最后**扩展 `step`——`step` 就是提交标记,`nstep` 只在 flush 后可见。spike 实测:尾部 inner chunk 的重写在编解码层是 tail-only 写(已完成的 shard 不动,一次 store 写,读回逐位相等),因此**决定 4 落在分支 A**:flush 落下含不满尾 chunk 的全部缓冲行,任意步都可提交。下表保留为历史记录(**spike resolved: A**);分支 B 不再是待定项,只是当初写下的回退设想:

| | 分支 A(倾向) | 分支 B(回退) |
|---|---|---|
| 前提 | partial encoding 能重写尾部 inner chunk(解码—追加—重编码 + 尾部索引,仍是 tail-only 写) | 尾 chunk 重写不可得 |
| `flush()` 落什么 | 全部缓冲行,含不满的尾 chunk | 只落整 chunk |
| `len()` 数什么 | 最近一次 `flush()` 之后的全部已 append 帧 | 最近一次 `flush()` 落下的**整 chunk 行**所覆盖的帧 |
| 崩溃丢什么 | 上次 `flush()` 之后 append 的帧 | 上次 `flush()` 之后 append 的帧,**外加**仍在内存里的不足 R 行的尾巴 |
| 提交粒度 | 任意步 | chunk 对齐(写进 `flush()` 的 rustdoc) |

**封盘压实(分支 A 的代价与它的收口)。** 分支 A 的每次尾 chunk 重写都会把被取代的那一份留在 shard 文件里当死字节(实测:一个 18 048 B 的 chunk 让 shard 从 146 399 B 长到 164 447 B)。逐帧 flush 会把**活动** shard 撑大到多出约 c 倍 chunk 尺寸;纯追加则一个死字节都不留。收口方式简单且有界:**跨过 shard 边界时,写入端把刚刚封盘的那个 shard 整个重编码一次,作为一次干净的整写**(没有死字节);`close(self)` 用同样的方式压实最后那个未满的 shard。代价:每个 shard 多付一次顺序整写(写入量约 2×,摊薄到整条 run;跨 shard 边界的那次 flush 因此带一次完成 shard 的重写尖峰——这一条与死字节一起写进 `flush()` 的 rustdoc);收益:**静止态的文件里一个死字节都没有,与 flush 频率无关**。活动 shard 在两次封盘之间仍可能带死字节——不藏。

**杠杆一:文件数(决定 13)。** `k`(chunks_per_shard)默认由字节目标推导:`k = max(1, floor(shard_target_bytes / chunk_bytes))`,`shard_target_bytes` 默认 **256 MiB**。旋钮只有两个,**每个都点名它的消费者**(审查 🔴#4:没有消费者的旋钮不生):

- `with_rows_per_chunk` —— 消费者是生命周期单测(ac-015 / ac-017):把 `R` 压到很小,几帧就能跨过 chunk 边界,否则要写满 512 KiB 才测得到。
- `with_chunks_per_shard` —— 消费者是文件数上界测试(ac-023):把 `S` 压小,让一个小 store 也横跨多个 shard,从而真的验到"文件数按字节走"。
- ~~`with_gzip_level`~~ —— **删掉**,没有任何在树内的调用者。gzip 停在库默认档;把"gzip 档位旋钮"移进 Out of scope,和 byte-shuffle 放在一起(它们是同一个尺寸/吞吐议题的两半)。

两个旋钮都遵循本仓唯一的 builder 约定 `with_*(mut self, v) -> Result<Self, E>`(`molrs/src/builder/graphene.rs:83`、`carbon_tube.rs:119`;全仓 `molrs/src` 无任何 `set_*` builder 旋钮)。推导出的默认值写在构造器里、在**被读取的地方**就地注释(PublisherConfig 的写法,`molrs/src/stream/publisher.rs`)。读粒度不动(inner chunk 仍约 512 KiB);shard 索引每次 flush 16 B × k 可忽略(k≈512 ⇒ 约 8 KiB)。**256 MiB 这个默认值是有条件的,而条件已经补上**:它只在写入端经 `PositionalWriteStore`(§三 `store.rs`)落盘时成立——库存 `FilesystemStore` 会把每次尾部写变成 256 MiB 的读+重写(§四 实测 247 ms),那种情况下这个默认值是灾难而不是杠杆。默认值不改、O(1) flush 不改、文件数杠杆不改;改的是**谁来承接那次写**。算例,3000 原子 xyz f64:行 24 B ⇒ R = floor(512 KiB / 24) = **21845** 行(≈7.28 帧——**帧跨 chunk,不向帧取整**;此前写的 21000/532/3724 是修订前"按帧取整"思路的残留,与本节公式矛盾,已按公式更正);chunk = 524 280 B;k = floor(256 MiB / 524 280) = **512**;每 shard R·k = 11 184 640 行 ≈ **3728** 帧 ⇒ 10⁴ 帧 **3** 个 shard 文件;整库 ≈ 元数据地板(每个数组至少 `zarr.json` + 1 chunk,约 50–60 个)+ 几个 shard ≈ **60–70 个文件,对比今天的约 15 万**。文件数按 `total_bytes/S + O(arrays)` 增长,不按 nstep。

**杠杆二:静止态单文件(决定 14)。** `.zarr.zip` 收工打包:运行阶段写目录 store(追加快路径需要就地部分写,zip 做不到——**明确否决**活写单文件);`pack()` 把目录打成一个 zip,**条目用 stored(不压缩)**(chunk 已经 gzip 过,打包只是拼接 + 中央目录),然后删除目录。形状约束三条:

- `pack(path) -> Result<PathBuf>` 是 `io/zarr/pack.rs` 里的**自由函数,吃一个已关闭的 store 路径**,**永远不吃活的写入端句柄**——"边写边打包"就是活写单文件,已被否决;`close(self)` 与 `pack(path)` 的组合是**调用者**的事(CLAUDE.md § Forbid all-in-one façade),不提供 `close_and_pack()`。
- 新依赖 `zip` 按 CLAUDE.md § Feature Flags 的规矩挂在它唯一的使用者上:`filesystem = [..., "dep:zip"]`,于是关掉 `filesystem` 的构建根本不编译它。
- 读路径就是 `zarrs_zip`(spike Q7 实测:stored 条目逐位读出,zarr-python 3.3.0 交叉读一致),**没有自制兜底**——曾经设想的"pack.rs 内最小只读 zip 读取"不再存在,不要为它留位置。`zarrs_zip` 与 `zip` 作为**可选依赖**接进 `filesystem` 档:`filesystem = [..., "dep:zip", "dep:zarrs_zip"]`,store 适配层因此只有 zarrs 的那一套加上 §三 的 `PositionalWriteStore`。

静止态文件数 = 1,与 DCD 齐平;随机访问保留(尾部中央目录 + 按条目 range 读——本地文件、对象存储、以后的 wasm HTTP-range 都成立)。

**分片布局对 HTTP-range / 对象存储友好**(取 shard 索引,再 range 取 inner chunk),wasm 与 S3 读者白拿这个收益——这也是保留 `ShardingIndexLocation::End`(默认)的理由之一:追加 = 一次 `partial_encode_many([(max_data_offset, chunk ++ index)])`,新数据恰好覆盖旧索引区,**在编解码层是一次尾部写**。这句话到 store 为止:库存 `FilesystemStore` 会把它变成全文件读+重写(§四),所以它只有经 `PositionalWriteStore` 落盘时才在磁盘层面成立;而分支 A 的尾 chunk 重写会在活动 shard 里留下被取代的死字节,由封盘压实(见上)收掉。

**不升 schema 版本(决定 10;架构审查 🔴#3 的答复,事实已按 `git show v0.13.2` 更正)。** 审查最初指出"不升版让丢数据的那个方向保持沉默",复审时把 v0.13.2 的实际行为查清楚了,**沉默这个前提是错的**,照实记录:

> 一个 molrs ≤ 0.13.2 的读者遇到本布局会**失败**,报 `trajectory.step length mismatch: expected 0, got N`(v0.13.2 的 `read_trajectory_section` 末尾调 `trajectory.validate()?`,而 `Trajectory::validate` 在 `step.len() != frames.len()` 时报错;新布局给它 `frames = []` 而 `step = [nstep]`)。**只有 nstep == 0 那个角落才是静默返回空。** 代价因此不是"沉默",而是"**响,但话说错了**"——消息指认的原因(step 长度)不是真正的原因(布局换了)。另外 v0.13.2 的 `read_meta` 本来就在 `record_schema_version != 1` 时硬报 `unsupported record_schema_version {version}`,那条**精确**的门已经在船上;`record_schema_version` 按裁定停在 1,因此这条精确路径不会被走到。

裁定不变(决定 10,维护者已裁):不升版。更正后的事实其实让裁定**更**站得住——失败已经是响的,升版买到的只是一句更准的话,而不是从沉默变成报错。同时如实记下:**这条裁定当初是在"会沉默"的前提下做出的**,而升版能以零代价换到那句精确消息;调用方会把更正后的前提呈递维护者,是否日后 supersede 由维护者定,**不由本规范代劳**。四个静默失败象限里唯一免费的那个仍然拿走:**新读者遇到含 `trajectory/frames/` 的 store 时报错** "legacy layout (written by molrs ≤ 0.13); re-write with 0.13"。反方向(旧读者遇到新 store)修不了,因此把它**写响在纸上**:记进 `.claude/notes/notes.md`,并记进 `../molrec/docs/spec/storage.md`(契约文档才是外部读者会看的地方),两处都用上面那段更正后的措辞,不用"静默返回空"。

**跨仓消费者必须同批告知(审查 🔴#3 后半,接受)。** 盘上布局变了,而 `molrs-cxxapi/src/lib.rs:15` 正是 `read_trajectory_file` / `write_trajectory_file` 的消费者,其下游是 Atomiverse 的 checkpoint 链(`cpu::ZarrReader`)。按 13 号的先例,老→新映射与"0.13 写的 store 必须用 0.13 重写"这句话进 `.claude/notes/notes.md` 的跨仓破坏项(带日期)。

### 二、边界裁决(决定 9)

`io/store/zarr/` → `io/zarr/`,`io/store/csv.rs` → `io/csv.rs`;molrec 的 `bindings/` 不动;molpy 得到 `io/zarr.py`。

14 号规范立下的规则是"**边界删了就再也接不回来**",所以删掉 `store/` 这一层必须当场交代:`store/` **不是**一条承重边界。它今天的非 zarr 住户只有 `csv.rs` 一个;它写在 `molrs/src/io/store/mod.rs:1-2` 的区分——"serialization of the store types themselves, as opposed to `io/data` and `io/trajectory`, which read molecular *file formats*"——是一**句话**,不是一个依赖门:没有任何编译单元因为这个目录而可门控,没有任何依赖因为它而被隔离(对比 14 号保住 `ff/potential/kspace` 是因为 FFT 依赖将来要能被 feature 门控出去)。

那句话的新家**只有一个**:`molrs/src/io/mod.rs` 的子模块枚举(它今天已经是这份枚举的所在地,`:1-9`),把原来的 `- [store] — persistence backends` 一条改写成 `- [zarr] / [csv] — serialization of the store types themselves, as opposed to [data] and [trajectory], which read molecular file formats`。`io/zarr/mod.rs` 与 `io/csv.rs` 的头文档**指向**它(`See the module list in [crate::io] for why these live beside the format readers rather than under them.`),**不各抄一份**——三份同义副本会各自漂移,这正是本规范在别处要消灭的东西。13 号的命名裁定不受影响:`zarr` 仍然是**适配层**模块名,在公开面上是死的——`molrs-python/tests/test_record.py:28-30` 的 `_EXEMPT_SUFFIXES` 由 `"/io/store/zarr/"` 改成 `"/io/zarr/"`,豁免跟着模块走,门的语义一字不改。

### 三、模块放置与公开面边界

`molrs/src/io/zarr/`:

- `mod.rs` —— 公开面 + 指向 `io/mod.rs` 的边界说明;spike 的 Q5/Q7 结论写在这里。
- `chunking.rs` —— 私有,`zarr`;自由函数 `plan(shape, itemsize) -> ChunkPlan`,形状镜像 molrec。这是**真正无属主的纯运算**(CLAUDE.md § Prefer 允许的唯一自由函数情形),且有两个调用点(固定尺寸路径与增长型路径的初始尺度),不违反"第二次调用前不外提"。
- `sequence.rs` —— 公开于 `molrs::io::zarr` 之内:`FrameSequenceWriter` + `FrameSequence` + `SequenceSchema`;`trajectory/` 布局的唯一编码器与唯一解码器;每步 typed meta 的编解码**先写在这里**(inline until second use)。其模块文档除 §〇 的判别规则外,还要用一句写清**错误词汇**:`FrameSequence::open` / `frame(i)` 与 `FrameSequenceWriter` 的所有门一律产出 `MolRsError`;`TrajectoryReader` 的三个 trait 方法保持 `io::Result`,`MolRsError` 在边界处转进去(**有损,但是故意的**——trait 是跨后端的公共形状,不该被某个后端的错误类型绑架)。
- `pack.rs` —— `filesystem`,`pack()` 与经 `zarrs_zip` 的 zip 读路径;`zip` / `zarrs_zip` 两个可选依赖只在这里被用到。
- `store.rs` —— `filesystem`;`PositionalWriteStore`,包住 `FilesystemStore` 的写入适配器,把 zarrs 的"部分写"落成**真正的定位写**(`set_partial_values` 以读写方式打开、**不 truncate**、在 offset 处 `write_at`;`set` 仍是整写;read / list 直接转发给被包住的 `FilesystemStore`),并**自己记账**它真正写下的字节数(`bytes_written()`,`AtomicU64` —— store 住在 `Arc<dyn … + Send + Sync>` 后面,循 CLAUDE.md "Cell<f64> is NOT Sync" 的本仓先例;记账只观察,不驱动任何行为,无 caller 可见的 reset)。`supports_set_partial()` 返回 `true`,而且这一次它在**磁盘层面**也是真的。类型文档写一行为什么叫 **Positional** 而不是 Partial:codec 层的 "partial encoding" 与 store 层的 `set_partial_values` 是两种不同的 partial,Positional 专指"落在 offset 上的真定位写"——防止后人把名字"修"回去。只用 `std::fs`,无新依赖。**吃 path 的写入门**(`write_record_file` / `write_trajectory_file`)与 `FrameSequenceWriter` 的 filesystem 路径经它落盘;读门(`read_record_file`)不包——包装只对写有意义。
- `frame_io.rs` —— 迁入并按 §五 泛化。
- `record_io.rs` —— 迁入并按 §一 的属主句重接。
- `error.rs` —— 原样迁入。
- 另加 `molrs/src/io/csv.rs`;`molrs/src/io/store/mod.rs` 删除。

**公开面到此为止。** `FrameSequence` / `FrameSequenceWriter` / `SequenceSchema` **不**在 `molrs::io` 或 crate 根 re-export——它们是适配层类型,住在 `molrs::io::zarr` 里,和 13 号"技术名活在适配层"的裁定一致;`ChunkPlan` 连这一层都不上,它是私有模块里私有函数的返回类型。将来若要给它们一扇吃 path 的 `filesystem` 门(`FrameSequence::open_path`),那是跟进项,不在本规范:今天没有 in-tree 调用者。

**绑定面上发生了什么(明写,免得被当成没变)**:`molrs.Trajectory.read` / `.write` 与 cxxapi 的 `write_frame` / `read_first_frame` **名字全不变**,变的是它们脚下的盘上布局——这是**破坏性**变更,记进 `notes.md`(见上)。`FrameSequence` **本轮不绑到 Python**:没有点名的消费者(molpy 的门面走 record/trajectory 那两扇门),按"第二次使用前不外提"不建这层。**本轮唯一的新消费者是 molrs-wasm 的 `RecordReader`**,并且它的 JS 签名**保持 `&self` 不变**:`FrameSequence` 需要可变借用,所以 `RecordReader` 内部持 `RefCell<FrameSequence>`,每次调用走 `try_borrow_mut()`,借用冲突(JS 侧重入)返回一个说明性的 `JsError` —— **绝不 panic**(`architecture-rules.md:39`:wasm 导出的可失败路径不许 panic)。

`FrameSequenceWriter` 与 `FrameSequence` 今天在任何仓里都不存在(干净的名字,不含 molrec 词,过 13 号的门);两者的模块文档必须写上 §〇 那条判别规则。

**生命周期(照抄 `FrameIndexBuilder` 的形,`molrs/src/io/streaming.rs:60`,其生命周期文档在 `:47-59`)**:feed / drain / finish(消费式)、文档化的生命周期、**没有 Drop**。`FrameSequenceWriter::close(self)` 消费自身;**Drop 不 flush** —— 一个科学写入端在 Drop 里吞掉 IO 错误就是无声的数据丢失。本仓唯一在 Drop 里做事的例子是 `stream/publisher.rs` 的 join 线程,引它作反例正好说明这是约定而非疏忽。

### 四、zarrs 0.23.13(钉死版本)—— 已核实的机制与 spike

`molrs/Cargo.toml:113` 声明 `zarrs = "0.23"`,`Cargo.lock:2059-2060` 锁在 **0.23.13**:

- `Array::set_shape` 增长前导轴并从冻结的 chunk 元数据重建网格 ⇒ shard 尺度创建时固定。
- shard/chunk 尺度必须整除,两者都是 `NonZeroU64`。
- chunk 尺度非零的**零长数组合法** ⇒ 所有数组以 `[0, ...]` 创建,store 从第一次 append 之前就是一条合法的空记录。
- `store_array_subset_opt` 会分解成逐 shard 的 `store_chunk_subset_opt` ⇒ 不需要手写 chunk 算术:`set_shape` 之后每个数组一次 subset 写。
- ⚠ `experimental_partial_encoding` **默认 false**,而非 `_opt` 方法各自新建 `CodecOptions::default()`,`set_codec_options` 到不了那里。因此**每一次写都必须显式**穿进 `global_config().codec_options().with_experimental_partial_encoding(true)`;绝不可改 `global_config_mut()`(库里的进程级 RwLock)。搞错了就是每次 flush O(shard) 且**无声**——256 MiB shard 下这是灾难,所以 spike Q6 把"每次 flush 写入字节 ≈ chunk + index"做成断言,不是做成注释。
- gzip 而非 zstd:zstd/blosc 在 C-linked 的 `zarr-codecs` 档里,过不了 wasm32(`molrs/Cargo.toml:83-89` 的两档注释就是这么写的),zstd 写出的 store molrs-wasm 读不了。
- 撕裂写关不严:`FilesystemStore::set` 是 truncate + `write_all_at`,没有 temp+rename。顺序(数据 → 元数据 → **step 最后**)让写与写之间的崩溃不可见;落在 `step` / `zarr.json` 那次小写内部的崩溃仍可见。`AtomicWriteStorageAdapter` 的 `supports_set_partial()` 返回 false,会直接杀死追加快路径,因此**不用它**。发布这个顺序、写明残余窗口、并把顺序做成可测的属性。
- ⚠ **本规范初稿把写放大的守卫放高了一层,spike 当场证伪。** `zarrs_filesystem` 0.3.12 的 `FilesystemStore::set_partial_many` 转给 `zarrs_storage::store_set_partial_many`,而后者**把整个值读进内存、打补丁、再整个 `set` 回去**(truncate + 全量重写)——同时 `supports_set_partial()` 仍报 `true`(编解码层正因为这个 `true` 才走部分写路径),`FilesystemStoreOptions` 里没有开关可以退出。实测:一次 16 B 的尾部写在 256 MiB 上要付一次全文件读+重写,**247 ms**。于是"一次 store 写"这句话在**编解码层是真的、在库存 `FilesystemStore` 上是假的**;而 Q6 若只在编解码层数字节,对这件事**结构性失明**——它数的是编解码器交出去的字节,不是磁盘吞下的字节。修法不是放弃 256 MiB,而是补上缺的那一层:`store.rs` 的 `PositionalWriteStore`(§三)。它的 `bytes_written()` 就是 Q6 的 **store 层那一半**,与编解码层那一半一起断言。

**spike(任务 1)已执行,裁决如下,证据在它写下的模块文档里**:Q1(追加只长一个 chunk)、Q2(`set_shape` / 网格冻结)、Q3(零长创建)、Q4(partial-encoding 穿过 `_opt`)、Q6(编解码层写放大)五钉全绿;**Q5 = 分支 A**——尾 inner chunk 可被重写为 tail-only 写,已完成的 shard 不动,一次 store 写,读回逐位相等;**Q7 = `zarrs_zip` 0.5.2**——它经 `Method::Store` 快路径逐位读出 stored 条目的 zip,且 zarr-python 3.3.0 交叉读同一个 zip 亦逐位相等,**因此不需要任何自制的兜底 zip 读路径**。spike 同时暴露出一条规范级缺陷(§四 写放大守卫放高了一层),已由上一条的 `PositionalWriteStore` 修正。

### 五、feature 门控修复

`record_io.rs` 的 `write_record_store`(`:78`)、`write_meta`(`:121`)、`write_json_group`(`:133`)、`write_observables`(`:185`),以及 `frame_io.rs` 里每一个写入函数,都挂着 `#[cfg(feature = "filesystem")]`,却**不碰** `FilesystemStore` —— 它们只用 `ReadableWritableListableStorage` 和核心 builder。真正需要 `filesystem` 的只有 `FilesystemStore::new` 的两处调用点及其 `Path` 签名的转发门——即**四扇吃 path 的门**:`write_record_file` / `read_record_file`(两处 `FilesystemStore::new`)加 `write_trajectory_file` / `read_trajectory_file`(吃 `Path`、转发前两者,签名依赖 `filesystem` 门下的导入)。由于 `filesystem = ["zarr", "zarr-codecs", "zarrs/filesystem"]` 会拖进 C-linked 的 zstd/blosc,**今天的 wasm 根本写不了 store**。把吃 store 的写入端一律改门到 `zarr`;`filesystem` 留在四扇吃 path 的门 + `pack.rs` + `store.rs` + 新增的 `dep:zip`。(实施注:模块声明处已有 `#[cfg(feature = "zarr")]`,模块内的 `zarr` 门是冗余重述,为 ac-003 的字面断言保留——simplify 不得删。)`zarrs/sharding` 已经在基础 `zarr` 档里。写前擦除是全新表面,`WritableStorageTraits` 属 zarrs 核心 ⇒ 只需 `zarr`。

### 六、发布次序(架构审查 🔴#7,接受)

molrs 先发,molpy 后跟(CLAUDE.md § Release before molpy 是本仓的 agent 铁律)。落到本规范:**P0–P3 全部在 molrs / molrec 内完成;P4 的 molpy 任务只在 `release-0-14-09-molpy-rebase` 产出的分支上执行,且必须在它所转发的 molrs 表面按 § Release before molpy 打过 tag、发布之后**;molpy 侧只做门面,能力留在 molrs(`release-0-14-10-molpy-mirror` 的 sink-to-molrs 治理)。这条依赖关系已写进 frontmatter 的 `depends_on`(09 / 10),现在也写在纸上。决定 11(单规范)与决定 10(不升版)两条裁定本身记进 `.claude/notes/notes.md`,免得下一个人重新辩论。

### 七、Reuse decision

- **generalize** `write_typed_array`(`frame_io.rs:98`)—— 全仓唯一的泛型数组写入器;给它 `ChunkPlan`,所有固定尺寸数组都走它,并把逐字孪生的 `write_float_array`(`frame_io.rs:88`)、`write_f64_array`(`record_io.rs:492`)、`write_i64_array`(`record_io.rs:506`)折进去。
- **generalize** `write_column`(`frame_io.rs:34`)—— 把 15 臂 match 劈出 `dtype_of(&Column) -> (DataType, FillValue)`,append 路径与 frame 路径共用**一张** dtype 表。它留在适配层做私有自由函数而不是挂到 `Column` 上:返回类型是 zarrs 的 `DataType`,core 不能认识 zarr(架构规则:`core` 不依赖任何其他模块)。
- **generalize** `read_column`(`frame_io.rs:124`)—— 参数化到 `&ArraySubset`(今天恒为整数组);`FrameSequence` 用同一套 15 臂分派读 `[offset[j]..offset[j+1]]`。
- **generalize** `DType`(`core/store/block/dtype.rs:17`)—— 增加 `DType::itemsize() -> Option<usize>`(String 返回 None,照 `Column::raw_bytes` 的先例,`block/column.rs:428`)。**两个调用点**:`chunking::plan` 的 `row_bytes`(固定尺寸路径)与 `SequenceSchema` 推导 `R` 时的 `row_bytes_widest`(增长型路径)。itemsize 从此有主,`chunking.rs` 里不放第二张尺寸表。
- **generalize** `TrajectoryReader`(`io/reader.rs:187`)—— 去掉 `Reader` 超 trait(`Reader`,`reader.rs:25`,要求 `type R: BufRead` 与 `fn new(Self::R)`,Zarr 背后的序列没有 `BufRead` 源,这就是它必须去掉的原因)。已核实泛型消费者只碰 `read_step` / `len`(`molrs-python/src/io/mod.rs:229-345` 的十个 `traj_*` 助手、`reader.rs:171` 的 `FrameIterator`),**但 trait 本身不得瘦身**:`build_index` 在五个 Python 类上是公开的(`molrs-python/src/io/mod.rs:408`、`:595`、`:737`、`:1777`、`:1898`)。同时把 `build_index` 的文档改成**后端中立**的契约——今天它写的是"index mapping step numbers to **byte offsets**",那是文件读者的实现细节;新措辞:*"Build and cache whatever per-step index the backend needs for random access. File readers cache byte offsets; `FrameSequence` caches the `step_index` / `offset` arrays."* 三个方法的错误类型都保持今天的 `io::Result`,`FrameSequence` 把 `MolRsError` 转进去(见 §三 的错误词汇句)。`FrameSequence` 实现 `TrajectoryReader`。
- **resolve** `molrs_dtype` bool 分支(`frame_io.rs:132`)—— 属性在这里被**读**,却在任何地方都不被**写**;在 molrs 写出的 store 上这条提升永远不会触发。裁决:删掉该分支,bool 走原生 bool dtype 往返(由 15-dtype 矩阵覆盖)。
- **reuse** `write_simbox` / `read_simbox`(`frame_io.rs:235` / `:272`)—— 每步 box 段直接调它们;boundary 语义修复只落在这一处(`:300-311` 的默认翻成全周期,缺失 `origin` 接受)。`cell_defined` 在 Zarr 路径上此前**双向都不落盘**(写入端不写、读取端硬编码 true;MessagePack 路径 serialize.rs:453 一直无损)——修法:盒组新增可选布尔属性 `cell_defined`,false 必写、缺席 ⇒ true(存量 store 全部产自定义胞语义,零破坏)。每步 box **不重新推导**任何几何。
- **reuse** `join_path`(`frame_io.rs:544`)、`BOX_GROUP`(`:342`)、`MolRsError::zarr` 与 `zerr` 垫片(`record_io.rs:555`)、未知 dtype 即 `Err` 的那一臂(`frame_io.rs:217`)作为错误模型。
- **reuse** `MetaValue`(`core/store/meta.rs:11`,`dtype()` 在 `:42`)作为每步 meta 的 dtype 词汇。
- **pattern** `FrameIndexBuilder`(`io/streaming.rs:60`)—— `FrameSequenceWriter` 照抄其显式生命周期形状(消费式收尾、无 Drop)。
- **pattern** `FrameIndex`(`io/reader.rs:125`)—— 偏移容器的形状照它(行偏移 vs 字节偏移),不发明第三种。
- **pattern** `PublisherConfig`(`stream/publisher.rs`)—— 推导默认值写在构造器、在被读处就地注释。
- **reuse** molrec `chunking.py::plan` 的常数与形状(`../molrec/src/molrec/chunking.py:30,:33`)。
- **new** `FrameSequenceWriter` / `FrameSequence` / `SequenceSchema` —— 全仓无对应物:没有任何按 step 索引做二分或 CSR 的既有代码(最近的先例只是 `core/store/schema/mod.rs:425` 的有序切片 `binary_search_by`;`Neighbors` 是 COO 不是 CSR),流式追加写入端也不存在。与 `Trajectory` 的关系由 §〇 的判别规则界定。
- **new** `io/zarr/pack.rs` —— 新概念(静止态单文件),新依赖,隔离一处。

### 八、molpy 侧:门面为什么可以叫 `zarr`(架构审查 🔴#6 的答复)

审查指出 `molpy/io/zarr.py` 既反转了 molpy `io/__init__.py` 现有的 docstring(它今天写着 "no `MolStore` / Zarr layer"),又把一个后端词放上了 molpy 的表面。**维护者裁定:文件保留**,理由落在 13 号规范自己给出的**豁免口**上:

> 13 号的规则是"名字里的技术词只有一种合法情形:它是**调用者自己选定的格式**,因而是对象的一部分(`read_pdb` / `write_xyz`)"。在 **molrs** 的表面上 `zarr` 是**未裁定的后端**(record 用什么存,维护者还没定),所以必须消失;在 **molpy** 的表面上,用户是拿着一个 `.zarr` 目录/文件来的,格式是**他选的**——这正是豁免口。molpy 的 io 层本来就按格式命名模块(`io/data/pdb.py`、`io/data/mol2.py`、`io/trajectory/xyz.py`),`io/zarr.py` 与它们同一形状,不是例外。

**放在 `io/` 根而不是 `io/trajectory/zarr.py`**:那些 kind 目录每个只装**一种** kind(data 一种、trajectory 一种、forcefield 一种),而一个 record store 横跨全部——frame + system + trajectory 都在里面,它自己就是一种根级 kind,不是某种 trajectory 格式。

配套两条,缺一不可:**函数仍按对象命名**(`read_record` / `write_record`,**绝不**是 `read_zarr`),以及 `io/__init__.py` 那句"no Zarr layer"必须在**同一个规范里**改掉——留着它就是让文档和代码互相指认对方是错的。至于 `src/molpy/io/store/`:它今天只剩 `__pycache__/{__init__,_h5,_zarr}.cpython-314.pyc`,`.py` 早已不在,目录本身让 `import molpy.io.store` 作为隐式命名空间包**成功返回**一个空模块(于是 `hasattr` 探测先答"有"再在调用时炸)。删它是**卫生**,不是搬迁——没有代码从那里搬到 `io/zarr.py`。

### 九、已知限制(明写,不藏)

不支持运行中新增列;撕裂写残余窗口(小写内部崩溃);活动 shard 在封盘或 `close()` 之前可能携带被取代的死字节(分支 A 的尾 chunk 重写所致,已由封盘压实在静止态收掉,并写在 `flush()` 的文档里);已发布的 0.13.2 读者遇到新 store 会以 `trajectory.step length mismatch: expected 0, got N` 失败——响,但消息指错了原因(nstep == 0 时才是静默返回空),这是不可修的那一侧,已按更正后的措辞记进 notes.md 与 storage.md;`zarrs_zip` 门在 `filesystem` 档,意味着浏览器读不了打包后的 `.zarr.zip`——这是选择不是疏漏(wasm `RecordReader` 吃 `js_sys::Map`,当前没有消费者需要 wasm 读 zip;wasm HTTP-range 读 zip 在 Out of scope);gzip 对未 shuffle f64 只有约 1.05–1.2×,且本轮不暴露档位旋钮;单写入端,无并发写。

## Files to create or modify

- `molrs/src/io/mod.rs`
- `molrs/src/io/store/mod.rs` (delete)
- `molrs/src/io/store/csv.rs` → `molrs/src/io/csv.rs` (moved)
- `molrs/src/io/store/zarr/mod.rs` → `molrs/src/io/zarr/mod.rs` (moved)
- `molrs/src/io/store/zarr/frame_io.rs` → `molrs/src/io/zarr/frame_io.rs` (moved)
- `molrs/src/io/store/zarr/record_io.rs` → `molrs/src/io/zarr/record_io.rs` (moved)
- `molrs/src/io/store/zarr/error.rs` → `molrs/src/io/zarr/error.rs` (moved)
- `molrs/src/io/zarr/chunking.rs` (new)
- `molrs/src/io/zarr/sequence.rs` (new)
- `molrs/src/io/zarr/pack.rs` (new)
- `molrs/src/io/zarr/store.rs` (new)
- `molrs/src/io/reader.rs`
- `molrs/src/core/store/meta.rs`
- `molrs/src/core/store/block/dtype.rs`
- `molrs/src/core/store/record.rs`
- `molrs/Cargo.toml`
- `molrs-python/src/core/store/record.rs`
- `molrs-python/src/core/store/trajectory.rs`
- `molrs-python/src/core/store/frame.rs`
- `molrs-python/src/io/mod.rs`
- `molrs-python/tests/test_record.py`
- `molrs-cxxapi/src/lib.rs`
- `molrs-wasm/src/io/zarr/mod.rs`
- `.claude/notes/notes.md`
- `.claude/notes/architecture.md`
- `regressions/release-0-14-15-molrec-zarr-trajectory.py` (new)
- `../molrec/docs/spec/record.md`
- `../molrec/docs/spec/trajectory.md`
- `../molrec/docs/spec/storage.md`
- `../molrec/docs/spec/conventions.md`
- `../molrec/src/molrec/core/model.py`
- `../molrec/src/molrec/core/suite.py`
- `../molrec/src/molrec/core/bindings/zarr.py`
- `../molrec/src/molrec/compare.py`
- `../molrec/tests/molrs_adapter.py`
- `../molrec/tests/test_core_conformance.py`
- `../molpy/src/molpy/io/store/` (delete)
- `../molpy/src/molpy/io/zarr.py` (new)
- `../molpy/src/molpy/io/__init__.py`
- `../molpy/tests/test_io/test_zarr.py` (new)

## Tasks

- [x] Record the executed spike's verdicts in the zarr module doc with the observation behind each — Q1–Q4 and Q6's codec half green, **Q5 = branch A** (tail-only trailing-chunk rewrite, completed shard untouched, bit-exact readback), **Q7 = `zarrs_zip` 0.5.2** (stored-entry reads bit-exact, zarr-python 3.3.0 cross-reads the same zip) — and keep its `#[cfg(test)] mod zarrs_pins` assertions live as the codec-layer half of ac-011; the zarrs_filesystem read-modify-write finding is superseded into the Design (`PositionalWriteStore`)
- [x] Drop the `Reader` supertrait from `TrajectoryReader` in `molrs/src/io/reader.rs`, keeping `build_index` / `read_step` / `len` and their `io::Result` error type, and reword `build_index`'s doc to a backend-neutral contract
- [x] Move `io/store/zarr/` to `io/zarr/` and `io/store/csv.rs` to `io/csv.rs`, delete `io/store/mod.rs` after rewriting its boundary sentence into the single children enumeration in `molrs/src/io/mod.rs:1-9` (with `io/zarr/mod.rs` and `io/csv.rs` pointing at it, not copying it), and update every call site plus the naming-gate exemption (`molrs/src/core/store/record.rs:5`, `molrs-python/src/core/store/{record,trajectory}.rs`, `molrs-python/src/io/mod.rs:2417,:2427`, `molrs-cxxapi/src/lib.rs:15`, `molrs-wasm/src/io/zarr/mod.rs:40,:49,:60`, `molrs-python/tests/test_record.py:28-30,:164`) — done; measured side effect: the old `"/io/store/zarr/"` exemption was dead (scanned `_PUBLIC_TREES` exclude `molrs/src`), the new `"/io/zarr/"` now also exempts `molrs-wasm/src/io/zarr/` — correct under 13's adapter carve-out, recorded as deliberate in the notes task
- [x] Regate the store-taking writers in `molrs/src/io/zarr/{record_io,frame_io}.rs` from `filesystem` to `zarr`, leaving `filesystem` on the four path-taking doors only — done; wasm32 check ran: `--no-default-features --features io,zarr --target wasm32-unknown-unknown` compiles the writers in (pre-edit probe failed E0432 "gated behind filesystem")
- [x] Write failing round-trip unit tests for all 15 `Column` dtypes, `structural_shape`, and a declared column-less block in `molrs/src/io/zarr/frame_io.rs` — landed as pins (defect was zero coverage, not breakage); arrival-width asserted at range extremes
- [x] Write failing unit tests for the three frame defects (absent `boundary` reads all-periodic; an envelope-shaped raw JSON meta value survives verbatim; a rewrite leaves no stale node) in `molrs/src/io/zarr/frame_io.rs` and `molrs/src/core/store/meta.rs` — 4 RED confirmed (incl. absent-origin errors instead of defaulting); tester also proved `cell_defined` survives in NEITHER direction on the Zarr path → folded into the read_simbox task as an optional box-group attribute, absent ⇒ true
- [x] Fix `MetaValue::from_attr_value` in `molrs/src/core/store/meta.rs:182` to decode raw JSON only, leaving the serde path on `from_json_value` untouched
- [x] Default an absent `boundary` attribute to all-periodic in `read_simbox` (`molrs/src/io/zarr/frame_io.rs`), accepting an absent `origin` (zero origin), and make `cell_defined` actually survive the Zarr path: `write_simbox` emits it as an optional box-group bool attribute (must-write when false), `read_simbox` defaults absent ⇒ true (every existing store was written under defined-cell semantics — zero breakage) — done; the false direction's test rides the next tester batch
- [x] Erase the target node before writing in the record and frame writers so a rewrite leaves no stale children — strategy: every writer erases its own target prefix (write_record_store erases the root, write_frame_group its own prefix; one rule, two sites, private `node_prefix` helper)
- [x] Delete the dead zarr surface — `UnitSystem`, `Provenance`, `write_f32_array`, `write_u8_array`, and the never-written `molrs_dtype` bool branch — without touching `ff::typifier::estimate::Provenance` (grep-guarded: no binder re-exported either; the only remaining `molrs_dtype` occurrences are the guard pin's own name/assert — ac-009 narrowed to non-test uses)
- [x] Write failing unit tests for `chunking::plan` and `DType::itemsize` against molrec's constants in `molrs/src/io/zarr/chunking.rs` and `molrs/src/core/store/block/dtype.rs` — 8 plan goldens (uv-run provenance, both sides of SHARD_ABOVE, itemsize=None, empty/scalar shapes) + exhaustive 15-variant width table + the owed `undefined_cell_round_trips_as_false` pin (green); compile-RED on `plan`/`ChunkPlan`/`itemsize`; also caught the stale worked-example numbers (fixed: R=21845/k=512/≈3728)
- [x] Implement `chunking::plan` returning a private named `ChunkPlan` in `molrs/src/io/zarr/chunking.rs`, and `DType::itemsize` in `molrs/src/core/store/block/dtype.rs` — 8 goldens + width table green; `pub(in crate::io::zarr)` (ac-022 holds)
- [x] Generalize `write_typed_array` in `molrs/src/io/zarr/frame_io.rs` to take a `ChunkPlan`, routing every fixed-size array through it and deleting `write_float_array` / `write_f64_array` / `write_i64_array` — record_io twins deleted; frame_io's `write_f64_array` survives as a 9-line delegating adapter (test call sites pin it; no longer a twin); chunks=None falls back to whole-shape + plain gzip (`GZIP_LEVEL = 5`, zarrs canonical); on-disk layout verified against the plan goldens
- [x] Generalize `write_column` into `dtype_of(&Column) -> (DataType, FillValue)` and parameterize `read_column` on `&ArraySubset` in `molrs/src/io/zarr/frame_io.rs` — done; eager path pays one extra metadata GET per column (accepted; sequence reader passes real subsets)
- [x] Write failing unit tests for `PositionalWriteStore` in `molrs/src/io/zarr/store.rs` (a partial write at an offset leaves the surrounding bytes and the file length untouched; a write past EOF extends the file correctly; a partial write never shortens the file; a subsequent `set` truncates to the new length; `set` still replaces the whole value; read and list results equal the wrapped `FilesystemStore`'s; `bytes_written()` equals the bytes handed to `set_partial_values`, not the file size; a 16 B tail write on a large file does not read or rewrite the file) — 9 tests compile-RED; constructor pinned `new(path)`; zarrs 0.23.13 note: the required override is `set_partial_many` (`set_partial` is provided sugar); the owed sharding golden also landed green (1 shard file measured for a 7-chunk column)
- [x] Implement `PositionalWriteStore` in `molrs/src/io/zarr/store.rs` (gated `filesystem`, `std::fs` only, open read+write without truncate and `write_at` at the offset, `supports_set_partial() == true` and true at the disk level, read/list delegated, own `AtomicU64` byte accounting), and route the two path-taking write doors and `FrameSequenceWriter`'s filesystem path through it — the read door `read_record_file` stays on plain `FilesystemStore`. Done (9/9; unix-only `write_all_at`, documented; inner store's file-handle cache must stay 0, documented at the constructor). `bytes_written()` carries a staged `#[allow(dead_code)]` whose removal is part of the sequence.rs task's definition of done
- [x] Write failing unit tests for the sequence layout primitives (CSR `offset` diff, `step_index` binary search, `R` and `k` derivation, typed-meta fill rule, reserved-name rejection at schema construction, union mint across heterogeneous frames and its conflicting-dtype error, schema-violation errors) in `molrs/src/io/zarr/sequence.rs` — 36 tests landed compile-RED, all bodies type-checked against a stub; nine API frictions surfaced and ruled (step/time producer → `append_at`; block-level sparsity; schema attrs authoritative for `open`)
- [x] Write failing unit tests for the writer/reader lifecycle (`create` on an occupied path errs; `open` errs on schema mismatch; append → `mem::forget` → reopen; reopen-and-append across a chunk boundary; `step`-extended-last ordering; legacy `trajectory/frames/` detection) in `molrs/src/io/zarr/sequence.rs` — same 36-test batch (groups D/E/F/G/H/I incl. ac-034 compaction and ac-011 store-half amp)
- [x] Implement the typed per-step meta arrays (`trajectory/meta/<key>`, `molrs_meta_dtype`, declared-fill rule) in `molrs/src/io/zarr/sequence.rs` — all 20 MetaValue variants bit-exact
- [x] Implement `FrameSequenceWriter` and `SequenceSchema` in `molrs/src/io/zarr/sequence.rs` (`SequenceSchema::from_frame` plus the derived-union `SequenceSchema::from_frames` that errors on conflicting dtype or trailing shape; create / open / append / flush / `close(self)`; branch-A commit with seal-on-complete compaction — re-encode each shard once as a clean full write when an append crosses its boundary, and compact the final partial shard in `close(self)`; `with_rows_per_chunk` and `with_chunks_per_shard` only; 256 MiB shard target over `PositionalWriteStore`; explicitly threaded `CodecOptions`; `step` extended last; no Drop; module doc carrying the `Trajectory` / `FrameSequence` / `FrameSequenceWriter` discriminating rule, the per-door error vocabulary, and branch A's durability contract including the active shard's superseded bytes) — done; determinism required `SubchunkWriteOrder::C` on every growth array (zarrs default lays subchunks in HashMap order) and compact = retrieve_chunk + store_chunk (whole-value set, truncates — cannot leave a superseded copy by construction); two layout limits in the module doc: a zero-row update is the absence marker (a genuinely-present zero-row block is indistinguishable), and box/ has no absence marker (latest cell wins)
- [x] Implement `FrameSequence` in `molrs/src/io/zarr/sequence.rs` (index-only open, `frame(i)`, `to_trajectory`, `impl TrajectoryReader`, legacy detection), exported no further than `molrs::io::zarr` — done; `append_at(frame, step, time)` landed as the single real door with `append` a two-line wrapper
- [x] Rewire `write_trajectory_file` / `read_trajectory_file` in `molrs/src/io/zarr/record_io.rs` into thin doors over `FrameSequenceWriter` / `FrameSequence`, deleting `write_trajectory_section`, `read_trajectory_section`, `read_frame_from_store` and `count_frames_in_store` so the layout has exactly one encoder and one decoder — done (plus orphaned read_float_values/read_i64_values); three behavior deltas recorded for notes.md: heterogeneous per-step meta errs at the eager door (decision 8's no-implicit-fill, accepted limitation — declare_meta is streaming-API-only), `step: None` reads back as `Some([0..n])` (step is the commit marker, always written), rewrite-over-same-path verified
- [x] Move molrs-wasm's `RecordReader` onto a `RefCell<FrameSequence>` in `molrs-wasm/src/io/zarr/mod.rs`, keeping its `&self` JS signatures and returning a descriptive `JsError` on `try_borrow_mut` failure rather than panicking — done; one frame per call, len() off the cached index; JS-surface change: a trajectory-less store now throws at construction (was an inconsistent countFrames()==1/readFrame(0)==null; zero in-tree JS consumers) — notes.md line
- [x] Add a file-count bound test generalizing `walk_json` (`molrs/src/io/zarr/record_io.rs:624`) that asserts the bound scales with `total_bytes / S`, not with nstep — measured 11 files at N=8 frames vs 13 at 2N=16 (bound 12/15, exact-equality asserted); plus the two owed runtime covers: rewrite-over-same-path and the eager door's legacy refusal
- [x] Implement `pack()` as a free function over a closed store path in `molrs/src/io/zarr/pack.rs` with `zarrs_zip`'s stored-entry read path, gated `filesystem`, promoting the spike's `zarrs_zip` + `zip` dev-dependencies to optional dependencies wired as `filesystem = [..., "dep:zip", "dep:zarrs_zip"]` in `molrs/Cargo.toml`, and rewriting the now-stale dev-dependency comment — done (`pack` + `open_packed`; coupling landed: `FrameSequence::open` widened to a generic read-only door, field typed `ReadableListableStorage` so no write capability survives; both RWL call sites demoted via `readable_listable()`; wasm normal graph zip-free with positive control)
- [x] Record in `.claude/notes/notes.md` the cross-repo on-disk layout break (`molrs-cxxapi/src/lib.rs:15` → Atomiverse `cpu::ZarrReader`, old→new plus "0.13-written stores must be re-written with 0.13"), the decision-10 no-bump ruling with its true cost (a molrs ≤ 0.13.2 reader fails with `trajectory.step length mismatch: expected 0, got N`, silent empty only when nstep == 0, and the message names the wrong cause; `record_schema_version` stays 1 so `read_meta`'s precise `unsupported record_schema_version` path is not taken), the decision-11 single-spec ruling, the `UnitSystem` / `Provenance` deletion with why compatibility was moot, the deliberate side effect that dropping `TrajectoryReader`'s `Reader` supertrait makes the trait dyn-compatible (`&mut dyn TrajectoryReader` becomes legal — re-erasing it later would be a breaking change), and the measured `zarrs_filesystem` 0.3.12 fact that `set_partial_many` is a whole-value read-modify-write while reporting `supports_set_partial() == true`, with `PositionalWriteStore` recorded as the fix so nobody simplifies it away, and the naming-gate exemption fact (the old `"/io/store/zarr/"` entry matched zero scanned files; the new `"/io/zarr/"` deliberately exempts `molrs-wasm/src/io/zarr/` too — the wasm zarr binding is adapter layer under 13's own carve-out); repoint the `io` row of `.claude/notes/architecture.md:35` to `io::zarr` and the `io::store::zarr` path string in `.claude/notes/notes.md:47`'s 13-naming entry to `io::zarr`
- [x] Amend rule 2 of `../molrec/docs/spec/record.md:67` to include `trajectory`, rewrite `trajectory.md` / `conventions.md` for the ragged + `step_index` form (the box section gains the optional `cell_defined` bool attribute, absent ⇒ true), and record in `storage.md` both the layout change and the true old-reader behaviour — done, plus consistency edits in overview.md/README.md (they restated rule 2); single legacy mention survives only in storage.md's cost note (grep-proved)
- [x] Widen `MolRec::validate` (`molrs/src/core/store/record.rs:143`) to accept a trajectory-only record, then delete the frame-0 duplication at `molrs/src/io/zarr/record_io.rs:461` — done in the required order; module-doc restatement widened in the same commit; `_lib.pyi:1641`'s stale docstring routed to docs Mode A
- [x] Add `TrajectoryModel` to `../molrec/src/molrec/core/model.py` and `TrajectorySuite` to `../molrec/src/molrec/core/suite.py`, with its store binding in `../molrec/src/molrec/core/bindings/zarr.py` — done (logical model + 6-case suite + Python reference codec writing byte-compatible array names/dtypes/shapes; RecordModel gains trajectory; schema JSON regenerated; molrec env wired to fresh molrs 0.14 wheel via path dev-dep, flip-to-published-pin comment left)
- [x] Collect the molrs adapter into `../molrec/tests/test_core_conformance.py` so the suite runs against both molrs and molrec's own codec — full (suite × implementation) matrix green, 40/0; en route: molrs read door relaxed to derive schema without the writer pin, `fill` ruled write-side-only (materialized, not recorded), trajectory adapter bridges bare stores via minimal-meta graft
- [x] Close the two molrec harness gaps: a store-level hand-built boundary-absent test, and a `__pydantic_extra__`-aware `_diff_model` in `../molrec/src/molrec/compare.py:44-57` — both closed and proven to bite (the extras case was empty-diff before the fix; molrec's reader was already all-periodic — the gap was suite reachability; molrs's absent-boundary read verified green on the fresh wheel)
- [ ] Delete `../molpy/src/molpy/io/store/` and add the `../molpy/src/molpy/io/zarr.py` façade (`read_record` / `write_record` forwarding to molrs), correcting the `../molpy/src/molpy/io/__init__.py` docstring — executes only on the post-`release-0-14-09` branch, after the molrs surface it forwards to is tagged and published per CLAUDE.md § Release before molpy
- [x] Add regression example `regressions/release-0-14-15-molrec-zarr-trajectory.py` (public API only; hard-coded goldens, no third-party runtime) — green, exit 0 on the fresh wheel; bit-pattern compares (−0.0/NaN), 2⁵³+1 i64, 2³²−1 u32, file bound 27≤27, legacy refusal; every assertion mutation-proven; pre-existing molrs-python schema numpy_dtype/width mismatch named → post-spec /mol:fix
- [x] Run full check + test suite — molrs: lib 1743/0, doctests 66/0, clippy ×2 clean, fmt clean, `uvx prek run --all-files --hook-stage pre-push` exit 0; molrec: pytest 40/0, ruff clean; molpy leg gated with its task. Simplify ran clean (1 doc-link fix; 3 manual → /mol:refactor); docs Mode A applied (13 error vocabularies, .pyi four-section rule, rustdoc errors 49→48 all pre-existing elsewhere)

## Testing strategy

molrs 的单测按本仓已捕获的规则放在**源码旁的 `#[cfg(test)]`**(CLAUDE.md § Testing Rules:"Prefer unit tests next to the code";本仓**没有** `molrs/tests/` 集成树,`architecture_gate.rs` 是默认门跑不到的 `[[test]]` 目标,13 号已裁定门不放那里)。每条单测只测**一个**函数/方法,单测绿 = `cargo test -p molcrafts-molrs --lib --features full,filesystem <module::path>`。Python 侧按镜像布局:molrs-python 平铺 `tests/test_*.py`,molpy `src/molpy/io/zarr.py` → `tests/test_io/test_zarr.py`,molrec `tests/test_core_conformance.py`。绝不在 `tests/` 下放 e2e;跨仓与长跑证据一律进 `regressions/`。

**逐仓门**:molrs —— `cargo test -p molcrafts-molrs --lib --features full,filesystem`、`cargo test --doc -p molcrafts-molrs --features full,filesystem`、`cargo clippy … -D warnings`、`prek run --all-files --hook-stage pre-push`。molrec —— `uvx ruff format --check .`、`uv sync --extra dev && uv run pytest -q`。molpy —— `ruff check src tests && ty check src/molpy/`、`uv run --extra dev python -m pytest tests/ -n auto`。

**happy path**
- 15 个 `Column` dtype × 往返:每个 dtype 写出读回 `assert_eq!` 逐位相等;`structural_shape` 声明后往返不变;声明了却无列的 block 往返仍保住 `nrows`(注意:早前"块计数缺陷"的说法**不复现**——`Block::resize(n)` 在空 block 上会置 `nrows=Some(n)`,`core/store/block/mod.rs:642`;本测试把这个既有行为**钉住**,不是修它)。
- 三帧 ragged 序列(原子数 3 / 5 / 4)append → flush → close → reopen:每帧坐标逐位相等,`offset` 为 `[0,3,8,12]`,`step` 为 `[0,1,2]`。
- 异构 trajectory:三帧各带不同的 block/列集合,经 `from_frames` 铸出并集后写出读回,每帧只带回**它自己**呈现过的 block(缺席不是空,是缺席)。
- 定胞 NVT:20 帧后 `box/step_index` 恰好 1 条;恒定拓扑 block 的 `step_index` 恰好 1 条。
- `FrameSequence` 经 `FrameIterator`(`io/reader.rs:166`)遍历,与逐个 `frame(i)` 结果相同——证明去掉 `Reader` 超 trait 后既有泛型消费者仍然可用。

**边界 / 属性**
- 构造与打开:`create` 打在已有序列的路径上 `Err` 并指名路径(**不覆盖**);`open` 遇到 dtype / trailing shape / chunk 尺度不符时 `Err`,消息含 `expected … found …`;保留名(`step`/`time`/`meta`/`box`/`offset`/`step_index`)在 `from_frame` / `from_frames` / `create` **当场**被拒,不是等到第一次 append;`from_frames` 遇到同名列在两帧里 dtype 或 trailing shape 冲突时 `Err` 并指名该列与两个值。
- 提交与耐久:append 若干帧 → `flush()` → `std::mem::forget(writer)` → 重新 open:`len()` 等于上次 `flush()` 之前 append 的全部帧(**分支 A**,spike 已裁),且每一帧都能读且逐位相等;测试**显式断言崩溃丢失边界**——丢的恰是上次 flush 之后 append 的帧,不多不少。
- 封盘压实:逐帧 flush 与一次性 flush 写出的同一批帧,`close()` 之后 shard 文件**逐字节相同**;`close()` 之前活动 shard 确实更大(证明压实那一步在干活,而不是断言恒真)。
- 顺序属性:在 `step` 扩展**之前**的任意写点截断,重新 open 得到的 `len()` 不包含未提交帧,且不报错。
- 跨 chunk 边界:close → reopen → 继续 append 直到越过一个 inner chunk(用 `with_rows_per_chunk` 压小 `R`),全部帧仍逐位相等,chunk / shard 尺度与创建时相同。
- 写放大:一次 flush 写出的字节 ≈ 一个 chunk + 一个 shard 索引,**不是** O(shard)(Q6 断言;这是 256 MiB shard 下唯一挡住无声灾难的东西)。
- 文件数上界:N 帧 store 的文件数 ≤ `total_bytes/S + O(arrays)`,常数取新默认值,并用 `with_chunks_per_shard` 压小 `S` 让小 store 也跨多个 shard;把 N 翻倍,文件数**不翻倍**。
- schema 违例:未声明的 block、未声明的列、dtype 变更、trailing shape 变更各自报错且消息指名;呈现声明并集的**真子集**的帧被接受。
- meta:省略已声明且无 fill 的 key 报错(**不写 NaN**);声明了 fill 的 key 省略时写 fill;`MetaValue` 的每个变体逐位往返。
- 旧布局:含 `trajectory/frames/` 的 store 在 `FrameSequence::open` 与 `read_trajectory_file` 两扇门上都报 "legacy layout (written by molrs ≤ 0.13); re-write with 0.13",不静默返回空。
- 信封:`{"dtype": "not-a-dtype", "value": 1}` 这样的普通 JSON meta 原样读回为 `MetaValue::Json`;`serialize.rs` 的 MessagePack 往返不受影响。
- boundary:缺失属性读作 `[true,true,true]`;显式 `[true,false,true]` 原样读回;缺失 `origin` 接受并落到零点。
- 打包:`pack()` 后目录消失、`.zarr.zip` 为单文件、zip 条目全是 stored(method 0)、从 zip 读回的每一帧与目录形式**逐位相等**;`pack` 的入参是路径而非活句柄(编译期即成立)。
- wasm 重入:`RecordReader` 的两个 JS 方法在借用冲突下返回 `JsError` 而非 panic。

**域验证(硬编码期望值)**
- `chunking::plan` 与 molrec 同参数结果一致,`ChunkPlan` 的两个字段各写死具体元组;`SHARD_ABOVE = 4` 的分界两侧各一例;`DType::itemsize()` 对 String 为 `None`。
- 算例回归(3000 原子 f64 xyz,R = 21000,k = 532):写死 `R`、`k`、每 shard 帧数 3724 与 10⁴ 帧的 shard 数 3。
- molrec 一致性套件对**两个**绑定(molrs 与 molrec 自带 codec)运行,且被 pytest **收集**;两处 harness 漏洞随之关闭:(a) `BoxModel._square_and_filled_in`(`../molrec/src/molrec/core/model.py:194`)在每次校验时把 boundary 物化成全 True,`_write_box`(`core/bindings/zarr.py:180`)又在非 None 时一律写出 ⇒ 套件永远造不出"属性缺失"的 store,因此需要一个**store 级手工搭建**的测试直接写一个没有该属性的 box 组;(b) meta 未知键"保留"这条今天什么也没断言 —— `_diff_model`(`../molrec/src/molrec/compare.py:46`)遍历 `type(expected).model_fields`,而 `extra="allow"` 的键住在 `__pydantic_extra__`,于是 `creator` / `x_vendor_local` 从未被比较过;修比较器,并用一个"额外键值不同"的用例证明它会红。

**回归样例** `regressions/release-0-14-15-molrec-zarr-trajectory.py`:纯公开 API(`import molrs`,`Trajectory` / `Record`——名字不变,布局在脚下换掉),写出一条 3 帧 ragged 轨迹(原子数 3 / 5 / 4,含 f64 / i64 / bool / u32 四种列与一个每步 meta 标量),读回后断言坐标与 meta **逐位相等**于脚本内**写死**的黄金值,断言文件数 ≤ 写死的上界,并断言读一个含 `trajectory/frames/` 的手工 store 会抛出指名的 legacy 错误。不 import 任何第三方科学软件,不在测试期调用外部工具。

## Open questions (maintainer ruling required)

无 —— 决定 1–14 全部由维护者裁定;架构审查提出的四条与裁定相撞的意见(命名、不升版、molpy 门面命名、单规范)按裁定执行,理由已写进 Design §〇 / §一 / §六 / §八,供后来者查阅而不是重新辩论。其中不升版一条的**事实前提**在复审中被更正(旧读者是响的,不是沉默的),裁定不变,更正后的前提由调用方呈递维护者。原先仅有的两个待定**事实**已由 spike 当场钉死:Q5 = 分支 A,Q7 = `zarrs_zip`;spike 另外证伪了初稿 §四 的一处规范级错误(写放大守卫放高了一层),已由 `PositionalWriteStore` 修正。至此本规范无待决项。

## Out of scope

- byte-shuffle / bitshuffle 编解码器与 **gzip 档位旋钮**(同一个尺寸/吞吐议题的两半,一起留给路线图;本轮 gzip 停在库默认档)
- 有损压缩(精度研究禁止)
- 活写单文件(明确否决:zip 不能就地部分写)
- wasm HTTP-range 读 zip(Q7 之后的跟进)
- 运行中扩展 schema(新增列)
- 多写入端并发
- `AtomicWriteStorageAdapter`(其 `supports_set_partial() -> false` 会杀死追加快路径)
- `FrameSequence` 的 Python 绑定与吃 path 的 `filesystem` 门(本轮无点名消费者)
- `/mol:map` 对 `.claude/notes/architecture.md` 的整体再生成(本规范只改 `io` 那一行)
- Atomiverse 侧的实际改动(本规范只负责告知与记录)
- molpy 除 zarr 门面外的 io 格式(10 号规范拥有)
- 13 号验收文档里对旧路径的散文提及(13 号按自己的节奏关闭并删除)
- `record_schema_version` 升版(决定 10,已裁定;更正后的代价记在 notes.md 与 storage.md,重议由维护者发起)
