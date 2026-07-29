---
slug: cell-grid-api
criteria:
  - id: ac-001
    summary: 非周期轴钳位、周期轴回绕，三轴互不影响
    type: code
    pass_when: |
      CellGrid::with_dims([4,4,4], [true,false,false]) 下：
      x 轴上 12.5 与 -7.5 与 2.5 落入同一 cell（回绕）；
      y 轴上 -30 落入 cell 0、+90 落入 cell 3（钳位到边缘，不回绕）；
      三轴同时越界时各按自己的 pbc 分派。
    status: done
  - id: ac-002
    summary: 两个非周期 cell 必须互相可见
    type: code
    pass_when: |
      CellGrid::with_dims([2,1,1], [false,false,false]) 下
      stencil_all(1) == [0] 且 stencil_all(0) == [1]。
      （被替换的实现在该轴上用 {0,+1} 偏移，cell 1 的全壳为空。）
    status: done
  - id: ac-003
    summary: stencil 返回值无重复、不含自身、forward 严格向前
    type: code
    pass_when: |
      celldim 分量取遍 {1,2,3,5} 的全部组合 × pbc 4 种组合下，
      stencil_all 与 stencil_forward 的返回切片内无重复索引且不含自身 cell；
      stencil_forward 的每个返回值严格大于入参 cell。
    status: done
  - id: ac-004
    summary: stencil 与独立邻接判据一致，且 forward 全扫描每对恰一次
    type: code
    pass_when: |
      以独立写出的逐轴邻接判据（相等 / 相差 1 / 周期且 dim>=2 时首尾相邻）为 oracle：
      stencil_all(i) 的集合等于该 oracle 给出的邻居集合；
      遍历所有 cell 收集 stencil_forward 得到的无序 cell 对集合等于 oracle 的无序邻接对集合，
      且总条目数等于该集合大小（即无重复、无遗漏）。
    status: done
  - id: ac-005
    summary: for_cutoff 按平面距离定尺，并声明盒子窄于 cutoff 的退化情形
    type: code
    pass_when: |
      六方胞（a=b=c=10, γ=120°）下每个方向的 cell 宽度
      nearest_plane_distance[k] / celldim[k] >= cutoff；
      且 celldim[k] <= npd[k] / cutoff（即确按平面距离而非棱长定尺）。
      盒子窄于 cutoff 时 celldim 退化为 1，该限制在 for_cutoff 的文档中写明。
    status: done
  - id: ac-006
    summary: LinkCell 与 BruteForce 的 pair 集合在完整矩阵上逐对相等
    type: runtime
    pass_when: |
      矩阵 = {正交, 六方 (a=b, γ=120°), 强倾斜三斜} × pbc {(t,t,t),(t,t,f),(f,f,f)}
      × cutoff 使 celldim 分量分别取到 1/2/3/5。每种组合下 LinkCell::visit_pairs
      产生的无序 pair 多重集与 BruteForce 完全相同（无重复、无遗漏），
      对应 dist_sq 差 <= 1e-12。粒子集合包含非周期轴上位于盒外的点，
      以及分数坐标恰为 0.0 与 1.0 的边界点。
    status: done
  - id: ac-007
    summary: 查询模式同样与 BruteForce 一致
    type: runtime
    pass_when: |
      同一矩阵下，NeighborQuery 的 cross-query 结果与独立 oracle 一致。
      矩阵必须覆盖非周期轴 celldim==2 —— ac-002 的缺陷在查询路径上最直接。
    status: done
  - id: ac-008
    summary: 函数级微基准不超基线 1%
    type: performance
    pass_when: |
      neighbors/cellgrid/cell_of/{ortho,triclinic} 与
      neighbors/cellgrid/stencil_fwd/{ortho,triclinic} 的 criterion 中位耗时
      <= spec 中记录的基线 * 1.01。这些基准是永久的回归防线，
      不保留被替换实现的 sentinel 副本。
    status: pending
  - id: ac-009
    summary: 调用方微基准不超基线 2%，三斜进入长期追踪
    type: performance
    pass_when: |
      neighbors/traversal/{build,visit_pairs}/{ortho,triclinic} 的中位耗时
      <= 记录基线 * 1.02（build 为 rayon 并行且实测抖动达 ±19%，仅作告警）；
      三斜条目出现在 bench.yml 的 bencher 输出中，可长期追踪。
    status: pending
  - id: ac-010
    summary: 端到端灾难告警
    type: performance
    pass_when: |
      既有的 neighbors/{build,update,build_soa,query_columns} 组整体 <= 基线 * 1.10。
    status: pending
  - id: ac-011
    summary: 被替换的私有实现已删除
    type: code
    pass_when: |
      linkcell.rs 中不再存在 get_cell3 / stencil_range / wrap /
      collect_stencil_into / stencil_fwd_into / stencil_all_into；
      全仓无 cell 分配或 stencil 的第二份实现。
    status: done
  - id: ac-012
    summary: 质量闸
    type: runtime
    pass_when: |
      cargo fmt --all --check、cargo clippy --all-targets -- -D warnings、
      cargo test -p molcrafts-molrs 全部 exit 0。
    status: done
---

# Acceptance — cell-grid-api

本 spec **不做向后兼容**：被替换的实现在非周期轴上是错的，与它一致不是目标。
正确性一律以 `BruteForce` 为准，不以旧代码为准；不保留 sentinel 副本，
也不要求 `sorted_idx` / `cell_start` 与旧实现逐字节相同。

## AC-001 — 非周期轴钳位、周期轴回绕

packer 采纳的前提。被替换的 `get_cell3` 对三个轴一律取模，而且分数坐标在
`make_fractional_fast_arr3` 里就已经被无条件折回 `[0,1)`，钳位所需的信息在到达
cell 分配之前就丢了——这就是为什么 `SimBox` 要多一个不折回的
`make_fractional_raw_arr3`。packer 的中间迭代常常把原子推出盒外，
回绕会让它对真正的邻居隐身、却和无关粒子配对。

## AC-002 — 两个非周期 cell 必须互相可见

实测发现的真实缺陷。旧的 `stencil_range` 在 `dim == 2` 时用 `{0, +1}` 偏移：
cell 1 的 `+1` 越界被跳过，`-1` 从不尝试，于是 cell 1 的全壳为空，
永远看不到 cell 0。

这里更正 spec 起草时的一个判断：当初担心的是 `dim == 2` 且周期时半壳会把
`{0,1}` 发两次。实际不会——`nc > cell` 过滤加排序去重已经挡住了。
真正的漏洞在**全壳**（查询路径），不在半壳（pair 路径）。

## AC-003 / AC-004 — stencil 的自洽性与完备性

用独立写出的逐轴邻接判据当 oracle，而不是另一份同构实现。
AC-004 的"多重集相等"同时抓住重复与遗漏两类错误。

## AC-005 — for_cutoff 按平面距离定尺

倾斜胞下按棱长定尺会让 cell 比 cutoff 薄，静默丢 pair。同时把退化情形写清楚：
盒子本身窄于 cutoff 时只能给一个 cell，此时 3×3×3 stencil 不再充分，
搜索退回最小镜像——这是最小镜像约定本身的限制，不是分区能修的。

## AC-006 / AC-007 — 与 BruteForce 的等价矩阵

AC-007 单列查询模式，因为 AC-002 那类缺陷**只在查询路径上可见**：
pair 路径的 forward 过滤会把它掩盖掉。只测 pair 会漏掉整整一类 bug。

## AC-008 / AC-009 / AC-010 — 性能三层闸

不保留 sentinel，改用永久微基准 + `bench.yml` 的长期追踪作为回归防线。
端到端基准的噪声底（±3%）与门限同量级，不能单独作为依据，所以函数级与调用方
两级都要有。`build` 是 rayon 并行、实测抖动 ±19%，只能当告警。

设计上的一个实测依据：wrap/clamp 分派放在调用边界之后时，`cell_of` 在正交盒上
比无条件回绕慢约 10%；内联之后反而在两种盒型上都更快（分派是循环不变量，
会被提出去；三斜路径还省掉了三次随即被覆盖的回绕）。因此 `cell3` / `cell_of`
的 `#[inline(always)]` 是承重的，不是装饰。

## AC-011 — 被替换的私有实现已删除

"下沉"必须真的完成。留着第二份实现意味着两套语义并存，日后必然分叉。

## AC-012 — 质量闸

项目标准闸门。
