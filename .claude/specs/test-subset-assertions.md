---
title: "四处手写的 fixture 子集 —— 恰好排除了唯一可能失败的分子"
slug: test-subset-assertions
status: approved
created: 2026-07-15
---

# 手写子集：`["e_ethane"]` 反模式的复发

## Summary

`chem-perceive-15-final-acceptance` 的 `no_test_asserts_on_a_subset_of_its_fixtures` 门禁抓到 **4 处手写的 fixture 子集**：

| 位置 | 子集 |
|---|---|
| `tests/ff/mmff/energy.rs:781` | 零电荷分子对 |
| `tests/ff/mmff/energy.rs:979` | `S_NAMES` |
| `tests/ff/typifier/mmff_variant.rs:69` | `N_FIXTURES` |
| `tests/ff/typifier/mmff_variant.rs:73` | `IDENTICAL_FIXTURES` |

**而门禁计算出来的分区显示：`e_caffeine` 和 `e_big` 确实带离域氮，却两个名单都不在。**

⇒ **从来没有任何东西断言过 MMFF94s 会改变咖啡因的能量。**

## Domain basis

这是这条链上**最贵的那个教训**的复发：

> `generic_path_total_energy_matches_rdkit` 曾只断言 `["e_ethane"]`——而乙烷恰好是**仅有的两个 MMFF 电荷全为零的分子之一**，即**唯一一类结构上不可能暴露"缺少静电项"的输入**。150 kcal/mol 的洞在测试全绿的情况下活了一个月。

**共同的病：子集恰好排除了唯一可能失败的分子。**

`mmff-orthogonal-01` 的修法是让 fixture 列表**由目录扫描产生**——只断言子集从此不是一件需要辩解的事，而是**一件写不出来的事**。这四处漏网了。

## Design

对每一处：**能计算的分区，就不许手写。**

- `N_FIXTURES` / `IDENTICAL_FIXTURES` 的判据是"这个分子带不带 type-10/40 的离域氮"——**这是可以从 fixture 算出来的**，不是需要人来记的。
- 零电荷分子对同理：`Σ|q| == 0` 可算。
- `S_NAMES`：查清它的判据是什么，然后计算它。

**如果某个判据真的算不出来**（存在这种可能），那么：**必须在测试里写明为什么排除了其余的**——而"还没实现"不是排除理由，**是失败的理由**。

## Tasks

- [ ] Replace `N_FIXTURES` / `IDENTICAL_FIXTURES` with a COMPUTED partition (does the molecule carry a type-10/40 delocalized N?) — the predicate is derivable from the fixture, not something a human should have to remember
- [ ] Replace the zero-charge pair at `energy.rs:781` with `sum|q| == 0`, computed
- [ ] Determine what `S_NAMES` (`energy.rs:979`) actually selects for, and compute it
- [ ] For any subset that genuinely CANNOT be computed: state the reason IN THE TEST. "Not yet implemented" is a reason to FAIL, not to exclude.
- [ ] `architecture_gate::no_test_asserts_on_a_subset_of_its_fixtures` goes green

## Testing strategy

转绿之后，**必须证明新的分区真的把 caffeine 和 e_big 收进去了**——否则只是换了个写法，洞还在。

具体地：MMFF94 与 MMFF94s 在 `e_caffeine` 上的总能量**必须不同**（它带离域氮），这条断言今天不存在。

## Out of scope

- 门禁自己的盲点（`N_CENTRE_ORACLE` 是一个 4 行的 oracle 表，其行携带非 fixture 字符串，因此不被标记——tester 明确说了这个代价）。另议。
