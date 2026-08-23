---
title: CI 双形态门:静态默认与动态 opt-in 各自被观测,不再只靠 pre-push
slug: link-mode-static-default-02-ci-gate
status: code-complete
created: 2026-08-23
last-updated: 2026-08-23
grilled: true
---

# CI 双形态门:静态默认与动态 opt-in 各自被观测,不再只靠 pre-push

## Summary

01 号 spec 落地后,静态是零参数默认、动态是命令行 opt-in,但两条路径在 CI 上
都没有被观测:今天两仓没有任何 workflow 调用 `verify-shared-dylib.sh`,动态形态
只被 molrs 的 pre-push 钩子(单机、单平台)守着。本 spec 新增 molrs 可复用
workflow `ci-link-form.yml`,内含 `static:`(零参数 wheel + 静态回归例)与
`dynamic:`(molrs/molpack 兄弟 checkout → 跨仓门脚本)两个 job,并在
`ci.yml` 挂成第五个 `uses:` 条目;molpack 侧在既有 `ci.yml` 追加一个
`link-dynamic:` job(非 PR 事件),复用兄弟 checkout 里的同一个脚本。同时把
"molpack 的 `MOLRS_GIT_REF` 不得早于 01 落地"的顺序前置条件写进注释,
因为那是一个会静默把 molpack 的静态 job 变成动态构建的真实危险。

## Design

### 为什么是一个新的可复用 workflow,而不是塞进 ci-rust.yml

`ci-rust.yml` 没有 uv / maturin / 兄弟 checkout 的准备步骤,而且它的 `env:` 块
正被 01 号 spec 删除 —— 在最容易冲突的文件里混两个 diff 是自找麻烦。新建
顶层 push/PR workflow 同样不行:那会绕开分支保护所依赖的 `CI` check 名。
因此按 `ci.yml:11-26` 的纯编排模式,新增
`.github/workflows/ci-link-form.yml`(`on: {workflow_call, workflow_dispatch}`),
`ci.yml` 里加第五条:

```yaml
  link-form:
    name: Link form
    uses: ./.github/workflows/ci-link-form.yml
```

命名上偏离 librarian 建议的 `ci-link-dynamic.yml`,理由明确:这道门要证明
**两个**形态(静态是发布形态、动态是必须不烂的 opt-in),只命名 dynamic 会让
`static:` job 无处安放,或者逼出第二个 workflow。

### 两个 job 的形状(house style 逐项对齐)

`static:`(ubuntu-latest,每次 CI 都跑)
- checkout(molrs `actions/checkout@v7`)→ `dtolnay/rust-toolchain@stable` →
  `mozilla-actions/sccache-action@v0.0.9` → `Swatinem/rust-cache@v2`
  (`shared-key: link-static`,`workspaces: ". -> target"`)→
  `astral-sh/setup-uv@v9.0.0`。
- **零参数**构建 molrs wheel(`maturin build --release`),装进一次性 venv,
  跑 `python regressions/link-mode-static-default.py`。job 里**不得出现任何
  flag / env** —— 这正是被测命题。

`dynamic:`(ubuntu-latest,每次 CI 都跑;molrs 是该不变量的所有者)
- **兄弟布局是硬要求**:molrs checkout 到 `path: molrs`,molpack
  (`repository: MolCrafts/molpack`)到 `path: molpack`。**ref 必须指向真正
  带着 01 的分支**:molrs CI 在 `dev` 与 `master` 上触发,而 molcrafts 的 PR
  先落 `dev`,所以落地窗口内 pin `dev`,不能 pin 裸 `master` —— 否则这道门会
  静默地去校验一个早于 01 的 molpack(grill 2026-08-23 发现)。所选 ref 以
  注释写在 checkout 步骤上。因为
  molpack 的 `.cargo/config.toml` 把 target 指向 `../molrs/target`(相对该配置
  文件),平铺布局会让两仓写进两个 target,门要么假红要么根本找不到 dylib。
  同理 `Swatinem/rust-cache@v2` 用 `workspaces: "molrs -> target"`,并且
  **绝不设置 `CARGO_TARGET_DIR`**(两仓都注释过原因)。
- `uv tool install maturin` 后 `cd molrs && bash scripts/verify-shared-dylib.sh`。
  `MOLPACK_ROOT` 无需设置:脚本默认 `$PROJECT_ROOT/../molpack` 在兄弟布局下
  正好命中。
- **job 里不重抄那两个 `--config` flag** —— 脚本自带(01 号 spec),CI 只是调用者。
  这维持"flag 只存在于脚本 + `docs/interop.md`"这条硬约束。

### molpack 侧

molpack 没有可复用 workflow 拆分(四个 job 都在 `ci.yml`),也有
"no scripts/" 政策(`.pre-commit-config.yaml:41`),所以在 `regression:`
(line 104)之后追加 `link-dynamic:` / `name: link (dynamic)`,内联命令调用
**兄弟 checkout 里的** `molrs/scripts/verify-shared-dylib.sh`(这不违反政策:
molpack 不拥有该脚本)。沿用 molpack 的 pin(`actions/checkout@v6`、
`setup-python@v6`、`setup-uv@v6`)与小写 job 名。成本控制沿用
`regression:` 的先例:`if: github.event_name != 'pull_request'` —— 该 job 要跑
两次 release wheel 构建,而 molrs 侧已经在每次 PR 上跑同一道门,molpack 侧
补的是"molpack 自身变更把 molrs 单元叉开"这条通路。

### 落地顺序前置条件(必须写进注释,不能假装不存在)

molpack 的四个 job 都 checkout `MOLRS_GIT_REF: v0.14.0`(`ci.yml:17`),而
**该 tag 目前不存在**(0.14 发布列车未推),因此 molpack CI 现在整体是红的,
新 job 不会让情况变差。真正的危险在之后:如果 `v0.14.0` 从一个仍带
`prefer-dynamic` 的 molrs 状态切出,而 molpack 已经(在 01 里)删掉了自己的
`RUSTFLAGS` 静态钉,那么 molpack 的"静态"job 会**静默地**构建成动态。因此:

> **01 必须先落到 molrs master,`v0.14.0` 才能切;`MOLRS_GIT_REF` 永远不得
> 指向早于 01 的 ref。**

这句话以注释形式写进 `molpack/.github/workflows/ci.yml` 的 env 块上方,并作为
docs 类验收项。

### 工具选择更正:hooks 是真源,CI 是镜像(实现期)

起草时把三条判据写成"在真实 runner 上观测",于是把 02 卡在了需要 push。
这与本仓既有设计相反:`.pre-commit-config.yaml` 头两行明写
"prek: pre-commit = rustfmt+clippy; pre-push = unit + python + wasm + capi.
**Mirrors ci.yml**",即**钩子是单一真源,`ci.yml` 是镜像**,而
`CLAUDE.md` 的 `ci.local` 也正是 `prek run --all-files --hook-stage pre-push`。

真正的缺口不在 CI,而在钩子:动态形态早有 `verify-shared-dylib` 钩子,
**静态形态一个都没有** —— 我们实际发布的那个形态,本地没有任何东西看着它。
因此本 spec 增补 `link-static` pre-push 钩子(与 CI `static:` job 同样的零参数
payload),两条链接形态判据随即本地可观测,不需要推任何分支;反向门也改为
本地脏树实验(改完即还原,永不提交)。CI 的两个 job 保留为镜像。

### Reuse decision

- `pattern` `molrs/.github/workflows/ci.yml:11-26` — 纯编排模式,新 workflow
  以一行 `uses:` 挂入,不在 `ci.yml` 里写 `steps:`。
- `pattern` `molrs/.github/workflows/ci-capi.yml:13-34` — 可复用 workflow 骨架
  (`on`、单 `test:` job、job 级 sccache env、rust-cache `shared-key` +
  `workspaces`)。
- `pattern` `molrs/.github/workflows/ci-python.yml:39-53` — uv + maturin 的既有
  写法(`astral-sh/setup-uv@v9.0.0` + `enable-cache`)。
- `pattern` `molpack/.github/workflows/ci.yml:70-84` — 跨仓 checkout 与
  `workspaces: "molpack -> ../molrs/target"`;`MOLRS_GIT_REF` 的注意事项按上节
  显式解决。
- `reuse unchanged` `molrs/scripts/verify-shared-dylib.sh` — 两个 dynamic job 的
  唯一实现,CI 只调用。**不新增第二个动态构建脚本、不新增 cargo alias、
  不新增包装脚本。**
- `reuse unchanged` `molrs/regressions/link-mode-static-default.py`(01 交付)与
  `molrs/regressions/ffi-shared-dylib.py` — 分别是 `static:` 与 `dynamic:` 的
  payload,不得互换。
- `reuse unchanged` `molrs/.pre-commit-config.yaml:77-83` — pre-push 钩子保持
  不变;CI 是补位,不是替代。
- `new — 无既有等价物`:`ci-link-form.yml` 与 molpack `link-dynamic:` job
  (两仓今天都没有任何 workflow 触碰链接形态)。

## Files to create or modify

- `molrs/.pre-commit-config.yaml` — 新增 `link-static` pre-push 钩子(静态形态缺失的那一半)+ 头部注释同步
- `molrs/.github/workflows/ci-link-form.yml` (new) — `static:` + `dynamic:` 两个 job(镜像上面两个钩子)
- `molrs/.github/workflows/ci.yml` — 在 `capi:` 之后加 `link-form:` 条目
- `molpack/.github/workflows/ci.yml` — 在 `regression:` 之后加 `link-dynamic:` job;env 块上方补落地顺序注释
- `molrs/docs/interop.md` — 在 § "Local link form" 的前置条件段补一句:动态形态现由 `CI Link Form` 的 `dynamic:` job 与 pre-push 双重守护

## Tasks

- [x] Add `.github/workflows/ci-link-form.yml` with a `static:` job (zero-flag molrs wheel + `regressions/link-mode-static-default.py`) and a `dynamic:` job (sibling molrs/molpack checkout → `scripts/verify-shared-dylib.sh`, no flags in the job)
- [x] Wire `link-form:` into `.github/workflows/ci.yml` as the fifth `uses:` entry
- [x] Add a `link-dynamic:` job to `molpack/.github/workflows/ci.yml` after `regression:` (non-PR events only) invoking the sibling `molrs/scripts/verify-shared-dylib.sh`
- [x] Document the `MOLRS_GIT_REF` landing-order precondition in `molpack/.github/workflows/ci.yml` and the CI/pre-push double guard in `docs/interop.md`
- [x] Add a `link-static` pre-push hook (the static form's missing half) and verify BOTH link-form hooks through `prek run --all-files --hook-stage pre-push`, plus the reverse-gate experiment locally
- [x] Run full check + test suite in both repos

## Testing strategy

本 spec 交付的是 CI 配置,验证同样必须落在**观测到的运行结果**上,而不是
YAML 文本:

- `static:` job 在真实 runner 上绿,且其日志显示回归例报告
  `libmolrs_ffi` 条目数 0(即 CI 与本地在零参数下产出同一形态)。
- `dynamic:` job 在真实 runner 上绿,日志显示两个扩展都有 `libmolrs_ffi`
  条目,以及两次 wheel 构建之间**同一个** sha256(路径相等是空断言,
  已由脚本 step 4b 的 sha 对比取代)。
- 反向门证明:在分支上临时把两个 `--config` 从
  `verify-shared-dylib.sh` 里去掉,`dynamic:` job 必须在
  `FAIL — no libmolrs_ffi dynamic-link entry` 处变红(证明这道门会咬人);
  该实验只在临时分支进行,不合入。
- `static:` job 内不得出现任何 flag / env:`rg -n "RUSTFLAGS|prefer-dynamic|
  CARGO_PROFILE_RELEASE_LTO" .github/workflows/ci-link-form.yml` 只应命中
  `dynamic:` job 里对脚本的调用注释(0 条可执行 flag)。
- 本 spec **不新增** `regressions/` 脚本:01 已交付静态回归例,动态 payload 是
  既有 `regressions/ffi-shared-dylib.py`;再加一个会是重复的门。

## Out of scope

- 发布态切换为动态、pip 层 build-id `==` 机制(与 01 一致,均未接线);
- Windows / macOS runner 上的链接形态门(脚本只实现 otool/ldd 两条分支,
  Windows 无 rpath 概念;扩展前需先在脚本里加分支);
- molpack 侧 `MOLRS_GIT_REF` 的自动化校验(molpack 硬规则:molrs pin 手工管理,
  不进 pre-commit / CI);
- `/mol:map` blueprint 刷新。
