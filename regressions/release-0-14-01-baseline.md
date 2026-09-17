# release-0-14-01-baseline — the 0.14.0 release baseline

What this checks: the 0.14 line is one merged tree with one version string
everywhere, and the published `Key` shape survived the merge. Every step below
is reproducible from a checkout with no access to the spec.

## 1. The merge happened

`dev` forked at `v0.13.0` and did not carry `v0.13.1` / `v0.13.2`, so releasing
0.14.0 from it would have withdrawn API that is already on PyPI and crates.io.

```bash
git merge-base --is-ancestor v0.13.1 HEAD && echo "v0.13.1 contained"
git merge-base --is-ancestor v0.13.2 HEAD && echo "v0.13.2 contained"
```

Both must print. Neither did before the merge.

## 2. Eight version strings, no prerelease suffix

| # | Location | Field |
|---|----------|-------|
| 1 | `Cargo.toml` | `[workspace.package] version` |
| 2 | `molrs-ffi/Cargo.toml` | `package.version` + the `molcrafts-molrs` pin |
| 3 | `molrs-wasm/Cargo.toml` | `package.version` + the `molcrafts-molrs` / `molcrafts-molrs-ffi` pins |
| 4 | `molrs-capi/Cargo.toml` | `package.version` + both pins |
| 5 | `molrs-cxxapi/Cargo.toml` | `package.version` + both pins |
| 6 | `molrs-python/Cargo.toml` | `package.version` + both pins |
| 7 | `molrs-python/pyproject.toml` | `project.version` |
| 8 | `molrs/Cargo.toml` | `package.version.workspace = true` (inherits #1) |

```bash
uv --directory molrs-python run --no-sync tox -e py -- tests/test_version_parity.py
```

`molrs-python/tests/test_version_parity.py` discovers the manifests by walking
the tree — it has no hand-written name list, so a new binder workspace is
covered the day it is added. It also rejects any `.dev` / `rc` / `aN` / `bN`
suffix, which is how the `0.13.2.dev1` prerelease is kept out.

## 3. `keys` resolved to the published `Key` shape

The merge rule was **published shape wins**: `master` had shipped
`molrs.keys.<CONST>` as a `Key` object in v0.13.1, `dev` still had plain `str`.
`Key` is what 0.14.0 ships.

```bash
grep -c "pub struct PyKey" molrs-python/src/schema.rs          # 1
grep -c "add_class::<PyKey>" molrs-python/src/schema.rs        # 1
diff <(git show v0.13.2:molrs/src/core/store/keys.rs | grep -oE "pub const [A-Z_0-9]+" | sort) \
     <(grep -oE "pub const [A-Z_0-9]+" molrs/src/core/store/keys.rs | sort)   # no output
```

`Key` compares equal to its own `str`, so call sites that pass plain strings
keep working. `molrs-python/tests/test_ecs_pybind.py` passes unmodified.

## 4. Four release records

`.claude/notes/release.md` carries `## v0.13.0`, `## v0.13.1`, `## v0.13.2` and
`## v0.14.0` in the same format as the older `## v0.12.1` entry. The 0.14.0
section is the content summary; its tag date is filled in by
`release-0-14-08-ship-molrs`.

```bash
grep -c "^## v0.13.0\|^## v0.13.1\|^## v0.13.2\|^## v0.14.0" .claude/notes/release.md   # 4
```

## 5. The merged tree is green

```bash
cargo test -p molcrafts-molrs --lib --features full,filesystem
cargo test --doc -p molcrafts-molrs --features full,filesystem
```

Run cargo with `CARGO_TARGET_DIR` pointing at node-local storage if the
checkout is on a shared filesystem: the repository's own `target/` lives on
Lustre, where cargo has been observed blocking indefinitely on flock (9h27m
elapsed, 0s CPU, `wchan = ldlm_flock_completion`).
