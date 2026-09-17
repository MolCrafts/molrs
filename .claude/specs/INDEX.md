# Specs

Live specs only. A spec is deleted when its work has landed — the record of what
shipped is the git history and `.claude/notes/release.md`, not a stack of closed
spec files.

| Spec | State |
|------|-------|
| [release-0-14-08-ship-molrs](release-0-14-08-ship-molrs.md) | in progress — merge to `master`, tag `v0.14.0`, publish to crates.io / npm / PyPI, replace the molnex `.dev1` wheel |
| [release-0-14-09-molpy-rebase](release-0-14-09-molpy-rebase.md) | blocked on 08 — re-branch molpy from `upstream/master`, pin `>=0.14.0,<0.15` |
| [release-0-14-10-molpy-mirror](release-0-14-10-molpy-mirror.md) | blocked on 09 — sink the duplicated formats and Box geometry into molrs with per-format bit-identical parity |
| [release-0-14-11-molpy-docs](release-0-14-11-molpy-docs.md) | blocked on 10 — typifier spelling sweep, molpy migration guide |
| [release-0-14-12-joint-smoke](release-0-14-12-joint-smoke.md) | blocked on 11 — molnex chain smoke on the released wheel, then tag molpy |

**Release order is fixed** (CLAUDE.md § *Release before molpy*): molrs tags and
publishes before molpy bumps its minor pin. 09–12 do not start before 08 is green.
