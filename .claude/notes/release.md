# Release discipline — molrs before molpy

**Supersedes:** monorepo-under-molpy merge idea (discarded).

## Rule (agents must obey)

1. **molrs is upstream of molpy.** Any API molpy will consume must land in
   molrs, hit **`master` with a version tag `vX.Y.Z`**, and be **published**
   before molpy pins or calls it.
2. **Master + tag.** Release-quality landings on `master` are not complete
   without the matching tag. Untagged master tip is not a pin target for molpy.
3. **No pin-parity scripts.** Do not add automation that fakes “CI will be
   fine.” Agents and operators **manually** verify tag + publish before telling
   molpy to bump. Process lives in harness notes, not `scripts/`.
4. **Same minor line.** molrs / molcrafts-molrs / molpy share **major.minor**
   when co-released; patch may drift. Bump the workspace version on the molrs
   release commit that the tag points at. Consumers pin
   `molcrafts-molrs>=X.Y.0,<X.(Y+1)` (or equivalent) and runtime-check
   major.minor only — never require exact patch equality.

## Manual checklist (before declaring a molrs release done)

- [ ] Version bump on the release commit
- [ ] Tag `vX.Y.Z` on that commit (canonical MolCrafts/molrs publish path)
- [ ] Publish finished (or confirmed already on the index)
- [ ] Only then: molpy may pin `molcrafts-molrs>=X.Y.0,<X.(Y+1)` and use new APIs

## Agent hard-stops

- molpy-facing API without a plan to tag+publish → **stop** or keep it
  private until release.
- “Just maturin develop and molpy can use it” as a landing plan → **stop**.
